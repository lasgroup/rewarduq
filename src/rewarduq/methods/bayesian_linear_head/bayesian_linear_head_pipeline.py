from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from rewarduq.methods.base import BasePipeline
from rewarduq.utils import get_logger

from .bayesian_linear_head_model import BayesianLinearHeadModel, BayesianLinearHeadModelConfig
from .bayesian_linear_head_trainer import BayesianLinearHeadTrainer, BayesianLinearHeadTrainerConfig

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

# Access the real underlying logger with logger.logger
if not logger.logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.logger.addHandler(handler)


class BayesianLinearHeadPipeline(BasePipeline):
    """Pipeline for training and making predictions with `BayesianLinearHeadModel`."""

    def __init__(
        self,
        model_config: BayesianLinearHeadModelConfig,
        trainer_config: BayesianLinearHeadTrainerConfig,
    ):
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.model = BayesianLinearHeadModel(model_config)
        self.trainer: BayesianLinearHeadTrainer | None = None

    def predict(
        self,
        prompts: list[list[dict[str, str]]],
        completions: list[list[dict[str, str]]],
    ) -> torch.Tensor:
        self.model.eval()
        batch_texts: list[str] = []

        for p_dialog, c_dialog in zip(prompts, completions):
            full_dialog = p_dialog + c_dialog
            text = self.model.tokenizer.apply_chat_template(full_dialog, tokenize=False, add_generation_prompt=False)
            batch_texts.append(text)

        if not batch_texts:
            return torch.empty(0, 3, device=self.model.head.linear.weight.device)

        if self.model.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id for batch padding in predict.")

        padded_inputs = self.model.tokenizer(
            batch_texts,
            max_length=self.trainer_config.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(input_ids=padded_inputs["input_ids"], attention_mask=padded_inputs["attention_mask"])
        return outputs["rewards"].detach().cpu()

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        resume_from_checkpoint: bool | str | None = None,
    ):
        # === STAGE 1: Train the Reward Model Head ===
        logger.info("Starting Stage 1: Training the reward model head...")
        self.trainer = BayesianLinearHeadTrainer(
            model=self.model,
            args=self.trainer_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.model.tokenizer,
        )
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Reward model head training finished.")

        # === STAGE 2: Post-hoc Hessian computation ===
        logger.info("Starting Stage 2: Post-hoc Hessian computation...")

        # Get the model (accelerate will unwrap if needed)
        inner = self.trainer.accelerator.unwrap_model(self.model)

        # Compute features for Hessian
        train_dataloader = self.trainer.get_train_dataloader()
        z_accumulator = []

        for inputs in tqdm(train_dataloader, desc="Hessian computation"):
            inputs = self.trainer._prepare_inputs(inputs)
            with torch.no_grad():
                chosen_features = self.trainer._get_model_outputs(
                    inner,
                    inputs,
                    "chosen",
                    output_only_features=True,
                )
                rejected_features = self.trainer._get_model_outputs(
                    inner,
                    inputs,
                    "rejected",
                    output_only_features=True,
                )
                delta_f = chosen_features - rejected_features
                z_accumulator.append(delta_f.to(inner.H.device, non_blocking=True))

        all_z_vectors = torch.cat(z_accumulator, dim=0)
        logger.info(f"Total samples: {all_z_vectors.shape[0]}")

        # === Compute Hessian ===
        logger.info("Computing Hessian...")
        inner.compute_and_set_final_hessian(all_z_vectors, self.trainer_config.final_hessian_mode)

        logger.info("Post-hoc Hessian computation completed.")

        # === Save model with computed Hessian ===
        # logger.info("Saving model with computed Hessian...")
        # self.trainer.save_model(self.trainer_config.output_dir)
        # logger.info(f"Model with Hessian saved to {self.trainer_config.output_dir}")

        # === Final evaluation with all metrics ===
        if self.trainer.args.do_eval and self.trainer.eval_dataset is not None:
            logger.info("Evaluating with final Hessian...")
            metrics = self.trainer.evaluate(metric_key_prefix="eval")
            logger.info(f"Final‑Hessian metrics: {metrics}")

            # === Save checkpoint with post-Hessian evaluation ===
            logger.info("Saving checkpoint with post-Hessian metrics...")

        acc = self.trainer.accelerator
        model_to_save = acc.unwrap_model(self.model)

        acc.wait_for_everyone()
        state = acc.get_state_dict(model_to_save)

        if acc.is_main_process:
            out_dir = Path(self.trainer.args.output_dir) / "checkpoint-final"

            self.trainer.save_state()
            model_to_save.save_pretrained(
                out_dir,
                state_dict=state,
                safe_serialization=True,
            )
            self.model.tokenizer.save_pretrained(out_dir)

        acc.wait_for_everyone()

    @staticmethod
    def from_config(config: DictConfig | ListConfig) -> BayesianLinearHeadPipeline:
        logger.info("BayesianLinearHeadPipeline: Creating from config...")

        model_cfg_dict = OmegaConf.to_container(config.model, resolve=True, throw_on_missing=True)
        trainer_cfg_dict_mutable = dict(
            OmegaConf.to_container(config.trainer, resolve=True, throw_on_missing=True) or {}
        )

        if not isinstance(model_cfg_dict, dict) or not isinstance(trainer_cfg_dict_mutable, dict):
            raise ValueError("Config sections 'model' and 'trainer' must be dictionaries.")

        model_config = BayesianLinearHeadModelConfig(**model_cfg_dict)

        if "output_dir" not in trainer_cfg_dict_mutable:
            default_output_dir = f"temp_outputs_bayesian_{model_config.base_model_name_or_path.replace('/', '_')}"
            logger.warning(f"'output_dir' not found in trainer_config, using default: '{default_output_dir}'")
            trainer_cfg_dict_mutable["output_dir"] = default_output_dir

        try:
            trainer_config = BayesianLinearHeadTrainerConfig(**trainer_cfg_dict_mutable)
        except TypeError as e:
            logger.error(f"Arguments that caused error: {trainer_cfg_dict_mutable}")
            raise e

        pipeline_instance = BayesianLinearHeadPipeline(model_config, trainer_config)
        logger.info("BayesianLinearHeadPipeline instance created successfully.")
        return pipeline_instance
