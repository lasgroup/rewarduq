from __future__ import annotations

import os
from collections import defaultdict

import torch
from datasets import Dataset
from omegaconf import DictConfig, ListConfig, OmegaConf

from rewarduq.methods.base import BasePipeline
from rewarduq.utils import get_logger, load_tokenizer_and_model, log_model_summary

from .lora_ensemble_model import LoraEnsembleModel, LoraEnsembleModelConfig
from .lora_ensemble_trainer import LoraEnsembleTrainer, LoraEnsembleTrainerConfig

logger = get_logger(__name__)


class LoraEnsemblePipeline(BasePipeline):
    """Pipeline for training and making predictions with `LoraEnsembleModel`."""

    def __init__(
        self,
        model_config: LoraEnsembleModelConfig,
        trainer_config: LoraEnsembleTrainerConfig,
        output_dirs: list[str] | None = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.trainer_config = trainer_config

        self.eval_schedule = None
        if output_dirs:  # Case: evaluate ensemble from existing adapters
            eval_schedule_dict = defaultdict(list)
            for output_dir in output_dirs:
                if not os.path.exists(output_dir):
                    raise ValueError(f"Output directory {output_dir} does not exist.")

                for checkpoint_name in os.listdir(output_dir):
                    if checkpoint_name.startswith("checkpoint-"):
                        step = int(checkpoint_name.split("-")[-1])
                        checkpoint_path = os.path.join(output_dir, checkpoint_name)
                        eval_schedule_dict[step].append(checkpoint_path)

            self.eval_schedule = [(step, adapter_paths) for step, adapter_paths in sorted(eval_schedule_dict.items())]
            model_config.adapter_paths = self.eval_schedule[0][1]  # Load first adapters initially

            self.model = LoraEnsembleModel(model_config)
            self.tokenizer = self.model.tokenizer
        else:  # Case: train single ensemble from scratch
            self.tokenizer, self.model = load_tokenizer_and_model(
                model_config.base_model_name_or_path,
                base_model_class=model_config.base_model_class,
                base_model_init_kwargs=model_config.base_model_init_kwargs,
                target_modules=None,
                peft_config=self.model_config.peft_config,
            )

            # Zero out the final score layer
            self.model.score.weight.data.zero_()
            if self.model.score.bias is not None:
                self.model.score.bias.data.zero_()

            # Enable gradient checkpointing for memory efficiency
            self.trainer_config.gradient_checkpointing = True

            # Print model summary
            log_model_summary(self.model)

    def predict(
        self,
        prompts: list[list[dict[str, str]]],
        completions: list[list[dict[str, str]]],
    ) -> torch.Tensor:
        raise NotImplementedError()

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        resume_from_checkpoint: bool | str | None = None,
    ):
        trainer = LoraEnsembleTrainer(
            model=self.model,
            args=self.trainer_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            peft_config=None,  # Handled above when loading the model
        )
        if self.eval_schedule:
            for step, adapter_paths in self.eval_schedule:
                logger.info(f"Evaluating ensemble at step {step} with adapters: {adapter_paths}")

                # Update adapters
                self.model.reload_adapters(adapter_paths)

                # Update global step
                trainer.state.global_step = step

                eval_result = trainer.evaluate()
                eval_result["step"] = step
                eval_result["train/global_step"] = step

                # Log evaluation metrics
                if "wandb" in trainer.args.report_to and trainer.accelerator.is_main_process:
                    import wandb

                    wandb.log(eval_result, step=step)

        else:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    @staticmethod
    def from_config(config: DictConfig | ListConfig) -> LoraEnsemblePipeline:
        model_config = LoraEnsembleModelConfig(
            **OmegaConf.to_container(config.model, resolve=True, throw_on_missing=True)
        )
        trainer_config = LoraEnsembleTrainerConfig(
            **OmegaConf.to_container(config.trainer, resolve=True, throw_on_missing=True)
        )

        output_dirs = None
        if "output_dirs" in config:
            output_dirs = OmegaConf.to_container(config.output_dirs, resolve=True)

        pipeline = LoraEnsemblePipeline(model_config, trainer_config, output_dirs=output_dirs)
        return pipeline
