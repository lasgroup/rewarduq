from __future__ import annotations

import torch
from datasets import Dataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.reward_trainer import _tokenize
from trl.trainer.utils import RewardDataCollatorWithPadding

from rewarduq.methods.base import BasePipeline
from rewarduq.utils import get_logger

from .mlp_head_ensemble_model import MLPHeadEnsembleModel, MLPHeadEnsembleModelConfig
from .mlp_head_ensemble_trainer import MLPHeadEnsembleTrainer, MLPHeadEnsembleTrainerConfig

logger = get_logger(__name__)


class MLPHeadEnsemblePipeline(BasePipeline):
    """Pipeline for training and making predictions with `MLPHeadEnsembleModel`."""

    def __init__(
        self,
        model_config: MLPHeadEnsembleModelConfig,
        trainer_config: MLPHeadEnsembleTrainerConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.trainer_config = trainer_config

        self.model = MLPHeadEnsembleModel(model_config)

    def predict(
        self,
        prompts: list[list[dict[str, str]]],
        completions: list[list[dict[str, str]]],
    ) -> torch.Tensor:
        # TODO Temporary hotfix, NOT GREAT tbh!!!

        tokenizer = self.model.tokenizer

        samples = [
            maybe_apply_chat_template(
                {"prompt": p, "chosen": c, "rejected": c},
                tokenizer=tokenizer,
            )
            for p, c in zip(prompts, completions)
        ]
        samples = {
            "prompts": [s["prompt"] for s in samples],
            "chosen": [s["chosen"] for s in samples],
            "rejected": [s["rejected"] for s in samples],
        }

        samples_tokenized = _tokenize(samples, tokenizer=tokenizer)
        samples_tokenized = [
            {
                "input_ids_chosen": input_ids_chosen,
                "attention_mask_chosen": attention_mask_chosen,
                "input_ids_rejected": input_ids_rejected,
                "attention_mask_rejected": attention_mask_rejected,
            }
            for input_ids_chosen, attention_mask_chosen, input_ids_rejected, attention_mask_rejected in zip(
                samples_tokenized["input_ids_chosen"],
                samples_tokenized["attention_mask_chosen"],
                samples_tokenized["input_ids_rejected"],
                samples_tokenized["attention_mask_rejected"],
            )
        ]

        collator = RewardDataCollatorWithPadding(tokenizer)
        samples_collated = collator(samples_tokenized)
        input_ids = samples_collated["input_ids_chosen"]
        attention_mask = samples_collated["attention_mask_chosen"]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            rewards = outputs["rewards"].detach()
        return rewards

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        resume_from_checkpoint: bool | str | None = None,
    ):
        self.trainer = MLPHeadEnsembleTrainer(
            model=self.model,
            args=self.trainer_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.model.tokenizer,
        )
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    @staticmethod
    def from_config(config: DictConfig | ListConfig) -> MLPHeadEnsemblePipeline:
        model_config = MLPHeadEnsembleModelConfig(
            **OmegaConf.to_container(config.model, resolve=True, throw_on_missing=True)
        )
        trainer_config = MLPHeadEnsembleTrainerConfig(
            **OmegaConf.to_container(config.trainer, resolve=True, throw_on_missing=True)
        )
        pipeline = MLPHeadEnsemblePipeline(model_config, trainer_config)
        return pipeline
