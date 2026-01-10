from __future__ import annotations

import torch
from datasets import Dataset
from omegaconf import DictConfig, ListConfig, OmegaConf

from rewarduq.methods.base import BasePipeline
from rewarduq.utils import encode_prompt_completion, get_logger

from .dpo_head_dropout_ensemble_model import DPOHeadDropoutEnsembleModel, DPOHeadDropoutEnsembleModelConfig
from .dpo_head_dropout_ensemble_trainer import DPOHeadDropoutEnsembleTrainer, DPOHeadDropoutEnsembleTrainerConfig

logger = get_logger(__name__)


class DPOHeadDropoutEnsemblePipeline(BasePipeline):
    """Pipeline for training and making predictions with `DPOHeadDropoutEnsembleModel`."""

    def __init__(
        self,
        model_config: DPOHeadDropoutEnsembleModelConfig,
        trainer_config: DPOHeadDropoutEnsembleTrainerConfig,
    ):
        # Set config
        self.model_config = model_config
        self.trainer_config = trainer_config

        # Initialize model
        self.model = DPOHeadDropoutEnsembleModel(self.model_config)
        self.model.init_ref_model()

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        resume_from_checkpoint: bool | str | None = None,
    ):
        self.trainer = DPOHeadDropoutEnsembleTrainer(  # TODO check Mehta et al. how to train
            model=self.model,
            processing_class=self.model.tokenizer,
            args=self.trainer_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def predict(
        self,
        prompts: list[list[dict[str, str]]],
        completions: list[list[dict[str, str]]],
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            features = encode_prompt_completion(
                prompts,
                completions,
                self.model.tokenizer,
                self.trainer_config.max_prompt_length,
                self.trainer_config.max_completion_length,
            )
            outputs = self.model(features)
            rewards = outputs["rewards"].detach().cpu()
        return rewards

    @staticmethod
    def from_config(config: DictConfig | ListConfig) -> DPOHeadDropoutEnsemblePipeline:
        model_config = DPOHeadDropoutEnsembleModelConfig(
            **OmegaConf.to_container(config.model, resolve=True, throw_on_missing=True)
        )
        trainer_config = DPOHeadDropoutEnsembleTrainerConfig(
            **OmegaConf.to_container(config.trainer, resolve=True, throw_on_missing=True)
        )
        pipeline = DPOHeadDropoutEnsemblePipeline(model_config, trainer_config)
        return pipeline
