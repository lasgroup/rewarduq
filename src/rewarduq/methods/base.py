"""Base classes for uncertainty-aware reward models."""

from __future__ import annotations

import torch
from datasets import Dataset
from omegaconf import DictConfig, ListConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from rewarduq.utils import get_logger

logger = get_logger(__name__)


class BasePipeline:
    """Base class for pipelines that train and make predictions with uncertainty-aware reward models."""

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        resume_from_checkpoint: str | bool | None = None,
    ):
        """Train the model on the provided dataset.

        Args:
            train_dataset (Dataset):
                The dataset to train on.

            eval_dataset (Dataset, optional):
                The dataset to evaluate on. Defaults to None.

            resume_from_checkpoint (str | bool, optional):
                The path to the checkpoint to resume training from. If True, will resume from the last
                checkpoint in the output directory. Defaults to None.
        """
        raise NotImplementedError()

    def predict(
        self,
        prompts: list[list[dict[str, str]]],
        completions: list[list[dict[str, str]]],
    ) -> torch.Tensor:
        """Predict the reward with uncertainty bounds for a given batch of prompts and completions.

        Args:
            prompts (list[list[dict[str, str]]]):
                Batched input prompts.

                Example of a single prompt:
                `[{"role": "user", "content": "What color is the sky?"}]`.

            completions (list[list[dict[str, str]]]):
                Batched generated completions.

                Example of a single completion:
                `[{"role": "assistant", "content": "It is blue."}]`.

        Returns:
            out (torch.Tensor of shape (batch_size, 3)):
                Tensor with reward scores, lower bounds, and upper bounds.
        """
        raise NotImplementedError()

    @staticmethod
    def from_config(config: DictConfig | ListConfig) -> BasePipeline:
        """Build the pipeline from the given config.

        Args:
            config (DictConfig | ListConfig):
                The configuration for the model and trainer inside the pipeline.

        Returns:
            pipeline (BasePipeline):
                The pipeline.
        """
        raise NotImplementedError()


class RewardUQModel(PreTrainedModel):
    """Base class for uncertainty-aware reward models."""

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
    ) -> tuple | dict | ModelOutput:
        """Forward pass of the model.

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length)):
                [What are input IDs?](https://github.com/huggingface/transformers/blob/v4.51.3/docs/source/en/glossary.md#input-ids)

            attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional):
                Mask to avoid performing attention on padding token indices.

                Mask values selected in `[0, 1]`, where
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

               [What are attention masks?](https://github.com/huggingface/transformers/blob/v4.51.3/docs/source/en/glossary.md#attention-mask)
        """
        raise NotImplementedError()
