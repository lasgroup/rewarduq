from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import pad_to_length

from rewarduq.trainers.trainer_extension import BatchMetricsTrainerMixin, TrainerExtension, TrainerExtensionConfig
from rewarduq.utils import check_supported_args, get_logger

logger = get_logger(__name__)


@dataclass
class DPORewardUQTrainerConfig(TrainerExtensionConfig, DPOConfig):
    """Configuration class for `DPORewardUQTrainer`."""

    pass


class DPORewardUQTrainer(BatchMetricsTrainerMixin, TrainerExtension, DPOTrainer):
    """Trainer class for DPO-based uncertainty-aware reward models."""

    def __init__(self, **kwargs):
        kwargs["ref_model"] = kwargs["model"].ref_model
        super().__init__(**kwargs)

        # Update reference model with the one prepared by the trainer
        self.model.ref_model = self.ref_model

        # Check for currently unsupported arguments
        check_supported_args(self.args, "use_logits_to_keep", [False], temp=True)
        check_supported_args(self.args, "padding_free", [False], temp=True)
        check_supported_args(self.args, "use_weighting", [False], temp=True)
        check_supported_args(self.args, "rpo_alpha", [None], temp=True)
        check_supported_args(self, "aux_loss_enabled", [False], temp=True)

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, list | torch.LongTensor],
        padding_value: int,
    ) -> dict[str, torch.LongTensor]:
        """
        Override `trl.DPOTrainer.concatenated_inputs` [1].

        Modifications:
        - Remove duplication of `prompt_input_ids`.
        - Remove duplication of `prompt_attention_mask`.
        - Add concatenation of `chosen_completion_mask` and `rejected_completion_mask`.

        References:
            [1] https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/dpo_trainer.py#L849
        """
        output = {}

        # Concatenate the chosen and rejected completions
        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        output["input_ids"] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
            dim=0,
        )
        output["attention_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
            dim=0,
        )
        output["completion_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_completion_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_completion_mask"], max_completion_length, pad_value=0),
            ),
            dim=0,
        )

        return output

    def compute_loss_context_manager(self):
        """Override `transformers.Trainer.compute_loss_context_manager` [1].

        Modifications:
        - Integrate `trl.DPOTrainer` logic for mixed precision [2] into `transformers.Trainer` context manager.

        References:
            [1] https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L3943
            [2] https://github.com/huggingface/trl/blob/v0.23.0/trl/trainer/dpo_trainer.py#L1764
        """
        ctx_stack = super().compute_loss_context_manager()

        # Add mixed precision context manager from `trl.DPOTrainer`
        if self._peft_has_been_casted_to_bf16:
            ctx_autocast = torch.autocast(self.accelerator.device.type)
            ctx_stack = ctx_stack.enter_context(ctx_autocast)

        return ctx_stack

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Override `trl.DPOTrainer.compute_loss` [1].

        Modifications:
        - Inline `trl.DPOTrainer.get_batch_loss_metrics` [2] and `trl.DPOTrainer.concatenated_forward` [3].
        - Return model outputs including chosen and rejected rewards instead of metrics.

        References:
            [1] https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/dpo_trainer.py#L1343
            [2] https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/dpo_trainer.py#L1281
            [3] https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/dpo_trainer.py#L1105
        """
        # Concatenate inputs
        inputs_concatenated = self.concatenated_inputs(inputs, padding_value=self.padding_value)

        # Create slices to separate chosen and rejected outputs
        num_examples = inputs_concatenated["input_ids"].shape[0] // 2
        CHOSEN = slice(0, num_examples)
        REJECTED = slice(num_examples, 2 * num_examples)

        # Forward pass
        outputs = model(**inputs_concatenated)

        outputs = dict(outputs)
        logprobs = outputs.pop("logprobs_all")  # Shape: (batch_size, num_samples)
        ref_logprobs = outputs.pop("ref_logprobs_all")  # Shape: (batch_size, num_samples)
        labels = outputs.pop("labels")  # Shape: (batch_size, sequence_length)
        loss_mask = outputs.pop("loss_mask")  # Shape: (batch_size, sequence_length)

        # Compute DPO loss
        # NOTE: We first perform all loss computations involving large tensors (i.e., with dimensions of size
        # `sequence_length` or `vocab_size`) in order to free up GPU memory as early as possible.
        if self.loss_type == "ipo":
            logprobs = logprobs / loss_mask.sum(-1).unsqueeze(-1)
            ref_logprobs = ref_logprobs / loss_mask.sum(-1).unsqueeze(-1)
        del labels, loss_mask

        loss_base, rewards_chosen, rewards_rejected = self.dpo_loss(
            logprobs[CHOSEN],
            logprobs[REJECTED],
            ref_logprobs[CHOSEN],
            ref_logprobs[REJECTED],
        )
        loss_base = loss_base.mean()

        # Compute final loss
        loss = loss_base

        # Compute batch metrics (on mean rewards across heads)
        rewards_chosen_mean = rewards_chosen.mean(dim=1)
        rewards_rejected_mean = rewards_rejected.mean(dim=1)
        batch_metrics = {
            "win_rate": (rewards_chosen_mean > rewards_rejected_mean).float(),
            "rewards/chosen": rewards_chosen_mean,
            "rewards/rejected": rewards_rejected_mean,
            "rewards/margins": rewards_chosen_mean - rewards_rejected_mean,
            "loss_base": loss_base,
        }
        self.cache_batch_metrics(batch_metrics)

        # Split remaining model outputs into chosen and rejected
        outputs_chosen = {}
        outputs_rejected = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs_chosen[key] = value[CHOSEN]
                outputs_rejected[key] = value[REJECTED]
            else:
                outputs_chosen[key] = value
                outputs_rejected[key] = value

        if return_outputs:
            return loss, {
                "chosen": outputs_chosen,
                "rejected": outputs_rejected,
            }
        else:
            return loss
