from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel

from rewarduq.trainers.dpo_rewarduq_trainer import DPORewardUQTrainer, DPORewardUQTrainerConfig
from rewarduq.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DPOHeadDropoutEnsembleTrainerConfig(DPORewardUQTrainerConfig):
    """Configuration class for `DPOHeadDropoutEnsembleTrainer`."""

    # Loss function parameters
    center_rewards_coefficient: float = 0.0
    std_rewards_coefficient: float = 0.0


class DPOHeadDropoutEnsembleTrainer(DPORewardUQTrainer):
    """Trainer class for `DPOHeadDropoutEnsembleModel`."""

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Override `rewarduq.DPORewardUQTrainer.compute_loss`.

        Modifications:
        - Add `loss_center_rewards` and `loss_std_rewards` to the final loss.
        """
        # concatenate inputs
        inputs_concatenated = self.concatenated_inputs(inputs, padding_value=self.padding_value)

        # create slices to separate chosen and rejected outputs
        num_examples = inputs_concatenated["input_ids"].shape[0] // 2
        CHOSEN = slice(0, num_examples)
        REJECTED = slice(num_examples, 2 * num_examples)

        # forward pass
        outputs = model(**inputs_concatenated)

        outputs = dict(outputs)
        logprobs = outputs.pop("logprobs_all")  # shape: (batch_size, num_samples)
        ref_logprobs = outputs.pop("ref_logprobs_all")  # shape: (batch_size, num_samples)
        labels = outputs.pop("labels")  # shape: (batch_size, sequence_length)
        loss_mask = outputs.pop("loss_mask")  # shape: (batch_size, sequence_length)

        # compute DPO loss
        # NOTE: We first perform all loss computations involving large tensors (i.e. with dimensions of size
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

        # compute regularization towards centered rewards per dropout sample
        loss_center_rewards = (self.args.center_rewards_coefficient or 0) * torch.mean(
            (rewards_chosen + rewards_rejected) ** 2
        )

        # compute regularization of std across dropout samples
        loss_std_rewards = (self.args.std_rewards_coefficient or 0) * (
            (rewards_chosen.std(dim=1).mean() + rewards_rejected.std(dim=1).mean()) * 0.5
        )

        # compute final loss
        loss = loss_base + loss_center_rewards + loss_std_rewards

        # compute batch metrics (on mean rewards across heads)
        rewards_chosen_mean = rewards_chosen.mean(dim=1)
        rewards_rejected_mean = rewards_rejected.mean(dim=1)
        batch_metrics = {
            "win_rate": (rewards_chosen_mean > rewards_rejected_mean).float(),
            "rewards/chosen": rewards_chosen_mean,
            "rewards/rejected": rewards_rejected_mean,
            "rewards/margins": rewards_chosen_mean - rewards_rejected_mean,
            "loss_base": loss_base,
            "loss_center_rewards": loss_center_rewards,
            "loss_std_rewards": loss_std_rewards,
        }
        self.cache_batch_metrics(batch_metrics)

        # split remaining model outputs into chosen and rejected
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
