from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rewarduq.trainers import RewardUQTrainer, RewardUQTrainerConfig
from rewarduq.utils import get_logger

from .mlp_head_ensemble_model import MLPHeadEnsembleModel

logger = get_logger(__name__)


@dataclass
class MLPHeadEnsembleTrainerConfig(RewardUQTrainerConfig):
    """Configuration class for `MLPHeadEnsembleTrainer`.

    Args:
        regularization_towards_initial_weights (float):
            The coefficient for the regularization term that encourages the weights of the ensemble head
            to stay close to their initial values.
    """

    # Loss function parameters
    regularization_towards_initial_weights: float = 0.5


class MLPHeadEnsembleTrainer(RewardUQTrainer):
    """Trainer class for `MLPHeadEnsembleModel`."""

    def compute_loss(
        self,
        model: MLPHeadEnsembleModel,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outputs_chosen = self._get_model_outputs(model, inputs, "chosen", output_individual_rewards=True)
        outputs_rejected = self._get_model_outputs(model, inputs, "rejected", output_individual_rewards=True)
        rewards_chosen = outputs_chosen["individual_rewards"]  # Shape: (batch_size, num_heads)
        rewards_rejected = outputs_rejected["individual_rewards"]  # Shape: (batch_size, num_heads)

        # Compute loss per head
        if "margin" in inputs:
            loss_base = -F.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"].unsqueeze(1)).mean()
        else:
            loss_base = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

        # Compute regularization towards initial weights per head
        model_unwrapped = self.accelerator.unwrap_model(model)
        weights = model_unwrapped.get_mlp_weights().to(self.args.device)  # Shape: (num_heads, weights_dim)
        weights_init = model_unwrapped.initial_weights.detach().to(self.args.device)  # Shape: (num_heads, weights_dim)
        weights_mse = ((weights - weights_init) ** 2).mean(dim=1)  # Shape: (num_heads,)
        loss_initial_weights = self.args.regularization_towards_initial_weights * weights_mse.mean()

        # Compute regularization towards centered rewards per head
        loss_center_rewards = (self.args.center_rewards_coefficient or 0) * torch.mean(
            (rewards_chosen + rewards_rejected) ** 2
        )

        # Compute final loss
        loss = loss_base + loss_initial_weights + loss_center_rewards

        # Compute batch metrics (on mean rewards across heads)
        rewards_chosen_mean = rewards_chosen.mean(dim=1)
        rewards_rejected_mean = rewards_rejected.mean(dim=1)
        batch_metrics = {
            "win_rate": (rewards_chosen_mean > rewards_rejected_mean).float(),
            "rewards/chosen": rewards_chosen_mean,
            "rewards/rejected": rewards_rejected_mean,
            "rewards/margins": rewards_chosen_mean - rewards_rejected_mean,
            "loss_base": loss_base,
            "loss_initial_weights": loss_initial_weights,
            "loss_center_rewards": loss_center_rewards,
        }
        self.cache_batch_metrics(batch_metrics)

        if return_outputs:
            return loss, {
                "chosen": outputs_chosen,
                "rejected": outputs_rejected,
            }
        return loss
