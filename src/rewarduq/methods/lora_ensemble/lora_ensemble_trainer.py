from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers.modeling_utils import unwrap_model

from rewarduq.trainers import RewardUQTrainer, RewardUQTrainerConfig
from rewarduq.utils import get_logger

from .lora_ensemble_model import LoraEnsembleModel

logger = get_logger(__name__)


@dataclass
class LoraEnsembleTrainerConfig(RewardUQTrainerConfig):
    """Configuration class for `LoraEnsembleTrainer`."""

    # Loss function parameters
    regularization_towards_initial_weights: float = 0.5


class LoraEnsembleTrainer(RewardUQTrainer):
    """Trainer class for `LoraEnsembleModel`."""

    def __init__(self, model: torch.nn.Module, **kwargs):
        # store initial weights for regularization if not LoraEnsembleModel
        if not isinstance(model, LoraEnsembleModel):
            initial_weights = self._get_model_weights(model).clone().detach()
            model.initial_weights = torch.nn.Parameter(initial_weights, requires_grad=False)

        super().__init__(model=model, **kwargs)

    def _get_model_weights(self, model: torch.nn.Module) -> torch.Tensor:
        """Get the flattened weights of the model."""
        return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad], dim=0)

    def compute_loss(
        self,
        model: LoraEnsembleModel,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ):
        rewards_chosen = self._get_model_outputs(model, inputs, "chosen")
        rewards_rejected = self._get_model_outputs(model, inputs, "rejected")

        if hasattr(rewards_chosen, "logits"):
            rewards_chosen = rewards_chosen.logits
            rewards_rejected = rewards_rejected.logits
        elif isinstance(rewards_chosen, dict) and "rewards" in rewards_chosen:
            rewards_chosen = rewards_chosen["rewards"]
            rewards_rejected = rewards_rejected["rewards"]
        else:
            raise ValueError(f"Unexpected model output type: {type(rewards_chosen)}")

        # compute loss per head
        if "margin" in inputs:
            loss_base = -F.logsigmoid(rewards_chosen[:, 0] - rewards_rejected[:, 0] - inputs["margin"]).mean()
        else:
            loss_base = -F.logsigmoid(rewards_chosen[:, 0] - rewards_rejected[:, 0]).mean()

        # compute regularization towards initial weights
        if isinstance(unwrap_model(model), LoraEnsembleModel):
            # in evaluation mode, we do not have access to the initial weights
            loss_initial_weights = torch.tensor(0.0, device=self.args.device)
        else:
            weights = self._get_model_weights(model).to(self.args.device)  # shape: (num_weights)
            initial_weights = self.model.initial_weights.to(self.args.device)  # shape: (num_weights)
            weights_norm = torch.norm(weights - initial_weights, p=2)  # shape: (1,)
            loss_initial_weights = self.args.regularization_towards_initial_weights * weights_norm  # shape: (1,)

        # compute regularization towards centered rewards per head
        loss_center_rewards = (self.args.center_rewards_coefficient or 0) * torch.mean(
            (rewards_chosen[:, 0] + rewards_rejected[:, 0]) ** 2
        )

        # compute final loss
        loss = loss_base + loss_initial_weights + loss_center_rewards

        # compute batch metrics
        batch_metrics = {
            "win_rate": (rewards_chosen[:, 0] > rewards_rejected[:, 0]).float(),
            "rewards/chosen": rewards_chosen[:, 0],
            "rewards/rejected": rewards_rejected[:, 0],
            "rewards/margins": rewards_chosen[:, 0] - rewards_rejected[:, 0],
            "loss_base": loss_base,
            "loss_center_rewards": loss_center_rewards,
            "loss_initial_weights": loss_initial_weights,
        }
        self.cache_batch_metrics(batch_metrics)

        if return_outputs:
            return loss, {
                "chosen": {"rewards": rewards_chosen},
                "rejected": {"rewards": rewards_rejected},
            }
        return loss
