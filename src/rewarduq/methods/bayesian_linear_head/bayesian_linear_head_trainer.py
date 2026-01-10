from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from rewarduq.trainers import RewardUQTrainer, RewardUQTrainerConfig
from rewarduq.utils import get_logger

from .bayesian_linear_head_model import BayesianLinearHeadModel

logger = get_logger(__name__)


@dataclass
class BayesianLinearHeadTrainerConfig(RewardUQTrainerConfig):
    """Configuration class for `BayesianLinearHeadTrainer`."""

    l2_reg: float = 0.0
    # NEW: Choose how the final Hessian is computed after training
    final_hessian_mode: Literal["unweighted", "weighted"] = "weighted"


class BayesianLinearHeadTrainer(RewardUQTrainer):
    """Trainer class for `BayesianLinearHeadModel`."""

    def compute_loss(
        self,
        model: BayesianLinearHeadModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        """Compute DPO loss for training the head. No Hessian updates here."""
        outputs_chosen = self._get_model_outputs(model, inputs, "chosen")
        outputs_rejected = self._get_model_outputs(model, inputs, "rejected")
        rewards_chosen = outputs_chosen["rewards"][:, 0]  # shape: (batch_size,)
        rewards_rejected = outputs_rejected["rewards"][:, 0]  # shape: (batch_size,)

        # compute standard loss
        loss_base = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

        # compute l2 regularization
        model_unwrapped = self.accelerator.unwrap_model(model)
        loss_reg = (self.args.l2_reg / 2) * (model_unwrapped.head.linear.weight**2).sum()

        # compute final loss
        loss = loss_base + loss_reg

        # compute batch metrics
        batch_metrics = {
            "win_rate": (rewards_chosen > rewards_rejected).float(),
            "rewards/chosen": rewards_chosen,
            "rewards/rejected": rewards_rejected,
            "rewards/margins": rewards_chosen - rewards_rejected,
            "loss_base": loss_base,
            "loss_reg": loss_reg,
        }
        self.cache_batch_metrics(batch_metrics)

        if return_outputs:
            return loss, {
                "chosen": outputs_chosen,
                "rejected": outputs_rejected,
            }
        return loss
