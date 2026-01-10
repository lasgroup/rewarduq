from __future__ import annotations

from typing import Any

import numpy as np

from rewarduq.metrics import combine_metrics, compute_default_metrics, normalize_report_to
from rewarduq.utils import get_logger

logger = get_logger(__name__)


def individual_heads_metrics(
    results: dict[str, Any],
    report_to: str | list | None = None,
    **kwargs,
) -> dict[str, float | np.ndarray]:
    """Compute metrics targeting the MLPHeadEnsembleModel."""
    if "outputs_chosen" not in results or len(results["outputs_chosen"]) == 0:
        raise ValueError("No chosen outputs found in results. Cannot compute metric.")
    if "outputs_rejected" not in results or len(results["outputs_rejected"]) == 0:
        raise ValueError("No rejected outputs found in results. Cannot compute metric.")
    report_to = normalize_report_to(report_to)

    num_samples = results["rewards"].shape[0]
    num_heads = results["outputs_chosen"][0]["mlp_outputs"].shape[1]

    wins_per_head = [0] * num_heads
    chosen_pred_mean_per_head = [0] * num_heads
    rejected_chosen_pred_mean_per_head = [0] * num_heads
    agreement = 0

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for chosen, rejected in zip(results["outputs_chosen"], results["outputs_rejected"]):
        prefs = sigmoid(chosen["mlp_outputs"] - rejected["mlp_outputs"])  # shape: (batch_size, num_heads)

        for i in range(prefs.shape[1]):  # For each head
            wins_per_head[i] += (prefs[:, i] > 0.5).sum()

            chosen_pred_mean_per_head[i] = chosen["mlp_outputs"][:, i].mean(axis=0)
            rejected_chosen_pred_mean_per_head[i] = rejected["mlp_outputs"][:, i].mean(axis=0)

        for i in range(prefs.shape[0]):  # For each sample
            agreement += (prefs[i] > 0.5).sum()
            agreement -= (prefs[i] < 0.5).sum()

    wandb_metrics = {}
    if "all" in report_to or "wandb" in report_to:
        import wandb

        for i in range(prefs.shape[1]):  # For each head
            wandb_metrics[f"chosen_pred_hist/head_{i}"] = wandb.Histogram(
                chosen["mlp_outputs"][:, i].tolist(),
            )
            wandb_metrics[f"rejected_pred_hist/head_{i}"] = wandb.Histogram(
                rejected["mlp_outputs"][:, i].tolist(),
            )

    return {
        **{f"win_rate/head_{i}": wins_per_head[i] / num_samples for i in range(num_heads)},
        **{f"rewards/chosen_pred_mean/head_{i}": chosen_pred_mean_per_head[i] for i in range(num_heads)},
        **{f"rewards/rejected_pred_mean/head_{i}": rejected_chosen_pred_mean_per_head[i] for i in range(num_heads)},
        "head_agreement": agreement / (num_samples * num_heads),
        **wandb_metrics,
    }


compute_mlp_head_ensemble_metrics = combine_metrics(
    compute_default_metrics,
    individual_heads_metrics,
)
