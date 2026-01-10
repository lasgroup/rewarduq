"""Metrics for evaluating the performance of uncertainty-aware reward models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np
import pandas as pd
from typing_extensions import Concatenate

from rewarduq.utils import normalize_report_to

MetricCallable = Callable[Concatenate[dict[str, Any], ...], dict[str, Union[float, np.ndarray]]]

CHOSEN = 0
REJECTED = 1

PRED = 0
LOWER = 1
UPPER = 2


def _safe_div(a, b, default: float = 0):
    return np.divide(a, b, out=np.full_like(a, default, dtype=float), where=b != 0)


def compute_reward_margins(rewards: np.ndarray) -> np.ndarray:
    """Compute reward margins and their lower and upper bounds."""
    margin_pred = rewards[:, CHOSEN, PRED] - rewards[:, REJECTED, PRED]
    margin_lower = rewards[:, CHOSEN, LOWER] - rewards[:, REJECTED, UPPER]
    margin_upper = rewards[:, CHOSEN, UPPER] - rewards[:, REJECTED, LOWER]
    margin = np.stack([margin_pred, margin_lower, margin_upper], axis=1)  # Shape: (batch_size, 3)
    return margin


def compute_pref_probs(rewards: np.ndarray) -> np.ndarray:
    """Compute preference probabilities and their lower and upper bounds."""

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pref_pred = sigmoid(rewards[:, CHOSEN, PRED] - rewards[:, REJECTED, PRED])
    pref_lower = sigmoid(rewards[:, CHOSEN, LOWER] - rewards[:, REJECTED, UPPER])
    pref_upper = sigmoid(rewards[:, CHOSEN, UPPER] - rewards[:, REJECTED, LOWER])
    pref = np.stack([pref_pred, pref_lower, pref_upper], axis=1)  # Shape: (batch_size, 3)
    return pref


def compute_statistics(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, float | np.ndarray]:
    """Compute basic statistics over predictions, lower bounds, and upper bounds."""
    if weights is None:
        weights = np.ones(len(values))

    def _statistics(values: np.ndarray, prefix: str) -> dict[str, float | np.ndarray]:
        quantiles = np.quantile(values, q=[0, 0.25, 0.5, 0.75, 1], method="inverted_cdf", weights=weights)
        return {
            f"{prefix}mean": np.average(values, weights=weights),
            f"{prefix}min": quantiles[0],
            f"{prefix}p25": quantiles[1],
            f"{prefix}median": quantiles[2],
            f"{prefix}p75": quantiles[3],
            f"{prefix}max": quantiles[4],
        }

    return {
        **_statistics(values[:, PRED], prefix=f"{prefix}pred_"),
        **_statistics(values[:, LOWER], prefix=f"{prefix}lower_"),
        **_statistics(values[:, UPPER], prefix=f"{prefix}upper_"),
        **_statistics(values[:, UPPER] - values[:, LOWER], prefix=f"{prefix}uncertainty_"),
    }


def reward_statistics(results: dict[str, Any], **kwargs) -> dict[str, float | np.ndarray]:
    """Compute statistics of predicted rewards."""
    if "rewards" not in results or len(results["rewards"]) == 0:
        raise ValueError("No rewards found in results. Cannot compute metric.")

    rewards = results["rewards"]
    weights = results.get("weights", np.ones(len(rewards)))

    margins = compute_reward_margins(rewards)

    return {
        **compute_statistics(rewards[:, CHOSEN], weights=weights, prefix="rewards/chosen_"),
        **compute_statistics(rewards[:, REJECTED], weights=weights, prefix="rewards/rejected_"),
        **compute_statistics(margins, weights=weights, prefix="rewards/margins_"),
    }


def preference_statistics(results: dict[str, Any], **kwargs) -> dict[str, float | np.ndarray]:
    """Compute statistics of predicted preferences."""
    if "rewards" not in results or len(results["rewards"]) == 0:
        raise ValueError("No rewards found in results. Cannot compute metric.")

    rewards = results["rewards"]
    weights = results.get("weights", np.ones(len(rewards)))

    prefs_all = compute_pref_probs(rewards)

    return {
        **compute_statistics(prefs_all, weights=weights, prefix="prefs/"),
    }


def win_rate(results: dict[str, Any], **kwargs) -> dict[str, float | np.ndarray]:
    """Compute the win rate for the predicted rewards.

    The `win_rate` corresponds to the percentage of
    - `reward(y_chosen) > reward(y_rejected)` or
    - `preference(y_chosen > y_rejected) > 0.5`.
    """
    if "rewards" not in results or len(results["rewards"]) == 0:
        raise ValueError("No rewards found in results. Cannot compute metric.")

    rewards = results["rewards"]
    weights = results.get("weights", np.ones(len(rewards)))

    win_rate = np.average(rewards[:, CHOSEN, PRED] > rewards[:, REJECTED, PRED], weights=weights)

    return {"win_rate": win_rate}


def calibration_statistics(
    results: dict[str, Any],
    n_bins: int = 20,
    return_output: bool = False,
    report_to: str | list | None = None,
    **kwargs,
) -> dict[str, float | np.ndarray]:
    """Compute calibration statistics for the predicted preferences.

    The `prefs/ece` and `prefs/mce` correspond to the expected and maximum
    calibration error as explained in [1] and the implementation is taken from
    [2].

    The `prefs/elce` and `prefs/mlce` correspond to the expected and maximum
    lower bound calibration error, and `prefs/euce` and `prefs/muce` for the
    upper bound respectively.

    The `prefs/brier_score` is the MSE between predicted and actual preferences.

    References
        [1] https://arxiv.org/pdf/1706.04599
        [2] https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/calibration.py#L927
    """
    if "rewards" not in results or len(results["rewards"]) == 0:
        raise ValueError("No rewards found in results. Cannot compute metric.")

    rewards = results["rewards"]
    weights = results.get("weights", np.ones(len(rewards)))

    report_to = normalize_report_to(report_to)

    # Compute predicted and true preferences
    pref_probs = compute_pref_probs(rewards)
    pref_true = np.ones_like(pref_probs)
    # Extend with flipped predictions and true labels (note: upper bounds become lower bounds and vice versa)
    pref_probs = np.concatenate(
        [pref_probs, 1 - np.stack([pref_probs[:, 0], pref_probs[:, 2], pref_probs[:, 1]], axis=1)]
    )
    pref_true = np.concatenate([pref_true, 1 - pref_true])
    weights = np.concatenate([weights, weights])

    # Define bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(pref_probs, bins[1:-1])

    # Compute calibration curve and error
    def compute_curve_and_error(
        binids: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        error_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        # Compute accuracy and average confidence per bin
        bin_count_true = np.bincount(binids, weights=y_true * weights, minlength=n_bins)
        bin_count_pred = np.bincount(binids, weights=y_pred * weights, minlength=n_bins)
        bin_count = np.bincount(binids, weights=weights, minlength=n_bins)
        bin_prob_true = _safe_div(bin_count_true, bin_count, default=np.nan)
        bin_prob_pred = _safe_div(bin_count_pred, bin_count, default=np.nan)
        curve = bin_prob_true, bin_prob_pred, bins, bin_count

        # Compute calibration error
        calibration_error = error_fn(bin_prob_true, bin_prob_pred)
        ece = np.nansum(bin_count / weights.sum() * calibration_error)
        mce = np.nanmax(calibration_error)

        return curve, ece, mce

    curve_pred, ece, mce = compute_curve_and_error(
        binids[:, PRED],
        pref_true[:, PRED],
        pref_probs[:, PRED],
        error_fn=lambda p_true, p_pred: np.abs(p_true - p_pred),
    )
    curve_lower, elce, mlce = compute_curve_and_error(
        binids[:, LOWER],
        pref_true[:, LOWER],
        pref_probs[:, LOWER],
        error_fn=lambda p_true, p_lower: np.maximum(p_lower - p_true, 0),
    )
    curve_upper, euce, muce = compute_curve_and_error(
        binids[:, UPPER],
        pref_true[:, UPPER],
        pref_probs[:, UPPER],
        error_fn=lambda p_true, p_upper: np.maximum(p_true - p_upper, 0),
    )

    # Compute brier score
    brier_score = np.average((pref_probs[:, PRED] - pref_true[:, PRED]) ** 2, weights=weights)

    if "all" in report_to or "wandb" in report_to:
        import wandb

        def plot_calibration_curve(curve, title):
            bin_prob_true, bin_prob_pred, bins, bin_count = curve
            table = wandb.Table(
                dataframe=pd.DataFrame(
                    {
                        "bin": bins[:-1],
                        "count": bin_count,
                        "p_true": bin_prob_true,
                        "p_pred": bin_prob_pred,
                    }
                )
            )
            plot = wandb.plot.line(table, "p_pred", "p_true", title=title)
            return plot

        wandb.log(
            {
                "eval/prefs/pred_calibration": plot_calibration_curve(curve_pred, title="Calibration Curve (pred)"),
                "eval/prefs/lower_calibration": plot_calibration_curve(curve_lower, title="Calibration Curve (lower)"),
                "eval/prefs/upper_calibration": plot_calibration_curve(curve_upper, title="Calibration Curve (upper)"),
            }
        )

    metrics = {
        "prefs/ece": ece,
        "prefs/mce": mce,
        "prefs/elce": elce,
        "prefs/mlce": mlce,
        "prefs/euce": euce,
        "prefs/muce": muce,
        "prefs/brier_score": brier_score,
    }
    if return_output:
        metrics["_output/calibration_statistics/pred"] = curve_pred
        metrics["_output/calibration_statistics/lower"] = curve_lower
        metrics["_output/calibration_statistics/upper"] = curve_upper
    return metrics


def confidence_statistics(results: dict[str, Any], **kwargs) -> dict[str, float | np.ndarray]:
    """Compute confidence statistics for predicted rewards.

    The `prefs/confident_true_rate` corresponds to the percentage of
    - non-overlapping (confident), correct (true) reward confidence intervals or
    - `reward_lower(y_chosen) > reward_upper(y_rejected)` or
    - `preference_lower(y_chosen > y_rejected) > 0.5`.

    The `prefs/confident_false_rate` corresponds to the percentage of
    - non-overlapping (confident), incorrect (false) reward confidence intervals or
    - `reward_upper(y_chosen) < reward_lower(y_rejected)` or
    - `preference_upper(y_chosen > y_rejected) <= 0.5`.

    The `prefs/unconfident_true_rate` corresponds to the percentage of
    - overlapping (unconfident), correct (true) reward confidence intervals or
    - `reward_lower(y_chosen) <= reward_upper(y_rejected) && reward(y_chosen) > reward(y_rejected)` or
    - `preference_lower(y_chosen > y_rejected) <= 0.5 && preference(y_chosen > y_rejected) > 0.5`.

    The `prefs/unconfident_false_rate` corresponds to the percentage of
    - overlapping (unconfident), incorrect (false) reward confidence intervals or
    - `reward_upper(y_chosen) >= reward_lower(y_rejected) && reward(y_chosen) <= reward(y_rejected)` or
    - `preference_upper(y_chosen > y_rejected) > 0.5 && preference(y_chosen > y_rejected) <= 0.5`.
    """
    if "rewards" not in results or len(results["rewards"]) == 0:
        raise ValueError("No rewards found in results. Cannot compute metric.")

    rewards = results["rewards"]
    weights = results.get("weights", np.ones(len(rewards)))

    true_rate = np.average(rewards[:, CHOSEN, PRED] > rewards[:, REJECTED, PRED], weights=weights)  # Aka win_rate
    false_rate = 1 - true_rate

    confident_true_rate = np.average(rewards[:, CHOSEN, LOWER] > rewards[:, REJECTED, UPPER], weights=weights)
    confident_false_rate = np.average(rewards[:, CHOSEN, UPPER] < rewards[:, REJECTED, LOWER], weights=weights)
    unconfident_true_rate = true_rate - confident_true_rate
    unconfident_false_rate = false_rate - confident_false_rate

    return {
        "prefs/confident_true_rate": confident_true_rate,
        "prefs/confident_false_rate": confident_false_rate,
        "prefs/unconfident_true_rate": unconfident_true_rate,
        "prefs/unconfident_false_rate": unconfident_false_rate,
    }


def ranking_scores(results: dict[str, Any], **kwargs) -> dict[str, float | np.ndarray]:
    """Compute ranking scores for predicted rewards."""
    if "rewards" not in results or len(results["rewards"]) == 0:
        raise ValueError("No rewards found in results. Cannot compute metric.")

    rewards = results["rewards"]
    weights = results.get("weights", np.ones(len(rewards)))

    true_rate = np.average(rewards[:, CHOSEN, PRED] > rewards[:, REJECTED, PRED], weights=weights)  # Aka win_rate
    false_rate = 1 - true_rate

    confident_true_rate = np.average(rewards[:, CHOSEN, LOWER] > rewards[:, REJECTED, UPPER], weights=weights)
    confident_false_rate = np.average(rewards[:, CHOSEN, UPPER] < rewards[:, REJECTED, LOWER], weights=weights)

    def _compute_score(beta):
        ranking_pos = _safe_div(confident_true_rate, (1 - beta) * true_rate + beta, default=0)
        ranking_neg = _safe_div(confident_false_rate, (1 - beta) * false_rate + beta, default=0)
        return ranking_pos - ranking_neg

    return {
        "ranking/0.0": _compute_score(0.0),
        "ranking/0.01": _compute_score(0.01),
        "ranking/0.2": _compute_score(0.2),
        "ranking/1.0": _compute_score(1.0),
    }


def combine_metrics(
    *metric_fns: MetricCallable,
    fn_kwargs: dict[str, dict[str, Any]] | None = None,
    **fn_kwargs_all,
) -> MetricCallable:
    """
    Combine multiple metric functions into a single metric function.

    Args:
        metric_fns (MetricCallable):
            A list of metric functions to combine.
        fn_kwargs (dict[str, dict[str, Any]] | None):
            A dictionary with keyword arguments passed to specific metric functions.
        fn_kwargs_all (dict[str, Any]):
            Additional keyword arguments passed to all metric functions.

    Returns:
        out (MetricCallable):
            A function that computes the combined metric.
    """
    if fn_kwargs is None:
        fn_kwargs = {}

    # Ensure all metric functions exist for which kwargs are provided
    metric_fn_names = [metric_fn.__name__ for metric_fn in metric_fns]
    for fn_name in fn_kwargs:
        if fn_name not in metric_fn_names:
            raise ValueError(
                f"Metric function '{fn_name}' cannot be found. "
                f"Please provide keyword arguments only for {metric_fn_names}."
            )

    def combined_metrics_fn(results: dict[str, Any], prefix: str = "", **kwargs) -> dict[str, float | np.ndarray]:
        metrics_all = {}
        for metric_fn in metric_fns:
            # Evaluate metric function
            metrics = metric_fn(results, **fn_kwargs_all, **kwargs, **fn_kwargs.get(metric_fn.__name__, {}))
            # Add metrics to dictionary
            for key, value in metrics.items():
                key = f"{prefix}{key}"
                if key in metrics_all:
                    raise ValueError(
                        f"Metric '{key}' already exists in the combined metrics. "
                        "Please ensure that the metric keys are unique."
                    )
                metrics_all[key] = value
        return metrics_all

    return combined_metrics_fn


compute_default_metrics = combine_metrics(
    reward_statistics,
    preference_statistics,
    win_rate,
    calibration_statistics,
    confidence_statistics,
    ranking_scores,
)
