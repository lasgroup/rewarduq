from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from natsort import natsort_keygen
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from rewarduq.metrics import PRED, UPPER, compute_default_metrics


def _reconstruct_rewards_mean_std(
    rewards_all: np.ndarray,
    beta_orig: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct rewards mean and std from predictions."""
    rewards_mean = rewards_all[:, :, PRED]
    rewards_std = (rewards_all[:, :, UPPER] - rewards_all[:, :, PRED]) / beta_orig
    return rewards_mean, rewards_std


def _optimize_over_beta(
    rewards_mean: np.ndarray,
    rewards_std: np.ndarray,
    metric_key: str,
    metric_weights: np.ndarray | None = None,
    betas: list[float] | None = None,
) -> float:
    """Compute metrics over beta sweep and find optimal beta."""
    if betas is None:
        betas = np.logspace(-2, 0, 10, endpoint=False).tolist() + np.linspace(1, 10, 91).tolist()

    # Compute metrics over beta
    results = []
    for beta in betas:
        rewards_lower = rewards_mean - beta * rewards_std
        rewards_upper = rewards_mean + beta * rewards_std
        rewards_all = np.stack([rewards_mean, rewards_lower, rewards_upper], axis=2)  # shape: (batch_size, 2, 3)
        metrics = compute_default_metrics({"rewards": rewards_all, "weights": metric_weights})
        results.append(metrics[metric_key])

    return betas[np.argmax(results)]


def load_configs(paths: list[str | Path], config_keys: list[str]) -> pd.DataFrame:
    """Load configs from results folders."""
    paths = [Path(path) for path in paths]

    results = []
    for path in paths:
        # Load config
        path_config = path / ".hydra" / "config.yaml"
        config = OmegaConf.load(path_config)

        result = {}
        for key in config_keys:
            result[key] = OmegaConf.select(config, key)
        result["path"] = str(path)
        results.append(result)

    # Create DataFrame and sort by config keys
    df_results = pd.DataFrame(results)
    df_results.sort_values(config_keys, inplace=True, key=natsort_keygen())
    return df_results


def load_metrics(
    paths: list[str | Path],
    config_keys: list[str],
    metric_keys: list[str],
    metric_weights: np.ndarray | None = None,
    beta_mapping: dict[str, str] | None = None,
    beta_optimal: bool = False,
    steps: int | list[int] | Literal["all", "final"] = "final",
) -> pd.DataFrame:
    """Load predictions from results folders and evaluate metrics."""
    paths = [Path(path) for path in paths]
    if isinstance(steps, int):
        steps = [steps]
    if beta_optimal and beta_mapping is None:
        raise ValueError("beta_mapping must be provided when beta_optimal is True.")

    results = []
    for path in tqdm(paths, desc="Loading metrics"):
        # Load config
        path_config = path / ".hydra" / "config.yaml"
        config = OmegaConf.load(path_config)

        # Resolve paths
        if isinstance(steps, list):
            path_predictions = [path / "predictions" / f"rewards_{step}.npy" for step in steps]
        elif steps == "all":
            path_predictions = (path / "predictions").glob("rewards_*.npy")
        elif steps == "final":
            path_predictions = (path / "predictions").glob("rewards_*.npy")
            path_predictions = [max(path_predictions, key=lambda path: int(path.stem.split("_")[1]))]
        else:
            raise ValueError(f"Unsupported steps specifier: {steps}")

        # Evaluate predictions
        for path_prediction in path_predictions:
            # Load predictions
            rewards_all = np.load(path_prediction)
            if beta_mapping is not None:
                beta_orig = OmegaConf.select(config, beta_mapping[config["pipeline"]])
                if beta_optimal:
                    # Reconstruct rewards mean and std and optimize beta
                    rewards_mean, rewards_std = _reconstruct_rewards_mean_std(rewards_all, beta_orig)
                    beta_opt = _optimize_over_beta(rewards_mean, rewards_std, "ranking/0.2", metric_weights)
                    # Reconstruct rewards_all with optimal beta
                    rewards_lower = rewards_mean - beta_opt * rewards_std
                    rewards_upper = rewards_mean + beta_opt * rewards_std
                    rewards_all = np.stack(
                        [rewards_mean, rewards_lower, rewards_upper], axis=2
                    )  # shape: (batch_size, 2, 3)
            # Compute metrics
            metrics = compute_default_metrics({"rewards": rewards_all, "weights": metric_weights}, return_output=True)
            # Add to result
            result = {}
            for key in config_keys:
                result[key] = OmegaConf.select(config, key)
            if beta_mapping is not None:
                result["beta_orig"] = beta_orig
                if beta_optimal:
                    result["beta_opt"] = beta_opt
            result["step"] = int(path_prediction.stem.split("_")[1])
            for key in metric_keys:
                result[key] = metrics[key]
            result["path"] = str(path)
            results.append(result)

    # Create DataFrame and sort by config keys and step
    df_results = pd.DataFrame(results)
    df_results.sort_values(config_keys + ["step"], inplace=True, key=natsort_keygen())
    return df_results


def load_predictions(
    paths: list[str | Path],
    config_keys: list[str],
    beta_mapping: dict[str, str],
    steps: int | list[int] | Literal["all", "final"] = "final",
) -> pd.DataFrame:
    """Load predictions from results folders and reconstruct rewards mean and std."""
    paths = [Path(path) for path in paths]
    if isinstance(steps, int):
        steps = [steps]

    results = []
    for path in tqdm(paths, desc="Loading predictions"):
        # Load config
        path_config = path / ".hydra" / "config.yaml"
        config = OmegaConf.load(path_config)

        # Resolve beta
        beta_orig = OmegaConf.select(config, beta_mapping[config["pipeline"]])

        # Resolve paths
        if isinstance(steps, list):
            path_predictions = [path / "predictions" / f"rewards_{step}.npy" for step in steps]
        elif steps == "all":
            path_predictions = (path / "predictions").glob("rewards_*.npy")
        elif steps == "final":
            path_predictions = (path / "predictions").glob("rewards_*.npy")
            path_predictions = [max(path_predictions, key=lambda path: int(path.stem.split("_")[1]))]
        else:
            raise ValueError(f"Unsupported steps specifier: {steps}")

        # Evaluate predictions
        for path_prediction in path_predictions:
            # Load predictions
            rewards_all = np.load(path_prediction)
            # Reconstruct rewards mean and std
            rewards_mean, rewards_std = _reconstruct_rewards_mean_std(rewards_all, beta_orig)
            # Add to result
            result = {}
            for key in config_keys:
                result[key] = OmegaConf.select(config, key)
            result["step"] = int(path_prediction.stem.split("_")[1])
            result["rewards_mean"] = rewards_mean
            result["rewards_std"] = rewards_std
            result["beta_orig"] = beta_orig
            result["path"] = str(path)
            results.append(result)

    # Create DataFrame and sort by config keys and step
    df_results = pd.DataFrame(results)
    df_results.sort_values(config_keys + ["step"], inplace=True, key=natsort_keygen())
    return df_results
