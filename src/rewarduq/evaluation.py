from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsort_keygen
from omegaconf import OmegaConf

from rewarduq.metrics import PRED, UPPER, compute_default_metrics


def load_configs(paths: list[Path], config_keys: list[str]) -> pd.DataFrame:
    """Load configs from results folders."""
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
    paths: list[Path],
    config_keys: list[str],
    metric_keys: list[str],
    metric_weights: np.ndarray,
    steps: str = "final",
) -> pd.DataFrame:
    """Load predictions from results folders and evaluate metrics."""
    results = []
    for path in paths:
        # Load config
        path_config = path / ".hydra" / "config.yaml"
        config = OmegaConf.load(path_config)

        # Resolve paths
        path_predictions = (path / "predictions").glob("rewards_*.npy")
        if steps == "all":
            pass
        elif steps == "final":
            path_predictions = [max(path_predictions, key=lambda path: int(path.stem.split("_")[1]))]
        else:
            raise ValueError(f"Unsupported steps specifier: {steps}")

        # Evaluate predictions
        for path_prediction in path_predictions:
            # Load predictions
            rewards_all = np.load(path_prediction)
            # Compute metrics
            metrics = compute_default_metrics(
                {"rewards": rewards_all, "weights": metric_weights},
                return_output=True,
                report_to="none",
            )
            # Add to result
            result = {}
            for key in config_keys:
                result[key] = OmegaConf.select(config, key)
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
    paths: list[Path],
    config_keys: list[str],
    beta_mapping: dict[str, str],
    steps: str = "final",
) -> pd.DataFrame:
    """Load predictions from results folders and reconstruct rewards mean and std."""
    results = []
    for path in paths:
        # Load config
        path_config = path / ".hydra" / "config.yaml"
        config = OmegaConf.load(path_config)

        # Resolve beta
        beta_orig = OmegaConf.select(config, beta_mapping[config["pipeline"]])

        # Resolve paths
        path_predictions = (path / "predictions").glob("rewards_*.npy")
        if steps == "all":
            pass
        elif steps == "final":
            path_predictions = [max(path_predictions, key=lambda path: int(path.stem.split("_")[1]))]
        else:
            raise ValueError(f"Unsupported steps specifier: {steps}")

        # Evaluate predictions
        for path_prediction in path_predictions:
            # Load predictions
            rewards_all = np.load(path_prediction)
            # Reconstruct rewards mean and std
            rewards_mean = rewards_all[:, :, PRED]
            rewards_std = (rewards_all[:, :, UPPER] - rewards_all[:, :, PRED]) / beta_orig
            # Add to result
            result = {}
            for key in config_keys:
                result[key] = OmegaConf.select(config, key)
            result["step"] = int(path_prediction.stem.split("_")[1])
            result["rewards_mean"] = rewards_mean
            result["rewards_std"] = rewards_std
            result["path"] = str(path)
            results.append(result)

    # Create DataFrame and sort by config keys and step
    df_results = pd.DataFrame(results)
    df_results.sort_values(config_keys + ["step"], inplace=True, key=natsort_keygen())
    return df_results
