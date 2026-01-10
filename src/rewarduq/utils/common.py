"""Common utility functions for configuration, logging, and reproducibility."""

from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import rich
import rich.console
import rich.syntax
import rich.tree
import torch
from accelerate import PartialState
from omegaconf import DictConfig, ListConfig, OmegaConf

from .logging import get_logger

logger = get_logger(__name__)


def get_config(config_path: str | Path, overwrite: list[str] | dict[str, Any] | None = None) -> DictConfig | ListConfig:
    """Load a configuration file and optionally overwrite specific values.

    Args:
        config_path (str | Path):
            Path to the configuration file to load.

        overwrite (list[str] | dict[str, Any] | None):
            Optional configuration overrides. Can be either:
            - A list of dotlist strings (e.g., ["key1=value1", "key2=value2"])
            - A dictionary of key-value pairs to override
            - None for no overrides

    Returns:
        config (DictConfig | ListConfig):
            The loaded configuration with any specified overrides applied.
    """
    if overwrite is None:
        overwrite = []
    elif isinstance(overwrite, dict):
        overwrite = [f"{key}={value if value is not None else 'null'}" for key, value in overwrite.items()]

    # Load the config file
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded config file: {config_path}")

    # Overwrite the config
    if len(overwrite) > 0:
        config.merge_with_dotlist(overwrite)
        logger.info(f"Overwrote config with:\n{OmegaConf.to_yaml(OmegaConf.from_dotlist(overwrite))}")

    return config


def print_config_tree(
    cfg: DictConfig,
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        cfg (DictConfig):
            The DictConfig to print.

        resolve (bool):
            Whether to resolve reference fields of DictConfig. Defaults to False.

        save_to_file (bool):
            Whether to export config to the hydra output folder. Defaults to False.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in cfg:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        branch_content: str
        if isinstance(config_group, DictConfig):
            yaml_output = OmegaConf.to_yaml(config_group, resolve=resolve)
            branch_content = str(yaml_output) if yaml_output is not None else ""
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # Print config tree
    if PartialState().is_main_process:
        rich.print(tree)

    # Save config tree to file
    if save_to_file:
        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Use a wider console for file output to avoid path truncation
        with open(output_dir / "config_tree.log", "w") as file:
            console = rich.console.Console(file=file, width=200)
            console.print(tree)


def get_wandb_context_from_config(config: DictConfig) -> contextlib.AbstractContextManager:
    """Create a Weights & Biases context manager from configuration.

    If W&B is enabled in the config and running on the main process, this returns
    an initialized wandb.init() context. Otherwise, returns a null context that
    does nothing.

    Args:
        config (DictConfig):
            Configuration containing:
            - `trainer.report_to`: Reporting integrations (e.g., ["wandb", "tensorboard"])
            - `wandb.entity`: W&B entity name
            - `wandb.project`: W&B project name
            - `wandb.resume`: Optional run ID to resume

    Returns:
        context (contextlib.AbstractContextManager):
            Either an active wandb.init() context or a null context manager.
    """
    report_to = normalize_report_to(config.trainer.report_to)

    if PartialState().is_main_process and ("all" in report_to or "wandb" in report_to):
        import wandb

        # Init wandb run
        kwargs = {
            "entity": config.wandb.entity,
            "project": config.wandb.project,
            "config": OmegaConf.to_object(config),
        }
        if config.wandb.resume is not None:
            kwargs["resume"] = "must"
            kwargs["id"] = config.wandb.resume
        context_wandb = wandb.init(**kwargs)
    else:
        context_wandb = contextlib.nullcontext()

    return context_wandb


def ensure_reproducibility(seed: int | None = None, deterministic: bool = False):
    """Set seeds and ensure usage of deterministic algorithms.

    Args:
        seed (int, optional): The seed set for each dataloader worker. Defaults
            to None.
        deterministic (bool, optional): Flag whether algorithms should be as
            deterministic as possible. Defaults to False.

    References:
        [1] https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Seed random number generators
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"Set seeds: {seed}")

    # Use deterministic algorithms
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.info("Enabled deterministic algorithms.")


def check_supported_args(args: Any, key: str, supported_values: list[Any], temp: bool = False) -> None:
    """Check that a configuration argument has a supported value.

    Args:
        args (Any):
            Object containing arguments (either a dataclass or dictionary).

        key (str):
            Name of the argument to check.

        supported_values (list[Any]):
            List of allowed values for this argument.

        temp (bool):
            If True, indicates this is a temporary limitation and error message
            will say "Not yet supported". If False, indicates permanent limitation
            and error message will say "Unsupported". Defaults to False.

    Raises:
        ValueError:
            If the argument value is not in the supported_values list.
    """
    value = getattr(args, key) if hasattr(args, key) else args[key]
    if value not in supported_values:
        raise ValueError(
            f"{'Not yet supported' if temp else 'Unsupported'} value for {key}: {value}."
            f" Supported values are: {supported_values}."
        )


def normalize_report_to(report_to: str | list | None) -> list[str]:
    """Normalize the `report_to` parameter into a list of reporting integrations."""
    if PartialState().is_main_process:
        if report_to is None:
            return ["all"]
        elif report_to == "none":
            return []
        elif isinstance(report_to, str):
            return [report_to]
        elif isinstance(report_to, list):
            return report_to
        else:
            raise ValueError(
                f"Invalid report_to value: {report_to}. Must be 'all', 'none', a string, or a list of strings."
            )
    else:
        return []
