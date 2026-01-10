from __future__ import annotations

import importlib
import pkgutil

from omegaconf import DictConfig

from .base import BasePipeline, RewardUQModel
from .mlp_head import MLPHead


def load_pipeline_from_config(config: DictConfig) -> BasePipeline:
    """Load a pipeline from a configuration.

    Args:
        config (DictConfig):
            The configuration.

    Returns:
        pipeline (BasePipeline):
            The pipeline.
    """
    module_name, class_name = config.pipeline.rsplit(".", maxsplit=1)
    PipelineClass = getattr(importlib.import_module(f".{module_name}", package=__package__), class_name)
    return PipelineClass.from_config(config)


def auto_register_models():
    """Automatically discover and import all model submodules to register them in transformers.

    This method is useful when using `AutoModel.from_pretrained` to load a model from this package.
    """
    # Import all submodules in the current package
    for module in pkgutil.iter_modules(__path__):
        importlib.import_module(f".{module.name}", package=__package__)
