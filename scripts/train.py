import datetime
import os

import hydra
from accelerate import Accelerator, InitProcessGroupKwargs
from omegaconf import DictConfig

from rewarduq.methods import load_pipeline_from_config
from rewarduq.utils import (
    ensure_reproducibility,
    get_logger,
    get_wandb_context_from_config,
    load_datasets_from_config,
    print_config_tree,
    setup_logging,
)

logger = get_logger(__name__)

# NOTE: We increase the timeout for NCCL from 10min to 30min to prevent timeouts while tokenizing large datasets.
# Importantly, this must be done in the first `Accelerator` instantiation globally.
# Reference: https://github.com/huggingface/accelerate/issues/223#issuecomment-1008792609
accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=1800))])


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(config: DictConfig) -> None:
    """Main training entry point for reward models.

    This function orchestrates the entire training workflow including:
    - Setting up logging and reproducibility
    - Loading training and evaluation datasets
    - Initializing the reward model pipeline
    - Training the model with optional checkpoint resumption

    Args:
        config (DictConfig):
            Hydra configuration containing all training parameters.
    """
    # Configure logging
    setup_logging(
        filterwarnings=[
            dict(
                action="ignore",
                message="functools.partial will be a method descriptor in future Python versions",
                category=FutureWarning,
                module="torch.distributed.algorithms.ddp_comm_hooks",
            )
        ],
    )

    # Print config
    print_config_tree(config)

    # Create output folder
    logger.info(f"Output folder: {config.paths.output_dir}")
    os.makedirs(config.paths.output_dir, exist_ok=True)

    # Initialize W&B context
    context_wandb = get_wandb_context_from_config(config)

    with context_wandb:
        # Set seeds and use deterministic algorithms
        ensure_reproducibility(seed=config.seed, deterministic=config.deterministic)

        # Load datasets
        train_dataset, eval_dataset = load_datasets_from_config(config)

        # Load reward model pipeline
        rm_pipeline = load_pipeline_from_config(config)
        logger.info(f"Loaded pipeline: {config.pipeline}")

        # Train reward model
        rm_pipeline.train(train_dataset, eval_dataset, resume_from_checkpoint=config.resume)
        logger.info("Finished training.")


if __name__ == "__main__":
    train()
