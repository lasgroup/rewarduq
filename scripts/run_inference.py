from __future__ import annotations

import argparse
import datetime
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_pt_utils import EvalLoopContainer, nested_detach

from rewarduq.methods import auto_register_models
from rewarduq.trainers import DataCollatorForPreference
from rewarduq.utils import (
    ensure_reproducibility,
    get_logger,
    load_dataset_from_config,
    prepare_preference_dataset,
    setup_logging,
)

logger = get_logger(__name__)

# NOTE: We increase the timeout for NCCL from 10min to 30min to prevent timeouts while tokenizing large datasets.
# Importantly, this must be done in the first `Accelerator` instantiation globally.
# Reference: https://github.com/huggingface/accelerate/issues/223#issuecomment-1008792609
accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=1800))])

auto_register_models()


def _prepare_input(data: torch.Tensor | Any) -> dict[str, torch.Tensor | Any]:
    """Taken from `transformers.Trainer._prepare_inputs` [1].

    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.

    References:
        [1] https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/trainer.py
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(accelerator.device)
    return data


def prediction_step(
    model: PreTrainedModel | nn.Module,
    inputs: dict[str, torch.Tensor | Any],
) -> dict[str, torch.Tensor | Any]:
    # Prepare inputs
    inputs = _prepare_input(inputs)

    # Compute outputs
    with torch.no_grad():
        outputs_chosen = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            completion_mask=inputs["chosen_completion_mask"],
        )
        outputs_rejected = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            completion_mask=inputs["rejected_completion_mask"],
        )
        outputs_chosen = nested_detach(outputs_chosen)
        outputs_rejected = nested_detach(outputs_rejected)

    # Keep only rewards in the outputs
    outputs_chosen = {
        "rewards": outputs_chosen["rewards"],
    }
    outputs_rejected = {
        "rewards": outputs_rejected["rewards"],
    }

    return {
        "chosen": outputs_chosen,
        "rejected": outputs_rejected,
    }


def predict(
    model: PreTrainedModel | nn.Module,
    processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
    dataset: Dataset,
    accumulation_steps: int | None = None,
    max_length: int = 2048,
    batch_size: int = 16,
):
    """Run inference on a preference dataset using a reward model."""

    # Prepare dataset and dataloader
    dataset = prepare_preference_dataset(
        dataset=dataset,
        processing_class=processing_class,
        dataset_name="eval",
        max_length=max_length,
    )
    data_collator = DataCollatorForPreference(pad_token_id=processing_class.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)

    # Prepare model
    model.eval()
    model = accelerator.prepare_model(model, evaluation_mode=True)

    # Prepare reference model if needed (only relevant for DPO models)
    if hasattr(model, "init_ref_model"):
        model.init_ref_model()
        model.ref_model = accelerator.prepare_model(model.ref_model, evaluation_mode=True)

    # Main inference loop
    outputs_chosen_all = EvalLoopContainer(do_nested_concat=False)
    outputs_rejected_all = EvalLoopContainer(do_nested_concat=False)
    for step, inputs in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        disable=not accelerator.is_main_process,
    ):
        # Predict outputs
        outputs = prediction_step(model, inputs)
        outputs_chosen = outputs["chosen"]
        outputs_rejected = outputs["rejected"]
        # Gather outputs across GPUs
        outputs_chosen = accelerator.gather_for_metrics(outputs_chosen)  # Shape: (batch_size * num_devices, 3)
        outputs_rejected = accelerator.gather_for_metrics(outputs_rejected)  # Shape: (batch_size * num_devices, 3)
        # Add outputs to containers
        outputs_chosen_all.add(outputs_chosen)
        outputs_rejected_all.add(outputs_rejected)

        # Move outputs to CPU every accumulation_steps
        if accumulation_steps is not None and (step + 1) % accumulation_steps == 0:
            outputs_chosen_all.to_cpu_and_numpy()
            outputs_rejected_all.to_cpu_and_numpy()

            del outputs_chosen, outputs_rejected
            torch.cuda.empty_cache()

    # Move remaining outputs to CPU and convert EvalLoopContainer to lists
    outputs_chosen_all = outputs_chosen_all.get_arrays()
    outputs_rejected_all = outputs_rejected_all.get_arrays()

    # Combine rewards to a single array
    rewards_chosen_all = np.concatenate([outputs["rewards"] for outputs in outputs_chosen_all], axis=0)
    rewards_rejected_all = np.concatenate([outputs["rewards"] for outputs in outputs_rejected_all], axis=0)
    rewards_all = np.stack([rewards_chosen_all, rewards_rejected_all], axis=1)  # Shape: (num_samples, 2, 3)

    # Retrieve number of samples
    num_samples = len(dataloader.dataset)
    if num_samples != len(rewards_all):
        raise RuntimeError(
            f"Number of evaluation samples ({num_samples}) does not match "
            f"the number of predictions ({len(rewards_all)})."
        )

    return rewards_all


def main(args):
    # Create output folder
    os.makedirs(args.out, exist_ok=True)
    logger.info(f"Output folder: {args.out}")

    # Set seeds and use deterministic algorithms
    ensure_reproducibility(seed=0, deterministic=args.deterministic)

    # Load dataset
    dataset = load_dataset_from_config(
        OmegaConf.create([{"path": args.dataset[0], "split": args.dataset[1], "num_samples": "all"}]),
        desc="eval",
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model,
        # device_map="auto",
        # dtype=torch.bfloat16, # TODO required when base model is loaded as bfloat16, but head is float32
    )
    # logger.info(f"Device map of loaded model: {model.hf_device_map}")

    if args.debug:
        # Limit dataset size
        size = int(1.5 * accelerator.num_processes * args.batch_size)
        dataset = dataset.select(range(size))

    # Run inference
    rewards = predict(
        model,
        tokenizer,
        dataset,
        batch_size=args.batch_size,
    )

    # Save predictions
    if accelerator.is_main_process:
        path = Path(args.out) / "rewards.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, rewards)
        logger.info(f"Saved predictions: {path}")


if __name__ == "__main__":
    # Configure logging
    setup_logging()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The pretrained model name or path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs=2,
        metavar=("DATASET", "SPLIT"),
        default=["trl-lib/ultrafeedback_binarized", "test"],
        help="The dataset and split to use for inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="The batch size to use for inference.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default="output",
        help="The path to the output folder. (default: output)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Flag whether to use deterministic algorithms. (default: False)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag whether to run in debug mode, which limits the dataset size. (default: False)",
    )
    args = parser.parse_args()

    main(args)
