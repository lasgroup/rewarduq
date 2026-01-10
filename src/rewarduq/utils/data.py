"""Dataset loading and preprocessing utilities for preference data."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np
from accelerate import PartialState
from datasets import (
    Dataset,
    Features,
    IterableDataset,
    Sequence,
    Value,
    concatenate_datasets,
    disable_progress_bars,
    enable_progress_bars,
    load_dataset,
)
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import PreTrainedTokenizerBase
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt

from .logging import get_logger

logger = get_logger(__name__)


def load_dataset_from_config(
    dataset_configs: ListConfig | DictConfig,
    extra_columns: dict[str, Any] | None = None,
    desc: str | None = None,
) -> Dataset:
    """Load datasets from the given configuration.

    Args:
        dataset_configs (ListConfig): A list of dataset configurations, where each
            configuration is a dictionary with the keys `path`, `split`, and `num_samples`.
            The specified dataset must contain the columns `chosen` and `rejected`.
            If `num_samples` is an integer, it specifies the number of samples to load.
            If `num_samples` is a float, it specifies the fraction of samples to load.

    Returns:
        Dataset: A concatenated dataset containing the samples from all specified datasets.
    """
    if OmegaConf.is_dict(dataset_configs):
        dataset_configs = ListConfig([dataset_configs])
    if extra_columns is None:
        extra_columns = {}

    datasets = []
    features = {
        "chosen": [{"role": Value(dtype="string"), "content": Value(dtype="string")}],
        "rejected": [{"role": Value(dtype="string"), "content": Value(dtype="string")}],
        "weight": Value(dtype="float32"),
        "source": Value(dtype="string"),
    }
    features = Features({**features, **extra_columns})

    # Disable dataset progress bars
    disable_progress_bars()

    # Load each dataset
    for dataset_config in dataset_configs:
        source = f"{dataset_config.path}/{dataset_config.split}:{dataset_config.num_samples}"

        # Load dataset
        name = dataset_config.get("name", None)
        dataset = load_dataset(dataset_config.path, name=name, split=dataset_config.split)

        # Filter dataset
        if "filter" in dataset_config:
            dataset = dataset.filter(
                lambda sample: sample[dataset_config.filter.column] in dataset_config.filter["values"],  # noqa: B023
            )

        # Sample from dataset
        if dataset_config.num_samples == "all":
            pass
        else:
            if isinstance(dataset_config.num_samples, int):
                num_samples = dataset_config.num_samples
            elif isinstance(dataset_config.num_samples, float):
                num_samples = int(len(dataset) * dataset_config.num_samples)
            else:
                raise ValueError(f"Unsupported num_samples: {dataset_config.num_samples}")
            rng = np.random.default_rng(seed=0)  # Use fixed random generator for reproducibility
            indices = rng.choice(len(dataset), size=num_samples, replace=False)
            dataset = dataset.select(indices)

        # Compute sample weights
        if dataset_config.path == "allenai/reward-bench" and dataset_config.split == "filtered":
            from .rewardbench import compute_rewardbench_weights

            weights = compute_rewardbench_weights(dataset)
        else:
            weights = np.ones(len(dataset))
        dataset = dataset.add_column("weight", weights)

        # Convert dataset to conversational if necessary
        def maybe_convert_to_conversational(sample):
            if "prompt" in sample and not isinstance(sample["prompt"], list):
                sample["prompt"] = [{"role": "user", "content": sample["prompt"]}]
            if "chosen" in sample and not isinstance(sample["chosen"], list):
                sample["chosen"] = [{"role": "assistant", "content": sample["chosen"]}]
            if "rejected" in sample and not isinstance(sample["rejected"], list):
                sample["rejected"] = [{"role": "assistant", "content": sample["rejected"]}]
            return sample

        dataset = dataset.map(maybe_convert_to_conversational)

        # Append prompt to chosen and rejected if necessary
        def append_prompt(sample):
            sample["chosen"] = sample["prompt"] + sample["chosen"]
            sample["rejected"] = sample["prompt"] + sample["rejected"]
            return sample

        if "prompt" in dataset.column_names:
            dataset = dataset.map(append_prompt, remove_columns=["prompt"])

        # Unify dataset format to allow concatenation
        if "source" in dataset.column_names:
            dataset = dataset.rename_column("source", "source_")
        dataset = dataset.add_column("source", [source] * len(dataset))
        dataset = dataset.select_columns(features.keys())
        dataset = dataset.cast(features)  # Ensure features have consistent structure

        datasets.append(dataset)

    # Concatenate datasets
    datasets = concatenate_datasets(datasets)

    # Log dataset summary
    counts = Counter(datasets["source"])
    total = sum(counts.values())  # For Python 3.9 compatibility, as `Counter.total()` is not available
    logger.info(
        f"Loaded dataset{'' if desc is None else f' ({desc})'}: {total} samples"
        f" from {len(counts)} dataset{'' if len(counts) == 1 else 's'}"
        "\n" + "\n".join(f"{source} ({count} samples)" for source, count in counts.items())
    )

    # Enable dataset progress bars
    enable_progress_bars()

    return datasets


def load_datasets_from_config(config: DictConfig):
    """Load train and eval datasets from the given configuration."""
    datasets = {
        dataset_group: load_dataset_from_config(config.dataset[dataset_group], desc=dataset_group)
        for dataset_group in config.dataset
    }

    if "train" in datasets:
        train_dataset = datasets.pop("train")
    else:
        raise ValueError("Training dataset must be specified in the config under `datasets.train`.")

    if "eval" in datasets:
        eval_dataset = datasets.pop("eval")
    else:
        eval_dataset = {
            dataset_group[len("eval/") :]: datasets.pop(dataset_group)
            for dataset_group in list(datasets.keys())  # Copy keys since we pop items from datasets
            if dataset_group.startswith("eval/")
        } or None

    if len(datasets) > 0:
        raise ValueError(f"Unknown dataset groups: {list(datasets.keys())}")

    return train_dataset, eval_dataset


def prepare_preference_dataset(
    dataset: Dataset | IterableDataset,
    processing_class: PreTrainedTokenizerBase,
    dataset_name: str,
    dataset_num_proc: int | None = None,
    tools: list[dict | Callable] | None = None,
    max_length: int | None = None,
) -> Dataset | IterableDataset:
    """Prepare preference dataset by applying common preprocessing steps.

    The dataset is processed in the following steps:
    1. Extract prompt from `chosen` and `rejected` if needed.
    2. Apply chat template to `prompt`, `chosen`, and `rejected` if needed.
    3. Tokenize `prompt`, `chosen`, and `rejected`.
    4. Remove samples with total length longer than `args.max_length`.

    The final dataset consists of the following columns:
    - `prompt_input_ids`: The tokenized prompt.
    - `chosen_input_ids`: The tokenized chosen completion.
    - `rejected_input_ids`: The tokenized rejected completion.

    This implementation is inspired from `trl.DPOTrainer._prepare_dataset` [1] and
    `trl.RewardTrainer._prepare_dataset` [2].

    References:
        [1] https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/dpo_trainer.py#L530
        [2] https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/reward_trainer.py#L436
    """
    map_kwargs = {}
    if isinstance(dataset, Dataset):
        map_kwargs["num_proc"] = dataset_num_proc

    with PartialState().main_process_first():
        # Extract prompt if needed
        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
        dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

        # Apply chat template if needed
        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
        dataset = dataset.map(
            maybe_apply_chat_template,
            fn_kwargs={"tokenizer": processing_class, "tools": tools},
            **map_kwargs,
        )

        # Tokenize the dataset
        def _tokenize(samples, processing_class):
            if not isinstance(samples["chosen"], list):
                raise ValueError("Expected input of `_tokenize` to be batched.")

            # Add EOS token if needed
            # NOTE: Chat templates often end with the EOS token followed by a newline, for which we also skip adding
            # another EOS token as in `trl.RewardTrainer._prepare_dataset` [2].
            eos_suffixes = (processing_class.eos_token, processing_class.eos_token + "\n")
            samples["chosen"] = [
                sample if sample.endswith(eos_suffixes) else sample + processing_class.eos_token
                for sample in samples["chosen"]
            ]
            samples["rejected"] = [
                sample if sample.endswith(eos_suffixes) else sample + processing_class.eos_token
                for sample in samples["rejected"]
            ]

            # Tokenize samples
            prompt_input_ids = processing_class(samples["prompt"])["input_ids"]
            chosen_input_ids = processing_class(samples["chosen"])["input_ids"]
            rejected_input_ids = processing_class(samples["rejected"])["input_ids"]

            return {
                "prompt_input_ids": prompt_input_ids,
                "chosen_input_ids": chosen_input_ids,
                "rejected_input_ids": rejected_input_ids,
            }

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
        if isinstance(dataset, IterableDataset):
            features = Features(
                {
                    "prompt_input_ids": Sequence(Value("int64")),
                    "chosen_input_ids": Sequence(Value("int64")),
                    "rejected_input_ids": Sequence(Value("int64")),
                }
            )
        else:
            features = None

        dataset = dataset.map(
            _tokenize,
            fn_kwargs={"processing_class": processing_class},
            batched=True,
            remove_columns=["prompt", "chosen", "rejected"],
            features=features,
            **map_kwargs,
        )

        if isinstance(dataset, IterableDataset) and features is not None:
            mapped_dataset = dataset
            dataset = IterableDataset.from_generator(lambda: (x for x in mapped_dataset), features=features)

        # Filter samples longer than max_length
        if max_length is not None:
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Filtering {dataset_name} dataset"
            dataset = dataset.filter(
                lambda sample: (
                    len(sample["prompt_input_ids"]) + len(sample["chosen_input_ids"]) <= max_length
                    and len(sample["prompt_input_ids"]) + len(sample["rejected_input_ids"]) <= max_length
                ),
                **map_kwargs,
            )

    return dataset
