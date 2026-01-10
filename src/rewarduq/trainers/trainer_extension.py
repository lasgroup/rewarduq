"""Extensions to HuggingFace Trainer for uncertainty-aware reward models."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, IterableDataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import EvalLoopContainer, nested_detach, nested_numpify
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize
from trl.trainer.utils import pad

from rewarduq.metrics import compute_default_metrics
from rewarduq.trainers.callbacks import EvaluateSaveCallback
from rewarduq.utils import check_supported_args, get_logger, prepare_preference_dataset

logger = get_logger(__name__)


@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing the following keys:
    - `prompt_input_ids`: The tokenized prompt.
    - `chosen_input_ids`: The tokenized chosen completion.
    - `rejected_input_ids`: The tokenized rejected completion.

    The collator returns a dictionary containing the input IDs as tensors, padded to the maximum length of the batch,
    and the corresponding attention masks.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.
    """

    pad_token_id: int
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        output = {}

        # Convert input IDs to tensors
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]

        # Create attention masks (before padding input IDs)
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]

        # Pad input IDs
        pad_kwargs = {"padding_value": self.pad_token_id, "pad_to_multiple_of": self.pad_to_multiple_of}
        prompt_input_ids = pad(prompt_input_ids, padding_side="left", **pad_kwargs)
        chosen_input_ids = pad(chosen_input_ids, padding_side="right", **pad_kwargs)
        rejected_input_ids = pad(rejected_input_ids, padding_side="right", **pad_kwargs)
        output["chosen_input_ids"] = torch.cat((prompt_input_ids, chosen_input_ids), dim=1)
        output["rejected_input_ids"] = torch.cat((prompt_input_ids, rejected_input_ids), dim=1)

        # Pad attention masks
        pad_kwargs = {"padding_value": 0, "pad_to_multiple_of": self.pad_to_multiple_of}
        prompt_attention_mask = pad(prompt_attention_mask, padding_side="left", **pad_kwargs)
        chosen_attention_mask = pad(chosen_attention_mask, padding_side="right", **pad_kwargs)
        rejected_attention_mask = pad(rejected_attention_mask, padding_side="right", **pad_kwargs)
        output["chosen_attention_mask"] = torch.cat((prompt_attention_mask, chosen_attention_mask), dim=1)
        output["rejected_attention_mask"] = torch.cat((prompt_attention_mask, rejected_attention_mask), dim=1)

        # Create completion masks (used as loss mask for DPO-based reward models)
        output["chosen_completion_mask"] = torch.cat(
            (torch.zeros_like(prompt_attention_mask), chosen_attention_mask),
            dim=1,
        )
        output["rejected_completion_mask"] = torch.cat(
            (torch.zeros_like(prompt_attention_mask), rejected_attention_mask),
            dim=1,
        )

        # Handle margin
        if "margin" in examples[0]:
            margin = [example["margin"] for example in examples]
            output["margin"] = torch.tensor(margin, dtype=torch.float)

        # Handle precomputed features (of fixed-size dimension, hence no padding needed)
        if "chosen_features" in examples[0]:
            chosen_features = [torch.tensor(example["chosen_features"]) for example in examples]
            rejected_features = [torch.tensor(example["rejected_features"]) for example in examples]
            output["chosen_features"] = torch.stack(chosen_features)
            output["rejected_features"] = torch.stack(rejected_features)

        # TODO: Handle precomputed ref_logps (for DPO-based models)

        return output


@dataclass
class TrainerExtensionConfig:
    """Extension class for `transformers.TrainingArguments`.

    IMPORTANT: This class must be placed before `TrainingArguments` in the inheritance order.

    Args:
        eval_on_epochs (list[float] | None):
            List of epochs when `EvaluateSaveCallback` should run evaluation.

        eval_on_end (bool):
            Flag whether `EvaluateSaveCallback` should run evaluation at the end of training.

        save_on_epochs (list[float] | None):
            List of epochs when `EvaluateSaveCallback` should save a checkpoint.

        save_on_end (bool):
            Flag whether `EvaluateSaveCallback` should save a checkpoint at the end of training.
    """

    # Evaluation parameters
    eval_on_epochs: list[float] | None = None  # Used by EvaluateSaveCallback
    eval_on_end: bool = False  # Used by EvaluateSaveCallback

    # Checkpointing parameters
    save_on_epochs: list[float] | None = None  # Used by EvaluateSaveCallback
    save_on_end: bool = False  # Used by EvaluateSaveCallback

    # Change default values of RewardConfig
    bf16: bool = False
    gradient_checkpointing: bool = False
    remove_unused_columns: bool = False  # Keep "weight" column for weighted metrics


class TrainerExtension:
    """Extension class for `transformers.Trainer`.

    Modifications:
    - Unify common initialization steps in `__init__`.
    - Unify `_prepare_dataset` to have common preprocessing steps.
    - Unify `data_collator` to handle commonly preprocessed data.
    - Unify `prediction_step` and `evaluation_loop` to work with the output of uncertainty-aware reward models.
    - Add `save_predictions` to save predictions during evaluation.

    IMPORTANT: This class must be placed before `Trainer` in the inheritance order.
    """

    def __init__(self, *args, **kwargs):
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [EvaluateSaveCallback()]
        super().__init__(*args, **kwargs)

        # Check for currently unsupported arguments
        check_supported_args(self.args, "label_smoothing_factor", [0.0], temp=True)
        check_supported_args(self.args, "batch_eval_metrics", [False], temp=True)
        # Check for permanently unsupported arguments
        check_supported_args(self.args, "include_for_metrics", [[]])
        check_supported_args(self.args, "use_legacy_prediction_loop", [False])
        check_supported_args(self.args, "remove_unused_columns", [False])

        # Set default data collator
        default_data_collator = DataCollatorForPreference(
            pad_token_id=self.processing_class.pad_token_id,
            pad_to_multiple_of=getattr(args, "pad_to_multiple_of", None),
        )
        self.data_collator = kwargs.get("data_collator", None) or default_data_collator
        # Set default compute_metrics function
        self.compute_metrics = kwargs.get("compute_metrics", None) or compute_default_metrics

        # Log number of samples in train and eval datasets
        if self.train_dataset is not None:
            try:
                num_samples = len(self.train_dataset)
                logger.info(f"Number of train samples: {num_samples}")
                if "wandb" in self.args.report_to and self.accelerator.is_main_process:
                    import wandb

                    wandb.log({"train/num_samples": num_samples})
            except TypeError:
                logger.info("Number of train samples: Unknown")
        if self.eval_dataset is not None:
            try:
                num_samples = len(self.eval_dataset)
                logger.info(f"Number of eval samples: {num_samples}")
                if "wandb" in self.args.report_to and self.accelerator.is_main_process:
                    import wandb

                    wandb.log({"eval/num_samples": num_samples})
            except TypeError:
                logger.info("Number of eval samples: Unknown")

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase,
        args: TrainerExtensionConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """Override `_prepare_dataset` of trainer class if it exists.

        Important: This method relies on being called in `__init__` by the trainer class as done by:
        - `trl.DPOTrainer` [1]
        - `trl.RewardTrainer` (only for trl>=0.24.0) [2]

        References:
            [1] https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/dpo_trainer.py#L530
            [2] https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/reward_trainer.py#L436
        """
        return prepare_preference_dataset(
            dataset=dataset,
            processing_class=processing_class,
            dataset_name=dataset_name,
            dataset_num_proc=getattr(args, "dataset_num_proc", None),
            tools=getattr(args, "tools", None),
            max_length=getattr(args, "max_length", None),
        )

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | Any] | None, torch.Tensor | None]:
        """Override `transformers.Trainer.prediction_step` [1].

        Modifications:
        - Return the entire `outputs` dictionary instead of a `logits` tensor. Hence, `ignore_keys` is unused.
        - Add `metric_key_prefix` argument and pass it through to `compute_loss`.
        - Set `labels` to `None` since it is clear that chosen is preferred over rejected.
        - Simplify code.

        References:
            [1] https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4804
        """
        # Prepare inputs
        inputs = self._prepare_inputs(inputs)

        # Compute loss and outputs
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()
        outputs = nested_detach(outputs)

        # Set labels to None (since it's clear that chosen is preferred over rejected)
        labels = None

        if prediction_loss_only:
            return loss, None, None
        else:
            return loss, outputs, labels

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Override `transformers.Trainer.evaluation_loop` [1].

        Modifications:
        - Receive `outputs` dictionary instead of a `logits` tensor from `prediction_step` and construct a single array
            of reward predictions and confidence bounds and pass it to the `compute_metrics` function.
        - Add call to `save_predictions`.
        - Simplify code.

        References:
            [1] https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4569
        """
        if prediction_loss_only is None:
            prediction_loss_only = self.args.prediction_loss_only

        ### PREPARE ###

        # If eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8" and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # For the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # Backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # If full FP16 or BF16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if self.args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=self.args.device)
            elif self.args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=self.args.device)

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader

        ### EVALUATE ###

        # Initialize containers
        losses_all = EvalLoopContainer(do_nested_concat=False)
        outputs_chosen_all = EvalLoopContainer(do_nested_concat=False)
        outputs_rejected_all = EvalLoopContainer(do_nested_concat=False)

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Predict outputs
            #   Shape of losses: (batch_size,)
            #   Shape of outputs_X["rewards"]: (batch_size, 3)
            losses, outputs, _ = self.prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )
            losses = losses.repeat(self.args.eval_batch_size)
            outputs_chosen = outputs["chosen"]
            outputs_rejected = outputs["rejected"]
            # Gather outputs across GPUs
            #   Shape of losses: (batch_size * num_devices,)
            #   Shape of outputs_X["rewards"]: (batch_size * num_devices, 3)
            losses = self.accelerator.gather_for_metrics(losses)
            outputs_chosen = self.accelerator.gather_for_metrics(outputs_chosen)
            outputs_rejected = self.accelerator.gather_for_metrics(outputs_rejected)
            # Add outputs to containers
            losses_all.add(losses)
            outputs_chosen_all.add(outputs_chosen)
            outputs_rejected_all.add(outputs_rejected)

            # Call prediction step callbacks
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Move outputs to CPU every eval_accumulation_steps
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                losses_all.to_cpu_and_numpy()
                outputs_chosen_all.to_cpu_and_numpy()
                outputs_rejected_all.to_cpu_and_numpy()

                del losses, outputs_chosen, outputs_rejected
                torch.cuda.empty_cache()

        # Move remaining outputs to CPU and convert EvalLoopContainer to lists
        losses_all = losses_all.get_arrays()
        outputs_chosen_all = outputs_chosen_all.get_arrays()
        outputs_rejected_all = outputs_rejected_all.get_arrays()

        # Combine losses to a single array
        losses_all = np.concatenate(losses_all, axis=0)
        # Combine rewards to a single array
        rewards_chosen_all = np.concatenate([outputs["rewards"] for outputs in outputs_chosen_all], axis=0)
        rewards_rejected_all = np.concatenate([outputs["rewards"] for outputs in outputs_rejected_all], axis=0)
        rewards_all = np.stack([rewards_chosen_all, rewards_rejected_all], axis=1)  # Shape: (num_samples, 2, 3)

        # Handle point-estimate models (no uncertainty)
        if rewards_all.shape[-1] == 1:
            rewards_all = np.repeat(rewards_all, 3, axis=-1)

        ### FINISH ###

        # Save predictions
        self.save_predictions(rewards_all)

        # Retrieve number of samples
        num_samples = len(dataloader.dataset)
        if num_samples != len(rewards_all):
            raise RuntimeError(
                f"Number of evaluation samples ({num_samples}) does not match "
                f"the number of predictions ({len(rewards_all)})."
            )

        # Compute metrics
        if self.compute_metrics is not None:
            if "weight" in dataloader.dataset.column_names:
                weights = np.array(dataloader.dataset["weight"])
            else:
                weights = np.ones(num_samples)
            metrics = self.compute_metrics(
                {
                    "rewards": rewards_all,
                    "weights": weights,
                    "outputs_chosen": outputs_chosen_all,
                    "outputs_rejected": outputs_rejected_all,
                },
                report_to=self.args.report_to,
            )
        else:
            metrics = {}
        metrics["loss"] = losses_all.mean()
        # Prefix all metric keys
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        # Ensure metrics are JSON-serializable
        metrics = denumpify_detensorize(metrics)

        return EvalLoopOutput(
            predictions=rewards_all,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )

    def save_predictions(self, rewards: np.ndarray) -> None:
        """Save reward predictions to disk during evaluation.

        Predictions are saved as NumPy arrays in the predictions subdirectory
        of the output directory, with filenames indicating the training step.

        Args:
            rewards (np.ndarray of shape (num_samples, 2, 3)):
                Array of reward predictions where:
                - First dimension: samples
                - Second dimension: [chosen, rejected]
                - Third dimension: [prediction, lower_bound, upper_bound]
        """
        if self.accelerator.is_main_process:
            path = Path(self.args.output_dir) / "predictions" / f"rewards_{self.state.global_step}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving reward predictions to {path}")
            np.save(path, rewards)


class BatchMetricsTrainerMixin:
    """Mixin class to enable logging of batch metrics from within `compute_loss`.

    This class provides the method `cache_batch_metrics` which can be called from within `compute_loss` to cache batch
    metrics. During the next call of `log`, the cached batch metrics corresponding to the current mode are averaged
    and added to the logs. The mode is inferred from the `metric_key_prefix` argument passed to `evaluation_loop`.

    IMPORTANT: This class must be placed before the Trainer class in the inheritance order.

    This implementation is inspired from `trl.DPOTrainer` [1].

    References:
        [1] https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/dpo_trainer.py#L1450

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__mode = "train"
        self.__batch_metrics = defaultdict(lambda: defaultdict(list))

    def cache_batch_metrics(self, metrics: dict[str, torch.Tensor]):
        """Cache batch metrics from `compute_loss`."""
        # Prepare per-sample metrics
        metrics = nested_detach(metrics)
        metrics = self.accelerator.gather_for_metrics(metrics)
        metrics = nested_numpify(metrics)

        # Cache batch metrics
        for key, batch_metrics in metrics.items():
            self.__batch_metrics[self.__mode][f"batch/{key}"].append(batch_metrics.mean().item())

    def log(self, logs: dict[str, float], *args, **kwargs):
        """Extend `transformers.Trainer.Trainer.log` [1].

        Modifications:
        - Add averaged batch metrics corresponding to `self.__mode` to logs.

        References:
            [1] https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L3769
        """
        # Add averaged batch metrics with corresponding prefix to logs
        metric_key_prefix = "" if self.__mode == "train" else f"{self.__mode}_"  # Log train metrics without prefix
        metrics = {
            f"{metric_key_prefix}{key}": np.mean(batch_metrics).item()
            for key, batch_metrics in self.__batch_metrics[self.__mode].items()
        }
        logs.update(metrics)
        self.__batch_metrics[self.__mode].clear()

        return super().log(logs, *args, **kwargs)

    def evaluate(self, *args, **kwargs) -> dict[str, float]:
        """Extend `transformers.Trainer.evaluate` [1].

        Modifications:
        - Set and reset `self.__mode` based on `metric_key_prefix` argument.

        References:
            [1] https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4401
        """
        try:
            self.__mode = kwargs.get("metric_key_prefix", "eval")
            return super().evaluate(*args, **kwargs)
        finally:
            self.__mode = "train"

    def predict(self, *args, **kwargs) -> dict[str, float]:
        """Extend `transformers.Trainer.predict` [1].

        Modifications:
        - Set and reset `self.__mode` based on `metric_key_prefix` argument.

        References:
            [1] https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4505
        """
        try:
            self.__mode = kwargs.get("metric_key_prefix", "test")
            return super().predict(*args, **kwargs)
        finally:
            self.__mode = "train"
