from __future__ import annotations

import importlib.metadata
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from datasets.fingerprint import Hasher
from packaging import version
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import nested_detach
from trl import RewardConfig, RewardTrainer

from rewarduq.trainers.trainer_extension import BatchMetricsTrainerMixin, TrainerExtension, TrainerExtensionConfig
from rewarduq.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RewardUQTrainerConfig(TrainerExtensionConfig, RewardConfig):
    """Configuration class for `RewardUQTrainer`.

    Args:
        precompute_features (bool):
            Whether to precompute the features. If `True`, the base model will be called
            only once for each sample in the dataset, and the outputs will be cached. This can
            speed up training significantly.

        precompute_features_path (str):
            The path to save the precomputed features. Only used if `precompute_features` is `True`.
    """

    # Precomputation parameters
    precompute_features: bool = False
    precompute_features_path: str = "features/"


class RewardUQTrainer(BatchMetricsTrainerMixin, TrainerExtension, RewardTrainer):
    """Trainer class for uncertainty-aware reward models."""

    def __init__(
        self,
        precomputed_features_train: torch.Tensor | None = None,
        precomputed_features_eval: torch.Tensor | None = None,
        **kwargs,
    ):
        # TODO: Can be removed after upgrading to trl>=0.24.0
        require_prepare_dataset = version.parse(importlib.metadata.version("trl")) < version.parse("0.24.0")
        if require_prepare_dataset:
            # Get kwargs
            train_dataset = kwargs.get("train_dataset", None)
            eval_dataset = kwargs.get("eval_dataset", None)
            processing_class = kwargs.get("processing_class", None)
            args = kwargs.get("args", None)
            # Prepare datasets
            train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
            if eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")
            # Rename columns to skip TRL's internal processing
            # Reference: https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/reward_trainer.py#L191
            if isinstance(train_dataset, IterableDataset) and train_dataset.features is not None:
                features = train_dataset.features.copy()
                features["input_ids_chosen"] = features.pop("chosen_input_ids")
                train_dataset = train_dataset.map(
                    lambda x: {"input_ids_chosen": x["chosen_input_ids"]},
                    remove_columns=["chosen_input_ids"],
                    features=features,
                )
            else:
                train_dataset = train_dataset.rename_column("chosen_input_ids", "input_ids_chosen")
            # Update kwargs
            kwargs["train_dataset"] = train_dataset
            kwargs["eval_dataset"] = eval_dataset

        super().__init__(**kwargs)

        # TODO: Can be removed after upgrading to trl>=0.24.0
        if require_prepare_dataset:
            # Rename columns to skip TRL's internal processing
            if isinstance(self.train_dataset, IterableDataset) and self.train_dataset.features is not None:
                features = self.train_dataset.features.copy()
                features["chosen_input_ids"] = features.pop("input_ids_chosen")
                self.train_dataset = self.train_dataset.map(
                    lambda x: {"chosen_input_ids": x["input_ids_chosen"]},
                    remove_columns=["input_ids_chosen"],
                    features=features,
                )
            else:
                self.train_dataset = self.train_dataset.rename_column("input_ids_chosen", "chosen_input_ids")

        # Precompute features
        if self.args.precompute_features:
            if self.processing_class is None:
                raise ValueError("A processing_class must be specified when using feature precomputation. ")

            if hasattr(self.args, "remove_unused_columns") and self.args.remove_unused_columns:
                raise ValueError(
                    "The `remove_unused_columns` argument is not supported when using "
                    "`precompute_features`. Please set it to `False`."
                )

            no_train_features = (
                self.train_dataset is not None
                and "chosen_features" not in self.train_dataset.column_names
                and "rejected_features" not in self.train_dataset.column_names
            )
            no_eval_features = (
                self.eval_dataset is not None
                and "chosen_features" not in self.eval_dataset.column_names
                and "rejected_features" not in self.eval_dataset.column_names
            )

            # TODO: Dirty hotfix for https://github.com/huggingface/transformers/issues/39961
            if (no_train_features and precomputed_features_train is None) or (
                no_eval_features and precomputed_features_eval is None
            ):
                num_train_epochs = self.args.num_train_epochs
                max_steps = self.args.max_steps
                eval_on_start = self.args.eval_on_start
                self.args.num_train_epochs = 0
                self.args.max_steps = 0
                self.args.eval_on_start = False
                self.train()
                self.args.num_train_epochs = num_train_epochs
                self.args.max_steps = max_steps
                self.args.eval_on_start = eval_on_start

            # Precompute features for train dataset
            if no_train_features:
                self.train_dataset = self._setup_precomputed_features(
                    "train",
                    self.train_dataset,
                    precomputed_features=precomputed_features_train,
                )

            # Precompute features for eval dataset
            if no_eval_features:
                self.eval_dataset = self._setup_precomputed_features(
                    "eval",
                    self.eval_dataset,
                    precomputed_features=precomputed_features_eval,
                )

    # Copied and adapted from `trl.RewardTrainer.compute_loss`
    # Reference: https://github.com/huggingface/trl/blob/e04f7eb3b993b5278733a793853ed3a85d1c69c8/trl/trainer/reward_trainer.py#L265
    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if num_items_in_batch is not None:
            logger.warning(
                "The num_items_in_batch argument is not supported in this trainer. "
                "It will be ignored and the loss will be computed on the whole batch."
            )

        outputs_chosen = self._get_model_outputs(model, inputs, "chosen")
        outputs_rejected = self._get_model_outputs(model, inputs, "rejected")
        rewards_chosen = outputs_chosen["rewards"]  # Shape: (batch_size,)
        rewards_rejected = outputs_rejected["rewards"]  # Shape: (batch_size,)

        # Compute loss
        if "margin" in inputs:
            margin = inputs["margin"]
            if margin.dim() == 1:
                margin = margin.unsqueeze(-1)
            loss_base = -F.logsigmoid(rewards_chosen - rewards_rejected - margin).mean()
        else:
            loss_base = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

        # Compute regularization towards centered rewards
        loss_center_rewards = (self.args.center_rewards_coefficient or 0) * torch.mean(
            (rewards_chosen + rewards_rejected) ** 2
        )

        # Compute final loss
        loss = loss_base + loss_center_rewards

        # Compute batch metrics
        batch_metrics = {
            "win_rate": (rewards_chosen > rewards_rejected).float(),
            "rewards/chosen": rewards_chosen,
            "rewards/rejected": rewards_rejected,
            "rewards/margins": rewards_chosen - rewards_rejected,
            "loss_base": loss_base,
            "loss_center_rewards": loss_center_rewards,
        }
        self.cache_batch_metrics(batch_metrics)

        if return_outputs:
            return loss, {
                "chosen": outputs_chosen,
                "rejected": outputs_rejected,
            }
        return loss

    def visualize_samples(self, *args, **kwargs):
        pass

    # ----------------------
    # Feature Precomputation
    # ----------------------

    def get_feature_dependencies(self, dataset: Dataset) -> dict[str, Any]:
        """Get the dependencies for the precomputed features.

        The returned dict should uniquely identify the precomputed features and must be hashable.
        """
        return {
            "model": self.model.get_feature_dependencies(),
            "dataset": {
                "name": dataset.info.dataset_name,
                "column_names": dataset.column_names,
                "length": len(dataset),
                "fingerprint": dataset._fingerprint,
            },
        }

    def _setup_precomputed_features(
        self,
        name: str,
        dataset: Dataset,
        precomputed_features: torch.Tensor | None = None,
    ) -> Dataset:
        if precomputed_features is None:
            # Load feature dependencies
            dependencies = self.get_feature_dependencies(dataset)
            logger.info(f"Feature dependencies for {name} dataset: {dependencies}")

            # Compute feature fingerprint
            fingerprint = Hasher.hash(dependencies)
            path_dependencies = Path(self.args.precompute_features_path) / f"features_{fingerprint}.json"
            path_features = Path(self.args.precompute_features_path) / f"features_{fingerprint}.pt"

            if path_features.exists():
                # Load precomputed features
                precomputed_features = torch.load(path_features)
                logger.info(f"Loaded precomputed features for {name} dataset: {path_features}")
            else:
                # Precompute features
                dataloader = self._get_dataloader(
                    dataset=dataset,
                    description=name,
                    batch_size=self.args.eval_batch_size,
                )
                precomputed_features = self._precompute_features(
                    dataloader,
                    tqdm_desc=f"Precomputing features for {name} dataset",
                )

                # Save feature dependencies
                path_dependencies.parent.mkdir(parents=True, exist_ok=True)
                with open(path_dependencies, "w") as f:
                    json.dump(dependencies, f, indent=2)

                # Save precomputed features
                path_features.parent.mkdir(parents=True, exist_ok=True)
                torch.save(precomputed_features, path_features)
                logger.info(f"Saved precomputed features for {name} dataset: {path_features}")

        # Add precomputed features to dataset
        dataset = dataset.add_column("chosen_features", precomputed_features[:, 0].tolist())
        dataset = dataset.add_column("rejected_features", precomputed_features[:, 1].tolist())

        return dataset

    def _precompute_features(
        self,
        dataloader,
        tqdm_desc: str | None = None,
    ) -> torch.Tensor:
        """Modification of `transformers.Trainer.evaluation_loop` [1] for just precomputing features.

        Modifications:
        - Instead of parsing logits from the model outputs, we construct a single array of rewards including upper and
            lower bounds and pass them to the `compute_metrics` function.
        - Simplified main loop.
        - Remove all metrics computation logic.
        - Custom tqdm bar for progress tracking.

        References:
            [1] https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/trainer.py
        """
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
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

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        # Setup tqdm bar
        if not self.args.disable_tqdm and self.state.is_world_process_zero:
            total_steps = len(dataloader)
            training_bar = tqdm(total=total_steps, dynamic_ncols=True, desc=tqdm_desc)

        chosen_features_all_gpu, rejected_features_all_gpu = [], []
        chosen_features_all, rejected_features_all = [], []

        # Main precomputation loop
        for step, inputs in enumerate(dataloader):
            # inputs = self._prepare_inputs(inputs)

            orig_flag = self.args.precompute_features
            self.args.precompute_features = False

            with torch.no_grad():
                chosen_features = nested_detach(
                    self._get_model_outputs(model, inputs, "chosen", output_only_features=True)
                )
                rejected_features = nested_detach(
                    self._get_model_outputs(model, inputs, "rejected", output_only_features=True)
                )

            self.args.precompute_features = orig_flag

            # Gather outputs across GPUs
            chosen_features = self.accelerator.gather_for_metrics([chosen_features])
            rejected_features = self.accelerator.gather_for_metrics([rejected_features])

            # Add features to list
            chosen_features_all_gpu += chosen_features
            rejected_features_all_gpu += rejected_features

            # Update tqdm bar
            if not self.args.disable_tqdm and self.state.is_world_process_zero:
                training_bar.update(1)

            # Move features to CPU every eval_accumulation_steps
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                chosen_features_all += [t.cpu() for t in chosen_features_all_gpu]
                rejected_features_all += [t.cpu() for t in rejected_features_all_gpu]
                chosen_features_all_gpu = []
                rejected_features_all_gpu = []

                del chosen_features, rejected_features
                torch.cuda.empty_cache()

        # Move remaining features to CPU
        chosen_features_all += [t.cpu() for t in chosen_features_all_gpu]
        rejected_features_all += [t.cpu() for t in rejected_features_all_gpu]

        # Combine rewards to a single tensor
        chosen_features_all = torch.cat(chosen_features_all, dim=0)
        rejected_features_all = torch.cat(rejected_features_all, dim=0)
        features_all = torch.stack(
            [chosen_features_all, rejected_features_all],
            axis=1,
        )  # Shape: (num_samples, 2, feature_dim)

        # Clean up tqdm bar
        if not self.args.disable_tqdm and self.state.is_world_process_zero:
            training_bar.close()

        # Compute number of samples
        num_samples = len(dataloader.dataset)
        if num_samples != len(features_all):
            raise ValueError(
                f"Number of samples ({num_samples}) does not match the number of "
                f"precomputed features ({len(features_all)})."
            )

        return features_all

    # -----
    # Other
    # -----

    def _get_model_outputs(
        self,
        model: PreTrainedModel | torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        key: str,
        **kwargs,
    ) -> dict[str, torch.Tensor | Any]:
        """Get the model outputs for the given inputs.

        Uses precomputed features if available.

        Args:
            model (PreTrainedModel | nn.Module):
                The model.

            inputs (dict[str, torch.Tensor | Any]):
                The inputs to the model. It should contain the keys "input_ids" and
                "attention_mask". If it contains the keys "chosen_features" and/or
                "rejected_features", the model will use these features instead of
                calling the base model again.

            key (str):
                The key to use for the model outputs. Normally either "chosen" or "rejected".

            kwargs (dict[str, Any]):
                Additional keyword arguments to pass to the model.

        Returns:
            outputs (dict[str, torch.Tensor | Any]):
                The model outputs.
        """
        if hasattr(self.args, "precompute_features") and self.args.precompute_features:
            if f"{key}_features" not in inputs or inputs[f"{key}_features"] is None:
                raise ValueError(
                    "Even though `precompute_features` is set to `True`, the dataset does not "
                    "contain the precomputed features. Something went wrong."
                )
            return model(
                features=inputs[f"{key}_features"],
                **kwargs,
            )

        return model(
            input_ids=inputs[f"{key}_input_ids"],
            attention_mask=inputs[f"{key}_attention_mask"],
            **kwargs,
        )
