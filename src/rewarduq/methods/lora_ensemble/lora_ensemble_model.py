from __future__ import annotations

from typing import Any

import torch
from peft import PeftModel
from transformers import PretrainedConfig

from rewarduq.methods import RewardUQModel
from rewarduq.utils import get_logger, load_tokenizer_and_model, log_model_summary

logger = get_logger(__name__)


class LoraEnsembleModelConfig(PretrainedConfig):
    """Configuration class for `LoraEnsembleModel`."""

    model_type = "lora_ensemble"

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen3-0.6B",
        base_model_class: str = "AutoModelForSequenceClassification",
        base_model_init_kwargs: dict[str, Any] | None = None,
        peft_config: dict[str, Any] | None = None,
        adapter_paths: list[str] | None = None,
        reward_function: str = "mean",
        bounds_function: str = "std",
        bounds_function_std_beta: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            base_model_name_or_path (str):
                The name or path of the Hugging Face base model.

            base_model_class (str):
                The class of the model to load. This should be a class from the `transformers` library, such as
                `AutoModel`, `AutoModelForCausalLM`, etc. Defaults to `AutoModel`.

            base_model_init_kwargs (dict[str, Any] | None):
                Additional keyword arguments to pass to the model loading function.

            peft (dict[str, Any] | None):
                If provided, the model will be loaded with a PEFT adapter. The dictionary should contain
                the configuration for the PEFT adapter, such as `lora_alpha`, `lora_r`, `lora_dropout`, etc.
                If `None`, no PEFT adapter will be loaded.

            adapter_paths (list[str] | None):
                A list of paths to the LoRA adapters to load.

            reward_function (str):
                The function that generates the reward from the different MLP outputs of the ensemble
                head. Let `outputs` be all outputs of the MLPs, then the following functions are
                available.

                "mean": Mean of the outputs.
                    - `reward = mean(outputs)`

                "median": Median of the outputs.
                    - `reward = median(outputs)`

            bounds_function (str):
                The function that generates the uncertainty bounds from the different MLP outputs
                of the ensemble head. Let `outputs` be all outputs of the MLPs, then the following
                functions are available.

                "std": Standard deviation with hyperparameter `beta` (`bounds_function_std_beta`).
                    - `lower_bound = mean(outputs) - beta * std(outputs)`
                    - `upper_bound = mean(outputs) + beta * std(outputs)`

                "min_max": Min lower bound and max upper bound.
                    - `lower_bound = min(outputs)`
                    - `upper_bound = max(outputs)`

            bounds_function_std_beta (float):
                The beta parameter for the "std" `bounds_function`.
                Ignored if `bounds_function` is not "std".
        """
        super().__init__(**kwargs)

        # Base model parameters
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_class = base_model_class
        self.base_model_init_kwargs = base_model_init_kwargs or {}
        # Convert peft_config to dict if it's a PeftConfig object (for JSON serialization)
        if peft_config is not None and hasattr(peft_config, "to_dict"):
            self.peft_config = peft_config.to_dict()
        else:
            self.peft_config = peft_config

        # LoRA ensemble parameters
        self.adapter_paths = adapter_paths

        # Output aggregation functions
        self.reward_function = reward_function
        self.bounds_function = bounds_function
        self.bounds_function_std_beta = bounds_function_std_beta

    @property
    def ensemble_num(self):
        return len(self.adapter_paths) if self.adapter_paths else None


class LoraEnsembleModel(RewardUQModel):
    """An uncertainty-aware reward model with a LoRA-based ensemble."""

    config_class = LoraEnsembleModelConfig

    def __init__(self, config: LoraEnsembleModelConfig):
        # Check that peft_config is provided
        if not config.peft_config:
            raise ValueError("`peft_config` must be provided for LoraEnsembleModel.")

        super().__init__(config)

        # Check for unsupported arguments
        if not config.adapter_paths:
            raise ValueError("LoraEnsembleModel requires adapter_paths to be set.")

        self.tokenizer, self.base_model_ = load_tokenizer_and_model(
            config.base_model_name_or_path,
            base_model_class=config.base_model_class,
            base_model_init_kwargs=config.base_model_init_kwargs,
            target_modules=[],  # Freeze all base model parameters
            peft_config=None,  # PEFT adapters will be loaded separately below
        )

        # Zero out the final score layer
        self.base_model_.score.weight.data.zero_()
        if self.base_model_.score.bias is not None:
            self.base_model_.score.bias.data.zero_()

        # Wrap base model with PeftModel and load the first adapter as default dummy
        # It will be loaded again below but with its correct name
        # We need to do this to make sure that the default adapter remains when reloading adapters
        self.base_model_ = PeftModel.from_pretrained(
            self.base_model_,
            model_id=config.adapter_paths[0],
        )

        # Load adapters
        for i, adapter_path in enumerate(config.adapter_paths, start=1):
            self.base_model_.load_adapter(model_id=adapter_path, adapter_name=f"model_{i}")

        self._no_split_modules = getattr(self.base_model_, "_no_split_modules", None)

        log_model_summary(self)

    def reload_adapters(self, adapter_paths: list[str]):
        """Reloads the adapters from the given paths."""
        if len(adapter_paths) != self.config.ensemble_num:
            logger.warning(
                f"Number of adapters changed from {self.config.ensemble_num} to {len(adapter_paths)}. "
                "This might cause issues if the model relies on specific adapter names."
            )

        # Activate default adapter to avoid issues when deleting adapters
        self.base_model_.set_adapter("default")

        # Delete existing adapters
        active_adapters = list(self.base_model_.peft_config.keys())
        for adapter_name in active_adapters:
            if adapter_name != "default":
                self.base_model_.delete_adapter(adapter_name)

        # Load new adapters
        for i, adapter_path in enumerate(adapter_paths, start=1):
            self.base_model_.load_adapter(model_id=adapter_path, adapter_name=f"model_{i}")

        self.config.adapter_paths = adapter_paths

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        output_features: bool = False,
        output_only_features: bool = False,
        output_individual_rewards: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor | Any]:
        """Forward pass of the model.

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length), optional):
                [What are input IDs?](https://github.com/huggingface/transformers/blob/v4.51.3/docs/source/en/glossary.md#input-ids)

            attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional):
                Mask to avoid performing attention on padding token indices.

                Mask values selected in `[0, 1]`, where
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

               [What are attention masks?](https://github.com/huggingface/transformers/blob/v4.51.3/docs/source/en/glossary.md#attention-mask)

            output_features (bool, optional):
                Whether to return the features extracted from the base model.
                If `True`, the features will be returned in the `features` key of the output dictionary.

            output_only_features (bool, optional):
                If `True`, only the features will be returned, and the head will not be called.

            output_individual_rewards (bool, optional):
                Whether to return the outputs of the individual MLPs in the ensemble head.
                If `True`, the outputs will be returned in the `individual_rewards` key of the output dictionary.
        """
        if output_features or output_only_features:
            raise NotImplementedError("Feature extraction not yet implemented.")  # TODO: Implement

        result = {}

        # Run model inference by switching between different LoRA adapters
        outputs = []
        for model_id in range(1, self.config.ensemble_num + 1):
            # Switch to the corresponding LoRA adapter
            self.base_model_.set_adapter(f"model_{model_id}")

            # Call the corresponding ensemble member
            individual_output: torch.Tensor = self.base_model_(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            ).logits[:, 0]  # Shape: (batch_size,)
            outputs.append(individual_output)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, num_heads)

        if self.config.reward_function == "mean":
            reward = outputs.mean(dim=1)
        elif self.config.reward_function == "median":
            reward = outputs.median(dim=1).values
        else:
            raise ValueError(f"Unknown reward function: {self.config.reward_function}")

        if self.config.bounds_function == "std":
            std = outputs.std(dim=1)
            lower_bound = reward - self.config.bounds_function_std_beta * std
            upper_bound = reward + self.config.bounds_function_std_beta * std
        elif self.config.bounds_function == "min_max":
            lower_bound = outputs.min(dim=1).values
            upper_bound = outputs.max(dim=1).values
        else:
            raise ValueError(f"Unknown bounds function: {self.config.bounds_function}")

        result["rewards"] = torch.stack([reward, lower_bound, upper_bound], dim=1)  # Shape: (batch_size, 3)

        if output_individual_rewards:
            result["individual_rewards"] = outputs  # Shape: (batch_size, num_heads)

        return result
