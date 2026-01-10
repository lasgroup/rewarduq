from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from rewarduq.methods import MLPHead, RewardUQModel
from rewarduq.utils import (
    apply_initialization_to_linear_layers,
    check_supported_args,
    extract_features,
    get_logger,
    load_tokenizer_and_model,
    log_model_summary,
)

logger = get_logger(__name__)


class MLPHeadEnsembleModelConfig(PretrainedConfig):
    """Configuration class for `MLPHeadEnsembleModel`."""

    model_type = "mlp_head_ensemble_model"

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen3-0.6B",
        base_model_class: str = "AutoModel",
        base_model_init_kwargs: dict[str, Any] | None = None,
        target_modules: list[str] | None = None,
        peft_config: dict[str, Any] | None = None,
        head_hidden_dim: int = 128,
        head_num_layers: int = 2,
        head_activation: str | None = "ReLU",
        head_initialization: str | None = "xavier_uniform_",
        head_initialization_kwargs: dict[str, Any] | None = None,
        num_heads: int = 20,
        reward_function: str = "mean",
        bounds_function: str = "std",
        bounds_function_std_beta: float = 1.0,
        feature_extraction_layer: str | int = "last_hidden_state",
        feature_extraction_pooling_strategy: str | None = "last",
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

            target_modules (list[str] | None):
                A list of module names to finetune. If `None`, all modules will be finetuned.
                If provided, only the specified modules will have their `requires_grad` set to `True`.

            peft (dict[str, Any] | None):
                If provided, the model will be loaded with a PEFT adapter. The dictionary should contain
                the configuration for the PEFT adapter, such as `lora_alpha`, `lora_r`, `lora_dropout`, etc.
                If `None`, no PEFT adapter will be loaded.

            head_hidden_dim (int):
                The dimension of the hidden layers in each MLP of the ensemble head.

            head_output_dim (int):
                The dimension of the output layer in each MLP of the ensemble head.

            head_num_layers (int):
                The number of layers in each MLP of the ensemble head.

            head_activation (str):
                The activation function to use in each MLP of the ensemble head.
                Must be a valid PyTorch activation function name (e.g. "ReLU", "Tanh", etc.).

            head_initialization (str):
                The initialization function to use for the weights of the ensemble head.
                Must be a valid PyTorch initialization function name (e.g. "xavier_uniform_", etc.).

            head_initialization_kwargs (dict[str, Any] | None):
                Additional keyword arguments to pass to the head_initialization function.

            num_heads (int):
                The number of MLPs in the ensemble head.

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

            feature_extraction_layer (str | int):
                The layer from which to extract features.

                If a string is provided, it will be used as the name of the attribute to extract
                from the outputs (e.g. `last_hidden_state` or `logits`).

                If an integer is provided, it will be used as the index of the `hidden_states`
                attribute of the outputs.

            feature_extraction_pooling_strategy (str | None):
                The strategy to use for selecting features from the base model outputs.
                Must be either "last", "min", "max", "mean" or None.
        """
        super().__init__(**kwargs)

        # Base model parameters
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_class = base_model_class
        self.base_model_init_kwargs = base_model_init_kwargs or {}
        self.target_modules = target_modules
        self.peft_config = peft_config

        # Head parameters
        self.head_hidden_dim = head_hidden_dim
        self.head_num_layers = head_num_layers
        self.head_activation = head_activation
        self.head_initialization = head_initialization
        self.head_initialization_kwargs = head_initialization_kwargs or {}
        self.num_heads = num_heads

        # Output aggregation functions
        self.reward_function = reward_function
        self.bounds_function = bounds_function
        self.bounds_function_std_beta = bounds_function_std_beta

        # Feature extraction parameters
        self.feature_extraction_layer = feature_extraction_layer
        self.feature_extraction_pooling_strategy = feature_extraction_pooling_strategy


class MLPHeadEnsembleModel(RewardUQModel):
    """An uncertainty-aware reward model with a multi-head ensemble."""

    config_class = MLPHeadEnsembleModelConfig

    def __init__(self, config: MLPHeadEnsembleModelConfig):
        super().__init__(config)

        # Check for currently unsupported arguments
        check_supported_args(config, "peft_config", [None], temp=True)

        self.tokenizer, self.base_model_ = load_tokenizer_and_model(
            config.base_model_name_or_path,
            base_model_class=config.base_model_class,
            base_model_init_kwargs=config.base_model_init_kwargs,
            target_modules=config.target_modules,
            peft_config=config.peft_config,
        )
        self._no_split_modules = getattr(self.base_model_, "_no_split_modules", None)

        # Freeze base model parameters
        for param in self.base_model_.parameters():
            param.requires_grad = False

        self.mlps = nn.ModuleList(
            [
                MLPHead(
                    input_dim=self.base_model_.config.hidden_size,
                    hidden_dim=config.head_hidden_dim,
                    num_layers=config.head_num_layers,
                    activation=getattr(nn, config.head_activation) if config.head_activation else None,
                )
                for _ in range(config.num_heads)
            ]
        )

        apply_initialization_to_linear_layers(
            self.mlps,
            initialization=getattr(nn.init, config.head_initialization) if config.head_initialization else None,
            initialization_kwargs=config.head_initialization_kwargs,
        )

        initial_weights = self.get_mlp_weights().clone().detach()
        self.initial_weights = nn.Parameter(initial_weights, requires_grad=False)

        log_model_summary(self)

    def get_mlp_weights(self) -> torch.Tensor:
        """Get the stacked weights of all MLPs in the ensemble head."""
        weights = torch.stack(
            [torch.cat([p.view(-1) for p in mlp.parameters() if p.requires_grad], dim=0) for mlp in self.mlps],
            dim=0,
        )
        return weights

    def get_feature_dependencies(self) -> dict[str]:
        return {
            "base_model_name_or_path": self.config.base_model_name_or_path,
            "pad_token_id": self.config.pad_token_id,
            "feature_layer": self.config.feature_extraction_layer,
            "pooling_strategy": self.config.feature_extraction_pooling_strategy,
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        features: torch.FloatTensor | None = None,
        output_features: bool = False,
        output_only_features: bool = False,
        output_individual_rewards: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor | Any]:
        """Forward pass of the model.

        Either provide `input_ids` and `attention_mask`, or directly the `base_outputs`.

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length), optional):
                [What are input IDs?](https://github.com/huggingface/transformers/blob/v4.51.3/docs/source/en/glossary.md#input-ids)

            attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional):
                Mask to avoid performing attention on padding token indices.

                Mask values selected in `[0, 1]`, where
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

               [What are attention masks?](https://github.com/huggingface/transformers/blob/v4.51.3/docs/source/en/glossary.md#attention-mask)

            features (torch.FloatTensor of shape (batch_size, sequence_length, input_dim), optional):
                If provided, the base model will not be called again.

            output_features (bool, optional):
                Whether to return the features extracted from the base model.
                If `True`, the features will be returned in the `features` key of the output dictionary.

            output_only_features (bool, optional):
                If `True`, only the features will be returned, and the head will not be called.

            output_individual_rewards (bool, optional):
                Whether to return the outputs of the individual MLPs in the ensemble head.
                If `True`, the outputs will be returned in the `individual_rewards` key of the output dictionary.
        """
        result = {}

        if features is None:
            base_model_outputs = self.base_model_(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            features = extract_features(
                base_model_outputs,
                feature_layer=self.config.feature_extraction_layer,
                pooling_strategy=self.config.feature_extraction_pooling_strategy,
                attention_mask=attention_mask,
            )  # Shape: (batch_size, feature_dim)

        if output_only_features:
            return features
        if output_features:
            result["features"] = features

        outputs: torch.Tensor = torch.stack(
            [mlp(features) for mlp in self.mlps], dim=1
        )  # Shape: (batch_size, num_heads, 1)

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

        result["rewards"] = torch.cat([reward, lower_bound, upper_bound], dim=1)  # Shape: (batch_size, 3)

        if output_individual_rewards:
            result["individual_rewards"] = outputs.squeeze(dim=-1)  # Shape: (batch_size, num_heads)

        return result
