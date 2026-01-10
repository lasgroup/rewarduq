from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import torch
from accelerate import Accelerator
from peft import PeftModel
from torch import nn
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from trl.trainer.utils import flush_left, selective_log_softmax

from rewarduq.methods import RewardUQModel
from rewarduq.utils import (
    check_supported_args,
    extract_features,
    get_logger,
    load_tokenizer_and_model,
    log_model_summary,
)

logger = get_logger(__name__)


def _unwrap(model):
    if isinstance(model, PeftModel):
        return model.model
    else:
        return model


# Reference: Sample Efficient Preference Alignment in LLMs via Active Exploration (Mehta et al., 2025)
class DropoutEnsembleModel(nn.Module):
    """A wrapper model that applies dropout-based ensembling for uncertainty estimation."""

    def __init__(
        self,
        *,
        model: nn.Module,
        feature_extractor: Callable[[Any], torch.Tensor] | None = None,
        head: nn.Module,
        p: float = 0.05,
    ):
        super().__init__()

        self.model = model
        self.feature_extractor = feature_extractor
        self.dropout = nn.Dropout(p)
        self.head = head

        self.p = p  # Remember p, since DPOTrainer sets it to 0

    def forward(self, num_samples=1, disable_dropout=False, **kwargs) -> Iterable[torch.Tensor]:
        # Set dropout mode
        self.dropout.p = self.p  # Set p again (since DPO trainer sets it to 0)
        if disable_dropout:
            self.dropout.eval()  # Disable dropout
        else:
            self.dropout.train()  # Enable dropout

        # Apply base model
        x = self.model(**kwargs)

        # Extract features
        if self.feature_extractor:
            x = self.feature_extractor(x)

        # Apply dropout head
        def dropout_head(x: torch.Tensor) -> torch.Tensor:
            x = self.dropout(x)
            logits = self.head(x)
            return logits

        # Return iterator of multiple sampled logits to compute loss and uncertainties
        # NOTE: We return a lazy iterator of logits to reduce GPU memory usage.
        if self.dropout.training and self.dropout.p > 0:
            logits_iterator = (dropout_head(x) for _ in range(num_samples))
        else:
            # Evaluate dropout head only once if dropout is disabled
            # NOTE: This is a small optimization, when dropout is disabled (e.g. for the DPO reference model).
            logits = dropout_head(x)
            logits_iterator = (logits for _ in range(num_samples))

        return logits_iterator

    # Reference: https://github.com/huggingface/peft/blob/v0.17.1/src/peft/peft_model.py#L853
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped model (e.g., `_no_split_modules`)."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":  # Prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)


class DPOHeadDropoutEnsembleModelConfig(PretrainedConfig):
    """Configuration class for `DPOHeadDropoutEnsembleModel`."""

    model_type = "dpo_head_dropout_ensemble"

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen3-0.6B",
        base_model_class: str = "AutoModelForCausalLM",
        base_model_init_kwargs: dict[str, Any] | None = None,
        ref_model_init_kwargs: dict[str, Any] | None = None,
        target_modules: list[str] | None = None,
        peft_config: dict[str, Any] | None = None,
        feature_extraction_layer: str | int = "last_hidden_state",
        feature_extraction_pooling_strategy: str | None = None,
        beta: float = 0.1,
        dropout: float = 0.05,
        num_samples_train: int = 1,
        num_samples_eval: int = 5,
        scale_bounds: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Base model parameters
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_class = base_model_class
        self.base_model_init_kwargs = base_model_init_kwargs or {}
        self.ref_model_init_kwargs = ref_model_init_kwargs or {}
        self.target_modules = target_modules
        self.peft_config = peft_config

        # Feature extraction parameters
        self.feature_extraction_layer = feature_extraction_layer
        self.feature_extraction_pooling_strategy = feature_extraction_pooling_strategy

        # DPO parameters
        self.beta = beta

        # Dropout-based ensemble parameters
        self.dropout = dropout
        self.num_samples_train = num_samples_train
        self.num_samples_eval = num_samples_eval
        self.scale_bounds = scale_bounds


@dataclass
class DPOHeadDropoutEnsembleModelOutput(ModelOutput):
    # Output for computing DPO loss
    logprobs_all: torch.Tensor | None = None
    ref_logprobs_all: torch.Tensor | None = None
    labels: torch.LongTensor | None = None
    loss_mask: torch.BoolTensor | None = None
    # Output for computing reward uncertainties
    rewards: torch.Tensor | None = None
    rewards_all: torch.Tensor | None = None


class DPOHeadDropoutEnsembleModel(RewardUQModel):
    """A DPO-based implicit reward model with dropout-based uncertainty estimation."""

    config_class = DPOHeadDropoutEnsembleModelConfig
    base_model_prefix = "model"

    _tied_weights_keys = ["model.head.weight"]  # Indicate shared tensors due to tied input and output embeddings

    def __init__(self, config: DPOHeadDropoutEnsembleModelConfig):
        super().__init__(config)
        self.config = config
        self.accelerator = Accelerator()

        # Check for currently unsupported arguments
        check_supported_args(self.config, "is_encoder_decoder", [False], temp=True)

        # Initialize model
        tokenizer, causal_lm = load_tokenizer_and_model(
            base_model_name_or_path=config.base_model_name_or_path,
            base_model_class=config.base_model_class,
            base_model_init_kwargs=config.base_model_init_kwargs,
            target_modules=config.target_modules,
            peft_config=config.peft_config,
        )
        model = DropoutEnsembleModel(
            model=_unwrap(causal_lm).model,
            feature_extractor=partial(
                extract_features,
                feature_layer=self.config.feature_extraction_layer,
                pooling_strategy=self.config.feature_extraction_pooling_strategy,
            ),
            head=_unwrap(causal_lm).lm_head,
            p=self.config.dropout,
        )
        self.tokenizer = tokenizer
        self.model = model

        # Defer initialization of reference model
        self._ref_model = None

        # Log summary of model
        log_model_summary(self)

    @property
    def ref_model(self):
        if self._ref_model is None:
            raise RuntimeError("Reference model is not initialized. Please call `init_ref_model()` first.")
        return self._ref_model[0]

    def __setattr__(self, name, value):
        if name == "ref_model":
            # Wrap reference model to prevent it from being exposed as submodule
            # NOTE 1: This ensures that its parameters are not included in `self.parameters()`. Hence,
            # - The reference model is unaffected by train/eval mode switches (i.e., it stays in eval mode),
            # - The reference model is unaffected by device placement (i.e., `prepare_model` must be called manually),
            # - The reference model is ignored while saving or loading weights.
            # NOTE 2: A standard setter function does not work here, as `torch.nn.Module` overwrites `__setattr__` and
            # intercepts the setting of `ref_model`. Hence, we have to intercept it here.
            self._ref_model = (value,)
        else:
            super().__setattr__(name, value)

    def init_ref_model(self):
        """Initialize the reference model.

        It is necessary to initialize the reference model separately by calling `init_ref_model()`, as it is bad
        practice to initialize submodules with `from_pretrained` within the constructor of a `PreTrainedModel` [1].

        References:
            [1] https://github.com/huggingface/transformers/issues/39900#issuecomment-3164359506
        """
        if not isinstance(self.model.model, PeftModel):
            # Initialize reference model
            _, ref_causal_lm = load_tokenizer_and_model(
                base_model_name_or_path=self.config.base_model_name_or_path,
                base_model_class=self.config.base_model_class,
                base_model_init_kwargs=self.config.ref_model_init_kwargs,
                target_modules=[],  # Freeze all parameters
            )
            ref_model = DropoutEnsembleModel(
                model=_unwrap(ref_causal_lm).model,
                feature_extractor=partial(
                    extract_features,
                    feature_layer=self.config.feature_extraction_layer,
                    pooling_strategy=self.config.feature_extraction_pooling_strategy,
                ),
                head=_unwrap(ref_causal_lm).lm_head,
                p=0.0,
            )
            ref_model.eval()
        else:
            ref_model = None

        self.ref_model = ref_model
        logger.info("Initialized reference model.")

    def get_output_embeddings(self):
        """Return the model's output embeddings (typically `lm_head`)."""
        # NOTE: This is important to enable tied input and output embedding weights.
        return self.model.head

    def set_output_embeddings(self, new_embeddings):
        """Set the model's output embeddings (typically `lm_head`)."""
        self.model.head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        completion_mask: torch.LongTensor | None = None,
        num_samples: int | None = None,
        output_rewards_all: bool = False,
        **kwargs,
    ) -> DPOHeadDropoutEnsembleModelOutput:
        """Compute multiple rewards for the completion."""
        loss_mask = completion_mask.bool()
        if num_samples is None:
            if self.training:
                num_samples = self.config.num_samples_train
            else:
                num_samples = self.config.num_samples_eval

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        # NOTE: Since the logits of shape `(batch_size, sequence_length, vocab_size)` allocate a large GPU memory block
        # (~11GB for batch size 8), we make sure to never allocate multiple logits at the same time, especially when
        # sampling multiple logits with the dropout head.
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "num_samples": num_samples,
            **kwargs,
        }

        # Compute reference outputs
        with torch.no_grad():
            if self.ref_model is None:
                # Use model without PEFT adapter
                with self.accelerator.unwrap_model(self.model.model).disable_adapter():
                    ref_outputs = self.forward_with_model(self.model, **forward_kwargs, disable_dropout=True)
            else:
                # Use reference model
                ref_outputs = self.forward_with_model(self.ref_model, **forward_kwargs, disable_dropout=True)
        # Delete unused outputs of the reference model
        del ref_outputs["labels"]

        # Compute outputs
        outputs = self.forward_with_model(self.model, **forward_kwargs)

        # Compute individual rewards
        rewards_all = self.config.beta * (
            outputs["logprobs_all"] - ref_outputs["logprobs_all"]
        )  # Shape: (batch_size, num_samples)

        # Compute reward statistics over dropout ensemble
        rewards_mean = torch.mean(rewards_all, dim=1)
        rewards_std = torch.std(rewards_all, dim=1) if num_samples > 1 else torch.zeros_like(rewards_mean)
        rewards_lower = rewards_mean - self.config.scale_bounds * rewards_std
        rewards_upper = rewards_mean + self.config.scale_bounds * rewards_std
        rewards = torch.stack([rewards_mean, rewards_lower, rewards_upper], dim=1)  # Shape: (batch_size, 3)

        return DPOHeadDropoutEnsembleModelOutput(
            # Return logprobs and Co.
            logprobs_all=outputs["logprobs_all"],
            ref_logprobs_all=ref_outputs["logprobs_all"],
            labels=outputs["labels"],
            loss_mask=loss_mask,
            # Return rewards
            rewards=rewards,
            rewards_all=rewards_all if output_rewards_all else None,
        )

    def forward_with_model(
        self,
        model: DropoutEnsembleModel,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        loss_mask: torch.LongTensor,
        num_samples: int,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute multiple log probabilities of the completion with the given model."""
        # Reference: https://github.com/huggingface/trl/blob/2bc182c4fb8180a3cb815c61ca6c528a86559d14/trl/trainer/dpo_trainer.py#L1105

        # Run forward pass with model
        logits_iterator = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_samples=num_samples,
            **kwargs,
        )  # Shape: num_samples x (batch_size, sequence_length, vocab_size)

        # Prepare labels for the loss computation
        labels = input_ids.clone()  # Prevent from in-place modification of input_ids
        labels[~loss_mask] = 0  # Dummy token; we'll ignore the losses on these tokens later

        # Compute logits and Co. of the last sample and log probabilities of all samples
        def compute_logprobs_per_token(logits: torch.Tensor) -> torch.Tensor:
            """Compute log probabilities of each token in the sequence.

            This is a more memory-efficient and faster version of [1], which replaces `torch.roll`, which allocates an
            additional memory block of the size `logits`, with simple slicing.

            Original implementation:
            ```python
            # shift logits forward by 1 to align with labels (due to the generated token)
            logits = torch.roll(logits, shifts=1, dims=1)
            # compute log probabilities of each token
            logprobs_per_token = selective_log_softmax(logits, labels)
            logprobs_per_token[~loss_mask] = 0
            ```

            References:
                [1] https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/dpo_trainer.py#L1214
            """
            # Align logits with labels (due to autoregressive prediction)
            # and compute log probabilities of each token
            #   Type:     [p, p, p, c, c] (p: prompt, c: completion)
            #   labels:   [0, 1, 2, 3, 4]
            #   logits:      [1, 2, 3, 4, next_token]
            #   logprobs:    [?, ?, ?, ?]
            logprobs_per_token = selective_log_softmax(logits[:, :-1], labels[:, 1:])
            # Ignore log probabilities of non-completion tokens
            #   Logprobs: [0, 0, ?, ?]
            logprobs_per_token[~loss_mask[:, 1:]] = 0
            # Add zero column for the first token
            #   Logprobs: [0, 0, 0, ?, ?]
            logprobs_per_token = torch.cat([torch.zeros_like(logprobs_per_token[:, :1]), logprobs_per_token], dim=1)
            return logprobs_per_token

        # NOTE: We manually iterate over `logits_iterator`, as a Python loop over `logits_iterator` allocates
        # `logits` of succeeding iterations with a small overlap in time and we want to avoid this.
        logprobs_all = []
        for _ in range(num_samples):
            logits = next(logits_iterator)  # Shape: (batch_size, sequence_length, vocab_size)
            # Compute log-probabilities of each token
            logprobs_per_token = compute_logprobs_per_token(logits)  # Shape: (batch_size, sequence_length)
            # Compute log-probabilities of each sequence
            logprobs = logprobs_per_token.sum(dim=-1)  # Shape: (batch_size)
            logprobs_all.append(logprobs)

            # Free GPU memory
            del logits, logprobs_per_token, logprobs
        logprobs_all = torch.stack(logprobs_all, dim=1)  # Shape: (batch_size, num_samples)

        return {
            "logprobs_all": logprobs_all,
            "labels": labels,
        }
