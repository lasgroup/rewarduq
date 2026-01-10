"""Model loading, tokenization, and feature extraction utilities."""

from __future__ import annotations

import importlib
from typing import Any, Literal

import torch
import torch.nn as nn
from datasets import Dataset
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.utils import ModelOutput
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import DPODataCollatorWithPadding

from .logging import get_logger

logger = get_logger(__name__)


def get_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Convert a string or torch.dtype to a torch.dtype."""
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid torch dtype: torch.{dtype}")

    return dtype


def load_tokenizer_and_model(
    base_model_name_or_path: str,
    base_model_class: str = "AutoModel",
    base_model_init_kwargs: dict[str, Any] | DictConfig | None = None,
    target_modules: list[str] | None = None,
    peft_config: dict[str, Any] | DictConfig | None = None,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, PreTrainedModel | None]:
    """Load a tokenizer and model from the specified model name or path.

    Args:
        base_model_name_or_path (str):
            The name or path of the model to load.

        base_model_class (str):
            The class of the model to load. This should be a class from the `transformers` library, such as `AutoModel`,
            `AutoModelForCausalLM`, etc. Defaults to `AutoModel`.

        base_model_init_kwargs (dict[str, Any] | DictConfig | None):
            Additional keyword arguments to pass to the model loading function.

        target_modules (list[str] | None):
            A list of module names to finetune. If `None`, all modules will be finetuned.
            If provided, only the specified modules will have their `requires_grad` set to `True`.

        peft_config (dict[str, Any] | DictConfig | None):
            If provided, the model will be loaded with a PEFT adapter. The dictionary should contain
            the configuration for the PEFT adapter, such as `lora_alpha`, `lora_r`, `lora_dropout`, etc.
            If `None`, no PEFT adapter will be loaded.

    Returns:
        tokenizer_and_model (tuple[PreTrainedTokenizerBase, PreTrainedModel, PreTrainedModel | None]):
            The loaded tokenizer, model, and an optional reference model.
    """
    if base_model_init_kwargs is None:
        base_model_init_kwargs = {}
    if "dtype" in base_model_init_kwargs and base_model_init_kwargs["dtype"] != "auto":
        base_model_init_kwargs["dtype"] = get_torch_dtype(base_model_init_kwargs["dtype"])

    # Load model from HuggingFace
    AutoModelClass = getattr(importlib.import_module("transformers"), base_model_class)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    model = AutoModelClass.from_pretrained(base_model_name_or_path, **base_model_init_kwargs)
    logger.info(f"Loaded base model: {base_model_name_or_path}")

    if peft_config is not None:
        # Load PEFT adapter
        model = get_peft_model(model, LoraConfig(**peft_config))
        # if config.bf16 and getattr(model, "is_loaded_in_4bit", False): # TODO check bf16
        #     peft_module_casting_to_bf16(model)
        #     # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
        #     self._peft_has_been_casted_to_bf16 = True
        logger.info(f"Loaded PEFT adapter: {model.peft_config}")

    # Finetune only specified target modules
    if target_modules is not None:
        for name, param in model.named_parameters():
            module_name = name.rsplit(".", maxsplit=1)[0]
            param.requires_grad = any(module_name.endswith(f".{module}") for module in target_modules)
        logger.info(f"Finetuning only target modules: {target_modules}")

    # Set defaults for tokenizer
    if tokenizer.padding_side is None:
        logger.warning(f"Tokenizer from {base_model_name_or_path} does not have a padding side. Setting it to 'left'.")
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        logger.warning(f"Tokenizer from {base_model_name_or_path} does not have a pad token. Setting it to eos token.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.chat_template is None:
        raise ValueError(f"Tokenizer from {base_model_name_or_path} does not have a chat template.")

    return tokenizer, model


def log_model_summary(model: torch.nn.Module):
    """Print a summary of the model including the number of trainable parameters.

    Args:
        model (torch.nn.Module):
            The model to summarize.
    """

    def get_num_params(param: torch.nn.Parameter) -> int:
        # Reference: https://github.com/huggingface/peft/blob/b5ace6a8c4b826f2938055c559078528727281ed/src/peft/peft_model.py#L813

        num_params = param.numel()

        # If using DS Zero 3 and the weights are initialized empty.
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters.
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes_per_param = param.element_size()
            elif hasattr(param, "quant_storage"):
                num_bytes_per_param = param.quant_storage.itemsize
            else:
                num_bytes_per_param = 1
            num_params *= num_bytes_per_param * 2

        return num_params

    # Count number of model parameters
    n_params_total = sum(get_num_params(p) for p in model.parameters())
    n_params_trainable = sum(get_num_params(p) for p in model.parameters() if p.requires_grad)

    # Log model summary
    logger.info(
        f"Loaded model: {model.__class__.__name__}"
        f"\n{model}"
        f"\n\nNumber of trainable parameters: {n_params_trainable:,d} / {n_params_total:,d}"
        f" ({100 * n_params_trainable / n_params_total:.2f}%)"
    )


def extract_features(
    outputs: ModelOutput,
    feature_layer: str | int = "last_hidden_state",
    pooling_strategy: Literal["last", "min", "max", "mean"] | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract features from the model outputs.

    If `attention_mask` is provided, the pooling strategy will be applied
    to the non-masked tokens. Otherwise, the pooling strategy will be applied to all tokens.

    Args:
        outputs:
            The model outputs to extract features from.

        feature_layer (str | int, optional):
            The layer from which to extract features.

            If a string is provided, it will be used as the name of the attribute to extract
            from the outputs (e.g. `last_hidden_state` or `logits`).

            If an integer is provided, it will be used as the index of the `hidden_states`
            attribute of the outputs.

        pooling_strategy (str | None, optional):
            The strategy to use for reducing the features over the sequence length.
            Must be either "last", "min", "max", "mean" or None.

        attention_mask (torch.Tensor of shape (batch_size, sequence_length), optional):
            The attention mask used to generate the outputs. This is used for the pooling strategy.

    Returns:
        features (torch.Tensor of shape (batch_size, feature_size)):
            The extracted features.
    """
    # Extract features from model outputs
    #   Shape: (batch_size, sequence_length, *feature_size)
    if isinstance(feature_layer, str):
        features = getattr(outputs, feature_layer)
    elif isinstance(feature_layer, int):
        features = outputs.hidden_states[feature_layer]
    else:
        raise ValueError(f"Invalid feature layer: {feature_layer}")

    if features is None:
        raise ValueError(f"Model outputs do not contain the feature layer: {feature_layer}")
    if torch.isnan(features).any():
        raise ValueError("Extracted features contain NaN values.")

    # apply pooling strategy over sequence length
    if pooling_strategy is None:
        return features  # Shape: (batch_size, sequence_length, *feature_size)
    else:
        if attention_mask is None:
            attention_mask = torch.ones(features.shape[:2], device=features.device, dtype=torch.int)
        else:
            attention_mask = attention_mask.to(features.device)

        if pooling_strategy == "last":
            # Extract features of the last non-masked tokens
            last_indices = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
            features_pooled = features[torch.arange(features.shape[0], device=features.device), last_indices]
        elif pooling_strategy == "min":
            # Compute min of features over non-masked tokens
            features[~attention_mask.bool()] = float("inf")
            features_pooled = features.min(dim=1).values
        elif pooling_strategy == "max":
            # Compute max of features over non-masked tokens
            features[~attention_mask.bool()] = float("-inf")
            features_pooled = features.max(dim=1).values
        elif pooling_strategy == "mean":
            # Compute mean of features over non-masked tokens
            features[~attention_mask.bool()] = float("nan")
            features_pooled = features.nanmean(dim=1)
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

        return features_pooled  # Shape: (batch_size, *feature_size)


def tokenize_sample(
    features,
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int,
    max_completion_length: int,
    add_special_tokens: bool = False,
) -> dict[str, torch.LongTensor]:
    """Tokenize samples into input IDs and attention masks.

    Args:
        features (dict[str, str]): A dictionary with the possible keys `prompt`,
            `completion`, `chosen` or `rejected`.
        tokenizer (PreTrainedTokenizer): The tokenizer used.
        max_prompt_length (int): The maximum length to which the prompt is
            truncated.
        max_completion_length (int): The maximum length to which a completion is
            truncated.
        add_special_tokens (bool, optional): Flag whether to add special tokens.
            Defaults to False.

    Returns:
        features_tokenized (dict[str, torch.LongTensor]): A dictionary with the keys
            `{key}_input_ids` and `{key}_attention_mask` for each key found in
            `features`.
    """
    # Reference: https://github.com/huggingface/trl/blob/2bc182c4fb8180a3cb815c61ca6c528a86559d14/trl/trainer/dpo_trainer.py#L575
    features_tokenized = {}

    # Tokenize prompt
    def tokenize_prompt(prompt, prefix=""):
        prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
        prompt_input_ids = prompt_tokenized["input_ids"]
        prompt_attention_mask = prompt_tokenized["attention_mask"]
        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
                prompt_attention_mask = [1] + prompt_attention_mask
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
                prompt_attention_mask = prompt_attention_mask + [1]
        # Truncate prompt
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
            prompt_attention_mask = prompt_attention_mask[-max_prompt_length:]
        return {
            f"{prefix}input_ids": prompt_input_ids,
            f"{prefix}attention_mask": prompt_attention_mask,
        }

    if "prompt" in features:
        features_tokenized.update(tokenize_prompt(features["prompt"], prefix="prompt_"))

    # Tokenize completions
    def tokenize_completion(completion, prefix=""):
        completion_tokenized = tokenizer(completion, add_special_tokens=False)
        completion_input_ids = completion_tokenized["input_ids"]
        completion_attention_mask = completion_tokenized["attention_mask"]
        # Add eos token
        completion_input_ids = completion_input_ids + [tokenizer.eos_token_id]
        completion_attention_mask = completion_attention_mask + [1]
        # Truncate completion
        if max_completion_length is not None:
            completion_input_ids = completion_input_ids[:max_completion_length]
            completion_attention_mask = completion_attention_mask[:max_completion_length]
        return {
            f"{prefix}input_ids": completion_input_ids,
            f"{prefix}attention_mask": completion_attention_mask,
        }

    if "completion" in features:
        features_tokenized.update(tokenize_completion(features["completion"], prefix="completion_"))
    if "chosen" in features:
        features_tokenized.update(tokenize_completion(features["chosen"], prefix="chosen_"))
    if "rejected" in features:
        features_tokenized.update(tokenize_completion(features["rejected"], prefix="rejected_"))

    return features_tokenized


def encode_prompt_completion(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int | None = None,
    max_completion_length: int | None = None,
) -> dict[str, torch.LongTensor]:
    """Encode the given (prompt, chosen, rejected) triples into a batch.

    Args:
        prompts (list[list[dict[str, str]]]): The list of prompts.
        completions (list[list[dict[str, str]]]): The list of chosen completions.
        tokenizer (PreTrainedTokenizer): The tokenizer used to process the text.
        max_prompt_length (int | None): Maximum length of the prompt sequence. If
            `None`, the prompt sequence is not truncated.
        max_completion_length (int | None): Maximum length of the completion sequences.
            If `None`, the completion sequences are not truncated.

    Returns:
        batch (dict[str, torch.LongTensor]): Tokenized sequences with the following keys:
            - prompt_input_ids
            - prompt_attention_mask
            - completion_input_ids
            - completion_attention_mask
    """
    # Reference: https://github.com/huggingface/trl/blob/2bc182c4fb8180a3cb815c61ca6c528a86559d14/trl/trainer/dpo_trainer.py#L530

    # create dataset for convenient processing
    dataset = Dataset.from_dict(
        {  # TODO avoid overhead of dataset?
            "prompt": prompts,
            "completion": completions,
        }
    )

    # Convert conversational to standard dataset if necessary
    dataset = dataset.map(
        maybe_apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Applying chat template",
    )

    # Tokenize dataset
    dataset = dataset.map(
        tokenize_sample,
        remove_columns=["prompt", "completion"],
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            # For enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
            "add_special_tokens": False,
        },
        desc="Tokenizing",
    )

    # Collate dataset into a single batch
    data_collator = DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id)
    batch = data_collator(dataset)

    return batch


def encode_prompt_chosen_rejected(
    prompts: list[list[dict[str, str]]],
    chosen: list[list[dict[str, str]]],
    rejected: list[list[dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int | None = None,
    max_completion_length: int | None = None,
) -> dict[str, torch.LongTensor]:
    """Encode the given (prompt, chosen, rejected) triples into a batch.

    Args:
        prompts (list[list[dict[str, str]]]): The list of prompts.
        chosen (list[list[dict[str, str]]]): The list of chosen completions.
        rejected (list[list[dict[str, str]]]): The list of rejected completions.
        tokenizer (PreTrainedTokenizer): The tokenizer used to process the text.
        max_prompt_length (int | None): Maximum length of the prompt sequence. If
            `None`, the prompt sequence is not truncated.
        max_completion_length (int | None): Maximum length of the completion sequences.
            If `None`, the completion sequences are not truncated.

    Returns:
        batch (dict[str, torch.LongTensor]): Tokenized sequences with the following keys:
            - prompt_input_ids
            - prompt_attention_mask
            - chosen_input_ids
            - chosen_attention_mask
            - rejected_input_ids
            - rejected_attention_mask
    """
    # Reference: https://github.com/huggingface/trl/blob/2bc182c4fb8180a3cb815c61ca6c528a86559d14/trl/trainer/dpo_trainer.py#L530

    # create dataset for convenient processing
    dataset = Dataset.from_dict(
        {  # TODO avoid overhead of dataset?
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
        }
    )

    # Convert conversational to standard dataset if necessary
    dataset = dataset.map(
        maybe_apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Applying chat template",
    )

    # Tokenize dataset
    dataset = dataset.map(
        tokenize_sample,
        remove_columns=["prompt", "chosen", "rejected"],
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            # For enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
            "add_special_tokens": False,
        },
        desc="Tokenizing",
    )

    # Collate dataset into a single batch
    data_collator = DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id)
    batch = data_collator(dataset)

    return batch


def apply_initialization_to_linear_layers(
    module: nn.Module,
    initialization: type[nn.Module],
    initialization_kwargs: dict[str, Any] | None = None,
) -> None:
    """Apply the given initialization to all Linear layers in the module.

    Bias terms are initialized to zero.
    """

    def _apply_initialization(m):
        if isinstance(m, nn.Linear):
            if initialization == nn.init.xavier_uniform_ or initialization == nn.init.xavier_normal_:
                initialization(m.weight, **(initialization_kwargs or {}))
            elif initialization:
                initialization(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    module.apply(_apply_initialization)
