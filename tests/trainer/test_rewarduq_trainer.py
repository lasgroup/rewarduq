# based on TRL tests
# https://github.com/huggingface/trl/blob/main/trl/trainer/reward_trainer.py#L86

from __future__ import annotations

import unittest
from typing import Any

import torch
from transformers import PretrainedConfig
from transformers.utils import is_peft_available

from rewarduq.methods.base import RewardUQModel
from rewarduq.trainers import DataCollatorForPreference, RewardUQTrainer, RewardUQTrainerConfig
from rewarduq.utils import extract_features, load_tokenizer_and_model

from ..base import BaseModelTest, BaseTrainerTest

if is_peft_available():
    pass


class DummyRewardUQModelConfig(PretrainedConfig):
    """Configuration class for DummyRewardUQModel."""

    model_type = "dummy_model"

    def __init__(
        self,
        base_model_name_or_path: str = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
        base_model_class: str = "AutoModel",
        base_model_init_kwargs: dict[str, Any] | None = None,
        target_modules: list[str] | None = None,
        peft_config: dict[str, Any] | None = None,
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
        """
        super().__init__(**kwargs)

        # Base model parameters
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_class = base_model_class
        self.base_model_init_kwargs = base_model_init_kwargs or {}
        if "attn_implementation" not in self.base_model_init_kwargs:
            self.base_model_init_kwargs["attn_implementation"] = "eager"
        self.target_modules = target_modules
        self.peft_config = peft_config


class DummyRewardUQModel(RewardUQModel):
    """A dummy model that simulates uncertainty-aware reward outputs based on a pre-trained base model."""

    config_class = DummyRewardUQModelConfig

    def __init__(self, config: DummyRewardUQModelConfig):
        super().__init__(config)

        self.tokenizer, self.base_model_ = load_tokenizer_and_model(
            config.base_model_name_or_path,
            base_model_class=config.base_model_class,
            base_model_init_kwargs=config.base_model_init_kwargs,
            target_modules=config.target_modules,
            peft_config=config.peft_config,
        )
        self.head = torch.nn.Linear(self.base_model_.config.hidden_size, 1, bias=False)

    def get_feature_dependencies(self) -> dict[str]:
        return {
            "base_model_name_or_path": self.base_model_name_or_path,
        }

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model."""
        self.base_model_.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model."""
        self.base_model_.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        """Return the input embeddings from the base model."""
        return self.base_model_.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        features: torch.FloatTensor | None = None,
        output_features: bool = False,
        output_only_features: bool = False,
        **kwargs,
    ):
        result = {}

        if features is None:
            base_model_outputs = self.base_model_(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            features = extract_features(
                base_model_outputs,
                pooling_strategy="last",
                attention_mask=attention_mask,
            )  # shape: (batch_size, feature_dim)

        if output_only_features:
            return features
        if output_features:
            result["features"] = features

        reward = self.head(features)  # shape: (batch_size, 1)
        lower_bound = reward - 0.1
        upper_bound = reward + 0.1

        result["rewards"] = torch.cat(
            [reward, lower_bound, upper_bound],
            dim=1,
        )  # shape: (batch_size, 3)

        return result


class TestDataCollatorForPreference:
    def test_basic_padding(self):
        """Test basic padding functionality without completion masks."""
        self.collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6, 7]},
            {"prompt_input_ids": [8, 9], "chosen_input_ids": [10, 11], "rejected_input_ids": [12, 13]},
        ]

        result = self.collator(examples)

        torch.testing.assert_close(result["chosen_input_ids"], torch.tensor([[1, 2, 3, 4, 5], [0, 8, 9, 10, 11]]))
        torch.testing.assert_close(result["rejected_input_ids"], torch.tensor([[1, 2, 3, 6, 7], [0, 8, 9, 12, 13]]))
        torch.testing.assert_close(result["chosen_attention_mask"], torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]))
        torch.testing.assert_close(result["rejected_attention_mask"], torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]))

    def test_pad_to_multiple_of(self):
        """Test padding to multiple of specified value."""
        collator = DataCollatorForPreference(pad_token_id=0, pad_to_multiple_of=4)
        examples = [
            {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6, 7]},
            {"prompt_input_ids": [8, 9], "chosen_input_ids": [10, 11], "rejected_input_ids": [12, 13]},
        ]

        result = collator(examples)

        torch.testing.assert_close(
            result["chosen_input_ids"], torch.tensor([[0, 1, 2, 3, 4, 5, 0, 0], [0, 0, 8, 9, 10, 11, 0, 0]])
        )
        torch.testing.assert_close(
            result["rejected_input_ids"], torch.tensor([[0, 1, 2, 3, 6, 7, 0, 0], [0, 0, 8, 9, 12, 13, 0, 0]])
        )
        torch.testing.assert_close(
            result["chosen_attention_mask"], torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0]])
        )
        torch.testing.assert_close(
            result["rejected_attention_mask"], torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0]])
        )

    def test_single_example(self):
        """Test collator with a single example."""
        self.collator = DataCollatorForPreference(pad_token_id=0)
        examples = [{"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6, 7]}]

        result = self.collator(examples)

        torch.testing.assert_close(result["chosen_input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
        torch.testing.assert_close(result["rejected_input_ids"], torch.tensor([[1, 2, 3, 6, 7]]))
        torch.testing.assert_close(result["chosen_attention_mask"], torch.tensor([[1, 1, 1, 1, 1]]))
        torch.testing.assert_close(result["rejected_attention_mask"], torch.tensor([[1, 1, 1, 1, 1]]))

    def test_different_pad_token_id(self):
        """Test with different pad token ID."""
        collator = DataCollatorForPreference(pad_token_id=999)
        examples = [
            {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6, 7]},
            {"prompt_input_ids": [8, 9], "chosen_input_ids": [10, 11], "rejected_input_ids": [12, 13]},
        ]

        result = collator(examples)

        torch.testing.assert_close(result["chosen_input_ids"], torch.tensor([[1, 2, 3, 4, 5], [999, 8, 9, 10, 11]]))
        torch.testing.assert_close(result["rejected_input_ids"], torch.tensor([[1, 2, 3, 6, 7], [999, 8, 9, 12, 13]]))
        torch.testing.assert_close(result["chosen_attention_mask"], torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]))
        torch.testing.assert_close(result["rejected_attention_mask"], torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]))

    def test_collate_with_margin(self):
        self.collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6, 7], "margin": 0.1},
            {"prompt_input_ids": [8, 9], "chosen_input_ids": [10, 11], "rejected_input_ids": [12, 13], "margin": 0.2},
        ]

        result = self.collator(examples)
        print(result)

        torch.testing.assert_close(result["chosen_input_ids"], torch.tensor([[1, 2, 3, 4, 5], [0, 8, 9, 10, 11]]))
        torch.testing.assert_close(result["rejected_input_ids"], torch.tensor([[1, 2, 3, 6, 7], [0, 8, 9, 12, 13]]))
        torch.testing.assert_close(result["chosen_attention_mask"], torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]))
        torch.testing.assert_close(result["rejected_attention_mask"], torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]))
        torch.testing.assert_close(result["margin"], torch.tensor([0.1, 0.2]))


class TestDummyRewardUQModel(BaseModelTest):
    def create_model(self):
        config = DummyRewardUQModelConfig(
            base_model_name_or_path="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        )
        model = DummyRewardUQModel(config)
        return model

    def load_model(self, path):
        model = DummyRewardUQModel.from_pretrained(path)
        return model


class TestRewardUQTrainer(BaseTrainerTest):
    def create_model(self, base_model_name_or_path):
        config = DummyRewardUQModelConfig(
            base_model_name_or_path=base_model_name_or_path,
        )
        model = DummyRewardUQModel(config)
        return model

    def create_trainer(self, trainer_kwargs, trainer_config_kwargs):
        trainer_config = RewardUQTrainerConfig(**trainer_config_kwargs)
        trainer = RewardUQTrainer(args=trainer_config, **trainer_kwargs)
        return trainer


if __name__ == "__main__":
    unittest.main()
