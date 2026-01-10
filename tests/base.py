# based on TRL tests
# https://github.com/huggingface/trl/blob/main/trl/trainer/reward_trainer.py#L86

from __future__ import annotations

import tempfile

import pytest
import torch
from datasets import load_dataset
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

if is_peft_available():
    from peft import LoraConfig, PeftModel, get_peft_model


class BaseModelTest:
    PASS_COMPLETION_MASK = False

    @pytest.fixture(autouse=True)
    def set_tmp_dir(self, tmp_path):
        self.tmp_dir = str(tmp_path)

    def create_model(self):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def test_initialization(self):
        """Test model can be initialized."""
        model = self.create_model()
        assert model is not None

    def test_save_and_load(self):
        """Test model can be saved and loaded with identical weights."""
        model = self.create_model()

        # Modify all weights to ensure we're testing actual save/load
        torch.manual_seed(12345)
        for _, param in model.named_parameters():
            param.data = torch.randn_like(param.data)

        # Store original weights for comparison
        original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            model.save_pretrained(tmp_dir)

            # Load model
            loaded_model = self.load_model(tmp_dir)
            assert loaded_model is not None

            # Verify all weights match exactly
            loaded_state_dict = loaded_model.state_dict()
            assert set(original_state_dict.keys()) == set(loaded_state_dict.keys())

            for key in original_state_dict:
                torch.testing.assert_close(
                    original_state_dict[key],
                    loaded_state_dict[key],
                    msg=f"Weight mismatch for {key}",
                )

    def test_forward(self):
        """Test forward pass returns expected outputs."""
        model = self.create_model()

        # Create dummy inputs
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        # Forward pass
        if self.PASS_COMPLETION_MASK:
            completion_mask = torch.ones(batch_size, seq_length)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, completion_mask=completion_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check outputs
        assert "rewards" in outputs
        assert outputs["rewards"].shape == (batch_size, 3)  # (rewards, lower, upper)


class BaseTrainerTest:
    TAGS = ["reward-trainer", "trl"]

    def create_model(self, base_model_name_or_path):
        raise NotImplementedError

    def create_trainer(self, trainer_kwargs, trainer_config_kwargs):
        raise NotImplementedError

    @pytest.fixture(autouse=True)
    def set_tmp_dir(self, tmp_path):
        self.tmp_dir = str(tmp_path)

    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_preference",
            "conversational_preference",
            "standard_implicit_prompt_preference",
            "conversational_implicit_prompt_preference",
        ],
    )
    def test_train_dataset_types(self, config_name):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        # Initialize the model
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if new_param.requires_grad:
                assert not torch.allclose(param, new_param), f"Trainable parameter {n} has not changed"
            else:
                assert torch.allclose(param, new_param), f"Non-trainable parameter {n} has changed"

    @pytest.mark.parametrize(
        "base_model_name_or_path",
        [
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            "trl-internal-testing/tiny-Qwen3MoeForSequenceClassification",
            "trl-internal-testing/tiny-LlamaForSequenceClassification-3.2",
        ],
    )
    def test_train_model(self, base_model_name_or_path):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the model
        model = self.create_model(base_model_name_or_path)

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if new_param.requires_grad:
                assert not torch.allclose(param, new_param), f"Trainable parameter {n} has not changed"
            else:
                assert torch.allclose(param, new_param), f"Non-trainable parameter {n} has changed"

    def test_train_from_causal_lm(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the model
        model = self.create_model("trl-internal-testing/tiny-Qwen3ForCausalLM")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if new_param.requires_grad:
                assert not torch.allclose(param, new_param), f"Trainable parameter {n} has not changed"
            else:
                assert torch.allclose(param, new_param), f"Non-trainable parameter {n} has changed"

    @pytest.mark.skip  # TODO: Enable when peft and gradient checkpointing is fully supported
    def test_train_with_gradient_checkpointing(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the model
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
                "gradient_checkpointing": True,
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if new_param.requires_grad:
                assert not torch.allclose(param, new_param), f"Trainable parameter {n} has not changed"
            else:
                assert torch.allclose(param, new_param), f"Non-trainable parameter {n} has changed"

    @require_peft
    @pytest.mark.skip  # TODO: Enable when peft and gradient checkpointing is fully supported
    def test_train_dense_with_peft_config(self):
        # Get the base model parameter names
        model_config = self.MODEL_CONFIG_CLASS(
            base_model_name_or_path="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            **self.MODEL_CONFIG_CLASS_KWARGS,
        )
        model = self.MODEL_CLASS(model_config)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
                "peft_config": LoraConfig(target_modules="all-linear"),
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model parameters to be the same
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    @pytest.mark.skip  # TODO: Enable when peft and gradient checkpointing is fully supported
    def test_train_moe_with_peft_config(self):
        # Get the base model parameter names
        model = self.create_model("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
                "peft_config": LoraConfig(target_modules=["up_proj", "down_proj", "score"]),
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model parameters to be the same
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    @pytest.mark.skip  # TODO: Enable when peft and gradient checkpointing is fully supported
    def test_train_peft_model(self):
        # Get the base model
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")

        # Get the base model parameter names
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Turn the model into a peft model
        lora_config = LoraConfig(target_modules="all-linear")
        model = get_peft_model(model, lora_config)

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model parameters to be the same
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    @pytest.mark.skip  # TODO: Enable when peft and gradient checkpointing is fully supported
    def test_train_dense_with_peft_config_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
                "peft_config": LoraConfig(target_modules="all-linear"),
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
                "gradient_checkpointing": True,
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model parameters to be the same
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    @pytest.mark.skip  # TODO: Enable when peft and gradient checkpointing is fully supported
    def test_train_moe_with_peft_config_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model = self.create_model("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
                "peft_config": LoraConfig(target_modules=["up_proj", "down_proj", "score"]),
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
                "gradient_checkpointing": True,
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model parameters to be the same
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    @pytest.mark.skip  # TODO: Enable when peft and gradient checkpointing is fully supported
    def test_train_with_peft_model_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        model = get_peft_model(model, LoraConfig(target_modules="all-linear"))

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
                "gradient_checkpointing": True,
            },
        )

        # Verify model is a PeftModel
        assert isinstance(trainer.model, PeftModel)

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model parameters to be the same
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_with_iterable_dataset(self):
        # Get the dataset
        dataset = load_dataset(
            "trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train", streaming=True
        )

        # Initialize the model
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
                "max_steps": 3,
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if new_param.requires_grad:
                assert not torch.allclose(param, new_param), f"Trainable parameter {n} has not changed"
            else:
                assert torch.allclose(param, new_param), f"Non-trainable parameter {n} has changed"

    def test_train_with_chat_template_kwargs(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the model and tokenizer
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")
        tokenizer = model.tokenizer

        # The following template is a simplified version of the Qwen chat template, where an additional argument
        # `role_capital` is used to control the capitalization of roles.
        tokenizer.chat_template = '{%- if messages[0]["role"] == "system" -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\n" + messages[0]["content"] + "<|im_end|>\\n" }}{%- else -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n" }}{%- endif -%}{%- for message in messages -%}    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) -%}        {{ "<|im_start|>" + (message.role.upper() if role_capital else message.role) + "\\n" + message.content + "<|im_end|>\\n" }}    {%- elif message.role == "assistant" -%}        {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") }}        {%- if message.content -%}            {{ "\\n" + message.content }}        {%- endif -%}        {{ "<|im_end|>\\n" }}    {%- elif message.role == "tool" -%}        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") -%}            {{ "<|im_start|>" + ("USER" if role_capital else "user") }}        {%- endif -%}        {{ "\\n<tool_response>\\n" + message.content + "\\n</tool_response>" }}        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") -%}            {{ "<|im_end|>\\n" }}        {%- endif -%}    {%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}    {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") + "\\n" }}{%- endif -%}'  # noqa: E501

        dataset.add_column("chat_template_kwargs", [{"role_capital": bool(i % 2)} for i in range(len(dataset))])

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if new_param.requires_grad:
                assert not torch.allclose(param, new_param), f"Trainable parameter {n} has not changed"
            else:
                assert torch.allclose(param, new_param), f"Non-trainable parameter {n} has changed"

    def test_train_with_eval(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference")

        # Initialize the model
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset["train"],
                "eval_dataset": dataset["test"],
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
                "eval_strategy": "steps",
                "eval_steps": 3,
            },
        )

        # Train the model
        trainer.train()

        # Check that the eval loss is not None
        assert trainer.state.log_history[0]["eval_loss"] is not None

    def test_train_with_multiple_eval_dataset(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference")

        # Initialize the model
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset["train"],
                "eval_dataset": {"data1": dataset["test"], "data2": dataset["test"]},
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
                "eval_strategy": "steps",
                "eval_steps": 3,
            },
        )

        # Train the model
        trainer.train()

        # Check that the eval losses are not None
        assert trainer.state.log_history[-3]["eval_data1_loss"] is not None
        assert trainer.state.log_history[-2]["eval_data2_loss"] is not None

    def test_tag_added(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the model
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
            },
        )

        for tag in self.TAGS:
            assert tag in trainer.model.model_tags

    @require_peft
    @pytest.mark.skip  # TODO: Enable when PEFT is supported
    def test_tag_added_peft(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the model
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
                "peft_config": LoraConfig(target_modules="all-linear"),
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
            },
        )

        for tag in self.TAGS:
            assert tag in trainer.model.model_tags

    def test_train_with_margin(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        def add_margin(example):
            # dummy margin based on the length of the chosen summary
            return {"margin": len(example["chosen"])}

        dataset = dataset.map(add_margin)

        # Initialize the model
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if new_param.requires_grad:
                assert not torch.allclose(param, new_param), f"Trainable parameter {n} has not changed"
            else:
                assert torch.allclose(param, new_param), f"Non-trainable parameter {n} has changed"

    def test_train_with_center_rewards_coefficient(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the model
        model = self.create_model("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")

        # Initialize the trainer
        trainer = self.create_trainer(
            trainer_kwargs={
                "model": model,
                "train_dataset": dataset,
                "processing_class": model.tokenizer,
            },
            trainer_config_kwargs={
                "output_dir": self.tmp_dir,
                "report_to": "none",
                "center_rewards_coefficient": 0.01,
            },
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if new_param.requires_grad:
                assert not torch.allclose(param, new_param), f"Trainable parameter {n} has not changed"
            else:
                assert torch.allclose(param, new_param), f"Non-trainable parameter {n} has changed"
