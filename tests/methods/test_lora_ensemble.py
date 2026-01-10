import tempfile
import unittest

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

from rewarduq.methods.lora_ensemble import (
    LoraEnsembleModel,
    LoraEnsembleModelConfig,
    LoraEnsembleTrainer,
    LoraEnsembleTrainerConfig,
)
from rewarduq.utils import load_tokenizer_and_model


class TestLoraEnsembleModel(unittest.TestCase):
    """Tests for LoraEnsembleModel with 3 independent adapters."""

    @classmethod
    def setUpClass(cls):
        """Create 3 independent LoRA adapters for testing."""
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.adapter_paths = []
        cls.base_model_name = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"

        # Create 3 independent LoRA models with different random initializations
        for i in range(3):
            # Each model gets fresh random initialization
            torch.manual_seed(i * 100)  # Different seed for each adapter
            model = AutoModelForSequenceClassification.from_pretrained(
                cls.base_model_name,
                num_labels=1,
            )
            peft_config = LoraConfig(target_modules="all-linear")
            model = get_peft_model(model, peft_config)

            # Randomize the LoRA weights to make each adapter distinct
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.data.normal_(mean=0.0, std=0.02)

            path = f"{cls.tmp_dir.name}/adapter_{i}"
            model.save_pretrained(path)
            cls.adapter_paths.append(path)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        cls.tmp_dir.cleanup()

    def _create_model(self) -> LoraEnsembleModel:
        """Create a LoraEnsembleModel with the test adapters."""
        config = LoraEnsembleModelConfig(
            base_model_name_or_path=self.base_model_name,
            peft_config=LoraConfig(target_modules="all-linear"),
            adapter_paths=self.adapter_paths,
        )
        return LoraEnsembleModel(config)

    def test_initialization(self):
        """Test model can be initialized with config."""
        model = self._create_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.base_model_)
        self.assertEqual(model.config.ensemble_num, 3)

    def test_forward(self):
        """Test forward pass returns expected outputs."""
        model = self._create_model()
        model.eval()

        # Create dummy inputs
        batch_size = 2
        seq_length = 10
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check outputs
        self.assertIn("rewards", outputs)
        self.assertEqual(outputs["rewards"].shape, (batch_size, 3))  # (reward, lower, upper)

    def test_forward_with_individual_rewards(self):
        """Test forward pass with individual rewards output."""
        model = self._create_model()
        model.eval()

        batch_size = 2
        seq_length = 10
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_individual_rewards=True,
            )

        self.assertIn("individual_rewards", outputs)
        self.assertEqual(outputs["individual_rewards"].shape, (batch_size, 3))  # 3 adapters

    def test_adapters_produce_different_outputs(self):
        """Test that each adapter produces different outputs (ensemble diversity)."""
        model = self._create_model()
        model.eval()

        batch_size = 2
        seq_length = 10
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_individual_rewards=True,
            )

        individual_rewards = outputs["individual_rewards"]  # (batch_size, 3)

        # Check that not all adapters produce identical outputs
        # (they should differ since we initialized them with different seeds)
        all_same = torch.allclose(individual_rewards[:, 0], individual_rewards[:, 1], atol=1e-6) and torch.allclose(
            individual_rewards[:, 1], individual_rewards[:, 2], atol=1e-6
        )
        self.assertFalse(all_same, "All adapters produce identical outputs - no ensemble diversity")

    def test_save_and_load(self):
        """Test model can be saved and loaded with identical weights.

        Note: LoraEnsembleModel saves config and adapters are loaded from
        their original paths. The state dict should match after loading.
        """
        model = self._create_model()

        # Store original weights for comparison
        original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as save_dir:
            # Save model config
            model.save_pretrained(save_dir)

            # Load model
            loaded_model = LoraEnsembleModel.from_pretrained(save_dir)

            # Check config
            self.assertEqual(model.config.ensemble_num, loaded_model.config.ensemble_num)
            self.assertEqual(model.config.adapter_paths, loaded_model.config.adapter_paths)

            # Verify all weights match exactly
            loaded_state_dict = loaded_model.state_dict()
            self.assertEqual(
                set(original_state_dict.keys()),
                set(loaded_state_dict.keys()),
                "State dict keys don't match",
            )

            for key in original_state_dict:
                torch.testing.assert_close(
                    original_state_dict[key],
                    loaded_state_dict[key],
                    msg=f"Weight mismatch for {key}",
                )

    def test_reload_adapters(self):
        """Test reloading adapters from different paths."""
        model = self._create_model()
        model.eval()

        batch_size = 2
        seq_length = 10
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        # Get initial outputs
        with torch.no_grad():
            outputs_initial = model(input_ids=input_ids, attention_mask=attention_mask)

        # Reload with same adapters
        model.reload_adapters(self.adapter_paths)

        # Get outputs after reload
        with torch.no_grad():
            outputs_after = model(input_ids=input_ids, attention_mask=attention_mask)

        # Should produce same outputs
        torch.testing.assert_close(
            outputs_initial["rewards"],
            outputs_after["rewards"],
            msg="Reloaded model produces different outputs",
        )


class TestLoraEnsembleTrainer(unittest.TestCase):
    """Tests for LoraEnsembleTrainer training a single LoRA model.

    Similar to the pipeline, this trains a normal LoRA model (not the ensemble).
    The ensemble is created by combining multiple independently trained LoRA models.
    """

    def test_train_single_lora_model(self):
        """Test training a single LoRA model similar to the pipeline."""
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the model similar to the pipeline (train case)
        base_model_name = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        peft_config = {"target_modules": "all-linear"}  # dict for load_tokenizer_and_model

        tokenizer, model = load_tokenizer_and_model(
            base_model_name,
            base_model_class="AutoModelForSequenceClassification",
            base_model_init_kwargs={"num_labels": 1},
            target_modules=None,
            peft_config=peft_config,
        )

        # Zero out the final score layer (like the pipeline)
        model.score.weight.data.zero_()
        if model.score.bias is not None:
            model.score.bias.data.zero_()

        # Save initial trainable params to verify they change
        previous_trainable_params = {n: param.clone() for n, param in model.named_parameters() if param.requires_grad}

        # Initialize the trainer
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = LoraEnsembleTrainerConfig(
                output_dir=tmp_dir,
                report_to="none",
                num_train_epochs=1,
                per_device_train_batch_size=2,
            )
            trainer = LoraEnsembleTrainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
            )

            # Train the model
            trainer.train()

            # Check that the training loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1].get("train_loss"))

            # Check the trainable params have changed
            for n, param in previous_trainable_params.items():
                new_param = model.get_parameter(n).cpu()
                if new_param.requires_grad:
                    self.assertFalse(
                        torch.allclose(param, new_param),
                        f"Trainable parameter {n} has not changed",
                    )

    def test_train_and_save_adapter(self):
        """Test training a LoRA model and saving the adapter."""
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        base_model_name = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        peft_config = {"target_modules": "all-linear"}  # dict for load_tokenizer_and_model

        tokenizer, model = load_tokenizer_and_model(
            base_model_name,
            base_model_class="AutoModelForSequenceClassification",
            base_model_init_kwargs={"num_labels": 1},
            target_modules=None,
            peft_config=peft_config,
        )

        model.score.weight.data.zero_()
        if model.score.bias is not None:
            model.score.bias.data.zero_()

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = LoraEnsembleTrainerConfig(
                output_dir=tmp_dir,
                report_to="none",
                num_train_epochs=1,
                per_device_train_batch_size=2,
            )
            trainer = LoraEnsembleTrainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
            )

            # Train
            trainer.train()

            # Save the adapter
            adapter_path = f"{tmp_dir}/adapter"
            model.save_pretrained(adapter_path)

            # Verify we can load it as an ensemble (with 1 adapter)
            ensemble_config = LoraEnsembleModelConfig(
                base_model_name_or_path=base_model_name,
                peft_config=peft_config,
                adapter_paths=[adapter_path],
            )
            ensemble_model = LoraEnsembleModel(ensemble_config)
            self.assertEqual(ensemble_model.config.ensemble_num, 1)

            # Verify forward pass works
            batch_size = 2
            seq_length = 10
            torch.manual_seed(42)
            input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

            ensemble_model.eval()
            with torch.no_grad():
                outputs = ensemble_model(input_ids=input_ids, attention_mask=attention_mask)

            self.assertIn("rewards", outputs)
            self.assertEqual(outputs["rewards"].shape, (batch_size, 3))


if __name__ == "__main__":
    unittest.main()
