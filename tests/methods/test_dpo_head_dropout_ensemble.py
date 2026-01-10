import unittest

import pytest

from rewarduq.methods.dpo_head_dropout_ensemble import (
    DPOHeadDropoutEnsembleModel,
    DPOHeadDropoutEnsembleModelConfig,
    DPOHeadDropoutEnsembleTrainer,
    DPOHeadDropoutEnsembleTrainerConfig,
)

from ..base import BaseModelTest, BaseTrainerTest


class TestDPOHeadDropoutEnsembleModel(BaseModelTest):
    PASS_COMPLETION_MASK = True

    def create_model(self):
        config = DPOHeadDropoutEnsembleModelConfig(
            base_model_name_or_path="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            num_samples_eval=3,
        )
        model = DPOHeadDropoutEnsembleModel(config)
        model.init_ref_model()
        return model

    def load_model(self, path):
        model = DPOHeadDropoutEnsembleModel.from_pretrained(path)
        return model

    @pytest.mark.skip
    def test_save_and_load(self):
        # TODO: Fix this test
        pass


class TestDPOHeadDropoutEnsembleTrainer(BaseTrainerTest):
    TAGS = ["trl", "dpo"]

    def create_model(self, base_model_name_or_path):
        base_model_name_or_path = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"  # Force causal model
        config = DPOHeadDropoutEnsembleModelConfig(
            base_model_name_or_path=base_model_name_or_path,
            num_samples_eval=3,
        )
        model = DPOHeadDropoutEnsembleModel(config)
        model.init_ref_model()
        return model

    def create_trainer(self, trainer_kwargs, trainer_config_kwargs):
        trainer_config_kwargs["learning_rate"] = 10  # Use high learning rate
        trainer_config = DPOHeadDropoutEnsembleTrainerConfig(**trainer_config_kwargs)
        trainer = DPOHeadDropoutEnsembleTrainer(args=trainer_config, **trainer_kwargs)
        return trainer

    def test_train_model(self):
        # Only test with causal model
        super().test_train_model("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")


if __name__ == "__main__":
    unittest.main()
