import unittest

from rewarduq.methods.mlp_head_ensemble import (
    MLPHeadEnsembleModel,
    MLPHeadEnsembleModelConfig,
    MLPHeadEnsembleTrainer,
    MLPHeadEnsembleTrainerConfig,
)

from ..base import BaseModelTest, BaseTrainerTest


class TestMLPHeadEnsembleModel(BaseModelTest):
    def create_model(self):
        config = MLPHeadEnsembleModelConfig(
            base_model_name_or_path="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            head_hidden_dim=16,
            num_heads=3,
        )
        model = MLPHeadEnsembleModel(config)
        return model

    def load_model(self, path):
        model = MLPHeadEnsembleModel.from_pretrained(path)
        return model


class TestMLPHeadEnsembleTrainer(BaseTrainerTest):
    def create_model(self, base_model_name_or_path):
        config = MLPHeadEnsembleModelConfig(
            base_model_name_or_path=base_model_name_or_path,
            head_hidden_dim=16,
            num_heads=3,
        )
        model = MLPHeadEnsembleModel(config)
        return model

    def create_trainer(self, trainer_kwargs, trainer_config_kwargs):
        trainer_config = MLPHeadEnsembleTrainerConfig(**trainer_config_kwargs)
        trainer = MLPHeadEnsembleTrainer(args=trainer_config, **trainer_kwargs)
        return trainer


if __name__ == "__main__":
    unittest.main()
