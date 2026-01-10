import unittest

from rewarduq.methods.bayesian_linear_head import (
    BayesianLinearHeadModel,
    BayesianLinearHeadModelConfig,
    BayesianLinearHeadTrainer,
    BayesianLinearHeadTrainerConfig,
)

from ..base import BaseModelTest, BaseTrainerTest


class TestBayesianLinearHeadModel(BaseModelTest):
    def create_model(self):
        config = BayesianLinearHeadModelConfig(
            base_model_name_or_path="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        )
        model = BayesianLinearHeadModel(config)
        return model

    def load_model(self, path):
        model = BayesianLinearHeadModel.from_pretrained(path)
        return model


class TestBayesianLinearHeadTrainer(BaseTrainerTest):
    def create_model(self, base_model_name_or_path):
        config = BayesianLinearHeadModelConfig(
            base_model_name_or_path=base_model_name_or_path,
        )
        model = BayesianLinearHeadModel(config)
        return model

    def create_trainer(self, trainer_kwargs, trainer_config_kwargs):
        trainer_config = BayesianLinearHeadTrainerConfig(**trainer_config_kwargs)
        trainer = BayesianLinearHeadTrainer(args=trainer_config, **trainer_kwargs)
        return trainer


if __name__ == "__main__":
    unittest.main()
