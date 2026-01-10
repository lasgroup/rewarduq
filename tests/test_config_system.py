"""Tests for the config system.

These tests validate that the config system correctly loads configs and
creates pipelines that can be trained for all methods.
"""

import tempfile
import unittest
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from rewarduq.methods import load_pipeline_from_config
from rewarduq.utils import get_config, load_dataset_from_config

# Get the path to configs directory relative to this test file
CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# All methods to test
METHODS = [
    "bayesian_linear_head",
    "dpo_head_dropout_ensemble",
    "lora_ensemble",
    "mlp_head_ensemble",
]


class ConfigLoadingTest(unittest.TestCase):
    """Tests for config loading functionality."""

    def test_get_config_loads_train_config(self):
        """Test that get_config loads the train.yaml config correctly."""
        config = get_config(CONFIGS_DIR / "train.yaml")
        self.assertIn("trainer", config)
        self.assertIn("seed", config)
        self.assertIn("task_name", config)

    def test_get_config_with_overwrite(self):
        """Test that get_config correctly applies overwrites."""
        config = get_config(
            CONFIGS_DIR / "train.yaml",
            overwrite=["seed=42", "trainer.max_steps=10"],
        )
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.trainer.max_steps, 10)

    def test_get_config_with_dict_overwrite(self):
        """Test that get_config correctly applies dict overwrites."""
        config = get_config(
            CONFIGS_DIR / "train.yaml",
            overwrite={"seed": 123, "trainer.max_steps": 5},
        )
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.trainer.max_steps, 5)


class DatasetLoadingTest(unittest.TestCase):
    """Tests for dataset loading functionality."""

    def test_load_dataset_from_config(self):
        """Test that load_dataset_from_config loads datasets correctly."""
        dataset_config = get_config(CONFIGS_DIR / "dataset" / "train" / "test_zen.yaml")
        dataset = load_dataset_from_config(dataset_config)
        self.assertIn("chosen", dataset.column_names)
        self.assertIn("rejected", dataset.column_names)


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear Hydra before and after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.mark.parametrize("method", METHODS)
def test_pipeline_training_flow(method):
    """Test that load_pipeline_from_config creates a pipeline correctly for a method."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with initialize_config_dir(config_dir=str(CONFIGS_DIR.absolute()), version_base=None):
            config = compose(
                config_name="train_test",
                overrides=[
                    f"method={method}/test_tiny",
                    f"trainer.output_dir={tmp_dir}",
                    f"paths.features_dir={tmp_dir}",
                ],
            )

        # Load pipeline
        pipeline = load_pipeline_from_config(config)
        assert pipeline is not None

        # Load dataset
        train_dataset = load_dataset_from_config(config.dataset.train)
        assert train_dataset is not None

        # Train (minimal)
        pipeline.train(train_dataset)


if __name__ == "__main__":
    unittest.main()
