from transformers import AutoConfig, AutoModel

from .lora_ensemble_model import LoraEnsembleModel, LoraEnsembleModelConfig
from .lora_ensemble_pipeline import LoraEnsemblePipeline
from .lora_ensemble_trainer import LoraEnsembleTrainer, LoraEnsembleTrainerConfig

AutoConfig.register("lora_ensemble", LoraEnsembleModelConfig)
AutoModel.register(LoraEnsembleModelConfig, LoraEnsembleModel)
