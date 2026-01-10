from __future__ import annotations

import glob
import json
import os
from typing import Any, Literal

import torch
from safetensors import safe_open
from torch import nn
from transformers import PretrainedConfig

from rewarduq.methods import RewardUQModel
from rewarduq.utils import extract_features, get_logger, load_tokenizer_and_model, log_model_summary

logger = get_logger(__name__)


class BayesianLinearHeadModelConfig(PretrainedConfig):
    """Configuration class for `BayesianLinearHeadModel`."""

    model_type = "bayesian_linear_head_model"

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen3-0.6B",
        base_model_class: str = "AutoModel",
        base_model_init_kwargs: dict[str, Any] | None = None,
        lambda_reg: float = 1e-5,
        std_beta: float = 1,
        feature_extraction_layer: str | int = "last_hidden_state",
        feature_extraction_pooling_strategy: str | None = "last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_class = base_model_class
        self.base_model_init_kwargs = base_model_init_kwargs or {}
        self.lambda_reg = lambda_reg
        self.std_beta = std_beta
        self.feature_extraction_layer = feature_extraction_layer
        self.feature_extraction_pooling_strategy = feature_extraction_pooling_strategy


class TrainableHead(nn.Module):
    """A simple module wrapper for the trainable linear layer."""

    def __init__(self, feat_dim: int, dtype: torch.dtype):
        super().__init__()
        self.linear = nn.Linear(feat_dim, 1, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BayesianLinearHeadModel(RewardUQModel):
    """An uncertainty-aware reward model with a Bayesian linear head."""

    config_class = BayesianLinearHeadModelConfig

    def __init__(self, config: BayesianLinearHeadModelConfig):
        super().__init__(config)

        tokenizer, base_model_ = load_tokenizer_and_model(
            config.base_model_name_or_path,
            base_model_class=config.base_model_class,
            base_model_init_kwargs=config.base_model_init_kwargs,
        )

        self.tokenizer = tokenizer
        self.base_model_ = base_model_
        self._no_split_modules = getattr(base_model_, "_no_split_modules", None)

        # Freeze the backbone
        self.base_model_.requires_grad_(False)

        feat_dim = base_model_.config.hidden_size
        self.feat_dim = feat_dim
        self.lambda_reg = config.lambda_reg

        self.head = TrainableHead(feat_dim=feat_dim, dtype=base_model_.dtype)
        self.head.requires_grad_(True)

        # Dual-dtype system
        self.safe_dtype = torch.float32  # FP32 for Hessian operations (numerical stability)
        self.fast_dtype = base_model_.dtype  # User-specified dtype for forward pass (performance)

        # Register H as a NON-PERSISTENT buffer to avoid FSDP2 issues
        self.register_buffer(
            "H",
            self.lambda_reg * torch.eye(self.feat_dim, dtype=self.safe_dtype),
            persistent=False,
        )

        # Keep H_inv as a regular attribute (not a buffer) during training
        self._H_inv: torch.Tensor | None = None
        self._H_inv_computed = False

        # Check if we're loading a saved model with Hessian data
        # This will be set to True when loading from a checkpoint
        self._loaded_from_checkpoint = False

        log_model_summary(self)

    def std(self, f_fast: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation for uncertainty estimation.

        Args:
            f_fast: Features in fast dtype (bf16/fp16)

        Returns:
            Standard deviation in safe dtype (FP32)
        """
        if self._H_inv is None:
            return torch.zeros(f_fast.shape[0], device=f_fast.device, dtype=self.safe_dtype)

        # Convert to safe dtype for Hessian operations
        f = f_fast.view(-1, self.feat_dim).to(device=self.H.device, dtype=self.safe_dtype)

        if f.shape[0] > 0:
            # Move H_inv to the same device as f for DataParallel compatibility
            H_inv_device = self._H_inv.to(f.device)
            var = torch.diag(f @ H_inv_device @ f.T)
        else:
            var = torch.tensor([], device=f.device, dtype=self.safe_dtype)

        return torch.sqrt(var.clamp_min(1e-9)).to(dtype=self.safe_dtype)

    @torch.no_grad()
    def compute_and_set_final_hessian(
        self,
        all_z_vectors: torch.Tensor,
        mode: Literal["unweighted", "weighted"],
    ):
        """Computes the final Hessian from all z_vectors and sets it on the model.

        Args:
            all_z_vectors: Feature differences (delta_f), shape (num_samples, feat_dim)
            mode: How to compute the Hessian - "unweighted" (simple outer products) or
                  "weighted" (with sigmoid weighting)
        """
        final_theta = self.head.linear.weight.data

        logger.info(f"Computing final Hessian from {all_z_vectors.shape[0]} samples using mode: '{mode}'")

        # Move to safe dtype and same device as H buffer
        Z_all = all_z_vectors.to(device=self.H.device, dtype=self.safe_dtype)

        final_H = self.lambda_reg * torch.eye(self.feat_dim, device=self.H.device, dtype=self.safe_dtype)

        if mode == "unweighted":
            # Simple unweighted Hessian: H = lambda*I + Z^T @ Z
            final_H += Z_all.T @ Z_all
        elif mode == "weighted":
            # Weighted Hessian with sigmoid derivatives
            theta_flat = final_theta.squeeze().to(device=self.H.device, dtype=self.safe_dtype)
            logits = Z_all @ theta_flat
            probs = torch.sigmoid(logits)
            sigma_dot_weights = probs * (1 - probs)
            # Efficiently compute weighted sum of outer products
            final_H += torch.einsum("i,id,ie->de", sigma_dot_weights, Z_all, Z_all)
        else:
            raise ValueError(f"Unknown Hessian computation mode: {mode}")

        # Update H buffer
        self.H.copy_(final_H)

        # Compute and store the inverse as regular attribute
        self._H_inv = torch.linalg.pinv(final_H)
        self._H_inv_computed = True

        logger.info("Final Hessian and its inverse have been computed and stored.")

    def save_pretrained(self, save_directory: str, **kwargs):
        """Override to save Hessian matrices in the state dict (TP/FSDP-safe)."""

        incoming_sd = kwargs.get("state_dict", None)
        will_inject_into_sd = incoming_sd is not None

        had_h_inv = self._H_inv is not None
        if had_h_inv and not will_inject_into_sd:
            H_data = self.H.data.clone()
            delattr(self, "H")
            self.register_buffer("H", H_data, persistent=True)
            self.register_buffer("H_inv", self._H_inv, persistent=True)
            # Bool dtype matters for some safetensors stacks
            self.register_buffer(
                "H_inv_computed", torch.tensor(self._H_inv_computed, dtype=torch.bool), persistent=True
            )

        if will_inject_into_sd and had_h_inv:
            sd = dict(incoming_sd)  # Shallow copy
            sd["H"] = self.H.detach().cpu()
            sd["H_inv"] = self._H_inv.detach().cpu()
            sd["H_inv_computed"] = torch.tensor(self._H_inv_computed, dtype=torch.bool)  # Bool
            kwargs["state_dict"] = sd

        had_tp_attr = hasattr(self, "_tp_plan")
        prev_tp = getattr(self, "_tp_plan", None)
        if prev_tp is None:
            self._tp_plan = {}  # HF TP code expects iterable

        try:
            super().save_pretrained(save_directory, **kwargs)
        finally:
            if not had_tp_attr:
                delattr(self, "_tp_plan")
            else:
                self._tp_plan = prev_tp

            if not self._loaded_from_checkpoint and had_h_inv and not will_inject_into_sd:
                H_data = self.H.data.clone()
                H_inv_data = self.H_inv.data.clone() if hasattr(self, "H_inv") else None

                if hasattr(self, "H"):
                    delattr(self, "H")
                if hasattr(self, "H_inv"):
                    delattr(self, "H_inv")
                if hasattr(self, "H_inv_computed"):
                    delattr(self, "H_inv_computed")

                self.register_buffer("H", H_data, persistent=False)
                self._H_inv = H_inv_data

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # Try to manually load Hessian tensors from the checkpoint folder
        folder = pretrained_model_name_or_path
        try:
            H, H_inv, H_inv_computed = cls._load_hessian_from_folder(folder)
        except Exception as e:
            logger.warning(f"Failed to probe Hessian tensors in '{folder}': {e}")
            H = H_inv = H_inv_computed = None

        if H is not None:
            # Re-register H as non-persistent buffer (FSDP-friendly) in safe dtype
            if hasattr(model, "H"):
                delattr(model, "H")
            model.register_buffer("H", H.to(dtype=model.safe_dtype), persistent=False)

            # Keep inverse as attribute
            model._H_inv = H_inv.to(dtype=model.safe_dtype) if H_inv is not None else None
            model._H_inv_computed = bool(H_inv_computed.item()) if H_inv_computed is not None else True
            model._loaded_from_checkpoint = True
            logger.info(f"Loaded Hessian (H: {tuple(H.shape)}) and inverse from '{folder}'.")
        else:
            logger.info(f"No Hessian tensors found in '{folder}'. Uncertainty will be zero.")

        return model

    @staticmethod
    def _load_hessian_from_folder(folder: str):
        """
        Return (H, H_inv, H_inv_computed) as tensors if present in the folder, else (None, None, None).
        Works with single-file and sharded safetensors using the index.
        """
        idx_path = os.path.join(folder, "model.safetensors.index.json")
        files = []

        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
            weight_map = idx.get("weight_map", {})
            files = sorted({os.path.join(folder, f) for f in weight_map.values()})
        else:
            # Single-file or shards
            st = os.path.join(folder, "model.safetensors")
            if os.path.exists(st):
                files = [st]
            else:
                files = sorted(glob.glob(os.path.join(folder, "model-*.safetensors")))

        if not files:
            return None, None, None

        H = H_inv = H_inv_computed = None
        # Scan files until we find the keys
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as sf:
                keys = set(sf.keys())
                if H is None and "H" in keys:
                    H = sf.get_tensor("H").cpu()
                if H_inv is None and "H_inv" in keys:
                    H_inv = sf.get_tensor("H_inv").cpu()
                if H_inv_computed is None and "H_inv_computed" in keys:
                    H_inv_computed = sf.get_tensor("H_inv_computed").cpu()
            if H is not None and H_inv is not None and H_inv_computed is not None:
                break

        return H, H_inv, H_inv_computed

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        features: torch.FloatTensor | None = None,
        output_features: bool = False,
        output_only_features: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        result = {}

        if features is None:
            # if input_ids is not None and input_ids.device != self.base_model_.device:
            #     input_ids = input_ids.to(self.base_model_.device)
            # if attention_mask is not None and attention_mask.device != self.base_model_.device:
            #     attention_mask = attention_mask.to(self.base_model_.device)

            base_model_outputs = self.base_model_(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            features = extract_features(
                base_model_outputs,
                feature_layer=self.config.feature_extraction_layer,
                pooling_strategy=self.config.feature_extraction_pooling_strategy,
                attention_mask=attention_mask,
            )  # Shape: (batch_size, feature_dim)

        if output_only_features:
            return features
        if output_features:
            result["features"] = features

        head_weight = self.head.linear.weight
        if features.device != head_weight.device:
            features = features.to(device=head_weight.device, dtype=head_weight.dtype)
        else:
            features = features.to(dtype=head_weight.dtype)

        rewards_mean = self.head(features).squeeze(-1).float()
        rewards_std = self.std(features)
        rewards_lower = rewards_mean - self.config.std_beta * rewards_std
        rewards_upper = rewards_mean + self.config.std_beta * rewards_std
        result["rewards"] = torch.stack([rewards_mean, rewards_lower, rewards_upper], dim=1)  # Shape: (batch_size, 3)

        return result

    def get_feature_dependencies(self) -> dict[str]:
        return {
            "base_model_name_or_path": self.config.base_model_name_or_path,
            "pad_token_id": self.config.pad_token_id,
            "feature_layer": self.config.feature_extraction_layer,
            "pooling_strategy": self.config.feature_extraction_pooling_strategy,
        }
