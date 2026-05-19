from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_text_prototypes(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        class_names = [str(item) for item in data["class_names"].tolist()]
        prompts = [str(item) for item in data["prompts"].tolist()]
        embeddings = data["text_embeddings"].astype(np.float32)
        model_name = str(data["model_name"].item()) if "model_name" in data.files else ""
        prompt_template = str(data["prompt_template"].item()) if "prompt_template" in data.files else ""
    return {
        "path": str(path),
        "class_names": class_names,
        "prompts": prompts,
        "text_embeddings": embeddings,
        "model_name": model_name,
        "prompt_template": prompt_template,
    }


class TextPrototypeHead(nn.Module):
    """Cosine classifier over fixed text prototypes."""

    def __init__(
        self,
        input_dim: int,
        text_prototypes: torch.Tensor | np.ndarray,
        temperature: float = 0.07,
        trainable_temperature: bool = True,
        use_projection: bool = True,
    ) -> None:
        super().__init__()
        prototypes = torch.as_tensor(text_prototypes, dtype=torch.float32)
        if prototypes.ndim != 2:
            raise ValueError(f"text_prototypes must be rank-2, got shape={tuple(prototypes.shape)}")

        text_dim = int(prototypes.shape[1])
        self.input_dim = int(input_dim)
        self.text_dim = text_dim
        self.num_classes = int(prototypes.shape[0])
        if self.input_dim == self.text_dim:
            self.projection = nn.Identity()
        elif use_projection:
            self.projection = nn.Linear(self.input_dim, self.text_dim)
        else:
            raise ValueError(
                f"input_dim={self.input_dim} does not match text_dim={self.text_dim}; "
                "enable use_projection or choose matching dimensions."
            )

        self.register_buffer("text_prototypes", F.normalize(prototypes, dim=-1, eps=1e-6))
        initial_logit_scale = torch.log(torch.tensor(1.0 / max(float(temperature), 1e-6)))
        if trainable_temperature:
            self.logit_scale = nn.Parameter(initial_logit_scale)
        else:
            self.register_buffer("logit_scale", initial_logit_scale)

    def point_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(features), dim=-1, eps=1e-6)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        point_embeddings = self.point_embeddings(features)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * point_embeddings @ self.text_prototypes.t()


def _strip_prefix(key: str, prefix: str) -> str:
    return key[len(prefix) :] if key.startswith(prefix) else key


def load_pointcept_backbone_weights(backbone: nn.Module, checkpoint_path: str | Path) -> tuple[list[str], list[str]]:
    """Load Stage 1 Pointcept DefaultSegmentor backbone weights, excluding the closed-set head."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    normalized_keys = [_strip_prefix(key, "module.") for key in state_dict]
    has_backbone_prefix = any(key.startswith("backbone.") for key in normalized_keys)
    backbone_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        key = _strip_prefix(key, "module.")
        if has_backbone_prefix:
            if not key.startswith("backbone."):
                continue
            key = _strip_prefix(key, "backbone.")
        if key.startswith("final."):
            continue
        backbone_state[key] = value
    load_info = backbone.load_state_dict(backbone_state, strict=False)
    return list(load_info.missing_keys), list(load_info.unexpected_keys)


try:
    from pointcept.models.builder import MODELS, build_model
    from pointcept.models.losses import build_criteria
except Exception:  # pragma: no cover - Pointcept is available only on the training server.
    MODELS = None
    build_model = None
    build_criteria = None


class PointceptOVHeadSegmentor(nn.Module):
    """Pointcept-compatible segmentor with an RA-owned text prototype head."""

    def __init__(
        self,
        backbone: dict[str, Any],
        text_prototypes_path: str,
        backbone_out_channels: int = 96,
        criteria: Any = None,
        temperature: float = 0.07,
        trainable_temperature: bool = True,
        use_projection: bool = True,
        freeze_backbone: bool = True,
        force_fp32_backbone: bool = True,
        backbone_weight_path: str | None = None,
    ) -> None:
        super().__init__()
        if build_model is None or build_criteria is None:
            raise ImportError("PointceptOVHeadSegmentor requires Pointcept on PYTHONPATH.")

        backbone = dict(backbone)
        backbone["num_classes"] = 0
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = bool(freeze_backbone)
        self.force_fp32_backbone = bool(force_fp32_backbone)

        prototype_data = load_text_prototypes(text_prototypes_path)
        self.class_names = prototype_data["class_names"]
        self.prompts = prototype_data["prompts"]
        self.text_model_name = prototype_data["model_name"]
        self.text_prototypes_path = prototype_data["path"]
        self.ov_head = TextPrototypeHead(
            input_dim=backbone_out_channels,
            text_prototypes=prototype_data["text_embeddings"],
            temperature=temperature,
            trainable_temperature=trainable_temperature,
            use_projection=use_projection,
        )

        if backbone_weight_path:
            missing, unexpected = load_pointcept_backbone_weights(self.backbone, backbone_weight_path)
            allowed_missing = [key for key in missing if key.startswith("final.")]
            disallowed_missing = sorted(set(missing) - set(allowed_missing))
            if disallowed_missing or unexpected:
                raise RuntimeError(
                    "Stage 1 backbone load mismatch: "
                    f"missing={disallowed_missing}, unexpected={unexpected}"
                )

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _run_backbone(self, input_dict: dict[str, Any]) -> torch.Tensor:
        backbone_input = dict(input_dict)
        if self.force_fp32_backbone and "feat" in backbone_input:
            backbone_input["feat"] = backbone_input["feat"].float()

        if self.force_fp32_backbone and torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=False):
                return self.backbone(backbone_input).float()
        return self.backbone(backbone_input).float()

    def forward(self, input_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.freeze_backbone:
            with torch.no_grad():
                point_features = self._run_backbone(input_dict)
        else:
            point_features = self._run_backbone(input_dict)

        seg_logits = self.ov_head(point_features)
        return_dict: dict[str, torch.Tensor] = {
            "seg_logits": seg_logits,
            "point_embeddings": self.ov_head.point_embeddings(point_features),
        }
        if "segment" in input_dict:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        return return_dict


if MODELS is not None:
    MODELS.register_module("RAOVHeadSegmentor")(PointceptOVHeadSegmentor)
