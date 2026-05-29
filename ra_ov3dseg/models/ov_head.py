from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ra_ov3dseg.training.losses import dense_logit_distillation_loss


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


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _strip_module_prefix(key: str) -> str:
    while key.startswith("module."):
        key = _strip_prefix(key, "module.")
    return key


def _tensor_shape_matches(target: torch.Tensor, value: Any) -> bool:
    return torch.is_tensor(value) and tuple(value.shape) == tuple(target.shape)


def _select_matching_state(
    module: nn.Module,
    state_dict: dict[str, Any],
    candidate_prefixes: tuple[str, ...],
) -> OrderedDict[str, torch.Tensor]:
    target_state = module.state_dict()
    best: OrderedDict[str, torch.Tensor] = OrderedDict()
    for prefix in candidate_prefixes:
        selected: OrderedDict[str, torch.Tensor] = OrderedDict()
        for raw_key, value in state_dict.items():
            key = _strip_module_prefix(str(raw_key))
            if prefix:
                if not key.startswith(prefix):
                    continue
                key = _strip_prefix(key, prefix)
            if key in target_state and _tensor_shape_matches(target_state[key], value):
                selected[key] = value
        if len(selected) > len(best):
            best = selected
    return best


def load_pointcept_backbone_weights(
    backbone: nn.Module, checkpoint_path: str | Path
) -> tuple[list[str], list[str], int]:
    """Load Stage 1 Pointcept DefaultSegmentor backbone weights, excluding the closed-set head."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    backbone_state = _select_matching_state(
        backbone,
        state_dict,
        candidate_prefixes=(
            "backbone.",
            "model.backbone.",
            "segmentor.backbone.",
            "",
        ),
    )
    load_info = backbone.load_state_dict(backbone_state, strict=False)
    return list(load_info.missing_keys), list(load_info.unexpected_keys), len(backbone_state)


def load_segmentor_weights(model: nn.Module, checkpoint_path: str | Path) -> tuple[list[str], list[str], int]:
    """Load a full RAOVHeadSegmentor checkpoint, including the text head."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    normalized_state = _select_matching_state(
        model,
        state_dict,
        candidate_prefixes=(
            "",
            "model.",
            "segmentor.",
        ),
    )
    load_info = model.load_state_dict(normalized_state, strict=False)
    return list(load_info.missing_keys), list(load_info.unexpected_keys), len(normalized_state)


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
        model_weight_path: str | None = None,
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
            missing, unexpected, loaded = load_pointcept_backbone_weights(self.backbone, backbone_weight_path)
            if loaded == 0 or unexpected:
                raise RuntimeError(
                    "Stage 1 backbone load mismatch: "
                    f"loaded={loaded}, missing={missing}, unexpected={unexpected}"
                )
            if missing:
                print(
                    "[PointceptOVHeadSegmentor] "
                    f"Stage 1 backbone partially loaded: loaded={loaded}, missing={len(missing)}"
                )

        if model_weight_path:
            missing, unexpected, loaded = load_segmentor_weights(self, model_weight_path)
            allowed_missing_prefixes = ("backbone.",) if backbone_weight_path else ()
            allowed_missing = [
                key for key in missing if any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
            ]
            allowed_missing.append("ov_head.text_prototypes")
            disallowed_missing = sorted(set(missing) - set(allowed_missing))
            if loaded == 0 or disallowed_missing or unexpected:
                raise RuntimeError(
                    "Stage 2 OV head checkpoint load mismatch: "
                    f"loaded={loaded}, missing={disallowed_missing}, unexpected={unexpected}"
                )
            if missing:
                print(
                    "[PointceptOVHeadSegmentor] "
                    f"Stage 2 OV checkpoint partially loaded: loaded={loaded}, missing={len(missing)}"
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
        seg_logits = self.forward_logits(input_dict)
        return_dict: dict[str, torch.Tensor] = {"seg_logits": seg_logits}
        if "segment" in input_dict:
            loss = self.criteria(seg_logits, input_dict["segment"])
            if self.training:
                return {"loss": loss}
            return_dict["loss"] = loss
        return return_dict

    def forward_logits(self, input_dict: dict[str, Any]) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                point_features = self._run_backbone(input_dict)
        else:
            point_features = self._run_backbone(input_dict)
        return self.ov_head(point_features)


class PointceptOVReliabilitySegmentor(PointceptOVHeadSegmentor):
    """Pointcept-compatible OV segmentor with reliability-weighted teacher KL."""

    def __init__(
        self,
        *args: Any,
        ce_loss_weight: float = 1.0,
        distill_loss_weight: float = 1.0,
        distill_temperature: float = 2.0,
        reliability_threshold: float = 0.5,
        require_teacher: bool = True,
        teacher_logits_key: str = "teacher_logits",
        teacher_valid_mask_key: str = "teacher_valid_mask",
        reliability_weight_key: str = "reliability_weight",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ce_loss_weight = float(ce_loss_weight)
        self.distill_loss_weight = float(distill_loss_weight)
        self.distill_temperature = float(distill_temperature)
        self.reliability_threshold = float(reliability_threshold)
        self.require_teacher = bool(require_teacher)
        self.teacher_logits_key = teacher_logits_key
        self.teacher_valid_mask_key = teacher_valid_mask_key
        self.reliability_weight_key = reliability_weight_key
        self._logged_distill_stats = False

    def _has_teacher(self, input_dict: dict[str, Any]) -> bool:
        return self.teacher_logits_key in input_dict and self.reliability_weight_key in input_dict

    @staticmethod
    def _as_tensor(value: Any, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=device)
        return torch.as_tensor(value, device=device)

    def _distillation_loss(
        self,
        seg_logits: torch.Tensor,
        input_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self._has_teacher(input_dict):
            if self.require_teacher:
                raise KeyError(
                    "Reliability distillation requires teacher fields in the Pointcept batch: "
                    f"{self.teacher_logits_key}, {self.reliability_weight_key}. "
                    "Check the RALoadReliabilityTeacher transform and Collect keys."
                )
            return seg_logits.sum() * 0.0, {"valid_ratio": 0.0, "mean_weight": 0.0}

        device = seg_logits.device
        teacher_logits = self._as_tensor(input_dict[self.teacher_logits_key], device=device).float()
        weights = self._as_tensor(input_dict[self.reliability_weight_key], device=device).float()
        if self.teacher_valid_mask_key in input_dict:
            valid_mask = self._as_tensor(input_dict[self.teacher_valid_mask_key], device=device).bool()
        else:
            valid_mask = torch.ones(weights.shape, dtype=torch.bool, device=device)

        if teacher_logits.ndim != 2:
            raise ValueError(f"teacher_logits must be rank-2, got shape={tuple(teacher_logits.shape)}")
        if teacher_logits.shape[0] != seg_logits.shape[0]:
            raise ValueError(
                f"teacher/student point count mismatch: teacher={teacher_logits.shape[0]}, student={seg_logits.shape[0]}"
            )
        if teacher_logits.shape[1] > seg_logits.shape[1]:
            teacher_logits = teacher_logits[:, : seg_logits.shape[1]]
        elif teacher_logits.shape[1] < seg_logits.shape[1]:
            raise ValueError(
                f"teacher class count {teacher_logits.shape[1]} is smaller than student class count {seg_logits.shape[1]}"
            )

        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        threshold_mask = weights >= self.reliability_threshold
        valid_mask = valid_mask & threshold_mask
        thresholded_weights = torch.where(threshold_mask, weights, torch.zeros_like(weights))
        loss = dense_logit_distillation_loss(
            student_logits=seg_logits,
            teacher_logits=teacher_logits,
            weights=thresholded_weights,
            valid_mask=valid_mask,
            temperature=self.distill_temperature,
        )
        valid_points = valid_mask.sum().item()
        total_points = max(int(valid_mask.numel()), 1)
        mean_weight = (
            float(thresholded_weights[valid_mask].mean().detach().cpu().item()) if valid_points else 0.0
        )
        return loss, {
            "valid_ratio": float(valid_points / total_points),
            "mean_weight": mean_weight,
        }

    def forward(self, input_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        seg_logits = self.forward_logits(input_dict)
        return_dict: dict[str, torch.Tensor] = {"seg_logits": seg_logits}
        if "segment" not in input_dict:
            return return_dict

        ce_loss = self.criteria(seg_logits, input_dict["segment"])
        if not self.training:
            return_dict["loss"] = ce_loss
            return return_dict

        distill_loss, stats = self._distillation_loss(seg_logits, input_dict)
        loss = self.ce_loss_weight * ce_loss + self.distill_loss_weight * distill_loss
        if not self._logged_distill_stats:
            print(
                "[RAOVReliabilitySegmentor] "
                f"threshold={self.reliability_threshold:.3f} "
                f"distill_valid_ratio={stats['valid_ratio']:.4f} "
                f"distill_mean_weight={stats['mean_weight']:.4f} "
                f"ce_weight={self.ce_loss_weight:.3f} "
                f"distill_weight={self.distill_loss_weight:.3f}"
            )
            self._logged_distill_stats = True
        return {"loss": loss}


if MODELS is not None:
    MODELS.register_module("RAOVHeadSegmentor")(PointceptOVHeadSegmentor)
    MODELS.register_module("RAOVReliabilitySegmentor")(PointceptOVReliabilitySegmentor)
