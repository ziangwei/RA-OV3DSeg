from __future__ import annotations

from dataclasses import dataclass

from ra_ov3dseg.models.point_mlp import PointMLP


DEBUG_BACKBONE = "debug_point_mlp"
SPARSE_UNET_BACKBONE = "sparse_unet_spconv"
SUPPORTED_BACKBONES = (DEBUG_BACKBONE, SPARSE_UNET_BACKBONE)


@dataclass(frozen=True)
class SegmentorSpec:
    backbone: str
    role: str
    is_debug_model: bool
    description: str


def describe_backbone(backbone: str) -> SegmentorSpec:
    if backbone == DEBUG_BACKBONE:
        return SegmentorSpec(
            backbone=backbone,
            role="debug_harness",
            is_debug_model=True,
            description=(
                "Minimal point-wise MLP used only to verify training IO, DDP, "
                "base/novel label masking, and reliability-weighted distillation."
            ),
        )
    if backbone == SPARSE_UNET_BACKBONE:
        return SegmentorSpec(
            backbone=backbone,
            role="production_backbone_interface",
            is_debug_model=False,
            description=(
                "Planned sparse-convolution 3D segmentation backbone interface. "
                "Implementation will be added in the next stage after selecting "
                "spconv/MinkowskiEngine dependency strategy."
            ),
        )
    raise ValueError(f"Unknown backbone={backbone}. Supported backbones: {SUPPORTED_BACKBONES}")


def build_segmentor(
    backbone: str,
    input_dim: int,
    hidden_dim: int,
    feature_dim: int,
    num_classes: int,
):
    """Build the 3D segmentation model used by the generic trainer.

    The current runnable implementation is intentionally named `debug_point_mlp`.
    That prevents the MVP harness from being mistaken for the final 3D model.
    """

    if backbone == DEBUG_BACKBONE:
        return PointMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            num_classes=num_classes,
        )

    if backbone == SPARSE_UNET_BACKBONE:
        raise NotImplementedError(
            "sparse_unet_spconv is the intended production backbone interface, "
            "but it is not implemented yet. Use --backbone debug_point_mlp for MVP-v4 "
            "smoke tests, or implement the spconv adapter in V5."
        )

    raise ValueError(f"Unknown backbone={backbone}. Supported backbones: {SUPPORTED_BACKBONES}")
