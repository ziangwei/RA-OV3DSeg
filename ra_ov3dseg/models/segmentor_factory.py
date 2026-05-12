from __future__ import annotations

from dataclasses import dataclass


POINTCEPT_SPUNET_BACKBONE = "pointcept_spunet"
SPCONV_BACKBONES = (POINTCEPT_SPUNET_BACKBONE,)
SUPPORTED_BACKBONES = (POINTCEPT_SPUNET_BACKBONE,)


@dataclass(frozen=True)
class SegmentorSpec:
    backbone: str
    role: str
    is_debug_model: bool
    description: str


def describe_backbone(backbone: str) -> SegmentorSpec:
    if backbone == POINTCEPT_SPUNET_BACKBONE:
        return SegmentorSpec(
            backbone=backbone,
            role="pip_installed_pointcept_sparse_unet",
            is_debug_model=False,
            description=(
                "Pointcept SpUNet v1m1 from the pip-installed editable Pointcept package. "
                "Stage 1 trains it through Pointcept's own launcher and recipe."
            ),
        )
    raise ValueError(f"Unknown backbone={backbone}. Supported backbones: {SUPPORTED_BACKBONES}")


def build_segmentor(
    backbone: str,
    input_dim: int,
    hidden_dim: int,
    feature_dim: int,
    num_classes: int,
    voxel_size: tuple[float, float, float] = (0.2, 0.2, 0.2),
    point_cloud_range: tuple[float, float, float, float, float, float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
    sparse_base_channels: int = 32,
):
    """Build RA-owned segmentors after Stage 2 integration."""

    if backbone == POINTCEPT_SPUNET_BACKBONE:
        raise NotImplementedError(
            "Pointcept SpUNet is trained through Pointcept's own launcher in Stage 1. "
            "The RA-OV3DSeg wrapper is introduced later for the OV head."
        )

    raise ValueError(f"Unknown backbone={backbone}. Supported backbones: {SUPPORTED_BACKBONES}")
