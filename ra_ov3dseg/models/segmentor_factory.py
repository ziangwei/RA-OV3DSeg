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
            role="sparse_3d_student",
            is_debug_model=False,
            description=(
                "MVP-v5 sparse-convolution 3D student using spconv voxelization, "
                "SparseUNet-Lite context aggregation, and point-wise feature gather."
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
        from ra_ov3dseg.models.sparse_unet_spconv import SparseUNetSpConv

        return SparseUNetSpConv(
            feature_dim=feature_dim,
            num_classes=num_classes,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            base_channels=sparse_base_channels,
        )

    raise ValueError(f"Unknown backbone={backbone}. Supported backbones: {SUPPORTED_BACKBONES}")
