from __future__ import annotations

from dataclasses import dataclass


DEBUG_BACKBONE = "debug_point_mlp"
SPARSE_UNET_BACKBONE = "sparse_unet_spconv"
SPCONV_RESUNET_BACKBONE = "spconv_resunet"
CYLINDER_SPUNET_BACKBONE = "cylinder_spconv_unet"
POINTCEPT_SPUNET_BACKBONE = "pointcept_spunet"
SPCONV_BACKBONES = (
    SPARSE_UNET_BACKBONE,
    SPCONV_RESUNET_BACKBONE,
    CYLINDER_SPUNET_BACKBONE,
    POINTCEPT_SPUNET_BACKBONE,
)
SUPPORTED_BACKBONES = (
    DEBUG_BACKBONE,
    SPARSE_UNET_BACKBONE,
    SPCONV_RESUNET_BACKBONE,
    CYLINDER_SPUNET_BACKBONE,
    POINTCEPT_SPUNET_BACKBONE,
)


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
    if backbone == SPCONV_RESUNET_BACKBONE:
        return SegmentorSpec(
            backbone=backbone,
            role="high_capacity_sparse_3d_student",
            is_debug_model=False,
            description=(
                "Higher-capacity in-repository spconv ResUNet with three sparse down/up stages "
                "and residual submanifold blocks. Use it for supervised upper-bound and serious "
                "student-capacity checks before drawing conclusions about open-vocabulary distillation."
            ),
        )
    if backbone == CYLINDER_SPUNET_BACKBONE:
        return SegmentorSpec(
            backbone=backbone,
            role="cylinder_sparse_3d_student",
            is_debug_model=False,
            description=(
                "Cylinder3D-style spconv U-Net using cylindrical LiDAR partitioning, "
                "asymmetric sparse residual blocks, and LiDAR intensity. This is the "
                "preferred supervised baseline backbone for V15."
            ),
        )
    if backbone == POINTCEPT_SPUNET_BACKBONE:
        return SegmentorSpec(
            backbone=backbone,
            role="vendored_pointcept_sparse_unet",
            is_debug_model=False,
            description=(
                "Vendored Pointcept SpConv SparseUNet v1m1 adapted through a thin RA-OV3DSeg "
                "adapter. This is the V17 mature supervised backbone path."
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

    The trainer stays stable while backbones are swapped through this factory.
    This keeps architecture changes isolated from data, loss, and eval code.
    """

    if backbone == DEBUG_BACKBONE:
        from ra_ov3dseg.models.point_mlp import PointMLP

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

    if backbone == SPCONV_RESUNET_BACKBONE:
        from ra_ov3dseg.models.spconv_resunet import SpConvResUNet

        return SpConvResUNet(
            feature_dim=feature_dim,
            num_classes=num_classes,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            base_channels=sparse_base_channels,
        )

    if backbone == CYLINDER_SPUNET_BACKBONE:
        from ra_ov3dseg.models.cylinder_spconv_unet import CylinderSpConvUNet

        return CylinderSpConvUNet(
            feature_dim=feature_dim,
            num_classes=num_classes,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            base_channels=sparse_base_channels,
        )

    if backbone == POINTCEPT_SPUNET_BACKBONE:
        from ra_ov3dseg.models.pointcept_spunet_adapter import PointceptSpUNetAdapter

        return PointceptSpUNetAdapter(
            feature_dim=feature_dim,
            num_classes=num_classes,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            base_channels=sparse_base_channels,
        )

    raise ValueError(f"Unknown backbone={backbone}. Supported backbones: {SUPPORTED_BACKBONES}")
