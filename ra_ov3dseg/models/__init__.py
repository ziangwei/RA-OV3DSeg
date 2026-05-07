"""Model builders and lightweight MVP modules."""

from .segmentor_factory import (
    DEBUG_BACKBONE,
    SPARSE_UNET_BACKBONE,
    SUPPORTED_BACKBONES,
    build_segmentor,
    describe_backbone,
)
from .voxelization import VoxelizationConfig, voxelize_point_features
from .teacher_registry import (
    CLIPSEG_DENSE,
    CLIP_PATCH_BASELINE,
    GROUNDED_SAM_MASK,
    OPENSEG_DENSE,
    SUPPORTED_TEACHERS,
    build_image_teacher,
    describe_teacher,
)

__all__ = [
    "DEBUG_BACKBONE",
    "SPARSE_UNET_BACKBONE",
    "SUPPORTED_BACKBONES",
    "build_segmentor",
    "describe_backbone",
    "CLIP_PATCH_BASELINE",
    "CLIPSEG_DENSE",
    "OPENSEG_DENSE",
    "GROUNDED_SAM_MASK",
    "SUPPORTED_TEACHERS",
    "build_image_teacher",
    "describe_teacher",
    "VoxelizationConfig",
    "voxelize_point_features",
]
