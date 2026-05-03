"""Model builders and lightweight MVP modules."""

from .segmentor_factory import (
    DEBUG_BACKBONE,
    SPARSE_UNET_BACKBONE,
    SUPPORTED_BACKBONES,
    build_segmentor,
    describe_backbone,
)
from .teacher_registry import (
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
    "OPENSEG_DENSE",
    "GROUNDED_SAM_MASK",
    "SUPPORTED_TEACHERS",
    "build_image_teacher",
    "describe_teacher",
]
