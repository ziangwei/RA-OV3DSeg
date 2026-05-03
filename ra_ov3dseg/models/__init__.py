"""Model builders and lightweight MVP modules."""

from .segmentor_factory import (
    DEBUG_BACKBONE,
    SPARSE_UNET_BACKBONE,
    SUPPORTED_BACKBONES,
    build_segmentor,
    describe_backbone,
)

__all__ = [
    "DEBUG_BACKBONE",
    "SPARSE_UNET_BACKBONE",
    "SUPPORTED_BACKBONES",
    "build_segmentor",
    "describe_backbone",
]
