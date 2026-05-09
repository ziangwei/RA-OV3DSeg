"""Model builders and lightweight MVP modules.

The package entrypoint uses lazy imports so metadata-only tools, such as teacher
backend checks and external teacher manifest generation, do not require PyTorch.
"""

from importlib import import_module

__all__ = [
    "DEBUG_BACKBONE",
    "SPARSE_UNET_BACKBONE",
    "SUPPORTED_BACKBONES",
    "build_segmentor",
    "describe_backbone",
    "CLIP_PATCH_BASELINE",
    "CLIPSEG_DENSE",
    "CATSEG_DENSE",
    "EXTERNAL_DENSE_LOGITS",
    "OPENSEG_DENSE",
    "GROUNDED_SAM_MASK",
    "SUPPORTED_TEACHERS",
    "build_image_teacher",
    "describe_teacher",
    "VoxelizationConfig",
    "voxelize_point_features",
]

_SEGMENTOR_EXPORTS = {
    "DEBUG_BACKBONE",
    "SPARSE_UNET_BACKBONE",
    "SUPPORTED_BACKBONES",
    "build_segmentor",
    "describe_backbone",
}
_TEACHER_EXPORTS = {
    "CLIP_PATCH_BASELINE",
    "CLIPSEG_DENSE",
    "CATSEG_DENSE",
    "EXTERNAL_DENSE_LOGITS",
    "OPENSEG_DENSE",
    "GROUNDED_SAM_MASK",
    "SUPPORTED_TEACHERS",
    "build_image_teacher",
    "describe_teacher",
}
_VOXELIZATION_EXPORTS = {"VoxelizationConfig", "voxelize_point_features"}


def __getattr__(name: str):
    if name in _SEGMENTOR_EXPORTS:
        segmentor_factory = import_module(f"{__name__}.segmentor_factory")
        return getattr(segmentor_factory, name)
    if name in _TEACHER_EXPORTS:
        teacher_registry = import_module(f"{__name__}.teacher_registry")
        return getattr(teacher_registry, name)
    if name in _VOXELIZATION_EXPORTS:
        voxelization = import_module(f"{__name__}.voxelization")
        return getattr(voxelization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
