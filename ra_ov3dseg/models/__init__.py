"""Model builders and lightweight MVP modules.

The package entrypoint uses lazy imports so metadata-only tools, such as teacher
backend checks, do not require PyTorch.
"""

from importlib import import_module

__all__ = [
    "POINTCEPT_SPUNET_BACKBONE",
    "SPCONV_BACKBONES",
    "SUPPORTED_BACKBONES",
    "build_segmentor",
    "describe_backbone",
    "CLIPSEG_DENSE",
    "GROUPVIT_DENSE",
    "SAM2_SIGLIP",
    "SUPPORTED_TEACHERS",
    "build_image_teacher",
    "describe_teacher",
    "SAM2SigLIPTeacher",
    "PointceptOVHeadSegmentor",
    "TextPrototypeHead",
    "load_pointcept_backbone_weights",
    "load_text_prototypes",
]

_SEGMENTOR_EXPORTS = {
    "POINTCEPT_SPUNET_BACKBONE",
    "SPCONV_BACKBONES",
    "SUPPORTED_BACKBONES",
    "build_segmentor",
    "describe_backbone",
}
_TEACHER_EXPORTS = {
    "CLIPSEG_DENSE",
    "GROUPVIT_DENSE",
    "SAM2_SIGLIP",
    "SUPPORTED_TEACHERS",
    "build_image_teacher",
    "describe_teacher",
}
_SAM2_TEACHER_EXPORTS = {"SAM2SigLIPTeacher"}
_OV_HEAD_EXPORTS = {
    "PointceptOVHeadSegmentor",
    "TextPrototypeHead",
    "load_pointcept_backbone_weights",
    "load_text_prototypes",
}


def __getattr__(name: str):
    if name in _SEGMENTOR_EXPORTS:
        segmentor_factory = import_module(f"{__name__}.segmentor_factory")
        return getattr(segmentor_factory, name)
    if name in _TEACHER_EXPORTS:
        teacher_registry = import_module(f"{__name__}.teacher_registry")
        return getattr(teacher_registry, name)
    if name in _SAM2_TEACHER_EXPORTS:
        sam2_siglip_teacher = import_module(f"{__name__}.sam2_siglip_teacher")
        return getattr(sam2_siglip_teacher, name)
    if name in _OV_HEAD_EXPORTS:
        ov_head = import_module(f"{__name__}.ov_head")
        return getattr(ov_head, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
