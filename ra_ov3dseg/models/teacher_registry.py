from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CLIPSEG_DENSE = "clipseg_dense"
GROUPVIT_DENSE = "groupvit_dense"
SAM2_SIGLIP = "sam2_siglip"
SUPPORTED_TEACHERS = (
    CLIPSEG_DENSE,
    GROUPVIT_DENSE,
    SAM2_SIGLIP,
)


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    role: str
    feature_granularity: str
    is_baseline: bool
    description: str


def describe_teacher(teacher_backend: str) -> TeacherSpec:
    if teacher_backend == CLIPSEG_DENSE:
        return TeacherSpec(
            name=teacher_backend,
            role="runnable_dense_teacher",
            feature_granularity="dense_class_logits",
            is_baseline=True,
            description=(
                "Runnable dense open-vocabulary segmentation teacher based on CLIPSeg. "
                "It produces per-class dense logits and is stronger than CLIP patch tokens, "
                "but remains a practical baseline before the GroupViT teacher."
            ),
        )

    if teacher_backend == GROUPVIT_DENSE:
        return TeacherSpec(
            name=teacher_backend,
            role="transformers_native_dense_teacher",
            feature_granularity="dense_class_logits",
            is_baseline=False,
            description=(
                "Transformers-native GroupViT zero-shot semantic segmentation teacher. "
                "It runs inside the RA-OV3DSeg environment through Hugging Face Transformers."
            ),
        )

    if teacher_backend == SAM2_SIGLIP:
        return TeacherSpec(
            name=teacher_backend,
            role="mask_then_classify_dense_teacher",
            feature_granularity="dense_class_logits",
            is_baseline=False,
            description=(
                "SAM2 automatic masks classified by SigLIP text/image similarity. "
                "This is the Stage 3 teacher candidate before reliability filtering."
            ),
        )

    raise ValueError(f"Unknown teacher_backend={teacher_backend}. Supported: {SUPPORTED_TEACHERS}")


def build_image_teacher(
    teacher_backend: str,
    model_name: str,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
):
    """Build an image teacher for dense-logit extraction."""

    if teacher_backend == CLIPSEG_DENSE:
        raise NotImplementedError(
            "clipseg_dense is a dense-logit teacher. Use scripts/extract_dense_teacher_logits.py "
            "or the Stage 3 SAM2+SigLIP teacher path."
        )

    if teacher_backend == GROUPVIT_DENSE:
        raise NotImplementedError(
            "groupvit_dense is a dense-logit teacher. Use scripts/extract_dense_teacher_logits.py "
            "or the Stage 3 SAM2+SigLIP teacher path."
        )

    if teacher_backend == SAM2_SIGLIP:
        raise NotImplementedError("sam2_siglip is built directly by scripts/extract_sam2_teacher.py.")

    raise ValueError(f"Unknown teacher_backend={teacher_backend}. Supported: {SUPPORTED_TEACHERS}")
