from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_CAMERA_COUNT = 6


@dataclass(frozen=True)
class DenseTeacherValidation:
    path: str
    valid: bool
    message: str
    num_cameras: int
    num_classes: int
    logit_height: int
    logit_width: int
    layout: str
    teacher_backend: str
    model_name: str


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def scalar_to_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value.tolist())
    return str(value)


def array_to_str_list(value: np.ndarray) -> list[str]:
    return [str(item) for item in value.tolist()]


def infer_dense_logit_layout(dense_logits: np.ndarray, num_classes: int) -> str:
    if dense_logits.ndim != 4:
        raise ValueError(f"dense_logits must be 4D, got shape={dense_logits.shape}")
    if dense_logits.shape[1] == num_classes:
        return "NCHW"
    if dense_logits.shape[-1] == num_classes:
        return "NHWC"
    raise ValueError(
        "Cannot infer dense_logits class dimension. Expected either "
        f"(camera, class, height, width) or (camera, height, width, class), "
        f"got shape={dense_logits.shape}, num_classes={num_classes}."
    )


def dense_logits_to_nchw(dense_logits: np.ndarray, num_classes: int) -> tuple[np.ndarray, str]:
    layout = infer_dense_logit_layout(dense_logits, num_classes)
    if layout == "NCHW":
        return dense_logits, layout
    return np.transpose(dense_logits, (0, 3, 1, 2)), layout


def validate_dense_teacher_npz(
    path: str | Path,
    expected_camera_names: list[str] | None = None,
    expected_class_names: list[str] | None = None,
) -> DenseTeacherValidation:
    path = Path(path)
    try:
        data = load_npz(path)
        required = [
            "sample_idx",
            "sample_token",
            "teacher_backend",
            "model_name",
            "camera_names",
            "camera_available",
            "image_widths",
            "image_heights",
            "class_names",
            "prompts",
            "dense_logits",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"missing_keys={missing}")

        camera_names = array_to_str_list(data["camera_names"])
        class_names = array_to_str_list(data["class_names"])
        dense_logits = data["dense_logits"]
        layout = infer_dense_logit_layout(dense_logits, len(class_names))
        dense_logits_nchw, _ = dense_logits_to_nchw(dense_logits, len(class_names))
        num_cameras, num_classes, logit_height, logit_width = dense_logits_nchw.shape

        if num_cameras != len(camera_names):
            raise ValueError(f"camera count mismatch: logits={num_cameras}, camera_names={len(camera_names)}")
        if num_cameras != CANONICAL_CAMERA_COUNT:
            raise ValueError(f"expected {CANONICAL_CAMERA_COUNT} cameras, got {num_cameras}")
        if data["camera_available"].shape[0] != num_cameras:
            raise ValueError("camera_available length does not match camera count")
        if data["image_widths"].shape[0] != num_cameras or data["image_heights"].shape[0] != num_cameras:
            raise ValueError("image_widths/image_heights length does not match camera count")
        if expected_camera_names is not None and camera_names != expected_camera_names:
            raise ValueError(f"camera_names mismatch: got={camera_names}, expected={expected_camera_names}")
        if expected_class_names is not None:
            if class_names[: len(expected_class_names)] != expected_class_names:
                raise ValueError("class_names must start with the expected lidarseg class order")
        if not np.isfinite(dense_logits_nchw).all():
            raise ValueError("dense_logits contains nan or inf")

        teacher_backend = scalar_to_str(data.get("teacher_backend"), default="external_dense_logits")
        model_name = scalar_to_str(data.get("model_name"), default="")
        return DenseTeacherValidation(
            path=str(path),
            valid=True,
            message="ok",
            num_cameras=num_cameras,
            num_classes=num_classes,
            logit_height=int(logit_height),
            logit_width=int(logit_width),
            layout=layout,
            teacher_backend=teacher_backend,
            model_name=model_name,
        )
    except Exception as exc:
        return DenseTeacherValidation(
            path=str(path),
            valid=False,
            message=str(exc),
            num_cameras=0,
            num_classes=0,
            logit_height=0,
            logit_width=0,
            layout="unknown",
            teacher_backend="",
            model_name="",
        )
