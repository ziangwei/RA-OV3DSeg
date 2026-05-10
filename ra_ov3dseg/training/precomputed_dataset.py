from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset
from ra_ov3dseg.training.augmentations import PointAugmentationConfig, augment_point_xyz
from ra_ov3dseg.training.labels import ClassSplit, map_labels_for_base_ce


IGNORE_INDEX = -100


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def default_point_feature_path(point_feature_dir: str | Path, sample_idx: int) -> Path:
    return Path(point_feature_dir).expanduser().resolve() / f"sample_{sample_idx:04d}_point_features.npz"


def default_reliability_path(reliability_dir: str | Path, sample_idx: int) -> Path:
    return Path(reliability_dir).expanduser().resolve() / f"sample_{sample_idx:04d}_reliability.npz"


def default_dense_point_path(dense_point_dir: str | Path, sample_idx: int) -> Path:
    return Path(dense_point_dir).expanduser().resolve() / f"sample_{sample_idx:04d}_dense_point_logits.npz"


def find_missing_precomputed_files(
    sample_indices: list[int],
    point_feature_dir: str | Path,
    reliability_dir: str | Path,
) -> list[dict[str, str | int]]:
    point_feature_dir = Path(point_feature_dir).expanduser().resolve()
    reliability_dir = Path(reliability_dir).expanduser().resolve()
    missing: list[dict[str, str | int]] = []
    for sample_idx in sample_indices:
        point_feature_path = default_point_feature_path(point_feature_dir, sample_idx)
        reliability_path = default_reliability_path(reliability_dir, sample_idx)
        missing_files = []
        if not point_feature_path.exists():
            missing_files.append(str(point_feature_path))
        if not reliability_path.exists():
            missing_files.append(str(reliability_path))
        if missing_files:
            missing.append(
                {
                    "sample_idx": int(sample_idx),
                    "missing_files": "\n".join(missing_files),
                }
            )
    return missing


def find_missing_dense_point_files(
    sample_indices: list[int],
    dense_point_dir: str | Path,
) -> list[dict[str, str | int]]:
    dense_point_dir = Path(dense_point_dir).expanduser().resolve()
    missing: list[dict[str, str | int]] = []
    for sample_idx in sample_indices:
        dense_point_path = default_dense_point_path(dense_point_dir, sample_idx)
        if not dense_point_path.exists():
            missing.append(
                {
                    "sample_idx": int(sample_idx),
                    "missing_files": str(dense_point_path),
                }
            )
    return missing


def subsample_indices(num_points: int, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or max_points <= 0 or num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=max_points, replace=False)).astype(np.int64)


def label_hist(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for label_id, count in zip(*np.unique(labels, return_counts=True)):
        label_id_int = int(label_id)
        name = class_names[label_id_int] if 0 <= label_id_int < len(class_names) else f"unknown_{label_id_int}"
        hist[name] = int(count)
    return hist


class PrecomputedPointFeatureDataset:
    """Training dataset backed by MVP-v1/v2 precomputed outputs.

    The current MVP does not run the 2D teacher online during training. Each sample
    must already have:
    - outputs/point_features/sample_XXXX_point_features.npz
    - outputs/reliability/sample_XXXX_reliability.npz
    - nuScenes lidarseg labels under dataroot
    """

    def __init__(
        self,
        nuscenes_dataset: NuScenesDataset,
        sample_indices: list[int],
        point_feature_dir: str | Path,
        reliability_dir: str | Path,
        class_split: ClassSplit,
        dense_point_dir: str | Path | None = None,
        load_dense_logits: bool = False,
        dense_logit_space: str = "base",
        max_points: int | None = None,
        seed: int = 0,
        ignore_index: int = IGNORE_INDEX,
        augment: bool = False,
        augmentation_config: PointAugmentationConfig | None = None,
    ) -> None:
        if not sample_indices:
            raise ValueError("sample_indices must not be empty.")

        self.nuscenes_dataset = nuscenes_dataset
        self.sample_indices = list(sample_indices)
        self.point_feature_dir = Path(point_feature_dir).expanduser().resolve()
        self.reliability_dir = Path(reliability_dir).expanduser().resolve()
        self.dense_point_dir = Path(dense_point_dir).expanduser().resolve() if dense_point_dir is not None else None
        self.load_dense_logits = load_dense_logits
        self.dense_logit_space = dense_logit_space
        self.class_split = class_split
        self.max_points = max_points
        self.seed = seed
        self.ignore_index = ignore_index
        self.augment = bool(augment)
        self.augmentation_config = augmentation_config or PointAugmentationConfig()
        self.epoch = 0
        if self.load_dense_logits and self.dense_point_dir is None:
            raise ValueError("dense_point_dir is required when load_dense_logits=True.")
        if self.dense_logit_space not in {"base", "all_lidarseg"}:
            raise ValueError("dense_logit_space must be one of: base, all_lidarseg.")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_idx = self.sample_indices[index]
        sample = self.nuscenes_dataset.get_sample_by_index(sample_idx)
        raw_labels = self.nuscenes_dataset.load_lidarseg_labels(sample)
        if raw_labels is None:
            raise FileNotFoundError(
                "lidarseg labels not found; training requires lidarseg labels for base-class CE loss."
            )

        point_feature_path = default_point_feature_path(self.point_feature_dir, sample_idx)
        reliability_path = default_reliability_path(self.reliability_dir, sample_idx)
        if not point_feature_path.exists():
            raise FileNotFoundError(f"point feature npz not found: {point_feature_path}")
        if not reliability_path.exists():
            raise FileNotFoundError(f"reliability npz not found: {reliability_path}")

        point_data = load_npz(point_feature_path)
        reliability_data = load_npz(reliability_path)

        point_xyz = point_data["point_xyz"].astype(np.float32)
        teacher_features = point_data["point_features"].astype(np.float32)
        teacher_valid_mask = point_data["point_valid_mask"].astype(bool)
        reliability_weight = reliability_data["reliability_weight"].astype(np.float32)
        dense_logit_dim = (
            self.class_split.num_classes if self.dense_logit_space == "all_lidarseg" else self.class_split.num_train_classes
        )
        dense_teacher_logits = np.zeros((point_xyz.shape[0], dense_logit_dim), dtype=np.float32)
        dense_teacher_valid_mask = np.zeros(point_xyz.shape[0], dtype=bool)
        dense_teacher_confidence = np.zeros(point_xyz.shape[0], dtype=np.float32)
        dense_point_path: Path | None = None

        if raw_labels.shape[0] != point_xyz.shape[0]:
            raise ValueError(
                "label/point count mismatch. This usually means stale precomputed outputs "
                "from a different nuScenes version or dataroot were reused. "
                f"sample_idx={sample_idx}, labels={raw_labels.shape[0]}, points={point_xyz.shape[0]}, "
                f"point_feature_path={point_feature_path}, reliability_path={reliability_path}"
            )
        if teacher_features.shape[0] != point_xyz.shape[0]:
            raise ValueError("teacher feature count does not match point count.")
        if teacher_valid_mask.shape[0] != point_xyz.shape[0]:
            raise ValueError("teacher valid mask count does not match point count.")
        if reliability_weight.shape[0] != point_xyz.shape[0]:
            raise ValueError("reliability count does not match point count.")

        if self.load_dense_logits:
            dense_point_path = default_dense_point_path(self.dense_point_dir, sample_idx)
            if not dense_point_path.exists():
                raise FileNotFoundError(f"dense point logits npz not found: {dense_point_path}")
            dense_data = load_npz(dense_point_path)
            point_teacher_logits = dense_data["point_teacher_logits"].astype(np.float32)
            point_dense_valid_mask = dense_data["point_dense_valid_mask"].astype(bool)
            if "point_xyz" in dense_data and not np.allclose(dense_data["point_xyz"].astype(np.float32), point_xyz):
                raise ValueError("dense point logits point_xyz does not match point feature point_xyz.")
            if "class_names" in dense_data:
                dense_class_names = [str(name) for name in dense_data["class_names"].tolist()]
                if dense_class_names[: self.class_split.num_classes] != self.class_split.class_names:
                    raise ValueError("dense point logits class_names order does not match lidarseg class_names.")
            if point_teacher_logits.shape[0] != point_xyz.shape[0]:
                raise ValueError("dense teacher logit count does not match point count.")
            if point_dense_valid_mask.shape[0] != point_xyz.shape[0]:
                raise ValueError("dense teacher valid mask count does not match point count.")
            if point_teacher_logits.shape[1] < self.class_split.num_classes:
                raise ValueError(
                    "dense teacher logits must contain all lidarseg classes before base-class selection: "
                    f"logits={point_teacher_logits.shape[1]}, expected>={self.class_split.num_classes}"
                )

            if self.dense_logit_space == "all_lidarseg":
                # The student emits the full lidarseg label space. CE still masks
                # ignored labels, while dense KL can keep teacher signal for all classes.
                dense_teacher_logits = point_teacher_logits[:, : self.class_split.num_classes].astype(np.float32)
            else:
                # Base mode keeps only supervised base classes so student logits
                # stay aligned with base-class CE train ids.
                dense_teacher_logits = point_teacher_logits[:, self.class_split.base_label_ids].astype(np.float32)
            dense_teacher_valid_mask = point_dense_valid_mask
            if "point_dense_pred_scores" in dense_data:
                dense_teacher_confidence = dense_data["point_dense_pred_scores"].astype(np.float32)
            else:
                shifted = point_teacher_logits - np.max(point_teacher_logits, axis=1, keepdims=True)
                probs = np.exp(shifted) / np.maximum(np.exp(shifted).sum(axis=1, keepdims=True), 1e-6)
                dense_teacher_confidence = probs.max(axis=1).astype(np.float32)
            dense_teacher_confidence = np.nan_to_num(dense_teacher_confidence, nan=0.0, posinf=0.0, neginf=0.0)
            dense_teacher_confidence = np.clip(dense_teacher_confidence, 0.0, 1.0).astype(np.float32)

        train_labels = map_labels_for_base_ce(raw_labels, self.class_split, ignore_index=self.ignore_index)
        all_class_train_labels = np.full(raw_labels.shape, self.ignore_index, dtype=np.int64)
        base_raw_mask = np.isin(raw_labels, self.class_split.base_label_ids)
        all_class_train_labels[base_raw_mask] = raw_labels[base_raw_mask].astype(np.int64)

        selected = subsample_indices(point_xyz.shape[0], self.max_points, self.seed + sample_idx)
        point_xyz_selected = point_xyz[selected]
        teacher_features_selected = teacher_features[selected]
        teacher_valid_mask_selected = teacher_valid_mask[selected]
        reliability_weight_selected = reliability_weight[selected]
        dense_teacher_logits_selected = dense_teacher_logits[selected]
        dense_teacher_valid_mask_selected = dense_teacher_valid_mask[selected]
        dense_teacher_confidence_selected = dense_teacher_confidence[selected]
        raw_labels_selected = raw_labels[selected].astype(np.int64)
        train_labels_selected = train_labels[selected].astype(np.int64)
        all_class_train_labels_selected = all_class_train_labels[selected].astype(np.int64)

        if self.augment:
            rng = np.random.default_rng(self.seed + sample_idx + 1000003 * self.epoch)
            point_xyz_selected, keep_mask = augment_point_xyz(point_xyz_selected, rng, self.augmentation_config)
            teacher_features_selected = teacher_features_selected[keep_mask]
            teacher_valid_mask_selected = teacher_valid_mask_selected[keep_mask]
            reliability_weight_selected = reliability_weight_selected[keep_mask]
            dense_teacher_logits_selected = dense_teacher_logits_selected[keep_mask]
            dense_teacher_valid_mask_selected = dense_teacher_valid_mask_selected[keep_mask]
            dense_teacher_confidence_selected = dense_teacher_confidence_selected[keep_mask]
            raw_labels_selected = raw_labels_selected[keep_mask]
            train_labels_selected = train_labels_selected[keep_mask]
            all_class_train_labels_selected = all_class_train_labels_selected[keep_mask]

        return {
            "sample_idx": sample_idx,
            "sample_token": sample["token"],
            "point_xyz": point_xyz_selected,
            "teacher_features": teacher_features_selected,
            "teacher_valid_mask": teacher_valid_mask_selected,
            "reliability_weight": reliability_weight_selected,
            "dense_teacher_logits": dense_teacher_logits_selected,
            "dense_teacher_valid_mask": dense_teacher_valid_mask_selected,
            "dense_teacher_confidence": dense_teacher_confidence_selected,
            "raw_labels": raw_labels_selected,
            "train_labels": train_labels_selected,
            "all_class_train_labels": all_class_train_labels_selected,
            "num_points_before_subsample": int(point_xyz.shape[0]),
            "point_feature_path": str(point_feature_path),
            "reliability_path": str(reliability_path),
            "dense_point_path": str(dense_point_path) if dense_point_path is not None else "",
        }


def collate_point_feature_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    def cat_array(key: str) -> np.ndarray:
        return np.concatenate([sample[key] for sample in samples], axis=0)

    return {
        "sample_indices": [int(sample["sample_idx"]) for sample in samples],
        "sample_tokens": [str(sample["sample_token"]) for sample in samples],
        "point_xyz": cat_array("point_xyz").astype(np.float32),
        "point_batch_indices": np.concatenate(
            [
                np.full(sample["point_xyz"].shape[0], batch_idx, dtype=np.int64)
                for batch_idx, sample in enumerate(samples)
            ],
            axis=0,
        ),
        "teacher_features": cat_array("teacher_features").astype(np.float32),
        "teacher_valid_mask": cat_array("teacher_valid_mask").astype(bool),
        "reliability_weight": cat_array("reliability_weight").astype(np.float32),
        "dense_teacher_logits": cat_array("dense_teacher_logits").astype(np.float32),
        "dense_teacher_valid_mask": cat_array("dense_teacher_valid_mask").astype(bool),
        "dense_teacher_confidence": cat_array("dense_teacher_confidence").astype(np.float32),
        "raw_labels": cat_array("raw_labels").astype(np.int64),
        "train_labels": cat_array("train_labels").astype(np.int64),
        "all_class_train_labels": cat_array("all_class_train_labels").astype(np.int64),
        "num_points_before_subsample": [int(sample["num_points_before_subsample"]) for sample in samples],
    }
