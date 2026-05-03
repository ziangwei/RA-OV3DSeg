from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset
from ra_ov3dseg.training.labels import ClassSplit, map_labels_for_base_ce


IGNORE_INDEX = -100


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def default_point_feature_path(point_feature_dir: str | Path, sample_idx: int) -> Path:
    return Path(point_feature_dir).expanduser().resolve() / f"sample_{sample_idx:04d}_point_features.npz"


def default_reliability_path(reliability_dir: str | Path, sample_idx: int) -> Path:
    return Path(reliability_dir).expanduser().resolve() / f"sample_{sample_idx:04d}_reliability.npz"


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
        max_points: int | None = None,
        seed: int = 0,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if not sample_indices:
            raise ValueError("sample_indices must not be empty.")

        self.nuscenes_dataset = nuscenes_dataset
        self.sample_indices = list(sample_indices)
        self.point_feature_dir = Path(point_feature_dir).expanduser().resolve()
        self.reliability_dir = Path(reliability_dir).expanduser().resolve()
        self.class_split = class_split
        self.max_points = max_points
        self.seed = seed
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.sample_indices)

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

        if raw_labels.shape[0] != point_xyz.shape[0]:
            raise ValueError(f"label/point count mismatch: labels={raw_labels.shape[0]}, points={point_xyz.shape[0]}")
        if teacher_features.shape[0] != point_xyz.shape[0]:
            raise ValueError("teacher feature count does not match point count.")
        if teacher_valid_mask.shape[0] != point_xyz.shape[0]:
            raise ValueError("teacher valid mask count does not match point count.")
        if reliability_weight.shape[0] != point_xyz.shape[0]:
            raise ValueError("reliability count does not match point count.")

        train_labels = map_labels_for_base_ce(raw_labels, self.class_split, ignore_index=self.ignore_index)

        selected = subsample_indices(point_xyz.shape[0], self.max_points, self.seed + sample_idx)
        return {
            "sample_idx": sample_idx,
            "sample_token": sample["token"],
            "point_xyz": point_xyz[selected],
            "teacher_features": teacher_features[selected],
            "teacher_valid_mask": teacher_valid_mask[selected],
            "reliability_weight": reliability_weight[selected],
            "raw_labels": raw_labels[selected].astype(np.int64),
            "train_labels": train_labels[selected].astype(np.int64),
            "num_points_before_subsample": int(point_xyz.shape[0]),
            "point_feature_path": str(point_feature_path),
            "reliability_path": str(reliability_path),
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
        "teacher_features": cat_array("teacher_features").astype(np.float32),
        "teacher_valid_mask": cat_array("teacher_valid_mask").astype(bool),
        "reliability_weight": cat_array("reliability_weight").astype(np.float32),
        "raw_labels": cat_array("raw_labels").astype(np.int64),
        "train_labels": cat_array("train_labels").astype(np.int64),
        "num_points_before_subsample": [int(sample["num_points_before_subsample"]) for sample in samples],
    }
