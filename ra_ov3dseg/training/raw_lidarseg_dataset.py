from __future__ import annotations

from typing import Any

import numpy as np

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset
from ra_ov3dseg.training.augmentations import PointAugmentationConfig, augment_point_xyz
from ra_ov3dseg.training.labels import (
    ClassSplit,
    map_labels_for_base_ce,
    map_official_16_for_ce,
    map_raw_lidarseg_to_official_16,
)
from ra_ov3dseg.training.precomputed_dataset import IGNORE_INDEX, subsample_indices


class RawLidarsegDataset:
    """Supervised nuScenes-lidarseg dataset that reads raw LiDAR and labels only.

    This is the right dataset for closed-set 3D upper-bound training. It avoids
    the V9/V12 precomputed 2D feature caches, which are unnecessary for pure
    lidarseg supervision and would make full trainval training impractical.
    """

    def __init__(
        self,
        nuscenes_dataset: NuScenesDataset,
        sample_indices: list[int],
        class_split: ClassSplit,
        max_points: int | None = None,
        seed: int = 0,
        ignore_index: int = IGNORE_INDEX,
        augment: bool = False,
        augmentation_config: PointAugmentationConfig | None = None,
        feature_dim: int = 512,
    ) -> None:
        if not sample_indices:
            raise ValueError("sample_indices must not be empty.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")

        self.nuscenes_dataset = nuscenes_dataset
        self.sample_indices = list(sample_indices)
        self.class_split = class_split
        self.max_points = max_points
        self.seed = seed
        self.ignore_index = ignore_index
        self.augment = bool(augment)
        self.augmentation_config = augmentation_config or PointAugmentationConfig()
        self.feature_dim = int(feature_dim)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.sample_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_idx = self.sample_indices[index]
        sample = self.nuscenes_dataset.get_sample_by_index(sample_idx)
        point_xyzi = self.nuscenes_dataset.load_lidar_points_xyzi(sample).astype(np.float32)
        point_xyz = point_xyzi[:, :3]
        point_input_features = point_xyzi[:, 3:4]
        raw_labels = self.nuscenes_dataset.load_lidarseg_labels(sample)
        if raw_labels is None:
            raise FileNotFoundError(f"lidarseg labels not found for sample_idx={sample_idx}")
        if raw_labels.shape[0] != point_xyz.shape[0]:
            raise ValueError(
                f"label/point count mismatch: sample_idx={sample_idx}, "
                f"labels={raw_labels.shape[0]}, points={point_xyz.shape[0]}"
            )

        train_labels = map_labels_for_base_ce(raw_labels, self.class_split, ignore_index=self.ignore_index)
        official_16_labels = map_raw_lidarseg_to_official_16(raw_labels)
        official_16_train_labels = map_official_16_for_ce(official_16_labels, ignore_index=self.ignore_index)
        all_class_train_labels = np.full(raw_labels.shape, self.ignore_index, dtype=np.int64)
        base_raw_mask = np.isin(raw_labels, self.class_split.base_label_ids)
        all_class_train_labels[base_raw_mask] = raw_labels[base_raw_mask].astype(np.int64)

        selected = subsample_indices(point_xyz.shape[0], self.max_points, self.seed + sample_idx)
        point_xyz_selected = point_xyz[selected]
        point_input_features_selected = point_input_features[selected]
        raw_labels_selected = raw_labels[selected].astype(np.int64)
        train_labels_selected = train_labels[selected].astype(np.int64)
        official_16_train_labels_selected = official_16_train_labels[selected].astype(np.int64)
        all_class_train_labels_selected = all_class_train_labels[selected].astype(np.int64)

        if self.augment:
            rng = np.random.default_rng(self.seed + sample_idx + 1000003 * self.epoch)
            point_xyz_selected, keep_mask = augment_point_xyz(point_xyz_selected, rng, self.augmentation_config)
            point_input_features_selected = point_input_features_selected[keep_mask]
            raw_labels_selected = raw_labels_selected[keep_mask]
            train_labels_selected = train_labels_selected[keep_mask]
            official_16_train_labels_selected = official_16_train_labels_selected[keep_mask]
            all_class_train_labels_selected = all_class_train_labels_selected[keep_mask]

        num_points = point_xyz_selected.shape[0]
        return {
            "sample_idx": sample_idx,
            "sample_token": sample["token"],
            "point_xyz": point_xyz_selected,
            "point_input_features": point_input_features_selected.astype(np.float32),
            "teacher_features": np.zeros((num_points, self.feature_dim), dtype=np.float32),
            "teacher_valid_mask": np.zeros(num_points, dtype=bool),
            "reliability_weight": np.zeros(num_points, dtype=np.float32),
            "dense_teacher_logits": np.zeros((num_points, 1), dtype=np.float32),
            "dense_teacher_valid_mask": np.zeros(num_points, dtype=bool),
            "dense_teacher_confidence": np.zeros(num_points, dtype=np.float32),
            "raw_labels": raw_labels_selected,
            "train_labels": train_labels_selected,
            "official_16_train_labels": official_16_train_labels_selected,
            "all_class_train_labels": all_class_train_labels_selected,
            "num_points_before_subsample": int(point_xyz.shape[0]),
            "point_feature_path": "",
            "reliability_path": "",
            "dense_point_path": "",
        }
