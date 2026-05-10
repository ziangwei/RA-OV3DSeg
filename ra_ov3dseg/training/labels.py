from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ra_ov3dseg.utils.config import load_yaml_config
from ra_ov3dseg.utils.io import load_text_lines


NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES = [
    "void",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

# nuScenes-lidarseg stores 32 raw labels, while the official lidarseg challenge
# evaluates 16 merged classes plus index 0 as void/ignore.
NUSCENES_RAW_TO_OFFICIAL_16 = np.asarray(
    [
        0,  # noise -> void
        0,  # animal -> void
        7,  # human.pedestrian.adult -> pedestrian
        7,  # human.pedestrian.child -> pedestrian
        7,  # human.pedestrian.construction_worker -> pedestrian
        0,  # human.pedestrian.personal_mobility -> void
        7,  # human.pedestrian.police_officer -> pedestrian
        0,  # human.pedestrian.stroller -> void
        0,  # human.pedestrian.wheelchair -> void
        1,  # movable_object.barrier -> barrier
        0,  # movable_object.debris -> void
        0,  # movable_object.pushable_pullable -> void
        8,  # movable_object.trafficcone -> traffic_cone
        0,  # static_object.bicycle_rack -> void
        2,  # vehicle.bicycle -> bicycle
        3,  # vehicle.bus.bendy -> bus
        3,  # vehicle.bus.rigid -> bus
        4,  # vehicle.car -> car
        5,  # vehicle.construction -> construction_vehicle
        0,  # vehicle.emergency.ambulance -> void
        0,  # vehicle.emergency.police -> void
        6,  # vehicle.motorcycle -> motorcycle
        9,  # vehicle.trailer -> trailer
        10,  # vehicle.truck -> truck
        11,  # flat.driveable_surface -> driveable_surface
        12,  # flat.other -> other_flat
        13,  # flat.sidewalk -> sidewalk
        14,  # terrain -> terrain
        15,  # static.manmade -> manmade
        0,  # static.other -> void
        16,  # static.vegetation -> vegetation
        0,  # vehicle.ego -> void
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class ClassSplit:
    class_names: list[str]
    base_class_names: list[str]
    novel_class_names: list[str]
    ignore_class_names: list[str]
    base_label_ids: np.ndarray
    novel_label_ids: np.ndarray
    ignore_label_ids: np.ndarray
    train_id_to_label_id: np.ndarray
    label_id_to_train_id: np.ndarray

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_train_classes(self) -> int:
        return len(self.base_label_ids)


def _names_to_ids(class_names: list[str], selected_names: list[str], field_name: str) -> np.ndarray:
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in selected_names if name not in name_to_id]
    if missing:
        raise ValueError(f"{field_name} contains unknown classes: {missing}")
    return np.asarray([name_to_id[name] for name in selected_names], dtype=np.int64)


def build_class_split(class_names_path: str | Path, split_config_path: str | Path) -> ClassSplit:
    class_names = load_text_lines(class_names_path)
    split_config: dict[str, Any] = load_yaml_config(split_config_path)

    base_class_names = list(split_config.get("base_classes", []))
    novel_class_names = list(split_config.get("novel_classes", []))
    ignore_class_names = list(split_config.get("ignore_classes", []))

    base_label_ids = _names_to_ids(class_names, base_class_names, "base_classes")
    novel_label_ids = _names_to_ids(class_names, novel_class_names, "novel_classes")
    ignore_label_ids = _names_to_ids(class_names, ignore_class_names, "ignore_classes")

    overlap = set(base_label_ids.tolist()) & set(novel_label_ids.tolist())
    overlap |= set(base_label_ids.tolist()) & set(ignore_label_ids.tolist())
    overlap |= set(novel_label_ids.tolist()) & set(ignore_label_ids.tolist())
    if overlap:
        overlap_names = [class_names[idx] for idx in sorted(overlap)]
        raise ValueError(f"class split has overlapping classes: {overlap_names}")

    label_id_to_train_id = np.full(len(class_names), -1, dtype=np.int64)
    for train_id, label_id in enumerate(base_label_ids.tolist()):
        label_id_to_train_id[label_id] = train_id

    return ClassSplit(
        class_names=class_names,
        base_class_names=base_class_names,
        novel_class_names=novel_class_names,
        ignore_class_names=ignore_class_names,
        base_label_ids=base_label_ids,
        novel_label_ids=novel_label_ids,
        ignore_label_ids=ignore_label_ids,
        train_id_to_label_id=base_label_ids.copy(),
        label_id_to_train_id=label_id_to_train_id,
    )


def map_labels_for_base_ce(labels: np.ndarray, class_split: ClassSplit, ignore_index: int = -100) -> np.ndarray:
    labels = labels.astype(np.int64)
    mapped = np.full(labels.shape, ignore_index, dtype=np.int64)
    valid = (labels >= 0) & (labels < class_split.label_id_to_train_id.shape[0])
    train_ids = np.full(labels.shape, -1, dtype=np.int64)
    train_ids[valid] = class_split.label_id_to_train_id[labels[valid]]
    base_mask = train_ids >= 0
    mapped[base_mask] = train_ids[base_mask]
    return mapped


def map_raw_lidarseg_to_official_16(raw_labels: np.ndarray) -> np.ndarray:
    """Map nuScenes-lidarseg raw 32-class ids to official 16-class ids.

    Output ids follow the nuScenes lidarseg challenge convention:
    0 is void/ignore, and 1..16 are evaluated semantic classes.
    """

    raw_labels = raw_labels.astype(np.int64)
    mapped = np.zeros(raw_labels.shape, dtype=np.int64)
    valid = (raw_labels >= 0) & (raw_labels < NUSCENES_RAW_TO_OFFICIAL_16.shape[0])
    mapped[valid] = NUSCENES_RAW_TO_OFFICIAL_16[raw_labels[valid]]
    return mapped


def map_official_16_for_ce(official_labels: np.ndarray, ignore_index: int = -100) -> np.ndarray:
    """Convert official lidarseg ids to contiguous CE ids.

    Official id 0 is ignored. Official ids 1..16 become train ids 0..15.
    """

    official_labels = official_labels.astype(np.int64)
    mapped = np.full(official_labels.shape, ignore_index, dtype=np.int64)
    valid = (official_labels >= 1) & (official_labels <= 16)
    mapped[valid] = official_labels[valid] - 1
    return mapped
