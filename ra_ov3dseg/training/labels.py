from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ra_ov3dseg.utils.config import load_yaml_config
from ra_ov3dseg.utils.io import load_text_lines


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
