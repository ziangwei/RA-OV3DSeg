from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset  # noqa: E402
from ra_ov3dseg.training.labels import (  # noqa: E402
    NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES,
    build_class_split,
    map_labels_for_base_ce,
    map_official_16_for_ce,
    map_raw_lidarseg_to_official_16,
)
from ra_ov3dseg.training.precomputed_dataset import IGNORE_INDEX  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute lidarseg class frequencies and CE weights for training.")
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--version", default="v1.0-trainval", type=str)
    parser.add_argument("--sample_idx", default=None, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--max_samples", default=1, type=int)
    parser.add_argument("--all_samples", action="store_true")
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--split_config", default="configs/all_lidarseg_supervised_split.yaml", type=str)
    parser.add_argument("--output_json", default="outputs/class_frequencies/class_frequencies.json", type=str)
    parser.add_argument("--min_count", default=1.0, type=float)
    parser.add_argument("--skip_missing_labels", action="store_true")
    return parser


def normalized_inverse_sqrt_weights(counts: np.ndarray, active_mask: np.ndarray, min_count: float) -> np.ndarray:
    counts = counts.astype(np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    active = active_mask.astype(bool)
    positive = active & (counts > 0)
    weights[positive] = 1.0 / np.sqrt(np.maximum(counts[positive], float(min_count)))
    if np.any(positive):
        weights[positive] *= float(np.sum(positive)) / np.sum(weights[positive])
    return weights.astype(np.float32)


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("compute_class_frequencies")
    dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=True)
    max_samples = None if args.all_samples else args.max_samples
    sample_indices = dataset.resolve_sample_indices(
        sample_idx=args.sample_idx,
        start_idx=args.start_idx,
        max_samples=max_samples,
    )
    class_split = build_class_split(args.class_names_path, args.split_config)
    num_classes = class_split.num_classes

    raw_counts = np.zeros(num_classes, dtype=np.int64)
    train_counts = np.zeros(class_split.num_train_classes, dtype=np.int64)
    official_16_counts = np.zeros(len(NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES), dtype=np.int64)
    official_16_train_counts = np.zeros(16, dtype=np.int64)
    missing_labels: list[int] = []
    for sample_idx in sample_indices:
        sample = dataset.get_sample_by_index(sample_idx)
        labels = dataset.load_lidarseg_labels(sample)
        if labels is None:
            if args.skip_missing_labels:
                missing_labels.append(int(sample_idx))
                continue
            raise FileNotFoundError(f"lidarseg labels not found for sample_idx={sample_idx}")

        valid_raw = labels[(labels >= 0) & (labels < num_classes)].astype(np.int64)
        raw_counts += np.bincount(valid_raw, minlength=num_classes)[:num_classes]
        train_labels = map_labels_for_base_ce(labels, class_split, ignore_index=IGNORE_INDEX)
        valid_train = train_labels[train_labels != IGNORE_INDEX].astype(np.int64)
        train_counts += np.bincount(valid_train, minlength=class_split.num_train_classes)[: class_split.num_train_classes]
        official_16_labels = map_raw_lidarseg_to_official_16(labels)
        official_16_counts += np.bincount(
            official_16_labels.astype(np.int64), minlength=len(NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES)
        )[: len(NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES)]
        official_16_train_labels = map_official_16_for_ce(official_16_labels, ignore_index=IGNORE_INDEX)
        valid_official_16_train = official_16_train_labels[official_16_train_labels != IGNORE_INDEX].astype(np.int64)
        official_16_train_counts += np.bincount(valid_official_16_train, minlength=16)[:16]

    raw_active_mask = np.zeros(num_classes, dtype=bool)
    raw_active_mask[class_split.base_label_ids] = True
    train_active_mask = np.ones(class_split.num_train_classes, dtype=bool)
    raw_weights = normalized_inverse_sqrt_weights(raw_counts, raw_active_mask, min_count=args.min_count)
    train_weights = normalized_inverse_sqrt_weights(train_counts, train_active_mask, min_count=args.min_count)
    official_16_active_mask = np.ones(16, dtype=bool)
    official_16_weights = normalized_inverse_sqrt_weights(
        official_16_train_counts,
        official_16_active_mask,
        min_count=args.min_count,
    )

    total_raw = int(raw_counts.sum())
    raw_freq = (raw_counts / max(total_raw, 1)).astype(np.float64)
    total_train = int(train_counts.sum())
    train_freq = (train_counts / max(total_train, 1)).astype(np.float64)
    total_official_16 = int(official_16_counts.sum())
    official_16_freq = (official_16_counts / max(total_official_16, 1)).astype(np.float64)
    total_official_16_train = int(official_16_train_counts.sum())
    official_16_train_freq = (official_16_train_counts / max(total_official_16_train, 1)).astype(np.float64)

    per_class: list[dict[str, Any]] = []
    base_ids = set(class_split.base_label_ids.tolist())
    novel_ids = set(class_split.novel_label_ids.tolist())
    ignore_ids = set(class_split.ignore_label_ids.tolist())
    for class_id, class_name in enumerate(class_split.class_names):
        if class_id in base_ids:
            split = "base"
        elif class_id in novel_ids:
            split = "novel"
        elif class_id in ignore_ids:
            split = "ignore"
        else:
            split = "unknown"
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "split": split,
                "raw_count": int(raw_counts[class_id]),
                "raw_frequency": float(raw_freq[class_id]),
                "raw_class_weight": float(raw_weights[class_id]),
            }
        )

    output_json = Path(args.output_json).expanduser().resolve()
    ensure_dir(output_json.parent)
    summary = {
        "version": args.version,
        "dataroot": str(Path(args.dataroot).expanduser().resolve()),
        "sample_indices": [int(idx) for idx in sample_indices],
        "num_samples_requested": len(sample_indices),
        "num_samples_used": len(sample_indices) - len(missing_labels),
        "missing_label_sample_indices": missing_labels,
        "class_names": class_split.class_names,
        "base_class_names": class_split.base_class_names,
        "ignore_class_names": class_split.ignore_class_names,
        "train_id_to_label_id": class_split.train_id_to_label_id.astype(int).tolist(),
        "raw_counts": raw_counts.astype(int).tolist(),
        "raw_frequencies": raw_freq.astype(float).tolist(),
        "raw_class_weights": raw_weights.astype(float).tolist(),
        "train_counts": train_counts.astype(int).tolist(),
        "train_frequencies": train_freq.astype(float).tolist(),
        "train_class_weights": train_weights.astype(float).tolist(),
        "official_16_class_names": NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES,
        "official_16_counts": official_16_counts.astype(int).tolist(),
        "official_16_frequencies": official_16_freq.astype(float).tolist(),
        "official_16_train_counts": official_16_train_counts.astype(int).tolist(),
        "official_16_train_frequencies": official_16_train_freq.astype(float).tolist(),
        "official_16_class_weights": official_16_weights.astype(float).tolist(),
        "per_class": per_class,
    }
    save_json(output_json, summary)
    logger.info(
        "class frequencies saved | samples=%d | total_points=%d | output=%s",
        summary["num_samples_used"],
        total_raw,
        output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
