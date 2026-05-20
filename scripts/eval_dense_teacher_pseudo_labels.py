from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset  # noqa: E402
from ra_ov3dseg.evaluation.metrics import (  # noqa: E402
    confusion_matrix,
    mean_iou_for_ids,
    safe_iou,
    segmentation_intersections_unions,
)
from ra_ov3dseg.training.labels import (  # noqa: E402
    NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES,
    build_class_split,
    map_official_16_for_ce,
    map_raw_lidarseg_to_official_16,
)
from ra_ov3dseg.utils.io import ensure_dir, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402
from ra_ov3dseg.utils.run_conclusion import RunConclusion  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate projected dense 2D teacher pseudo labels against nuScenes lidarseg labels."
    )
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--version", default="v1.0-trainval", type=str)
    parser.add_argument("--sample_idx", default=None, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--max_samples", default=1, type=int)
    parser.add_argument("--dense_point_npz", default=None, type=str)
    parser.add_argument("--dense_point_dir", default="outputs/dense_point_logits", type=str)
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--split_config", default="configs/base_novel_split.yaml", type=str)
    parser.add_argument(
        "--label_space",
        default="raw32",
        choices=["raw32", "official16"],
        help="raw32 evaluates raw labels; official16 evaluates merged official lidarseg classes.",
    )
    parser.add_argument("--output_dir", default="outputs/teacher_quality", type=str)
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def build_jobs(args, dataset: NuScenesDataset) -> list[tuple[int, Path]]:
    if args.dense_point_npz is not None:
        if args.sample_idx is None:
            raise ValueError("--sample_idx is required with --dense_point_npz.")
        return [(args.sample_idx, Path(args.dense_point_npz).expanduser().resolve())]

    sample_indices = dataset.resolve_sample_indices(
        sample_idx=args.sample_idx,
        start_idx=args.start_idx,
        max_samples=args.max_samples,
    )
    dense_point_dir = Path(args.dense_point_dir).expanduser().resolve()
    return [(sample_idx, dense_point_dir / f"sample_{sample_idx:04d}_dense_point_logits.npz") for sample_idx in sample_indices]


def teacher_predictions(dense_data: dict[str, np.ndarray], num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = dense_data["point_dense_valid_mask"].astype(bool)
    if "point_dense_pred_label_indices" in dense_data:
        pred = dense_data["point_dense_pred_label_indices"].astype(np.int64)
    else:
        logits = dense_data["point_teacher_logits"].astype(np.float32)
        pred = np.argmax(logits[:, :num_classes], axis=1).astype(np.int64)
    pred = pred.copy()
    pred[(~valid_mask) | (pred < 0) | (pred >= num_classes)] = -1
    return pred, valid_mask


def summarize_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    class_names: list[str],
    base_ids: np.ndarray,
    novel_ids: np.ndarray,
    ignore_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    num_classes = len(class_names)
    valid_gt_mask = (gt >= 0) & (gt < num_classes)
    valid_pred_mask = (pred >= 0) & (pred < num_classes)
    intersections, unions, gt_counts = segmentation_intersections_unions(
        pred_labels=pred,
        gt_labels=gt,
        num_classes=num_classes,
        valid_gt_mask=valid_gt_mask,
    )
    ious = safe_iou(intersections, unions)
    pred_counts = np.asarray([int(np.sum(valid_gt_mask & (pred == class_id))) for class_id in range(num_classes)])
    conf = confusion_matrix(pred_labels=pred, gt_labels=gt, num_classes=num_classes, valid_gt_mask=valid_gt_mask)

    eval_class_ids = np.asarray(sorted(set(base_ids.tolist()) | set(novel_ids.tolist())), dtype=np.int64)
    base_set = set(base_ids.tolist())
    novel_set = set(novel_ids.tolist())
    ignore_set = set(ignore_ids.tolist())
    per_class = []
    for class_id, class_name in enumerate(class_names):
        if class_id in base_set:
            split = "base"
        elif class_id in novel_set:
            split = "novel"
        elif class_id in ignore_set:
            split = "ignore"
        else:
            split = "unknown"
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "split": split,
                "iou": finite_or_none(float(ious[class_id])),
                "intersection": int(intersections[class_id]),
                "union": int(unions[class_id]),
                "gt_count": int(gt_counts[class_id]),
                "pred_count": int(pred_counts[class_id]),
            }
        )

    metrics = {
        "all_miou": finite_or_none(mean_iou_for_ids(ious, eval_class_ids)),
        "base_miou": finite_or_none(mean_iou_for_ids(ious, base_ids)),
        "novel_miou": finite_or_none(mean_iou_for_ids(ious, novel_ids)),
        "ignore_miou": finite_or_none(mean_iou_for_ids(ious, ignore_ids)),
        "num_points": int(gt.shape[0]),
        "num_valid_pred_points": int(valid_pred_mask.sum()),
        "prediction_coverage": float(valid_pred_mask.sum() / max(gt.shape[0], 1)),
        "per_class": per_class,
    }
    arrays = {
        "intersections": intersections,
        "unions": unions,
        "gt_counts": gt_counts,
        "pred_counts": pred_counts,
        "confusion_matrix": conf,
        "ious": ious.astype(np.float32),
    }
    return metrics, arrays


def main() -> int:
    start_time = time.monotonic()
    args = build_parser().parse_args()
    logger = setup_logger("eval_dense_teacher_pseudo_labels")
    output_dir = ensure_dir(args.output_dir)
    dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=True)
    if args.label_space == "official16":
        class_names = list(NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES[1:]) + ["background"]
        base_label_ids = np.arange(16, dtype=np.int64)
        novel_label_ids = np.asarray([], dtype=np.int64)
        ignore_label_ids = np.asarray([16], dtype=np.int64)
    else:
        class_split = build_class_split(args.class_names_path, args.split_config)
        class_names = class_split.class_names
        base_label_ids = class_split.base_label_ids
        novel_label_ids = class_split.novel_label_ids
        ignore_label_ids = class_split.ignore_label_ids
    jobs = build_jobs(args, dataset)

    num_classes = len(class_names)
    total_intersections = np.zeros(num_classes, dtype=np.int64)
    total_unions = np.zeros(num_classes, dtype=np.int64)
    total_gt_counts = np.zeros(num_classes, dtype=np.int64)
    total_pred_counts = np.zeros(num_classes, dtype=np.int64)
    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_points = 0
    total_valid_pred_points = 0
    outputs = []

    for sample_idx, dense_point_npz in jobs:
        if not dense_point_npz.exists():
            raise FileNotFoundError(f"dense point logits npz not found: {dense_point_npz}")
        prefix = f"sample_{sample_idx:04d}"
        eval_npz = output_dir / f"{prefix}_teacher_pseudo_eval.npz"
        summary_json = output_dir / f"{prefix}_teacher_pseudo_eval_summary.json"
        if args.skip_existing and eval_npz.exists() and summary_json.exists():
            logger.info("skip existing teacher pseudo eval for sample_idx=%d", sample_idx)
            existing = load_npz(eval_npz)
            total_intersections += existing["intersections"].astype(np.int64)
            total_unions += existing["unions"].astype(np.int64)
            total_gt_counts += existing["gt_counts"].astype(np.int64)
            total_pred_counts += existing["pred_counts"].astype(np.int64)
            total_confusion += existing["confusion_matrix"].astype(np.int64)
            total_points += int(existing["num_points"].item())
            total_valid_pred_points += int(existing["num_valid_pred_points"].item())
            outputs.append({"sample_idx": sample_idx, "status": "skipped_existing", "summary_json": str(summary_json)})
            continue

        sample = dataset.get_sample_by_index(sample_idx)
        gt = dataset.load_lidarseg_labels(sample)
        if gt is None:
            raise FileNotFoundError(f"lidarseg labels not found for sample_idx={sample_idx}")
        if args.label_space == "official16":
            gt = map_official_16_for_ce(map_raw_lidarseg_to_official_16(gt), ignore_index=-1)
        dense_data = load_npz(dense_point_npz)
        pred, teacher_valid_mask = teacher_predictions(dense_data, num_classes=num_classes)
        if pred.shape[0] != gt.shape[0]:
            raise ValueError(f"teacher prediction/label count mismatch: pred={pred.shape[0]}, labels={gt.shape[0]}")

        metrics, arrays = summarize_metrics(
            pred=pred,
            gt=gt.astype(np.int64),
            class_names=class_names,
            base_ids=base_label_ids,
            novel_ids=novel_label_ids,
            ignore_ids=ignore_label_ids,
        )
        total_intersections += arrays["intersections"].astype(np.int64)
        total_unions += arrays["unions"].astype(np.int64)
        total_gt_counts += arrays["gt_counts"].astype(np.int64)
        total_pred_counts += arrays["pred_counts"].astype(np.int64)
        total_confusion += arrays["confusion_matrix"].astype(np.int64)
        total_points += int(gt.shape[0])
        total_valid_pred_points += int(np.sum((pred >= 0) & (pred < num_classes)))

        save_npz(
            eval_npz,
            sample_idx=np.asarray(sample_idx, dtype=np.int32),
            sample_token=np.asarray(sample["token"]),
            pred_label_indices=pred.astype(np.int32),
            teacher_valid_mask=teacher_valid_mask.astype(bool),
            class_names=np.asarray(class_names),
            base_label_ids=base_label_ids,
            novel_label_ids=novel_label_ids,
            ignore_label_ids=ignore_label_ids,
            num_points=np.asarray(gt.shape[0], dtype=np.int64),
            num_valid_pred_points=np.asarray(metrics["num_valid_pred_points"], dtype=np.int64),
            **arrays,
        )
        summary = {
            "sample_idx": sample_idx,
            "sample_token": str(sample["token"]),
            "dense_point_npz": str(dense_point_npz),
            "eval_npz": str(eval_npz),
            "metrics": metrics,
        }
        save_json(summary_json, summary)
        outputs.append({"sample_idx": sample_idx, "status": "done", "summary_json": str(summary_json), "metrics": metrics})
        logger.info(
            "teacher pseudo eval saved | sample_idx=%d | all_miou=%s | base_miou=%s | novel_miou=%s",
            sample_idx,
            metrics["all_miou"],
            metrics["base_miou"],
            metrics["novel_miou"],
        )

    aggregate_metrics, _ = summarize_metrics(
        pred=np.zeros((0,), dtype=np.int64),
        gt=np.zeros((0,), dtype=np.int64),
        class_names=class_names,
        base_ids=base_label_ids,
        novel_ids=novel_label_ids,
        ignore_ids=ignore_label_ids,
    )
    aggregate_ious = safe_iou(total_intersections, total_unions)
    eval_class_ids = np.asarray(
        sorted(set(base_label_ids.tolist()) | set(novel_label_ids.tolist())), dtype=np.int64
    )
    aggregate_metrics.update(
        {
            "all_miou": finite_or_none(mean_iou_for_ids(aggregate_ious, eval_class_ids)),
            "base_miou": finite_or_none(mean_iou_for_ids(aggregate_ious, base_label_ids)),
            "novel_miou": finite_or_none(mean_iou_for_ids(aggregate_ious, novel_label_ids)),
            "ignore_miou": finite_or_none(mean_iou_for_ids(aggregate_ious, ignore_label_ids)),
            "num_points": int(total_points),
            "num_valid_pred_points": int(total_valid_pred_points),
            "prediction_coverage": float(total_valid_pred_points / max(total_points, 1)),
        }
    )
    base_set = set(base_label_ids.tolist())
    novel_set = set(novel_label_ids.tolist())
    ignore_set = set(ignore_label_ids.tolist())
    aggregate_metrics["per_class"] = []
    for class_id, class_name in enumerate(class_names):
        if class_id in base_set:
            split = "base"
        elif class_id in novel_set:
            split = "novel"
        elif class_id in ignore_set:
            split = "ignore"
        else:
            split = "unknown"
        aggregate_metrics["per_class"].append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "split": split,
                "iou": finite_or_none(float(aggregate_ious[class_id])),
                "intersection": int(total_intersections[class_id]),
                "union": int(total_unions[class_id]),
                "gt_count": int(total_gt_counts[class_id]),
                "pred_count": int(total_pred_counts[class_id]),
            }
        )

    batch_eval_npz = output_dir / "batch_teacher_pseudo_eval.npz"
    batch_summary_json = output_dir / "batch_teacher_pseudo_eval_summary.json"
    save_npz(
        batch_eval_npz,
        class_names=np.asarray(class_names),
        intersections=total_intersections,
        unions=total_unions,
        gt_counts=total_gt_counts,
        pred_counts=total_pred_counts,
        ious=aggregate_ious.astype(np.float32),
        confusion_matrix=total_confusion,
        base_label_ids=base_label_ids,
        novel_label_ids=novel_label_ids,
        ignore_label_ids=ignore_label_ids,
    )
    save_json(
        batch_summary_json,
        {
            "version": args.version,
            "label_space": args.label_space,
            "num_samples": len(jobs),
            "outputs": outputs,
            "aggregate_metrics": aggregate_metrics,
            "batch_eval_npz": str(batch_eval_npz),
        },
    )
    logger.info(
        "batch teacher pseudo eval saved | samples=%d | all_miou=%s | base_miou=%s | novel_miou=%s",
        len(jobs),
        aggregate_metrics["all_miou"],
        aggregate_metrics["base_miou"],
        aggregate_metrics["novel_miou"],
    )
    primary = aggregate_metrics["all_miou"]
    primary_value = float(primary) if primary is not None else 0.0
    secondary = {}
    for key in ("base_miou", "novel_miou", "prediction_coverage"):
        value = aggregate_metrics.get(key)
        if value is not None and np.isfinite(float(value)):
            secondary[key] = float(value)
    conclusion = RunConclusion(
        stage="stage-teacher",
        experiment="eval_dense_teacher_pseudo_labels",
        status="success",
        gate="projected teacher mIoU >= 0.10",
        gate_passed=primary_value >= 0.10,
        primary_metric_name="teacher_miou",
        primary_metric_value=primary_value,
        secondary=secondary,
        runtime_seconds=time.monotonic() - start_time,
        checkpoint=None,
        artifacts=[str(batch_summary_json), str(batch_eval_npz)],
        next_step="proceed to Stage 3 diagnostic review if gate passed",
        notes=f"label_space={args.label_space}; samples={len(jobs)}",
    )
    conclusion.append_to_recap(Path("docs/EXPERIMENT_RECAP.md"))
    conclusion.print_block()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
