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
from ra_ov3dseg.evaluation.metrics import (  # noqa: E402
    confusion_matrix,
    mean_iou_for_ids,
    safe_iou,
    segmentation_intersections_unions,
)
from ra_ov3dseg.training.labels import build_class_split  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, load_json, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate 3D point predictions against nuScenes lidarseg labels.")
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes dataroot.")
    parser.add_argument("--version", default="v1.0-mini", type=str)
    parser.add_argument("--sample_idx", default=None, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--max_samples", default=1, type=int)
    parser.add_argument("--prediction_npz", default=None, type=str, help="Explicit prediction npz for one sample.")
    parser.add_argument("--prediction_dir", default="outputs/predictions3d", type=str)
    parser.add_argument(
        "--prediction_file_template",
        default="sample_{sample_idx:04d}_3d_predictions.npz",
        type=str,
        help="Prediction filename template used in batch mode.",
    )
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--split_config", default="configs/base_novel_split.yaml", type=str)
    parser.add_argument("--output_dir", default="outputs/evaluation3d", type=str)
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_jobs(args, dataset: NuScenesDataset) -> list[tuple[int, Path]]:
    if args.prediction_npz is not None:
        if args.sample_idx is None:
            raise ValueError("--sample_idx is required with --prediction_npz.")
        return [(args.sample_idx, Path(args.prediction_npz).expanduser().resolve())]

    sample_indices = dataset.resolve_sample_indices(
        sample_idx=args.sample_idx,
        start_idx=args.start_idx,
        max_samples=args.max_samples,
    )
    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    return [
        (sample_idx, prediction_dir / args.prediction_file_template.format(sample_idx=sample_idx))
        for sample_idx in sample_indices
    ]


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def summarize_sample(
    sample_idx: int,
    sample_token: str,
    pred: np.ndarray,
    gt: np.ndarray,
    class_names: list[str],
    base_ids: np.ndarray,
    novel_ids: np.ndarray,
    ignore_ids: np.ndarray,
) -> dict[str, Any]:
    num_classes = len(class_names)
    valid_gt_mask = (gt >= 0) & (gt < num_classes)
    valid_pred_mask = (pred >= 0) & (pred < num_classes)
    eval_class_ids = np.asarray(sorted(set(base_ids.tolist()) | set(novel_ids.tolist())), dtype=np.int64)
    intersections, unions, gt_counts = segmentation_intersections_unions(
        pred_labels=pred,
        gt_labels=gt,
        num_classes=num_classes,
        valid_gt_mask=valid_gt_mask,
    )
    ious = safe_iou(intersections, unions)
    conf = confusion_matrix(pred_labels=pred, gt_labels=gt, num_classes=num_classes, valid_gt_mask=valid_gt_mask)

    per_class = []
    for class_id, class_name in enumerate(class_names):
        if class_id in set(base_ids.tolist()):
            split = "base"
        elif class_id in set(novel_ids.tolist()):
            split = "novel"
        elif class_id in set(ignore_ids.tolist()):
            split = "ignore"
        else:
            split = "unknown"
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "split": split,
                "iou": finite_or_none(float(ious[class_id])),
                "intersection": int(intersections[class_id]),
                "union": int(unions[class_id]),
                "gt_count": int(gt_counts[class_id]),
                "pred_count": int(np.sum(valid_gt_mask & (pred == class_id))),
            }
        )

    metrics = {
        "all_miou": finite_or_none(mean_iou_for_ids(ious, eval_class_ids)),
        "base_miou": finite_or_none(mean_iou_for_ids(ious, base_ids)),
        "novel_miou": finite_or_none(mean_iou_for_ids(ious, novel_ids)),
        "ignore_miou": finite_or_none(mean_iou_for_ids(ious, ignore_ids)),
        "num_points": int(gt.shape[0]),
        "num_valid_gt_points": int(valid_gt_mask.sum()),
        "num_valid_pred_points": int(valid_pred_mask.sum()),
        "prediction_coverage": float(valid_pred_mask.sum() / max(gt.shape[0], 1)),
    }
    return {
        "sample_idx": sample_idx,
        "sample_token": sample_token,
        "metrics": metrics,
        "per_class": per_class,
        "intersections": intersections,
        "unions": unions,
        "gt_counts": gt_counts,
        "confusion_matrix": conf,
    }


def aggregate_metric_json(
    intersections: np.ndarray,
    unions: np.ndarray,
    gt_counts: np.ndarray,
    pred_counts: np.ndarray,
    class_names: list[str],
    base_ids: np.ndarray,
    novel_ids: np.ndarray,
    ignore_ids: np.ndarray,
    num_points: int,
    num_valid_pred_points: int,
) -> dict[str, Any]:
    ious = safe_iou(intersections, unions)
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
                "class_id": class_id,
                "class_name": class_name,
                "split": split,
                "iou": finite_or_none(float(ious[class_id])),
                "intersection": int(intersections[class_id]),
                "union": int(unions[class_id]),
                "gt_count": int(gt_counts[class_id]),
                "pred_count": int(pred_counts[class_id]),
            }
        )
    return {
        "all_miou": finite_or_none(mean_iou_for_ids(ious, eval_class_ids)),
        "base_miou": finite_or_none(mean_iou_for_ids(ious, base_ids)),
        "novel_miou": finite_or_none(mean_iou_for_ids(ious, novel_ids)),
        "ignore_miou": finite_or_none(mean_iou_for_ids(ious, ignore_ids)),
        "num_points": int(num_points),
        "num_valid_pred_points": int(num_valid_pred_points),
        "prediction_coverage": float(num_valid_pred_points / max(num_points, 1)),
        "per_class": per_class,
    }


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("eval_lidarseg")
    output_dir = ensure_dir(args.output_dir)
    dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=True)
    class_split = build_class_split(args.class_names_path, args.split_config)
    jobs = build_jobs(args, dataset)

    num_classes = class_split.num_classes
    total_intersections = np.zeros(num_classes, dtype=np.int64)
    total_unions = np.zeros(num_classes, dtype=np.int64)
    total_gt_counts = np.zeros(num_classes, dtype=np.int64)
    total_pred_counts = np.zeros(num_classes, dtype=np.int64)
    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_points = 0
    total_valid_pred_points = 0
    batch_outputs = []

    for sample_idx, prediction_npz in jobs:
        if not prediction_npz.exists():
            raise FileNotFoundError(f"prediction npz not found: {prediction_npz}")
        prefix = f"sample_{sample_idx:04d}"
        eval_npz = output_dir / f"{prefix}_3d_eval.npz"
        summary_json = output_dir / f"{prefix}_3d_eval_summary.json"
        if args.skip_existing and eval_npz.exists() and summary_json.exists():
            logger.info("skip existing eval outputs for sample_idx=%d", sample_idx)
            existing_eval = load_npz(eval_npz)
            existing_summary = load_json(summary_json)
            total_intersections += existing_eval["intersections"].astype(np.int64)
            total_unions += existing_eval["unions"].astype(np.int64)
            total_gt_counts += existing_eval["gt_counts"].astype(np.int64)
            total_pred_counts += existing_eval["pred_counts"].astype(np.int64)
            total_confusion += existing_eval["confusion_matrix"].astype(np.int64)
            metrics = existing_summary.get("metrics", {})
            total_points += int(metrics.get("num_points", int(existing_eval["gt_counts"].sum())))
            total_valid_pred_points += int(metrics.get("num_valid_pred_points", int(existing_eval["pred_counts"].sum())))
            batch_outputs.append(
                {
                    "sample_idx": sample_idx,
                    "status": "skipped_existing",
                    "eval_npz": str(eval_npz),
                    "summary_json": str(summary_json),
                    "metrics": metrics,
                }
            )
            continue

        sample = dataset.get_sample_by_index(sample_idx)
        gt = dataset.load_lidarseg_labels(sample)
        if gt is None:
            raise FileNotFoundError(f"lidarseg labels not found for sample_idx={sample_idx}")
        pred_data = load_npz(prediction_npz)
        if "class_names" in pred_data:
            pred_class_names = [str(name) for name in pred_data["class_names"].tolist()]
            if pred_class_names[: class_split.num_classes] != class_split.class_names:
                raise ValueError("prediction class_names order does not match lidarseg class_names.")
        pred = pred_data["pred_label_indices"].astype(np.int64)
        if pred.shape[0] != gt.shape[0]:
            raise ValueError(f"prediction/label count mismatch: pred={pred.shape[0]}, labels={gt.shape[0]}")

        sample_result = summarize_sample(
            sample_idx=sample_idx,
            sample_token=str(sample["token"]),
            pred=pred,
            gt=gt.astype(np.int64),
            class_names=class_split.class_names,
            base_ids=class_split.base_label_ids,
            novel_ids=class_split.novel_label_ids,
            ignore_ids=class_split.ignore_label_ids,
        )
        intersections = sample_result["intersections"]
        unions = sample_result["unions"]
        gt_counts = sample_result["gt_counts"]
        conf = sample_result["confusion_matrix"]
        valid_gt_mask = (gt >= 0) & (gt < num_classes)
        valid_pred_mask = (pred >= 0) & (pred < num_classes)
        pred_counts = np.asarray([int(np.sum(valid_gt_mask & (pred == class_id))) for class_id in range(num_classes)])

        total_intersections += intersections
        total_unions += unions
        total_gt_counts += gt_counts
        total_pred_counts += pred_counts
        total_confusion += conf
        total_points += int(gt.shape[0])
        total_valid_pred_points += int(valid_pred_mask.sum())

        save_npz(
            eval_npz,
            sample_idx=np.array(sample_idx, dtype=np.int32),
            sample_token=np.asarray(sample["token"]),
            class_names=np.asarray(class_split.class_names),
            intersections=intersections,
            unions=unions,
            gt_counts=gt_counts,
            pred_counts=pred_counts,
            ious=safe_iou(intersections, unions).astype(np.float32),
            confusion_matrix=conf,
            base_label_ids=class_split.base_label_ids,
            novel_label_ids=class_split.novel_label_ids,
            ignore_label_ids=class_split.ignore_label_ids,
        )
        summary = {
            "sample_idx": sample_idx,
            "sample_token": str(sample["token"]),
            "prediction_npz": str(prediction_npz),
            "eval_npz": str(eval_npz),
            "metrics": sample_result["metrics"],
            "per_class": sample_result["per_class"],
        }
        save_json(summary_json, summary)
        logger.info(
            "eval saved | sample_idx=%d | all_miou=%s | base_miou=%s | novel_miou=%s",
            sample_idx,
            summary["metrics"]["all_miou"],
            summary["metrics"]["base_miou"],
            summary["metrics"]["novel_miou"],
        )
        batch_outputs.append(
            {
                "sample_idx": sample_idx,
                "status": "done",
                "eval_npz": str(eval_npz),
                "summary_json": str(summary_json),
                "metrics": sample_result["metrics"],
            }
        )

    if len(jobs) > 1:
        aggregate = aggregate_metric_json(
            intersections=total_intersections,
            unions=total_unions,
            gt_counts=total_gt_counts,
            pred_counts=total_pred_counts,
            class_names=class_split.class_names,
            base_ids=class_split.base_label_ids,
            novel_ids=class_split.novel_label_ids,
            ignore_ids=class_split.ignore_label_ids,
            num_points=total_points,
            num_valid_pred_points=total_valid_pred_points,
        )
        batch_eval_npz = output_dir / "batch_3d_eval.npz"
        batch_summary_json = output_dir / "batch_3d_eval_summary.json"
        save_npz(
            batch_eval_npz,
            class_names=np.asarray(class_split.class_names),
            intersections=total_intersections,
            unions=total_unions,
            gt_counts=total_gt_counts,
            pred_counts=total_pred_counts,
            ious=safe_iou(total_intersections, total_unions).astype(np.float32),
            confusion_matrix=total_confusion,
            base_label_ids=class_split.base_label_ids,
            novel_label_ids=class_split.novel_label_ids,
            ignore_label_ids=class_split.ignore_label_ids,
        )
        save_json(
            batch_summary_json,
            {
                "version": args.version,
                "num_samples": len(jobs),
                "outputs": batch_outputs,
                "aggregate_metrics": aggregate,
                "batch_eval_npz": str(batch_eval_npz),
            },
        )
        logger.info(
            "batch eval saved | samples=%d | all_miou=%s | base_miou=%s | novel_miou=%s",
            len(jobs),
            aggregate["all_miou"],
            aggregate["base_miou"],
            aggregate["novel_miou"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
