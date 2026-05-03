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
from ra_ov3dseg.models.point_mlp import PointMLP  # noqa: E402
from ra_ov3dseg.training.labels import build_class_split, map_labels_for_base_ce  # noqa: E402
from ra_ov3dseg.training.losses import cosine_distillation_loss, supervised_ce_loss  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


IGNORE_INDEX = -100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one MVP-v3 training dry-run step on one nuScenes sample.")
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes dataroot.")
    parser.add_argument("--version", default="v1.0-mini", type=str, help="nuScenes version.")
    parser.add_argument("--sample_idx", default=0, type=int, help="Sample index.")
    parser.add_argument(
        "--point_feature_npz",
        default=None,
        type=str,
        help="Point teacher feature .npz. Defaults to outputs/point_features/sample_XXXX_point_features.npz.",
    )
    parser.add_argument(
        "--reliability_npz",
        default=None,
        type=str,
        help="Reliability .npz. Defaults to outputs/reliability/sample_XXXX_reliability.npz.",
    )
    parser.add_argument("--point_feature_dir", default="outputs/point_features", type=str)
    parser.add_argument("--reliability_dir", default="outputs/reliability", type=str)
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--split_config", default="configs/base_novel_split.yaml", type=str)
    parser.add_argument("--device", default="cpu", type=str, help="cpu/cuda/auto.")
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--distill_weight", default=1.0, type=float)
    parser.add_argument("--ce_weight", default=1.0, type=float)
    parser.add_argument("--max_points", default=20000, type=int, help="Subsample points for CPU dry-run.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output_dir", default="outputs/training_dryrun", type=str)
    return parser


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def resolve_device(torch_module, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return requested


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


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("dry_run_training_step")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("dry_run_training_step requires PyTorch. Install torch before running MVP-v3.") from exc

    device = resolve_device(torch, args.device)
    output_dir = ensure_dir(args.output_dir)
    prefix = f"sample_{args.sample_idx:04d}"

    point_feature_npz = (
        Path(args.point_feature_npz).resolve()
        if args.point_feature_npz is not None
        else Path(args.point_feature_dir).resolve() / f"{prefix}_point_features.npz"
    )
    reliability_npz = (
        Path(args.reliability_npz).resolve()
        if args.reliability_npz is not None
        else Path(args.reliability_dir).resolve() / f"{prefix}_reliability.npz"
    )
    if not point_feature_npz.exists():
        raise FileNotFoundError(f"point feature npz not found: {point_feature_npz}")
    if not reliability_npz.exists():
        raise FileNotFoundError(f"reliability npz not found: {reliability_npz}")

    dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=False)
    sample = dataset.get_sample_by_index(args.sample_idx)
    raw_labels = dataset.load_lidarseg_labels(sample)
    if raw_labels is None:
        raise FileNotFoundError("lidarseg labels not found; MVP-v3 dry-run requires labels for CE loss.")

    point_data = load_npz(point_feature_npz)
    reliability_data = load_npz(reliability_npz)
    point_xyz = point_data["point_xyz"].astype(np.float32)
    teacher_features = point_data["point_features"].astype(np.float32)
    teacher_valid_mask = point_data["point_valid_mask"].astype(bool)
    reliability_weight = reliability_data["reliability_weight"].astype(np.float32)

    if raw_labels.shape[0] != point_xyz.shape[0]:
        raise ValueError(f"label/point count mismatch: labels={raw_labels.shape[0]}, points={point_xyz.shape[0]}")
    if teacher_features.shape[0] != point_xyz.shape[0]:
        raise ValueError("teacher feature count does not match point count.")
    if reliability_weight.shape[0] != point_xyz.shape[0]:
        raise ValueError("reliability count does not match point count.")

    class_split = build_class_split(args.class_names_path, args.split_config)
    train_labels = map_labels_for_base_ce(raw_labels, class_split, ignore_index=IGNORE_INDEX)

    selected = subsample_indices(point_xyz.shape[0], args.max_points, args.seed)
    point_xyz = point_xyz[selected]
    teacher_features = teacher_features[selected]
    teacher_valid_mask = teacher_valid_mask[selected]
    reliability_weight = reliability_weight[selected]
    raw_labels = raw_labels[selected]
    train_labels = train_labels[selected]

    torch.manual_seed(args.seed)
    point_xyz_t = torch.from_numpy(point_xyz).to(device)
    teacher_features_t = torch.from_numpy(teacher_features).to(device)
    train_labels_t = torch.from_numpy(train_labels).long().to(device)
    teacher_valid_mask_t = torch.from_numpy(teacher_valid_mask).bool().to(device)
    reliability_weight_t = torch.from_numpy(reliability_weight).float().to(device)

    model = PointMLP(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        feature_dim=teacher_features.shape[1],
        num_classes=class_split.num_train_classes,
    ).to(device)
    model.train()

    outputs = model(point_xyz_t)
    ce_loss = supervised_ce_loss(outputs["logits"], train_labels_t, ignore_index=IGNORE_INDEX)
    distill_loss = cosine_distillation_loss(
        student_features=outputs["point_features"],
        teacher_features=teacher_features_t,
        weights=reliability_weight_t,
        valid_mask=teacher_valid_mask_t,
    )
    total_loss = args.ce_weight * ce_loss + args.distill_weight * distill_loss

    total_loss.backward()
    grad_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm_sq += float(torch.sum(param.grad.detach() ** 2).cpu().item())
    grad_norm = grad_norm_sq ** 0.5

    base_supervised_points = int(np.sum(train_labels != IGNORE_INDEX))
    ignored_points = int(train_labels.shape[0] - base_supervised_points)
    distill_points = int(np.sum(teacher_valid_mask & np.isfinite(reliability_weight) & (reliability_weight > 0.0)))

    summary: dict[str, Any] = {
        "sample_idx": args.sample_idx,
        "sample_token": sample["token"],
        "device": device,
        "num_input_points": int(point_data["point_xyz"].shape[0]),
        "num_used_points": int(point_xyz.shape[0]),
        "feature_dim": int(teacher_features.shape[1]),
        "num_all_classes": int(class_split.num_classes),
        "num_base_train_classes": int(class_split.num_train_classes),
        "base_classes": class_split.base_class_names,
        "novel_classes": class_split.novel_class_names,
        "ignore_classes": class_split.ignore_class_names,
        "base_supervised_points": base_supervised_points,
        "ignored_points": ignored_points,
        "distill_points": distill_points,
        "raw_label_hist": label_hist(raw_labels, class_split.class_names),
        "loss": {
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "distill_loss": float(distill_loss.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
            "ce_weight": float(args.ce_weight),
            "distill_weight": float(args.distill_weight),
        },
        "grad_norm": grad_norm,
        "point_feature_npz": str(point_feature_npz),
        "reliability_npz": str(reliability_npz),
    }
    summary_path = output_dir / f"{prefix}_training_dryrun_summary.json"
    save_json(summary_path, summary)

    logger.info(
        "dry-run PASS | used=%d | base_ce_points=%d | distill_points=%d | total_loss=%.6f | grad_norm=%.6f",
        summary["num_used_points"],
        base_supervised_points,
        distill_points,
        summary["loss"]["total_loss"],
        grad_norm,
    )
    logger.info("summary saved to: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
