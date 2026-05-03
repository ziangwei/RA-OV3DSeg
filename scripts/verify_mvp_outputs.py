from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.utils.io import ensure_dir, load_json, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


EXPECTED_CAMERA_COUNT = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 MVP-v0/v1 已生成输出是否符合最小规格。")
    parser.add_argument("--sample_idx", default=0, type=int, help="要检查的 sample 索引。")
    parser.add_argument("--outputs_dir", default="outputs", type=str, help="outputs 根目录。")
    parser.add_argument(
        "--stage",
        default="v1",
        choices=["v0", "v1", "v2", "v3", "v4"],
        help=(
            "Stage to verify: v0 projection, v1 features/zero-shot, v2 reliability, "
            "v3 training dry-run, v4 train checkpoint."
        ),
    )
    parser.add_argument(
        "--min_projection_ratio",
        default=0.01,
        type=float,
        help="至少一个相机需要达到的有效投影比例下限。",
    )
    parser.add_argument(
        "--min_point_feature_ratio",
        default=0.01,
        type=float,
        help="3D 点中至少多少比例应获得 2D feature。",
    )
    parser.add_argument(
        "--expected_feature_dim",
        default=512,
        type=int,
        help="CLIP/SigLIP text-space feature 维度期望值。",
    )
    parser.add_argument(
        "--expected_num_classes",
        default=32,
        type=int,
        help="zero-shot 类别数量期望值。",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/verification",
        type=str,
        help="验证摘要输出目录。",
    )
    parser.add_argument(
        "--training_v4_dir",
        default=None,
        type=str,
        help="Optional V4 training output directory. Defaults to outputs_dir/training_v4.",
    )
    return parser


def check_file(path: Path, checks: list[dict[str, Any]], label: str, required: bool = True) -> bool:
    exists = path.exists()
    status = "pass" if exists else ("fail" if required else "warn")
    checks.append(
        {
            "name": label,
            "status": status,
            "path": str(path),
            "message": "exists" if exists else "missing",
        }
    )
    return exists


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, message: str, details: Any = None) -> None:
    item = {
        "name": name,
        "status": "pass" if passed else "fail",
        "message": message,
    }
    if details is not None:
        item["details"] = details
    checks.append(item)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def verify_projection(outputs_dir: Path, sample_idx: int, args, checks: list[dict[str, Any]]) -> None:
    prefix = f"sample_{sample_idx:04d}"
    projection_npz = outputs_dir / "projections" / f"{prefix}_projection.npz"
    projection_summary = outputs_dir / "projections" / f"{prefix}_projection_summary.json"

    if not check_file(projection_npz, checks, "projection_npz"):
        return
    check_file(projection_summary, checks, "projection_summary_json")

    data = load_npz(projection_npz)
    required_keys = [
        "point_xyz",
        "camera_names",
        "uv",
        "depth",
        "valid_masks",
        "visible_camera_count",
    ]
    missing_keys = [key for key in required_keys if key not in data]
    add_check(checks, "projection_required_keys", not missing_keys, f"missing_keys={missing_keys}", missing_keys)
    if missing_keys:
        return

    point_xyz = data["point_xyz"]
    camera_names = data["camera_names"]
    uv = data["uv"]
    depth = data["depth"]
    valid_masks = data["valid_masks"].astype(bool)
    visible_camera_count = data["visible_camera_count"]

    add_check(
        checks,
        "projection_camera_count",
        int(camera_names.shape[0]) == EXPECTED_CAMERA_COUNT,
        f"camera_count={int(camera_names.shape[0])}",
    )
    add_check(
        checks,
        "projection_shapes",
        (
            point_xyz.ndim == 2
            and point_xyz.shape[1] == 3
            and uv.shape == (EXPECTED_CAMERA_COUNT, point_xyz.shape[0], 2)
            and depth.shape == (EXPECTED_CAMERA_COUNT, point_xyz.shape[0])
            and valid_masks.shape == (EXPECTED_CAMERA_COUNT, point_xyz.shape[0])
            and visible_camera_count.shape[0] == point_xyz.shape[0]
        ),
        (
            f"point_xyz={point_xyz.shape}, uv={uv.shape}, depth={depth.shape}, "
            f"valid_masks={valid_masks.shape}, visible_camera_count={visible_camera_count.shape}"
        ),
    )

    valid_ratios = valid_masks.sum(axis=1) / max(point_xyz.shape[0], 1)
    max_ratio = float(valid_ratios.max()) if valid_ratios.shape[0] else 0.0
    add_check(
        checks,
        "projection_valid_ratio",
        max_ratio >= args.min_projection_ratio,
        f"max_camera_valid_ratio={max_ratio:.6f}",
        valid_ratios.astype(float).tolist(),
    )


def verify_overlays(outputs_dir: Path, sample_idx: int, checks: list[dict[str, Any]]) -> None:
    prefix = f"sample_{sample_idx:04d}"
    manifest = outputs_dir / "visualizations" / f"{prefix}_overlay_manifest.json"
    if not check_file(manifest, checks, "overlay_manifest_json", required=False):
        return

    manifest_data = load_json(manifest)
    outputs = manifest_data.get("outputs", [])
    near_paths = [Path(item.get("overlay_near_path", "")) for item in outputs]
    full_paths = [Path(item.get("overlay_full_path", "")) for item in outputs]
    existing_near = sum(path.exists() for path in near_paths)
    existing_full = sum(path.exists() for path in full_paths)
    add_check(
        checks,
        "overlay_images",
        existing_near > 0 and existing_full > 0,
        f"near={existing_near}, full={existing_full}",
    )


def verify_image_features(outputs_dir: Path, sample_idx: int, args, checks: list[dict[str, Any]]) -> None:
    prefix = f"sample_{sample_idx:04d}"
    feature_npz = outputs_dir / "features2d" / f"{prefix}_image_features.npz"
    feature_summary = outputs_dir / "features2d" / f"{prefix}_image_features_summary.json"

    if not check_file(feature_npz, checks, "image_feature_npz"):
        return
    check_file(feature_summary, checks, "image_feature_summary_json")

    data = load_npz(feature_npz)
    required_keys = [
        "feature_maps",
        "image_embeddings",
        "camera_names",
        "camera_available",
        "model_name",
    ]
    missing_keys = [key for key in required_keys if key not in data]
    add_check(checks, "image_feature_required_keys", not missing_keys, f"missing_keys={missing_keys}", missing_keys)
    if missing_keys:
        return

    feature_maps = data["feature_maps"]
    image_embeddings = data["image_embeddings"]
    camera_available = data["camera_available"].astype(bool)
    feature_dim = int(feature_maps.shape[-1]) if feature_maps.ndim == 4 else -1
    image_dim = int(image_embeddings.shape[-1]) if image_embeddings.ndim == 2 else -1

    add_check(
        checks,
        "image_feature_shapes",
        feature_maps.ndim == 4 and image_embeddings.ndim == 2 and feature_maps.shape[0] == EXPECTED_CAMERA_COUNT,
        f"feature_maps={feature_maps.shape}, image_embeddings={image_embeddings.shape}",
    )
    add_check(
        checks,
        "image_feature_dim",
        feature_dim == args.expected_feature_dim and image_dim == args.expected_feature_dim,
        f"feature_dim={feature_dim}, image_dim={image_dim}, expected={args.expected_feature_dim}",
    )
    add_check(
        checks,
        "image_feature_available_cameras",
        int(camera_available.sum()) == EXPECTED_CAMERA_COUNT,
        f"available_cameras={int(camera_available.sum())}",
    )


def verify_point_features(outputs_dir: Path, sample_idx: int, args, checks: list[dict[str, Any]]) -> None:
    prefix = f"sample_{sample_idx:04d}"
    point_feature_npz = outputs_dir / "point_features" / f"{prefix}_point_features.npz"
    point_feature_summary = outputs_dir / "point_features" / f"{prefix}_point_features_summary.json"

    if not check_file(point_feature_npz, checks, "point_feature_npz"):
        return
    check_file(point_feature_summary, checks, "point_feature_summary_json")

    data = load_npz(point_feature_npz)
    required_keys = ["point_xyz", "point_features", "point_valid_mask", "point_camera_count", "model_name"]
    missing_keys = [key for key in required_keys if key not in data]
    add_check(checks, "point_feature_required_keys", not missing_keys, f"missing_keys={missing_keys}", missing_keys)
    if missing_keys:
        return

    point_xyz = data["point_xyz"]
    point_features = data["point_features"]
    valid_mask = data["point_valid_mask"].astype(bool)
    valid_ratio = float(valid_mask.sum() / max(point_xyz.shape[0], 1))

    add_check(
        checks,
        "point_feature_shapes",
        (
            point_xyz.ndim == 2
            and point_xyz.shape[1] == 3
            and point_features.ndim == 2
            and point_features.shape[0] == point_xyz.shape[0]
        ),
        f"point_xyz={point_xyz.shape}, point_features={point_features.shape}",
    )
    add_check(
        checks,
        "point_feature_dim",
        int(point_features.shape[-1]) == args.expected_feature_dim,
        f"point_feature_dim={int(point_features.shape[-1])}, expected={args.expected_feature_dim}",
    )
    add_check(
        checks,
        "point_feature_valid_ratio",
        valid_ratio >= args.min_point_feature_ratio,
        f"valid_ratio={valid_ratio:.6f}, valid_points={int(valid_mask.sum())}, total={int(point_xyz.shape[0])}",
    )


def verify_zero_shot(outputs_dir: Path, sample_idx: int, args, checks: list[dict[str, Any]]) -> None:
    prefix = f"sample_{sample_idx:04d}"
    zero_npz = outputs_dir / "zero_shot" / f"{prefix}_zero_shot_predictions.npz"
    zero_summary = outputs_dir / "zero_shot" / f"{prefix}_zero_shot_summary.json"
    bev_path = outputs_dir / "zero_shot" / f"{prefix}_zero_shot_bev.png"
    ply_path = outputs_dir / "zero_shot" / f"{prefix}_zero_shot_points.ply"

    if not check_file(zero_npz, checks, "zero_shot_npz"):
        return
    check_file(zero_summary, checks, "zero_shot_summary_json")
    check_file(bev_path, checks, "zero_shot_bev_png")
    check_file(ply_path, checks, "zero_shot_points_ply")

    data = load_npz(zero_npz)
    required_keys = ["pred_label_indices", "pred_scores", "class_names", "point_valid_mask", "text_embeddings"]
    missing_keys = [key for key in required_keys if key not in data]
    add_check(checks, "zero_shot_required_keys", not missing_keys, f"missing_keys={missing_keys}", missing_keys)
    if missing_keys:
        return

    class_names = data["class_names"]
    pred_label_indices = data["pred_label_indices"]
    valid_mask = data["point_valid_mask"].astype(bool)
    valid_pred_mask = valid_mask & (pred_label_indices >= 0)
    unique_pred_count = int(np.unique(pred_label_indices[valid_pred_mask]).shape[0]) if np.any(valid_pred_mask) else 0

    add_check(
        checks,
        "zero_shot_num_classes",
        int(class_names.shape[0]) == args.expected_num_classes,
        f"num_classes={int(class_names.shape[0])}, expected={args.expected_num_classes}",
    )
    add_check(
        checks,
        "zero_shot_valid_predictions",
        int(valid_pred_mask.sum()) > 0 and unique_pred_count > 0,
        f"valid_predictions={int(valid_pred_mask.sum())}, unique_pred_classes={unique_pred_count}",
    )

    if zero_summary.exists():
        summary = load_json(zero_summary)
        class_hist = summary.get("class_hist", {})
        add_check(
            checks,
            "zero_shot_class_hist",
            len(class_hist) > 0,
            f"class_hist_entries={len(class_hist)}",
            class_hist,
        )


def verify_reliability(outputs_dir: Path, sample_idx: int, checks: list[dict[str, Any]]) -> None:
    prefix = f"sample_{sample_idx:04d}"
    reliability_npz = outputs_dir / "reliability" / f"{prefix}_reliability.npz"
    reliability_summary = outputs_dir / "reliability" / f"{prefix}_reliability_summary.json"
    bev_path = outputs_dir / "reliability" / f"{prefix}_reliability_bev.png"
    ply_path = outputs_dir / "reliability" / f"{prefix}_reliability_points.ply"

    if not check_file(reliability_npz, checks, "reliability_npz"):
        return
    check_file(reliability_summary, checks, "reliability_summary_json")
    check_file(bev_path, checks, "reliability_bev_png")
    check_file(ply_path, checks, "reliability_points_ply")

    data = load_npz(reliability_npz)
    required_keys = [
        "point_xyz",
        "point_valid_mask",
        "distance_weight",
        "geometric_weight",
        "semantic_weight",
        "reliability_weight",
    ]
    missing_keys = [key for key in required_keys if key not in data]
    add_check(checks, "reliability_required_keys", not missing_keys, f"missing_keys={missing_keys}", missing_keys)
    if missing_keys:
        return

    point_xyz = data["point_xyz"]
    valid_mask = data["point_valid_mask"].astype(bool)
    reliability_weight = data["reliability_weight"].astype(np.float32)
    shape_ok = reliability_weight.shape[0] == point_xyz.shape[0]
    finite_valid = np.isfinite(reliability_weight[valid_mask]).all() if np.any(valid_mask) else False
    in_range = bool(np.all((reliability_weight >= -1e-6) & (reliability_weight <= 1.0 + 1e-6)))
    nonzero_valid = int(np.sum(reliability_weight[valid_mask] > 0.0)) if np.any(valid_mask) else 0

    add_check(
        checks,
        "reliability_shapes",
        shape_ok,
        f"point_xyz={point_xyz.shape}, reliability_weight={reliability_weight.shape}",
    )
    add_check(
        checks,
        "reliability_finite_and_range",
        finite_valid and in_range,
        f"finite_valid={finite_valid}, in_range={in_range}",
    )
    add_check(
        checks,
        "reliability_nonzero_valid",
        nonzero_valid > 0,
        f"nonzero_valid={nonzero_valid}, valid_points={int(valid_mask.sum())}",
    )


def verify_training_dryrun(outputs_dir: Path, sample_idx: int, checks: list[dict[str, Any]]) -> None:
    prefix = f"sample_{sample_idx:04d}"
    summary_path = outputs_dir / "training_dryrun" / f"{prefix}_training_dryrun_summary.json"

    if not check_file(summary_path, checks, "training_dryrun_summary_json"):
        return

    summary = load_json(summary_path)
    loss = summary.get("loss", {})
    total_loss = float(loss.get("total_loss", float("nan")))
    ce_loss = float(loss.get("ce_loss", float("nan")))
    distill_loss = float(loss.get("distill_loss", float("nan")))
    grad_norm = float(summary.get("grad_norm", float("nan")))
    base_points = int(summary.get("base_supervised_points", 0))
    distill_points = int(summary.get("distill_points", 0))

    add_check(
        checks,
        "training_dryrun_losses",
        np.isfinite(total_loss) and np.isfinite(ce_loss) and np.isfinite(distill_loss) and total_loss >= 0.0,
        f"ce={ce_loss:.6f}, distill={distill_loss:.6f}, total={total_loss:.6f}",
    )
    add_check(
        checks,
        "training_dryrun_grad",
        np.isfinite(grad_norm) and grad_norm > 0.0,
        f"grad_norm={grad_norm:.6f}",
    )
    add_check(
        checks,
        "training_dryrun_supervision",
        base_points > 0 and distill_points > 0,
        f"base_supervised_points={base_points}, distill_points={distill_points}",
    )


def verify_training_v4(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    training_dir = Path(args.training_v4_dir) if args.training_v4_dir is not None else outputs_dir / "training_v4"
    summary_path = training_dir / "train_summary.json"
    default_latest = training_dir / "point_mlp_latest.pt"

    if not check_file(summary_path, checks, "training_v4_summary_json"):
        return

    summary = load_json(summary_path)
    latest_path = Path(summary.get("latest_checkpoint", str(default_latest)))
    check_file(latest_path, checks, "training_v4_latest_checkpoint")

    epoch_logs = summary.get("epoch_logs", [])
    final_log = epoch_logs[-1] if epoch_logs else {}
    avg_total_loss = float(final_log.get("avg_total_loss", float("nan")))
    avg_ce_loss = float(final_log.get("avg_ce_loss", float("nan")))
    avg_distill_loss = float(final_log.get("avg_distill_loss", float("nan")))
    avg_grad_norm = float(final_log.get("avg_grad_norm", float("nan")))
    epochs_completed = int(summary.get("epochs_completed", 0))
    num_samples = int(summary.get("num_samples", 0))
    points = int(final_log.get("points", 0))
    base_points = int(final_log.get("base_supervised_points", 0))
    distill_points = int(final_log.get("distill_points", 0))

    add_check(
        checks,
        "training_v4_status",
        summary.get("status") == "pass" and epochs_completed > 0 and num_samples > 0,
        f"status={summary.get('status')}, epochs={epochs_completed}, samples={num_samples}",
    )
    add_check(
        checks,
        "training_v4_losses",
        (
            np.isfinite(avg_total_loss)
            and np.isfinite(avg_ce_loss)
            and np.isfinite(avg_distill_loss)
            and avg_total_loss >= 0.0
        ),
        f"ce={avg_ce_loss:.6f}, distill={avg_distill_loss:.6f}, total={avg_total_loss:.6f}",
    )
    add_check(
        checks,
        "training_v4_grad",
        np.isfinite(avg_grad_norm) and avg_grad_norm > 0.0,
        f"avg_grad_norm={avg_grad_norm:.6f}",
    )
    add_check(
        checks,
        "training_v4_supervision",
        points > 0 and base_points > 0 and distill_points > 0,
        f"points={points}, base_supervised_points={base_points}, distill_points={distill_points}",
    )


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("verify_mvp_outputs")
    outputs_dir = Path(args.outputs_dir)
    output_dir = ensure_dir(args.output_dir)
    checks: list[dict[str, Any]] = []

    verify_projection(outputs_dir, args.sample_idx, args, checks)
    verify_overlays(outputs_dir, args.sample_idx, checks)
    if args.stage in {"v1", "v2", "v3", "v4"}:
        verify_image_features(outputs_dir, args.sample_idx, args, checks)
        verify_point_features(outputs_dir, args.sample_idx, args, checks)
        verify_zero_shot(outputs_dir, args.sample_idx, args, checks)
    if args.stage in {"v2", "v3", "v4"}:
        verify_reliability(outputs_dir, args.sample_idx, checks)
    if args.stage == "v3":
        verify_training_dryrun(outputs_dir, args.sample_idx, checks)
    if args.stage == "v4":
        verify_training_v4(outputs_dir, args, checks)

    failed = [check for check in checks if check["status"] == "fail"]
    warned = [check for check in checks if check["status"] == "warn"]
    passed = [check for check in checks if check["status"] == "pass"]
    summary = {
        "sample_idx": args.sample_idx,
        "stage": args.stage,
        "outputs_dir": str(outputs_dir.resolve()),
        "num_passed": len(passed),
        "num_warned": len(warned),
        "num_failed": len(failed),
        "status": "pass" if not failed else "fail",
        "checks": checks,
    }
    summary_path = output_dir / f"sample_{args.sample_idx:04d}_mvp_verify_summary.json"
    save_json(summary_path, summary)

    logger.info(
        "MVP verification %s | passed=%d warned=%d failed=%d | summary=%s",
        summary["status"].upper(),
        len(passed),
        len(warned),
        len(failed),
        summary_path,
    )
    for check in checks:
        level = logger.info if check["status"] == "pass" else logger.warning
        if check["status"] == "fail":
            level = logger.error
        level("[%s] %s | %s", check["status"].upper(), check["name"], check["message"])

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
