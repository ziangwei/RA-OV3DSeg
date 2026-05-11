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
        choices=[
            "v0",
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "v7",
            "v8",
            "v9",
            "v10",
            "v11",
            "v12",
            "v13",
            "v14",
            "v15",
            "v16a",
        ],
        help=(
            "Stage to verify: v0 projection, v1 features/zero-shot, v2 reliability, "
            "v3 training dry-run, v4 train checkpoint, v5 sparse backbone checkpoint, "
            "v6 dense teacher logits, v7 dense-logit distillation training, v8 3D prediction/eval, "
            "v9 mini experiment protocol, v10 open-vocabulary inference/eval, "
            "v11 text-aligned 3D embedding training/eval, "
            "v12 GroupViT dense teacher training/eval, "
            "v13 teacher-quality and supervised-backbone diagnostics, "
            "v14 improved supervised 3D recipe, "
            "v15 cylinder-grid supervised 3D baseline, "
            "v16a official-16 cylinder baseline."
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
    parser.add_argument(
        "--training_v5_dir",
        default=None,
        type=str,
        help="Optional V5 training output directory. Defaults to outputs_dir/training_v5.",
    )
    parser.add_argument(
        "--training_v7_dir",
        default=None,
        type=str,
        help="Optional V7 training output directory. Defaults to outputs_dir/training_v7.",
    )
    parser.add_argument(
        "--dense_teacher_dir",
        default=None,
        type=str,
        help="Optional V6 dense teacher logit directory. Defaults to outputs_dir/dense_teacher_logits.",
    )
    parser.add_argument(
        "--dense_point_dir",
        default=None,
        type=str,
        help="Optional V6 dense point logit directory. Defaults to outputs_dir/dense_point_logits.",
    )
    parser.add_argument(
        "--prediction_dir",
        default=None,
        type=str,
        help="Optional V8 prediction directory. Defaults to outputs_dir/predictions3d.",
    )
    parser.add_argument(
        "--evaluation_dir",
        default=None,
        type=str,
        help="Optional V8 evaluation directory. Defaults to outputs_dir/evaluation3d.",
    )
    parser.add_argument(
        "--experiment_dir",
        default=None,
        type=str,
        help="Optional V9 experiment directory. Defaults to outputs_dir/experiments/mini_v9.",
    )
    parser.add_argument(
        "--open_vocab_prediction_dir",
        default=None,
        type=str,
        help="Optional V10 open-vocabulary prediction directory.",
    )
    parser.add_argument(
        "--open_vocab_evaluation_dir",
        default=None,
        type=str,
        help="Optional V10 open-vocabulary evaluation directory.",
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


def verify_training_v5(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    training_dir = Path(args.training_v5_dir) if args.training_v5_dir is not None else outputs_dir / "training_v5"
    summary_path = training_dir / "train_summary.json"

    if not check_file(summary_path, checks, "training_v5_summary_json"):
        return

    summary = load_json(summary_path)
    latest_path = Path(summary.get("latest_checkpoint", str(training_dir / "sparse_unet_spconv_latest.pt")))
    check_file(latest_path, checks, "training_v5_latest_checkpoint")

    backbone = summary.get("backbone", {})
    backbone_name = backbone.get("backbone", "")
    epoch_logs = summary.get("epoch_logs", [])
    final_log = epoch_logs[-1] if epoch_logs else {}
    avg_total_loss = float(final_log.get("avg_total_loss", float("nan")))
    avg_ce_loss = float(final_log.get("avg_ce_loss", float("nan")))
    avg_distill_loss = float(final_log.get("avg_distill_loss", float("nan")))
    avg_grad_norm = float(final_log.get("avg_grad_norm", float("nan")))
    epochs_completed = int(summary.get("epochs_completed", 0))
    points = int(final_log.get("points", 0))
    base_points = int(final_log.get("base_supervised_points", 0))
    distill_points = int(final_log.get("distill_points", 0))

    add_check(
        checks,
        "training_v5_backbone",
        backbone_name == "sparse_unet_spconv",
        f"backbone={backbone_name}",
    )
    add_check(
        checks,
        "training_v5_status",
        summary.get("status") == "pass" and epochs_completed > 0,
        f"status={summary.get('status')}, epochs={epochs_completed}",
    )
    add_check(
        checks,
        "training_v5_losses",
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
        "training_v5_grad",
        np.isfinite(avg_grad_norm) and avg_grad_norm > 0.0,
        f"avg_grad_norm={avg_grad_norm:.6f}",
    )
    add_check(
        checks,
        "training_v5_supervision",
        points > 0 and base_points > 0 and distill_points > 0,
        f"points={points}, base_supervised_points={base_points}, distill_points={distill_points}",
    )


def verify_training_v7(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    training_dir = Path(args.training_v7_dir) if args.training_v7_dir is not None else outputs_dir / "training_v7"
    summary_path = training_dir / "train_summary.json"

    if not check_file(summary_path, checks, "training_v7_summary_json"):
        return

    summary = load_json(summary_path)
    latest_path = Path(summary.get("latest_checkpoint", str(training_dir / "sparse_unet_spconv_latest.pt")))
    check_file(latest_path, checks, "training_v7_latest_checkpoint")

    backbone = summary.get("backbone", {})
    backbone_name = backbone.get("backbone", "")
    teacher_mode = str(summary.get("teacher_mode", ""))
    student_output_space = str(summary.get("student_output_space", ""))
    num_output_classes = int(summary.get("num_output_classes", 0))
    epoch_logs = summary.get("epoch_logs", [])
    final_log = epoch_logs[-1] if epoch_logs else {}
    avg_total_loss = float(final_log.get("avg_total_loss", float("nan")))
    avg_ce_loss = float(final_log.get("avg_ce_loss", float("nan")))
    avg_dense_loss = float(final_log.get("avg_dense_logit_loss", float("nan")))
    avg_grad_norm = float(final_log.get("avg_grad_norm", float("nan")))
    epochs_completed = int(summary.get("epochs_completed", 0))
    points = int(final_log.get("points", 0))
    base_points = int(final_log.get("base_supervised_points", 0))
    dense_points = int(final_log.get("dense_distill_points", 0))

    add_check(
        checks,
        "training_v7_backbone",
        backbone_name == "sparse_unet_spconv",
        f"backbone={backbone_name}",
    )
    add_check(
        checks,
        "training_v7_teacher_mode",
        teacher_mode in {"dense_logit_distill", "hybrid"},
        f"teacher_mode={teacher_mode}",
    )
    add_check(
        checks,
        "training_v7_output_space",
        student_output_space == "all_lidarseg" and num_output_classes == 32,
        f"student_output_space={student_output_space}, num_output_classes={num_output_classes}",
    )
    add_check(
        checks,
        "training_v7_status",
        summary.get("status") == "pass" and epochs_completed > 0,
        f"status={summary.get('status')}, epochs={epochs_completed}",
    )
    add_check(
        checks,
        "training_v7_losses",
        (
            np.isfinite(avg_total_loss)
            and np.isfinite(avg_ce_loss)
            and np.isfinite(avg_dense_loss)
            and avg_total_loss >= 0.0
            and avg_dense_loss >= 0.0
        ),
        f"ce={avg_ce_loss:.6f}, dense_logit={avg_dense_loss:.6f}, total={avg_total_loss:.6f}",
    )
    add_check(
        checks,
        "training_v7_grad",
        np.isfinite(avg_grad_norm) and avg_grad_norm > 0.0,
        f"avg_grad_norm={avg_grad_norm:.6f}",
    )
    add_check(
        checks,
        "training_v7_supervision",
        points > 0 and base_points > 0 and dense_points > 0,
        f"points={points}, base_supervised_points={base_points}, dense_distill_points={dense_points}",
    )


def verify_dense_teacher_v6(outputs_dir: Path, sample_idx: int, args, checks: list[dict[str, Any]]) -> None:
    prefix = f"sample_{sample_idx:04d}"
    dense_teacher_dir = Path(args.dense_teacher_dir) if args.dense_teacher_dir is not None else outputs_dir / "dense_teacher_logits"
    dense_point_dir = Path(args.dense_point_dir) if args.dense_point_dir is not None else outputs_dir / "dense_point_logits"
    dense_npz = dense_teacher_dir / f"{prefix}_dense_teacher_logits.npz"
    dense_summary = dense_teacher_dir / f"{prefix}_dense_teacher_logits_summary.json"
    point_npz = dense_point_dir / f"{prefix}_dense_point_logits.npz"
    point_summary = dense_point_dir / f"{prefix}_dense_point_logits_summary.json"

    if not check_file(dense_npz, checks, "dense_teacher_npz"):
        return
    check_file(dense_summary, checks, "dense_teacher_summary_json")
    dense = load_npz(dense_npz)
    dense_required = ["dense_logits", "camera_names", "camera_available", "class_names", "teacher_backend", "model_name"]
    missing_dense = [key for key in dense_required if key not in dense]
    add_check(checks, "dense_teacher_required_keys", not missing_dense, f"missing_keys={missing_dense}", missing_dense)
    if missing_dense:
        return

    dense_logits = dense["dense_logits"]
    camera_available = dense["camera_available"].astype(bool)
    class_names = dense["class_names"]
    teacher_backend = str(dense["teacher_backend"].item())
    dense_shape_ok = (
        dense_logits.ndim == 4
        and dense_logits.shape[0] == EXPECTED_CAMERA_COUNT
        and dense_logits.shape[1] == class_names.shape[0]
        and dense_logits.shape[2] > 0
        and dense_logits.shape[3] > 0
    )
    add_check(
        checks,
        "dense_teacher_shapes",
        dense_shape_ok,
        f"dense_logits={dense_logits.shape}, class_names={class_names.shape}, backend={teacher_backend}",
    )
    add_check(
        checks,
        "dense_teacher_available_cameras",
        int(camera_available.sum()) == EXPECTED_CAMERA_COUNT,
        f"available_cameras={int(camera_available.sum())}",
    )
    add_check(
        checks,
        "dense_teacher_backend",
        teacher_backend == "clipseg_dense",
        f"teacher_backend={teacher_backend}",
    )

    if not check_file(point_npz, checks, "dense_point_npz"):
        return
    check_file(point_summary, checks, "dense_point_summary_json")
    point = load_npz(point_npz)
    point_required = [
        "point_xyz",
        "point_teacher_logits",
        "point_dense_valid_mask",
        "point_dense_pred_label_indices",
        "class_names",
    ]
    missing_point = [key for key in point_required if key not in point]
    add_check(checks, "dense_point_required_keys", not missing_point, f"missing_keys={missing_point}", missing_point)
    if missing_point:
        return

    point_xyz = point["point_xyz"]
    point_logits = point["point_teacher_logits"]
    point_valid = point["point_dense_valid_mask"].astype(bool)
    point_classes = point["class_names"]
    valid_ratio = float(point_valid.sum() / max(point_xyz.shape[0], 1))
    add_check(
        checks,
        "dense_point_shapes",
        (
            point_xyz.ndim == 2
            and point_xyz.shape[1] == 3
            and point_logits.ndim == 2
            and point_logits.shape[0] == point_xyz.shape[0]
            and point_logits.shape[1] == point_classes.shape[0]
        ),
        f"point_xyz={point_xyz.shape}, point_teacher_logits={point_logits.shape}, class_names={point_classes.shape}",
    )
    add_check(
        checks,
        "dense_point_valid_ratio",
        valid_ratio >= args.min_point_feature_ratio,
        f"valid_ratio={valid_ratio:.6f}, valid_points={int(point_valid.sum())}, total={int(point_xyz.shape[0])}",
    )


def verify_prediction_eval_v8(outputs_dir: Path, sample_idx: int, args, checks: list[dict[str, Any]]) -> None:
    prefix = f"sample_{sample_idx:04d}"
    prediction_dir = Path(args.prediction_dir) if args.prediction_dir is not None else outputs_dir / "predictions3d"
    evaluation_dir = Path(args.evaluation_dir) if args.evaluation_dir is not None else outputs_dir / "evaluation3d"

    pred_npz = prediction_dir / f"{prefix}_3d_predictions.npz"
    pred_summary = prediction_dir / f"{prefix}_3d_prediction_summary.json"
    pred_ply = prediction_dir / f"{prefix}_3d_predictions.ply"
    pred_bev = prediction_dir / f"{prefix}_3d_predictions_bev.png"
    eval_npz = evaluation_dir / f"{prefix}_3d_eval.npz"
    eval_summary = evaluation_dir / f"{prefix}_3d_eval_summary.json"

    if not check_file(pred_npz, checks, "prediction_npz"):
        return
    check_file(pred_summary, checks, "prediction_summary_json")
    check_file(pred_ply, checks, "prediction_points_ply")
    check_file(pred_bev, checks, "prediction_bev_png")

    pred = load_npz(pred_npz)
    pred_required = [
        "point_xyz",
        "model_valid_mask",
        "pred_output_indices",
        "pred_label_indices",
        "pred_scores",
        "class_names",
        "checkpoint_path",
        "student_output_space",
    ]
    missing_pred = [key for key in pred_required if key not in pred]
    add_check(checks, "prediction_required_keys", not missing_pred, f"missing_keys={missing_pred}", missing_pred)
    if missing_pred:
        return

    point_xyz = pred["point_xyz"]
    pred_labels = pred["pred_label_indices"]
    pred_scores = pred["pred_scores"]
    model_valid = pred["model_valid_mask"].astype(bool)
    class_names = pred["class_names"]
    valid_pred = model_valid & (pred_labels >= 0)
    output_space = str(pred["student_output_space"].item())

    add_check(
        checks,
        "prediction_shapes",
        (
            point_xyz.ndim == 2
            and point_xyz.shape[1] == 3
            and pred_labels.shape[0] == point_xyz.shape[0]
            and pred_scores.shape[0] == point_xyz.shape[0]
            and model_valid.shape[0] == point_xyz.shape[0]
        ),
        (
            f"point_xyz={point_xyz.shape}, pred_label_indices={pred_labels.shape}, "
            f"pred_scores={pred_scores.shape}, model_valid_mask={model_valid.shape}"
        ),
    )
    add_check(
        checks,
        "prediction_output_space",
        output_space in {"all_lidarseg", "base"},
        f"student_output_space={output_space}",
    )
    add_check(
        checks,
        "prediction_valid_ratio",
        int(valid_pred.sum()) > 0,
        f"valid_predictions={int(valid_pred.sum())}, total={int(point_xyz.shape[0])}, classes={int(class_names.shape[0])}",
    )

    if not check_file(eval_npz, checks, "evaluation_npz"):
        return
    check_file(eval_summary, checks, "evaluation_summary_json")
    eval_data = load_npz(eval_npz)
    eval_required = [
        "class_names",
        "intersections",
        "unions",
        "gt_counts",
        "pred_counts",
        "ious",
        "confusion_matrix",
        "base_label_ids",
        "novel_label_ids",
        "ignore_label_ids",
    ]
    missing_eval = [key for key in eval_required if key not in eval_data]
    add_check(checks, "evaluation_required_keys", not missing_eval, f"missing_keys={missing_eval}", missing_eval)
    if missing_eval:
        return

    ious = eval_data["ious"].astype(np.float32)
    unions = eval_data["unions"].astype(np.int64)
    confusion = eval_data["confusion_matrix"]
    present_iou = ious[np.isfinite(ious) & (unions > 0)]
    add_check(
        checks,
        "evaluation_shapes",
        (
            ious.shape[0] == class_names.shape[0]
            and unions.shape[0] == class_names.shape[0]
            and confusion.shape == (class_names.shape[0], class_names.shape[0])
        ),
        f"ious={ious.shape}, unions={unions.shape}, confusion_matrix={confusion.shape}",
    )
    add_check(
        checks,
        "evaluation_iou_values",
        present_iou.shape[0] > 0 and bool(np.all((present_iou >= 0.0) & (present_iou <= 1.0))),
        f"present_iou_classes={int(present_iou.shape[0])}",
    )
    if eval_summary.exists():
        summary = load_json(eval_summary)
        metrics = summary.get("metrics", {})
        add_check(
            checks,
            "evaluation_metrics_json",
            "all_miou" in metrics and "base_miou" in metrics and "novel_miou" in metrics,
            f"metrics_keys={sorted(metrics.keys())}",
            metrics,
        )


def verify_mini_experiment_v9(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir is not None else outputs_dir / "experiments" / "mini_v9"
    summary_path = experiment_dir / "summary.json"
    if not check_file(summary_path, checks, "v9_experiment_summary_json"):
        return

    summary = load_json(summary_path)
    status = str(summary.get("status", ""))
    commands = summary.get("commands", [])
    failed_commands = [item for item in commands if item.get("status") == "failed"]
    done_or_dry = [item for item in commands if item.get("status") in {"done", "dry_run"}]
    add_check(
        checks,
        "v9_experiment_status",
        status in {"pass", "dry_run"} and not failed_commands,
        f"status={status}, commands={len(commands)}, failed={len(failed_commands)}",
    )
    add_check(
        checks,
        "v9_experiment_commands",
        len(commands) > 0 and len(done_or_dry) == len(commands),
        f"commands={len(commands)}, completed_or_dry={len(done_or_dry)}",
    )

    artifact_dirs = summary.get("artifact_dirs", {})
    training = summary.get("training", {})
    latest_checkpoint = Path(str(training.get("latest_checkpoint", ""))) if training.get("latest_checkpoint") else None
    if status == "pass":
        if latest_checkpoint is not None:
            check_file(latest_checkpoint, checks, "v9_latest_checkpoint")
        add_check(
            checks,
            "v9_training_summary",
            training.get("status") == "pass" and int(training.get("epochs_completed", 0)) > 0,
            f"training_status={training.get('status')}, epochs={training.get('epochs_completed')}",
        )

    evaluation = summary.get("evaluation", {})
    aggregate_metrics = evaluation.get("aggregate_metrics", {})
    metric_keys = {"all_miou", "base_miou", "novel_miou", "prediction_coverage"}
    add_check(
        checks,
        "v9_aggregate_metrics",
        status == "dry_run" or metric_keys.issubset(set(aggregate_metrics.keys())),
        f"metrics_keys={sorted(aggregate_metrics.keys())}",
        aggregate_metrics,
    )
    if status == "pass":
        eval_source = evaluation.get("source", "")
        add_check(
            checks,
            "v9_evaluation_source",
            bool(eval_source) and Path(eval_source).exists(),
            f"evaluation_source={eval_source}",
        )
        for key in ["experiment", "training", "predictions", "evaluation"]:
            path = Path(str(artifact_dirs.get(key, "")))
            add_check(
                checks,
                f"v9_artifact_dir_{key}",
                bool(str(path)) and path.exists(),
                f"path={path}",
            )


def verify_open_vocab_v10(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    if args.stage == "v12":
        default_experiment_name = "trainval_v12_groupvit_128"
    elif args.stage == "v11":
        default_experiment_name = "trainval_v11_text_align_128"
    else:
        default_experiment_name = "trainval_v10_open_vocab_128"
    experiment_dir = (
        Path(args.experiment_dir)
        if args.experiment_dir is not None
        else outputs_dir / "experiments" / default_experiment_name
    )
    prediction_dir = (
        Path(args.open_vocab_prediction_dir)
        if args.open_vocab_prediction_dir is not None
        else experiment_dir / "open_vocab_predictions3d"
    )
    evaluation_dir = (
        Path(args.open_vocab_evaluation_dir)
        if args.open_vocab_evaluation_dir is not None
        else experiment_dir / "open_vocab_evaluation3d"
    )
    prefix = f"sample_{args.sample_idx:04d}"
    pred_npz = prediction_dir / f"{prefix}_open_vocab_predictions.npz"
    pred_summary = prediction_dir / f"{prefix}_open_vocab_prediction_summary.json"
    pred_ply = prediction_dir / f"{prefix}_open_vocab_predictions.ply"
    pred_bev = prediction_dir / f"{prefix}_open_vocab_predictions_bev.png"
    batch_pred_summary = prediction_dir / "batch_open_vocab_prediction_summary.json"
    eval_npz = evaluation_dir / f"{prefix}_3d_eval.npz"
    eval_summary = evaluation_dir / f"{prefix}_3d_eval_summary.json"
    batch_eval_summary = evaluation_dir / "batch_3d_eval_summary.json"

    if not check_file(pred_npz, checks, "open_vocab_prediction_npz"):
        return
    check_file(pred_summary, checks, "open_vocab_prediction_summary_json")
    check_file(pred_ply, checks, "open_vocab_prediction_points_ply", required=False)
    check_file(pred_bev, checks, "open_vocab_prediction_bev_png", required=False)
    check_file(batch_pred_summary, checks, "open_vocab_batch_prediction_summary_json", required=False)

    pred = load_npz(pred_npz)
    pred_required = [
        "point_xyz",
        "model_valid_mask",
        "pred_query_indices",
        "pred_label_indices",
        "pred_scores",
        "query_class_names",
        "text_embeddings",
    ]
    missing_pred = [key for key in pred_required if key not in pred]
    add_check(checks, "open_vocab_prediction_required_keys", not missing_pred, f"missing_keys={missing_pred}", missing_pred)
    if missing_pred:
        return

    point_xyz = pred["point_xyz"]
    point_embeddings = pred["point_embeddings"] if "point_embeddings" in pred else None
    text_embeddings = pred["text_embeddings"]
    pred_query_indices = pred["pred_query_indices"]
    pred_label_indices = pred["pred_label_indices"]
    pred_scores = pred["pred_scores"]
    model_valid = pred["model_valid_mask"].astype(bool)
    query_class_names = pred["query_class_names"]
    add_check(
        checks,
        "open_vocab_prediction_shapes",
        (
            point_xyz.ndim == 2
            and point_xyz.shape[1] == 3
            and text_embeddings.ndim == 2
            and pred_query_indices.shape[0] == point_xyz.shape[0]
            and pred_label_indices.shape[0] == point_xyz.shape[0]
            and pred_scores.shape[0] == point_xyz.shape[0]
            and (
                point_embeddings is None
                or (
                    point_embeddings.ndim == 2
                    and point_embeddings.shape[0] == point_xyz.shape[0]
                    and point_embeddings.shape[1] == text_embeddings.shape[1]
                )
            )
        ),
        (
            f"point_xyz={point_xyz.shape}, "
            f"point_embeddings={None if point_embeddings is None else point_embeddings.shape}, "
            f"text_embeddings={text_embeddings.shape}, query_classes={query_class_names.shape}"
        ),
    )
    valid_predictions = model_valid & (pred_query_indices >= 0)
    mapped_predictions = model_valid & (pred_label_indices >= 0)
    add_check(
        checks,
        "open_vocab_valid_predictions",
        int(valid_predictions.sum()) > 0,
        f"valid_query_predictions={int(valid_predictions.sum())}, total={int(point_xyz.shape[0])}",
    )
    add_check(
        checks,
        "open_vocab_mapped_predictions",
        int(mapped_predictions.sum()) > 0,
        f"mapped_lidarseg_predictions={int(mapped_predictions.sum())}, total={int(point_xyz.shape[0])}",
    )

    if check_file(eval_npz, checks, "open_vocab_evaluation_npz", required=False):
        check_file(eval_summary, checks, "open_vocab_evaluation_summary_json", required=False)
        eval_data = load_npz(eval_npz)
        eval_required = ["class_names", "intersections", "unions", "ious", "confusion_matrix"]
        missing_eval = [key for key in eval_required if key not in eval_data]
        add_check(checks, "open_vocab_evaluation_required_keys", not missing_eval, f"missing_keys={missing_eval}", missing_eval)
    check_file(batch_eval_summary, checks, "open_vocab_batch_evaluation_summary_json", required=False)
    if batch_eval_summary.exists():
        batch_summary = load_json(batch_eval_summary)
        aggregate = batch_summary.get("aggregate_metrics", {})
        metric_keys = {"all_miou", "base_miou", "novel_miou", "prediction_coverage"}
        add_check(
            checks,
            "open_vocab_aggregate_metrics",
            metric_keys.issubset(set(aggregate.keys())),
            f"metrics_keys={sorted(aggregate.keys())}",
            aggregate,
        )


def verify_text_aligned_training_v11(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    default_experiment_name = (
        "trainval_v12_groupvit_128" if args.stage == "v12" else "trainval_v11_text_align_128"
    )
    experiment_dir = (
        Path(args.experiment_dir)
        if args.experiment_dir is not None
        else outputs_dir / "experiments" / default_experiment_name
    )
    training_dir = experiment_dir / "training"
    summary_path = training_dir / "train_summary.json"
    latest_checkpoint = training_dir / "sparse_unet_spconv_latest.pt"

    if not check_file(summary_path, checks, f"{args.stage}_training_summary_json"):
        return
    check_file(latest_checkpoint, checks, f"{args.stage}_latest_checkpoint")

    summary = load_json(summary_path)
    text_alignment = summary.get("text_alignment", {})
    loss_weights = summary.get("loss_weights", {})
    epoch_logs = summary.get("epoch_logs", [])
    final_epoch = epoch_logs[-1] if epoch_logs else {}
    init_checkpoint = str(summary.get("init_checkpoint", ""))

    add_check(
        checks,
        f"{args.stage}_text_alignment_enabled",
        bool(text_alignment.get("enabled", False)),
        f"text_alignment={text_alignment}",
        text_alignment,
    )
    add_check(
        checks,
        f"{args.stage}_text_align_weight",
        float(loss_weights.get("text_align_weight", 0.0)) > 0.0,
        f"text_align_weight={loss_weights.get('text_align_weight')}",
        loss_weights,
    )
    add_check(
        checks,
        f"{args.stage}_epoch_logs",
        bool(epoch_logs) and "avg_text_align_loss" in final_epoch,
        f"epochs={len(epoch_logs)}, final_keys={sorted(final_epoch.keys())}",
        final_epoch,
    )
    if "avg_text_align_loss" in final_epoch:
        text_loss = float(final_epoch["avg_text_align_loss"])
        add_check(
            checks,
            f"{args.stage}_text_align_loss_finite",
            np.isfinite(text_loss) and text_loss >= 0.0,
            f"avg_text_align_loss={text_loss}",
            final_epoch,
        )
    add_check(
        checks,
        f"{args.stage}_warm_start_checkpoint",
        bool(init_checkpoint) and Path(init_checkpoint).exists(),
        f"init_checkpoint={init_checkpoint}",
    )


def verify_groupvit_dense_teacher_v12(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    experiment_dir = (
        Path(args.experiment_dir)
        if args.experiment_dir is not None
        else outputs_dir / "experiments" / "trainval_v12_groupvit_128"
    )
    dense_teacher_dir = experiment_dir / "precompute" / "dense_teacher_logits"
    dense_teacher_files = sorted(dense_teacher_dir.glob("sample_*_dense_teacher_logits.npz"))
    dense_teacher_summary = dense_teacher_dir / "batch_dense_teacher_logits_summary.json"
    dense_point_dir = experiment_dir / "precompute" / "dense_point_logits"
    dense_point_files = sorted(dense_point_dir.glob("sample_*_dense_point_logits.npz"))
    dense_point_summary = dense_point_dir / "batch_dense_point_logits_summary.json"

    check_file(dense_teacher_dir, checks, "v12_dense_teacher_dir")
    add_check(
        checks,
        "v12_dense_teacher_files",
        len(dense_teacher_files) > 0,
        f"num_dense_teacher_files={len(dense_teacher_files)}",
    )
    check_file(dense_teacher_summary, checks, "v12_dense_teacher_batch_summary_json", required=False)
    if dense_teacher_files:
        dense_teacher = load_npz(dense_teacher_files[0])
        required = ["dense_logits", "class_names", "camera_names", "camera_available", "teacher_backend"]
        missing = [key for key in required if key not in dense_teacher]
        add_check(checks, "v12_dense_teacher_required_keys", not missing, f"missing_keys={missing}", missing)
        if not missing:
            dense_logits = dense_teacher["dense_logits"]
            class_names = dense_teacher["class_names"]
            camera_names = dense_teacher["camera_names"]
            teacher_backend = str(dense_teacher["teacher_backend"].item())
            class_axis_ok = dense_logits.ndim == 4 and (
                dense_logits.shape[1] >= args.expected_num_classes
                or dense_logits.shape[-1] >= args.expected_num_classes
            )
            add_check(
                checks,
                "v12_dense_teacher_shapes",
                dense_logits.ndim == 4 and camera_names.shape[0] == dense_logits.shape[0] and class_axis_ok,
                (
                    f"dense_logits={dense_logits.shape}, camera_names={camera_names.shape}, "
                    f"classes={class_names.shape}, teacher_backend={teacher_backend}"
                ),
            )
            add_check(
                checks,
                "v12_dense_teacher_backend",
                teacher_backend == "groupvit_dense",
                f"teacher_backend={teacher_backend}",
            )

    check_file(dense_point_dir, checks, "v12_dense_point_dir")
    add_check(
        checks,
        "v12_dense_point_files",
        len(dense_point_files) > 0,
        f"num_dense_point_files={len(dense_point_files)}",
    )
    check_file(dense_point_summary, checks, "v12_dense_point_batch_summary_json", required=False)
    if dense_point_files:
        dense_point = load_npz(dense_point_files[0])
        required = ["point_teacher_logits", "point_dense_valid_mask", "class_names", "teacher_backend"]
        missing = [key for key in required if key not in dense_point]
        add_check(checks, "v12_dense_point_required_keys", not missing, f"missing_keys={missing}", missing)
        if not missing:
            logits = dense_point["point_teacher_logits"]
            valid_mask = dense_point["point_dense_valid_mask"].astype(bool)
            teacher_backend = str(dense_point["teacher_backend"].item())
            add_check(
                checks,
                "v12_dense_point_shapes",
                logits.ndim == 2 and logits.shape[0] == valid_mask.shape[0] and logits.shape[1] >= args.expected_num_classes,
                f"logits={logits.shape}, valid_mask={valid_mask.shape}, teacher_backend={teacher_backend}",
            )
            add_check(
                checks,
                "v12_dense_point_valid_points",
                int(valid_mask.sum()) > 0,
                f"valid_points={int(valid_mask.sum())}, total={int(valid_mask.shape[0])}",
            )


def verify_diagnostics_v13(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    experiment_dir = (
        Path(args.experiment_dir)
        if args.experiment_dir is not None
        else outputs_dir / "experiments" / "trainval_v13_diagnostics_128"
    )
    teacher_summary_path = experiment_dir / "teacher_quality" / "batch_teacher_pseudo_eval_summary.json"
    training_dir = experiment_dir / "supervised_training"
    training_summary_path = training_dir / "train_summary.json"
    latest_checkpoint = training_dir / "spconv_resunet_latest.pt"
    eval_summary_path = experiment_dir / "supervised_evaluation3d" / "batch_3d_eval_summary.json"

    if check_file(teacher_summary_path, checks, "v13_teacher_quality_summary_json"):
        teacher_summary = load_json(teacher_summary_path)
        metrics = teacher_summary.get("aggregate_metrics", {})
        add_check(
            checks,
            "v13_teacher_quality_metrics",
            {"all_miou", "base_miou", "prediction_coverage"}.issubset(metrics.keys()),
            f"metrics_keys={sorted(metrics.keys())}",
            metrics,
        )

    if check_file(training_summary_path, checks, "v13_supervised_training_summary_json"):
        training_summary = load_json(training_summary_path)
        backbone = training_summary.get("backbone", {})
        epoch_logs = training_summary.get("epoch_logs", [])
        final_epoch = epoch_logs[-1] if epoch_logs else {}
        add_check(
            checks,
            "v13_supervised_backbone",
            backbone.get("backbone") == "spconv_resunet",
            f"backbone={backbone.get('backbone')}",
            backbone,
        )
        add_check(
            checks,
            "v13_supervised_epoch_logs",
            bool(epoch_logs) and "avg_ce_loss" in final_epoch,
            f"epochs={len(epoch_logs)}, final_keys={sorted(final_epoch.keys())}",
            final_epoch,
        )
    check_file(latest_checkpoint, checks, "v13_supervised_latest_checkpoint")

    if check_file(eval_summary_path, checks, "v13_supervised_eval_summary_json"):
        eval_summary = load_json(eval_summary_path)
        metrics = eval_summary.get("aggregate_metrics", {})
        add_check(
            checks,
            "v13_supervised_eval_metrics",
            {"all_miou", "base_miou", "prediction_coverage"}.issubset(metrics.keys()),
            f"metrics_keys={sorted(metrics.keys())}",
            metrics,
        )


def verify_supervised_recipe_v14(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    experiment_dir = (
        Path(args.experiment_dir)
        if args.experiment_dir is not None
        else outputs_dir / "experiments" / "trainval_v14_supervised_resunet_128"
    )
    class_freq_path = experiment_dir / "class_frequencies.json"
    training_dir = experiment_dir / "training"
    training_summary_path = training_dir / "train_summary.json"
    best_checkpoint = training_dir / "spconv_resunet_best.pt"
    latest_checkpoint = training_dir / "spconv_resunet_latest.pt"
    eval_summary_path = experiment_dir / "evaluation3d" / "batch_3d_eval_summary.json"

    if check_file(class_freq_path, checks, "v14_class_frequencies_json"):
        class_freq = load_json(class_freq_path)
        has_weights = "raw_class_weights" in class_freq and "train_class_weights" in class_freq
        add_check(
            checks,
            "v14_class_frequency_weights",
            has_weights,
            f"keys={sorted(class_freq.keys())}",
            {key: class_freq.get(key) for key in ["num_samples_used", "raw_counts", "raw_class_weights"]},
        )

    if check_file(training_summary_path, checks, "v14_training_summary_json"):
        training_summary = load_json(training_summary_path)
        backbone = training_summary.get("backbone", {})
        loss_weights = training_summary.get("loss_weights", {})
        class_weights = training_summary.get("class_weights", {})
        augmentation = training_summary.get("augmentation", {})
        eval_info = training_summary.get("eval_during_training", {})
        epoch_logs = training_summary.get("epoch_logs", [])
        final_epoch = epoch_logs[-1] if epoch_logs else {}
        add_check(
            checks,
            "v14_data_source_raw_lidarseg",
            training_summary.get("data_source") == "raw_lidarseg",
            f"data_source={training_summary.get('data_source')}",
            training_summary.get("data_source"),
        )
        add_check(checks, "v14_backbone", backbone.get("backbone") == "spconv_resunet", f"backbone={backbone.get('backbone')}", backbone)
        add_check(
            checks,
            "v14_lovasz_or_dice_enabled",
            float(loss_weights.get("lovasz_weight", 0.0)) > 0.0 or float(loss_weights.get("dice_weight", 0.0)) > 0.0,
            f"loss_weights={loss_weights}",
            loss_weights,
        )
        add_check(checks, "v14_class_weights_enabled", bool(class_weights), f"class_weights={class_weights}", class_weights)
        add_check(checks, "v14_augmentation_recorded", "enabled" in augmentation, f"augmentation={augmentation}", augmentation)
        add_check(checks, "v14_eval_during_training", bool(eval_info.get("enabled", False)), f"eval={eval_info}", eval_info)
        add_check(
            checks,
            "v14_epoch_logs",
            bool(epoch_logs) and "avg_lovasz_loss" in final_epoch and "eval" in final_epoch,
            f"epochs={len(epoch_logs)}, final_keys={sorted(final_epoch.keys())}",
            final_epoch,
        )
    check_file(best_checkpoint, checks, "v14_best_checkpoint")
    check_file(latest_checkpoint, checks, "v14_latest_checkpoint")

    if check_file(eval_summary_path, checks, "v14_eval_summary_json"):
        eval_summary = load_json(eval_summary_path)
        metrics = eval_summary.get("aggregate_metrics", {})
        add_check(
            checks,
            "v14_eval_metrics",
            {"all_miou", "base_miou", "prediction_coverage"}.issubset(metrics.keys()),
            f"metrics_keys={sorted(metrics.keys())}",
            metrics,
        )


def verify_cylinder_baseline_v15(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    experiment_dir = (
        Path(args.experiment_dir)
        if args.experiment_dir is not None
        else outputs_dir / "experiments" / "trainval_v15_cylinder_1024"
    )
    class_freq_path = experiment_dir / "class_frequencies.json"
    training_dir = experiment_dir / "training"
    training_summary_path = training_dir / "train_summary.json"
    best_checkpoint = training_dir / "cylinder_spconv_unet_best.pt"
    latest_checkpoint = training_dir / "cylinder_spconv_unet_latest.pt"
    eval_summary_path = experiment_dir / "evaluation3d" / "batch_3d_eval_summary.json"

    if check_file(class_freq_path, checks, "v15_class_frequencies_json"):
        class_freq = load_json(class_freq_path)
        has_weights = "raw_class_weights" in class_freq and "train_class_weights" in class_freq
        add_check(
            checks,
            "v15_class_frequency_weights",
            has_weights,
            f"keys={sorted(class_freq.keys())}",
            {key: class_freq.get(key) for key in ["num_samples_used", "raw_counts", "raw_class_weights"]},
        )

    if check_file(training_summary_path, checks, "v15_training_summary_json"):
        training_summary = load_json(training_summary_path)
        backbone = training_summary.get("backbone", {})
        loss_weights = training_summary.get("loss_weights", {})
        class_weights = training_summary.get("class_weights", {})
        eval_info = training_summary.get("eval_during_training", {})
        epoch_logs = training_summary.get("epoch_logs", [])
        final_epoch = epoch_logs[-1] if epoch_logs else {}
        add_check(
            checks,
            "v15_data_source_raw_lidarseg",
            training_summary.get("data_source") == "raw_lidarseg",
            f"data_source={training_summary.get('data_source')}",
            training_summary.get("data_source"),
        )
        add_check(
            checks,
            "v15_backbone",
            backbone.get("backbone") == "cylinder_spconv_unet",
            f"backbone={backbone.get('backbone')}",
            backbone,
        )
        add_check(
            checks,
            "v15_cylindrical_range",
            training_summary.get("point_cloud_range", [None])[0] == 0.0,
            f"point_cloud_range={training_summary.get('point_cloud_range')}",
            training_summary.get("point_cloud_range"),
        )
        add_check(
            checks,
            "v15_lovasz_or_dice_enabled",
            float(loss_weights.get("lovasz_weight", 0.0)) > 0.0 or float(loss_weights.get("dice_weight", 0.0)) > 0.0,
            f"loss_weights={loss_weights}",
            loss_weights,
        )
        add_check(checks, "v15_class_weights_enabled", bool(class_weights), f"class_weights={class_weights}", class_weights)
        add_check(checks, "v15_eval_during_training", bool(eval_info.get("enabled", False)), f"eval={eval_info}", eval_info)
        add_check(
            checks,
            "v15_epoch_logs",
            bool(epoch_logs) and "avg_lovasz_loss" in final_epoch and "eval" in final_epoch,
            f"epochs={len(epoch_logs)}, final_keys={sorted(final_epoch.keys())}",
            final_epoch,
        )
    check_file(best_checkpoint, checks, "v15_best_checkpoint")
    check_file(latest_checkpoint, checks, "v15_latest_checkpoint")

    if check_file(eval_summary_path, checks, "v15_eval_summary_json"):
        eval_summary = load_json(eval_summary_path)
        metrics = eval_summary.get("aggregate_metrics", {})
        add_check(
            checks,
            "v15_eval_metrics",
            {"all_miou", "base_miou", "prediction_coverage"}.issubset(metrics.keys()),
            f"metrics_keys={sorted(metrics.keys())}",
            metrics,
        )


def verify_official16_cylinder_v16a(outputs_dir: Path, args, checks: list[dict[str, Any]]) -> None:
    experiment_dir = (
        Path(args.experiment_dir)
        if args.experiment_dir is not None
        else outputs_dir / "experiments" / "trainval_v16a_official16_cylinder_128"
    )
    class_freq_path = experiment_dir / "class_frequencies.json"
    precheck_summary_path = experiment_dir / "reports" / "pre_v16_sanity_summary.json"
    training_dir = experiment_dir / "training"
    training_summary_path = training_dir / "train_summary.json"
    best_checkpoint = training_dir / "cylinder_spconv_unet_best.pt"
    latest_checkpoint = training_dir / "cylinder_spconv_unet_latest.pt"
    eval_summary_path = experiment_dir / "evaluation3d" / "batch_3d_eval_summary.json"
    compact_summary_path = experiment_dir / "compact_summary.json"

    if check_file(precheck_summary_path, checks, "v16a_precheck_summary_json"):
        precheck = load_json(precheck_summary_path)
        aggregate = precheck.get("aggregate", {})
        range_ratio = float(aggregate.get("cylinder_range_inside_ratio", 0.0))
        add_check(
            checks,
            "v16a_expanded_range_coverage",
            range_ratio >= 0.98,
            f"cylinder_range_inside_ratio={range_ratio:.6f}",
            aggregate,
        )

    if check_file(class_freq_path, checks, "v16a_class_frequencies_json"):
        class_freq = load_json(class_freq_path)
        weights = class_freq.get("official_16_class_weights", [])
        add_check(
            checks,
            "v16a_official16_class_weights",
            len(weights) == 16,
            f"num_weights={len(weights)}",
            {"official_16_train_counts": class_freq.get("official_16_train_counts"), "weights": weights},
        )

    if check_file(training_summary_path, checks, "v16a_training_summary_json"):
        training_summary = load_json(training_summary_path)
        backbone = training_summary.get("backbone", {})
        loss_weights = training_summary.get("loss_weights", {})
        class_weights = training_summary.get("class_weights", {})
        eval_info = training_summary.get("eval_during_training", {})
        epoch_logs = training_summary.get("epoch_logs", [])
        final_epoch = epoch_logs[-1] if epoch_logs else {}
        add_check(
            checks,
            "v16a_output_space",
            training_summary.get("student_output_space") == "official_lidarseg_16",
            f"student_output_space={training_summary.get('student_output_space')}",
            training_summary.get("student_output_space"),
        )
        add_check(
            checks,
            "v16a_num_output_classes",
            int(training_summary.get("num_output_classes", -1)) == 16,
            f"num_output_classes={training_summary.get('num_output_classes')}",
            training_summary.get("num_output_classes"),
        )
        add_check(
            checks,
            "v16a_backbone",
            backbone.get("backbone") == "cylinder_spconv_unet",
            f"backbone={backbone.get('backbone')}",
            backbone,
        )
        add_check(
            checks,
            "v16a_expanded_cylinder_range_recorded",
            float(training_summary.get("point_cloud_range", [0, 0, 0, 0])[3]) >= 80.0,
            f"point_cloud_range={training_summary.get('point_cloud_range')}",
            training_summary.get("point_cloud_range"),
        )
        add_check(
            checks,
            "v16a_lovasz_or_dice_enabled",
            float(loss_weights.get("lovasz_weight", 0.0)) > 0.0 or float(loss_weights.get("dice_weight", 0.0)) > 0.0,
            f"loss_weights={loss_weights}",
            loss_weights,
        )
        add_check(checks, "v16a_class_weights_enabled", bool(class_weights), f"class_weights={class_weights}", class_weights)
        add_check(checks, "v16a_eval_during_training", bool(eval_info.get("enabled", False)), f"eval={eval_info}", eval_info)
        add_check(
            checks,
            "v16a_epoch_logs",
            bool(epoch_logs) and "avg_lovasz_loss" in final_epoch and "eval" in final_epoch,
            f"epochs={len(epoch_logs)}, final_keys={sorted(final_epoch.keys())}",
            final_epoch,
        )
    check_file(best_checkpoint, checks, "v16a_best_checkpoint")
    check_file(latest_checkpoint, checks, "v16a_latest_checkpoint")

    if check_file(eval_summary_path, checks, "v16a_eval_summary_json"):
        eval_summary = load_json(eval_summary_path)
        metrics = eval_summary.get("aggregate_metrics", {})
        add_check(
            checks,
            "v16a_eval_metrics",
            {"all_miou", "base_miou", "prediction_coverage"}.issubset(metrics.keys()),
            f"metrics_keys={sorted(metrics.keys())}",
            metrics,
        )
        coverage = float(metrics.get("prediction_coverage", 0.0))
        add_check(
            checks,
            "v16a_prediction_coverage",
            coverage >= 0.98,
            f"prediction_coverage={coverage:.6f}",
            metrics,
        )

    check_file(compact_summary_path, checks, "v16a_compact_summary_json")


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("verify_mvp_outputs")
    outputs_dir = Path(args.outputs_dir)
    output_dir = ensure_dir(args.output_dir)
    checks: list[dict[str, Any]] = []

    if args.stage not in {"v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16a"}:
        verify_projection(outputs_dir, args.sample_idx, args, checks)
        verify_overlays(outputs_dir, args.sample_idx, checks)
    if args.stage in {"v1", "v2", "v3", "v4", "v5", "v7"}:
        verify_image_features(outputs_dir, args.sample_idx, args, checks)
        verify_point_features(outputs_dir, args.sample_idx, args, checks)
        verify_zero_shot(outputs_dir, args.sample_idx, args, checks)
    if args.stage in {"v2", "v3", "v4", "v5", "v7"}:
        verify_reliability(outputs_dir, args.sample_idx, checks)
    if args.stage == "v3":
        verify_training_dryrun(outputs_dir, args.sample_idx, checks)
    if args.stage == "v4":
        verify_training_v4(outputs_dir, args, checks)
    if args.stage == "v5":
        verify_training_v5(outputs_dir, args, checks)
    if args.stage in {"v6", "v7"}:
        verify_dense_teacher_v6(outputs_dir, args.sample_idx, args, checks)
    if args.stage == "v7":
        verify_training_v7(outputs_dir, args, checks)
    if args.stage == "v8":
        verify_prediction_eval_v8(outputs_dir, args.sample_idx, args, checks)
    if args.stage == "v9":
        verify_mini_experiment_v9(outputs_dir, args, checks)
    if args.stage == "v10":
        verify_open_vocab_v10(outputs_dir, args, checks)
    if args.stage == "v11":
        verify_text_aligned_training_v11(outputs_dir, args, checks)
        verify_open_vocab_v10(outputs_dir, args, checks)
    if args.stage == "v12":
        verify_groupvit_dense_teacher_v12(outputs_dir, args, checks)
        verify_text_aligned_training_v11(outputs_dir, args, checks)
        verify_open_vocab_v10(outputs_dir, args, checks)
    if args.stage == "v13":
        verify_diagnostics_v13(outputs_dir, args, checks)
    if args.stage == "v14":
        verify_supervised_recipe_v14(outputs_dir, args, checks)
    if args.stage == "v15":
        verify_cylinder_baseline_v15(outputs_dir, args, checks)
    if args.stage == "v16a":
        verify_official16_cylinder_v16a(outputs_dir, args, checks)

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

    print("========== SEND_THIS_TO_CODEX ==========")
    print(f"stage={args.stage}")
    print(f"verification_status={summary['status']}")
    print(f"passed={len(passed)}")
    print(f"warned={len(warned)}")
    print(f"failed={len(failed)}")
    print(f"summary_json={summary_path}")
    print("failed_checks=" + ",".join(check["name"] for check in failed))
    print("warned_checks=" + ",".join(check["name"] for check in warned))
    print("========================================")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
