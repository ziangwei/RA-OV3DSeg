from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.utils.io import ensure_dir, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402
from ra_ov3dseg.models.external_dense_teacher import dense_logits_to_nchw, scalar_to_str  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample dense teacher class logits at projected LiDAR points.")
    parser.add_argument("--sample_idx", default=None, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--max_samples", default=1, type=int)
    parser.add_argument("--projection_npz", default=None, type=str)
    parser.add_argument("--dense_teacher_npz", default=None, type=str)
    parser.add_argument("--projection_dir", default="outputs/projections", type=str)
    parser.add_argument("--dense_teacher_dir", default="outputs/dense_teacher_logits", type=str)
    parser.add_argument("--output_dir", default="outputs/dense_point_logits", type=str)
    parser.add_argument("--aggregation", default="mean", choices=["mean", "closest_camera"])
    parser.add_argument("--output_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def infer_sample_idx_from_path(path: Path) -> int | None:
    match = re.search(r"sample_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(np.sum(exp, axis=axis, keepdims=True), 1e-6, None)


def build_jobs(args) -> list[tuple[int, Path, Path]]:
    if args.projection_npz is not None or args.dense_teacher_npz is not None:
        if args.projection_npz is None or args.dense_teacher_npz is None:
            raise ValueError("projection_npz and dense_teacher_npz must be provided together.")
        projection_npz = Path(args.projection_npz).resolve()
        dense_teacher_npz = Path(args.dense_teacher_npz).resolve()
        sample_idx = args.sample_idx
        if sample_idx is None:
            sample_idx = infer_sample_idx_from_path(projection_npz)
        if sample_idx is None:
            raise ValueError("sample_idx is required if it cannot be inferred from projection filename.")
        return [(sample_idx, projection_npz, dense_teacher_npz)]

    if args.sample_idx is not None:
        sample_indices = [args.sample_idx]
    else:
        sample_indices = list(range(args.start_idx, args.start_idx + args.max_samples))
    projection_dir = Path(args.projection_dir).resolve()
    dense_teacher_dir = Path(args.dense_teacher_dir).resolve()
    jobs = []
    for sample_idx in sample_indices:
        prefix = f"sample_{sample_idx:04d}"
        jobs.append(
            (
                sample_idx,
                projection_dir / f"{prefix}_projection.npz",
                dense_teacher_dir / f"{prefix}_dense_teacher_logits.npz",
            )
        )
    return jobs


def assign_logits(projection: dict[str, np.ndarray], dense_teacher: dict[str, np.ndarray], aggregation: str) -> dict:
    point_xyz = projection["point_xyz"].astype(np.float32)
    uv = projection["uv"].astype(np.float32)
    depth = projection["depth"].astype(np.float32)
    valid_masks = projection["valid_masks"].astype(bool)
    camera_names = [str(name) for name in projection["camera_names"].tolist()]
    teacher_camera_names = [str(name) for name in dense_teacher["camera_names"].tolist()]
    if camera_names != teacher_camera_names:
        raise ValueError("Camera order mismatch between projection and dense teacher logits.")

    class_names = [str(name) for name in dense_teacher["class_names"].tolist()]
    dense_logits, dense_logit_layout = dense_logits_to_nchw(
        dense_teacher["dense_logits"].astype(np.float32),
        num_classes=len(class_names),
    )
    camera_available = dense_teacher["camera_available"].astype(bool)
    image_widths = projection["image_widths"].astype(np.float32)
    image_heights = projection["image_heights"].astype(np.float32)
    num_cameras, num_points, _ = uv.shape
    num_classes = int(dense_logits.shape[1])

    point_logits_sum = np.zeros((num_points, num_classes), dtype=np.float32)
    point_logits_count = np.zeros(num_points, dtype=np.int32)
    point_depth_closest = np.full(num_points, np.inf, dtype=np.float32)
    point_logits_closest = np.zeros((num_points, num_classes), dtype=np.float32)
    selected_camera_index = np.full(num_points, -1, dtype=np.int32)

    for camera_idx in range(num_cameras):
        if not camera_available[camera_idx]:
            continue
        valid_mask = valid_masks[camera_idx]
        if not np.any(valid_mask):
            continue
        camera_logits = dense_logits[camera_idx]
        _, logit_height, logit_width = camera_logits.shape
        point_indices = np.nonzero(valid_mask)[0]
        point_uv = uv[camera_idx, valid_mask]
        point_depth = depth[camera_idx, valid_mask]
        x = np.floor(point_uv[:, 0] * logit_width / max(image_widths[camera_idx], 1.0)).astype(np.int32)
        y = np.floor(point_uv[:, 1] * logit_height / max(image_heights[camera_idx], 1.0)).astype(np.int32)
        x = np.clip(x, 0, logit_width - 1)
        y = np.clip(y, 0, logit_height - 1)
        sampled_logits = camera_logits[:, y, x].T.astype(np.float32)

        if aggregation == "mean":
            point_logits_sum[point_indices] += sampled_logits
            point_logits_count[point_indices] += 1
        else:
            better_mask = point_depth < point_depth_closest[point_indices]
            chosen_indices = point_indices[better_mask]
            point_logits_closest[chosen_indices] = sampled_logits[better_mask]
            point_depth_closest[chosen_indices] = point_depth[better_mask]
            selected_camera_index[chosen_indices] = camera_idx
            point_logits_count[point_indices] += 1

    if aggregation == "mean":
        point_valid_mask = point_logits_count > 0
        point_teacher_logits = np.zeros_like(point_logits_sum)
        point_teacher_logits[point_valid_mask] = (
            point_logits_sum[point_valid_mask] / point_logits_count[point_valid_mask, None].astype(np.float32)
        )
    else:
        point_valid_mask = selected_camera_index >= 0
        point_teacher_logits = point_logits_closest

    point_teacher_probs = np.zeros_like(point_teacher_logits, dtype=np.float32)
    if np.any(point_valid_mask):
        point_teacher_probs[point_valid_mask] = softmax_np(point_teacher_logits[point_valid_mask], axis=1)
    pred_label_indices = np.full(num_points, -1, dtype=np.int32)
    pred_scores = np.full(num_points, np.nan, dtype=np.float32)
    if np.any(point_valid_mask):
        pred_label_indices[point_valid_mask] = np.argmax(point_teacher_probs[point_valid_mask], axis=1).astype(np.int32)
        pred_scores[point_valid_mask] = np.max(point_teacher_probs[point_valid_mask], axis=1).astype(np.float32)

    return {
        "point_xyz": point_xyz,
        "point_teacher_logits": point_teacher_logits.astype(np.float32),
        "point_teacher_probs": point_teacher_probs.astype(np.float32),
        "point_dense_valid_mask": point_valid_mask.astype(bool),
        "point_dense_camera_count": point_logits_count.astype(np.int32),
        "point_dense_pred_label_indices": pred_label_indices,
        "point_dense_pred_scores": pred_scores,
        "selected_camera_index": selected_camera_index.astype(np.int32),
        "camera_names": np.asarray(camera_names),
        "dense_logit_layout": np.array(dense_logit_layout),
    }


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("assign_dense_logits_to_points")
    output_dir = ensure_dir(args.output_dir)
    output_dtype = np.float16 if args.output_dtype == "float16" else np.float32
    jobs = build_jobs(args)
    batch_summary = {"aggregation": args.aggregation, "jobs": []}

    for sample_idx, projection_npz, dense_teacher_npz in jobs:
        logger.info("========== sample_idx=%d ==========", sample_idx)
        if not projection_npz.exists():
            raise FileNotFoundError(f"projection npz not found: {projection_npz}")
        if not dense_teacher_npz.exists():
            raise FileNotFoundError(f"dense teacher npz not found: {dense_teacher_npz}")
        prefix = f"sample_{sample_idx:04d}"
        output_npz = output_dir / f"{prefix}_dense_point_logits.npz"
        summary_json = output_dir / f"{prefix}_dense_point_logits_summary.json"
        if args.skip_existing and output_npz.exists() and summary_json.exists():
            logger.info("skip existing dense point logits for sample_idx=%d", sample_idx)
            batch_summary["jobs"].append({"sample_idx": sample_idx, "status": "skipped_existing"})
            continue

        projection = load_npz(projection_npz)
        dense_teacher = load_npz(dense_teacher_npz)
        assignment = assign_logits(projection, dense_teacher, aggregation=args.aggregation)
        assignment["point_teacher_logits"] = assignment["point_teacher_logits"].astype(output_dtype)
        assignment["point_teacher_probs"] = assignment["point_teacher_probs"].astype(output_dtype)
        valid_points = int(assignment["point_dense_valid_mask"].sum())

        save_npz(
            output_npz,
            **assignment,
            sample_idx=np.array(sample_idx, dtype=np.int32),
            sample_token=projection["sample_token"],
            teacher_backend=(
                dense_teacher["teacher_backend"]
                if "teacher_backend" in dense_teacher
                else np.array("external_dense_logits")
            ),
            teacher_role=dense_teacher["teacher_role"] if "teacher_role" in dense_teacher else np.array("unknown"),
            teacher_feature_granularity=(
                dense_teacher["teacher_feature_granularity"]
                if "teacher_feature_granularity" in dense_teacher
                else np.array("dense_class_logits")
            ),
            model_name=dense_teacher["model_name"] if "model_name" in dense_teacher else np.array("external"),
            class_names=dense_teacher["class_names"],
            prompts=dense_teacher["prompts"] if "prompts" in dense_teacher else dense_teacher["class_names"],
            aggregation=np.array(args.aggregation),
        )
        class_hist = {}
        class_names = [str(name) for name in dense_teacher["class_names"].tolist()]
        for class_idx, class_name in enumerate(class_names):
            count = int(np.sum(assignment["point_dense_pred_label_indices"] == class_idx))
            if count > 0:
                class_hist[class_name] = count
        summary = {
            "sample_idx": sample_idx,
            "sample_token": str(projection["sample_token"].item()),
            "teacher_backend": scalar_to_str(dense_teacher.get("teacher_backend"), default="external_dense_logits"),
            "model_name": scalar_to_str(dense_teacher.get("model_name"), default="external"),
            "dense_logit_layout": scalar_to_str(assignment["dense_logit_layout"]),
            "aggregation": args.aggregation,
            "num_points": int(assignment["point_xyz"].shape[0]),
            "num_valid_points": valid_points,
            "valid_ratio": float(valid_points / max(assignment["point_xyz"].shape[0], 1)),
            "num_classes": len(class_names),
            "class_hist": class_hist,
            "projection_npz": str(projection_npz),
            "dense_teacher_npz": str(dense_teacher_npz),
            "output_npz": str(output_npz),
        }
        save_json(summary_json, summary)
        batch_summary["jobs"].append(
            {"sample_idx": sample_idx, "status": "done", "output_npz": str(output_npz), "summary_json": str(summary_json)}
        )
        logger.info("dense point logits saved | valid_points=%d | output=%s", valid_points, output_npz)

    if len(jobs) > 1:
        batch_summary_path = output_dir / "batch_dense_point_logits_summary.json"
        save_json(batch_summary_path, batch_summary)
        logger.info("batch dense point logits summary saved to: %s", batch_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
