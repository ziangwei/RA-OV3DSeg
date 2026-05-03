from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.models.reliability import boundary_weight, compute_point_reliability  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402
from ra_ov3dseg.visualization.visualize_points import (  # noqa: E402
    save_bev_score_plot,
    save_score_point_cloud_ply,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute MVP-v2 point-level reliability scores.")
    parser.add_argument("--sample_idx", default=None, type=int, help="Single sample index.")
    parser.add_argument("--start_idx", default=0, type=int, help="Batch start sample index.")
    parser.add_argument("--max_samples", default=1, type=int, help="Number of samples in batch mode.")
    parser.add_argument("--projection_npz", default=None, type=str, help="Single projection .npz path.")
    parser.add_argument("--zero_shot_npz", default=None, type=str, help="Single zero-shot .npz path.")
    parser.add_argument("--projection_dir", default="outputs/projections", type=str, help="Projection output dir.")
    parser.add_argument("--zero_shot_dir", default="outputs/zero_shot", type=str, help="Zero-shot output dir.")
    parser.add_argument("--output_dir", default="outputs/reliability", type=str, help="Reliability output dir.")
    parser.add_argument("--max_distance", default=60.0, type=float, help="Distance decay upper distance.")
    parser.add_argument("--min_distance_weight", default=0.1, type=float, help="Distance weight lower bound.")
    parser.add_argument("--boundary_margin_ratio", default=0.05, type=float, help="Image border margin ratio.")
    parser.add_argument("--semantic_min_similarity", default=0.0, type=float, help="Semantic weight lower bound.")
    parser.add_argument("--semantic_max_similarity", default=0.35, type=float, help="Semantic weight upper bound.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples with existing outputs.")
    return parser


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def summarize_array(values: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float | int]:
    if valid_mask is not None:
        values = values[valid_mask]
    values = values[np.isfinite(values)]
    if values.shape[0] == 0:
        return {"count": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": int(values.shape[0]),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def build_jobs(args) -> list[tuple[int, Path, Path]]:
    if args.projection_npz is not None or args.zero_shot_npz is not None:
        if args.projection_npz is None or args.zero_shot_npz is None:
            raise ValueError("projection_npz and zero_shot_npz must be provided together.")
        if args.sample_idx is None:
            raise ValueError("sample_idx is required in explicit single-file mode.")
        return [(args.sample_idx, Path(args.projection_npz).resolve(), Path(args.zero_shot_npz).resolve())]

    if args.sample_idx is not None:
        sample_indices = [args.sample_idx]
    else:
        sample_indices = list(range(args.start_idx, args.start_idx + args.max_samples))

    projection_dir = Path(args.projection_dir).resolve()
    zero_shot_dir = Path(args.zero_shot_dir).resolve()
    jobs = []
    for sample_idx in sample_indices:
        prefix = f"sample_{sample_idx:04d}"
        jobs.append(
            (
                sample_idx,
                projection_dir / f"{prefix}_projection.npz",
                zero_shot_dir / f"{prefix}_zero_shot_predictions.npz",
            )
        )
    return jobs


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("compute_reliability")
    output_dir = ensure_dir(args.output_dir)
    jobs = build_jobs(args)
    batch_summary: dict[str, Any] = {"jobs": []}

    for sample_idx, projection_npz, zero_shot_npz in jobs:
        logger.info("========== sample_idx=%d ==========", sample_idx)
        if not projection_npz.exists():
            raise FileNotFoundError(f"projection npz not found: {projection_npz}")
        if not zero_shot_npz.exists():
            raise FileNotFoundError(f"zero-shot npz not found: {zero_shot_npz}")

        prefix = f"sample_{sample_idx:04d}"
        reliability_npz = output_dir / f"{prefix}_reliability.npz"
        summary_json = output_dir / f"{prefix}_reliability_summary.json"
        bev_path = output_dir / f"{prefix}_reliability_bev.png"
        ply_path = output_dir / f"{prefix}_reliability_points.ply"

        if args.skip_existing and reliability_npz.exists() and summary_json.exists():
            logger.info("skip existing reliability outputs for sample_idx=%d", sample_idx)
            batch_summary["jobs"].append(
                {"sample_idx": sample_idx, "status": "skipped_existing", "reliability_npz": str(reliability_npz)}
            )
            continue

        projection = load_npz(projection_npz)
        zero_shot = load_npz(zero_shot_npz)

        point_xyz = projection["point_xyz"].astype(np.float32)
        point_valid_mask = zero_shot["point_valid_mask"].astype(bool)
        pred_scores = zero_shot["pred_scores"].astype(np.float32)
        visible_camera_count = projection["visible_camera_count"].astype(np.int32)
        distances = np.linalg.norm(point_xyz, axis=1).astype(np.float32)

        if point_xyz.shape[0] != pred_scores.shape[0]:
            raise ValueError(
                f"point count mismatch: projection={point_xyz.shape[0]}, zero_shot={pred_scores.shape[0]}"
            )

        boundary_weights = boundary_weight(
            uv=projection["uv"].astype(np.float32),
            image_widths=projection["image_widths"].astype(np.float32),
            image_heights=projection["image_heights"].astype(np.float32),
            valid_masks=projection["valid_masks"].astype(bool),
            margin_ratio=args.boundary_margin_ratio,
        )
        max_similarity = np.nan_to_num(pred_scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        max_similarity[~point_valid_mask] = 0.0

        weights = compute_point_reliability(
            distances=distances,
            visible_camera_count=visible_camera_count,
            boundary_weights=boundary_weights,
            max_similarity=max_similarity,
            max_distance=args.max_distance,
            min_distance_weight=args.min_distance_weight,
            semantic_min_similarity=args.semantic_min_similarity,
            semantic_max_similarity=args.semantic_max_similarity,
        )
        reliability_weight = weights["reliability_weight"]
        reliability_weight[~point_valid_mask] = 0.0

        save_bev_score_plot(
            point_xyz=point_xyz,
            scores=reliability_weight,
            output_path=bev_path,
            valid_mask=point_valid_mask,
            title="Point Reliability (MVP-v2)",
        )
        save_score_point_cloud_ply(
            point_xyz=point_xyz,
            scores=reliability_weight,
            output_path=ply_path,
            valid_mask=point_valid_mask,
        )

        save_npz(
            reliability_npz,
            sample_idx=np.array(sample_idx, dtype=np.int32),
            sample_token=zero_shot["sample_token"],
            point_xyz=point_xyz,
            point_valid_mask=point_valid_mask,
            distances=distances,
            visible_camera_count=visible_camera_count,
            max_similarity=max_similarity,
            boundary_weight=boundary_weights.astype(np.float32),
            distance_weight=weights["distance_weight"].astype(np.float32),
            geometric_weight=weights["geometric_weight"].astype(np.float32),
            semantic_weight=weights["semantic_weight"].astype(np.float32),
            reliability_weight=reliability_weight.astype(np.float32),
        )

        valid_reliability = point_valid_mask & np.isfinite(reliability_weight)
        high_reliability = valid_reliability & (reliability_weight >= 0.5)
        summary = {
            "sample_idx": sample_idx,
            "sample_token": str(zero_shot["sample_token"].item()),
            "num_points": int(point_xyz.shape[0]),
            "num_valid_points": int(point_valid_mask.sum()),
            "num_high_reliability_points": int(high_reliability.sum()),
            "high_reliability_ratio_among_valid": float(high_reliability.sum() / max(point_valid_mask.sum(), 1)),
            "distance_weight": summarize_array(weights["distance_weight"], point_valid_mask),
            "boundary_weight": summarize_array(boundary_weights, point_valid_mask),
            "geometric_weight": summarize_array(weights["geometric_weight"], point_valid_mask),
            "semantic_weight": summarize_array(weights["semantic_weight"], point_valid_mask),
            "reliability_weight": summarize_array(reliability_weight, point_valid_mask),
            "params": {
                "max_distance": args.max_distance,
                "min_distance_weight": args.min_distance_weight,
                "boundary_margin_ratio": args.boundary_margin_ratio,
                "semantic_min_similarity": args.semantic_min_similarity,
                "semantic_max_similarity": args.semantic_max_similarity,
            },
            "projection_npz": str(projection_npz),
            "zero_shot_npz": str(zero_shot_npz),
            "reliability_npz": str(reliability_npz),
            "bev_path": str(bev_path),
            "ply_path": str(ply_path),
        }
        save_json(summary_json, summary)
        logger.info(
            "reliability saved | valid=%d | high>=0.5=%d | mean=%.4f | npz=%s",
            summary["num_valid_points"],
            summary["num_high_reliability_points"],
            summary["reliability_weight"]["mean"],
            reliability_npz,
        )
        batch_summary["jobs"].append(
            {
                "sample_idx": sample_idx,
                "status": "done",
                "reliability_npz": str(reliability_npz),
                "summary_json": str(summary_json),
            }
        )

    if len(jobs) > 1:
        batch_summary_path = output_dir / "batch_reliability_summary.json"
        save_json(batch_summary_path, batch_summary)
        logger.info("batch reliability summary saved to: %s", batch_summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
