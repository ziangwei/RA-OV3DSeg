from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.models.point_feature_assigner import PointFeatureAssigner  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="根据投影结果把 2D patch features 赋给 3D 点。")
    parser.add_argument("--sample_idx", default=None, type=int, help="指定单个 sample 索引。")
    parser.add_argument("--start_idx", default=0, type=int, help="批量赋值时的起始 sample 索引。")
    parser.add_argument("--max_samples", default=1, type=int, help="批量赋值多少个 sample。")
    parser.add_argument("--projection_npz", default=None, type=str, help="单个投影结果 .npz 路径。")
    parser.add_argument("--image_feature_npz", default=None, type=str, help="单个 2D feature .npz 路径。")
    parser.add_argument("--projection_dir", default="outputs/projections", type=str, help="投影目录。")
    parser.add_argument("--image_feature_dir", default="outputs/features2d", type=str, help="2D feature 目录。")
    parser.add_argument("--output_dir", default="outputs/point_features", type=str, help="点特征输出目录。")
    parser.add_argument(
        "--aggregation",
        default="mean",
        choices=["mean", "closest_camera"],
        help="多相机可见时如何融合 2D features。",
    )
    parser.add_argument("--skip_existing", action="store_true", help="如果点特征文件已存在，则跳过。")
    return parser


def infer_sample_idx_from_path(path: Path) -> int | None:
    match = re.search(r"sample_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def load_npz_as_dict(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("assign_2d_features_to_points")
    output_dir = ensure_dir(args.output_dir)
    assigner = PointFeatureAssigner(aggregation=args.aggregation, normalize_output=True)

    jobs = []
    if args.projection_npz is not None or args.image_feature_npz is not None:
        if args.projection_npz is None or args.image_feature_npz is None:
            raise ValueError("projection_npz and image_feature_npz must be provided together for single-sample mode.")
        projection_npz = Path(args.projection_npz).resolve()
        image_feature_npz = Path(args.image_feature_npz).resolve()
        sample_idx = args.sample_idx
        if sample_idx is None:
            sample_idx = infer_sample_idx_from_path(projection_npz)
        if sample_idx is None:
            raise ValueError("sample_idx is required if it cannot be inferred from projection_npz filename.")
        jobs.append((sample_idx, projection_npz, image_feature_npz))
    else:
        if args.sample_idx is not None:
            sample_indices = [args.sample_idx]
        else:
            sample_indices = list(range(args.start_idx, args.start_idx + args.max_samples))
        projection_dir = Path(args.projection_dir).resolve()
        image_feature_dir = Path(args.image_feature_dir).resolve()
        for sample_idx in sample_indices:
            prefix = f"sample_{sample_idx:04d}"
            projection_npz = projection_dir / f"{prefix}_projection.npz"
            image_feature_npz = image_feature_dir / f"{prefix}_image_features.npz"
            if not projection_npz.exists():
                raise FileNotFoundError(f"projection npz not found: {projection_npz}")
            if not image_feature_npz.exists():
                raise FileNotFoundError(f"image feature npz not found: {image_feature_npz}")
            jobs.append((sample_idx, projection_npz, image_feature_npz))

    batch_summary = {
        "aggregation": args.aggregation,
        "jobs": [],
    }

    for sample_idx, projection_npz, image_feature_npz in jobs:
        logger.info("========== sample_idx=%d ==========", sample_idx)
        prefix = f"sample_{sample_idx:04d}"
        output_npz = output_dir / f"{prefix}_point_features.npz"
        summary_json = output_dir / f"{prefix}_point_features_summary.json"

        if args.skip_existing and output_npz.exists() and summary_json.exists():
            logger.info("skip existing point features for sample_idx=%d", sample_idx)
            batch_summary["jobs"].append(
                {
                    "sample_idx": sample_idx,
                    "status": "skipped_existing",
                    "output_npz": str(output_npz),
                }
            )
            continue

        projection_result = load_npz_as_dict(projection_npz)
        image_features = load_npz_as_dict(image_feature_npz)
        assignment = assigner.assign(projection_result, image_features)

        npz_data = {
            **assignment,
            "sample_idx": np.array(sample_idx, dtype=np.int32),
            "sample_token": projection_result["sample_token"],
            "teacher_backend": (
                image_features["teacher_backend"]
                if "teacher_backend" in image_features
                else np.array("legacy_unknown")
            ),
            "teacher_role": (
                image_features["teacher_role"]
                if "teacher_role" in image_features
                else np.array("legacy_unknown")
            ),
            "teacher_feature_granularity": (
                image_features["teacher_feature_granularity"]
                if "teacher_feature_granularity" in image_features
                else np.array("legacy_unknown")
            ),
            "is_baseline_teacher": (
                image_features["is_baseline_teacher"]
                if "is_baseline_teacher" in image_features
                else np.array(True)
            ),
            "model_name": image_features["model_name"],
            "cache_dir": image_features["cache_dir"] if "cache_dir" in image_features else np.array(""),
            "local_files_only": (
                image_features["local_files_only"]
                if "local_files_only" in image_features
                else np.array(False)
            ),
        }
        summary = {
            "sample_idx": sample_idx,
            "sample_token": str(projection_result["sample_token"].item()),
            "aggregation": args.aggregation,
            "num_points": int(assignment["point_xyz"].shape[0]),
            "num_valid_points": int(assignment["point_valid_mask"].sum()),
            "teacher_backend": (
                str(image_features["teacher_backend"].item())
                if "teacher_backend" in image_features
                else "legacy_unknown"
            ),
            "teacher_role": (
                str(image_features["teacher_role"].item())
                if "teacher_role" in image_features
                else "legacy_unknown"
            ),
            "teacher_feature_granularity": (
                str(image_features["teacher_feature_granularity"].item())
                if "teacher_feature_granularity" in image_features
                else "legacy_unknown"
            ),
            "is_baseline_teacher": (
                bool(image_features["is_baseline_teacher"].item())
                if "is_baseline_teacher" in image_features
                else True
            ),
            "model_name": str(image_features["model_name"].item()),
            "cache_dir": str(image_features["cache_dir"].item()) if "cache_dir" in image_features else "",
            "local_files_only": (
                bool(image_features["local_files_only"].item())
                if "local_files_only" in image_features
                else False
            ),
            "projection_npz": str(projection_npz),
            "image_feature_npz": str(image_feature_npz),
            "output_npz": str(output_npz),
        }
        save_npz(output_npz, **npz_data)
        save_json(summary_json, summary)
        logger.info(
            "point features saved to: %s | valid_points=%d / %d",
            output_npz,
            summary["num_valid_points"],
            summary["num_points"],
        )

        batch_summary["jobs"].append(
            {
                "sample_idx": sample_idx,
                "status": "done",
                "output_npz": str(output_npz),
                "summary_json": str(summary_json),
            }
        )

    if len(jobs) > 1:
        batch_summary_path = output_dir / "batch_point_features_summary.json"
        save_json(batch_summary_path, batch_summary)
        logger.info("batch point feature summary saved to: %s", batch_summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
