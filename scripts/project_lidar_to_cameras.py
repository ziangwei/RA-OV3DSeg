from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset  # noqa: E402
from ra_ov3dseg.geometry.projection import project_lidar_points_to_cameras  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将一个或多个 nuScenes LiDAR sample 投影到 6 个相机平面。")
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes 数据根目录。")
    parser.add_argument("--version", default="v1.0-mini", type=str, help="nuScenes 版本。")
    parser.add_argument("--sample_idx", default=None, type=int, help="指定单个 sample 索引。")
    parser.add_argument("--start_idx", default=0, type=int, help="批量投影时的起始 sample 索引。")
    parser.add_argument("--max_samples", default=1, type=int, help="批量投影多少个 sample。")
    parser.add_argument(
        "--output_dir",
        default="outputs/projections",
        type=str,
        help="投影结果输出目录。",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="如果目标 .npz 和 summary 已存在，则跳过该 sample。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("project_lidar_to_cameras")

    dataset = NuScenesDataset(
        dataroot=args.dataroot,
        version=args.version,
        verbose=False,
    )
    sample_indices = dataset.resolve_sample_indices(
        sample_idx=args.sample_idx,
        start_idx=args.start_idx,
        max_samples=args.max_samples,
    )
    output_dir = ensure_dir(args.output_dir)
    batch_summary = {
        "version": args.version,
        "dataroot": str(Path(args.dataroot).resolve()),
        "num_total_samples": len(dataset),
        "requested_sample_indices": sample_indices,
        "projection_output_dir": str(output_dir.resolve()),
        "samples": [],
    }

    for sample_idx in sample_indices:
        logger.info("========== sample_idx=%d ==========", sample_idx)
        prefix = f"sample_{sample_idx:04d}"
        projection_npz = output_dir / f"{prefix}_projection.npz"
        summary_json = output_dir / f"{prefix}_projection_summary.json"

        if args.skip_existing and projection_npz.exists() and summary_json.exists():
            logger.info("skip existing projection outputs for sample_idx=%d", sample_idx)
            batch_summary["samples"].append(
                {
                    "sample_idx": sample_idx,
                    "projection_npz": str(projection_npz),
                    "summary_json": str(summary_json),
                    "status": "skipped_existing",
                }
            )
            continue

        sample = dataset.get_sample_by_index(sample_idx)
        npz_data, summary = project_lidar_points_to_cameras(dataset, sample)

        save_npz(projection_npz, **npz_data)
        save_json(summary_json, summary)

        logger.info("projection npz saved to: %s", projection_npz)
        logger.info("projection summary saved to: %s", summary_json)

        for camera_summary in summary["cameras"]:
            logger.info(
                "%s | available=%s | total=%d | depth>0=%d | in_image=%d | valid_ratio=%.6f",
                camera_summary["camera_name"],
                camera_summary["available"],
                camera_summary["total_points"],
                camera_summary["positive_depth_points"],
                camera_summary["inside_image_points"],
                camera_summary["valid_projection_ratio"],
            )

        batch_summary["samples"].append(
            {
                "sample_idx": sample_idx,
                "projection_npz": str(projection_npz),
                "summary_json": str(summary_json),
                "status": "done",
                "summary": summary,
            }
        )

    if len(sample_indices) > 1:
        batch_summary_path = output_dir / "batch_projection_summary.json"
        save_json(batch_summary_path, batch_summary)
        logger.info("batch projection summary saved to: %s", batch_summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
