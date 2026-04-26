from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_mini_dataset import NuScenesMiniDataset  # noqa: E402
from ra_ov3dseg.geometry.projection import project_lidar_points_to_cameras  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 nuScenes LiDAR 点云投影到 6 个相机平面。")
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes 数据根目录。")
    parser.add_argument("--version", default="v1.0-mini", type=str, help="nuScenes 版本。")
    parser.add_argument("--sample_idx", default=0, type=int, help="按时间排序后的 sample 索引。")
    parser.add_argument(
        "--output_dir",
        default="outputs/projections",
        type=str,
        help="投影结果输出目录。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("project_lidar_to_cameras")

    dataset = NuScenesMiniDataset(
        dataroot=args.dataroot,
        version=args.version,
        verbose=False,
    )
    sample = dataset.get_sample_by_index(args.sample_idx)

    npz_data, summary = project_lidar_points_to_cameras(dataset, sample)

    output_dir = ensure_dir(args.output_dir)
    prefix = f"sample_{args.sample_idx:04d}"
    projection_npz = output_dir / f"{prefix}_projection.npz"
    summary_json = output_dir / f"{prefix}_projection_summary.json"

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
