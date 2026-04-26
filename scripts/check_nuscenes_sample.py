from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_mini_dataset import (  # noqa: E402
    CAMERA_CHANNELS,
    NuScenesMiniDataset,
)
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 nuScenes 单个 sample 的传感器与标签状态。")
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes 数据根目录。")
    parser.add_argument("--version", default="v1.0-mini", type=str, help="nuScenes 版本。")
    parser.add_argument("--sample_idx", default=0, type=int, help="按时间排序后的 sample 索引。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("check_nuscenes_sample")

    dataset = NuScenesMiniDataset(
        dataroot=args.dataroot,
        version=args.version,
        verbose=False,
    )
    sample = dataset.get_sample_by_index(args.sample_idx)
    scene = dataset.get_scene_record(sample)

    logger.info("sample_idx: %d", args.sample_idx)
    logger.info("sample token: %s", sample["token"])
    logger.info("scene token: %s", scene["token"])
    logger.info("scene name: %s", scene["name"])
    logger.info("timestamp: %s", sample["timestamp"])

    lidar_token = dataset.get_sensor_token(sample, "LIDAR_TOP")
    logger.info("LIDAR_TOP exists: %s", "yes" if lidar_token else "no")

    for camera_name in CAMERA_CHANNELS:
        camera_token = dataset.get_sensor_token(sample, camera_name)
        logger.info("%s exists: %s", camera_name, "yes" if camera_token else "no")

    logger.info("sensor file paths:")
    for sensor_name in ["LIDAR_TOP", *CAMERA_CHANNELS]:
        sensor_path = dataset.get_sample_data_path_from_channel(sample, sensor_name)
        if sensor_path is None:
            logger.info("  - %s: missing", sensor_name)
        else:
            logger.info("  - %s: %s", sensor_name, sensor_path)

    if lidar_token is None:
        logger.error("sample does not contain LIDAR_TOP, stop.")
        return 1

    lidar_points = dataset.load_lidar_points(sample)
    logger.info("LiDAR point count: %d", lidar_points.shape[0])

    lidarseg_path = dataset.get_lidarseg_path(sample)
    if lidarseg_path is None:
        logger.warning("lidarseg labels not found, skip label check.")
        return 0

    lidar_labels = dataset.load_lidarseg_labels(sample)
    if lidar_labels is None:
        logger.warning("lidarseg labels not found, skip label check.")
        return 0

    logger.info("lidarseg label path: %s", lidarseg_path)
    logger.info("lidarseg label count: %d", lidar_labels.shape[0])

    if lidar_labels.shape[0] == lidar_points.shape[0]:
        logger.info("label check: PASS (point count matches lidarseg count)")
        return 0

    logger.error(
        "label check: FAIL (LiDAR point count=%d, lidarseg count=%d)",
        lidar_points.shape[0],
        lidar_labels.shape[0],
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
