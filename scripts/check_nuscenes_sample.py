from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import CAMERA_CHANNELS, NuScenesDataset  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查一个或多个 nuScenes sample 的传感器与标签状态。")
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes 数据根目录。")
    parser.add_argument("--version", default="v1.0-mini", type=str, help="nuScenes 版本。")
    parser.add_argument("--sample_idx", default=None, type=int, help="指定单个 sample 索引。")
    parser.add_argument("--start_idx", default=0, type=int, help="批量检查时的起始 sample 索引。")
    parser.add_argument("--max_samples", default=1, type=int, help="批量检查多少个 sample。")
    parser.add_argument(
        "--output_dir",
        default="outputs/checks",
        type=str,
        help="检查结果 JSON 输出目录。",
    )
    return parser


def inspect_sample(dataset: NuScenesDataset, sample_idx: int, logger) -> tuple[dict, int]:
    sample = dataset.get_sample_by_index(sample_idx)
    scene = dataset.get_scene_record(sample)

    logger.info("sample_idx: %d", sample_idx)
    logger.info("sample token: %s", sample["token"])
    logger.info("scene token: %s", scene["token"])
    logger.info("scene name: %s", scene["name"])
    logger.info("timestamp: %s", sample["timestamp"])

    lidar_token = dataset.get_sensor_token(sample, "LIDAR_TOP")
    logger.info("LIDAR_TOP exists: %s", "yes" if lidar_token else "no")

    camera_status = {}
    for camera_name in CAMERA_CHANNELS:
        camera_token = dataset.get_sensor_token(sample, camera_name)
        logger.info("%s exists: %s", camera_name, "yes" if camera_token else "no")
        camera_status[camera_name] = camera_token is not None

    logger.info("sensor file paths:")
    sensor_paths = {}
    for sensor_name in ["LIDAR_TOP", *CAMERA_CHANNELS]:
        sensor_path = dataset.get_sample_data_path_from_channel(sample, sensor_name)
        if sensor_path is None:
            logger.info("  - %s: missing", sensor_name)
            sensor_paths[sensor_name] = ""
        else:
            logger.info("  - %s: %s", sensor_name, sensor_path)
            sensor_paths[sensor_name] = str(sensor_path)

    summary = {
        "sample_idx": sample_idx,
        "sample_token": sample["token"],
        "scene_token": scene["token"],
        "scene_name": scene["name"],
        "timestamp": int(sample["timestamp"]),
        "lidar_top_exists": lidar_token is not None,
        "camera_status": camera_status,
        "sensor_paths": sensor_paths,
        "lidar_point_count": None,
        "lidarseg_label_path": "",
        "lidarseg_label_count": None,
        "label_check": "not_run",
        "status": "ok",
    }

    if lidar_token is None:
        logger.error("sample does not contain LIDAR_TOP, stop.")
        summary["status"] = "missing_lidar_top"
        return summary, 1

    lidar_points = dataset.load_lidar_points(sample)
    logger.info("LiDAR point count: %d", lidar_points.shape[0])
    summary["lidar_point_count"] = int(lidar_points.shape[0])

    lidarseg_path = dataset.get_lidarseg_path(sample)
    if lidarseg_path is None:
        logger.warning("lidarseg labels not found, skip label check.")
        summary["label_check"] = "labels_not_found"
        return summary, 0

    lidar_labels = dataset.load_lidarseg_labels(sample)
    if lidar_labels is None:
        logger.warning("lidarseg labels not found, skip label check.")
        summary["label_check"] = "labels_not_found"
        return summary, 0

    logger.info("lidarseg label path: %s", lidarseg_path)
    logger.info("lidarseg label count: %d", lidar_labels.shape[0])
    summary["lidarseg_label_path"] = str(lidarseg_path)
    summary["lidarseg_label_count"] = int(lidar_labels.shape[0])

    if lidar_labels.shape[0] == lidar_points.shape[0]:
        logger.info("label check: PASS (point count matches lidarseg count)")
        summary["label_check"] = "pass"
        return summary, 0

    logger.error(
        "label check: FAIL (LiDAR point count=%d, lidarseg count=%d)",
        lidar_points.shape[0],
        lidar_labels.shape[0],
    )
    summary["label_check"] = "fail"
    summary["status"] = "label_count_mismatch"
    return summary, 2


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("check_nuscenes_sample")

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
        "samples": [],
    }
    max_exit_code = 0

    for sample_idx in sample_indices:
        logger.info("========== sample_idx=%d ==========", sample_idx)
        summary, exit_code = inspect_sample(dataset, sample_idx, logger)
        summary_path = output_dir / f"sample_{sample_idx:04d}_check_summary.json"
        save_json(summary_path, summary)
        logger.info("check summary saved to: %s", summary_path)
        batch_summary["samples"].append(summary)
        max_exit_code = max(max_exit_code, exit_code)

    if len(sample_indices) > 1:
        batch_summary_path = output_dir / "batch_check_summary.json"
        save_json(batch_summary_path, batch_summary)
        logger.info("batch check summary saved to: %s", batch_summary_path)

    return max_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
