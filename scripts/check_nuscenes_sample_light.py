from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
TARGET_CHANNELS = ["LIDAR_TOP", *CAMERA_CHANNELS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight nuScenes sample checker. This avoids NuScenes devkit full "
            "metadata indexing and is intended for trainval checks on memory-limited nodes."
        )
    )
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes root directory.")
    parser.add_argument("--version", default="v1.0-trainval", type=str, help="nuScenes version.")
    parser.add_argument("--sample_idx", default=0, type=int, help="Sample index to inspect.")
    parser.add_argument(
        "--output_dir",
        default="outputs/checks_light",
        type=str,
        help="Directory for JSON summary output.",
    )
    return parser


def iter_json_objects(json_path: Path) -> Iterable[dict]:
    """Stream top-level JSON-array objects without loading the full table into RAM.

    nuScenes metadata tables are JSON arrays of objects. The standard devkit loads
    every table, including large trainval annotation tables. Here we scan only the
    required tables and emit one object at a time, so memory stays near one record.
    """

    depth = 0
    collecting = False
    in_string = False
    escape = False
    chars: list[str] = []

    with json_path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break

            for char in chunk:
                if not collecting:
                    if char == "{":
                        collecting = True
                        depth = 1
                        in_string = False
                        escape = False
                        chars = ["{"]
                    continue

                chars.append(char)

                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        yield json.loads("".join(chars))
                        collecting = False
                        chars = []


def find_by_index(json_path: Path, index: int) -> dict:
    for current_index, record in enumerate(iter_json_objects(json_path)):
        if current_index == index:
            return record
    raise IndexError(f"sample_idx out of range: {index}")


def find_by_token(json_path: Path, token: str) -> dict | None:
    for record in iter_json_objects(json_path):
        if record.get("token") == token:
            return record
    return None


def find_sample_data_by_channel_for_sample(
    json_path: Path,
    sample_token: str,
    calibrated_token_to_channel: dict[str, str],
    target_channels: list[str],
) -> tuple[dict[str, dict], int]:
    records_by_channel: dict[str, dict] = {}
    num_matching_keyframes = 0
    target_channel_set = set(target_channels)

    for record in iter_json_objects(json_path):
        if record.get("sample_token") == sample_token and bool(record.get("is_key_frame", False)):
            num_matching_keyframes += 1
            channel = calibrated_token_to_channel.get(record.get("calibrated_sensor_token", ""))
            if channel in target_channel_set and channel not in records_by_channel:
                records_by_channel[channel] = record
                if len(records_by_channel) == len(target_channel_set):
                    break
    return records_by_channel, num_matching_keyframes


def load_sensor_channels(json_path: Path) -> dict[str, str]:
    token_to_channel: dict[str, str] = {}
    for record in iter_json_objects(json_path):
        token = record.get("token")
        channel = record.get("channel")
        if token and channel:
            token_to_channel[token] = channel
    return token_to_channel


def load_calibrated_sensor_channels(
    json_path: Path,
    sensor_token_to_channel: dict[str, str],
) -> dict[str, str]:
    calibrated_token_to_channel: dict[str, str] = {}
    for record in iter_json_objects(json_path):
        token = record.get("token")
        sensor_token = record.get("sensor_token")
        channel = sensor_token_to_channel.get(sensor_token or "")
        if token and channel:
            calibrated_token_to_channel[token] = channel
    return calibrated_token_to_channel


def find_lidarseg_record(json_path: Path, lidar_sample_data_token: str) -> dict | None:
    for record in iter_json_objects(json_path):
        if record.get("sample_data_token") == lidar_sample_data_token:
            return record
    return None


def count_lidar_points(lidar_path: Path) -> tuple[int | None, str]:
    if not lidar_path.is_file():
        return None, "missing"

    file_size = lidar_path.stat().st_size
    point_stride_bytes = 5 * 4
    if file_size % point_stride_bytes != 0:
        return None, f"invalid_size_bytes={file_size}"
    return file_size // point_stride_bytes, "ok"


def count_lidarseg_labels(label_path: Path) -> tuple[int | None, str]:
    if not label_path.is_file():
        return None, "missing"
    # nuScenes lidarseg labels are uint8, one byte per LiDAR point.
    return label_path.stat().st_size, "ok"


def resolve_data_path(dataroot: Path, filename: str | None) -> Path | None:
    if not filename:
        return None
    path = Path(filename)
    if path.is_absolute():
        return path
    return dataroot / path


def info(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)


def main() -> int:
    args = build_parser().parse_args()
    dataroot = Path(args.dataroot).expanduser().resolve()
    version_dir = dataroot / args.version
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_json = version_dir / "sample.json"
    scene_json = version_dir / "scene.json"
    sample_data_json = version_dir / "sample_data.json"
    calibrated_sensor_json = version_dir / "calibrated_sensor.json"
    sensor_json = version_dir / "sensor.json"
    lidarseg_json = version_dir / "lidarseg.json"

    for required_path in [sample_json, scene_json, sample_data_json, calibrated_sensor_json, sensor_json]:
        if not required_path.exists():
            error(f"missing metadata file: {required_path}")
            return 2

    info(f"dataroot={dataroot}")
    info(f"version={args.version}")
    info(f"sample_idx={args.sample_idx}")

    sample = find_by_index(sample_json, args.sample_idx)
    sample_token = sample["token"]
    scene_token = sample["scene_token"]
    timestamp = int(sample["timestamp"])
    scene = find_by_token(scene_json, scene_token) or {}
    info(f"sample token: {sample_token}")
    info(f"scene token: {scene_token}")
    if scene:
        info(f"scene name: {scene.get('name', '')}")
    info(f"timestamp: {timestamp}")

    info("loading sensor calibration metadata")
    sensor_token_to_channel = load_sensor_channels(sensor_json)
    calibrated_token_to_channel = load_calibrated_sensor_channels(
        calibrated_sensor_json,
        sensor_token_to_channel,
    )
    info("scanning sample_data.json for target keyframe sensors")
    sample_data_records_by_channel, num_matching_keyframes = find_sample_data_by_channel_for_sample(
        sample_data_json,
        sample_token,
        calibrated_token_to_channel,
        TARGET_CHANNELS,
    )
    info(
        "target keyframe sample_data records found: "
        f"{len(sample_data_records_by_channel)}/{len(TARGET_CHANNELS)} "
        f"(matching_keyframes_seen={num_matching_keyframes})"
    )

    sensor_paths: dict[str, str] = {}
    sensor_status: dict[str, dict] = {}
    exit_code = 0

    for sensor_name in TARGET_CHANNELS:
        record = sample_data_records_by_channel.get(sensor_name)
        token = record.get("token") if record else None
        exists_in_sample = token is not None
        data_path = resolve_data_path(dataroot, record.get("filename") if record else None)
        file_exists = bool(data_path and data_path.exists())

        if data_path is not None:
            sensor_paths[sensor_name] = str(data_path)
        else:
            sensor_paths[sensor_name] = ""

        sensor_status[sensor_name] = {
            "token": token or "",
            "exists_in_sample": exists_in_sample,
            "sample_data_record_found": record is not None,
            "file_exists": file_exists,
            "path": sensor_paths[sensor_name],
        }

        info(
            f"{sensor_name}: exists_in_sample={exists_in_sample} "
            f"record_found={record is not None} file_exists={file_exists} "
            f"path={sensor_paths[sensor_name] or 'missing'}"
        )

        if not exists_in_sample or record is None or not file_exists:
            exit_code = max(exit_code, 2)

    lidar_path = Path(sensor_paths.get("LIDAR_TOP", ""))
    lidar_point_count, lidar_count_status = count_lidar_points(lidar_path)
    info(f"LiDAR point count: {lidar_point_count} ({lidar_count_status})")
    if lidar_point_count is None:
        exit_code = max(exit_code, 2)

    lidarseg_label_path = ""
    lidarseg_label_count = None
    label_check = "not_run"

    lidar_record = sample_data_records_by_channel.get("LIDAR_TOP")
    lidar_token = lidar_record.get("token") if lidar_record else None
    if not lidarseg_json.exists() or not lidar_token:
        warn("lidarseg labels not found, skip label check.")
        label_check = "labels_not_found"
    else:
        lidarseg_record = find_lidarseg_record(lidarseg_json, lidar_token)
        lidarseg_path = resolve_data_path(
            dataroot, lidarseg_record.get("filename") if lidarseg_record else None
        )
        if lidarseg_path is None:
            warn("lidarseg labels not found, skip label check.")
            label_check = "labels_not_found"
        else:
            lidarseg_label_path = str(lidarseg_path)
            lidarseg_label_count, label_count_status = count_lidarseg_labels(lidarseg_path)
            info(f"lidarseg label path: {lidarseg_path}")
            info(f"lidarseg label count: {lidarseg_label_count} ({label_count_status})")

            if lidarseg_label_count is None:
                label_check = "labels_not_found"
            elif lidar_point_count == lidarseg_label_count:
                label_check = "pass"
                info("label check: PASS (point count matches lidarseg count)")
            else:
                label_check = "fail"
                error(
                    "label check: FAIL "
                    f"(LiDAR point count={lidar_point_count}, lidarseg count={lidarseg_label_count})"
                )
                exit_code = max(exit_code, 2)

    summary = {
        "mode": "lightweight_no_devkit",
        "dataroot": str(dataroot),
        "version": args.version,
        "sample_idx": args.sample_idx,
        "sample_token": sample_token,
        "scene_token": scene_token,
        "scene_name": scene.get("name", ""),
        "timestamp": timestamp,
        "sensor_status": sensor_status,
        "sensor_paths": sensor_paths,
        "lidar_point_count": lidar_point_count,
        "lidarseg_label_path": lidarseg_label_path,
        "lidarseg_label_count": lidarseg_label_count,
        "label_check": label_check,
        "status": "ok" if exit_code == 0 else "failed",
    }
    summary_path = output_dir / f"sample_{args.sample_idx:04d}_check_light_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    info(f"light check summary saved to: {summary_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
