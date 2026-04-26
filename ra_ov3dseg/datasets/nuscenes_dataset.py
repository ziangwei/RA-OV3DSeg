from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class NuScenesDataset:
    """nuScenes 数据访问封装。

    当前工程虽然先以 `v1.0-mini` 做链路验证，但底层接口按通用 `version`
    设计，后续切换到 `v1.0-trainval` 时不需要重写数据读取逻辑。
    """

    def __init__(self, dataroot: str | Path, version: str = "v1.0-mini", verbose: bool = False) -> None:
        self.dataroot = Path(dataroot).expanduser().resolve()
        if not self.dataroot.exists():
            raise FileNotFoundError(f"nuScenes dataroot does not exist: {self.dataroot}")

        self.version = version
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=verbose)
        self.samples = sorted(
            self.nusc.sample,
            key=lambda record: (record["scene_token"], record["timestamp"]),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def get_sample_by_index(self, sample_idx: int) -> dict[str, Any]:
        if sample_idx < 0 or sample_idx >= len(self.samples):
            raise IndexError(f"sample_idx out of range: {sample_idx}, total samples={len(self.samples)}")
        return self.samples[sample_idx]

    def resolve_sample_indices(
        self,
        sample_idx: int | None = None,
        start_idx: int = 0,
        max_samples: int | None = 1,
    ) -> list[int]:
        """统一处理单样本与小批量样本选择逻辑。"""

        if sample_idx is not None:
            self.get_sample_by_index(sample_idx)
            return [sample_idx]

        if start_idx < 0 or start_idx >= len(self.samples):
            raise IndexError(f"start_idx out of range: {start_idx}, total samples={len(self.samples)}")

        if max_samples is None:
            end_idx = len(self.samples)
        else:
            if max_samples <= 0:
                raise ValueError(f"max_samples must be positive or None, got {max_samples}")
            end_idx = min(start_idx + max_samples, len(self.samples))

        return list(range(start_idx, end_idx))

    def get_scene_record(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self.nusc.get("scene", sample["scene_token"])

    def get_sensor_token(self, sample: dict[str, Any], channel: str) -> str | None:
        return sample["data"].get(channel)

    def get_sample_data_record_from_channel(self, sample: dict[str, Any], channel: str) -> dict[str, Any] | None:
        token = self.get_sensor_token(sample, channel)
        if token is None:
            return None
        return self.nusc.get("sample_data", token)

    def get_sample_data_path_from_channel(self, sample: dict[str, Any], channel: str) -> Path | None:
        record = self.get_sample_data_record_from_channel(sample, channel)
        if record is None:
            return None
        return (self.dataroot / record["filename"]).resolve()

    def get_sample_data_relpath_from_channel(self, sample: dict[str, Any], channel: str) -> str | None:
        record = self.get_sample_data_record_from_channel(sample, channel)
        if record is None:
            return None
        return str(record["filename"])

    def load_lidar_points(self, sample: dict[str, Any]) -> np.ndarray:
        lidar_path = self.get_sample_data_path_from_channel(sample, "LIDAR_TOP")
        if lidar_path is None:
            raise KeyError("sample does not contain LIDAR_TOP")
        if not lidar_path.exists():
            raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")

        lidar_point_cloud = LidarPointCloud.from_file(str(lidar_path))
        return lidar_point_cloud.points[:3, :].T.astype(np.float32)

    def get_lidarseg_record(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        lidar_token = self.get_sensor_token(sample, "LIDAR_TOP")
        if lidar_token is None:
            return None
        try:
            return self.nusc.get("lidarseg", lidar_token)
        except Exception:
            return None

    def get_lidarseg_path(self, sample: dict[str, Any]) -> Path | None:
        record = self.get_lidarseg_record(sample)
        if record is None:
            return None
        label_path = (self.dataroot / record["filename"]).resolve()
        if not label_path.exists():
            return None
        return label_path

    def load_lidarseg_labels(self, sample: dict[str, Any]) -> np.ndarray | None:
        label_path = self.get_lidarseg_path(sample)
        if label_path is None:
            return None
        return np.fromfile(label_path, dtype=np.uint8)
