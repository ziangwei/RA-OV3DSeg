from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ra_ov3dseg.datasets.nuscenes_mini_dataset import CAMERA_CHANNELS, NuScenesMiniDataset
from ra_ov3dseg.geometry.transforms import inverse_transform_points, transform_points


def _load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def project_lidar_points_to_cameras(
    dataset: NuScenesMiniDataset,
    sample: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """将 LiDAR 点云投影到 6 个相机平面，并返回可保存的数组与摘要。

    坐标变换链路说明：

    1. LiDAR 坐标系：
       原始点云文件中的点，位于 `LIDAR_TOP` 传感器自身坐标系。
    2. ego 坐标系：
       先通过 `LIDAR_TOP` 的外参，把点从 LiDAR 坐标系变到该时刻自车坐标系。
    3. global 坐标系：
       再通过 LiDAR 时刻的 `ego_pose`，把点从自车坐标系变到全局坐标系。
    4. camera 时刻 ego 坐标系：
       因为相机和 LiDAR 可能不是完全同一时刻采样，所以需要用相机自己的 `ego_pose`
       把 global 坐标系下的点逆变换到相机时刻的自车坐标系。
    5. camera 坐标系：
       再通过相机外参的逆变换，把点从相机时刻 ego 坐标系变到相机坐标系。
    6. camera intrinsic 投影：
       对相机坐标系下的 3D 点应用相机内参矩阵 `K`，得到齐次像素坐标，再除以深度 `z`
       得到最终的 `(u, v)` 图像坐标。
    """

    lidar_token = dataset.get_sensor_token(sample, "LIDAR_TOP")
    if lidar_token is None:
        raise KeyError("sample does not contain LIDAR_TOP")

    point_xyz = dataset.load_lidar_points(sample)
    num_points = point_xyz.shape[0]

    nusc = dataset.nusc
    lidar_sd = nusc.get("sample_data", lidar_token)
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    # 先把 LiDAR 点统一提升到 global 坐标系。
    # 后面投影到每个相机时，只需要再从 global 逆变换到对应相机坐标系即可。
    points_ego_lidar = transform_points(point_xyz, lidar_cs["rotation"], lidar_cs["translation"])
    points_global = transform_points(points_ego_lidar, lidar_pose["rotation"], lidar_pose["translation"])

    camera_names = list(CAMERA_CHANNELS)
    uv = np.full((len(camera_names), num_points, 2), np.nan, dtype=np.float32)
    depth = np.full((len(camera_names), num_points), np.nan, dtype=np.float32)
    positive_depth_masks = np.zeros((len(camera_names), num_points), dtype=bool)
    inside_image_masks = np.zeros((len(camera_names), num_points), dtype=bool)
    valid_masks = np.zeros((len(camera_names), num_points), dtype=bool)
    visible_camera_count = np.zeros(num_points, dtype=np.int32)
    image_widths = np.zeros(len(camera_names), dtype=np.int32)
    image_heights = np.zeros(len(camera_names), dtype=np.int32)
    image_rel_paths: list[str] = []
    summary_cameras: list[dict[str, Any]] = []

    for camera_idx, camera_name in enumerate(camera_names):
        camera_token = dataset.get_sensor_token(sample, camera_name)
        if camera_token is None:
            image_rel_paths.append("")
            summary_cameras.append(
                {
                    "camera_name": camera_name,
                    "available": False,
                    "image_path": "",
                    "image_width": 0,
                    "image_height": 0,
                    "total_points": int(num_points),
                    "positive_depth_points": 0,
                    "inside_image_points": 0,
                    "valid_projection_ratio": 0.0,
                }
            )
            continue

        camera_sd = nusc.get("sample_data", camera_token)
        camera_cs = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
        camera_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])
        image_path = dataset.get_sample_data_path_from_channel(sample, camera_name)
        image_rel_path = dataset.get_sample_data_relpath_from_channel(sample, camera_name)

        if image_path is None or image_rel_path is None:
            image_rel_paths.append("")
            summary_cameras.append(
                {
                    "camera_name": camera_name,
                    "available": False,
                    "image_path": "",
                    "image_width": 0,
                    "image_height": 0,
                    "total_points": int(num_points),
                    "positive_depth_points": 0,
                    "inside_image_points": 0,
                    "valid_projection_ratio": 0.0,
                }
            )
            continue

        if not image_path.exists():
            raise FileNotFoundError(f"camera image not found: {image_path}")

        image_width, image_height = _load_image_size(image_path)
        image_widths[camera_idx] = image_width
        image_heights[camera_idx] = image_height
        image_rel_paths.append(image_rel_path)

        # global -> 相机时刻 ego -> 相机坐标系
        points_ego_camera = inverse_transform_points(
            points_global,
            camera_pose["rotation"],
            camera_pose["translation"],
        )
        points_camera = inverse_transform_points(
            points_ego_camera,
            camera_cs["rotation"],
            camera_cs["translation"],
        )

        camera_depth = points_camera[:, 2].astype(np.float32)
        depth[camera_idx] = camera_depth

        positive_depth = camera_depth > 1e-6
        positive_depth_masks[camera_idx] = positive_depth

        if np.any(positive_depth):
            # 相机内参投影：
            # [u', v', w'] = K * [x, y, z]
            # 最终像素坐标为 [u, v] = [u'/w', v'/w']，在 pinhole 模型下 w' = z。
            intrinsic = np.asarray(camera_cs["camera_intrinsic"], dtype=np.float32)
            homogeneous_pixels = points_camera[positive_depth] @ intrinsic.T
            uv[camera_idx, positive_depth] = homogeneous_pixels[:, :2] / homogeneous_pixels[:, 2:3]

        inside_image = (
            positive_depth
            & (uv[camera_idx, :, 0] >= 0.0)
            & (uv[camera_idx, :, 0] < image_width)
            & (uv[camera_idx, :, 1] >= 0.0)
            & (uv[camera_idx, :, 1] < image_height)
        )

        inside_image_masks[camera_idx] = inside_image
        valid_masks[camera_idx] = inside_image
        visible_camera_count += inside_image.astype(np.int32)

        summary_cameras.append(
            {
                "camera_name": camera_name,
                "available": True,
                "image_path": str(image_path),
                "image_width": int(image_width),
                "image_height": int(image_height),
                "total_points": int(num_points),
                "positive_depth_points": int(positive_depth.sum()),
                "inside_image_points": int(inside_image.sum()),
                "valid_projection_ratio": float(inside_image.sum() / max(num_points, 1)),
            }
        )

    scene = dataset.get_scene_record(sample)
    npz_data = {
        "sample_token": np.array(sample["token"]),
        "scene_token": np.array(scene["token"]),
        "timestamp": np.array(sample["timestamp"], dtype=np.int64),
        "point_xyz": point_xyz.astype(np.float32),
        "camera_names": np.asarray(camera_names),
        "image_rel_paths": np.asarray(image_rel_paths),
        "image_widths": image_widths,
        "image_heights": image_heights,
        "uv": uv,
        "depth": depth,
        "positive_depth_masks": positive_depth_masks,
        "inside_image_masks": inside_image_masks,
        "valid_masks": valid_masks,
        "visible_camera_count": visible_camera_count,
    }
    summary = {
        "sample_token": sample["token"],
        "scene_token": scene["token"],
        "scene_name": scene["name"],
        "timestamp": int(sample["timestamp"]),
        "num_points": int(num_points),
        "num_cameras": len(camera_names),
        "cameras": summary_cameras,
    }
    return npz_data, summary
