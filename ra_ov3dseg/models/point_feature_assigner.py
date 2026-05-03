from __future__ import annotations

from typing import Any

import numpy as np


def l2_normalize(features: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(features, axis=axis, keepdims=True)
    return features / np.clip(norms, eps, None)


class PointFeatureAssigner:
    """将 2D image patch feature 赋给 3D 点。

    MVP-v1 采用最简单的 nearest-patch assignment：
    1. 先根据投影点 `(u, v)` 找到其在 resize 后图像上的位置。
    2. 再映射到 patch feature grid 的 cell 索引。
    3. 如果一个点被多个相机看到，可选：
       - `mean`：对多相机特征取平均。
       - `closest_camera`：保留深度最小的那个相机特征。
    """

    def __init__(self, aggregation: str = "mean", normalize_output: bool = True) -> None:
        if aggregation not in {"mean", "closest_camera"}:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        self.aggregation = aggregation
        self.normalize_output = normalize_output

    def assign(self, projection_result: dict[str, Any], image_features: dict[str, Any]) -> dict[str, Any]:
        point_xyz = projection_result["point_xyz"].astype(np.float32)
        uv = projection_result["uv"].astype(np.float32)
        depth = projection_result["depth"].astype(np.float32)
        valid_masks = projection_result["valid_masks"].astype(bool)
        camera_names = [str(name) for name in projection_result["camera_names"].tolist()]

        feature_maps = image_features["feature_maps"].astype(np.float32)
        feature_camera_names = [str(name) for name in image_features["camera_names"].tolist()]
        if camera_names != feature_camera_names:
            raise ValueError("Camera order mismatch between projection result and image features.")

        image_widths = projection_result["image_widths"].astype(np.float32)
        image_heights = projection_result["image_heights"].astype(np.float32)
        resized_widths = image_features["resized_widths"].astype(np.float32)
        resized_heights = image_features["resized_heights"].astype(np.float32)
        camera_available = image_features["camera_available"].astype(bool)

        num_cameras, num_points, _ = uv.shape
        feature_dim = int(feature_maps.shape[-1])

        point_feature_sum = np.zeros((num_points, feature_dim), dtype=np.float32)
        point_feature_count = np.zeros(num_points, dtype=np.int32)
        point_depth_sum = np.zeros(num_points, dtype=np.float32)
        point_selected_camera = np.full(num_points, -1, dtype=np.int32)
        point_feature_closest = np.zeros((num_points, feature_dim), dtype=np.float32)
        point_depth_closest = np.full(num_points, np.inf, dtype=np.float32)

        for camera_idx in range(num_cameras):
            if not camera_available[camera_idx]:
                continue

            valid_mask = valid_masks[camera_idx]
            if not np.any(valid_mask):
                continue

            feature_map = feature_maps[camera_idx]
            grid_height, grid_width, _ = feature_map.shape
            point_indices = np.nonzero(valid_mask)[0]
            point_uv = uv[camera_idx, valid_mask]
            point_depth = depth[camera_idx, valid_mask]

            x_resized = point_uv[:, 0] * resized_widths[camera_idx] / max(image_widths[camera_idx], 1.0)
            y_resized = point_uv[:, 1] * resized_heights[camera_idx] / max(image_heights[camera_idx], 1.0)

            stride_x = resized_widths[camera_idx] / max(grid_width, 1)
            stride_y = resized_heights[camera_idx] / max(grid_height, 1)
            patch_x = np.floor(x_resized / max(stride_x, 1e-6)).astype(np.int32)
            patch_y = np.floor(y_resized / max(stride_y, 1e-6)).astype(np.int32)
            patch_x = np.clip(patch_x, 0, grid_width - 1)
            patch_y = np.clip(patch_y, 0, grid_height - 1)

            sampled_features = feature_map[patch_y, patch_x].astype(np.float32)

            if self.aggregation == "mean":
                point_feature_sum[point_indices] += sampled_features
                point_feature_count[point_indices] += 1
                point_depth_sum[point_indices] += point_depth
            else:
                better_mask = point_depth < point_depth_closest[point_indices]
                chosen_indices = point_indices[better_mask]
                point_feature_closest[chosen_indices] = sampled_features[better_mask]
                point_depth_closest[chosen_indices] = point_depth[better_mask]
                point_selected_camera[chosen_indices] = camera_idx
                point_feature_count[point_indices] += 1

        if self.aggregation == "mean":
            point_valid_mask = point_feature_count > 0
            point_features = np.zeros_like(point_feature_sum)
            point_features[point_valid_mask] = (
                point_feature_sum[point_valid_mask]
                / point_feature_count[point_valid_mask, None].astype(np.float32)
            )
            point_mean_depth = np.full(num_points, np.nan, dtype=np.float32)
            point_mean_depth[point_valid_mask] = (
                point_depth_sum[point_valid_mask]
                / point_feature_count[point_valid_mask].astype(np.float32)
            )
        else:
            point_valid_mask = point_selected_camera >= 0
            point_features = point_feature_closest
            point_mean_depth = np.where(point_valid_mask, point_depth_closest, np.nan).astype(np.float32)

        if self.normalize_output and np.any(point_valid_mask):
            point_features[point_valid_mask] = l2_normalize(point_features[point_valid_mask])

        return {
            "point_xyz": point_xyz.astype(np.float32),
            "point_features": point_features.astype(np.float32),
            "point_valid_mask": point_valid_mask.astype(bool),
            "point_camera_count": point_feature_count.astype(np.int32),
            "point_mean_depth": point_mean_depth.astype(np.float32),
            "selected_camera_index": point_selected_camera.astype(np.int32),
            "camera_names": np.asarray(camera_names),
            "aggregation": np.array(self.aggregation),
        }
