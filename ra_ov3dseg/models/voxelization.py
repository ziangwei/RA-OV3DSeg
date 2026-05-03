from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VoxelizationConfig:
    voxel_size: tuple[float, float, float] = (0.2, 0.2, 0.2)
    point_cloud_range: tuple[float, float, float, float, float, float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)

    @property
    def grid_size_xyz(self) -> tuple[int, int, int]:
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        vx, vy, vz = self.voxel_size
        return (
            int((x_max - x_min) / vx),
            int((y_max - y_min) / vy),
            int((z_max - z_min) / vz),
        )

    @property
    def spatial_shape_zyx(self) -> list[int]:
        grid_x, grid_y, grid_z = self.grid_size_xyz
        return [grid_z, grid_y, grid_x]


def make_point_input_features(
    point_xyz: torch.Tensor,
    point_cloud_range: tuple[float, float, float, float, float, float],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build lightweight per-point input features for sparse voxel aggregation."""

    pc_range = torch.as_tensor(point_cloud_range, dtype=point_xyz.dtype, device=point_xyz.device)
    pc_min = pc_range[:3]
    pc_max = pc_range[3:]
    center = (pc_min + pc_max) * 0.5
    scale = torch.clamp(pc_max - pc_min, min=eps)
    xyz_normalized = (point_xyz - center) / scale
    distance = torch.linalg.norm(point_xyz, dim=1, keepdim=True) / torch.clamp(torch.linalg.norm(scale), min=eps)
    return torch.cat([xyz_normalized, distance], dim=1)


def hash_spconv_indices(indices: torch.Tensor, spatial_shape_zyx: list[int]) -> torch.Tensor:
    """Hash [batch, z, y, x] sparse indices for coordinate matching on GPU."""

    z_size, y_size, x_size = [int(value) for value in spatial_shape_zyx]
    indices_long = indices.long()
    return (
        ((indices_long[:, 0] * (z_size + 1) + indices_long[:, 1]) * (y_size + 1) + indices_long[:, 2])
        * (x_size + 1)
        + indices_long[:, 3]
    )


def voxelize_point_features(
    point_xyz: torch.Tensor,
    point_features: torch.Tensor,
    point_batch_indices: torch.Tensor,
    config: VoxelizationConfig,
) -> dict[str, torch.Tensor | list[int]]:
    """Voxelize batched points and average point features inside each voxel.

    Coordinates follow spconv's convention: [batch_idx, z_idx, y_idx, x_idx].
    """

    if point_xyz.ndim != 2 or point_xyz.shape[1] != 3:
        raise ValueError(f"point_xyz must have shape [N, 3], got {tuple(point_xyz.shape)}")
    if point_features.ndim != 2 or point_features.shape[0] != point_xyz.shape[0]:
        raise ValueError("point_features must have shape [N, C] and align with point_xyz.")
    if point_batch_indices.ndim != 1 or point_batch_indices.shape[0] != point_xyz.shape[0]:
        raise ValueError("point_batch_indices must have shape [N].")

    device = point_xyz.device
    voxel_size = torch.as_tensor(config.voxel_size, dtype=point_xyz.dtype, device=device)
    pc_range = torch.as_tensor(config.point_cloud_range, dtype=point_xyz.dtype, device=device)
    pc_min = pc_range[:3]
    pc_max = pc_range[3:]
    grid_size_xyz = torch.as_tensor(config.grid_size_xyz, dtype=torch.long, device=device)

    coords_xyz = torch.floor((point_xyz - pc_min) / voxel_size).long()
    valid_point_mask = torch.all((point_xyz >= pc_min) & (point_xyz < pc_max), dim=1)
    valid_point_mask &= torch.all((coords_xyz >= 0) & (coords_xyz < grid_size_xyz), dim=1)

    point_to_voxel = torch.full((point_xyz.shape[0],), -1, dtype=torch.long, device=device)
    if not torch.any(valid_point_mask):
        raise ValueError("No points fall inside the configured point_cloud_range.")

    valid_indices = torch.nonzero(valid_point_mask, as_tuple=False).squeeze(1)
    valid_coords_xyz = coords_xyz[valid_indices]
    valid_batch = point_batch_indices[valid_indices].long()
    valid_coords_zyx = torch.stack(
        [
            valid_batch,
            valid_coords_xyz[:, 2],
            valid_coords_xyz[:, 1],
            valid_coords_xyz[:, 0],
        ],
        dim=1,
    )

    voxel_coords, inverse = torch.unique(valid_coords_zyx, sorted=True, return_inverse=True, dim=0)
    voxel_count = int(voxel_coords.shape[0])
    feature_dim = int(point_features.shape[1])
    voxel_features = point_features.new_zeros((voxel_count, feature_dim))
    voxel_features.index_add_(0, inverse, point_features[valid_indices])
    counts = point_features.new_zeros((voxel_count, 1))
    counts.index_add_(0, inverse, torch.ones((valid_indices.shape[0], 1), dtype=point_features.dtype, device=device))
    voxel_features = voxel_features / torch.clamp(counts, min=1.0)
    point_to_voxel[valid_indices] = inverse.long()

    return {
        "voxel_features": voxel_features,
        "voxel_coords": voxel_coords.int(),
        "point_to_voxel": point_to_voxel,
        "valid_point_mask": valid_point_mask,
        "spatial_shape": config.spatial_shape_zyx,
        "num_voxels": torch.tensor(voxel_count, dtype=torch.long, device=device),
    }
