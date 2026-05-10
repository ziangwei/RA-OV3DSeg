from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CylindricalVoxelizationConfig:
    voxel_size: tuple[float, float, float] = (0.125, 0.017453292519943295, 0.25)
    point_cloud_range: tuple[float, float, float, float, float, float] = (
        0.0,
        -3.141592653589793,
        -5.0,
        60.0,
        3.141592653589793,
        3.0,
    )

    @property
    def grid_size_rpz(self) -> tuple[int, int, int]:
        r_min, phi_min, z_min, r_max, phi_max, z_max = self.point_cloud_range
        vr, vphi, vz = self.voxel_size
        return (
            int((r_max - r_min) / vr),
            int((phi_max - phi_min) / vphi),
            int((z_max - z_min) / vz),
        )

    @property
    def spatial_shape_zpr(self) -> list[int]:
        grid_r, grid_phi, grid_z = self.grid_size_rpz
        return [grid_z, grid_phi, grid_r]


def cartesian_to_cylinder(point_xyz: torch.Tensor) -> torch.Tensor:
    rho = torch.linalg.norm(point_xyz[:, :2], dim=1)
    phi = torch.atan2(point_xyz[:, 1], point_xyz[:, 0])
    return torch.stack([rho, phi, point_xyz[:, 2]], dim=1)


def make_cylinder_point_features(
    point_xyz: torch.Tensor,
    point_input_features: torch.Tensor | None,
    config: CylindricalVoxelizationConfig,
    eps: float = 1e-6,
) -> torch.Tensor:
    cyl = cartesian_to_cylinder(point_xyz)
    cyl_range = torch.as_tensor(config.point_cloud_range, dtype=point_xyz.dtype, device=point_xyz.device)
    cyl_min = cyl_range[:3]
    cyl_max = cyl_range[3:]
    cyl_scale = torch.clamp(cyl_max - cyl_min, min=eps)
    cyl_normalized = (cyl - cyl_min) / cyl_scale

    xy_range = torch.as_tensor((-54.0, -54.0, -5.0, 54.0, 54.0, 3.0), dtype=point_xyz.dtype, device=point_xyz.device)
    xyz_center = (xy_range[:3] + xy_range[3:]) * 0.5
    xyz_scale = torch.clamp(xy_range[3:] - xy_range[:3], min=eps)
    xyz_normalized = (point_xyz - xyz_center) / xyz_scale

    sin_phi = torch.sin(cyl[:, 1:2])
    cos_phi = torch.cos(cyl[:, 1:2])
    distance = cyl[:, 0:1] / max(float(config.point_cloud_range[3]), eps)
    if point_input_features is None or point_input_features.numel() == 0:
        intensity = torch.zeros((point_xyz.shape[0], 1), dtype=point_xyz.dtype, device=point_xyz.device)
    else:
        intensity = point_input_features[:, :1].to(dtype=point_xyz.dtype, device=point_xyz.device)
    return torch.cat([xyz_normalized, cyl_normalized, sin_phi, cos_phi, distance, intensity], dim=1)


def voxelize_cylinder_point_features(
    point_xyz: torch.Tensor,
    point_features: torch.Tensor,
    point_batch_indices: torch.Tensor,
    config: CylindricalVoxelizationConfig,
) -> dict[str, torch.Tensor | list[int]]:
    if point_xyz.ndim != 2 or point_xyz.shape[1] != 3:
        raise ValueError(f"point_xyz must have shape [N, 3], got {tuple(point_xyz.shape)}")
    if point_features.ndim != 2 or point_features.shape[0] != point_xyz.shape[0]:
        raise ValueError("point_features must have shape [N, C] and align with point_xyz.")
    if point_batch_indices.ndim != 1 or point_batch_indices.shape[0] != point_xyz.shape[0]:
        raise ValueError("point_batch_indices must have shape [N].")

    device = point_xyz.device
    cyl = cartesian_to_cylinder(point_xyz)
    voxel_size = torch.as_tensor(config.voxel_size, dtype=point_xyz.dtype, device=device)
    cyl_range = torch.as_tensor(config.point_cloud_range, dtype=point_xyz.dtype, device=device)
    cyl_min = cyl_range[:3]
    cyl_max = cyl_range[3:]
    grid_size_rpz = torch.as_tensor(config.grid_size_rpz, dtype=torch.long, device=device)

    coords_rpz = torch.floor((cyl - cyl_min) / voxel_size).long()
    valid_point_mask = torch.all((cyl >= cyl_min) & (cyl < cyl_max), dim=1)
    valid_point_mask &= torch.all((coords_rpz >= 0) & (coords_rpz < grid_size_rpz), dim=1)

    point_to_voxel = torch.full((point_xyz.shape[0],), -1, dtype=torch.long, device=device)
    if not torch.any(valid_point_mask):
        raise ValueError("No points fall inside the configured cylindrical point_cloud_range.")

    valid_indices = torch.nonzero(valid_point_mask, as_tuple=False).squeeze(1)
    valid_coords_rpz = coords_rpz[valid_indices]
    valid_batch = point_batch_indices[valid_indices].long()
    valid_coords_zpr = torch.stack(
        [
            valid_batch,
            valid_coords_rpz[:, 2],
            valid_coords_rpz[:, 1],
            valid_coords_rpz[:, 0],
        ],
        dim=1,
    )

    voxel_coords, inverse = torch.unique(valid_coords_zpr, sorted=True, return_inverse=True, dim=0)
    voxel_count = int(voxel_coords.shape[0])
    voxel_features = point_features.new_zeros((voxel_count, int(point_features.shape[1])))
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
        "spatial_shape": config.spatial_shape_zpr,
        "num_voxels": torch.tensor(voxel_count, dtype=torch.long, device=device),
    }
