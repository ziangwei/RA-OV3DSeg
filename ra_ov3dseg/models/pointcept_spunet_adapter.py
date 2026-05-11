from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ra_ov3dseg.models.voxelization import VoxelizationConfig
from third_party.pointcept_spunet import SpUNetBase


class PointceptSpUNetAdapter(nn.Module):
    """Adapter from RA-OV3DSeg point batches to vendored Pointcept SpUNet.

    Pointcept's nuScenes SpUNet recipe uses a local Cartesian voxel grid,
    `coord + strength` input features, and official lidarseg output classes.
    This adapter keeps all data/loss/eval code in RA-OV3DSeg while using the
    mature SpUNet backend.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        voxel_size: tuple[float, float, float] = (0.05, 0.05, 0.05),
        point_cloud_range: tuple[float, float, float, float, float, float] = (
            -120.0,
            -120.0,
            -10.0,
            120.0,
            120.0,
            10.0,
        ),
        input_channels: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.voxel_config = VoxelizationConfig(voxel_size=voxel_size, point_cloud_range=point_cloud_range)
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.num_classes = int(num_classes)

        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8,
            base_channels * 4,
            base_channels * 3,
            base_channels * 3,
        )
        decoder_dim = int(channels[-1])
        self.feature_dim = decoder_dim
        self.requested_feature_dim = int(feature_dim)
        self.backbone = SpUNetBase(
            in_channels=input_channels,
            num_classes=0,
            base_channels=base_channels,
            channels=channels,
            layers=(2, 3, 4, 6, 2, 2, 2, 2),
        )
        # Per-voxel linear classification matches Pointcept's 1x1 supervised head
        # while preserving RA-OV3DSeg's point_features interface.
        self.classifier = nn.Linear(decoder_dim, num_classes)

    def _make_point_features(self, point_xyz: torch.Tensor, point_input_features: torch.Tensor | None) -> torch.Tensor:
        if point_input_features is None or point_input_features.numel() == 0:
            intensity = torch.zeros((point_xyz.shape[0], 1), dtype=point_xyz.dtype, device=point_xyz.device)
        else:
            intensity = point_input_features[:, :1].to(dtype=point_xyz.dtype, device=point_xyz.device)
        return torch.cat([point_xyz, intensity], dim=1)

    def _voxel_offsets(self, voxel_coords_bxyz: torch.Tensor) -> torch.Tensor:
        batch_indices = voxel_coords_bxyz[:, 0].long()
        batch_size = int(batch_indices.max().detach().cpu().item()) + 1 if batch_indices.numel() else 1
        counts = torch.bincount(batch_indices, minlength=batch_size)
        return torch.cumsum(counts, dim=0).long()

    @staticmethod
    def _hash_indices(indices: torch.Tensor, spatial_shape: list[int]) -> torch.Tensor:
        """Hash [batch, x, y, z] sparse indices for coordinate matching."""

        x_size, y_size, z_size = [int(value) for value in spatial_shape]
        indices_long = indices.long()
        return (
            ((indices_long[:, 0] * (x_size + 1) + indices_long[:, 1]) * (y_size + 1) + indices_long[:, 2])
            * (z_size + 1)
            + indices_long[:, 3]
        )

    def _voxelize_pointcept_grid(
        self,
        point_xyz: torch.Tensor,
        point_features: torch.Tensor,
        point_batch_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[int]]:
        """Voxelize with Pointcept-style per-sample local grid coordinates."""

        device = point_xyz.device
        voxel_size = torch.as_tensor(self.voxel_config.voxel_size, dtype=point_xyz.dtype, device=device)
        coords_xyz_abs = torch.floor(point_xyz / voxel_size).long()
        batch_indices = point_batch_indices.long()
        coords_xyz = torch.empty_like(coords_xyz_abs)
        valid_point_mask = torch.zeros((point_xyz.shape[0],), dtype=torch.bool, device=device)

        for batch_id in torch.unique(batch_indices, sorted=True):
            batch_mask = batch_indices == batch_id
            if not torch.any(batch_mask):
                continue
            batch_coords = coords_xyz_abs[batch_mask]
            coords_xyz[batch_mask] = batch_coords - torch.min(batch_coords, dim=0).values
            valid_point_mask[batch_mask] = True

        if not torch.any(valid_point_mask):
            raise ValueError("No points available for Pointcept local-grid voxelization.")

        valid_indices = torch.nonzero(valid_point_mask, as_tuple=False).squeeze(1)
        valid_coords_bxyz = torch.cat(
            [batch_indices[valid_indices].unsqueeze(1), coords_xyz[valid_indices]],
            dim=1,
        )
        voxel_coords, inverse = torch.unique(valid_coords_bxyz, sorted=True, return_inverse=True, dim=0)

        voxel_features = point_features.new_zeros((voxel_coords.shape[0], point_features.shape[1]))
        voxel_features.index_add_(0, inverse, point_features[valid_indices])
        counts = point_features.new_zeros((voxel_coords.shape[0], 1))
        counts.index_add_(0, inverse, torch.ones((valid_indices.shape[0], 1), dtype=point_features.dtype, device=device))
        voxel_features = voxel_features / torch.clamp(counts, min=1.0)

        point_to_voxel = torch.full((point_xyz.shape[0],), -1, dtype=torch.long, device=device)
        point_to_voxel[valid_indices] = inverse.long()
        voxel_point_indices = torch.full(
            (int(voxel_coords.shape[0]),),
            point_xyz.shape[0],
            dtype=torch.long,
            device=device,
        )
        voxel_point_indices.scatter_reduce_(0, inverse.long(), valid_indices.long(), reduce="amin", include_self=True)
        spatial_shape_xyz = (torch.max(voxel_coords[:, 1:], dim=0).values + 96).detach().cpu().tolist()

        return {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords.int(),
            "voxel_point_indices": voxel_point_indices,
            "point_to_voxel": point_to_voxel,
            "valid_point_mask": valid_point_mask,
            "spatial_shape": [int(value) for value in spatial_shape_xyz],
            "num_voxels": torch.tensor(int(voxel_coords.shape[0]), dtype=torch.long, device=device),
        }

    def _gather_to_points(
        self,
        sparse_tensor: Any,
        input_voxel_coords: torch.Tensor,
        spatial_shape: list[int],
        point_to_voxel: torch.Tensor,
        valid_point_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        point_voxel_features = sparse_tensor.features.new_zeros((point_to_voxel.shape[0], sparse_tensor.features.shape[1]))
        final_hash = self._hash_indices(sparse_tensor.indices, spatial_shape)
        input_hash = self._hash_indices(input_voxel_coords, spatial_shape)
        sorted_hash, sorted_order = torch.sort(final_hash)
        lookup_pos = torch.searchsorted(sorted_hash, input_hash)
        in_bounds = lookup_pos < sorted_hash.shape[0]
        matched = torch.zeros_like(in_bounds)
        matched[in_bounds] = sorted_hash[lookup_pos[in_bounds]] == input_hash[in_bounds]

        voxel_to_final = torch.full((input_voxel_coords.shape[0],), -1, dtype=torch.long, device=input_voxel_coords.device)
        voxel_to_final[matched] = sorted_order[lookup_pos[matched]]

        point_voxel_index = voxel_to_final[torch.clamp(point_to_voxel, min=0)]
        model_valid_mask = valid_point_mask & (point_to_voxel >= 0) & (point_voxel_index >= 0)
        point_voxel_features[model_valid_mask] = sparse_tensor.features[point_voxel_index[model_valid_mask]]
        return point_voxel_features, model_valid_mask, point_voxel_index, voxel_to_final

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        point_xyz = batch["point_xyz"]
        point_batch_indices = batch["point_batch_indices"].long()
        point_input_features = batch.get("point_input_features")
        point_features_in = self._make_point_features(point_xyz, point_input_features)

        voxelized = self._voxelize_pointcept_grid(
            point_xyz=point_xyz,
            point_features=point_features_in,
            point_batch_indices=point_batch_indices,
        )
        voxel_coords_bxyz = voxelized["voxel_coords"].long()
        input_dict = {
            "grid_coord": voxel_coords_bxyz[:, 1:].contiguous(),
            "feat": voxelized["voxel_features"],
            "offset": self._voxel_offsets(voxel_coords_bxyz),
            "sparse_shape": torch.as_tensor(voxelized["spatial_shape"], dtype=torch.long, device=point_xyz.device),
        }
        sparse_out = self.backbone(input_dict, return_sparse_tensor=True)
        point_embeddings_raw, model_valid_mask, point_voxel_index, voxel_to_final = self._gather_to_points(
            sparse_tensor=sparse_out,
            input_voxel_coords=voxelized["voxel_coords"],
            spatial_shape=voxelized["spatial_shape"],
            point_to_voxel=voxelized["point_to_voxel"],
            valid_point_mask=voxelized["valid_point_mask"],
        )
        voxel_logits = self.classifier(sparse_out.features)
        logits = voxel_logits.new_zeros((point_xyz.shape[0], self.num_classes))
        logits[model_valid_mask] = voxel_logits[point_voxel_index[model_valid_mask]]

        input_voxel_valid = voxel_to_final >= 0
        supervised_logits = voxel_logits[voxel_to_final[input_voxel_valid]]
        supervised_label_indices = voxelized["voxel_point_indices"][input_voxel_valid]
        point_features = F.normalize(point_embeddings_raw, dim=-1)
        return {
            "point_features": point_features,
            "logits": logits,
            "supervised_logits": supervised_logits,
            "supervised_label_indices": supervised_label_indices,
            "model_valid_mask": model_valid_mask,
            "num_voxels": voxelized["num_voxels"],
        }
