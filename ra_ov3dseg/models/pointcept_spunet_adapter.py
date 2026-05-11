from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ra_ov3dseg.models.voxelization import VoxelizationConfig, hash_spconv_indices, voxelize_point_features
from third_party.pointcept_spunet import SpUNetBase


class PointceptSpUNetAdapter(nn.Module):
    """Adapter from RA-OV3DSeg point batches to vendored Pointcept SpUNet.

    Pointcept's nuScenes SpUNet recipe uses a Cartesian grid, `coord + strength`
    input features, and 16 official lidarseg output classes. This adapter keeps
    all data/loss/eval code in RA-OV3DSeg while using the mature SpUNet backend.
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
        self.feature_dim = int(feature_dim)
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
        self.backbone = SpUNetBase(
            in_channels=input_channels,
            num_classes=feature_dim,
            base_channels=base_channels,
            channels=channels,
            layers=(2, 3, 4, 6, 2, 2, 2, 2),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _make_point_features(self, point_xyz: torch.Tensor, point_input_features: torch.Tensor | None) -> torch.Tensor:
        if point_input_features is None or point_input_features.numel() == 0:
            intensity = torch.zeros((point_xyz.shape[0], 1), dtype=point_xyz.dtype, device=point_xyz.device)
        else:
            intensity = point_input_features[:, :1].to(dtype=point_xyz.dtype, device=point_xyz.device)
        return torch.cat([point_xyz, intensity], dim=1)

    def _voxel_offsets(self, voxel_coords_zyx: torch.Tensor) -> torch.Tensor:
        batch_indices = voxel_coords_zyx[:, 0].long()
        batch_size = int(batch_indices.max().detach().cpu().item()) + 1 if batch_indices.numel() else 1
        counts = torch.bincount(batch_indices, minlength=batch_size)
        return torch.cumsum(counts, dim=0).long()

    def _gather_to_points(
        self,
        sparse_tensor: Any,
        input_voxel_coords: torch.Tensor,
        point_to_voxel: torch.Tensor,
        valid_point_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_voxel_features = sparse_tensor.features.new_zeros((point_to_voxel.shape[0], sparse_tensor.features.shape[1]))
        final_hash = hash_spconv_indices(sparse_tensor.indices, self.voxel_config.spatial_shape_zyx)
        input_hash = hash_spconv_indices(input_voxel_coords, self.voxel_config.spatial_shape_zyx)
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
        return point_voxel_features, model_valid_mask

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        point_xyz = batch["point_xyz"]
        point_batch_indices = batch["point_batch_indices"].long()
        point_input_features = batch.get("point_input_features")
        point_features_in = self._make_point_features(point_xyz, point_input_features)

        voxelized = voxelize_point_features(
            point_xyz=point_xyz,
            point_features=point_features_in,
            point_batch_indices=point_batch_indices,
            config=self.voxel_config,
        )
        voxel_coords_zyx = voxelized["voxel_coords"].long()
        input_dict = {
            "grid_coord": voxel_coords_zyx[:, [3, 2, 1]].contiguous(),
            "feat": voxelized["voxel_features"],
            "offset": self._voxel_offsets(voxel_coords_zyx),
            "sparse_shape": torch.as_tensor(self.voxel_config.grid_size_xyz, dtype=torch.long, device=point_xyz.device),
        }
        sparse_out = self.backbone(input_dict, return_sparse_tensor=True)
        point_embeddings_raw, model_valid_mask = self._gather_to_points(
            sparse_tensor=sparse_out,
            input_voxel_coords=voxelized["voxel_coords"],
            point_to_voxel=voxelized["point_to_voxel"],
            valid_point_mask=voxelized["valid_point_mask"],
        )
        logits = self.classifier(point_embeddings_raw)
        point_features = F.normalize(point_embeddings_raw, dim=-1)
        return {
            "point_features": point_features,
            "logits": logits,
            "model_valid_mask": model_valid_mask,
            "num_voxels": voxelized["num_voxels"],
        }
