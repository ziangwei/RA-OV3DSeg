from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ra_ov3dseg.models.voxelization import (
    VoxelizationConfig,
    hash_spconv_indices,
    make_point_input_features,
    voxelize_point_features,
)


def _import_spconv():
    try:
        import spconv.pytorch as spconv
    except ImportError as exc:
        raise ImportError(
            "sparse_unet_spconv requires spconv. Install a CUDA-matched wheel, "
            "for example `pip install spconv-cu118` or `pip install spconv-cu120`."
        ) from exc
    return spconv


class SparseUNetSpConv(nn.Module):
    """Minimal spconv SparseUNet-style student for MVP-v5.

    This is intentionally compact: one sparse downsample / inverse upsample stage,
    point-wise gather, then CLIP-space feature and base-class classification heads.
    It is the first non-debug 3D student, not the final high-capacity backbone.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        voxel_size: tuple[float, float, float] = (0.2, 0.2, 0.2),
        point_cloud_range: tuple[float, float, float, float, float, float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        input_channels: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        spconv = _import_spconv()
        self.spconv = spconv
        self.voxel_config = VoxelizationConfig(voxel_size=voxel_size, point_cloud_range=point_cloud_range)
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, base_channels, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.down1 = spconv.SparseSequential(
            spconv.SparseConv3d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                indice_key="down1",
            ),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(
                base_channels * 2,
                base_channels * 2,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            ),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.up1 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                base_channels * 2,
                base_channels,
                kernel_size=3,
                bias=False,
                indice_key="down1",
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = spconv.SparseSequential(
            spconv.SubMConv3d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False, indice_key="fuse"),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False, indice_key="fuse"),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.fallback_fuse = spconv.SparseSequential(
            spconv.SubMConv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False, indice_key="fallback"),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.feature_head = nn.Linear(base_channels, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _batch_size(self, point_batch_indices: torch.Tensor) -> int:
        if point_batch_indices.numel() == 0:
            return 1
        return int(point_batch_indices.max().detach().cpu().item()) + 1

    def _gather_to_points(
        self,
        sparse_tensor: Any,
        input_voxel_coords: torch.Tensor,
        point_to_voxel: torch.Tensor,
        valid_point_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_voxel_features = sparse_tensor.features.new_zeros((point_to_voxel.shape[0], sparse_tensor.features.shape[1]))
        model_valid_mask = torch.zeros_like(valid_point_mask)

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
        point_input_features = make_point_input_features(point_xyz, self.voxel_config.point_cloud_range)
        voxelized = voxelize_point_features(
            point_xyz=point_xyz,
            point_features=point_input_features,
            point_batch_indices=point_batch_indices,
            config=self.voxel_config,
        )

        sparse_tensor = self.spconv.SparseConvTensor(
            features=voxelized["voxel_features"],
            indices=voxelized["voxel_coords"],
            spatial_shape=voxelized["spatial_shape"],
            batch_size=self._batch_size(point_batch_indices),
        )
        x0 = self.stem(sparse_tensor)
        x1 = self.down1(x0)
        x_up = self.up1(x1)

        if x_up.features.shape[0] == x0.features.shape[0] and torch.equal(x_up.indices, x0.indices):
            x = x_up.replace_feature(torch.cat([x_up.features, x0.features], dim=1))
            x = self.fuse(x)
        else:
            x = self.fallback_fuse(x_up)

        point_voxel_features, model_valid_mask = self._gather_to_points(
            sparse_tensor=x,
            input_voxel_coords=voxelized["voxel_coords"],
            point_to_voxel=voxelized["point_to_voxel"],
            valid_point_mask=voxelized["valid_point_mask"],
        )
        point_features = F.normalize(self.feature_head(point_voxel_features), dim=-1)
        logits = self.classifier(point_features)
        return {
            "point_features": point_features,
            "logits": logits,
            "model_valid_mask": model_valid_mask,
            "num_voxels": voxelized["num_voxels"],
        }
