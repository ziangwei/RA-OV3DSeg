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
            "spconv_resunet requires spconv. Install a CUDA-matched wheel, "
            "for example `pip install spconv-cu120` for a CUDA 12.x environment."
        ) from exc
    return spconv


class SparseResidualBlock(nn.Module):
    """Submanifold sparse residual block operating on a SparseConvTensor."""

    def __init__(self, spconv, channels: int, indice_key: str) -> None:
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, sparse_tensor):
        identity = sparse_tensor.features
        out = self.conv1(sparse_tensor)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + identity))
        return out


def sparse_conv_norm_relu(spconv, in_channels: int, out_channels: int, indice_key: str):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


def sparse_downsample(spconv, in_channels: int, out_channels: int, indice_key: str):
    return spconv.SparseSequential(
        spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            indice_key=indice_key,
        ),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


def sparse_inverse_upsample(spconv, in_channels: int, out_channels: int, indice_key: str):
    return spconv.SparseSequential(
        spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size=3, bias=False, indice_key=indice_key),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


class SpConvResUNet(nn.Module):
    """Higher-capacity sparse ResUNet student for outdoor LiDAR upper-bound checks.

    This is still an in-repository implementation, not a vendor repo transplant.
    It uses a deeper encoder-decoder than `sparse_unet_spconv` so we can test
    whether the current project is limited by the 3D student capacity.
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

        c0 = base_channels
        c1 = base_channels * 2
        c2 = base_channels * 4
        c3 = base_channels * 8

        self.stem = sparse_conv_norm_relu(spconv, input_channels, c0, indice_key="res_stem")
        self.enc0 = nn.ModuleList(
            [SparseResidualBlock(spconv, c0, "res_enc0_a"), SparseResidualBlock(spconv, c0, "res_enc0_b")]
        )

        self.down1 = sparse_downsample(spconv, c0, c1, indice_key="res_down1")
        self.enc1 = nn.ModuleList(
            [SparseResidualBlock(spconv, c1, "res_enc1_a"), SparseResidualBlock(spconv, c1, "res_enc1_b")]
        )

        self.down2 = sparse_downsample(spconv, c1, c2, indice_key="res_down2")
        self.enc2 = nn.ModuleList(
            [SparseResidualBlock(spconv, c2, "res_enc2_a"), SparseResidualBlock(spconv, c2, "res_enc2_b")]
        )

        self.down3 = sparse_downsample(spconv, c2, c3, indice_key="res_down3")
        self.bottleneck = nn.ModuleList(
            [SparseResidualBlock(spconv, c3, "res_bottleneck_a"), SparseResidualBlock(spconv, c3, "res_bottleneck_b")]
        )

        self.up3 = sparse_inverse_upsample(spconv, c3, c2, indice_key="res_down3")
        self.fuse2 = sparse_conv_norm_relu(spconv, c2 * 2, c2, indice_key="res_fuse2")
        self.fallback2 = sparse_conv_norm_relu(spconv, c2, c2, indice_key="res_fallback2")
        self.dec2 = SparseResidualBlock(spconv, c2, "res_dec2")

        self.up2 = sparse_inverse_upsample(spconv, c2, c1, indice_key="res_down2")
        self.fuse1 = sparse_conv_norm_relu(spconv, c1 * 2, c1, indice_key="res_fuse1")
        self.fallback1 = sparse_conv_norm_relu(spconv, c1, c1, indice_key="res_fallback1")
        self.dec1 = SparseResidualBlock(spconv, c1, "res_dec1")

        self.up1 = sparse_inverse_upsample(spconv, c1, c0, indice_key="res_down1")
        self.fuse0 = sparse_conv_norm_relu(spconv, c0 * 2, c0, indice_key="res_fuse0")
        self.fallback0 = sparse_conv_norm_relu(spconv, c0, c0, indice_key="res_fallback0")
        self.dec0 = SparseResidualBlock(spconv, c0, "res_dec0")

        self.feature_head = nn.Sequential(
            nn.Linear(c0, c0 * 2),
            nn.ReLU(inplace=True),
            nn.Linear(c0 * 2, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _batch_size(self, point_batch_indices: torch.Tensor) -> int:
        if point_batch_indices.numel() == 0:
            return 1
        return int(point_batch_indices.max().detach().cpu().item()) + 1

    def _apply_blocks(self, sparse_tensor, blocks: nn.ModuleList):
        out = sparse_tensor
        for block in blocks:
            out = block(out)
        return out

    def _fuse_skip(self, up_tensor, skip_tensor, fuse_module, fallback_module):
        if up_tensor.features.shape[0] == skip_tensor.features.shape[0] and torch.equal(
            up_tensor.indices, skip_tensor.indices
        ):
            return fuse_module(up_tensor.replace_feature(torch.cat([up_tensor.features, skip_tensor.features], dim=1)))
        return fallback_module(up_tensor)

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

        x0 = self._apply_blocks(self.stem(sparse_tensor), self.enc0)
        x1 = self._apply_blocks(self.down1(x0), self.enc1)
        x2 = self._apply_blocks(self.down2(x1), self.enc2)
        x3 = self._apply_blocks(self.down3(x2), self.bottleneck)

        y2 = self.dec2(self._fuse_skip(self.up3(x3), x2, self.fuse2, self.fallback2))
        y1 = self.dec1(self._fuse_skip(self.up2(y2), x1, self.fuse1, self.fallback1))
        y0 = self.dec0(self._fuse_skip(self.up1(y1), x0, self.fuse0, self.fallback0))

        point_voxel_features, model_valid_mask = self._gather_to_points(
            sparse_tensor=y0,
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
