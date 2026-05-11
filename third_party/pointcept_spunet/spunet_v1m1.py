"""Minimal vendored Pointcept SpUNet v1m1.

Vendored from:
    https://github.com/Pointcept/Pointcept
    pointcept/models/sparse_unet/spconv_unet_v1m1_base.py
Commit:
    d74c646db6abec569d0f23e0c34e7ddfce142789
License:
    MIT, see ./LICENSE
Modifications:
    - removed Pointcept registry imports/decorators;
    - removed torch_geometric/timm dependencies;
    - return SparseConvTensor when requested for RA-OV3DSeg point gather.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial

import torch
from torch import nn

try:
    import spconv.pytorch as spconv
except ImportError as exc:  # pragma: no cover - exercised on server env.
    raise ImportError(
        "Pointcept SpUNet requires spconv. Install a CUDA-matched spconv wheel "
        "such as spconv-cu120 for CUDA 12.x."
    ) from exc


def offset2batch(offset: torch.Tensor) -> torch.Tensor:
    bincount = torch.diff(offset.long(), prepend=torch.zeros(1, device=offset.device, dtype=torch.long))
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        embed_channels: int,
        stride: int = 1,
        norm_fn=None,
        indice_key: str | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if norm_fn is None:
            raise ValueError("norm_fn must be provided.")

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size=1, bias=False),
                norm_fn(embed_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))
        return out


class SpUNetBase(nn.Module):
    """Pointcept SpConv SparseUNet v1m1 backend.

    The module operates on voxel-level input dictionaries:
        grid_coord: [M, 3] int tensor in xyz order
        feat: [M, C] float tensor
        offset: cumulative voxel counts per batch
        sparse_shape: optional [x, y, z] shape
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        channels: tuple[int, ...] = (32, 64, 128, 256, 256, 128, 96, 96),
        layers: tuple[int, ...] = (2, 3, 4, 6, 2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(layers) % 2 != 0 or len(layers) != len(channels):
            raise ValueError("layers must have even length and align with channels.")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.channels = tuple(int(channel) for channel in channels)
        self.layers = tuple(int(layer) for layer in layers)
        self.num_stages = len(self.layers) // 2

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for stage in range(self.num_stages):
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[stage],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{stage + 1}",
                    ),
                    norm_fn(channels[stage]),
                    nn.ReLU(),
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                f"block{i}",
                                BasicBlock(
                                    channels[stage],
                                    channels[stage],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{stage + 1}",
                                ),
                            )
                            for i in range(layers[stage])
                        ]
                    )
                )
            )
            self.up.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        channels[len(channels) - stage - 2],
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{stage + 1}",
                    ),
                    norm_fn(dec_channels),
                    nn.ReLU(),
                )
            )
            self.dec.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                f"block{i}",
                                BasicBlock(
                                    dec_channels + enc_channels if i == 0 else dec_channels,
                                    dec_channels,
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{stage}",
                                ),
                            )
                            for i in range(layers[len(channels) - stage - 1])
                        ]
                    )
                )
            )

            enc_channels = channels[stage]
            dec_channels = channels[len(channels) - stage - 2]

        self.final = (
            spconv.SubMConv3d(channels[-1], num_classes, kernel_size=1, padding=1, bias=True)
            if num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, spconv.SubMConv3d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, input_dict: dict[str, torch.Tensor], return_sparse_tensor: bool = False):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        batch = offset2batch(offset)

        if "sparse_shape" in input_dict:
            sparse_shape_xyz = [int(value) for value in input_dict["sparse_shape"]]
        else:
            sparse_shape_xyz = torch.add(torch.max(grid_coord, dim=0).values, 96).detach().cpu().tolist()

        sparse_indices = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()],
            dim=1,
        ).contiguous()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=sparse_indices,
            spatial_shape=sparse_shape_xyz,
            batch_size=int(batch[-1].detach().cpu().item()) + 1,
        )
        x = self.conv_input(x)
        skips = [x]
        for stage in range(self.num_stages):
            x = self.down[stage](x)
            x = self.enc[stage](x)
            skips.append(x)

        x = skips.pop(-1)
        for stage in reversed(range(self.num_stages)):
            x = self.up[stage](x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[stage](x)

        x = self.final(x)
        if return_sparse_tensor:
            return x
        return x.features
