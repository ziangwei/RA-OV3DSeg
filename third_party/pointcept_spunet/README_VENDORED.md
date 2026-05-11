# Vendored Pointcept SpUNet

Source: https://github.com/Pointcept/Pointcept

Commit: `d74c646db6abec569d0f23e0c34e7ddfce142789`

License: MIT, copied in `LICENSE`.

Vendored files:

- `spunet_v1m1.py`: adapted from `pointcept/models/sparse_unet/spconv_unet_v1m1_base.py`.

Modifications:

- Removed Pointcept registry imports and decorators.
- Removed `torch_geometric` dependency and encoder classification mode.
- Replaced `timm.layers.trunc_normal_` with `torch.nn.init.trunc_normal_`.
- Added `return_sparse_tensor=True` path so RA-OV3DSeg can gather voxel features back to original points.
- V17 adapter instantiates this module with `num_classes=0` and applies RA-OV3DSeg's own per-voxel linear classifier on decoder features. This avoids an extra embedding bottleneck while keeping checkpoint/prediction code centralized in RA-OV3DSeg.
- Kept Pointcept's sparse coordinate convention (`[batch, x, y, z]`). The adapter performs Pointcept-style per-sample local grid shifting before calling this module.
- Training uses voxel representative labels to match Pointcept `GridSample(mode="train")`; point-level logits are still gathered for prediction/evaluation.
- Kept the SpConv SparseUNet encoder-decoder blocks and default channel/layer recipe from Pointcept's nuScenes config.

Import rule:

- `third_party/pointcept_spunet/` must not import `ra_ov3dseg`.
- Integration logic belongs in `ra_ov3dseg/models/pointcept_spunet_adapter.py`.
