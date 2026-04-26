from __future__ import annotations

from ra_ov3dseg.datasets.nuscenes_dataset import CAMERA_CHANNELS, NuScenesDataset


class NuScenesMiniDataset(NuScenesDataset):
    """兼容旧命名。

    当前类仅作为向后兼容别名保留，后续统一使用 `NuScenesDataset`。
    """
