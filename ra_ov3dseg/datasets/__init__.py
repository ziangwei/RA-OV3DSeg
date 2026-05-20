"""Dataset helpers for RA-OV3DSeg.

The package entrypoint stays lazy so metadata-only tools do not require the
nuScenes devkit.
"""

__all__ = ["CAMERA_CHANNELS", "NuScenesDataset"]


def __getattr__(name: str):
    if name in __all__:
        from . import nuscenes_dataset

        return getattr(nuscenes_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
