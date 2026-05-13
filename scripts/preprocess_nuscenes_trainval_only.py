"""Pointcept nuScenes preprocessing, trainval-only.

This wrapper bypasses Pointcept's hard-coded v1.0-test loading path. It imports
Pointcept's preprocessing helpers as a library and only writes:

  <output_root>/info/nuscenes_infos_<max_sweeps>sweeps_train.pkl
  <output_root>/info/nuscenes_infos_<max_sweeps>sweeps_val.pkl

It does not modify files under third_party/Pointcept.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import types
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

sys.modules.setdefault("pointops", types.ModuleType("pointops"))

from pointcept.datasets.preprocessing.nuscenes.preprocess_nuscenes_info import (
    fill_trainval_infos,
    get_available_scenes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Pointcept nuScenes preprocessing for train/val only.")
    parser.add_argument("--dataset_root", required=True, type=str, help="Raw nuScenes root with v1.0-trainval.")
    parser.add_argument("--output_root", required=True, type=str, help="Pointcept processed nuScenes output root.")
    parser.add_argument("--max_sweeps", default=10, type=int, help="Max number of sweeps.")
    parser.add_argument("--with_camera", action="store_true", help="Include camera metadata.")
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Allow missing scenes instead of requiring all 850 trainval scenes.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    print("[trainval-only] loading v1.0-trainval...")
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(dataset_root), verbose=False)
    available = get_available_scenes(nusc)
    available_names = [scene["name"] for scene in available]
    print(f"[trainval-only] total scenes: {len(nusc.scene)}, available: {len(available)}")
    if not args.allow_partial and len(available) != len(nusc.scene):
        raise RuntimeError(
            "Not all v1.0-trainval scenes are available. If this is intentional for a smoke run, "
            "rerun with --allow_partial."
        )

    train_scenes = {
        available[available_names.index(scene_name)]["token"]
        for scene_name in splits.train
        if scene_name in available_names
    }
    print(f"[trainval-only] train scene tokens: {len(train_scenes)}")
    print(
        "[trainval-only] filling trainval information "
        f"(max_sweeps={args.max_sweeps}, with_camera={args.with_camera})..."
    )
    train_infos, val_infos = fill_trainval_infos(
        str(dataset_root),
        nusc,
        train_scenes,
        test=False,
        max_sweeps=args.max_sweeps,
        with_camera=args.with_camera,
    )
    print(f"[trainval-only] train samples: {len(train_infos)}, val samples: {len(val_infos)}")

    info_dir = output_root / "info"
    os.makedirs(info_dir, exist_ok=True)
    train_pkl = info_dir / f"nuscenes_infos_{args.max_sweeps}sweeps_train.pkl"
    val_pkl = info_dir / f"nuscenes_infos_{args.max_sweeps}sweeps_val.pkl"
    with train_pkl.open("wb") as f:
        pickle.dump(train_infos, f)
    with val_pkl.open("wb") as f:
        pickle.dump(val_infos, f)
    print(f"[trainval-only] wrote:\n  {train_pkl}\n  {val_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
