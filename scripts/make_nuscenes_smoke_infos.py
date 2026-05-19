"""Create tiny Pointcept nuScenes info files for launcher smoke tests."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create tiny train/val Pointcept nuScenes info files.")
    parser.add_argument("--source_root", required=True, type=str, help="Pointcept processed nuScenes root.")
    parser.add_argument("--output_root", required=True, type=str, help="Smoke processed nuScenes root.")
    parser.add_argument("--max_sweeps", default=1, type=int)
    parser.add_argument("--train_samples", default=8, type=int)
    parser.add_argument("--val_samples", default=4, type=int)
    return parser


def load_infos(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def dump_infos(path: Path, infos) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(infos, f)


def main() -> int:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    info_name = f"nuscenes_infos_{args.max_sweeps}sweeps"

    source_train = source_root / "info" / f"{info_name}_train.pkl"
    source_val = source_root / "info" / f"{info_name}_val.pkl"
    if not source_train.exists():
        raise FileNotFoundError(f"train info not found: {source_train}")
    if not source_val.exists():
        raise FileNotFoundError(f"val info not found: {source_val}")

    train_infos = load_infos(source_train)[: args.train_samples]
    val_infos = load_infos(source_val)[: args.val_samples]
    if not train_infos:
        raise RuntimeError(f"No train infos loaded from {source_train}")
    if not val_infos:
        raise RuntimeError(f"No val infos loaded from {source_val}")

    dump_infos(output_root / "info" / f"{info_name}_train.pkl", train_infos)
    dump_infos(output_root / "info" / f"{info_name}_val.pkl", val_infos)

    raw_target = source_root / "raw"
    raw_link = output_root / "raw"
    if raw_link.exists() or raw_link.is_symlink():
        raw_link.unlink()
    os.symlink(raw_target, raw_link, target_is_directory=True)

    print(f"[smoke-infos] source_root={source_root}")
    print(f"[smoke-infos] output_root={output_root}")
    print(f"[smoke-infos] train={len(train_infos)} val={len(val_infos)}")
    print(f"[smoke-infos] raw -> {raw_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
