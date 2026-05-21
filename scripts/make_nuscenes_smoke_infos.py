"""Create tiny Pointcept nuScenes info files for launcher smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create tiny train/val Pointcept nuScenes info files.")
    parser.add_argument("--source_root", required=True, type=str, help="Pointcept processed nuScenes root.")
    parser.add_argument("--output_root", required=True, type=str, help="Smoke processed nuScenes root.")
    parser.add_argument("--max_sweeps", default=1, type=int)
    parser.add_argument("--train_samples", default=8, type=int)
    parser.add_argument("--val_samples", default=4, type=int)
    parser.add_argument(
        "--sample_indices_path",
        default=None,
        type=str,
        help="Optional Stage 3/4 JSON manifest with samples[].sample_token; filters smoke infos to cached samples.",
    )
    parser.add_argument(
        "--cache_reliability_dir",
        default=None,
        type=str,
        help=(
            "Optional Stage 4 reliability cache dir. When set, materializes "
            "subset raw LiDAR files from cache point_xyz so Pointcept coord "
            "order exactly matches the teacher cache."
        ),
    )
    return parser


def load_infos(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def dump_infos(path: Path, infos) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(infos, f)


def reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def scalar_to_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (str, int)):
        return str(value)
    return None


def info_token(info: dict[str, Any]) -> str | None:
    for key in ("sample_token", "token"):
        value = scalar_to_str(info.get(key))
        if value:
            return value
    return None


def timestamp_candidates(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return []

    # Pointcept preprocessing stores seconds, while nuScenes sample records store
    # microseconds. Do not keep lossy integer-second candidates: multiple samples
    # can share the same second and would be mapped to the wrong teacher cache.
    if abs(numeric) < 1_000_000_000_000:
        return [int(round(numeric * 1_000_000))]
    return [int(round(numeric))]


def info_timestamps(info: dict[str, Any]) -> list[int]:
    candidates: list[int] = []
    for key in ("timestamp", "lidar_timestamp"):
        candidates.extend(timestamp_candidates(info.get(key)))
    return list(dict.fromkeys(candidates))


def load_manifest_keys(path: Path) -> tuple[set[str], set[int], dict[str, int], dict[int, int], dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens = {
        str(item["sample_token"])
        for item in data.get("samples", [])
        if isinstance(item, dict) and item.get("sample_token") is not None
    }
    timestamp_index_sets: dict[int, set[int]] = {}
    for item in data.get("samples", []):
        if not isinstance(item, dict) or item.get("sample_idx") is None:
            continue
        sample_idx = int(item["sample_idx"])
        for timestamp in timestamp_candidates(item.get("timestamp")):
            timestamp_index_sets.setdefault(timestamp, set()).add(sample_idx)
    timestamp_to_index = {
        timestamp: next(iter(sample_indices))
        for timestamp, sample_indices in timestamp_index_sets.items()
        if len(sample_indices) == 1
    }
    timestamps = set(timestamp_to_index)
    token_to_index = {
        str(item["sample_token"]): int(item["sample_idx"])
        for item in data.get("samples", [])
        if isinstance(item, dict)
        and item.get("sample_token") is not None
        and item.get("sample_idx") is not None
    }
    index_to_token = {
        int(item["sample_idx"]): str(item["sample_token"])
        for item in data.get("samples", [])
        if isinstance(item, dict)
        and item.get("sample_idx") is not None
        and item.get("sample_token") is not None
    }
    if not tokens and not timestamps:
        raise ValueError(f"No samples[].sample_token or samples[].timestamp entries found in {path}")
    return tokens, timestamps, token_to_index, timestamp_to_index, index_to_token


def filter_by_manifest(
    infos: list[dict[str, Any]],
    token_to_index: dict[str, int],
    timestamp_to_index: dict[int, int],
    index_to_token: dict[int, str],
) -> list[dict[str, Any]]:
    filtered = []
    for info in infos:
        token = info_token(info)
        timestamp_matches = [timestamp for timestamp in info_timestamps(info) if timestamp in timestamp_to_index]
        sample_idx = None
        if token in token_to_index:
            sample_idx = token_to_index[token]
        elif timestamp_matches:
            sample_idx = timestamp_to_index[timestamp_matches[0]]
        if sample_idx is None:
            continue

        copied = dict(info)
        # Pointcept NuScenesDataset exposes only lidar_token as data_dict["name"].
        # Preserve original Pointcept identifiers separately and inject manifest
        # sample identity so reliability cache lookup uses one namespace.
        copied["original_lidar_token"] = copied.get("lidar_token")
        copied["original_token"] = copied.get("token")
        copied["original_sample_token"] = copied.get("sample_token")
        copied["sample_idx"] = sample_idx
        if sample_idx in index_to_token:
            copied["sample_token"] = index_to_token[sample_idx]
        elif token in token_to_index:
            copied["sample_token"] = token
        else:
            copied.pop("sample_token", None)
        copied["lidar_token"] = f"sample_{sample_idx:04d}"
        filtered.append(copied)
    return filtered


def load_lidar_bin(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 5 != 0:
        return None
    return raw.reshape(-1, 5)


def materialize_cache_backed_raw(
    infos: list[dict[str, Any]],
    source_root: Path,
    output_root: Path,
    reliability_dir: Path,
) -> None:
    raw_root = output_root / "raw"
    reset_path(raw_root)
    lidar_dir = raw_root / "ra_cache_lidar"
    label_dir = raw_root / "ra_cache_lidarseg"
    lidar_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for info in infos:
        if info.get("sample_idx") is None:
            raise ValueError("cache-backed raw requires every info row to have sample_idx")
        sample_idx = int(info["sample_idx"])
        prefix = f"sample_{sample_idx:04d}"
        reliability_path = reliability_dir / f"{prefix}_reliability.npz"
        if not reliability_path.exists():
            raise FileNotFoundError(f"reliability cache not found for cache-backed raw: {reliability_path}")

        with np.load(reliability_path, allow_pickle=False) as reliability_npz:
            point_xyz = reliability_npz["point_xyz"][:, :3].astype(np.float32)
        num_points = int(point_xyz.shape[0])

        strength_ring = np.zeros((num_points, 2), dtype=np.float32)
        source_lidar_rel = scalar_to_str(info.get("lidar_path"))
        if source_lidar_rel:
            source_lidar = load_lidar_bin(source_root / "raw" / source_lidar_rel)
            if source_lidar is not None and source_lidar.shape[0] == num_points:
                strength_ring = source_lidar[:, 3:5].astype(np.float32)

        cache_lidar_rel = f"ra_cache_lidar/{prefix}.bin"
        cache_lidar_path = raw_root / cache_lidar_rel
        np.concatenate([point_xyz, strength_ring], axis=1).astype(np.float32).tofile(cache_lidar_path)

        info["original_lidar_path"] = info.get("lidar_path")
        info["lidar_path"] = cache_lidar_rel

        source_label_rel = scalar_to_str(info.get("gt_segment_path"))
        if source_label_rel:
            source_label_path = source_root / "raw" / source_label_rel
            if not source_label_path.exists():
                raise FileNotFoundError(f"gt segment file not found for cache-backed raw: {source_label_path}")
            labels = np.fromfile(source_label_path, dtype=np.uint8)
            if labels.shape[0] != num_points:
                raise ValueError(
                    f"gt segment count mismatch for {prefix}: labels={labels.shape[0]} cache={num_points}"
                )
            cache_label_rel = f"ra_cache_lidarseg/{prefix}.bin"
            cache_label_path = raw_root / cache_label_rel
            labels.tofile(cache_label_path)
            info["original_gt_segment_path"] = info.get("gt_segment_path")
            info["gt_segment_path"] = cache_label_rel


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

    source_train_infos = load_infos(source_train)
    source_val_infos = load_infos(source_val)
    if args.sample_indices_path:
        manifest_path = Path(args.sample_indices_path).expanduser().resolve()
        tokens, timestamps, token_to_index, timestamp_to_index, index_to_token = load_manifest_keys(manifest_path)
        train_infos = filter_by_manifest(
            source_train_infos,
            token_to_index,
            timestamp_to_index,
            index_to_token,
        )[: args.train_samples]
        val_infos = filter_by_manifest(
            source_val_infos,
            token_to_index,
            timestamp_to_index,
            index_to_token,
        )[: args.val_samples]
        if not train_infos or not val_infos:
            combined = filter_by_manifest(
                source_train_infos + source_val_infos,
                token_to_index,
                timestamp_to_index,
                index_to_token,
            )
            train_infos = combined[: args.train_samples]
            val_start = min(len(train_infos), len(combined))
            val_infos = combined[val_start : val_start + args.val_samples]
    else:
        train_infos = source_train_infos[: args.train_samples]
        val_infos = source_val_infos[: args.val_samples]
    if not train_infos:
        raise RuntimeError(f"No train infos loaded from {source_train}")
    if not val_infos:
        raise RuntimeError(f"No val infos loaded from {source_val}")

    cache_reliability_dir = (
        Path(args.cache_reliability_dir).expanduser().resolve() if args.cache_reliability_dir else None
    )
    if cache_reliability_dir is not None:
        materialize_cache_backed_raw(train_infos + val_infos, source_root, output_root, cache_reliability_dir)

    dump_infos(output_root / "info" / f"{info_name}_train.pkl", train_infos)
    dump_infos(output_root / "info" / f"{info_name}_val.pkl", val_infos)

    raw_target = source_root / "raw"
    raw_link = output_root / "raw"
    if cache_reliability_dir is None:
        reset_path(raw_link)
        os.symlink(raw_target, raw_link, target_is_directory=True)

    print(f"[smoke-infos] source_root={source_root}")
    print(f"[smoke-infos] output_root={output_root}")
    print(f"[smoke-infos] train={len(train_infos)} val={len(val_infos)}")
    if cache_reliability_dir is None:
        print(f"[smoke-infos] raw -> {raw_target}")
    else:
        print(f"[smoke-infos] raw=cache-backed from {cache_reliability_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
