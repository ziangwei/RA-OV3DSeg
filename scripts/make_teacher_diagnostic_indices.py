from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a reproducible cross-scene sample-index list for Stage 3 teacher diagnostics."
    )
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes data root.")
    parser.add_argument("--version", default="v1.0-trainval", type=str, help="nuScenes version.")
    parser.add_argument("--max_samples", default=32, type=int, help="Number of diagnostic samples to select.")
    parser.add_argument(
        "--output_path",
        default="outputs/diagnostics/stage3_teacher_indices_32.json",
        type=str,
        help="Output JSON manifest path.",
    )
    return parser


def evenly_spaced_positions(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))

    raw_positions = np.linspace(0, total - 1, count)
    selected: list[int] = []
    used = set()
    for raw_position in raw_positions:
        index = int(round(float(raw_position)))
        if index in used:
            left = index - 1
            right = index + 1
            while left >= 0 or right < total:
                if left >= 0 and left not in used:
                    index = left
                    break
                if right < total and right not in used:
                    index = right
                    break
                left -= 1
                right += 1
        used.add(index)
        selected.append(index)
    return selected


def group_samples_by_scene(dataset: NuScenesDataset) -> list[dict[str, Any]]:
    scene_groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for sample_idx, sample in enumerate(dataset.samples):
        scene_token = str(sample["scene_token"])
        if scene_token not in scene_groups:
            scene = dataset.get_scene_record(sample)
            scene_groups[scene_token] = {
                "scene_token": scene_token,
                "scene_name": scene["name"],
                "sample_indices": [],
            }
        scene_groups[scene_token]["sample_indices"].append(sample_idx)
    return list(scene_groups.values())


def select_cross_scene_indices(dataset: NuScenesDataset, max_samples: int) -> list[int]:
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")

    scene_groups = group_samples_by_scene(dataset)
    if not scene_groups:
        return []

    selected: list[int] = []
    selected_set = set()
    scene_positions = evenly_spaced_positions(len(scene_groups), min(max_samples, len(scene_groups)))
    for scene_position in scene_positions:
        sample_indices = scene_groups[scene_position]["sample_indices"]
        sample_idx = sample_indices[len(sample_indices) // 2]
        selected.append(sample_idx)
        selected_set.add(sample_idx)

    if len(selected) < max_samples:
        remaining = [idx for idx in range(len(dataset)) if idx not in selected_set]
        extra_positions = evenly_spaced_positions(len(remaining), max_samples - len(selected))
        for position in extra_positions:
            selected.append(remaining[position])

    return selected[:max_samples]


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("make_teacher_diagnostic_indices")
    dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=False)
    sample_indices = select_cross_scene_indices(dataset, max_samples=args.max_samples)
    output_path = Path(args.output_path).expanduser().resolve()
    ensure_dir(output_path.parent)

    samples = []
    for sample_idx in sample_indices:
        sample = dataset.get_sample_by_index(sample_idx)
        scene = dataset.get_scene_record(sample)
        samples.append(
            {
                "sample_idx": sample_idx,
                "sample_token": sample["token"],
                "scene_token": scene["token"],
                "scene_name": scene["name"],
                "timestamp": int(sample["timestamp"]),
            }
        )

    save_json(
        output_path,
        {
            "version": args.version,
            "dataroot": str(Path(args.dataroot).expanduser().resolve()),
            "strategy": "evenly spaced scenes, center sample per selected scene",
            "num_total_samples": len(dataset),
            "num_selected_samples": len(sample_indices),
            "sample_indices": sample_indices,
            "samples": samples,
        },
    )
    logger.info("teacher diagnostic indices saved | samples=%d | output=%s", len(sample_indices), output_path)
    print(f"saved {len(sample_indices)} teacher diagnostic sample indices to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
