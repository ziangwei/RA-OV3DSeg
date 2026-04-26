from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402
from ra_ov3dseg.visualization.visualize_projection import save_projection_overlay  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将投影结果绘制到 6 张相机图像上。")
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes 数据根目录。")
    parser.add_argument("--version", default="v1.0-mini", type=str, help="nuScenes 版本。")
    parser.add_argument("--sample_idx", default=None, type=int, help="指定单个 sample 索引。")
    parser.add_argument("--start_idx", default=0, type=int, help="批量可视化时的起始 sample 索引。")
    parser.add_argument("--max_samples", default=1, type=int, help="批量可视化多少个 sample。")
    parser.add_argument("--projection_npz", default=None, type=str, help="单个投影结果 .npz 文件路径。")
    parser.add_argument(
        "--projection_dir",
        default="outputs/projections",
        type=str,
        help="批量可视化时，投影 .npz 所在目录。",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/visualizations",
        type=str,
        help="overlay 图输出目录。",
    )
    parser.add_argument(
        "--near_max_depth",
        default=30.0,
        type=float,
        help="近距离 sanity check 图的最大深度，单位米。",
    )
    parser.add_argument(
        "--full_max_points",
        default=15000,
        type=int,
        help="全量 overlay 最多绘制多少个点，避免图像过于拥挤。",
    )
    parser.add_argument(
        "--near_max_points",
        default=6000,
        type=int,
        help="近距离 overlay 最多绘制多少个点，便于人工检查。",
    )
    parser.add_argument(
        "--point_size",
        default=4.0,
        type=float,
        help="绘制点大小。",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="如果某个 sample 的 overlay 已存在，则跳过。",
    )
    return parser


def infer_sample_idx_from_path(path: Path) -> int | None:
    match = re.search(r"sample_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("visualize_projection")

    dataset = NuScenesDataset(
        dataroot=args.dataroot,
        version=args.version,
        verbose=False,
    )
    output_dir = ensure_dir(args.output_dir)
    projection_dir = Path(args.projection_dir).resolve()

    if args.projection_npz is not None:
        projection_jobs = []
        projection_npz = Path(args.projection_npz).resolve()
        if not projection_npz.exists():
            raise FileNotFoundError(f"projection npz not found: {projection_npz}")
        inferred_sample_idx = args.sample_idx
        if inferred_sample_idx is None:
            inferred_sample_idx = infer_sample_idx_from_path(projection_npz)
        if inferred_sample_idx is None:
            raise ValueError("sample_idx is required when projection_npz filename does not contain sample_XXXX.")
        projection_jobs.append((inferred_sample_idx, projection_npz))
    else:
        sample_indices = dataset.resolve_sample_indices(
            sample_idx=args.sample_idx,
            start_idx=args.start_idx,
            max_samples=args.max_samples,
        )
        projection_jobs = []
        for sample_idx in sample_indices:
            projection_npz = projection_dir / f"sample_{sample_idx:04d}_projection.npz"
            if not projection_npz.exists():
                raise FileNotFoundError(f"projection npz not found: {projection_npz}")
            projection_jobs.append((sample_idx, projection_npz))

    batch_manifest = {
        "version": args.version,
        "dataroot": str(Path(args.dataroot).resolve()),
        "projection_dir": str(projection_dir),
        "output_dir": str(output_dir.resolve()),
        "samples": [],
    }

    for sample_idx, projection_npz in projection_jobs:
        logger.info("========== sample_idx=%d ==========", sample_idx)
        sample = dataset.get_sample_by_index(sample_idx)
        data = np.load(projection_npz, allow_pickle=False)
        camera_names = [str(name) for name in data["camera_names"].tolist()]
        image_rel_paths = [str(path) for path in data["image_rel_paths"].tolist()]
        uv = data["uv"]
        depth = data["depth"]
        valid_masks = data["valid_masks"].astype(bool)

        prefix = f"sample_{sample_idx:04d}"
        manifest = {
            "sample_idx": sample_idx,
            "sample_token": str(data["sample_token"].item()),
            "projection_npz": str(projection_npz),
            "outputs": [],
        }

        should_skip_sample = False
        if args.skip_existing:
            probe_path = output_dir / f"{prefix}_{camera_names[0]}_overlay_near.png"
            should_skip_sample = probe_path.exists()
        if should_skip_sample:
            logger.info("skip existing overlays for sample_idx=%d", sample_idx)
            manifest["status"] = "skipped_existing"
            batch_manifest["samples"].append(manifest)
            continue

        for camera_idx, camera_name in enumerate(camera_names):
            image_path = None
            if image_rel_paths[camera_idx]:
                candidate = Path(args.dataroot) / image_rel_paths[camera_idx]
                if candidate.exists():
                    image_path = candidate

            if image_path is None:
                fallback = dataset.get_sample_data_path_from_channel(sample, camera_name)
                if fallback is not None and fallback.exists():
                    image_path = fallback

            if image_path is None:
                logger.warning("%s image not found, skip overlay.", camera_name)
                continue

            camera_valid_count = int(valid_masks[camera_idx].sum())

            overlay_full_path = output_dir / f"{prefix}_{camera_name}_overlay_full.png"
            overlay_near_path = output_dir / f"{prefix}_{camera_name}_overlay_near.png"

            full_stats = save_projection_overlay(
                image_path=image_path,
                uv=uv[camera_idx],
                depth=depth[camera_idx],
                valid_mask=valid_masks[camera_idx],
                output_path=overlay_full_path,
                title=f"{prefix} | {camera_name} | valid={camera_valid_count}",
                min_depth=0.0,
                max_depth=None,
                max_points=args.full_max_points,
                point_size=args.point_size,
                alpha=0.75,
            )
            near_stats = save_projection_overlay(
                image_path=image_path,
                uv=uv[camera_idx],
                depth=depth[camera_idx],
                valid_mask=valid_masks[camera_idx],
                output_path=overlay_near_path,
                title=f"{prefix} | {camera_name} | near<= {args.near_max_depth:.1f}m",
                min_depth=0.0,
                max_depth=args.near_max_depth,
                max_points=args.near_max_points,
                point_size=max(args.point_size, 5.0),
                alpha=0.85,
            )

            manifest["outputs"].append(
                {
                    "camera_name": camera_name,
                    "image_path": str(image_path),
                    "valid_points": camera_valid_count,
                    "overlay_full_path": str(overlay_full_path),
                    "overlay_near_path": str(overlay_near_path),
                    "full_overlay_stats": full_stats,
                    "near_overlay_stats": near_stats,
                }
            )
            logger.info(
                "%s overlays saved | full=%s (%d drawn) | near=%s (%d drawn)",
                camera_name,
                overlay_full_path,
                full_stats["drawn_points"],
                overlay_near_path,
                near_stats["drawn_points"],
            )

        manifest["status"] = "done"
        manifest_path = output_dir / f"{prefix}_overlay_manifest.json"
        save_json(manifest_path, manifest)
        logger.info("overlay manifest saved to: %s", manifest_path)
        batch_manifest["samples"].append(manifest)

    if len(projection_jobs) > 1:
        batch_manifest_path = output_dir / "batch_overlay_manifest.json"
        save_json(batch_manifest_path, batch_manifest)
        logger.info("batch overlay manifest saved to: %s", batch_manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
