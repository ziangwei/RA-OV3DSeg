from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_mini_dataset import NuScenesMiniDataset  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402
from ra_ov3dseg.visualization.visualize_projection import save_projection_overlay  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将投影结果绘制到 6 张相机图像上。")
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes 数据根目录。")
    parser.add_argument("--version", default="v1.0-mini", type=str, help="nuScenes 版本。")
    parser.add_argument("--sample_idx", default=0, type=int, help="按时间排序后的 sample 索引。")
    parser.add_argument("--projection_npz", required=True, type=str, help="投影结果 .npz 文件路径。")
    parser.add_argument(
        "--output_dir",
        default="outputs/visualizations",
        type=str,
        help="overlay 图输出目录。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("visualize_projection")

    dataset = NuScenesMiniDataset(
        dataroot=args.dataroot,
        version=args.version,
        verbose=False,
    )
    sample = dataset.get_sample_by_index(args.sample_idx)

    projection_npz = Path(args.projection_npz).resolve()
    if not projection_npz.exists():
        raise FileNotFoundError(f"projection npz not found: {projection_npz}")

    data = np.load(projection_npz, allow_pickle=False)
    camera_names = [str(name) for name in data["camera_names"].tolist()]
    image_rel_paths = [str(path) for path in data["image_rel_paths"].tolist()]
    uv = data["uv"]
    depth = data["depth"]
    valid_masks = data["valid_masks"].astype(bool)

    output_dir = ensure_dir(args.output_dir)
    prefix = f"sample_{args.sample_idx:04d}"
    manifest = {
        "sample_idx": args.sample_idx,
        "sample_token": str(data["sample_token"].item()),
        "projection_npz": str(projection_npz),
        "outputs": [],
    }

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

        overlay_path = output_dir / f"{prefix}_{camera_name}_overlay.png"
        save_projection_overlay(
            image_path=image_path,
            uv=uv[camera_idx],
            depth=depth[camera_idx],
            valid_mask=valid_masks[camera_idx],
            output_path=overlay_path,
            title=f"{prefix} | {camera_name}",
        )
        manifest["outputs"].append(
            {
                "camera_name": camera_name,
                "image_path": str(image_path),
                "overlay_path": str(overlay_path),
            }
        )
        logger.info("overlay saved: %s", overlay_path)

    manifest_path = output_dir / f"{prefix}_overlay_manifest.json"
    save_json(manifest_path, manifest)
    logger.info("overlay manifest saved to: %s", manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
