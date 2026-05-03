from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.models.voxelization import (  # noqa: E402
    VoxelizationConfig,
    make_point_input_features,
    voxelize_point_features,
)
from ra_ov3dseg.utils.io import ensure_dir, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check MVP-v5 voxelization on precomputed point features.")
    parser.add_argument("--sample_idx", default=0, type=int)
    parser.add_argument("--point_feature_npz", default=None, type=str)
    parser.add_argument("--point_feature_dir", default="outputs/point_features", type=str)
    parser.add_argument("--voxel_size", default=(0.2, 0.2, 0.2), nargs=3, type=float, metavar=("VX", "VY", "VZ"))
    parser.add_argument(
        "--point_cloud_range",
        default=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        nargs=6,
        type=float,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
    )
    parser.add_argument("--output_dir", default="outputs/voxelization", type=str)
    return parser


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("check_voxelization")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("check_voxelization.py requires PyTorch.") from exc

    prefix = f"sample_{args.sample_idx:04d}"
    point_feature_npz = (
        Path(args.point_feature_npz).resolve()
        if args.point_feature_npz is not None
        else Path(args.point_feature_dir).resolve() / f"{prefix}_point_features.npz"
    )
    if not point_feature_npz.exists():
        raise FileNotFoundError(f"point feature npz not found: {point_feature_npz}")

    output_dir = ensure_dir(args.output_dir)
    point_data = load_npz(point_feature_npz)
    point_xyz_np = point_data["point_xyz"].astype(np.float32)
    point_xyz = torch.from_numpy(point_xyz_np)
    point_batch_indices = torch.zeros(point_xyz.shape[0], dtype=torch.long)
    config = VoxelizationConfig(
        voxel_size=tuple(args.voxel_size),
        point_cloud_range=tuple(args.point_cloud_range),
    )
    point_input_features = make_point_input_features(point_xyz, config.point_cloud_range)
    voxelized = voxelize_point_features(
        point_xyz=point_xyz,
        point_features=point_input_features,
        point_batch_indices=point_batch_indices,
        config=config,
    )

    valid_mask = voxelized["valid_point_mask"].detach().cpu().numpy().astype(bool)
    voxel_coords = voxelized["voxel_coords"].detach().cpu().numpy()
    point_to_voxel = voxelized["point_to_voxel"].detach().cpu().numpy()
    valid_point_to_voxel = point_to_voxel[valid_mask]
    unique_voxel_refs = int(np.unique(valid_point_to_voxel).shape[0]) if valid_point_to_voxel.shape[0] else 0

    summary = {
        "status": "pass",
        "sample_idx": args.sample_idx,
        "point_feature_npz": str(point_feature_npz),
        "num_points": int(point_xyz.shape[0]),
        "num_valid_points": int(valid_mask.sum()),
        "valid_point_ratio": float(valid_mask.sum() / max(point_xyz.shape[0], 1)),
        "num_voxels": int(voxel_coords.shape[0]),
        "unique_voxel_refs": unique_voxel_refs,
        "voxel_size": list(args.voxel_size),
        "point_cloud_range": list(args.point_cloud_range),
        "spatial_shape_zyx": config.spatial_shape_zyx,
        "voxel_coord_min_bzyx": voxel_coords.min(axis=0).astype(int).tolist() if voxel_coords.shape[0] else [],
        "voxel_coord_max_bzyx": voxel_coords.max(axis=0).astype(int).tolist() if voxel_coords.shape[0] else [],
    }
    summary_path = output_dir / f"{prefix}_voxelization_summary.json"
    save_json(summary_path, summary)
    logger.info(
        "voxelization PASS | points=%d | valid=%d | voxels=%d | summary=%s",
        summary["num_points"],
        summary["num_valid_points"],
        summary["num_voxels"],
        summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
