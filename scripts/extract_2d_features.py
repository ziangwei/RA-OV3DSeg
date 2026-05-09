from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import CAMERA_CHANNELS, NuScenesDataset  # noqa: E402
from ra_ov3dseg.models.teacher_registry import (  # noqa: E402
    CLIP_PATCH_BASELINE,
    build_image_teacher,
    describe_teacher,
)
from ra_ov3dseg.utils.io import ensure_dir, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="提取一个或多个 nuScenes sample 的 2D image patch features。")
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes 数据根目录。")
    parser.add_argument("--version", default="v1.0-mini", type=str, help="nuScenes 版本。")
    parser.add_argument("--sample_idx", default=None, type=int, help="指定单个 sample 索引。")
    parser.add_argument("--start_idx", default=0, type=int, help="批量提取时的起始 sample 索引。")
    parser.add_argument("--max_samples", default=1, type=int, help="批量提取多少个 sample。")
    parser.add_argument(
        "--teacher_backend",
        default=CLIP_PATCH_BASELINE,
        choices=[CLIP_PATCH_BASELINE],
        help=(
            "2D patch-feature backend for MVP-v1. Dense teachers use "
            "scripts/extract_dense_teacher_logits.py instead."
        ),
    )
    parser.add_argument(
        "--model_name",
        default="openai/clip-vit-base-patch16",
        type=str,
        help="Hugging Face 上的图像/文本共享模型名，例如 openai/clip-vit-base-patch16。",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Hugging Face 模型缓存目录，例如 /path/to/huggingface_cache。",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="只读取本地缓存，不联网下载模型。",
    )
    parser.add_argument("--device", default="auto", type=str, help="运行设备：auto/cpu/cuda。")
    parser.add_argument(
        "--feature_dtype",
        default="float16",
        choices=["float16", "float32"],
        help="保存 feature map 时使用的 dtype。",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/features2d",
        type=str,
        help="2D feature 输出目录。",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="如果该 sample 的 2D feature 文件已存在，则跳过。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("extract_2d_features")

    dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=False)
    sample_indices = dataset.resolve_sample_indices(
        sample_idx=args.sample_idx,
        start_idx=args.start_idx,
        max_samples=args.max_samples,
    )
    output_dir = ensure_dir(args.output_dir)
    teacher_spec = describe_teacher(args.teacher_backend)
    encoder = build_image_teacher(
        teacher_backend=args.teacher_backend,
        model_name=args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    feature_dtype = np.float16 if args.feature_dtype == "float16" else np.float32

    batch_summary = {
        "version": args.version,
        "dataroot": str(Path(args.dataroot).resolve()),
        "teacher_backend": args.teacher_backend,
        "teacher_role": teacher_spec.role,
        "teacher_feature_granularity": teacher_spec.feature_granularity,
        "is_baseline_teacher": teacher_spec.is_baseline,
        "model_name": args.model_name,
        "cache_dir": args.cache_dir or "",
        "local_files_only": args.local_files_only,
        "device": encoder.device,
        "sample_indices": sample_indices,
        "samples": [],
    }

    for sample_idx in sample_indices:
        logger.info("========== sample_idx=%d ==========", sample_idx)
        prefix = f"sample_{sample_idx:04d}"
        feature_npz = output_dir / f"{prefix}_image_features.npz"
        summary_json = output_dir / f"{prefix}_image_features_summary.json"

        if args.skip_existing and feature_npz.exists() and summary_json.exists():
            logger.info("skip existing 2D features for sample_idx=%d", sample_idx)
            batch_summary["samples"].append(
                {
                    "sample_idx": sample_idx,
                    "status": "skipped_existing",
                    "feature_npz": str(feature_npz),
                    "summary_json": str(summary_json),
                }
            )
            continue

        sample = dataset.get_sample_by_index(sample_idx)
        feature_maps = None
        image_embeddings = None
        camera_available = np.zeros(len(CAMERA_CHANNELS), dtype=bool)
        image_rel_paths: list[str] = []
        image_widths = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
        image_heights = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
        resized_widths = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
        resized_heights = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
        summary_cameras = []

        for camera_idx, camera_name in enumerate(CAMERA_CHANNELS):
            image_path = dataset.get_sample_data_path_from_channel(sample, camera_name)
            image_rel_path = dataset.get_sample_data_relpath_from_channel(sample, camera_name)
            if image_path is None or image_rel_path is None or not image_path.exists():
                image_rel_paths.append("")
                summary_cameras.append(
                    {
                        "camera_name": camera_name,
                        "available": False,
                        "image_path": "",
                    }
                )
                continue

            encoded = encoder.encode_image(image_path)
            patch_feature_map = encoded["patch_feature_map"]
            image_embedding = encoded["image_embedding"]
            metadata = encoded["metadata"]

            if feature_maps is None:
                feature_maps = np.zeros(
                    (len(CAMERA_CHANNELS),) + patch_feature_map.shape,
                    dtype=feature_dtype,
                )
                image_embeddings = np.zeros(
                    (len(CAMERA_CHANNELS), image_embedding.shape[0]),
                    dtype=feature_dtype,
                )

            feature_maps[camera_idx] = patch_feature_map.astype(feature_dtype)
            image_embeddings[camera_idx] = image_embedding.astype(feature_dtype)
            camera_available[camera_idx] = True
            image_rel_paths.append(image_rel_path)
            image_widths[camera_idx] = metadata["original_width"]
            image_heights[camera_idx] = metadata["original_height"]
            resized_widths[camera_idx] = metadata["resized_width"]
            resized_heights[camera_idx] = metadata["resized_height"]
            summary_cameras.append(
                {
                    "camera_name": camera_name,
                    "available": True,
                    "image_path": str(image_path),
                    "feature_grid_height": metadata["feature_grid_height"],
                    "feature_grid_width": metadata["feature_grid_width"],
                    "feature_dim": metadata["feature_dim"],
                }
            )
            logger.info(
                "%s | grid=%dx%d | dim=%d",
                camera_name,
                metadata["feature_grid_height"],
                metadata["feature_grid_width"],
                metadata["feature_dim"],
            )

        if feature_maps is None or image_embeddings is None:
            raise RuntimeError(f"No camera image available for sample_idx={sample_idx}")

        npz_data = {
            "sample_idx": np.array(sample_idx, dtype=np.int32),
            "sample_token": np.array(sample["token"]),
            "camera_names": np.asarray(CAMERA_CHANNELS),
            "camera_available": camera_available,
            "image_rel_paths": np.asarray(image_rel_paths),
            "image_widths": image_widths,
            "image_heights": image_heights,
            "resized_widths": resized_widths,
            "resized_heights": resized_heights,
            "feature_maps": feature_maps,
            "image_embeddings": image_embeddings,
            "teacher_backend": np.array(args.teacher_backend),
            "teacher_role": np.array(teacher_spec.role),
            "teacher_feature_granularity": np.array(teacher_spec.feature_granularity),
            "is_baseline_teacher": np.array(teacher_spec.is_baseline),
            "model_name": np.array(args.model_name),
            "cache_dir": np.array(args.cache_dir or ""),
            "local_files_only": np.array(args.local_files_only),
        }
        summary = {
            "sample_idx": sample_idx,
            "sample_token": sample["token"],
            "teacher_backend": args.teacher_backend,
            "teacher_role": teacher_spec.role,
            "teacher_feature_granularity": teacher_spec.feature_granularity,
            "is_baseline_teacher": teacher_spec.is_baseline,
            "teacher_description": teacher_spec.description,
            "model_name": args.model_name,
            "cache_dir": args.cache_dir or "",
            "local_files_only": args.local_files_only,
            "feature_dtype": args.feature_dtype,
            "cameras": summary_cameras,
        }
        save_npz(feature_npz, **npz_data)
        save_json(summary_json, summary)
        logger.info("image feature npz saved to: %s", feature_npz)
        logger.info("image feature summary saved to: %s", summary_json)

        batch_summary["samples"].append(
            {
                "sample_idx": sample_idx,
                "status": "done",
                "feature_npz": str(feature_npz),
                "summary_json": str(summary_json),
            }
        )

    if len(sample_indices) > 1:
        batch_summary_path = output_dir / "batch_image_features_summary.json"
        save_json(batch_summary_path, batch_summary)
        logger.info("batch image feature summary saved to: %s", batch_summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
