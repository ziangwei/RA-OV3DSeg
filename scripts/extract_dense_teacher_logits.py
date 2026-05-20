from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import CAMERA_CHANNELS, NuScenesDataset  # noqa: E402
from ra_ov3dseg.models.clipseg_dense_teacher import CLIPSegDenseTeacher  # noqa: E402
from ra_ov3dseg.models.groupvit_dense_teacher import GroupViTDenseTeacher  # noqa: E402
from ra_ov3dseg.models.teacher_registry import CLIPSEG_DENSE, GROUPVIT_DENSE, describe_teacher  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, is_valid_npz, load_sample_indices, load_text_lines, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract dense open-vocabulary teacher logits for nuScenes cameras.")
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--version", default="v1.0-mini", type=str)
    parser.add_argument("--sample_idx", default=None, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--max_samples", default=1, type=int)
    parser.add_argument(
        "--sample_indices_path",
        default=None,
        type=str,
        help="JSON or text file with explicit sample indices. Overrides start_idx/max_samples.",
    )
    parser.add_argument("--teacher_backend", default=CLIPSEG_DENSE, choices=[CLIPSEG_DENSE, GROUPVIT_DENSE])
    parser.add_argument("--model_name", default="CIDAS/clipseg-rd64-refined", type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--prompt_template", default="a {} in a driving scene", type=str)
    parser.add_argument("--prompt_batch_size", default=8, type=int)
    parser.add_argument("--logit_height", default=0, type=int, help="Optional saved dense logit height. <=0 keeps teacher size.")
    parser.add_argument("--logit_width", default=0, type=int, help="Optional saved dense logit width. <=0 keeps teacher size.")
    parser.add_argument("--logit_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--output_dir", default="outputs/dense_teacher_logits", type=str)
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("extract_dense_teacher_logits")
    dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=False)
    if args.sample_indices_path is not None:
        if args.sample_idx is not None:
            raise ValueError("--sample_idx cannot be combined with --sample_indices_path.")
        sample_indices = load_sample_indices(args.sample_indices_path)
        for sample_idx in sample_indices:
            dataset.get_sample_by_index(sample_idx)
    else:
        sample_indices = dataset.resolve_sample_indices(
            sample_idx=args.sample_idx,
            start_idx=args.start_idx,
            max_samples=args.max_samples,
        )
    class_names = load_text_lines(args.class_names_path)
    if not class_names:
        raise ValueError("class_names_path is empty.")
    output_dir = ensure_dir(args.output_dir)
    teacher_spec = describe_teacher(args.teacher_backend)
    if args.teacher_backend == CLIPSEG_DENSE:
        teacher = CLIPSegDenseTeacher(
            model_name=args.model_name,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    elif args.teacher_backend == GROUPVIT_DENSE:
        teacher = GroupViTDenseTeacher(
            model_name=args.model_name,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            output_height=args.logit_height,
            output_width=args.logit_width,
        )
    else:
        raise ValueError(f"Unsupported dense teacher backend: {args.teacher_backend}")
    logit_dtype = np.float16 if args.logit_dtype == "float16" else np.float32

    batch_summary = {
        "version": args.version,
        "teacher_backend": args.teacher_backend,
        "teacher_role": teacher_spec.role,
        "model_name": args.model_name,
        "sample_indices": sample_indices,
        "samples": [],
    }

    for sample_idx in sample_indices:
        logger.info("========== sample_idx=%d ==========", sample_idx)
        prefix = f"sample_{sample_idx:04d}"
        output_npz = output_dir / f"{prefix}_dense_teacher_logits.npz"
        summary_json = output_dir / f"{prefix}_dense_teacher_logits_summary.json"
        if args.skip_existing and summary_json.exists() and is_valid_npz(output_npz, required_keys=("dense_logits",)):
            logger.info("skip existing dense teacher logits for sample_idx=%d", sample_idx)
            batch_summary["samples"].append({"sample_idx": sample_idx, "status": "skipped_existing"})
            continue
        if args.skip_existing and output_npz.exists() and not is_valid_npz(output_npz, required_keys=("dense_logits",)):
            logger.warning("invalid existing dense teacher npz, recomputing: %s", output_npz)

        sample = dataset.get_sample_by_index(sample_idx)
        dense_logits = None
        camera_available = np.zeros(len(CAMERA_CHANNELS), dtype=bool)
        image_rel_paths: list[str] = []
        image_widths = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
        image_heights = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
        logit_widths = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
        logit_heights = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
        summary_cameras = []
        prompts = None

        for camera_idx, camera_name in enumerate(CAMERA_CHANNELS):
            image_path = dataset.get_sample_data_path_from_channel(sample, camera_name)
            image_rel_path = dataset.get_sample_data_relpath_from_channel(sample, camera_name)
            if image_path is None or image_rel_path is None or not image_path.exists():
                image_rel_paths.append("")
                summary_cameras.append({"camera_name": camera_name, "available": False, "image_path": ""})
                continue

            encoded = teacher.encode_image_logits(
                image_path=image_path,
                class_names=class_names,
                prompt_template=args.prompt_template,
                prompt_batch_size=args.prompt_batch_size,
            )
            camera_logits = encoded["dense_logits"].astype(logit_dtype)
            metadata = encoded["metadata"]
            if dense_logits is None:
                dense_logits = np.zeros(
                    (len(CAMERA_CHANNELS),) + camera_logits.shape,
                    dtype=logit_dtype,
                )
                prompts = encoded["prompts"]
            dense_logits[camera_idx] = camera_logits
            camera_available[camera_idx] = True
            image_rel_paths.append(image_rel_path)
            image_widths[camera_idx] = metadata["original_width"]
            image_heights[camera_idx] = metadata["original_height"]
            logit_widths[camera_idx] = metadata["logit_width"]
            logit_heights[camera_idx] = metadata["logit_height"]
            summary_cameras.append(
                {
                    "camera_name": camera_name,
                    "available": True,
                    "image_path": str(image_path),
                    "logit_height": metadata["logit_height"],
                    "logit_width": metadata["logit_width"],
                    "num_classes": metadata["num_classes"],
                }
            )
            logger.info(
                "%s | dense_logits=%dx%dx%d",
                camera_name,
                metadata["num_classes"],
                metadata["logit_height"],
                metadata["logit_width"],
            )

        if dense_logits is None or prompts is None:
            raise RuntimeError(f"No camera image available for sample_idx={sample_idx}")

        save_npz(
            output_npz,
            sample_idx=np.array(sample_idx, dtype=np.int32),
            sample_token=np.array(sample["token"]),
            teacher_backend=np.array(args.teacher_backend),
            teacher_role=np.array(teacher_spec.role),
            teacher_feature_granularity=np.array(teacher_spec.feature_granularity),
            is_baseline_teacher=np.array(teacher_spec.is_baseline),
            model_name=np.array(args.model_name),
            cache_dir=np.array(args.cache_dir or ""),
            local_files_only=np.array(args.local_files_only),
            camera_names=np.asarray(CAMERA_CHANNELS),
            camera_available=camera_available,
            image_rel_paths=np.asarray(image_rel_paths),
            image_widths=image_widths,
            image_heights=image_heights,
            logit_widths=logit_widths,
            logit_heights=logit_heights,
            class_names=np.asarray(class_names),
            prompts=np.asarray(prompts),
            dense_logits=dense_logits,
        )
        summary = {
            "sample_idx": sample_idx,
            "sample_token": sample["token"],
            "teacher_backend": args.teacher_backend,
            "teacher_role": teacher_spec.role,
            "teacher_description": teacher_spec.description,
            "model_name": args.model_name,
            "cache_dir": args.cache_dir or "",
            "local_files_only": args.local_files_only,
            "logit_dtype": args.logit_dtype,
            "num_classes": len(class_names),
            "prompt_template": args.prompt_template,
            "cameras": summary_cameras,
            "output_npz": str(output_npz),
        }
        save_json(summary_json, summary)
        batch_summary["samples"].append(
            {
                "sample_idx": sample_idx,
                "status": "done",
                "output_npz": str(output_npz),
                "summary_json": str(summary_json),
            }
        )
        logger.info("dense teacher logits saved to: %s", output_npz)

    if len(sample_indices) > 1:
        batch_summary_path = output_dir / "batch_dense_teacher_logits_summary.json"
        save_json(batch_summary_path, batch_summary)
        logger.info("batch dense teacher summary saved to: %s", batch_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
