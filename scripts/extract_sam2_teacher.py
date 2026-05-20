from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import CAMERA_CHANNELS, NuScenesDataset  # noqa: E402
from ra_ov3dseg.models.sam2_siglip_teacher import SAM2SigLIPTeacher  # noqa: E402
from ra_ov3dseg.models.teacher_registry import SAM2_SIGLIP, describe_teacher  # noqa: E402
from ra_ov3dseg.training.labels import NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, is_valid_npz, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402
from ra_ov3dseg.utils.run_conclusion import RunConclusion  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract SAM2+SigLIP dense teacher logits for nuScenes cameras.")
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--version", default="v1.0-trainval", type=str)
    parser.add_argument("--sample_idx", default=None, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--max_samples", default=1, type=int)
    parser.add_argument("--sam_model_id", default="facebook/sam2.1-hiera-small", type=str)
    parser.add_argument("--siglip_model_name", default="google/siglip-base-patch16-224", type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prompt_template", default="a photo of a {}", type=str)
    parser.add_argument("--points_per_side", default=24, type=int)
    parser.add_argument("--points_per_batch", default=64, type=int)
    parser.add_argument("--pred_iou_thresh", default=0.80, type=float)
    parser.add_argument("--stability_score_thresh", default=0.92, type=float)
    parser.add_argument("--min_mask_region_area", default=100, type=int)
    parser.add_argument("--classification_batch_size", default=16, type=int)
    parser.add_argument("--logit_temperature", default=0.07, type=float)
    parser.add_argument("--logit_height", default=450, type=int)
    parser.add_argument("--logit_width", default=800, type=int)
    parser.add_argument("--logit_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--output_dir", default="outputs/teacher_caches/sam2_siglip", type=str)
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def main() -> int:
    start_time = time.monotonic()
    args = build_parser().parse_args()
    logger = setup_logger("extract_sam2_teacher")
    output_dir = ensure_dir(args.output_dir)
    logit_dtype = np.float16 if args.logit_dtype == "float16" else np.float32
    status = "success"
    notes = "SAM2+SigLIP dense teacher extraction"
    artifacts: list[str] = []
    processed_cameras = 0

    try:
        dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=False)
        sample_indices = dataset.resolve_sample_indices(
            sample_idx=args.sample_idx,
            start_idx=args.start_idx,
            max_samples=args.max_samples,
        )
        teacher_spec = describe_teacher(SAM2_SIGLIP)
        class_names = list(NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES[1:])
        teacher = SAM2SigLIPTeacher(
            sam_model_id=args.sam_model_id,
            siglip_model_name=args.siglip_model_name,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            min_mask_region_area=args.min_mask_region_area,
            classification_batch_size=args.classification_batch_size,
            logit_temperature=args.logit_temperature,
            output_height=args.logit_height,
            output_width=args.logit_width,
        )
        batch_summary = {
            "version": args.version,
            "teacher_backend": SAM2_SIGLIP,
            "teacher_role": teacher_spec.role,
            "sam_model_id": args.sam_model_id,
            "siglip_model_name": args.siglip_model_name,
            "sample_indices": sample_indices,
            "samples": [],
        }

        for sample_idx in sample_indices:
            logger.info("========== sample_idx=%d ==========", sample_idx)
            sample = dataset.get_sample_by_index(sample_idx)
            prefix = f"sample_{sample_idx:04d}"
            output_npz = output_dir / f"{prefix}_dense_teacher_logits.npz"
            summary_json = output_dir / f"{prefix}_dense_teacher_logits_summary.json"
            if args.skip_existing and summary_json.exists() and is_valid_npz(output_npz, required_keys=("dense_logits",)):
                logger.info("skip existing SAM2+SigLIP teacher for sample_idx=%d", sample_idx)
                batch_summary["samples"].append({"sample_idx": sample_idx, "status": "skipped_existing"})
                artifacts.append(str(output_npz))
                continue

            sample_dir = ensure_dir(output_dir / str(sample["token"]))
            dense_logits = None
            dense_confidence = None
            dense_pred_label = None
            camera_available = np.zeros(len(CAMERA_CHANNELS), dtype=bool)
            image_rel_paths: list[str] = []
            image_widths = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
            image_heights = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
            logit_widths = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
            logit_heights = np.zeros(len(CAMERA_CHANNELS), dtype=np.int32)
            summary_cameras = []
            prompts = None
            output_class_names = None

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
                )
                camera_logits = encoded["dense_logits"].astype(logit_dtype)
                camera_conf = encoded["dense_confidence"].astype(np.float16)
                camera_pred = encoded["dense_pred_label"].astype(np.int16)
                metadata = encoded["metadata"]
                if dense_logits is None:
                    dense_logits = np.zeros((len(CAMERA_CHANNELS),) + camera_logits.shape, dtype=logit_dtype)
                    dense_confidence = np.zeros((len(CAMERA_CHANNELS),) + camera_conf.shape, dtype=np.float16)
                    dense_pred_label = np.full((len(CAMERA_CHANNELS),) + camera_pred.shape, -1, dtype=np.int16)
                    prompts = encoded["prompts"]
                    output_class_names = encoded["class_names"]

                dense_logits[camera_idx] = camera_logits
                dense_confidence[camera_idx] = camera_conf
                dense_pred_label[camera_idx] = camera_pred
                camera_available[camera_idx] = True
                image_rel_paths.append(image_rel_path)
                image_widths[camera_idx] = metadata["original_width"]
                image_heights[camera_idx] = metadata["original_height"]
                logit_widths[camera_idx] = metadata["logit_width"]
                logit_heights[camera_idx] = metadata["logit_height"]
                processed_cameras += 1

                camera_npz = sample_dir / f"{camera_name}.npz"
                save_npz(
                    camera_npz,
                    dense_logits=camera_logits,
                    dense_confidence=camera_conf,
                    dense_pred_label=camera_pred,
                    class_names=np.asarray(encoded["class_names"]),
                    prompts=np.asarray(encoded["prompts"]),
                    camera_name=np.asarray(camera_name),
                    image_rel_path=np.asarray(image_rel_path),
                    sample_idx=np.asarray(sample_idx, dtype=np.int32),
                    sample_token=np.asarray(sample["token"]),
                )
                summary_cameras.append(
                    {
                        "camera_name": camera_name,
                        "available": True,
                        "image_path": str(image_path),
                        "num_masks": metadata["num_masks"],
                        "logit_height": metadata["logit_height"],
                        "logit_width": metadata["logit_width"],
                        "camera_npz": str(camera_npz),
                    }
                )
                logger.info(
                    "%s | masks=%d | logits=%dx%dx%d",
                    camera_name,
                    metadata["num_masks"],
                    metadata["num_classes"],
                    metadata["logit_height"],
                    metadata["logit_width"],
                )

            if dense_logits is None or prompts is None or output_class_names is None:
                raise RuntimeError(f"No camera image available for sample_idx={sample_idx}")

            save_npz(
                output_npz,
                sample_idx=np.array(sample_idx, dtype=np.int32),
                sample_token=np.array(sample["token"]),
                teacher_backend=np.array(SAM2_SIGLIP),
                teacher_role=np.array(teacher_spec.role),
                teacher_feature_granularity=np.array(teacher_spec.feature_granularity),
                is_baseline_teacher=np.array(teacher_spec.is_baseline),
                model_name=np.array(f"{args.sam_model_id}+{args.siglip_model_name}"),
                sam_model_id=np.array(args.sam_model_id),
                siglip_model_name=np.array(args.siglip_model_name),
                cache_dir=np.array(args.cache_dir or ""),
                local_files_only=np.array(args.local_files_only),
                camera_names=np.asarray(CAMERA_CHANNELS),
                camera_available=camera_available,
                image_rel_paths=np.asarray(image_rel_paths),
                image_widths=image_widths,
                image_heights=image_heights,
                logit_widths=logit_widths,
                logit_heights=logit_heights,
                class_names=np.asarray(output_class_names),
                prompts=np.asarray(prompts),
                dense_logits=dense_logits,
                dense_confidence=dense_confidence,
                dense_pred_label=dense_pred_label,
            )
            summary = {
                "sample_idx": sample_idx,
                "sample_token": sample["token"],
                "teacher_backend": SAM2_SIGLIP,
                "teacher_role": teacher_spec.role,
                "sam_model_id": args.sam_model_id,
                "siglip_model_name": args.siglip_model_name,
                "logit_dtype": args.logit_dtype,
                "prompt_template": args.prompt_template,
                "num_classes": len(output_class_names),
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
            artifacts.append(str(output_npz))
            logger.info("SAM2+SigLIP teacher saved to: %s", output_npz)

        if len(sample_indices) > 1:
            batch_summary_path = output_dir / "batch_sam2_siglip_teacher_summary.json"
            save_json(batch_summary_path, batch_summary)
            artifacts.append(str(batch_summary_path))
    except Exception as exc:
        logger.exception("SAM2+SigLIP extraction failed")
        status = "failed"
        notes = f"{type(exc).__name__}: {exc}"

    conclusion = RunConclusion(
        stage="stage-teacher",
        experiment="extract_sam2_teacher",
        status=status,
        gate="SAM2+SigLIP teacher cache produced for diagnostic samples",
        gate_passed=status == "success" and processed_cameras > 0,
        primary_metric_name="processed_cameras",
        primary_metric_value=float(processed_cameras),
        secondary={},
        runtime_seconds=time.monotonic() - start_time,
        checkpoint=None,
        artifacts=artifacts,
        next_step="project dense teacher logits to LiDAR points" if status == "success" else "fix extraction failure",
        notes=notes,
    )
    conclusion.append_to_recap(Path("docs/EXPERIMENT_RECAP.md"))
    conclusion.print_block()
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
