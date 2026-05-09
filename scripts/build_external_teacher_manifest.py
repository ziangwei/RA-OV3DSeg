from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import CAMERA_CHANNELS, NuScenesDataset  # noqa: E402
from ra_ov3dseg.models.teacher_registry import CATSEG_DENSE, EXTERNAL_DENSE_LOGITS, OPENSEG_DENSE  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, load_text_lines, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a JSONL manifest for running an external dense open-vocabulary "
            "teacher such as CAT-Seg/OpenSeg in a separate environment."
        )
    )
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--version", default="v1.0-trainval", type=str)
    parser.add_argument("--sample_idx", default=None, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--max_samples", default=1, type=int)
    parser.add_argument(
        "--teacher_backend",
        default=CATSEG_DENSE,
        choices=[CATSEG_DENSE, OPENSEG_DENSE, EXTERNAL_DENSE_LOGITS],
        help="External teacher name to record in the manifest.",
    )
    parser.add_argument("--model_name", default="catseg_external", type=str)
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--prompt_template", default="a {} in a driving scene", type=str)
    parser.add_argument(
        "--dense_teacher_dir",
        default="outputs/external_teachers/catseg_dense",
        type=str,
        help="Directory where the external teacher should write canonical sample_XXXX_dense_teacher_logits.npz files.",
    )
    parser.add_argument("--output_dir", default="outputs/external_teacher_manifests", type=str)
    parser.add_argument("--output_name", default="external_teacher_manifest", type=str)
    return parser


def prettify_label_name(label_name: str) -> str:
    return label_name.replace(".", " ").replace("_", " ").strip()


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("build_external_teacher_manifest")
    dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=False)
    sample_indices = dataset.resolve_sample_indices(
        sample_idx=args.sample_idx,
        start_idx=args.start_idx,
        max_samples=args.max_samples,
    )
    class_names = load_text_lines(args.class_names_path)
    prompts = [args.prompt_template.format(prettify_label_name(name)) for name in class_names]
    output_dir = ensure_dir(args.output_dir)
    dense_teacher_dir = Path(args.dense_teacher_dir).expanduser().resolve()
    manifest_path = output_dir / f"{args.output_name}.jsonl"
    summary_path = output_dir / f"{args.output_name}_summary.json"

    records: list[dict[str, Any]] = []
    for sample_idx in sample_indices:
        sample = dataset.get_sample_by_index(sample_idx)
        cameras = []
        for camera_name in CAMERA_CHANNELS:
            image_path = dataset.get_sample_data_path_from_channel(sample, camera_name)
            image_rel_path = dataset.get_sample_data_relpath_from_channel(sample, camera_name)
            record = dataset.get_sample_data_record_from_channel(sample, camera_name)
            width, height = image_size(image_path) if image_path is not None and image_path.exists() else (0, 0)
            cameras.append(
                {
                    "camera_name": camera_name,
                    "sample_data_token": "" if record is None else str(record["token"]),
                    "image_path": "" if image_path is None else str(image_path),
                    "image_rel_path": "" if image_rel_path is None else image_rel_path,
                    "image_exists": bool(image_path is not None and image_path.exists()),
                    "image_width": width,
                    "image_height": height,
                }
            )

        prefix = f"sample_{sample_idx:04d}"
        records.append(
            {
                "sample_idx": sample_idx,
                "sample_token": sample["token"],
                "scene_token": sample["scene_token"],
                "timestamp": int(sample["timestamp"]),
                "teacher_backend": args.teacher_backend,
                "model_name": args.model_name,
                "class_names": class_names,
                "prompts": prompts,
                "camera_names": CAMERA_CHANNELS,
                "cameras": cameras,
                "expected_output_npz": str(dense_teacher_dir / f"{prefix}_dense_teacher_logits.npz"),
                "canonical_output_keys": [
                    "sample_idx",
                    "sample_token",
                    "teacher_backend",
                    "model_name",
                    "camera_names",
                    "camera_available",
                    "image_widths",
                    "image_heights",
                    "class_names",
                    "prompts",
                    "dense_logits",
                ],
                "canonical_dense_logits_layout": "camera,class,height,width",
            }
        )

    write_jsonl(manifest_path, records)
    summary = {
        "version": args.version,
        "dataroot": str(Path(args.dataroot).expanduser().resolve()),
        "teacher_backend": args.teacher_backend,
        "model_name": args.model_name,
        "sample_indices": sample_indices,
        "num_samples": len(sample_indices),
        "num_classes": len(class_names),
        "dense_teacher_dir": str(dense_teacher_dir),
        "manifest_jsonl": str(manifest_path),
    }
    save_json(summary_path, summary)
    logger.info("external teacher manifest saved | samples=%d | path=%s", len(records), manifest_path)
    logger.info("summary saved to: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
