from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export CAT-Seg per-camera dense maps into the canonical V12 dense teacher npz format. "
            "Run this script from a CAT-Seg checkout/environment, not from the RA-OV3DSeg env."
        )
    )
    parser.add_argument("--catseg_root", default=".", type=str, help="Path to the CAT-Seg repository root.")
    parser.add_argument("--manifest_jsonl", required=True, type=str, help="Manifest from build_external_teacher_manifest.py.")
    parser.add_argument("--output_dir", required=True, type=str, help="Canonical dense teacher output directory.")
    parser.add_argument("--config_file", default="configs/vitb_384.yaml", type=str)
    parser.add_argument("--weights", required=True, type=str, help="CAT-Seg checkpoint, e.g. model_base.pth.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model_name", default="catseg_vitb_384", type=str)
    parser.add_argument("--teacher_backend", default="catseg_dense", type=str)
    parser.add_argument("--logit_height", default=180, type=int, help="Saved dense map height. <=0 keeps model output size.")
    parser.add_argument("--logit_width", default=320, type=int, help="Saved dense map width. <=0 keeps model output size.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--start_record", default=0, type=int)
    parser.add_argument("--max_records", default=None, type=int)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--class_text_mode",
        default="pretty_class_names",
        choices=["class_names", "pretty_class_names"],
        help=(
            "Class strings passed to CAT-Seg. pretty_class_names converts nuScenes names like "
            "vehicle.car to vehicle car while preserving canonical output class_names."
        ),
    )
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Extra Detectron2/CAT-Seg config opts.")
    return parser


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prettify_class_name(name: str) -> str:
    return name.replace(".", " ").replace("_", " ").strip()


def write_class_json(path: Path, class_names: list[str], class_text_mode: str) -> list[str]:
    if class_text_mode == "pretty_class_names":
        class_texts = [prettify_class_name(name) for name in class_names]
    else:
        class_texts = list(class_names)
    with path.open("w", encoding="utf-8") as file:
        json.dump(class_texts, file, ensure_ascii=False, indent=2)
    return class_texts


def import_catseg(catseg_root: Path):
    catseg_root = catseg_root.resolve()
    sys.path.insert(0, str(catseg_root))
    sys.path.insert(0, str(catseg_root / "demo"))

    try:
        import torch
        import torch.nn.functional as F
        from detectron2.config import get_cfg
        from detectron2.data.detection_utils import read_image
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.projects.deeplab import add_deeplab_config

        from cat_seg import add_cat_seg_config
    except ImportError as exc:
        raise ImportError(
            "CAT-Seg export requires the CAT-Seg environment with detectron2, torch, opencv, and CAT-Seg imports. "
            f"catseg_root={catseg_root}"
        ) from exc

    return torch, F, get_cfg, read_image, DefaultPredictor, add_deeplab_config, add_cat_seg_config


def setup_predictor(args: argparse.Namespace, class_json_path: Path, num_classes: int):
    (
        torch,
        F,
        get_cfg,
        read_image,
        DefaultPredictor,
        add_deeplab_config,
        add_cat_seg_config,
    ) = import_catseg(Path(args.catseg_root))

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(str(Path(args.catseg_root) / args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(Path(args.weights).expanduser().resolve())
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = str(class_json_path)
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = str(class_json_path)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(num_classes)
    cfg.freeze()
    return torch, F, read_image, DefaultPredictor(cfg)


def sem_seg_to_logits(torch_module, F_module, sem_seg, out_height: int, out_width: int) -> np.ndarray:
    sem_seg = sem_seg.detach().float().cpu()
    if out_height > 0 and out_width > 0 and tuple(sem_seg.shape[-2:]) != (out_height, out_width):
        sem_seg = F_module.interpolate(
            sem_seg.unsqueeze(0),
            size=(out_height, out_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    # CAT-Seg eval returns sigmoid score maps. Convert probabilities back to logits
    # so RA-OV3DSeg can apply a consistent softmax/temperature downstream.
    if torch_module.isfinite(sem_seg).all() and float(sem_seg.min()) >= 0.0 and float(sem_seg.max()) <= 1.0:
        sem_seg = torch_module.logit(torch_module.clamp(sem_seg, 1e-4, 1.0 - 1e-4))
    return sem_seg.numpy().astype(np.float32)


def save_sample_npz(
    output_path: Path,
    record: dict[str, Any],
    dense_logits: np.ndarray,
    camera_available: np.ndarray,
    image_widths: np.ndarray,
    image_heights: np.ndarray,
    class_texts: list[str],
    args: argparse.Namespace,
) -> None:
    output_dtype = np.float16 if args.dtype == "float16" else np.float32
    dense_logits = dense_logits.astype(output_dtype)
    np.savez_compressed(
        output_path,
        sample_idx=np.array(int(record["sample_idx"]), dtype=np.int32),
        sample_token=np.array(str(record["sample_token"])),
        teacher_backend=np.array(args.teacher_backend),
        teacher_role=np.array("main_dense_teacher_candidate"),
        teacher_feature_granularity=np.array("dense_class_logits"),
        model_name=np.array(args.model_name),
        camera_names=np.asarray(record["camera_names"]),
        camera_available=camera_available.astype(bool),
        image_widths=image_widths.astype(np.int32),
        image_heights=image_heights.astype(np.int32),
        logit_widths=np.full(len(record["camera_names"]), dense_logits.shape[-1], dtype=np.int32),
        logit_heights=np.full(len(record["camera_names"]), dense_logits.shape[-2], dtype=np.int32),
        class_names=np.asarray(record["class_names"]),
        class_texts=np.asarray(class_texts),
        prompts=np.asarray(record.get("prompts", record["class_names"])),
        dense_logits=dense_logits,
    )


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest_jsonl).expanduser().resolve()
    output_dir = ensure_dir(args.output_dir)
    records = load_manifest(manifest_path)
    if args.max_records is not None:
        records = records[args.start_record : args.start_record + args.max_records]
    else:
        records = records[args.start_record :]
    if not records:
        raise ValueError("No manifest records selected.")

    class_names = [str(name) for name in records[0]["class_names"]]
    with tempfile.TemporaryDirectory(prefix="ra_ov3dseg_catseg_") as tmp_dir:
        class_json_path = Path(tmp_dir) / "nuscenes_lidarseg_classes.json"
        class_texts = write_class_json(class_json_path, class_names, args.class_text_mode)
        torch_module, F_module, read_image, predictor = setup_predictor(args, class_json_path, len(class_names))

        for record_idx, record in enumerate(records, start=args.start_record):
            sample_idx = int(record["sample_idx"])
            output_path = output_dir / f"sample_{sample_idx:04d}_dense_teacher_logits.npz"
            if args.skip_existing and output_path.exists():
                print(f"[INFO] skip existing sample_idx={sample_idx} path={output_path}", flush=True)
                continue

            dense_logits = None
            camera_available = np.zeros(len(record["camera_names"]), dtype=bool)
            image_widths = np.zeros(len(record["camera_names"]), dtype=np.int32)
            image_heights = np.zeros(len(record["camera_names"]), dtype=np.int32)

            for camera_idx, camera in enumerate(record["cameras"]):
                image_path = Path(camera["image_path"])
                if not camera.get("image_exists", False) or not image_path.exists():
                    continue
                image = read_image(str(image_path), format="BGR")
                predictions = predictor(image)
                if "sem_seg" not in predictions:
                    raise RuntimeError(f"CAT-Seg output missing sem_seg for image={image_path}")
                camera_logits = sem_seg_to_logits(
                    torch_module,
                    F_module,
                    predictions["sem_seg"],
                    out_height=args.logit_height,
                    out_width=args.logit_width,
                )
                if dense_logits is None:
                    dense_logits = np.zeros(
                        (len(record["camera_names"]),) + camera_logits.shape,
                        dtype=np.float32,
                    )
                dense_logits[camera_idx] = camera_logits
                camera_available[camera_idx] = True
                image_heights[camera_idx] = int(camera.get("image_height", image.shape[0]) or image.shape[0])
                image_widths[camera_idx] = int(camera.get("image_width", image.shape[1]) or image.shape[1])
                print(
                    f"[INFO] sample_idx={sample_idx} camera={camera['camera_name']} logits={camera_logits.shape}",
                    flush=True,
                )

            if dense_logits is None:
                raise RuntimeError(f"No camera logits generated for sample_idx={sample_idx}")
            save_sample_npz(
                output_path=output_path,
                record=record,
                dense_logits=dense_logits,
                camera_available=camera_available,
                image_widths=image_widths,
                image_heights=image_heights,
                class_texts=class_texts,
                args=args,
            )
            print(
                f"[INFO] saved sample_idx={sample_idx} record={record_idx} output={output_path}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
