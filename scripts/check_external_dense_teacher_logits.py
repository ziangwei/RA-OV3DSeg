from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import CAMERA_CHANNELS  # noqa: E402
from ra_ov3dseg.models.external_dense_teacher import load_npz, validate_dense_teacher_npz  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, load_text_lines, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate canonical external dense teacher logits.")
    parser.add_argument("--sample_idx", default=None, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--max_samples", default=1, type=int)
    parser.add_argument("--dense_teacher_npz", default=None, type=str)
    parser.add_argument("--dense_teacher_dir", default="outputs/external_teachers/catseg_dense", type=str)
    parser.add_argument("--projection_dir", default=None, type=str, help="Optional projection dir for point-count sanity.")
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--output_dir", default="outputs/external_teacher_checks", type=str)
    return parser


def build_jobs(args: argparse.Namespace) -> list[tuple[int, Path]]:
    if args.dense_teacher_npz is not None:
        sample_idx = args.sample_idx
        if sample_idx is None:
            stem = Path(args.dense_teacher_npz).stem
            for token in stem.split("_"):
                if token.isdigit():
                    sample_idx = int(token)
                    break
        if sample_idx is None:
            raise ValueError("--sample_idx is required when it cannot be inferred from dense_teacher_npz.")
        return [(sample_idx, Path(args.dense_teacher_npz).expanduser().resolve())]

    if args.sample_idx is not None:
        sample_indices = [args.sample_idx]
    else:
        sample_indices = list(range(args.start_idx, args.start_idx + args.max_samples))
    dense_teacher_dir = Path(args.dense_teacher_dir).expanduser().resolve()
    return [
        (sample_idx, dense_teacher_dir / f"sample_{sample_idx:04d}_dense_teacher_logits.npz")
        for sample_idx in sample_indices
    ]


def projection_shape_summary(projection_dir: str | None, sample_idx: int) -> dict[str, Any]:
    if projection_dir is None:
        return {}
    path = Path(projection_dir).expanduser().resolve() / f"sample_{sample_idx:04d}_projection.npz"
    if not path.exists():
        return {"projection_npz": str(path), "projection_exists": False}
    data = load_npz(path)
    return {
        "projection_npz": str(path),
        "projection_exists": True,
        "point_xyz_shape": list(data["point_xyz"].shape) if "point_xyz" in data else [],
        "uv_shape": list(data["uv"].shape) if "uv" in data else [],
    }


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("check_external_dense_teacher_logits")
    output_dir = ensure_dir(args.output_dir)
    expected_class_names = load_text_lines(args.class_names_path)
    jobs = build_jobs(args)

    results: list[dict[str, Any]] = []
    failed = 0
    for sample_idx, dense_teacher_npz in jobs:
        validation = validate_dense_teacher_npz(
            dense_teacher_npz,
            expected_camera_names=CAMERA_CHANNELS,
            expected_class_names=expected_class_names,
        )
        status = "pass" if validation.valid else "fail"
        if not validation.valid:
            failed += 1
        record = {
            "sample_idx": sample_idx,
            "status": status,
            "dense_teacher_npz": str(dense_teacher_npz),
            "message": validation.message,
            "num_cameras": validation.num_cameras,
            "num_classes": validation.num_classes,
            "logit_height": validation.logit_height,
            "logit_width": validation.logit_width,
            "layout": validation.layout,
            "teacher_backend": validation.teacher_backend,
            "model_name": validation.model_name,
            "projection": projection_shape_summary(args.projection_dir, sample_idx),
        }
        results.append(record)
        logger.info(
            "[%s] sample_idx=%d | classes=%d | logits=%dx%d | layout=%s | %s",
            status.upper(),
            sample_idx,
            validation.num_classes,
            validation.logit_height,
            validation.logit_width,
            validation.layout,
            validation.message,
        )

    summary = {
        "status": "pass" if failed == 0 else "fail",
        "num_checked": len(results),
        "num_failed": failed,
        "results": results,
    }
    summary_path = output_dir / "external_dense_teacher_check_summary.json"
    save_json(summary_path, summary)
    logger.info("external dense teacher check %s | summary=%s", summary["status"].upper(), summary_path)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
