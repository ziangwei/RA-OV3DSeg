from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset  # noqa: E402
from ra_ov3dseg.training.labels import (  # noqa: E402
    NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES,
    NUSCENES_RAW_TO_OFFICIAL_16,
    build_class_split,
    map_official_16_for_ce,
    map_raw_lidarseg_to_official_16,
)
from ra_ov3dseg.training.raw_lidarseg_dataset import RawLidarsegDataset  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-V16 sanity check for nuScenes-lidarseg label space, ignore masks, "
            "point/label counts, and V15 cylinder range coverage."
        )
    )
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes dataroot.")
    parser.add_argument("--version", default="v1.0-trainval", type=str)
    parser.add_argument("--sample_idx", default=None, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--max_samples", default=32, type=int)
    parser.add_argument("--max_points", default=50000, type=int)
    parser.add_argument("--class_names_path", default=str(ROOT / "configs/nuscenes_lidarseg_class_names.txt"), type=str)
    parser.add_argument("--split_config", default=str(ROOT / "configs/all_lidarseg_supervised_split.yaml"), type=str)
    parser.add_argument("--output_dir", default=str(ROOT / "outputs/reports"), type=str)
    parser.add_argument(
        "--cylinder_voxel_size",
        default=(0.125, 0.017453292519943295, 0.25),
        nargs=3,
        type=float,
        metavar=("VR", "VPHI", "VZ"),
    )
    parser.add_argument(
        "--cylinder_point_cloud_range",
        default=(0.0, -math.pi, -5.0, 60.0, math.pi, 3.0),
        nargs=6,
        type=float,
        metavar=("R_MIN", "PHI_MIN", "Z_MIN", "R_MAX", "PHI_MAX", "Z_MAX"),
    )
    parser.add_argument("--strict", action="store_true", help="Return non-zero if hard checks fail.")
    return parser


def hist_entries(counts: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    entries = []
    total = int(np.sum(counts))
    for class_id, count in enumerate(counts.astype(np.int64).tolist()):
        if count <= 0:
            continue
        entries.append(
            {
                "id": int(class_id),
                "name": names[class_id],
                "count": int(count),
                "ratio": float(count / max(total, 1)),
            }
        )
    return entries


def cylinder_range_mask(
    point_xyz: np.ndarray,
    voxel_size: tuple[float, float, float],
    point_cloud_range: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    rho = np.linalg.norm(point_xyz[:, :2], axis=1)
    phi = np.arctan2(point_xyz[:, 1], point_xyz[:, 0])
    z = point_xyz[:, 2]
    cyl = np.stack([rho, phi, z], axis=1)
    mins = np.asarray(point_cloud_range[:3], dtype=np.float32)
    maxs = np.asarray(point_cloud_range[3:], dtype=np.float32)
    vsize = np.asarray(voxel_size, dtype=np.float32)
    grid_size = np.floor((maxs - mins) / vsize).astype(np.int64)
    coords = np.floor((cyl - mins) / vsize).astype(np.int64)
    valid = np.all((cyl >= mins) & (cyl < maxs), axis=1)
    valid &= np.all((coords >= 0) & (coords < grid_size), axis=1)
    return valid


def run_loss_mask_test() -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as torch_f

        from ra_ov3dseg.training.losses import supervised_ce_loss
    except Exception as exc:
        return {"status": "skipped", "reason": f"torch import failed: {exc}"}

    generator = torch.Generator().manual_seed(0)
    logits = torch.randn((9, 4), generator=generator)
    labels = torch.tensor([0, 1, -100, 2, -100, 3, 1, -100, 0], dtype=torch.long)
    valid = labels != -100
    loss_project = supervised_ce_loss(logits, labels, ignore_index=-100)
    loss_manual = torch_f.cross_entropy(logits[valid], labels[valid])
    diff = float(torch.abs(loss_project - loss_manual).detach().cpu().item())
    return {
        "status": "pass" if diff < 1e-7 else "fail",
        "project_loss": float(loss_project.detach().cpu().item()),
        "manual_valid_only_loss": float(loss_manual.detach().cpu().item()),
        "absolute_diff": diff,
    }


def make_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Pre-V16 Sanity Report",
        "",
        f"- status: {summary['status']}",
        f"- version: {summary['version']}",
        f"- dataroot: `{summary['dataroot']}`",
        f"- sample_indices: {summary['sample_indices']}",
        "",
        "## Checks",
        "",
    ]
    for check in summary["checks"]:
        lines.append(f"- {check['status'].upper()} `{check['name']}`: {check['message']}")

    totals = summary["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- total_points: {totals['total_points']}",
            f"- current_raw_supervised_points: {totals['current_raw_supervised_points']}",
            f"- official_16_supervised_points: {totals['official_16_supervised_points']}",
            f"- official_void_points: {totals['official_void_points']}",
            f"- cylinder_range_inside_ratio: {totals['cylinder_range_inside_ratio']:.6f}",
            f"- max_points_subsample_would_drop_ratio: {totals['max_points_subsample_would_drop_ratio']:.6f}",
            f"- max_observed_rho: {totals['max_observed_rho']:.3f}",
            f"- observed_z_range: [{totals['min_observed_z']}, {totals['max_observed_z']}]",
            "",
            "## Current Split Vs Official 16",
            "",
        ]
    )
    for item in summary["current_split_supervises_official_void"]:
        lines.append(
            f"- raw_id={item['raw_id']} `{item['raw_name']}` count={item['count']} "
            f"is currently supervised but official lidarseg maps it to void/ignore"
        )
    if not summary["current_split_supervises_official_void"]:
        lines.append("- No current supervised raw class maps to official void in the checked samples.")

    lines.extend(["", "## Official Class Histogram", ""])
    for entry in summary["official_histogram"]:
        lines.append(f"- {entry['id']:02d} `{entry['name']}`: {entry['count']} ({entry['ratio']:.6f})")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("pre_v16_sanity_check")
    output_dir = ensure_dir(args.output_dir)

    dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=False)
    sample_indices = dataset.resolve_sample_indices(
        sample_idx=args.sample_idx,
        start_idx=args.start_idx,
        max_samples=args.max_samples,
    )
    class_split = build_class_split(args.class_names_path, args.split_config)

    raw_counts = np.zeros(len(class_split.class_names), dtype=np.int64)
    official_counts = np.zeros(len(NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES), dtype=np.int64)
    current_supervised_counts = np.zeros(len(class_split.class_names), dtype=np.int64)
    sample_summaries = []
    checks = []
    total_points = 0
    total_inside_cylinder = 0
    total_subsample_drop = 0
    max_rho = 0.0
    min_z = float("inf")
    max_z = float("-inf")
    mismatches = []

    logger.info("dataroot=%s", Path(args.dataroot).expanduser().resolve())
    logger.info("version=%s | samples=%s", args.version, sample_indices)

    first_direct_points = None
    first_raw_dataset_points = None

    for local_idx, sample_idx in enumerate(sample_indices):
        sample = dataset.get_sample_by_index(sample_idx)
        point_xyzi = dataset.load_lidar_points_xyzi(sample)
        point_xyz = point_xyzi[:, :3]
        raw_labels = dataset.load_lidarseg_labels(sample)
        if raw_labels is None:
            mismatches.append({"sample_idx": int(sample_idx), "reason": "missing_lidarseg"})
            continue
        if raw_labels.shape[0] != point_xyz.shape[0]:
            mismatches.append(
                {
                    "sample_idx": int(sample_idx),
                    "reason": "point_label_count_mismatch",
                    "points": int(point_xyz.shape[0]),
                    "labels": int(raw_labels.shape[0]),
                }
            )
            continue

        official_labels = map_raw_lidarseg_to_official_16(raw_labels)
        official_ce_labels = map_official_16_for_ce(official_labels)
        current_supervised_mask = np.isin(raw_labels, class_split.base_label_ids)
        official_supervised_mask = official_ce_labels != -100
        cylinder_valid = cylinder_range_mask(
            point_xyz,
            voxel_size=tuple(args.cylinder_voxel_size),
            point_cloud_range=tuple(args.cylinder_point_cloud_range),
        )
        rho = np.linalg.norm(point_xyz[:, :2], axis=1)
        max_rho = max(max_rho, float(np.max(rho)) if rho.size else 0.0)
        min_z = min(min_z, float(np.min(point_xyz[:, 2])) if point_xyz.shape[0] else min_z)
        max_z = max(max_z, float(np.max(point_xyz[:, 2])) if point_xyz.shape[0] else max_z)

        raw_counts += np.bincount(raw_labels.astype(np.int64), minlength=len(class_split.class_names))[
            : len(class_split.class_names)
        ]
        official_counts += np.bincount(
            official_labels.astype(np.int64), minlength=len(NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES)
        )[: len(NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES)]
        current_supervised_counts += np.bincount(
            raw_labels[current_supervised_mask].astype(np.int64), minlength=len(class_split.class_names)
        )[: len(class_split.class_names)]

        total_points += int(point_xyz.shape[0])
        total_inside_cylinder += int(np.sum(cylinder_valid))
        if args.max_points > 0:
            total_subsample_drop += int(max(0, point_xyz.shape[0] - args.max_points))

        sample_summaries.append(
            {
                "sample_idx": int(sample_idx),
                "sample_token": sample["token"],
                "num_points": int(point_xyz.shape[0]),
                "current_raw_supervised_points": int(np.sum(current_supervised_mask)),
                "official_16_supervised_points": int(np.sum(official_supervised_mask)),
                "official_void_points": int(np.sum(official_labels == 0)),
                "cylinder_range_inside_points": int(np.sum(cylinder_valid)),
                "cylinder_range_inside_ratio": float(np.mean(cylinder_valid)),
                "max_points_subsample_would_drop": int(max(0, point_xyz.shape[0] - args.max_points))
                if args.max_points > 0
                else 0,
            }
        )

        if local_idx == 0:
            first_direct_points = point_xyz.copy()

    if first_direct_points is not None:
        raw_dataset = RawLidarsegDataset(
            nuscenes_dataset=dataset,
            sample_indices=[sample_indices[0]],
            class_split=class_split,
            max_points=None,
            seed=0,
            augment=False,
            feature_dim=1,
        )
        first_raw_dataset_points = raw_dataset[0]["point_xyz"]

    if mismatches:
        checks.append(
            {
                "name": "point_label_count",
                "status": "fail",
                "message": f"{len(mismatches)} sample(s) missing labels or mismatched point/label counts",
            }
        )
    else:
        checks.append(
            {
                "name": "point_label_count",
                "status": "pass",
                "message": "all checked samples have matching LiDAR point and lidarseg label counts",
            }
        )

    if first_direct_points is not None and first_raw_dataset_points is not None:
        same_shape = first_direct_points.shape == first_raw_dataset_points.shape
        max_abs_diff = (
            float(np.max(np.abs(first_direct_points - first_raw_dataset_points))) if same_shape and first_direct_points.size else 0.0
        )
        checks.append(
            {
                "name": "raw_dataset_coordinate_consistency",
                "status": "pass" if same_shape and max_abs_diff == 0.0 else "fail",
                "message": f"same_shape={same_shape}, max_abs_diff={max_abs_diff}",
            }
        )

    loss_mask = run_loss_mask_test()
    checks.append(
        {
            "name": "loss_ignore_mask",
            "status": loss_mask["status"],
            "message": str(loss_mask),
        }
    )

    inside_ratio = float(total_inside_cylinder / max(total_points, 1))
    checks.append(
        {
            "name": "v15_cylinder_range_coverage",
            "status": "pass" if inside_ratio >= 0.98 else "warn",
            "message": f"inside_ratio={inside_ratio:.6f}; below 1.0 means V15 cannot predict all points without fill/range fix",
        }
    )

    current_supervises_void = []
    for raw_id, count in enumerate(current_supervised_counts.tolist()):
        if count <= 0:
            continue
        official_id = int(NUSCENES_RAW_TO_OFFICIAL_16[raw_id])
        if official_id == 0:
            current_supervises_void.append(
                {
                    "raw_id": int(raw_id),
                    "raw_name": class_split.class_names[raw_id],
                    "count": int(count),
                }
            )
    checks.append(
        {
            "name": "current_split_vs_official_16",
            "status": "warn" if current_supervises_void else "pass",
            "message": (
                f"{len(current_supervises_void)} current supervised raw class(es) map to official void; "
                "V16 should train/evaluate the official 16-class space"
            ),
        }
    )

    hard_fail = any(check["status"] == "fail" for check in checks)
    status = "fail" if hard_fail else ("warn" if any(check["status"] == "warn" for check in checks) else "pass")
    summary = {
        "status": status,
        "dataroot": str(Path(args.dataroot).expanduser().resolve()),
        "version": args.version,
        "sample_indices": [int(idx) for idx in sample_indices],
        "class_names_path": str(Path(args.class_names_path).expanduser().resolve()),
        "split_config": str(Path(args.split_config).expanduser().resolve()),
        "checks": checks,
        "loss_mask_test": loss_mask,
        "mismatches": mismatches,
        "aggregate": {
            "total_points": int(total_points),
            "current_raw_supervised_points": int(np.sum(current_supervised_counts)),
            "official_16_supervised_points": int(np.sum(official_counts[1:])),
            "official_void_points": int(official_counts[0]),
            "cylinder_range_inside_points": int(total_inside_cylinder),
            "cylinder_range_inside_ratio": inside_ratio,
            "max_points": int(args.max_points),
            "max_points_subsample_would_drop": int(total_subsample_drop),
            "max_points_subsample_would_drop_ratio": float(total_subsample_drop / max(total_points, 1)),
            "max_observed_rho": float(max_rho),
            "min_observed_z": None if not np.isfinite(min_z) else float(min_z),
            "max_observed_z": None if not np.isfinite(max_z) else float(max_z),
        },
        "raw_histogram": hist_entries(raw_counts, class_split.class_names),
        "official_histogram": hist_entries(official_counts, NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES),
        "current_split_supervises_official_void": current_supervises_void,
        "samples": sample_summaries,
    }

    summary_path = save_json(output_dir / "pre_v16_sanity_summary.json", summary)
    report_path = output_dir / "pre_v16_sanity.md"
    report_path.write_text(make_markdown_report(summary), encoding="utf-8")

    logger.info("pre-V16 sanity %s | summary=%s | report=%s", status.upper(), summary_path, report_path)
    for check in checks:
        logger.info("[%s] %s | %s", check["status"].upper(), check["name"], check["message"])

    print("========== RUN_CONCLUSION ==========")
    print(f"stage=pre_v16_sanity")
    print(f"status={status}")
    print(f"total_points={summary['aggregate']['total_points']}")
    print(f"official_16_supervised_points={summary['aggregate']['official_16_supervised_points']}")
    print(f"official_void_points={summary['aggregate']['official_void_points']}")
    print(f"cylinder_range_inside_ratio={summary['aggregate']['cylinder_range_inside_ratio']:.6f}")
    print(f"max_observed_rho={summary['aggregate']['max_observed_rho']:.3f}")
    print(f"min_observed_z={summary['aggregate']['min_observed_z']}")
    print(f"max_observed_z={summary['aggregate']['max_observed_z']}")
    print(f"current_split_supervises_official_void={len(current_supervises_void)}")
    print(f"summary_json={summary_path}")
    print(f"report_md={report_path}")
    print("checks=" + "; ".join(f"{check['name']}:{check['status']}" for check in checks))
    print("====================================")

    if args.strict and hard_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
