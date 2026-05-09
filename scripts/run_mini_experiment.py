from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small nuScenes-mini train/eval experiment protocol.")
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes dataroot.")
    parser.add_argument("--version", default="v1.0-mini", type=str)
    parser.add_argument("--outputs_dir", default="outputs", type=str, help="Shared precompute output root.")
    parser.add_argument("--experiment_dir", default="outputs/experiments/mini_v9", type=str)
    parser.add_argument("--checkpoint", default=None, type=str, help="Optional existing checkpoint for skip_train/predict.")
    parser.add_argument("--train_start_idx", default=0, type=int)
    parser.add_argument("--train_max_samples", default=32, type=int)
    parser.add_argument("--eval_start_idx", default=32, type=int)
    parser.add_argument("--eval_max_samples", default=32, type=int)
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch16", type=str)
    parser.add_argument("--dense_model_name", default="CIDAS/clipseg-rd64-refined", type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--precompute_device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--train_device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--teacher_mode", default="hybrid", choices=["feature_distill", "dense_logit_distill", "hybrid"])
    parser.add_argument(
        "--student_output_space",
        default="all_lidarseg",
        choices=["auto", "base", "all_lidarseg"],
        help="Use all_lidarseg for dense/hybrid experiments unless intentionally running a base-only ablation.",
    )
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--max_points", default=50000, type=int)
    parser.add_argument("--sparse_base_channels", default=32, type=int)
    parser.add_argument("--voxel_size", default=(0.2, 0.2, 0.2), nargs=3, type=float)
    parser.add_argument(
        "--point_cloud_range",
        default=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        nargs=6,
        type=float,
    )
    parser.add_argument("--ce_weight", default=1.0, type=float)
    parser.add_argument("--distill_weight", default=1.0, type=float)
    parser.add_argument("--dense_logit_weight", default=1.0, type=float)
    parser.add_argument("--dense_temperature", default=1.0, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--prompt_batch_size", default=8, type=int)
    parser.add_argument("--feature_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--dense_logit_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--nproc_per_node", default=1, type=int, help="Use torchrun DDP for training when >1.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", help="Pass skip_existing to precompute/predict steps.")
    parser.add_argument("--skip_precompute", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_predict", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print and save commands without executing them.")
    return parser


def add_optional(command: list[str], flag: str, value: str | None) -> None:
    if value is not None and value != "":
        command.extend([flag, value])


def add_bool(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def range_args(start_idx: int, max_samples: int) -> list[str]:
    return ["--start_idx", str(start_idx), "--max_samples", str(max_samples)]


def command_to_text(command: list[str]) -> str:
    return shlex.join([str(item) for item in command])


def run_command(
    command: list[str],
    step_name: str,
    logger,
    command_log: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    started = time.time()
    command_text = command_to_text(command)
    logger.info("========== STEP START: %s ==========", step_name)
    logger.info("[%s] %s", "DRY-RUN" if dry_run else "RUN", command_text)
    entry: dict[str, Any] = {
        "step": step_name,
        "command": command_text,
        "status": "dry_run" if dry_run else "running",
        "started_at_unix": started,
    }
    command_log.append(entry)
    if dry_run:
        entry["elapsed_sec"] = 0.0
        logger.info("========== STEP DRY-RUN: %s ==========", step_name)
        return

    result = subprocess.run(command, cwd=str(ROOT), check=False)
    entry["returncode"] = int(result.returncode)
    entry["elapsed_sec"] = float(time.time() - started)
    if result.returncode != 0:
        entry["status"] = "failed"
        logger.error(
            "========== STEP FAIL: %s | elapsed=%.1fs | returncode=%d ==========",
            step_name,
            entry["elapsed_sec"],
            result.returncode,
        )
        raise RuntimeError(f"step failed: {step_name} | returncode={result.returncode}")
    entry["status"] = "done"
    logger.info("========== STEP DONE: %s | elapsed=%.1fs ==========", step_name, entry["elapsed_sec"])


def python_script(script_name: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script_name)]


def build_dirs(outputs_dir: Path, experiment_dir: Path) -> dict[str, Path]:
    return {
        "outputs": outputs_dir,
        "experiment": experiment_dir,
        "projections": outputs_dir / "projections",
        "features2d": outputs_dir / "features2d",
        "point_features": outputs_dir / "point_features",
        "zero_shot": outputs_dir / "zero_shot",
        "reliability": outputs_dir / "reliability",
        "dense_teacher_logits": outputs_dir / "dense_teacher_logits",
        "dense_point_logits": outputs_dir / "dense_point_logits",
        "training": experiment_dir / "training",
        "predictions": experiment_dir / "predictions3d",
        "evaluation": experiment_dir / "evaluation3d",
        "verification": experiment_dir / "verification",
    }


def precompute_ranges(args: argparse.Namespace) -> list[tuple[int, int]]:
    ranges = [
        (int(args.train_start_idx), int(args.train_max_samples)),
        (int(args.eval_start_idx), int(args.eval_max_samples)),
    ]
    deduped: list[tuple[int, int]] = []
    for item in ranges:
        if item[1] <= 0:
            raise ValueError("train/eval max_samples must be positive.")
        if item not in deduped:
            deduped.append(item)
    return deduped


def build_execution_plan(args: argparse.Namespace) -> list[str]:
    plan: list[str] = []
    if not args.skip_precompute:
        for start_idx, max_samples in precompute_ranges(args):
            plan.extend(
                [
                    f"project_lidar_to_cameras start={start_idx} max={max_samples}",
                    f"extract_2d_features start={start_idx} max={max_samples}",
                    f"assign_2d_features_to_points start={start_idx} max={max_samples}",
                ]
            )
        plan.extend(
            [
                f"zero_shot_eval train_start={args.train_start_idx} train_max={args.train_max_samples}",
                f"compute_reliability train_start={args.train_start_idx} train_max={args.train_max_samples}",
            ]
        )
        if args.teacher_mode in {"dense_logit_distill", "hybrid"}:
            plan.extend(
                [
                    f"extract_dense_teacher_logits train_start={args.train_start_idx} train_max={args.train_max_samples}",
                    f"assign_dense_logits_to_points train_start={args.train_start_idx} train_max={args.train_max_samples}",
                ]
            )
    if not args.skip_train:
        plan.append(f"train_3d_segmentor epochs={args.epochs} train_max={args.train_max_samples}")
    if not args.skip_predict:
        plan.append(f"predict_3d_segmentor eval_start={args.eval_start_idx} eval_max={args.eval_max_samples}")
    if not args.skip_eval:
        plan.append(f"eval_lidarseg eval_start={args.eval_start_idx} eval_max={args.eval_max_samples}")
    return plan


def collect_metrics(experiment_dir: Path, eval_start_idx: int, eval_max_samples: int) -> dict[str, Any]:
    evaluation_dir = experiment_dir / "evaluation3d"
    batch_summary = evaluation_dir / "batch_3d_eval_summary.json"
    if batch_summary.exists():
        data = load_json(batch_summary)
        return {
            "source": str(batch_summary),
            "aggregate_metrics": data.get("aggregate_metrics", {}),
        }

    sample_summary = evaluation_dir / f"sample_{eval_start_idx:04d}_3d_eval_summary.json"
    if eval_max_samples == 1 and sample_summary.exists():
        data = load_json(sample_summary)
        return {
            "source": str(sample_summary),
            "aggregate_metrics": data.get("metrics", {}),
        }
    return {"source": "", "aggregate_metrics": {}}


def collect_training_summary(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "training" / "train_summary.json"
    if not path.exists():
        return {}
    data = load_json(path)
    epoch_logs = data.get("epoch_logs", [])
    return {
        "source": str(path),
        "status": data.get("status"),
        "teacher_mode": data.get("teacher_mode"),
        "student_output_space": data.get("student_output_space"),
        "num_samples": data.get("num_samples"),
        "epochs_completed": data.get("epochs_completed"),
        "latest_checkpoint": data.get("latest_checkpoint"),
        "final_epoch": epoch_logs[-1] if epoch_logs else {},
    }


def save_experiment_summary(
    path: Path,
    args: argparse.Namespace,
    dirs: dict[str, Path],
    commands: list[dict[str, Any]],
    status: str,
    error: str | None = None,
) -> None:
    metrics = collect_metrics(dirs["experiment"], args.eval_start_idx, args.eval_max_samples)
    training = collect_training_summary(dirs["experiment"])
    summary: dict[str, Any] = {
        "status": status,
        "error": error or "",
        "version": args.version,
        "dataroot": str(Path(args.dataroot).expanduser().resolve()),
        "train_range": {"start_idx": args.train_start_idx, "max_samples": args.train_max_samples},
        "eval_range": {"start_idx": args.eval_start_idx, "max_samples": args.eval_max_samples},
        "teacher_mode": args.teacher_mode,
        "student_output_space": args.student_output_space,
        "experiment_dir": str(dirs["experiment"].resolve()),
        "artifact_dirs": {key: str(value.resolve()) for key, value in dirs.items()},
        "training": training,
        "evaluation": metrics,
        "commands": commands,
    }
    save_json(path, summary)


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("run_mini_experiment")
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    experiment_dir = ensure_dir(Path(args.experiment_dir).expanduser().resolve())
    dirs = build_dirs(outputs_dir=outputs_dir, experiment_dir=experiment_dir)
    for path in dirs.values():
        ensure_dir(path)
    summary_path = experiment_dir / "summary.json"
    commands: list[dict[str, Any]] = []

    if args.teacher_mode in {"dense_logit_distill", "hybrid"} and args.student_output_space == "base":
        raise ValueError("Dense/hybrid V9 should use --student_output_space all_lidarseg or auto, not base.")

    execution_plan = build_execution_plan(args)
    logger.info("========== EXPERIMENT PLAN ==========")
    logger.info("version=%s | teacher_mode=%s | student_output_space=%s", args.version, args.teacher_mode, args.student_output_space)
    logger.info(
        "train_range=start:%d max:%d | eval_range=start:%d max:%d",
        args.train_start_idx,
        args.train_max_samples,
        args.eval_start_idx,
        args.eval_max_samples,
    )
    for step_idx, step_text in enumerate(execution_plan, start=1):
        logger.info("plan_step=%02d/%02d | %s", step_idx, len(execution_plan), step_text)

    try:
        if not args.skip_precompute:
            for start_idx, max_samples in precompute_ranges(args):
                common_range = range_args(start_idx, max_samples)

                command = python_script("project_lidar_to_cameras.py")
                command.extend(["--dataroot", args.dataroot, "--version", args.version, *common_range])
                command.extend(["--output_dir", str(dirs["projections"])])
                add_bool(command, "--skip_existing", args.skip_existing)
                run_command(command, f"project_{start_idx}_{max_samples}", logger, commands, args.dry_run)

                command = python_script("extract_2d_features.py")
                command.extend(["--dataroot", args.dataroot, "--version", args.version, *common_range])
                command.extend(["--teacher_backend", "clip_patch_baseline", "--model_name", args.clip_model_name])
                add_optional(command, "--cache_dir", args.cache_dir)
                command.extend(["--device", args.precompute_device, "--feature_dtype", args.feature_dtype])
                command.extend(["--output_dir", str(dirs["features2d"])])
                add_bool(command, "--local_files_only", args.local_files_only)
                add_bool(command, "--skip_existing", args.skip_existing)
                run_command(command, f"extract_2d_{start_idx}_{max_samples}", logger, commands, args.dry_run)

                command = python_script("assign_2d_features_to_points.py")
                command.extend(common_range)
                command.extend(["--projection_dir", str(dirs["projections"])])
                command.extend(["--image_feature_dir", str(dirs["features2d"])])
                command.extend(["--output_dir", str(dirs["point_features"])])
                add_bool(command, "--skip_existing", args.skip_existing)
                run_command(command, f"assign_2d_{start_idx}_{max_samples}", logger, commands, args.dry_run)

            train_range = range_args(args.train_start_idx, args.train_max_samples)
            command = python_script("zero_shot_eval.py")
            command.extend(train_range)
            command.extend(["--point_feature_dir", str(dirs["point_features"])])
            add_optional(command, "--cache_dir", args.cache_dir)
            command.extend(["--device", args.precompute_device])
            command.extend(["--output_dir", str(dirs["zero_shot"])])
            add_bool(command, "--local_files_only", args.local_files_only)
            add_bool(command, "--skip_existing", args.skip_existing)
            run_command(command, "zero_shot_train_range", logger, commands, args.dry_run)

            command = python_script("compute_reliability.py")
            command.extend(train_range)
            command.extend(["--projection_dir", str(dirs["projections"])])
            command.extend(["--zero_shot_dir", str(dirs["zero_shot"])])
            command.extend(["--output_dir", str(dirs["reliability"])])
            add_bool(command, "--skip_existing", args.skip_existing)
            run_command(command, "reliability_train_range", logger, commands, args.dry_run)

            if args.teacher_mode in {"dense_logit_distill", "hybrid"}:
                command = python_script("extract_dense_teacher_logits.py")
                command.extend(["--dataroot", args.dataroot, "--version", args.version, *train_range])
                command.extend(["--teacher_backend", "clipseg_dense", "--model_name", args.dense_model_name])
                add_optional(command, "--cache_dir", args.cache_dir)
                command.extend(["--device", args.precompute_device, "--prompt_batch_size", str(args.prompt_batch_size)])
                command.extend(["--logit_dtype", args.dense_logit_dtype])
                command.extend(["--output_dir", str(dirs["dense_teacher_logits"])])
                add_bool(command, "--local_files_only", args.local_files_only)
                add_bool(command, "--skip_existing", args.skip_existing)
                run_command(command, "dense_teacher_train_range", logger, commands, args.dry_run)

                command = python_script("assign_dense_logits_to_points.py")
                command.extend(train_range)
                command.extend(["--projection_dir", str(dirs["projections"])])
                command.extend(["--dense_teacher_dir", str(dirs["dense_teacher_logits"])])
                command.extend(["--output_dir", str(dirs["dense_point_logits"])])
                add_bool(command, "--skip_existing", args.skip_existing)
                run_command(command, "dense_point_train_range", logger, commands, args.dry_run)

        checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else dirs["training"] / "sparse_unet_spconv_latest.pt"
        if not args.skip_train:
            train_core = python_script("train_3d_segmentor.py")
            train_core.extend(["--dataroot", args.dataroot, "--version", args.version])
            train_core.extend(range_args(args.train_start_idx, args.train_max_samples))
            train_core.extend(["--backbone", "sparse_unet_spconv"])
            train_core.extend(["--teacher_mode", args.teacher_mode, "--student_output_space", args.student_output_space])
            train_core.extend(["--point_feature_dir", str(dirs["point_features"])])
            train_core.extend(["--reliability_dir", str(dirs["reliability"])])
            train_core.extend(["--dense_point_dir", str(dirs["dense_point_logits"])])
            train_core.extend(["--device", args.train_device])
            train_core.extend(["--epochs", str(args.epochs), "--batch_size", str(args.batch_size)])
            train_core.extend(["--num_workers", str(args.num_workers), "--max_points", str(args.max_points)])
            train_core.extend(["--voxel_size", *[str(value) for value in args.voxel_size]])
            train_core.extend(["--point_cloud_range", *[str(value) for value in args.point_cloud_range]])
            train_core.extend(["--sparse_base_channels", str(args.sparse_base_channels)])
            train_core.extend(["--lr", str(args.lr), "--weight_decay", str(args.weight_decay)])
            train_core.extend(["--ce_weight", str(args.ce_weight), "--distill_weight", str(args.distill_weight)])
            train_core.extend(["--dense_logit_weight", str(args.dense_logit_weight)])
            train_core.extend(["--dense_temperature", str(args.dense_temperature)])
            train_core.extend(["--output_dir", str(dirs["training"])])
            add_bool(train_core, "--amp", args.amp)
            if args.nproc_per_node > 1:
                command = ["torchrun", "--standalone", f"--nproc_per_node={args.nproc_per_node}", *train_core[1:]]
            else:
                command = train_core
            run_command(command, "train_sparse_student", logger, commands, args.dry_run)

        if not args.skip_predict:
            command = python_script("predict_3d_segmentor.py")
            command.extend(["--checkpoint", str(checkpoint)])
            command.extend(range_args(args.eval_start_idx, args.eval_max_samples))
            command.extend(["--point_feature_dir", str(dirs["point_features"])])
            command.extend(["--device", args.train_device, "--output_dir", str(dirs["predictions"])])
            add_bool(command, "--skip_existing", args.skip_existing)
            run_command(command, "predict_eval_range", logger, commands, args.dry_run)

        if not args.skip_eval:
            command = python_script("eval_lidarseg.py")
            command.extend(["--dataroot", args.dataroot, "--version", args.version])
            command.extend(range_args(args.eval_start_idx, args.eval_max_samples))
            command.extend(["--prediction_dir", str(dirs["predictions"])])
            command.extend(["--output_dir", str(dirs["evaluation"])])
            add_bool(command, "--skip_existing", args.skip_existing)
            run_command(command, "eval_lidarseg_range", logger, commands, args.dry_run)

        status = "dry_run" if args.dry_run else "pass"
        save_experiment_summary(summary_path, args, dirs, commands, status=status)
        logger.info("mini experiment %s | summary=%s", status.upper(), summary_path)
        return 0
    except Exception as exc:
        save_experiment_summary(summary_path, args, dirs, commands, status="fail", error=str(exc))
        logger.error("mini experiment FAIL | summary=%s | error=%s", summary_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
