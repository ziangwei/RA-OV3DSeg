from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print a compact experiment summary for pasting back to Codex.")
    parser.add_argument("--experiment_dir", required=True, type=str)
    parser.add_argument("--stage", default="", type=str)
    parser.add_argument("--output_json", default=None, type=str)
    return parser


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compact_float(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    args = build_parser().parse_args()
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    train_summary = load_json_if_exists(experiment_dir / "training" / "train_summary.json")
    eval_summary = load_json_if_exists(experiment_dir / "evaluation3d" / "batch_3d_eval_summary.json")

    eval_metrics = eval_summary.get("aggregate_metrics", {})
    best_eval = train_summary.get("eval_during_training", {})
    best_train_metrics = best_eval.get("best_eval_metrics", {})

    summary = {
        "stage": args.stage,
        "experiment_dir": str(experiment_dir),
        "train_status": train_summary.get("status", "missing"),
        "backbone": train_summary.get("backbone", {}).get("backbone", "missing"),
        "student_output_space": train_summary.get("student_output_space", "missing"),
        "epochs_completed": train_summary.get("epochs_completed"),
        "num_samples": train_summary.get("num_samples"),
        "best_train_eval_miou": best_eval.get("best_eval_miou"),
        "best_train_eval_epoch": best_train_metrics.get("epoch"),
        "final_eval_all_miou": eval_metrics.get("all_miou"),
        "final_eval_base_miou": eval_metrics.get("base_miou"),
        "final_eval_novel_miou": eval_metrics.get("novel_miou"),
        "final_eval_coverage": eval_metrics.get("prediction_coverage"),
        "final_eval_num_points": eval_metrics.get("num_points"),
        "final_eval_num_valid_pred_points": eval_metrics.get("num_valid_pred_points"),
        "best_checkpoint": best_eval.get("best_checkpoint", ""),
    }

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    print("========== RUN_CONCLUSION ==========")
    print(f"stage={summary['stage']}")
    print(f"experiment_dir={summary['experiment_dir']}")
    print(f"train_status={summary['train_status']}")
    print(f"backbone={summary['backbone']}")
    print(f"student_output_space={summary['student_output_space']}")
    print(f"epochs_completed={summary['epochs_completed']}")
    print(f"num_samples={summary['num_samples']}")
    print(f"best_train_eval_miou={compact_float(summary['best_train_eval_miou'])}")
    print(f"best_train_eval_epoch={summary['best_train_eval_epoch']}")
    print(f"final_eval_all_miou={compact_float(summary['final_eval_all_miou'])}")
    print(f"final_eval_base_miou={compact_float(summary['final_eval_base_miou'])}")
    print(f"final_eval_novel_miou={compact_float(summary['final_eval_novel_miou'])}")
    print(f"final_eval_coverage={compact_float(summary['final_eval_coverage'])}")
    print(f"final_eval_num_points={summary['final_eval_num_points']}")
    print(f"final_eval_num_valid_pred_points={summary['final_eval_num_valid_pred_points']}")
    print(f"best_checkpoint={summary['best_checkpoint']}")
    print("====================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
