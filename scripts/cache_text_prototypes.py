from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.models.text_encoder import TextEncoder  # noqa: E402
from ra_ov3dseg.training.labels import NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES  # noqa: E402
from ra_ov3dseg.utils.io import save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.run_conclusion import RunConclusion  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache SigLIP text prototypes for the 16 nuScenes lidarseg classes.")
    parser.add_argument("--output", default="outputs/text_prototypes/nuscenes_siglip_16.npz", type=str)
    parser.add_argument("--model_name", default="google/siglip-base-patch16-224", type=str)
    parser.add_argument("--prompt_template", default="a photo of a {}", type=str)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = output_path.with_suffix(".json")
    artifacts: list[str] = []
    status = "success"
    gate_passed = False
    metric = 0.0
    notes = "SigLIP text prototypes cached"

    try:
        class_names = NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES[1:]
        encoder = TextEncoder(
            model_name=args.model_name,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        result = encoder.encode_texts(
            class_names,
            prompt_template=args.prompt_template,
            normalize=True,
        )
        embeddings = result["text_embeddings"].astype(np.float32)
        save_npz(
            output_path,
            class_names=np.asarray(result["class_names"]),
            prompts=np.asarray(result["prompts"]),
            text_embeddings=embeddings,
            model_name=np.asarray(result["model_name"]),
            prompt_template=np.asarray(args.prompt_template),
        )
        save_json(
            summary_path,
            {
                "class_names": result["class_names"],
                "prompts": result["prompts"],
                "model_name": result["model_name"],
                "prompt_template": args.prompt_template,
                "embedding_shape": list(embeddings.shape),
                "output": str(output_path),
            },
        )
        artifacts = [str(output_path), str(summary_path)]
        metric = float(embeddings.shape[0])
        gate_passed = embeddings.shape[0] == 16 and embeddings.ndim == 2
    except Exception as exc:
        status = "failed"
        notes = f"{type(exc).__name__}: {exc}"

    conclusion = RunConclusion(
        stage="stage-ov-head",
        experiment="cache_text_prototypes",
        status=status,
        gate="16 normalized nuScenes text prototypes cached",
        gate_passed=gate_passed,
        primary_metric_name="num_text_prototypes",
        primary_metric_value=metric,
        secondary={},
        runtime_seconds=0.0,
        checkpoint=None,
        artifacts=artifacts,
        next_step="run Stage 2 OV head smoke fine-tune" if gate_passed else "fix text prototype cache failure",
        notes=notes,
    )
    conclusion.append_to_recap(Path("docs/EXPERIMENT_RECAP.md"))
    conclusion.print_block()
    return 0 if status == "success" and gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
