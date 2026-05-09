from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.evaluation.openvocab_eval import zero_shot_logits  # noqa: E402
from ra_ov3dseg.models.segmentor_factory import build_segmentor  # noqa: E402
from ra_ov3dseg.models.text_encoder import TextEncoder  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, is_valid_npz, load_text_lines, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402
from ra_ov3dseg.visualization.visualize_points import (  # noqa: E402
    save_bev_prediction_plot,
    save_point_cloud_ply,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open-vocabulary 3D point prediction: model point embeddings are compared "
            "against arbitrary text embeddings by cosine similarity."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to train_3d_segmentor checkpoint .pt.")
    parser.add_argument("--sample_idx", default=None, type=int, help="Single sample index.")
    parser.add_argument("--start_idx", default=0, type=int, help="Batch start sample index.")
    parser.add_argument("--max_samples", default=1, type=int, help="Number of samples in batch mode.")
    parser.add_argument("--point_feature_npz", default=None, type=str, help="Explicit point feature npz for one sample.")
    parser.add_argument("--point_feature_dir", default="outputs/point_features", type=str)
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--class_names_csv", default=None, type=str, help="Comma-separated query class names.")
    parser.add_argument(
        "--lidarseg_class_names_path",
        default="configs/nuscenes_lidarseg_class_names.txt",
        type=str,
        help="Known lidarseg class names used to map query names back to label ids when possible.",
    )
    parser.add_argument("--text_model_name", default=None, type=str, help="CLIP/SigLIP text encoder name.")
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--prompt_template", default="a {} in a driving scene", type=str)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_dir", default="outputs/open_vocab_predictions3d", type=str)
    parser.add_argument("--save_point_embeddings", action="store_true", help="Save 3D point embeddings for debugging.")
    parser.add_argument("--save_similarities", action="store_true", help="Save full point x text similarity matrix.")
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("predict_3d_open_vocab.py requires PyTorch.") from exc
    return torch


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def torch_load_checkpoint(torch_module, checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(checkpoint_path, map_location="cpu")


def resolve_device(torch_module, requested: str):
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        return torch_module.device("cuda")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def scalar_to_str(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value.tolist())
    return str(value)


def load_class_names(args: argparse.Namespace) -> list[str]:
    if args.class_names_csv is not None:
        return [item.strip() for item in args.class_names_csv.split(",") if item.strip()]
    return load_text_lines(args.class_names_path)


def build_point_feature_paths(args: argparse.Namespace) -> list[Path]:
    if args.point_feature_npz is not None:
        return [Path(args.point_feature_npz).expanduser().resolve()]
    if args.sample_idx is not None:
        sample_indices = [args.sample_idx]
    else:
        sample_indices = list(range(args.start_idx, args.start_idx + args.max_samples))
    point_feature_dir = Path(args.point_feature_dir).expanduser().resolve()
    return [point_feature_dir / f"sample_{sample_idx:04d}_point_features.npz" for sample_idx in sample_indices]


def build_model_from_checkpoint(torch_module, checkpoint: dict[str, Any], device):
    state_dict = checkpoint["model_state_dict"]
    ckpt_args = checkpoint.get("args", {})
    classifier_weight = state_dict["classifier.weight"]
    num_output_classes = int(classifier_weight.shape[0])
    feature_dim = int(classifier_weight.shape[1])
    backbone = ckpt_args.get("backbone", checkpoint.get("backbone", {}).get("backbone", "sparse_unet_spconv"))

    model = build_segmentor(
        backbone=backbone,
        input_dim=3,
        hidden_dim=int(ckpt_args.get("hidden_dim", 128)),
        feature_dim=feature_dim,
        num_classes=num_output_classes,
        voxel_size=tuple(ckpt_args.get("voxel_size", (0.2, 0.2, 0.2))),
        point_cloud_range=tuple(ckpt_args.get("point_cloud_range", (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0))),
        sparse_base_channels=int(ckpt_args.get("sparse_base_channels", 32)),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, backbone, feature_dim, num_output_classes


def infer_text_model_name(args: argparse.Namespace, point_feature_paths: list[Path]) -> str:
    if args.text_model_name:
        return args.text_model_name
    for path in point_feature_paths:
        if path.exists():
            data = load_npz(path)
            if "model_name" in data:
                value = scalar_to_str(data["model_name"])
                if value:
                    return value
    return "openai/clip-vit-base-patch16"


def map_query_indices_to_lidarseg(
    pred_query_indices: np.ndarray,
    model_valid_mask: np.ndarray,
    query_class_names: list[str],
    lidarseg_class_names: list[str],
) -> np.ndarray:
    name_to_label_id = {name: idx for idx, name in enumerate(lidarseg_class_names)}
    query_to_label = np.asarray([name_to_label_id.get(name, -1) for name in query_class_names], dtype=np.int32)
    pred_label_indices = np.full(pred_query_indices.shape[0], -1, dtype=np.int32)
    valid = model_valid_mask & (pred_query_indices >= 0) & (pred_query_indices < len(query_class_names))
    mapped = query_to_label[pred_query_indices[valid]]
    keep = mapped >= 0
    valid_positions = np.flatnonzero(valid)
    pred_label_indices[valid_positions[keep]] = mapped[keep]
    return pred_label_indices


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("predict_3d_open_vocab")
    torch = import_torch()
    device = resolve_device(torch, args.device)
    output_dir = ensure_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch_load_checkpoint(torch, checkpoint_path)
    model, backbone, feature_dim, num_output_classes = build_model_from_checkpoint(torch, checkpoint, device)
    point_feature_paths = build_point_feature_paths(args)
    query_class_names = load_class_names(args)
    if not query_class_names:
        raise ValueError("No query class names provided.")
    lidarseg_class_names = load_text_lines(args.lidarseg_class_names_path)
    text_model_name = infer_text_model_name(args, point_feature_paths)
    text_encoder = TextEncoder(
        model_name=text_model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    text_result = text_encoder.encode_texts(query_class_names, prompt_template=args.prompt_template, normalize=True)
    text_embeddings = text_result["text_embeddings"].astype(np.float32)
    if text_embeddings.shape[1] != feature_dim:
        raise ValueError(
            "Text embedding dimension does not match 3D point embedding dimension: "
            f"text_dim={text_embeddings.shape[1]}, point_feature_dim={feature_dim}, model={text_model_name}"
        )

    batch_summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "backbone": backbone,
        "feature_dim": feature_dim,
        "num_closed_set_output_classes": num_output_classes,
        "text_model_name": text_model_name,
        "prompt_template": args.prompt_template,
        "num_query_classes": len(query_class_names),
        "outputs": [],
    }
    logger.info(
        "open-vocab predict start | checkpoint=%s | backbone=%s | feature_dim=%d | queries=%d | device=%s",
        checkpoint_path,
        backbone,
        feature_dim,
        len(query_class_names),
        device,
    )

    for point_feature_path in point_feature_paths:
        if not point_feature_path.exists():
            raise FileNotFoundError(f"point feature npz not found: {point_feature_path}")
        point_data = load_npz(point_feature_path)
        sample_idx = int(point_data["sample_idx"].item())
        prefix = f"sample_{sample_idx:04d}"
        output_npz = output_dir / f"{prefix}_open_vocab_predictions.npz"
        summary_json = output_dir / f"{prefix}_open_vocab_prediction_summary.json"
        ply_path = output_dir / f"{prefix}_open_vocab_predictions.ply"
        bev_path = output_dir / f"{prefix}_open_vocab_predictions_bev.png"
        if args.skip_existing and summary_json.exists() and is_valid_npz(
            output_npz,
            required_keys=("point_xyz", "pred_query_indices", "pred_label_indices", "pred_scores"),
        ):
            logger.info("skip existing open-vocab prediction for sample_idx=%d", sample_idx)
            batch_summary["outputs"].append(
                {"sample_idx": sample_idx, "status": "skipped_existing", "output_npz": str(output_npz)}
            )
            continue

        point_xyz = point_data["point_xyz"].astype(np.float32)
        torch_batch = {
            "point_xyz": torch.from_numpy(point_xyz).to(device),
            "point_batch_indices": torch.zeros(point_xyz.shape[0], dtype=torch.long, device=device),
        }
        with torch.no_grad():
            outputs = model(torch_batch)
            point_embeddings = outputs["point_features"].detach().float().cpu().numpy().astype(np.float32)
            model_valid_mask = outputs.get(
                "model_valid_mask",
                torch.ones(point_xyz.shape[0], dtype=torch.bool, device=device),
            ).detach().cpu().numpy().astype(bool)

        pred_query_indices = np.full(point_xyz.shape[0], -1, dtype=np.int32)
        pred_scores = np.full(point_xyz.shape[0], np.nan, dtype=np.float32)
        similarities = None
        if np.any(model_valid_mask):
            valid_similarities = zero_shot_logits(point_embeddings[model_valid_mask], text_embeddings).astype(np.float32)
            valid_pred_indices = np.argmax(valid_similarities, axis=1).astype(np.int32)
            valid_pred_scores = valid_similarities[np.arange(valid_similarities.shape[0]), valid_pred_indices]
            pred_query_indices[model_valid_mask] = valid_pred_indices
            pred_scores[model_valid_mask] = valid_pred_scores.astype(np.float32)
            if args.save_similarities:
                similarities = np.full((point_xyz.shape[0], len(query_class_names)), np.nan, dtype=np.float32)
                similarities[model_valid_mask] = valid_similarities

        pred_label_indices = map_query_indices_to_lidarseg(
            pred_query_indices=pred_query_indices,
            model_valid_mask=model_valid_mask,
            query_class_names=query_class_names,
            lidarseg_class_names=lidarseg_class_names,
        )
        valid_lidarseg_prediction_mask = model_valid_mask & (pred_label_indices >= 0)

        save_bev_prediction_plot(
            point_xyz=point_xyz,
            label_indices=pred_query_indices,
            output_path=bev_path,
            valid_mask=model_valid_mask,
            num_classes=len(query_class_names),
        )
        save_point_cloud_ply(
            point_xyz=point_xyz,
            label_indices=pred_query_indices,
            output_path=ply_path,
            valid_mask=model_valid_mask,
            num_classes=len(query_class_names),
        )

        save_kwargs: dict[str, Any] = {
            "sample_idx": np.array(sample_idx, dtype=np.int32),
            "sample_token": point_data["sample_token"],
            "point_xyz": point_xyz,
            "model_valid_mask": model_valid_mask,
            "pred_query_indices": pred_query_indices,
            "pred_label_indices": pred_label_indices,
            "pred_scores": pred_scores,
            "class_names": np.asarray(query_class_names),
            "query_class_names": np.asarray(query_class_names),
            "lidarseg_class_names": np.asarray(lidarseg_class_names),
            "prompts": np.asarray(text_result["prompts"]),
            "text_embeddings": text_embeddings,
            "checkpoint_path": np.asarray(str(checkpoint_path)),
            "backbone": np.asarray(backbone),
            "teacher_mode": np.asarray(checkpoint.get("args", {}).get("teacher_mode", "")),
            "student_output_space": np.asarray(checkpoint.get("args", {}).get("student_output_space", "")),
            "text_model_name": np.asarray(text_model_name),
            "prompt_template": np.asarray(args.prompt_template),
            "feature_dim": np.array(feature_dim, dtype=np.int32),
            "num_closed_set_output_classes": np.array(num_output_classes, dtype=np.int32),
        }
        if args.save_point_embeddings:
            save_kwargs["point_embeddings"] = point_embeddings
        if similarities is not None:
            save_kwargs["similarities"] = similarities
        save_npz(output_npz, **save_kwargs)

        class_hist = {}
        for class_idx, class_name in enumerate(query_class_names):
            count = int(np.sum(pred_query_indices == class_idx))
            if count > 0:
                class_hist[class_name] = count
        mapped_query_count = int(sum(1 for name in query_class_names if name in set(lidarseg_class_names)))
        summary = {
            "sample_idx": sample_idx,
            "sample_token": scalar_to_str(point_data["sample_token"]),
            "checkpoint": str(checkpoint_path),
            "backbone": backbone,
            "teacher_mode": str(checkpoint.get("args", {}).get("teacher_mode", "")),
            "student_output_space": str(checkpoint.get("args", {}).get("student_output_space", "")),
            "feature_dim": feature_dim,
            "num_closed_set_output_classes": num_output_classes,
            "text_model_name": text_model_name,
            "prompt_template": args.prompt_template,
            "num_query_classes": len(query_class_names),
            "num_lidarseg_mapped_query_classes": mapped_query_count,
            "num_points": int(point_xyz.shape[0]),
            "num_model_valid_points": int(model_valid_mask.sum()),
            "num_lidarseg_mapped_predictions": int(valid_lidarseg_prediction_mask.sum()),
            "model_valid_ratio": float(model_valid_mask.sum() / max(point_xyz.shape[0], 1)),
            "lidarseg_mapped_prediction_ratio": float(valid_lidarseg_prediction_mask.sum() / max(point_xyz.shape[0], 1)),
            "class_hist": class_hist,
            "point_feature_npz": str(point_feature_path),
            "output_npz": str(output_npz),
            "ply_path": str(ply_path),
            "bev_path": str(bev_path),
            "save_point_embeddings": bool(args.save_point_embeddings),
            "save_similarities": bool(args.save_similarities),
        }
        save_json(summary_json, summary)
        logger.info(
            "open-vocab prediction saved | sample_idx=%d | valid=%d/%d | mapped=%d | npz=%s",
            sample_idx,
            summary["num_model_valid_points"],
            summary["num_points"],
            summary["num_lidarseg_mapped_predictions"],
            output_npz,
        )
        batch_summary["outputs"].append(
            {
                "sample_idx": sample_idx,
                "status": "done",
                "output_npz": str(output_npz),
                "summary_json": str(summary_json),
            }
        )

    if len(point_feature_paths) > 1:
        batch_summary_path = output_dir / "batch_open_vocab_prediction_summary.json"
        save_json(batch_summary_path, batch_summary)
        logger.info("batch open-vocab prediction summary saved to: %s", batch_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
