from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.models.segmentor_factory import build_segmentor  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, load_text_lines, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402
from ra_ov3dseg.visualization.visualize_points import (  # noqa: E402
    save_bev_prediction_plot,
    save_point_cloud_ply,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run point-level inference with a trained 3D segmentor checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to train_3d_segmentor checkpoint .pt.")
    parser.add_argument("--sample_idx", default=None, type=int, help="Single sample index.")
    parser.add_argument("--start_idx", default=0, type=int, help="Batch start sample index.")
    parser.add_argument("--max_samples", default=1, type=int, help="Number of samples in batch mode.")
    parser.add_argument("--point_feature_npz", default=None, type=str, help="Explicit point feature npz for one sample.")
    parser.add_argument("--point_feature_dir", default="outputs/point_features", type=str)
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_dir", default="outputs/predictions3d", type=str)
    parser.add_argument("--save_logits", action="store_true", help="Also save raw model logits in prediction npz.")
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("predict_3d_segmentor.py requires PyTorch.") from exc
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


def build_point_feature_paths(args) -> list[Path]:
    if args.point_feature_npz is not None:
        return [Path(args.point_feature_npz).expanduser().resolve()]
    if args.sample_idx is not None:
        sample_indices = [args.sample_idx]
    else:
        sample_indices = list(range(args.start_idx, args.start_idx + args.max_samples))
    point_feature_dir = Path(args.point_feature_dir).expanduser().resolve()
    return [point_feature_dir / f"sample_{sample_idx:04d}_point_features.npz" for sample_idx in sample_indices]


def infer_checkpoint_output_space(checkpoint: dict[str, Any], num_output_classes: int, class_names: list[str]) -> str:
    class_split = checkpoint.get("class_split", {})
    train_id_to_label_id = class_split.get("train_id_to_label_id", [])
    if num_output_classes == len(class_names):
        return "all_lidarseg"
    if num_output_classes == len(train_id_to_label_id):
        return "base"
    return "unknown"


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


def output_indices_to_lidarseg_labels(
    pred_output_indices: np.ndarray,
    model_valid_mask: np.ndarray,
    output_space: str,
    checkpoint: dict[str, Any],
) -> np.ndarray:
    pred_label_indices = np.full(pred_output_indices.shape[0], -1, dtype=np.int32)
    valid = model_valid_mask & (pred_output_indices >= 0)
    if output_space == "all_lidarseg":
        pred_label_indices[valid] = pred_output_indices[valid].astype(np.int32)
        return pred_label_indices

    if output_space == "base":
        train_id_to_label_id = np.asarray(checkpoint["class_split"]["train_id_to_label_id"], dtype=np.int32)
        in_range = valid & (pred_output_indices < train_id_to_label_id.shape[0])
        pred_label_indices[in_range] = train_id_to_label_id[pred_output_indices[in_range]]
        return pred_label_indices

    raise ValueError(f"Cannot map output indices for unknown output_space={output_space}.")


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("predict_3d_segmentor")
    torch = import_torch()
    device = resolve_device(torch, args.device)
    output_dir = ensure_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch_load_checkpoint(torch, checkpoint_path)
    class_names = checkpoint.get("class_split", {}).get("class_names")
    if class_names is None:
        class_names = load_text_lines(args.class_names_path)
    class_names = [str(name) for name in class_names]
    model, backbone, feature_dim, num_output_classes = build_model_from_checkpoint(torch, checkpoint, device)
    output_space = infer_checkpoint_output_space(checkpoint, num_output_classes, class_names)
    if output_space == "unknown":
        raise ValueError(f"Cannot infer output space from num_output_classes={num_output_classes}.")

    point_feature_paths = build_point_feature_paths(args)
    batch_summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "backbone": backbone,
        "output_space": output_space,
        "outputs": [],
    }

    logger.info(
        "predict start | checkpoint=%s | backbone=%s | output_space=%s | classes=%d | device=%s",
        checkpoint_path,
        backbone,
        output_space,
        num_output_classes,
        device,
    )

    for point_feature_path in point_feature_paths:
        if not point_feature_path.exists():
            raise FileNotFoundError(f"point feature npz not found: {point_feature_path}")
        point_data = load_npz(point_feature_path)
        sample_idx = int(point_data["sample_idx"].item())
        prefix = f"sample_{sample_idx:04d}"
        output_npz = output_dir / f"{prefix}_3d_predictions.npz"
        summary_json = output_dir / f"{prefix}_3d_prediction_summary.json"
        ply_path = output_dir / f"{prefix}_3d_predictions.ply"
        bev_path = output_dir / f"{prefix}_3d_predictions_bev.png"
        if args.skip_existing and output_npz.exists() and summary_json.exists():
            logger.info("skip existing prediction outputs for sample_idx=%d", sample_idx)
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
            logits = outputs["logits"].detach().float().cpu()
            probs = torch.softmax(logits, dim=-1).numpy().astype(np.float32)
            pred_output_indices = np.argmax(probs, axis=1).astype(np.int32)
            pred_scores = probs[np.arange(probs.shape[0]), pred_output_indices].astype(np.float32)
            model_valid_mask = outputs.get(
                "model_valid_mask",
                torch.ones(point_xyz.shape[0], dtype=torch.bool, device=device),
            ).detach().cpu().numpy().astype(bool)

        pred_output_indices[~model_valid_mask] = -1
        pred_scores[~model_valid_mask] = np.nan
        pred_label_indices = output_indices_to_lidarseg_labels(
            pred_output_indices=pred_output_indices,
            model_valid_mask=model_valid_mask,
            output_space=output_space,
            checkpoint=checkpoint,
        )
        valid_prediction_mask = model_valid_mask & (pred_label_indices >= 0)

        save_bev_prediction_plot(
            point_xyz=point_xyz,
            label_indices=pred_label_indices,
            output_path=bev_path,
            valid_mask=valid_prediction_mask,
            num_classes=len(class_names),
        )
        save_point_cloud_ply(
            point_xyz=point_xyz,
            label_indices=pred_label_indices,
            output_path=ply_path,
            valid_mask=valid_prediction_mask,
            num_classes=len(class_names),
        )

        save_kwargs: dict[str, Any] = {
            "sample_idx": np.array(sample_idx, dtype=np.int32),
            "sample_token": point_data["sample_token"],
            "point_xyz": point_xyz,
            "model_valid_mask": model_valid_mask,
            "pred_output_indices": pred_output_indices,
            "pred_label_indices": pred_label_indices,
            "pred_scores": pred_scores,
            "class_names": np.asarray(class_names),
            "checkpoint_path": np.asarray(str(checkpoint_path)),
            "backbone": np.asarray(backbone),
            "teacher_mode": np.asarray(checkpoint.get("args", {}).get("teacher_mode", "")),
            "student_output_space": np.asarray(output_space),
            "feature_dim": np.array(feature_dim, dtype=np.int32),
            "num_output_classes": np.array(num_output_classes, dtype=np.int32),
        }
        if args.save_logits:
            save_kwargs["logits"] = logits.numpy().astype(np.float32)
        save_npz(output_npz, **save_kwargs)

        class_hist = {}
        for class_idx, class_name in enumerate(class_names):
            count = int(np.sum(pred_label_indices == class_idx))
            if count > 0:
                class_hist[class_name] = count
        summary = {
            "sample_idx": sample_idx,
            "sample_token": scalar_to_str(point_data["sample_token"]),
            "checkpoint": str(checkpoint_path),
            "backbone": backbone,
            "teacher_mode": str(checkpoint.get("args", {}).get("teacher_mode", "")),
            "student_output_space": output_space,
            "feature_dim": feature_dim,
            "num_output_classes": num_output_classes,
            "num_points": int(point_xyz.shape[0]),
            "num_valid_predictions": int(valid_prediction_mask.sum()),
            "valid_prediction_ratio": float(valid_prediction_mask.sum() / max(point_xyz.shape[0], 1)),
            "class_hist": class_hist,
            "point_feature_npz": str(point_feature_path),
            "output_npz": str(output_npz),
            "ply_path": str(ply_path),
            "bev_path": str(bev_path),
            "save_logits": bool(args.save_logits),
        }
        save_json(summary_json, summary)
        logger.info(
            "prediction saved | sample_idx=%d | valid=%d/%d | npz=%s",
            sample_idx,
            summary["num_valid_predictions"],
            summary["num_points"],
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
        batch_summary_path = output_dir / "batch_3d_prediction_summary.json"
        save_json(batch_summary_path, batch_summary)
        logger.info("batch prediction summary saved to: %s", batch_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
