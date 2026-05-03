from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.evaluation.openvocab_eval import zero_shot_predict  # noqa: E402
from ra_ov3dseg.models.text_encoder import TextEncoder  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, load_text_lines, save_json, save_npz  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402
from ra_ov3dseg.visualization.visualize_points import (  # noqa: E402
    save_bev_prediction_plot,
    save_point_cloud_ply,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 3D 点特征做 zero-shot 文本分类。")
    parser.add_argument("--sample_idx", default=None, type=int, help="指定单个 sample 索引。")
    parser.add_argument("--start_idx", default=0, type=int, help="批量评估时的起始 sample 索引。")
    parser.add_argument("--max_samples", default=1, type=int, help="批量评估多少个 sample。")
    parser.add_argument("--point_feature_npz", default=None, type=str, help="单个点特征 .npz 路径。")
    parser.add_argument("--point_feature_dir", default="outputs/point_features", type=str, help="点特征目录。")
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="文本编码模型名。默认从点特征文件中读取与图像侧相同的模型名。",
    )
    parser.add_argument(
        "--class_names_path",
        default="configs/nuscenes_lidarseg_class_names.txt",
        type=str,
        help="类别名 txt 文件，每行一个类别。",
    )
    parser.add_argument(
        "--class_names_csv",
        default=None,
        type=str,
        help="可选：直接用逗号分隔的类别名字符串覆盖 class_names_path。",
    )
    parser.add_argument(
        "--prompt_template",
        default="a {} in a driving scene",
        type=str,
        help="zero-shot 文本提示模板，必须包含一个 `{}` 占位符。",
    )
    parser.add_argument("--device", default="auto", type=str, help="运行设备：auto/cpu/cuda。")
    parser.add_argument("--output_dir", default="outputs/zero_shot", type=str, help="zero-shot 输出目录。")
    parser.add_argument("--skip_existing", action="store_true", help="如果 zero-shot 输出已存在，则跳过。")
    return parser


def load_npz_as_dict(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_class_names(args) -> list[str]:
    if args.class_names_csv is not None:
        return [item.strip() for item in args.class_names_csv.split(",") if item.strip()]
    return load_text_lines(args.class_names_path)


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("zero_shot_eval")
    output_dir = ensure_dir(args.output_dir)

    if args.point_feature_npz is not None:
        point_feature_paths = [Path(args.point_feature_npz).resolve()]
    else:
        point_feature_dir = Path(args.point_feature_dir).resolve()
        if args.sample_idx is not None:
            sample_indices = [args.sample_idx]
        else:
            sample_indices = list(range(args.start_idx, args.start_idx + args.max_samples))
        point_feature_paths = [
            point_feature_dir / f"sample_{sample_idx:04d}_point_features.npz"
            for sample_idx in sample_indices
        ]

    class_names = load_class_names(args)
    if len(class_names) == 0:
        raise ValueError("No class names provided.")

    text_encoder = None
    batch_summary = {
        "class_names_path": args.class_names_path,
        "num_classes": len(class_names),
        "outputs": [],
    }

    for point_feature_path in point_feature_paths:
        if not point_feature_path.exists():
            raise FileNotFoundError(f"point feature npz not found: {point_feature_path}")

        data = load_npz_as_dict(point_feature_path)
        sample_idx = int(data["sample_idx"].item())
        prefix = f"sample_{sample_idx:04d}"
        output_npz = output_dir / f"{prefix}_zero_shot_predictions.npz"
        summary_json = output_dir / f"{prefix}_zero_shot_summary.json"
        ply_path = output_dir / f"{prefix}_zero_shot_points.ply"
        bev_path = output_dir / f"{prefix}_zero_shot_bev.png"

        if args.skip_existing and output_npz.exists() and summary_json.exists():
            logger.info("skip existing zero-shot outputs for sample_idx=%d", sample_idx)
            batch_summary["outputs"].append(
                {
                    "sample_idx": sample_idx,
                    "status": "skipped_existing",
                    "output_npz": str(output_npz),
                }
            )
            continue

        model_name = args.model_name or str(data["model_name"].item())
        if text_encoder is None or text_encoder.model_name != model_name:
            text_encoder = TextEncoder(model_name=model_name, device=args.device)
        text_result = text_encoder.encode_texts(class_names, prompt_template=args.prompt_template, normalize=True)

        point_xyz = data["point_xyz"].astype(np.float32)
        point_features = data["point_features"].astype(np.float32)
        point_valid_mask = data["point_valid_mask"].astype(bool)

        pred_label_indices = np.full(point_xyz.shape[0], -1, dtype=np.int32)
        pred_scores = np.full(point_xyz.shape[0], np.nan, dtype=np.float32)
        if np.any(point_valid_mask):
            valid_pred_indices, valid_pred_scores = zero_shot_predict(
                point_features[point_valid_mask],
                text_result["text_embeddings"],
            )
            pred_label_indices[point_valid_mask] = valid_pred_indices
            pred_scores[point_valid_mask] = valid_pred_scores

        class_hist = {}
        for class_idx, class_name in enumerate(class_names):
            class_count = int(np.sum(pred_label_indices == class_idx))
            if class_count > 0:
                class_hist[class_name] = class_count

        save_bev_prediction_plot(
            point_xyz=point_xyz,
            label_indices=pred_label_indices,
            output_path=bev_path,
            valid_mask=point_valid_mask,
            num_classes=len(class_names),
        )
        save_point_cloud_ply(
            point_xyz=point_xyz,
            label_indices=pred_label_indices,
            output_path=ply_path,
            valid_mask=point_valid_mask,
            num_classes=len(class_names),
        )

        save_npz(
            output_npz,
            sample_idx=data["sample_idx"],
            sample_token=data["sample_token"],
            point_xyz=point_xyz,
            point_valid_mask=point_valid_mask,
            pred_label_indices=pred_label_indices,
            pred_scores=pred_scores,
            class_names=np.asarray(class_names),
            prompts=np.asarray(text_result["prompts"]),
            text_embeddings=text_result["text_embeddings"].astype(np.float32),
            model_name=np.array(model_name),
        )
        summary = {
            "sample_idx": sample_idx,
            "sample_token": str(data["sample_token"].item()),
            "model_name": model_name,
            "num_points": int(point_xyz.shape[0]),
            "num_valid_points": int(point_valid_mask.sum()),
            "num_classes": len(class_names),
            "class_hist": class_hist,
            "prompt_template": args.prompt_template,
            "point_feature_npz": str(point_feature_path),
            "output_npz": str(output_npz),
            "ply_path": str(ply_path),
            "bev_path": str(bev_path),
        }
        save_json(summary_json, summary)
        logger.info(
            "zero-shot outputs saved | sample_idx=%d | valid_points=%d | classes=%d",
            sample_idx,
            summary["num_valid_points"],
            len(class_names),
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
        batch_summary_path = output_dir / "batch_zero_shot_summary.json"
        save_json(batch_summary_path, batch_summary)
        logger.info("batch zero-shot summary saved to: %s", batch_summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
