from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _resolve_device(torch_module, device: str) -> str:
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


class ImageEncoder:
    """基于 Hugging Face Transformers 的 2D image encoder 封装。

    当前优先支持 CLIP / SigLIP 这类同时具备：
    1. `vision_model`
    2. `get_image_features`

    的视觉语言模型。MVP-v1 只做最小链路，因此这里采用“整张图直接 resize 到模型输入尺寸、
    不做 center crop”的策略，便于把原图像素坐标稳定映射回 patch feature grid。
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = str(Path(cache_dir).expanduser().resolve()) if cache_dir is not None else None
        self.local_files_only = local_files_only

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "ImageEncoder requires `torch` and `transformers`. "
                "Please install PyTorch for your server CUDA/CPU environment, then install requirements."
            ) from exc

        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )

        if not hasattr(self.model, "vision_model") or not hasattr(self.model, "get_image_features"):
            raise ValueError(
                f"Model `{model_name}` does not expose `vision_model` and `get_image_features`; "
                "use a CLIP/SigLIP style model such as `openai/clip-vit-base-patch16`. "
                "Pure timm vision backbones are not enough for the current zero-shot text pipeline."
            )

        self.model.eval()
        self.model.to(self.device)
        self.target_height, self.target_width = self._resolve_target_hw()
        self.patch_size = self._resolve_patch_size()
        self.image_mean = np.asarray(getattr(self.image_processor, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.image_std = np.asarray(getattr(self.image_processor, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)

    def _resolve_target_hw(self) -> tuple[int, int]:
        size = getattr(self.image_processor, "size", None)
        if isinstance(size, dict):
            if "height" in size and "width" in size:
                return int(size["height"]), int(size["width"])
            if "shortest_edge" in size:
                edge = int(size["shortest_edge"])
                return edge, edge
        if isinstance(size, int):
            return int(size), int(size)

        vision_config = getattr(self.model.config, "vision_config", self.model.config)
        image_size = getattr(vision_config, "image_size", 224)
        if isinstance(image_size, (tuple, list)):
            return int(image_size[0]), int(image_size[1])
        return int(image_size), int(image_size)

    def _resolve_patch_size(self) -> int | None:
        vision_config = getattr(self.model.config, "vision_config", self.model.config)
        patch_size = getattr(vision_config, "patch_size", None)
        if patch_size is None:
            return None
        return int(patch_size)

    def _preprocess_image(self, image_path: str | Path) -> tuple[Any, dict[str, Any]]:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            resized_image = image.resize(
                (self.target_width, self.target_height),
                resample=Image.Resampling.BICUBIC,
            )

        image_np = np.asarray(resized_image, dtype=np.float32) / 255.0
        image_np = (image_np - self.image_mean.reshape(1, 1, 3)) / self.image_std.reshape(1, 1, 3)
        image_np = np.transpose(image_np, (2, 0, 1))
        pixel_values = self.torch.from_numpy(image_np).unsqueeze(0).to(self.device)
        metadata = {
            "image_path": str(image_path),
            "original_width": int(original_width),
            "original_height": int(original_height),
            "resized_width": int(self.target_width),
            "resized_height": int(self.target_height),
        }
        return pixel_values, metadata

    def _extract_patch_feature_map(self, vision_outputs: Any) -> np.ndarray:
        tokens = vision_outputs.last_hidden_state[0]
        num_tokens = int(tokens.shape[0])

        grid_height = None
        grid_width = None
        if self.patch_size is not None and self.patch_size > 0:
            grid_height = self.target_height // self.patch_size
            grid_width = self.target_width // self.patch_size

        if grid_height is not None and grid_width is not None:
            expected_tokens = grid_height * grid_width
            if num_tokens == expected_tokens + 1:
                tokens = tokens[1:]
                num_tokens = int(tokens.shape[0])
            if num_tokens == expected_tokens:
                return tokens.reshape(grid_height, grid_width, -1).detach().cpu().numpy().astype(np.float32)

        if num_tokens > 1:
            inferred = int(round((num_tokens - 1) ** 0.5))
            if inferred * inferred == num_tokens - 1:
                tokens = tokens[1:]
                return tokens.reshape(inferred, inferred, -1).detach().cpu().numpy().astype(np.float32)

        inferred = int(round(num_tokens**0.5))
        if inferred * inferred != num_tokens:
            raise ValueError(
                f"Cannot infer patch grid from {num_tokens} tokens for model `{self.model_name}`."
            )
        return tokens.reshape(inferred, inferred, -1).detach().cpu().numpy().astype(np.float32)

    def encode_image(
        self,
        image_path: str | Path,
        normalize: bool = True,
    ) -> dict[str, Any]:
        pixel_values, metadata = self._preprocess_image(image_path)

        with self.torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            image_embedding = self.model.get_image_features(pixel_values=pixel_values)[0]

        patch_feature_map = self._extract_patch_feature_map(vision_outputs)
        image_embedding_np = image_embedding.detach().cpu().numpy().astype(np.float32)

        if normalize:
            patch_norm = np.linalg.norm(patch_feature_map, axis=-1, keepdims=True)
            patch_feature_map = patch_feature_map / np.clip(patch_norm, 1e-6, None)

            image_norm = np.linalg.norm(image_embedding_np, axis=-1, keepdims=True)
            image_embedding_np = image_embedding_np / np.clip(image_norm, 1e-6, None)

        metadata.update(
            {
                "feature_grid_height": int(patch_feature_map.shape[0]),
                "feature_grid_width": int(patch_feature_map.shape[1]),
                "feature_dim": int(patch_feature_map.shape[2]),
                "model_name": self.model_name,
                "cache_dir": self.cache_dir or "",
            }
        )
        return {
            "patch_feature_map": patch_feature_map.astype(np.float32),
            "image_embedding": image_embedding_np.astype(np.float32),
            "metadata": metadata,
        }
