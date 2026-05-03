from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ra_ov3dseg.models.image_encoder import _resolve_device


def prettify_label_name(label_name: str) -> str:
    """将 nuScenes 风格标签名转换成更适合文本提示的自然语言形式。"""

    return label_name.replace(".", " ").replace("_", " ").strip()


class TextEncoder:
    """基于 Hugging Face Transformers 的 text encoder 封装。"""

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
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TextEncoder requires `torch` and `transformers`. "
                "Please install PyTorch for your server CUDA/CPU environment, then install requirements."
            ) from exc

        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
            use_safetensors=True,
        )

        if not hasattr(self.model, "get_text_features"):
            raise ValueError(
                f"Model `{model_name}` does not expose `get_text_features`; "
                "use a CLIP/SigLIP style model such as `openai/clip-vit-base-patch16`."
            )

        self.model.eval()
        self.model.to(self.device)

    def _apply_text_projection(self, features):
        if hasattr(self.model, "text_projection"):
            return self.model.text_projection(features)
        return features

    def _extract_text_embeddings(self, inputs) -> np.ndarray:
        text_outputs = self.model.get_text_features(**inputs)
        if self.torch.is_tensor(text_outputs):
            return text_outputs.detach().cpu().numpy().astype(np.float32)

        if hasattr(text_outputs, "text_embeds") and self.torch.is_tensor(text_outputs.text_embeds):
            return text_outputs.text_embeds.detach().cpu().numpy().astype(np.float32)

        if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            text_embeddings = self._apply_text_projection(text_outputs.pooler_output)
            return text_embeddings.detach().cpu().numpy().astype(np.float32)

        raise ValueError(f"Cannot extract text embeddings for model `{self.model_name}`.")

    def build_prompts(self, class_names: list[str], prompt_template: str | None = None) -> list[str]:
        if prompt_template is None:
            return [prettify_label_name(class_name) for class_name in class_names]
        return [prompt_template.format(prettify_label_name(class_name)) for class_name in class_names]

    def encode_texts(
        self,
        class_names: list[str],
        prompt_template: str | None = None,
        normalize: bool = True,
    ) -> dict[str, Any]:
        prompts = self.build_prompts(class_names, prompt_template=prompt_template)
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            text_embeddings_np = self._extract_text_embeddings(inputs)

        if normalize:
            text_norm = np.linalg.norm(text_embeddings_np, axis=-1, keepdims=True)
            text_embeddings_np = text_embeddings_np / np.clip(text_norm, 1e-6, None)

        return {
            "class_names": class_names,
            "prompts": prompts,
            "text_embeddings": text_embeddings_np.astype(np.float32),
            "model_name": self.model_name,
            "cache_dir": self.cache_dir or "",
        }

    def encode_text(self, class_names: list[str], prompt_template: str | None = None) -> np.ndarray:
        return self.encode_texts(class_names, prompt_template=prompt_template)["text_embeddings"]
