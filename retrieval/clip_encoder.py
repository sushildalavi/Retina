from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def select_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_embeddings(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return array / norm


@dataclass
class RetinaClipEncoder:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"

    def __post_init__(self) -> None:
        self.device = select_device(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _move_inputs(self, inputs: dict) -> dict:
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _encode_batches(self, batches: Iterable[dict], feature_fn, output_attr: str | None = None) -> np.ndarray:
        vectors: List[np.ndarray] = []
        with torch.no_grad():
            for batch in batches:
                outputs = feature_fn(**self._move_inputs(batch))
                tensor = outputs
                if not torch.is_tensor(tensor):
                    if output_attr and hasattr(outputs, output_attr):
                        tensor = getattr(outputs, output_attr)
                    elif hasattr(outputs, "pooler_output"):
                        tensor = outputs.pooler_output
                    elif hasattr(outputs, "image_embeds"):
                        tensor = outputs.image_embeds
                    elif hasattr(outputs, "text_embeds"):
                        tensor = outputs.text_embeds
                    else:
                        raise TypeError(f"Unsupported CLIP output type: {type(outputs)!r}")
                tensor = F.normalize(tensor, dim=-1)
                vectors.append(tensor.detach().cpu().numpy().astype(np.float32))
        if not vectors:
            return np.zeros((0, self.model.config.projection_dim), dtype=np.float32)
        return np.concatenate(vectors, axis=0)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        batches = []
        for start in range(0, len(texts), batch_size):
            items = list(texts[start : start + batch_size])
            batches.append(self.processor(text=items, return_tensors="pt", padding=True, truncation=True))
        return self._encode_batches(batches, self.model.get_text_features, "text_embeds")

    def encode_image_paths(self, image_paths: Sequence[str | Path], batch_size: int = 16) -> np.ndarray:
        batches = []
        for start in range(0, len(image_paths), batch_size):
            paths = image_paths[start : start + batch_size]
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            batches.append(self.processor(images=images, return_tensors="pt"))
        return self._encode_batches(batches, self.model.get_image_features, "image_embeds")
