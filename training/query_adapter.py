from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class QueryAdapter(nn.Module):
    """A lightweight trainable projection head for CLIP query embeddings."""

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(256, embedding_dim * 2)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.backbone = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.backbone(embeddings)
        projected = projected + embeddings
        return F.normalize(projected, dim=-1)

    def config(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }


def save_query_adapter(model: QueryAdapter, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "model_type": "retina_query_adapter",
        "config": model.config(),
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_query_adapter(path: str | Path, device: str = "cpu") -> tuple[QueryAdapter, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = checkpoint.get("config") or {}
    embedding_dim = int(config.get("embedding_dim"))
    hidden_dim = int(config.get("hidden_dim", max(256, embedding_dim * 2)))
    dropout = float(config.get("dropout", 0.1))
    model = QueryAdapter(embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    metadata = dict(checkpoint.get("metadata") or {})
    return model, metadata


def apply_query_adapter(embeddings: np.ndarray, adapter: QueryAdapter | None, device: str = "cpu") -> np.ndarray:
    if adapter is None:
        return np.asarray(embeddings, dtype=np.float32)
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    with torch.no_grad():
        tensor = torch.as_tensor(vectors, dtype=torch.float32, device=device)
        adapted = adapter(tensor).detach().cpu().numpy().astype(np.float32)
    return adapted
