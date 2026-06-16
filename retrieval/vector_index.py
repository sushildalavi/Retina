from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import json

import faiss
import numpy as np


@dataclass
class FaissVectorIndex:
    dimension: int
    ids: List[str]
    index: faiss.Index

    @classmethod
    def build(cls, embeddings: np.ndarray, ids: Sequence[str]) -> "FaissVectorIndex":
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if embeddings.shape[0] != len(ids):
            raise ValueError("embeddings and ids must match")
        vectors = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(dimension=vectors.shape[1], ids=list(ids), index=index)

    def search(self, query_embeddings: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        vectors = np.asarray(query_embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        faiss.normalize_L2(vectors)
        scores, positions = self.index.search(vectors, top_k)
        return positions, scores

    def save(self, index_path: str | Path, meta_path: str | Path) -> None:
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(json.dumps({"dimension": self.dimension, "ids": self.ids}, indent=2))

    @classmethod
    def load(cls, index_path: str | Path, meta_path: str | Path) -> "FaissVectorIndex":
        meta = json.loads(Path(meta_path).read_text())
        index = faiss.read_index(str(index_path))
        return cls(dimension=int(meta["dimension"]), ids=list(meta["ids"]), index=index)

