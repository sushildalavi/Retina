from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import json
import pandas as pd

from retrieval.vector_index import FaissVectorIndex


@dataclass
class RetinaSearchEngine:
    encoder: RetinaClipEncoder
    index: FaissVectorIndex
    metadata: pd.DataFrame

    @classmethod
    def load(
        cls,
        model_name: str,
        device: str,
        metadata_path: str | Path,
        index_path: str | Path,
        index_meta_path: str | Path,
    ) -> "RetinaSearchEngine":
        from retrieval.clip_encoder import RetinaClipEncoder

        metadata_path = Path(metadata_path)
        if metadata_path.suffix.lower() in {".jsonl", ".ndjson"}:
            metadata = pd.read_json(metadata_path, lines=True)
        else:
            metadata = pd.read_csv(metadata_path)
        if "caption" not in metadata.columns and "captions" in metadata.columns:
            metadata["caption"] = metadata["captions"].apply(
                lambda value: value[0] if isinstance(value, list) and value else ""
            )
        encoder = RetinaClipEncoder(model_name=model_name, device=device)
        index = FaissVectorIndex.load(index_path, index_meta_path)
        return cls(encoder=encoder, index=index, metadata=metadata)

    def _format_results(self, positions, scores) -> List[Dict[str, Any]]:
        rows = []
        for rank, (pos, score) in enumerate(zip(positions, scores), start=1):
            if pos < 0:
                continue
            row = self.metadata.iloc[int(pos)].to_dict()
            if "caption" not in row and "captions" in row:
                captions = row["captions"]
                row["caption"] = captions[0] if isinstance(captions, list) and captions else ""
            row["score"] = float(score)
            row["rank"] = rank
            rows.append(row)
        return rows

    def search_text(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.encoder.encode_texts([query], batch_size=1)
        positions, scores = self.index.search(query_embedding, top_k=top_k)
        return self._format_results(positions[0], scores[0])

    def search_image(self, image_path: str | Path, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.encoder.encode_image_paths([image_path], batch_size=1)
        positions, scores = self.index.search(query_embedding, top_k=top_k)
        return self._format_results(positions[0], scores[0])
