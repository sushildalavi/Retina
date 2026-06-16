from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import ast

import numpy as np
import pandas as pd

from retrieval.clip_encoder import RetinaClipEncoder
from retrieval.vector_index import FaissVectorIndex


def _parse_caption_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    items = parsed
                else:
                    items = [text]
            except Exception:
                items = [text]
        else:
            items = [text]
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _load_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        metadata = pd.read_json(path, lines=True)
    else:
        metadata = pd.read_csv(path)
    if "captions" not in metadata.columns and "caption" in metadata.columns:
        metadata["captions"] = metadata["caption"].apply(_parse_caption_value)
    else:
        metadata["captions"] = metadata["captions"].apply(_parse_caption_value)
    if "caption" not in metadata.columns:
        metadata["caption"] = metadata["captions"].apply(lambda captions: captions[0] if captions else "")
    return metadata.reset_index(drop=True)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return vector / norm


def _first_caption(captions: Any) -> str:
    parsed = _parse_caption_value(captions)
    return parsed[0] if parsed else ""


@dataclass
class RetinaRecommender:
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
    ) -> "RetinaRecommender":
        encoder = RetinaClipEncoder(model_name=model_name, device=device)
        index = FaissVectorIndex.load(index_path, index_meta_path)
        metadata = _load_metadata(metadata_path)
        return cls(encoder=encoder, index=index, metadata=metadata)

    def __post_init__(self) -> None:
        self.metadata = self.metadata.reset_index(drop=True).copy()
        if "captions" not in self.metadata.columns:
            self.metadata["captions"] = self.metadata["caption"].apply(_parse_caption_value)
        if "caption" not in self.metadata.columns:
            self.metadata["caption"] = self.metadata["captions"].apply(_first_caption)
        self._image_lookup = {
            str(row["image_id"]): idx for idx, row in self.metadata.iterrows() if "image_id" in row and pd.notna(row["image_id"])
        }

    def _row_to_result(
        self,
        position: int,
        score: float,
        rank: int,
        recommendation_reason: str,
    ) -> Dict[str, Any]:
        row = self.metadata.iloc[int(position)].to_dict()
        row["captions"] = _parse_caption_value(row.get("captions"))
        row["caption"] = row["captions"][0] if row["captions"] else str(row.get("caption", ""))
        row["score"] = float(score)
        row["rank"] = int(rank)
        row["index_position"] = int(position)
        row["recommendation_reason"] = recommendation_reason
        return row

    def _search_embedding(
        self,
        embedding: np.ndarray,
        top_k: int,
        recommendation_reason: str,
        exclude_image_ids: Sequence[str] | None = None,
    ) -> List[Dict[str, Any]]:
        search_k = max(top_k + (len(exclude_image_ids) if exclude_image_ids else 0) + 5, top_k)
        positions, scores = self.index.search(embedding, top_k=search_k)
        excluded = {str(image_id) for image_id in exclude_image_ids or []}
        results: List[Dict[str, Any]] = []
        for position, score in zip(positions[0], scores[0]):
            if position < 0:
                continue
            image_id = str(self.metadata.iloc[int(position)]["image_id"])
            if image_id in excluded:
                continue
            results.append(self._row_to_result(int(position), float(score), len(results) + 1, recommendation_reason))
            if len(results) >= top_k:
                break
        return results

    def recommend_text(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        embedding = self.encoder.encode_texts([query], batch_size=1)
        return self._search_embedding(embedding, top_k, "high_clip_similarity_to_text_query")

    def recommend_image(
        self,
        image_id: str | None = None,
        image_path: str | Path | None = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if image_id is not None:
            lookup = self._image_lookup.get(str(image_id))
            if lookup is None:
                raise KeyError(f"Unknown image_id: {image_id}")
            image_path = self.metadata.iloc[lookup]["image_path"]
            exclude = [str(image_id)]
        elif image_path is not None:
            exclude = []
        else:
            raise ValueError("image_id or image_path is required")
        embedding = self.encoder.encode_image_paths([image_path], batch_size=1)
        return self._search_embedding(embedding, top_k, "high_visual_similarity_to_seed_image", exclude_image_ids=exclude)

    def recommend_profile(
        self,
        text_queries: Sequence[str] | None = None,
        liked_image_ids: Sequence[str] | None = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        text_queries = [query for query in (text_queries or []) if str(query).strip()]
        liked_image_ids = [str(image_id) for image_id in (liked_image_ids or []) if str(image_id).strip()]
        vectors: List[np.ndarray] = []
        if text_queries:
            vectors.append(self.encoder.encode_texts(text_queries, batch_size=max(1, len(text_queries))))
        for image_id in liked_image_ids:
            lookup = self._image_lookup.get(str(image_id))
            if lookup is None:
                continue
            image_path = self.metadata.iloc[lookup]["image_path"]
            vectors.append(self.encoder.encode_image_paths([image_path], batch_size=1))
        if not vectors:
            raise ValueError("Provide at least one text query or liked image id")
        combined = np.concatenate(vectors, axis=0).mean(axis=0, keepdims=True)
        combined = _normalize_vector(combined.astype(np.float32))
        return self._search_embedding(
            combined,
            top_k,
            "high_similarity_to_content_profile",
            exclude_image_ids=liked_image_ids,
        )

    def search_text(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.recommend_text(query, top_k=top_k)

    def search_image(self, image_path: str | Path, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.recommend_image(image_path=image_path, top_k=top_k)
