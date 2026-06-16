from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json
from typing import Any, Dict, List, Sequence

import pandas as pd
from fastapi import Request

from retrieval.search import RetinaSearchEngine
from serving.errors import RetinaInvalidRequestError, RetinaMissingArtifactError, RetinaModelLoadError, RetinaNotFoundError
from serving.settings import RetinaSettings


def _parse_captions(value: Any) -> List[str]:
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
                import ast

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


def _normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.reset_index(drop=True).copy()
    if "captions" not in metadata.columns:
        if "caption" in metadata.columns:
            metadata["captions"] = metadata["caption"].apply(_parse_captions)
        else:
            metadata["captions"] = [[] for _ in range(len(metadata))]
    else:
        metadata["captions"] = metadata["captions"].apply(_parse_captions)
    if "caption" not in metadata.columns:
        metadata["caption"] = metadata["captions"].apply(lambda captions: captions[0] if captions else "")
    return metadata


def _load_metadata_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


@dataclass
class RetinaService:
    settings: RetinaSettings
    engine: RetinaSearchEngine
    metadata: pd.DataFrame

    @classmethod
    def load(cls, config_path: str | Path) -> "RetinaService":
        settings = RetinaSettings.from_yaml(config_path)
        settings.check_artifacts()
        try:
            engine = RetinaSearchEngine.load(
                model_name=settings.model_name,
                device=settings.device,
                metadata_path=settings.metadata_path,
                index_path=settings.index_path,
                index_meta_path=settings.index_meta_path,
            )
        except RetinaMissingArtifactError:
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise RetinaModelLoadError(
                "Failed to load the CLIP model or FAISS index",
                details={"setup_command": settings.setup_command, "config_path": str(settings.config_path)},
            ) from exc
        metadata = _normalize_metadata(engine.metadata if hasattr(engine, "metadata") else _load_metadata_table(settings.metadata_path))
        return cls(settings=settings, engine=engine, metadata=metadata)

    @property
    def indexed_image_count(self) -> int:
        if "image_id" not in self.metadata.columns:
            return int(len(self.metadata))
        return int(self.metadata["image_id"].nunique())

    @property
    def indexed_caption_count(self) -> int:
        if "captions" not in self.metadata.columns:
            return int(len(self.metadata))
        return int(sum(len(_parse_captions(captions)) for captions in self.metadata["captions"]))

    def service_metadata(self) -> Dict[str, Any]:
        return {
            "config_path": str(self.settings.config_path),
            "model_name": self.settings.model_name,
            "dataset_name": self.settings.dataset_name,
            "indexed_image_count": self.indexed_image_count,
            "indexed_caption_count": self.indexed_caption_count,
            "device": self.settings.device,
            "report_paths": self.settings.report_paths(),
            "setup_command": self.settings.setup_command,
        }

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "ready": True,
            "checks": {
                "config": self.settings.config_path.exists(),
                "metadata": self.settings.metadata_path.exists(),
                "index": self.settings.index_path.exists(),
                "index_meta": self.settings.index_meta_path.exists(),
                "images": self.settings.image_artifact_dir.exists(),
                "model": True,
            },
        }

    def metrics_summary(self) -> Dict[str, Any]:
        summary_path = self.settings.reports_dir / "final_retina_metrics_summary.json"
        if not summary_path.exists():
            raise RetinaMissingArtifactError(
                "Metrics summary report is missing",
                details={"summary_path": str(summary_path), "setup_command": self.settings.setup_command},
            )
        return {"summary_path": str(summary_path), "metrics": json.loads(summary_path.read_text())}

    def _image_url(self, image_path: str | Path | None) -> str | None:
        if not image_path:
            return None
        path = Path(image_path)
        root = self.settings.image_artifact_dir
        try:
            relative = path.relative_to(root)
            return f"/artifacts/images/{relative.as_posix()}"
        except Exception:
            pass
        try:
            relative = path.resolve().relative_to(root.resolve())
        except Exception:
            return None
        return f"/artifacts/images/{relative.as_posix()}"

    def _format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(result)
        captions = _parse_captions(payload.get("captions"))
        payload["captions"] = captions
        payload["caption"] = captions[0] if captions else str(payload.get("caption", ""))
        payload["image_url"] = self._image_url(payload.get("image_path"))
        payload.setdefault("metadata", {})
        return payload

    def _method(self, primary: str, fallback: str):
        method = getattr(self.engine, primary, None)
        if method is None:
            method = getattr(self.engine, fallback)
        return method

    def recommend_text(self, query: str, top_k: int) -> tuple[List[Dict[str, Any]], float]:
        start = perf_counter()
        method = self._method("recommend_text", "search_text")
        results = [self._format_result(item) for item in method(query, top_k=top_k)]
        return results, (perf_counter() - start) * 1000.0

    def recommend_image(self, image_id: str, top_k: int) -> tuple[List[Dict[str, Any]], float]:
        start = perf_counter()
        method = self._method("recommend_image", "search_image")
        try:
            results = method(image_id=image_id, top_k=top_k)
        except TypeError:
            results = method(image_id, top_k=top_k)
        except KeyError as exc:
            raise RetinaNotFoundError(str(exc), details={"image_id": image_id}) from exc
        results = [self._format_result(item) for item in results]
        return results, (perf_counter() - start) * 1000.0

    def recommend_profile(
        self,
        text_queries: Sequence[str],
        liked_image_ids: Sequence[str],
        top_k: int,
    ) -> tuple[List[Dict[str, Any]], float]:
        if not any(str(item).strip() for item in text_queries) and not any(str(item).strip() for item in liked_image_ids):
            raise RetinaInvalidRequestError(
                "Provide at least one text query or liked image id",
                details={"setup_command": self.settings.setup_command},
            )
        start = perf_counter()
        method = self._method("recommend_profile", "search_text")
        try:
            results = method(text_queries=list(text_queries), liked_image_ids=list(liked_image_ids), top_k=top_k)
        except TypeError:
            results = method(" ".join(text_queries), top_k=top_k)
        results = [self._format_result(item) for item in results]
        return results, (perf_counter() - start) * 1000.0


def get_service(request: Request) -> RetinaService:
    service = getattr(request.app.state, "retina_service", None)
    if service is None:
        raise RetinaModelLoadError(
            "Retina service is not loaded",
            details={"setup_command": "make api", "config_path": "configs/retina.yaml"},
        )
    return service


def load_service(config_path: str | Path) -> RetinaService:
    return RetinaService.load(config_path)
