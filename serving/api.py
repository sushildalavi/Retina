from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from retrieval.search import RetinaSearchEngine


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class ProfileRequest(BaseModel):
    text_queries: List[str] = Field(default_factory=list)
    liked_image_ids: List[str] = Field(default_factory=list)
    top_k: int = 10


@dataclass
class RetinaAPIState:
    engine: RetinaSearchEngine
    config_path: str


def create_app(engine: RetinaSearchEngine, config_path: str = "configs/retina.yaml") -> FastAPI:
    app = FastAPI(title="Retina API", version="0.1.0")
    state = RetinaAPIState(engine=engine, config_path=config_path)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> Dict[str, Any]:
        return {
            "config_path": state.config_path,
            "rows": int(len(state.engine.metadata)),
            "model_name": state.engine.encoder.model_name,
            "device": state.engine.encoder.device,
        }

    @app.get("/recommend/text")
    def recommend_text(query: str, top_k: int = 10) -> Dict[str, Any]:
        method = getattr(state.engine, "recommend_text", state.engine.search_text)
        results = method(query, top_k=top_k)
        return {"query": query, "top_k": top_k, "results": results}

    @app.get("/recommend/image")
    def recommend_image(image_id: str, top_k: int = 10) -> Dict[str, Any]:
        method = getattr(state.engine, "recommend_image", None)
        if method is None:
            method = getattr(state.engine, "search_image")
            results = method(image_id, top_k=top_k)
        else:
            results = method(image_id=image_id, top_k=top_k)
        return {"image_id": image_id, "top_k": top_k, "results": results}

    @app.post("/recommend/profile")
    def recommend_profile(payload: ProfileRequest) -> Dict[str, Any]:
        method = getattr(state.engine, "recommend_profile", None)
        if method is None:
            results = state.engine.search_text(" ".join(payload.text_queries), top_k=payload.top_k)
        else:
            results = method(
                text_queries=payload.text_queries,
                liked_image_ids=payload.liked_image_ids,
                top_k=payload.top_k,
            )
        return {
            "text_queries": payload.text_queries,
            "liked_image_ids": payload.liked_image_ids,
            "top_k": payload.top_k,
            "results": results,
        }

    @app.post("/search/text")
    def search_text(payload: SearchRequest) -> Dict[str, Any]:
        method = getattr(state.engine, "recommend_text", state.engine.search_text)
        results = method(payload.query, top_k=payload.top_k)
        return {"query": payload.query, "top_k": payload.top_k, "results": results}

    return app
