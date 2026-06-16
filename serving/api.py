from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from retrieval.search import RetinaSearchEngine


class SearchRequest(BaseModel):
    query: str
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

    @app.post("/search/text")
    def search_text(payload: SearchRequest) -> Dict[str, Any]:
        results = state.engine.search_text(payload.query, top_k=payload.top_k)
        return {"query": payload.query, "top_k": payload.top_k, "results": results}

    return app

