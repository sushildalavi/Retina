from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from serving.api import create_app
from serving.dependencies import load_service
from serving.errors import RetinaMissingArtifactError
from serving.schemas import (
    ErrorResponse,
    HealthResponse,
    ImageRecommendationResponse,
    MetadataResponse,
    MetricsSummaryResponse,
    ProfileRecommendationResponse,
    TextRecommendationResponse,
)


class FakeService:
    def __init__(self, image_root: Path):
        self.settings = SimpleNamespace(
            config_path=Path("configs/retina.yaml"),
            model_name="fake-model",
            dataset_name="flickr8k",
            device="cpu",
            setup_command="make all-local",
            cors_origins=["http://localhost:5173"],
            metadata_path=Path("data/processed/retina_metadata.jsonl"),
            index_path=Path("data/processed/index/retina_image_index.faiss"),
            index_meta_path=Path("data/processed/index/retina_image_index.json"),
            reports_dir=Path("reports"),
            image_artifact_dir=image_root,
            report_paths=lambda: {
                "final_summary": "reports/final_retina_metrics_summary.json",
                "retrieval_eval": "reports/flickr8k_full_retrieval_eval.json",
            },
        )
        self._service_metadata = {
            "config_path": str(self.settings.config_path),
            "model_name": self.settings.model_name,
            "dataset_name": self.settings.dataset_name,
            "indexed_image_count": 2,
            "indexed_caption_count": 2,
            "device": self.settings.device,
            "report_paths": self.settings.report_paths(),
            "setup_command": self.settings.setup_command,
        }

    def service_metadata(self):
        return self._service_metadata

    def health(self):
        return {"status": "ok", "ready": True, "checks": {"config": True, "metadata": True, "index": True, "index_meta": True, "images": True, "model": True}}

    def metrics_summary(self):
        return {
            "summary_path": "reports/final_retina_metrics_summary.json",
            "metrics": {"recall_at_10": 0.641925},
        }

    def recommend_text(self, query: str, top_k: int):
        return (
            [
                {
                    "image_id": "img_1",
                    "image_path": str(self.settings.image_artifact_dir / "train" / "a.jpg"),
                    "image_url": "/artifacts/images/train/a.jpg",
                    "captions": ["a dog"],
                    "score": 0.9,
                    "rank": 1,
                    "recommendation_reason": "high_clip_similarity_to_text_query",
                    "metadata": {"split": "train"},
                }
            ],
            12.34,
        )

    def recommend_image(self, image_id: str, top_k: int):
        return (
            [
                {
                    "image_id": "img_2",
                    "image_path": str(self.settings.image_artifact_dir / "train" / "b.jpg"),
                    "image_url": "/artifacts/images/train/b.jpg",
                    "captions": ["a cat"],
                    "score": 0.8,
                    "rank": 1,
                    "recommendation_reason": "high_visual_similarity_to_seed_image",
                }
            ],
            23.45,
        )

    def recommend_profile(self, text_queries, liked_image_ids, top_k: int):
        return (
            [
                {
                    "image_id": "img_1",
                    "image_path": str(self.settings.image_artifact_dir / "train" / "a.jpg"),
                    "image_url": "/artifacts/images/train/a.jpg",
                    "captions": ["a dog"],
                    "score": 0.7,
                    "rank": 1,
                    "recommendation_reason": "high_similarity_to_content_profile",
                }
            ],
            34.56,
        )


@pytest.fixture()
def image_root(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    (root / "train").mkdir(parents=True)
    (root / "train" / "a.jpg").write_bytes(b"fake-image")
    (root / "train" / "b.jpg").write_bytes(b"fake-image")
    return root


@pytest.fixture()
def client(image_root: Path) -> TestClient:
    return TestClient(create_app(service=FakeService(image_root=image_root)))


def test_health_and_metadata_endpoints(client: TestClient):
    health = client.get("/health")
    metadata = client.get("/metadata")
    assert health.status_code == 200
    assert metadata.status_code == 200
    assert HealthResponse.model_validate(health.json()).status == "ok"
    meta = MetadataResponse.model_validate(metadata.json())
    assert meta.service.model_name == "fake-model"


def test_metrics_summary_endpoint(client: TestClient):
    response = client.get("/metrics/summary")
    payload = MetricsSummaryResponse.model_validate(response.json())
    assert payload.metrics["recall_at_10"] == 0.641925


def test_text_recommendation_endpoint_with_schema_validation(client: TestClient):
    response = client.get("/recommend/text", params={"query": "a dog", "top_k": 1})
    assert response.status_code == 200
    payload = TextRecommendationResponse.model_validate(response.json())
    assert payload.latency_ms == pytest.approx(12.34)
    assert payload.results[0].image_id == "img_1"
    assert payload.results[0].image_url.endswith("/train/a.jpg")

    alias = client.post("/search/text", json={"query": "a dog", "top_k": 1})
    assert alias.status_code == 200


def test_image_recommendation_endpoint_with_schema_validation(client: TestClient):
    response = client.get("/recommend/image", params={"image_id": "img_1", "top_k": 1})
    assert response.status_code == 200
    payload = ImageRecommendationResponse.model_validate(response.json())
    assert payload.results[0].recommendation_reason == "high_visual_similarity_to_seed_image"


def test_profile_recommendation_endpoint_with_schema_validation(client: TestClient):
    response = client.post(
        "/recommend/profile",
        json={"text_queries": ["a dog"], "liked_image_ids": ["img_1"], "top_k": 1},
    )
    payload = ProfileRecommendationResponse.model_validate(response.json())
    assert payload.results[0].recommendation_reason == "high_similarity_to_content_profile"


def test_invalid_request_behavior(client: TestClient):
    response = client.get("/recommend/text", params={"query": " ", "top_k": 1})
    assert response.status_code == 422
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error.code == "invalid_request"

    invalid_top_k = client.get("/recommend/text", params={"query": "a dog", "top_k": 0})
    assert invalid_top_k.status_code == 422


def test_missing_artifact_behavior(tmp_path: Path):
    config_path = tmp_path / "retina.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "Retina"},
                "dataset": {"name": "flickr8k"},
                "model": {"name": "openai/clip-vit-base-patch32", "device": "cpu"},
                "artifacts": {
                    "metadata_path": str(tmp_path / "missing-metadata.jsonl"),
                    "index_path": str(tmp_path / "missing-index.faiss"),
                    "index_meta_path": str(tmp_path / "missing-index.json"),
                    "reports_dir": str(tmp_path / "reports"),
                    "image_dir": str(tmp_path / "images"),
                },
                "server": {"cors_origins": ["http://localhost:5173"]},
                "setup": {"command": "make all-local"},
            }
        )
    )
    with pytest.raises(RetinaMissingArtifactError):
        load_service(config_path)


def test_static_artifact_serving_and_path_safety(client: TestClient):
    ok = client.get("/artifacts/images/train/a.jpg")
    assert ok.status_code == 200
    safe_error = client.get("/artifacts/images/../../etc/passwd")
    assert safe_error.status_code in {404, 422}
