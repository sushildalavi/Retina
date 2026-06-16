from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from serving.api import create_app


class FakeEngine:
    def __init__(self):
        self.metadata = pd.DataFrame(
            [
                {"image_id": "img_1", "image_path": "images/a.jpg", "caption": "a dog", "captions": ["a dog"]},
                {"image_id": "img_2", "image_path": "images/b.jpg", "caption": "a cat", "captions": ["a cat"]},
            ]
        )
        self.encoder = SimpleNamespace(model_name="fake-model", device="cpu")

    def search_text(self, query: str, top_k: int = 10):
        return [
            {"image_id": "img_1", "image_path": "images/a.jpg", "caption": "a dog", "score": 0.9, "rank": 1}
        ]

    def recommend_text(self, query: str, top_k: int = 10):
        return self.search_text(query, top_k=top_k)

    def recommend_image(self, image_id: str | None = None, image_path: str | None = None, top_k: int = 10):
        return self.search_text("image", top_k=top_k)

    def recommend_profile(self, text_queries=None, liked_image_ids=None, top_k: int = 10):
        return self.search_text("profile", top_k=top_k)


def test_api_health_and_search():
    client = TestClient(create_app(FakeEngine()))
    assert client.get("/health").json() == {"status": "ok"}
    payload = client.post("/search/text", json={"query": "dog", "top_k": 1}).json()
    assert payload["results"][0]["image_id"] == "img_1"
    recommend_text = client.get("/recommend/text", params={"query": "dog", "top_k": 1}).json()
    assert recommend_text["results"][0]["image_id"] == "img_1"
    recommend_image = client.get("/recommend/image", params={"image_id": "img_1", "top_k": 1}).json()
    assert recommend_image["results"][0]["image_id"] == "img_1"
    recommend_profile = client.post(
        "/recommend/profile",
        json={"text_queries": ["dog"], "liked_image_ids": ["img_1"], "top_k": 1},
    ).json()
    assert recommend_profile["results"][0]["image_id"] == "img_1"
