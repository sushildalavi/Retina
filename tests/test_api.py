from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from serving.api import create_app


class FakeEngine:
    def __init__(self):
        self.metadata = pd.DataFrame(
            [
                {"image_id": "img_1", "image_path": "images/a.jpg", "caption": "a dog"},
                {"image_id": "img_2", "image_path": "images/b.jpg", "caption": "a cat"},
            ]
        )
        self.encoder = SimpleNamespace(model_name="fake-model", device="cpu")

    def search_text(self, query: str, top_k: int = 10):
        return [
            {"image_id": "img_1", "image_path": "images/a.jpg", "caption": "a dog", "score": 0.9, "rank": 1}
        ]


def test_api_health_and_search():
    client = TestClient(create_app(FakeEngine()))
    assert client.get("/health").json() == {"status": "ok"}
    payload = client.post("/search/text", json={"query": "dog", "top_k": 1}).json()
    assert payload["results"][0]["image_id"] == "img_1"

