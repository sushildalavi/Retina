from types import SimpleNamespace

import pandas as pd

from retrieval.search import RetinaSearchEngine


class FakeEncoder:
    model_name = "fake-model"
    device = "cpu"

    def encode_texts(self, texts, batch_size=16):
        return [[1.0, 0.0]]

    def encode_image_paths(self, image_paths, batch_size=16):
        return [[1.0, 0.0]]


class FakeIndex:
    def search(self, query_embeddings, top_k=10):
        return [[0, 1]], [[0.9, 0.1]]


def test_search_engine_formats_text_results():
    engine = RetinaSearchEngine(
        encoder=FakeEncoder(),
        index=FakeIndex(),
        metadata=pd.DataFrame(
            [
                {"image_id": "img_1", "image_path": "a.jpg", "caption": "cap a"},
                {"image_id": "img_2", "image_path": "b.jpg", "caption": "cap b"},
            ]
        ),
    )
    results = engine.search_text("query", top_k=2)
    assert results[0]["image_id"] == "img_1"
    assert results[0]["rank"] == 1


def test_search_engine_formats_image_results():
    engine = RetinaSearchEngine(
        encoder=FakeEncoder(),
        index=FakeIndex(),
        metadata=pd.DataFrame(
            [
                {"image_id": "img_1", "image_path": "a.jpg", "caption": "cap a"},
                {"image_id": "img_2", "image_path": "b.jpg", "caption": "cap b"},
            ]
        ),
    )
    results = engine.search_image("a.jpg", top_k=2)
    assert results[0]["image_id"] == "img_1"

