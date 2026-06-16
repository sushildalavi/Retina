from types import SimpleNamespace

import pandas as pd

from app.gradio_app import build_demo


class FakeEngine:
    def __init__(self):
        self.metadata = pd.DataFrame(
            [{"image_id": "img_1", "image_path": "images/a.jpg", "caption": "a dog"}]
        )

    def search_text(self, query: str, top_k: int = 10):
        return [
            {"image_id": "img_1", "image_path": "images/a.jpg", "caption": "a dog", "score": 0.9, "rank": 1}
        ]


def test_gradio_demo_builds():
    demo = build_demo(FakeEngine())
    assert demo is not None

