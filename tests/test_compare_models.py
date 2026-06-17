from __future__ import annotations

from scripts.compare_models import build_summary, render_markdown


def test_build_summary_selects_best_rows():
    payload = {
        "status": "measured",
        "sample_size": 500,
        "comparison": [
            {
                "model_name": "base",
                "device": "mps",
                "sample_size": 500,
                "recall_at_1": 0.64,
                "recall_at_5": 0.87,
                "recall_at_10": 0.93,
                "mrr": 0.74,
                "ndcg_at_10": 0.78,
                "image_embeddings_per_sec": 58.07,
                "end_to_end_search_p95_ms": 14.53,
            },
            {
                "model_name": "large",
                "device": "mps",
                "sample_size": 500,
                "recall_at_1": 0.66,
                "recall_at_5": 0.90,
                "recall_at_10": 0.95,
                "mrr": 0.76,
                "ndcg_at_10": 0.81,
                "image_embeddings_per_sec": 2.43,
                "end_to_end_search_p95_ms": 114.11,
            },
        ],
    }

    summary = build_summary(payload)
    assert summary["best_recall_at_10"]["model_name"] == "large"
    assert summary["fastest_image_embeddings"]["model_name"] == "base"
    assert "strongest Recall@10" in summary["takeaways"][0]


def test_render_markdown_includes_table():
    markdown = render_markdown(
        {
            "status": "measured",
            "sample_size": 500,
            "comparison": [
                {
                    "model_name": "base",
                    "device": "mps",
                    "sample_size": 500,
                    "recall_at_1": 0.64,
                    "recall_at_5": 0.87,
                    "recall_at_10": 0.93,
                    "mrr": 0.74,
                    "ndcg_at_10": 0.78,
                    "image_embeddings_per_sec": 58.07,
                    "end_to_end_search_p95_ms": 14.53,
                }
            ],
            "takeaways": [],
        }
    )
    assert "Retina Model Comparison" in markdown
    assert "image_embeddings_per_sec" not in markdown
    assert "58.07" in markdown
