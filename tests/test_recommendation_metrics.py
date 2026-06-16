import numpy as np

from evaluation.recommendation_metrics import (
    average_precision_at_k,
    coverage_at_k,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    summarize_recommendations,
)


def test_recommendation_metrics_basic():
    relevance = [1, 0, 0, 0]
    assert precision_at_k(relevance, 1) == 1.0
    assert recall_at_k(relevance, 1) == 1.0
    assert average_precision_at_k(relevance, 10) == 1.0
    assert ndcg_at_k(relevance, 10) == 1.0


def test_recommendation_metrics_coverage_and_diversity():
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    assert coverage_at_k([["a", "b"], ["b", "c"]], 4) == 0.75
    assert intra_list_diversity(embeddings, [0, 1]) > 0.0


def test_recommendation_summary_schema():
    summary = summarize_recommendations(
        relevances=[[1, 0, 0], [0, 1, 0]],
        latencies_ms=[1.0, 2.0],
        recommended_ids=[["a", "b", "c"], ["d", "e", "f"]],
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.25, 0.75], [0.75, 0.25]], dtype=np.float32),
        positions=[[0, 1, 2], [3, 4, 5]],
        scores=[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]],
        total_candidates=6,
    ).to_dict()
    assert "precision_at_1" in summary
    assert "ndcg_at_10" in summary
    assert "coverage_at_10" in summary
