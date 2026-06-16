from evaluation.retrieval_metrics import evaluate_ranks, mean_reciprocal_rank, recall_at_k
from evaluation.retrieval_metrics import median_rank, percentile


def test_retrieval_metrics_basic():
    ranks = [1, 2, 0, 5]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.75
    assert mean_reciprocal_rank(ranks) > 0.0
    metrics = evaluate_ranks(ranks, [10.0, 20.0, 30.0, 40.0])
    assert metrics.recall_at_10 == 0.75
    assert metrics.latency_p50_ms == 25.0


def test_retrieval_metrics_handles_empty_input():
    metrics = evaluate_ranks([], [])
    assert metrics.recall_at_1 == 0.0
    assert metrics.mrr == 0.0
    assert metrics.median_rank == 0.0


def test_percentile_and_median_rank():
    assert median_rank([1, 3, 5]) == 3.0
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0
