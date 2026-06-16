from evaluation.retrieval_metrics import evaluate_ranks, mean_reciprocal_rank, recall_at_k


def test_retrieval_metrics_basic():
    ranks = [1, 2, 0, 5]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.75
    assert mean_reciprocal_rank(ranks) > 0.0
    metrics = evaluate_ranks(ranks, [10.0, 20.0, 30.0, 40.0])
    assert metrics.recall_at_10 == 0.75
    assert metrics.latency_p50_ms == 25.0

