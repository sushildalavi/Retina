from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def precision_at_k(relevance: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    items = np.asarray(list(relevance)[:k], dtype=np.float32)
    if len(items) == 0:
        return 0.0
    return float(np.mean(items > 0))


def recall_at_k(relevance: Sequence[int], k: int, total_relevant: int | None = None) -> float:
    total_relevant = int(total_relevant if total_relevant is not None else sum(int(value > 0) for value in relevance))
    if total_relevant <= 0:
        return 0.0
    hits = int(np.sum(np.asarray(list(relevance)[:k], dtype=np.float32) > 0))
    return float(hits / total_relevant)


def reciprocal_rank(relevance: Sequence[int]) -> float:
    for index, value in enumerate(relevance, start=1):
        if value:
            return float(1.0 / index)
    return 0.0


def average_precision_at_k(relevance: Sequence[int], k: int) -> float:
    values = np.asarray(list(relevance)[:k], dtype=np.float32)
    positives = int(np.sum(values > 0))
    if positives == 0:
        return 0.0
    score = 0.0
    seen = 0
    for index, value in enumerate(values, start=1):
        if value > 0:
            seen += 1
            score += seen / index
    return float(score / positives)


def dcg_at_k(relevance: Sequence[int], k: int) -> float:
    values = np.asarray(list(relevance)[:k], dtype=np.float32)
    if len(values) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(values) + 2))
    gains = (2 ** values - 1.0) / discounts
    return float(np.sum(gains))


def ndcg_at_k(relevance: Sequence[int], k: int) -> float:
    actual = dcg_at_k(relevance, k)
    ideal = dcg_at_k(sorted(relevance, reverse=True), k)
    return float(actual / ideal) if ideal > 0 else 0.0


def coverage_at_k(recommended_ids: Sequence[Sequence[str]], total_candidates: int) -> float:
    if total_candidates <= 0:
        return 0.0
    unique_ids = set()
    for row in recommended_ids:
        unique_ids.update(str(item) for item in row)
    return float(len(unique_ids) / total_candidates)


def intra_list_diversity(embeddings: np.ndarray, positions: Sequence[int]) -> float:
    valid_positions = [int(position) for position in positions if int(position) >= 0]
    if len(valid_positions) < 2:
        return 0.0
    vectors = embeddings[valid_positions]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    similarity = vectors @ vectors.T
    tri = similarity[np.triu_indices_from(similarity, k=1)]
    if len(tri) == 0:
        return 0.0
    return float(1.0 - np.mean(tri))


def mean_similarity(scores: Sequence[Sequence[float]]) -> float:
    values = [float(np.mean(row)) for row in scores if len(row)]
    return float(np.mean(values)) if values else 0.0


@dataclass
class RecommendationMetrics:
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    map_at_10: float
    ndcg_at_10: float
    coverage_at_10: float
    intra_list_diversity: float
    average_similarity_score: float
    latency_p50_ms: float
    latency_p95_ms: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def summarize_recommendations(
    relevances: Sequence[Sequence[int]],
    latencies_ms: Sequence[float],
    recommended_ids: Sequence[Sequence[str]] | None = None,
    embeddings: np.ndarray | None = None,
    positions: Sequence[Sequence[int]] | None = None,
    scores: Sequence[Sequence[float]] | None = None,
    total_candidates: int | None = None,
) -> RecommendationMetrics:
    flat_rr = [reciprocal_rank(relevance) for relevance in relevances]
    map_scores = [average_precision_at_k(relevance, 10) for relevance in relevances]
    ndcg_scores = [ndcg_at_k(relevance, 10) for relevance in relevances]
    precision_1 = [precision_at_k(relevance, 1) for relevance in relevances]
    precision_5 = [precision_at_k(relevance, 5) for relevance in relevances]
    precision_10 = [precision_at_k(relevance, 10) for relevance in relevances]
    recall_1 = [recall_at_k(relevance, 1, total_relevant=1 if relevance else 0) for relevance in relevances]
    recall_5 = [recall_at_k(relevance, 5, total_relevant=1 if relevance else 0) for relevance in relevances]
    recall_10 = [recall_at_k(relevance, 10, total_relevant=1 if relevance else 0) for relevance in relevances]

    coverage = 0.0
    diversity = 0.0
    if recommended_ids is not None and total_candidates is not None:
        coverage = coverage_at_k(recommended_ids, total_candidates)
    if embeddings is not None and positions is not None:
        diversities = [intra_list_diversity(embeddings, row) for row in positions]
        diversity = float(np.mean(diversities)) if diversities else 0.0
    similarity_score = mean_similarity(scores or [])
    return RecommendationMetrics(
        precision_at_1=float(np.mean(precision_1)) if precision_1 else 0.0,
        precision_at_5=float(np.mean(precision_5)) if precision_5 else 0.0,
        precision_at_10=float(np.mean(precision_10)) if precision_10 else 0.0,
        recall_at_1=float(np.mean(recall_1)) if recall_1 else 0.0,
        recall_at_5=float(np.mean(recall_5)) if recall_5 else 0.0,
        recall_at_10=float(np.mean(recall_10)) if recall_10 else 0.0,
        mrr=float(np.mean(flat_rr)) if flat_rr else 0.0,
        map_at_10=float(np.mean(map_scores)) if map_scores else 0.0,
        ndcg_at_10=float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        coverage_at_10=coverage,
        intra_list_diversity=diversity,
        average_similarity_score=similarity_score,
        latency_p50_ms=float(np.percentile(np.asarray(latencies_ms, dtype=np.float32), 50)) if latencies_ms else 0.0,
        latency_p95_ms=float(np.percentile(np.asarray(latencies_ms, dtype=np.float32), 95)) if latencies_ms else 0.0,
    )
