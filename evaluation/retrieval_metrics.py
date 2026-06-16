from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import json
import math

import numpy as np


def recall_at_k(ranks: Sequence[int], k: int) -> float:
    ranks = np.asarray(ranks, dtype=np.int32)
    if len(ranks) == 0:
        return 0.0
    hits = (ranks > 0) & (ranks <= k)
    return float(np.mean(hits))


def mean_reciprocal_rank(ranks: Sequence[int]) -> float:
    ranks = np.asarray(ranks, dtype=np.float32)
    if len(ranks) == 0:
        return 0.0
    reciprocal = np.zeros_like(ranks, dtype=np.float32)
    valid = ranks > 0
    reciprocal[valid] = 1.0 / ranks[valid]
    return float(np.mean(reciprocal))


def median_rank(ranks: Sequence[int]) -> float:
    ranks = np.asarray(ranks, dtype=np.float32)
    valid = ranks[ranks > 0]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


def percentile(values: Sequence[float], q: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


@dataclass
class RetrievalEvaluation:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    median_rank: float
    latency_p50_ms: float
    latency_p95_ms: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate_ranks(ranks: Sequence[int], latencies_ms: Sequence[float]) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        recall_at_1=recall_at_k(ranks, 1),
        recall_at_5=recall_at_k(ranks, 5),
        recall_at_10=recall_at_k(ranks, 10),
        mrr=mean_reciprocal_rank(ranks),
        median_rank=median_rank(ranks),
        latency_p50_ms=percentile(latencies_ms, 50),
        latency_p95_ms=percentile(latencies_ms, 95),
    )


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def format_report(title: str, metrics: dict) -> str:
    lines = [f"# {title}", ""]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)
