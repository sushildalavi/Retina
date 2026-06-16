from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from evaluation.retrieval_metrics import evaluate_ranks, format_report, save_json
from retrieval.search import RetinaSearchEngine


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = RetinaSearchEngine.load(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        metadata_path=config["artifacts"]["metadata_path"],
        index_path=config["artifacts"]["index_path"],
        index_meta_path=config["artifacts"]["index_meta_path"],
    )

    metadata = engine.metadata
    ranks = []
    latencies = []
    failures = []
    for _, row in metadata.iterrows():
        start = time.perf_counter()
        results = engine.search_text(row["caption"], top_k=int(config["search"]["top_k"]))
        latencies.append((time.perf_counter() - start) * 1000.0)
        rank = 0
        for result in results:
            if result["image_id"] == row["image_id"]:
                rank = int(result["rank"])
                break
        ranks.append(rank)
        if rank == 0:
            failures.append(
                {
                    "query_caption": row["caption"],
                    "expected_image_id": row["image_id"],
                    "expected_image_path": row["image_path"],
                    "top_results": results,
                }
            )

    metrics = evaluate_ranks(ranks, latencies).to_dict()
    metrics.update(
        {
            "queries": int(len(metadata)),
            "failures": int(len(failures)),
        }
    )
    reports_dir = Path(config["artifacts"]["reports_dir"])
    save_json(reports_dir / "retina_retrieval_eval.json", metrics)
    (reports_dir / "retina_retrieval_eval.md").write_text(format_report("Retina Retrieval Evaluation", metrics))
    (reports_dir / "retina_retrieval_failures.jsonl").write_text(
        "\n".join(json.dumps(item) for item in failures[:20]) + ("\n" if failures else "")
    )


if __name__ == "__main__":
    main()

