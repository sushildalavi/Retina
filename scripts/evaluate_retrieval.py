from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
import yaml

from evaluation.retrieval_metrics import evaluate_ranks, format_report, save_json
from retrieval.search import RetinaSearchEngine
from scripts.prepare_dataset import parse_caption_value, read_metadata_table


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    parser.add_argument("--report-prefix", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = RetinaSearchEngine.load(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        metadata_path=config["artifacts"]["metadata_path"],
        index_path=config["artifacts"]["index_path"],
        index_meta_path=config["artifacts"]["index_meta_path"],
    )

    metadata = read_metadata_table(config["artifacts"]["metadata_path"])
    if "captions" not in metadata.columns and "caption" in metadata.columns:
        metadata["captions"] = metadata["caption"].apply(parse_caption_value)
    elif "captions" in metadata.columns:
        metadata["captions"] = metadata["captions"].apply(parse_caption_value)

    query_rows = []
    for _, row in metadata.iterrows():
        for index, caption in enumerate(parse_caption_value(row["captions"])):
            query_rows.append(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "caption": caption,
                    "caption_index": index,
                    "split": row.get("split", "train"),
                }
            )
    query_frame = pd.DataFrame(query_rows)
    ranks = []
    latencies = []
    failures = []
    for _, row in query_frame.iterrows():
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
                    "failure_category": "no_hit",
                    "split": row.get("split", "train"),
                    "top_results": results,
                }
            )

    metrics = evaluate_ranks(ranks, latencies).to_dict()
    metrics.update(
        {
            "dataset_name": str(metadata["source"].iloc[0]) if "source" in metadata.columns and len(metadata) else "synthetic",
            "queries": int(len(query_frame)),
            "sample_size": int(metadata["image_id"].nunique()),
            "failures": int(len(failures)),
            "images": int(metadata["image_id"].nunique()),
            "model_name": config["model"]["name"],
            "device": engine.encoder.device,
            "split_counts": metadata["split"].value_counts().to_dict() if "split" in metadata.columns else {},
        }
    )
    reports_dir = Path(config["artifacts"]["reports_dir"])
    save_json(reports_dir / f"{args.report_prefix}retrieval_eval.json", metrics)
    (reports_dir / f"{args.report_prefix}retrieval_eval.md").write_text(format_report("Retina Retrieval Evaluation", metrics))
    (reports_dir / f"{args.report_prefix}retrieval_failures.jsonl").write_text(
        "\n".join(json.dumps(item) for item in failures[:20]) + ("\n" if failures else "")
    )


if __name__ == "__main__":
    main()
