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
    text_embeddings = np.load(Path(config["artifacts"]["embeddings_dir"]) / "retina_text_embeddings.npy")
    ranks = []
    latencies = []
    failures = []
    top_k = int(config["search"]["top_k"])
    for idx, row in query_frame.iterrows():
        batch_start = time.perf_counter()
        query_embedding = text_embeddings[idx : idx + 1]
        positions, scores = engine.index.search(query_embedding, top_k=top_k)
        latencies.append((time.perf_counter() - batch_start) * 1000.0)
        result_positions = positions[0]
        result_scores = scores[0]
        results = []
        rank = 0
        for rank_index, (pos, score) in enumerate(zip(result_positions, result_scores), start=1):
            if pos < 0:
                continue
            result = engine._row_to_result(int(pos), float(score), rank_index, "high_clip_similarity_to_text_query")
            results.append(result)
            if str(result["image_id"]) == str(row["image_id"]):
                rank = int(rank_index)
        ranks.append(rank)
        if rank == 0:
            retrieved_image_ids = [str(result.get("image_id")) for result in results]
            retrieved_image_paths = [str(result.get("image_path")) for result in results]
            retrieved_captions = [str(result.get("caption", "")) for result in results]
            retrieved_scores = [float(result.get("score", 0.0)) for result in results]
            failures.append(
                {
                    "query_caption": row["caption"],
                    "expected_image_id": row["image_id"],
                    "expected_image_path": row["image_path"],
                    "retrieved_image_ids": retrieved_image_ids,
                    "retrieved_image_paths": retrieved_image_paths,
                    "retrieved_captions": retrieved_captions,
                    "scores": retrieved_scores,
                    "rank_of_correct_image": None,
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
        "\n".join(json.dumps(item) for item in failures[:50]) + ("\n" if failures else "")
    )


if __name__ == "__main__":
    main()
