from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from evaluation.recommendation_metrics import summarize_recommendations
from retrieval.recommender import RetinaRecommender
from scripts.prepare_dataset import parse_caption_value, read_metadata_table


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _collect_caption_queries(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metadata.iterrows():
        for index, caption in enumerate(parse_caption_value(row["captions"])):
            rows.append(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "caption": caption,
                    "caption_index": index,
                    "split": row.get("split", "train"),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_text_queries(
    engine: RetinaRecommender,
    metadata: pd.DataFrame,
    top_k: int,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> dict:
    query_frame = _collect_caption_queries(metadata)
    relevances = []
    latencies = []
    recommended_ids = []
    positions = []
    scores = []
    failures = []
    for idx, row in query_frame.iterrows():
        batch_start = time.perf_counter()
        query_embeddings = text_embeddings[idx : idx + 1]
        batch_positions, batch_scores = engine.index.search(query_embeddings, top_k=top_k)
        latencies.append((time.perf_counter() - batch_start) * 1000.0)
        row_relevance = []
        row_ids = []
        row_positions = []
        row_scores = []
        matched = False
        results = []
        for rank, (pos, score) in enumerate(zip(batch_positions[0], batch_scores[0]), start=1):
            if pos < 0:
                continue
            result = engine._row_to_result(int(pos), float(score), rank, "high_clip_similarity_to_text_query")
            results.append(result)
            row_ids.append(str(result["image_id"]))
            row_positions.append(int(result["index_position"]))
            row_scores.append(float(result["score"]))
            hit = int(str(result["image_id"]) == str(row["image_id"]))
            row_relevance.append(hit)
            matched = matched or bool(hit)
        if not matched:
            failures.append(
                {
                    "query_caption": row["caption"],
                    "expected_image_id": row["image_id"],
                    "expected_image_path": row["image_path"],
                    "failure_category": "exact_target_missing_from_top_k",
                    "top_results": results,
                }
            )
        relevances.append(row_relevance)
        recommended_ids.append(row_ids)
        positions.append(row_positions)
        scores.append(row_scores)
    summary = summarize_recommendations(
        relevances,
        latencies,
        recommended_ids=recommended_ids,
        embeddings=image_embeddings,
        positions=positions,
        scores=scores,
        total_candidates=int(metadata["image_id"].nunique()),
    ).to_dict()
    summary.update(
        {
            "queries": int(len(query_frame)),
            "failures": int(len(failures)),
        }
    )
    return {"metrics": summary, "failures": failures}


def _evaluate_profile_queries(
    engine: RetinaRecommender,
    metadata: pd.DataFrame,
    top_k: int,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> dict:
    rows = []
    caption_lengths = []
    for _, row in metadata.iterrows():
        captions = parse_caption_value(row["captions"])
        if not captions:
            continue
        rows.append(row)
        caption_lengths.append(len(captions))
    relevances = []
    latencies = []
    recommended_ids = []
    positions = []
    scores = []
    query_embeddings = text_embeddings
    offset = 0
    profile_embeddings = []
    for length in caption_lengths:
        group_embeddings = query_embeddings[offset : offset + length]
        offset += length
        group_embeddings = group_embeddings[: min(3, len(group_embeddings))]
        profile_embedding = group_embeddings.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(profile_embedding, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        profile_embeddings.append(profile_embedding / norm)
    profile_embeddings = np.concatenate(profile_embeddings, axis=0) if profile_embeddings else np.zeros((0, query_embeddings.shape[1]), dtype=np.float32)
    for idx, row in enumerate(rows):
        batch_start = time.perf_counter()
        batch_positions, batch_scores = engine.index.search(profile_embeddings[idx : idx + 1], top_k=top_k)
        latencies.append((time.perf_counter() - batch_start) * 1000.0)
        row_relevance = []
        row_ids = []
        row_positions = []
        row_scores = []
        for rank, (pos, score) in enumerate(zip(batch_positions[0], batch_scores[0]), start=1):
            if pos < 0:
                continue
            result = engine._row_to_result(int(pos), float(score), rank, "high_similarity_to_content_profile")
            row_ids.append(str(result["image_id"]))
            row_positions.append(int(result["index_position"]))
            row_scores.append(float(result["score"]))
            row_relevance.append(int(str(result["image_id"]) == str(row["image_id"])))
        relevances.append(row_relevance)
        recommended_ids.append(row_ids)
        positions.append(row_positions)
        scores.append(row_scores)
    summary = summarize_recommendations(
        relevances,
        latencies,
        recommended_ids=recommended_ids,
        embeddings=image_embeddings,
        positions=positions,
        scores=scores,
        total_candidates=int(metadata["image_id"].nunique()),
    ).to_dict()
    summary.update({"queries": int(len(rows))})
    return summary


def _evaluate_image_queries(engine: RetinaRecommender, metadata: pd.DataFrame, top_k: int, image_embeddings: np.ndarray) -> dict:
    latencies = []
    similarity_scores = []
    image_rows = metadata.drop_duplicates("image_id").reset_index(drop=True)
    for idx, _ in enumerate(image_rows.itertuples(index=False)):
        batch_start = time.perf_counter()
        batch_positions, batch_scores = engine.index.search(image_embeddings[idx : idx + 1], top_k=top_k)
        latencies.append((time.perf_counter() - batch_start) * 1000.0)
        similarity_scores.append([float(score) for score in batch_scores[0]])
    payload = {
        "queries": int(len(image_rows)),
        "latency_p50_ms": float(np.percentile(np.asarray(latencies, dtype=np.float32), 50)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(np.asarray(latencies, dtype=np.float32), 95)) if latencies else 0.0,
        "average_similarity_score": float(np.mean([np.mean(row) for row in similarity_scores])) if similarity_scores else 0.0,
        "qualitative_only": True,
    }
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    parser.add_argument("--report-prefix", default="")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    metadata = read_metadata_table(config["artifacts"]["metadata_path"])
    if "captions" not in metadata.columns and "caption" in metadata.columns:
        metadata["captions"] = metadata["caption"].apply(parse_caption_value)
    elif "captions" in metadata.columns:
        metadata["captions"] = metadata["captions"].apply(parse_caption_value)

    text_embeddings = np.load(Path(config["artifacts"]["embeddings_dir"]) / "retina_text_embeddings.npy")
    image_embeddings = np.load(Path(config["artifacts"]["embeddings_dir"]) / "retina_image_embeddings.npy")
    engine = RetinaRecommender.load(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        metadata_path=config["artifacts"]["metadata_path"],
        index_path=config["artifacts"]["index_path"],
        index_meta_path=config["artifacts"]["index_meta_path"],
    )

    top_k = int(config["search"]["top_k"])
    text_eval = _evaluate_text_queries(engine, metadata, top_k, image_embeddings, text_embeddings)
    profile_eval = _evaluate_profile_queries(engine, metadata, top_k, image_embeddings, text_embeddings)
    image_eval = _evaluate_image_queries(engine, metadata, top_k, image_embeddings)

    payload = {
        "dataset_name": str(metadata["source"].iloc[0]) if "source" in metadata.columns and len(metadata) else "synthetic",
        "model_name": engine.encoder.model_name,
        "device": engine.encoder.device,
        "image_count": int(metadata["image_id"].nunique()),
        "caption_count": int(sum(len(parse_caption_value(captions)) for captions in metadata["captions"])),
        "sample_size": int(metadata["image_id"].nunique()),
        "split_counts": metadata["split"].value_counts().to_dict() if "split" in metadata.columns else {},
        "text_to_image": text_eval["metrics"],
        "profile_recommendations": profile_eval,
        "image_to_image": image_eval,
    }

    reports_dir = Path(config["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{args.report_prefix}recommendation_eval.json").write_text(json.dumps(payload, indent=2))
    md_lines = ["# Retina Recommendation Evaluation", ""]
    for key in ("dataset_name", "model_name", "device", "image_count", "caption_count", "sample_size"):
        md_lines.append(f"- {key}: {payload[key]}")
    md_lines.append("")
    md_lines.append("## Text To Image")
    for key, value in text_eval["metrics"].items():
        md_lines.append(f"- {key}: {value}")
    md_lines.append("")
    md_lines.append("## Profile Recommendations")
    for key, value in profile_eval.items():
        md_lines.append(f"- {key}: {value}")
    md_lines.append("")
    md_lines.append("## Image To Image")
    for key, value in image_eval.items():
        md_lines.append(f"- {key}: {value}")
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("- text-to-image and profile recommendations are content-based CLIP similarity results")
    md_lines.append("- image-to-image recommendations are reported as qualitative latency-only measurements")
    md_lines.append("")
    (reports_dir / f"{args.report_prefix}recommendation_eval.md").write_text("\n".join(md_lines))
    (reports_dir / f"{args.report_prefix}recommendation_failures.jsonl").write_text(
        "\n".join(json.dumps(item) for item in text_eval["failures"][:50]) + ("\n" if text_eval["failures"] else "")
    )


if __name__ == "__main__":
    main()
