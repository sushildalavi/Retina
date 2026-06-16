from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from retrieval.clip_encoder import RetinaClipEncoder
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
    metadata = read_metadata_table(config["artifacts"]["metadata_path"])
    if "captions" not in metadata.columns and "caption" in metadata.columns:
        metadata["captions"] = metadata["caption"].apply(parse_caption_value)
    elif "captions" in metadata.columns:
        metadata["captions"] = metadata["captions"].apply(parse_caption_value)
    sample = metadata.head(min(16, len(metadata)))
    encoder = RetinaClipEncoder(config["model"]["name"], config["model"]["device"])

    image_times = []
    text_times = []
    for path in sample["image_path"].tolist():
        start = time.perf_counter()
        encoder.encode_image_paths([path], batch_size=1)
        image_times.append((time.perf_counter() - start) * 1000.0)
    sample_queries = []
    for _, row in sample.iterrows():
        sample_queries.extend(parse_caption_value(row["captions"]))
    sample_queries = sample_queries[:16]
    for caption in sample_queries:
        start = time.perf_counter()
        encoder.encode_texts([caption], batch_size=1)
        text_times.append((time.perf_counter() - start) * 1000.0)

    engine = RetinaSearchEngine.load(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        metadata_path=config["artifacts"]["metadata_path"],
        index_path=config["artifacts"]["index_path"],
        index_meta_path=config["artifacts"]["index_meta_path"],
    )
    search_times = []
    for caption in sample_queries:
        start = time.perf_counter()
        engine.search_text(caption, top_k=int(config["search"]["top_k"]))
        search_times.append((time.perf_counter() - start) * 1000.0)

    payload = {
        "dataset_name": str(metadata["source"].iloc[0]) if "source" in metadata.columns and len(metadata) else "synthetic",
        "model_name": encoder.model_name,
        "device": encoder.device,
        "sample_size": int(len(sample)),
        "caption_sample_size": int(len(sample_queries)),
        "split_counts": metadata["split"].value_counts().to_dict() if "split" in metadata.columns else {},
        "image_latency_p50_ms": float(np.percentile(image_times, 50)) if image_times else 0.0,
        "image_latency_p95_ms": float(np.percentile(image_times, 95)) if image_times else 0.0,
        "text_latency_p50_ms": float(np.percentile(text_times, 50)) if text_times else 0.0,
        "text_latency_p95_ms": float(np.percentile(text_times, 95)) if text_times else 0.0,
        "image_throughput_per_sec": float(len(image_times) / (sum(image_times) / 1000.0)) if image_times else 0.0,
        "text_throughput_per_sec": float(len(text_times) / (sum(text_times) / 1000.0)) if text_times else 0.0,
        "search_queries_per_sec": float(len(search_times) / (sum(search_times) / 1000.0)) if search_times else 0.0,
        "search_latency_p50_ms": float(np.percentile(search_times, 50)) if search_times else 0.0,
        "search_latency_p95_ms": float(np.percentile(search_times, 95)) if search_times else 0.0,
    }
    reports_dir = Path(config["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{args.report_prefix}runtime_benchmark.json").write_text(json.dumps(payload, indent=2))
    (reports_dir / f"{args.report_prefix}runtime_benchmark.md").write_text(
        "# Retina Runtime Benchmark\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in payload.items())
        + "\n"
    )


if __name__ == "__main__":
    main()
