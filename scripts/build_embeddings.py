from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from retrieval.clip_encoder import RetinaClipEncoder
from scripts.prepare_dataset import parse_caption_value, read_metadata_table


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    parser.add_argument("--report-prefix", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["artifacts"]["embeddings_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata_table(config["artifacts"]["metadata_path"])
    if "captions" not in metadata.columns and "caption" in metadata.columns:
        metadata["captions"] = metadata["caption"].apply(parse_caption_value)
    elif "captions" in metadata.columns:
        metadata["captions"] = metadata["captions"].apply(parse_caption_value)

    encoder = RetinaClipEncoder(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
    )

    image_rows = metadata.drop_duplicates("image_id").reset_index(drop=True)
    query_rows = []
    for _, row in metadata.iterrows():
        captions = parse_caption_value(row["captions"])
        for index, caption in enumerate(captions):
            query_rows.append(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "caption": caption,
                    "caption_index": index,
                    "split": row.get("split", "train"),
                    "source": row.get("source", "unknown"),
                }
            )
    text_rows = pd.DataFrame(query_rows)

    start = time.perf_counter()
    image_embeddings = encoder.encode_image_paths(
        image_rows["image_path"].tolist(),
        batch_size=int(config["model"]["batch_size"]),
    )
    text_embeddings = encoder.encode_texts(
        text_rows["caption"].tolist(),
        batch_size=int(config["model"]["batch_size"]),
    )
    elapsed = time.perf_counter() - start

    image_count = int(len(image_rows))
    caption_count = int(len(text_rows))
    image_throughput = float(image_count / elapsed) if elapsed else 0.0
    text_throughput = float(caption_count / elapsed) if elapsed else 0.0
    embedding_dim = int(image_embeddings.shape[1]) if image_embeddings.size else 0

    np.save(output_dir / "retina_image_embeddings.npy", image_embeddings)
    np.save(output_dir / "retina_text_embeddings.npy", text_embeddings)
    image_rows.to_csv(output_dir / "retina_image_metadata.csv", index=False)
    text_rows.to_csv(output_dir / "retina_text_metadata.csv", index=False)
    payload = {
        "dataset_name": str(metadata["source"].iloc[0]) if "source" in metadata.columns and len(metadata) else "synthetic",
        "model_name": encoder.model_name,
        "device": encoder.device,
        "image_count": image_count,
        "caption_count": caption_count,
        "sample_size": image_count,
        "split_counts": metadata["split"].value_counts().to_dict() if "split" in metadata.columns else {},
        "embedding_dim": embedding_dim,
        "batch_size": int(config["model"]["batch_size"]),
        "runtime_seconds": elapsed,
        "image_embeddings_per_sec": image_throughput,
        "text_embeddings_per_sec": text_throughput,
    }
    reports_dir = Path(config["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{args.report_prefix}embedding_benchmark.json").write_text(json.dumps(payload, indent=2))
    (reports_dir / f"{args.report_prefix}embedding_benchmark.md").write_text(
        "# Retina Embedding Benchmark\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in payload.items())
        + "\n"
    )


if __name__ == "__main__":
    main()
