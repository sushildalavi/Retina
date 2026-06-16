from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from retrieval.clip_encoder import RetinaClipEncoder


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["artifacts"]["embeddings_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(config["artifacts"]["metadata_path"])

    encoder = RetinaClipEncoder(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
    )

    image_rows = metadata.drop_duplicates("image_id").reset_index(drop=True)
    text_rows = metadata.reset_index(drop=True)

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
        "model_name": encoder.model_name,
        "device": encoder.device,
        "image_count": image_count,
        "caption_count": caption_count,
        "embedding_dim": embedding_dim,
        "batch_size": int(config["model"]["batch_size"]),
        "runtime_seconds": elapsed,
        "image_embeddings_per_sec": image_throughput,
        "text_embeddings_per_sec": text_throughput,
    }
    reports_dir = Path(config["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "embedding_benchmark.json").write_text(json.dumps(payload, indent=2))
    (reports_dir / "embedding_benchmark.md").write_text(
        "# Retina Embedding Benchmark\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in payload.items())
        + "\n"
    )


if __name__ == "__main__":
    main()
