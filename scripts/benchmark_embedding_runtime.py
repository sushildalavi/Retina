from __future__ import annotations

import argparse
import json
import statistics
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
    metadata = pd.read_csv(config["artifacts"]["metadata_path"])
    sample = metadata.head(min(16, len(metadata)))
    encoder = RetinaClipEncoder(config["model"]["name"], config["model"]["device"])

    image_times = []
    text_times = []
    for path in sample["image_path"].tolist():
        start = time.perf_counter()
        encoder.encode_image_paths([path], batch_size=1)
        image_times.append((time.perf_counter() - start) * 1000.0)
    for caption in sample["caption"].tolist():
        start = time.perf_counter()
        encoder.encode_texts([caption], batch_size=1)
        text_times.append((time.perf_counter() - start) * 1000.0)

    payload = {
        "model_name": encoder.model_name,
        "device": encoder.device,
        "image_latency_p50_ms": float(np.percentile(image_times, 50)) if image_times else 0.0,
        "image_latency_p95_ms": float(np.percentile(image_times, 95)) if image_times else 0.0,
        "text_latency_p50_ms": float(np.percentile(text_times, 50)) if text_times else 0.0,
        "text_latency_p95_ms": float(np.percentile(text_times, 95)) if text_times else 0.0,
        "image_throughput_per_sec": float(len(image_times) / (sum(image_times) / 1000.0)) if image_times else 0.0,
        "text_throughput_per_sec": float(len(text_times) / (sum(text_times) / 1000.0)) if text_times else 0.0,
    }
    reports_dir = Path(config["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "retina_runtime_benchmark.json").write_text(json.dumps(payload, indent=2))
    (reports_dir / "retina_runtime_benchmark.md").write_text(
        "# Retina Runtime Benchmark\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in payload.items())
        + "\n"
    )


if __name__ == "__main__":
    main()

