from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    args = parser.parse_args()
    run(["python3", "scripts/prepare_dataset.py", "--config", args.config])
    run(["python3", "scripts/build_embeddings.py", "--config", args.config])
    run(["python3", "scripts/build_image_text_index.py", "--config", args.config])
    run(["python3", "scripts/evaluate_retrieval.py", "--config", args.config])
    run(["python3", "scripts/analyze_retrieval_failures.py", "--config", args.config])
    run(["python3", "scripts/benchmark_embedding_runtime.py", "--config", args.config])


if __name__ == "__main__":
    main()

