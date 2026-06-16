from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    args = parser.parse_args(argv)
    python = sys.executable
    run([python, "-m", "scripts.prepare_dataset", "--config", args.config])
    run([python, "-m", "scripts.build_embeddings", "--config", args.config])
    run([python, "-m", "scripts.build_image_text_index", "--config", args.config])
    run([python, "-m", "scripts.evaluate_retrieval", "--config", args.config])
    run([python, "-m", "scripts.analyze_retrieval_failures", "--config", args.config])
    run([python, "-m", "scripts.benchmark_embedding_runtime", "--config", args.config])


if __name__ == "__main__":
    main()
