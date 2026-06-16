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
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--hf-dataset", default="")
    parser.add_argument("--sample-size", default="")
    parser.add_argument("--report-prefix", default="")
    args = parser.parse_args(argv)
    python = sys.executable
    prepare_cmd = [python, "-m", "scripts.prepare_dataset", "--config", args.config, "--dataset", args.dataset, "--report-prefix", args.report_prefix]
    if args.hf_dataset:
        prepare_cmd.extend(["--hf-dataset", args.hf_dataset])
    if args.sample_size:
        prepare_cmd.extend(["--sample-size", args.sample_size])
    run(prepare_cmd)
    run([python, "-m", "scripts.build_embeddings", "--config", args.config, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.train_query_adapter", "--config", args.config, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.build_image_text_index", "--config", args.config])
    run([python, "-m", "scripts.evaluate_retrieval", "--config", args.config, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.evaluate_recommendations", "--config", args.config, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.evaluate_random_baseline", "--config", args.config, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.analyze_retrieval_failures", "--config", args.config, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.benchmark_embedding_runtime", "--config", args.config, "--report-prefix", args.report_prefix])


if __name__ == "__main__":
    main()
