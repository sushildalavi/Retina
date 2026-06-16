from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from _bootstrap import bootstrap, resolve_repo_path
except ImportError:  # pragma: no cover - module execution path
    from scripts._bootstrap import bootstrap, resolve_repo_path

REPO_ROOT = bootstrap()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--hf-dataset", default="")
    parser.add_argument("--sample-size", default="")
    parser.add_argument("--report-prefix", default="")
    args = parser.parse_args(argv)
    python = sys.executable
    config_path = str(resolve_repo_path(args.config))
    prepare_cmd = [
        python,
        "-m",
        "scripts.prepare_dataset",
        "--config",
        config_path,
        "--dataset",
        args.dataset,
        "--report-prefix",
        args.report_prefix,
    ]
    if args.hf_dataset:
        prepare_cmd.extend(["--hf-dataset", args.hf_dataset])
    if args.sample_size:
        prepare_cmd.extend(["--sample-size", args.sample_size])
    run(prepare_cmd)
    run([python, "-m", "scripts.build_embeddings", "--config", config_path, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.train_query_adapter", "--config", config_path, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.build_image_text_index", "--config", config_path])
    run([python, "-m", "scripts.evaluate_retrieval", "--config", config_path, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.evaluate_recommendations", "--config", config_path, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.evaluate_random_baseline", "--config", config_path, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.analyze_retrieval_failures", "--config", config_path, "--report-prefix", args.report_prefix])
    run([python, "-m", "scripts.benchmark_embedding_runtime", "--config", config_path, "--report-prefix", args.report_prefix])


if __name__ == "__main__":
    main()
