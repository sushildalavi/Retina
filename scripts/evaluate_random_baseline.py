from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from evaluation.retrieval_metrics import evaluate_ranks
from scripts.prepare_dataset import parse_caption_value, read_metadata_table


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    parser.add_argument("--report-prefix", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    metadata = read_metadata_table(config["artifacts"]["metadata_path"])
    if "captions" not in metadata.columns and "caption" in metadata.columns:
        metadata["captions"] = metadata["caption"].apply(parse_caption_value)
    elif "captions" in metadata.columns:
        metadata["captions"] = metadata["captions"].apply(parse_caption_value)

    query_rows = []
    for _, row in metadata.iterrows():
        for caption in parse_caption_value(row["captions"]):
            query_rows.append(
                {
                    "image_id": row["image_id"],
                    "caption": caption,
                }
            )

    n_candidates = int(metadata["image_id"].nunique())
    rng = np.random.default_rng(args.seed)
    ranks = rng.integers(1, n_candidates + 1, size=len(query_rows)).tolist()
    metrics = evaluate_ranks(ranks, [0.0] * len(ranks)).to_dict()
    ndcg_scores = [float(1.0 / np.log2(rank + 1)) if rank <= 10 else 0.0 for rank in ranks]
    payload = {
        "dataset_name": str(metadata["source"].iloc[0]) if "source" in metadata.columns and len(metadata) else "synthetic",
        "sample_size": int(metadata["image_id"].nunique()),
        "queries": int(len(query_rows)),
        "candidate_images": n_candidates,
        "seed": args.seed,
        "method": "random",
        "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        **metrics,
    }
    reports_dir = Path(config["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{args.report_prefix}random_baseline.json").write_text(json.dumps(payload, indent=2))
    md_lines = ["# Retina Random Baseline", ""]
    for key, value in payload.items():
        md_lines.append(f"- {key}: {value}")
    md_lines.append("")
    (reports_dir / f"{args.report_prefix}random_baseline.md").write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
