from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")
    required = {"image_id", "image_path", "caption"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df[["image_id", "image_path", "caption"]].copy()


def split_by_image(df: pd.DataFrame, ratios: List[float], seed: int) -> pd.DataFrame:
    image_ids = df["image_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)
    n = len(image_ids)
    train_n = int(n * ratios[0])
    val_n = int(n * ratios[1])
    train_ids = set(image_ids[:train_n])
    val_ids = set(image_ids[train_n : train_n + val_n])
    test_ids = set(image_ids[train_n + val_n :])
    split = []
    for image_id in df["image_id"]:
        if image_id in train_ids:
            split.append("train")
        elif image_id in val_ids:
            split.append("val")
        else:
            split.append("test")
    out = df.copy()
    out["split"] = split
    return out


def save_reports(df: pd.DataFrame, output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "rows": int(len(df)),
        "images": int(df["image_id"].nunique()),
        "train_rows": int((df["split"] == "train").sum()),
        "val_rows": int((df["split"] == "val").sum()),
        "test_rows": int((df["split"] == "test").sum()),
        "config": config["project"]["name"],
    }
    (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (output_dir / "dataset_stats.md").write_text(
        "# Dataset Stats\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in stats.items())
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = Path(args.manifest or config["dataset"]["input_manifest"])
    output_dir = Path(args.output_dir or config["dataset"]["output_dir"])
    df = read_manifest(manifest_path)
    if args.limit or config["dataset"].get("sample_limit"):
        limit = args.limit or int(config["dataset"]["sample_limit"])
        df = df.head(limit)
    df = split_by_image(df, config["dataset"]["split_ratios"], int(config["dataset"]["seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "retina_metadata.csv"
    df.to_csv(metadata_path, index=False)
    save_reports(df, Path(config["artifacts"]["reports_dir"]), config)


if __name__ == "__main__":
    main()
