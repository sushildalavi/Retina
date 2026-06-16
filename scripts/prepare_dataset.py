from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Tuple

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


def read_metadata_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def parse_caption_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    items = parsed
                else:
                    items = [text]
            except Exception:
                items = [text]
        else:
            items = [text]
    else:
        items = [value]
    captions = [str(item).strip() for item in items if str(item).strip()]
    return captions


def extract_captions_from_record(record: dict) -> List[str]:
    if "captions" in record and record["captions"] is not None:
        captions = parse_caption_value(record["captions"])
    else:
        captions = []
        for key in sorted(record):
            if key.startswith("caption_"):
                captions.extend(parse_caption_value(record[key]))
        if not captions and "caption" in record:
            captions.extend(parse_caption_value(record["caption"]))
    return captions


def build_canonical_rows_from_manifest(df: pd.DataFrame, source: str) -> List[dict]:
    rows: List[dict] = []
    for row in df.to_dict(orient="records"):
        captions = parse_caption_value(row.get("caption"))
        rows.append(
            {
                "image_id": str(row["image_id"]),
                "image_path": str(row["image_path"]),
                "captions": captions,
                "split": str(row.get("split", "train")),
                "source": source,
                "metadata": {
                    "caption_count": len(captions),
                },
            }
        )
    return rows


def _coerce_sample_size(sample_size: Any, default: int | None = None) -> int | str | None:
    if sample_size is None:
        return default
    if isinstance(sample_size, str):
        text = sample_size.strip().lower()
        if not text:
            return default
        if text == "full":
            return "full"
        return int(text)
    return int(sample_size)


def build_canonical_rows_from_hf(
    records: Iterable[dict] | Any,
    source: str,
    image_dir: Path,
    sample_size: int | str,
    seed: int,
) -> List[dict]:
    rows: List[dict] = []
    rng = np.random.default_rng(seed)
    if hasattr(records, "items") and not isinstance(records, list):
        flattened: List[dict] = []
        for split_name, split_records in records.items():
            for record in split_records:
                item = dict(record)
                item["_hf_split"] = split_name
                flattened.append(item)
        rng.shuffle(flattened)
        iterable = enumerate(flattened)
    else:
        iterable = enumerate(records)
    accepted = 0
    target = None if sample_size == "full" else int(sample_size)
    for index, record in iterable:
        if target is not None and accepted >= target:
            break
        captions = extract_captions_from_record(record)
        if not captions:
            continue
        image = record.get("image")
        if image is None:
            continue
        file_name = record.get("file_name") or f"{accepted:06d}.jpg"
        suffix = Path(file_name).suffix or ".jpg"
        split_name = str(record.get("split") or record.get("_hf_split") or "train")
        dataset_dir = image_dir / source / split_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        image_id = Path(file_name).stem or f"{source}_{accepted:06d}"
        local_path = dataset_dir / f"{image_id}{suffix}"
        if not local_path.exists():
            image.save(local_path)
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(local_path),
                "captions": captions,
                "split": split_name,
                "source": source,
                "metadata": {
                    "file_name": file_name,
                    "caption_count": len(captions),
                    "source_split": split_name,
                    "sample_index": index,
                },
            }
        )
        accepted += 1
    if target is not None and accepted < target:
        raise RuntimeError(f"Only collected {accepted} rows out of requested {target}")
    return rows


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


def build_report_paths(output_dir: Path, base_name: str, report_prefix: str = "", dataset_name: str | None = None) -> List[Path]:
    paths = [output_dir / f"{report_prefix}{base_name}.json", output_dir / f"{report_prefix}{base_name}.md"]
    if dataset_name and report_prefix and base_name == "dataset_stats":
        paths.extend(
            [
                output_dir / f"{base_name}_{dataset_name}.json",
                output_dir / f"{base_name}_{dataset_name}.md",
            ]
        )
    return paths


def write_dataset_stats(
    rows: List[dict],
    output_dir: Path,
    report_prefix: str,
    dataset_name: str,
    sample_size: int | str,
    image_dir: Path,
    metadata_path: Path,
    config: dict,
) -> None:
    counts = Counter(row["split"] for row in rows)
    caption_count = sum(len(row["captions"]) for row in rows)
    payload = {
        "dataset_name": dataset_name,
        "sample_size": sample_size,
        "requested_sample_size": sample_size,
        "images": len(rows),
        "captions": caption_count,
        "average_captions_per_image": float(caption_count / len(rows)) if rows else 0.0,
        "split_counts": dict(counts),
        "metadata_path": str(metadata_path),
        "image_artifact_path": str(image_dir),
        "reports_dir": str(output_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source": dataset_name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report_prefix}dataset_stats.json"
    md_path = output_dir / f"{report_prefix}dataset_stats.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_lines = ["# Dataset Stats", ""]
    for key, value in payload.items():
        if isinstance(value, dict):
            md_lines.append(f"- {key}: {json.dumps(value, sort_keys=True)}")
        else:
            md_lines.append(f"- {key}: {value}")
    md_lines.append("")
    md_path.write_text("\n".join(md_lines))
    if report_prefix and dataset_name == "flickr8k":
        (output_dir / f"dataset_stats_{dataset_name}.json").write_text(json.dumps(payload, indent=2))
        (output_dir / f"dataset_stats_{dataset_name}.md").write_text("\n".join(md_lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "hf_flickr8k", "hf_flickr30k"])
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--sample-size", default=None)
    parser.add_argument("--report-prefix", default="")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["dataset"]["output_dir"])
    reports_dir = Path(config["artifacts"]["reports_dir"])
    metadata_jsonl = output_dir / "retina_metadata.jsonl"
    metadata_csv = output_dir / "retina_metadata.csv"
    image_dir = Path("data/artifacts/images")

    if args.dataset == "synthetic":
        manifest_path = Path(args.manifest or config["dataset"]["input_manifest"])
        df = read_manifest(manifest_path)
        if args.limit or config["dataset"].get("sample_limit"):
            limit = args.limit or int(config["dataset"]["sample_limit"])
            df = df.head(limit)
        df = split_by_image(df, config["dataset"]["split_ratios"], int(config["dataset"]["seed"]))
        canonical_rows = build_canonical_rows_from_manifest(df, source="synthetic")
        metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)
        metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        with metadata_jsonl.open("w") as f:
            for row in canonical_rows:
                f.write(json.dumps(row) + "\n")
        pd.DataFrame(
            [
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "caption": row["captions"][0] if row["captions"] else "",
                    "captions": json.dumps(row["captions"]),
                    "split": row["split"],
                    "source": row["source"],
                    "metadata": json.dumps(row["metadata"]),
                }
                for row in canonical_rows
            ]
        ).to_csv(metadata_csv, index=False)
        save_reports(df, reports_dir, config)
        return

    hf_dataset = args.hf_dataset
    if not hf_dataset:
        raise ValueError("--hf-dataset is required for real HF datasets")
    requested_sample_size = _coerce_sample_size(args.sample_size, int(config["dataset"].get("sample_limit") or 500))
    from datasets import load_dataset

    dataset = load_dataset(hf_dataset)
    canonical_rows = build_canonical_rows_from_hf(
        dataset,
        source=args.dataset.replace("hf_", ""),
        image_dir=image_dir,
        sample_size=requested_sample_size,
        seed=int(config["dataset"]["seed"]),
    )
    canonical_df = pd.DataFrame(canonical_rows)
    metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with metadata_jsonl.open("w") as f:
        for row in canonical_df.to_dict(orient="records"):
            f.write(json.dumps(row) + "\n")
    pd.DataFrame(
        [
            {
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "caption": row["captions"][0] if row["captions"] else "",
                "captions": json.dumps(row["captions"]),
                "split": row["split"],
                "source": row["source"],
                "metadata": json.dumps(row["metadata"]),
            }
            for row in canonical_df.to_dict(orient="records")
        ]
    ).to_csv(metadata_csv, index=False)
    write_dataset_stats(
        canonical_df.to_dict(orient="records"),
        reports_dir,
        args.report_prefix,
        dataset_name=args.dataset.replace("hf_", ""),
        sample_size=requested_sample_size if requested_sample_size is not None else len(canonical_rows),
        image_dir=image_dir,
        metadata_path=metadata_jsonl,
        config=config,
    )


if __name__ == "__main__":
    main()
