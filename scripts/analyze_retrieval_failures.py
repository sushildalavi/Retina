from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    reports_dir = Path(config["artifacts"]["reports_dir"])
    failure_path = reports_dir / "retina_retrieval_failures.jsonl"
    markdown_path = reports_dir / "retina_retrieval_failures.md"
    if not failure_path.exists():
        markdown_path.write_text("# Retina Retrieval Failures\n\nNo failures recorded yet.\n")
        return
    entries = [json.loads(line) for line in failure_path.read_text().splitlines() if line.strip()]
    lines = ["# Retina Retrieval Failures", ""]
    for i, item in enumerate(entries[:20], start=1):
        lines.append(f"## Failure {i}")
        lines.append(f"- query: {item['query_caption']}")
        lines.append(f"- expected_image_id: {item['expected_image_id']}")
        lines.append(f"- expected_image_path: {item['expected_image_path']}")
        lines.append(f"- top_results: {len(item['top_results'])}")
        lines.append("")
    markdown_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

