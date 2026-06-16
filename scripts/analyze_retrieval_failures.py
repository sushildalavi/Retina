from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


COLOR_WORDS = {
    "red",
    "green",
    "blue",
    "yellow",
    "black",
    "white",
    "brown",
    "gray",
    "grey",
    "orange",
    "purple",
    "pink",
}

ACTION_WORDS = {
    "running",
    "jumping",
    "riding",
    "standing",
    "sitting",
    "holding",
    "walking",
    "playing",
    "eating",
    "looking",
}

STOP_WORDS = {"a", "an", "the", "of", "on", "in", "with", "and", "to", "for", "at", "is", "are", "was", "were"}

SCENE_WORDS = {
    "beach",
    "street",
    "park",
    "snow",
    "water",
    "field",
    "kitchen",
    "mountain",
    "indoors",
    "outdoors",
    "city",
    "room",
}


def classify_failure(query_caption: str, top_results: list[dict]) -> str:
    text = query_caption.lower()
    top_text = " ".join(str(item.get("caption", "")) for item in top_results).lower()
    tokens = {token.strip(".,;:!?") for token in text.split()}
    if not top_results:
        return "exact_target_missing_from_top_k"
    content_tokens = [token for token in tokens if token not in STOP_WORDS]
    if len(content_tokens) <= 2:
        return "generic_caption"
    if tokens & COLOR_WORDS and not (set(top_text.split()) & COLOR_WORDS):
        return "attribute_mismatch"
    if tokens & ACTION_WORDS and not (set(top_text.split()) & ACTION_WORDS):
        return "action_mismatch"
    if tokens & SCENE_WORDS and not (set(top_text.split()) & SCENE_WORDS):
        return "scene_mismatch"
    overlap_counts = []
    for item in top_results[:3]:
        result_tokens = {token.strip(".,;:!?") for token in str(item.get("caption", "")).lower().split()}
        overlap_counts.append(len(result_tokens & set(content_tokens)))
    if len(overlap_counts) >= 2 and overlap_counts[0] >= 2 and overlap_counts[1] >= 2:
        return "multiple_valid_matches"
    if any(token in top_text for token in tokens if len(token) > 3):
        return "visually_similar_negative"
    if len(content_tokens) > 5:
        return "ambiguous_caption"
    return "object_mismatch"


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    parser.add_argument("--report-prefix", default="")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    reports_dir = Path(config["artifacts"]["reports_dir"])
    failure_path = reports_dir / f"{args.report_prefix}retrieval_failures.jsonl"
    markdown_path = reports_dir / f"{args.report_prefix}retrieval_failures.md"
    if not failure_path.exists():
        markdown_path.write_text("# Retina Retrieval Failures\n\nNo failures recorded yet.\n")
        return
    entries = [json.loads(line) for line in failure_path.read_text().splitlines() if line.strip()]
    lines = ["# Retina Retrieval Failures", ""]
    counts = {}
    for item in entries:
        category = classify_failure(item["query_caption"], item.get("top_results", []))
        counts[category] = counts.get(category, 0) + 1
    lines.extend(["## Summary", ""])
    for key, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {key}: {value}")
    lines.append("")
    for i, item in enumerate(entries[:50], start=1):
        category = classify_failure(item["query_caption"], item.get("top_results", []))
        note = "likely visually similar negative" if category == "visually_similar_negative" else "conservative heuristic classification"
        lines.append(f"## Failure {i}")
        lines.append(f"- query: {item['query_caption']}")
        lines.append(f"- expected_image_id: {item['expected_image_id']}")
        lines.append(f"- expected_image_path: {item['expected_image_path']}")
        lines.append(f"- failure_category: {category}")
        lines.append(f"- conservative_note: {note}")
        lines.append(f"- retrieved_image_ids: {item.get('retrieved_image_ids', [])}")
        lines.append(f"- retrieved_image_paths: {item.get('retrieved_image_paths', [])}")
        lines.append(f"- retrieved_captions: {item.get('retrieved_captions', [])}")
        lines.append(f"- scores: {item.get('scores', [])}")
        lines.append(f"- rank_of_correct_image: {item.get('rank_of_correct_image')}")
        lines.append("")
    markdown_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
