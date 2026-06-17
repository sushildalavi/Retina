from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "model_baseline_comparison.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize local Retina model comparison artifacts.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--from-existing-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "pending",
            "sample_size": None,
            "comparison": [],
            "notes": [f"Missing comparison artifact: {path}"],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _best_by(comparison: list[dict[str, Any]], key: str, reverse: bool = True) -> dict[str, Any] | None:
    values = [row for row in comparison if row.get(key) is not None]
    if not values:
        return None
    return sorted(values, key=lambda row: row[key], reverse=reverse)[0]


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    comparison = list(payload.get("comparison", []))
    best_recall = _best_by(comparison, "recall_at_10", reverse=True)
    best_mrr = _best_by(comparison, "mrr", reverse=True)
    fastest_embeddings = _best_by(comparison, "image_embeddings_per_sec", reverse=True)
    lowest_latency = _best_by(comparison, "end_to_end_search_p95_ms", reverse=False)

    takeaways: list[str] = []
    if best_recall:
        takeaways.append(
            f"{best_recall['model_name']} had the strongest Recall@10 ({best_recall['recall_at_10']:.4f})."
        )
    if best_mrr:
        takeaways.append(f"{best_mrr['model_name']} had the strongest MRR ({best_mrr['mrr']:.4f}).")
    if fastest_embeddings:
        takeaways.append(
            f"{fastest_embeddings['model_name']} encoded images the fastest ({fastest_embeddings['image_embeddings_per_sec']:.2f} images/sec)."
        )
    if lowest_latency:
        takeaways.append(
            f"{lowest_latency['model_name']} had the lowest end-to-end p95 latency ({lowest_latency['end_to_end_search_p95_ms']:.2f} ms)."
        )

    return {
        "status": payload.get("status", "pending"),
        "sample_size": payload.get("sample_size"),
        "comparison": comparison,
        "best_recall_at_10": best_recall,
        "best_mrr": best_mrr,
        "fastest_image_embeddings": fastest_embeddings,
        "lowest_end_to_end_latency": lowest_latency,
        "takeaways": takeaways,
        "notes": payload.get("notes", []),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Retina Model Comparison",
        "",
        f"- status: {summary['status']}",
        f"- sample_size: {summary['sample_size']}",
        "",
    ]

    comparison = summary.get("comparison", [])
    if not comparison:
        lines.extend(["No comparison rows were available.", ""])
        notes = summary.get("notes", [])
        if notes:
            lines.extend(["## Notes", ""])
            for note in notes:
                lines.append(f"- {note}")
        return "\n".join(lines)

    lines.extend(
        [
            "| Model | Device | Sample | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Image embeddings/sec | End-to-end p95 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison:
        lines.append(
            "| {model_name} | {device} | {sample_size} | {recall_at_1:.4f} | {recall_at_5:.4f} | {recall_at_10:.4f} | {mrr:.4f} | {ndcg_at_10:.4f} | {image_embeddings_per_sec:.2f} | {end_to_end_search_p95_ms:.2f} ms |".format(
                **row
            )
        )

    if summary.get("takeaways"):
        lines.extend(["", "## Takeaways", ""])
        for item in summary["takeaways"]:
            lines.append(f"- {item}")

    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if args.dry_run:
        print(f"would read {input_path}")
        if args.output_json:
            print(f"would write json to {args.output_json}")
        if args.output_md:
            print(f"would write markdown to {args.output_md}")
        return

    payload = _load_payload(input_path) if args.from_existing_results else {"status": "pending", "comparison": []}
    summary = build_summary(payload)
    markdown = render_markdown(summary)

    if args.output_json:
        output_json = Path(args.output_json)
        if not output_json.is_absolute():
            output_json = ROOT / output_json
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.output_md:
        output_md = Path(args.output_md)
        if not output_md.is_absolute():
            output_md = ROOT / output_md
        output_md.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
