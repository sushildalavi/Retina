import json
from pathlib import Path

import yaml

from scripts.analyze_retrieval_failures import classify_failure, main as analyze_main


def test_failure_report_schema(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    failure_path = reports_dir / "retrieval_failures.jsonl"
    failure_path.write_text(
        json.dumps(
            {
                "query_caption": "cap",
                "expected_image_id": "img_1",
                "expected_image_path": "a.jpg",
                "failure_category": "no_hit",
                "top_results": [],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        "scripts.analyze_retrieval_failures.load_config",
        lambda path: {"artifacts": {"reports_dir": str(reports_dir)}},
    )
    analyze_main([])
    markdown = (reports_dir / "retrieval_failures.md").read_text()
    assert "failure_category: exact_target_missing_from_top_k" in markdown


def test_failure_category_heuristics():
    assert classify_failure("a red car on the street", []) == "exact_target_missing_from_top_k"
    assert classify_failure(
        "a red car on the street",
        [{"caption": "a blue car on the street"}],
    ) in {"color_attribute_mismatch", "object_mismatch", "scene_mismatch", "visually_similar_negative"}


def test_failure_category_generic_caption():
    assert classify_failure("a dog", [{"caption": "a dog"}]) in {"generic_caption", "object_mismatch"}
