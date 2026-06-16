import json

from evaluation.retrieval_metrics import format_report, save_json


def test_save_json_writes_content(tmp_path):
    path = tmp_path / "report.json"
    save_json(path, {"x": 1})
    assert json.loads(path.read_text()) == {"x": 1}


def test_format_report_renders_keys():
    report = format_report("Title", {"a": 1.2345, "b": "x"})
    assert "# Title" in report
    assert "- a: 1.2345" in report
    assert "- b: x" in report

