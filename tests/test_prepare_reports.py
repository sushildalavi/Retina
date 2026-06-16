import json

import pandas as pd

from scripts.prepare_dataset import save_reports


def test_save_reports_writes_dataset_stats(tmp_path):
    df = pd.DataFrame(
        [
            {"image_id": "a", "image_path": "a.jpg", "caption": "cap a", "split": "train"},
            {"image_id": "b", "image_path": "b.jpg", "caption": "cap b", "split": "val"},
        ]
    )
    reports_dir = tmp_path / "reports"
    save_reports(df, reports_dir, {"project": {"name": "Retina"}})
    stats = json.loads((reports_dir / "dataset_stats.json").read_text())
    assert stats["rows"] == 2
    assert "Dataset Stats" in (reports_dir / "dataset_stats.md").read_text()

