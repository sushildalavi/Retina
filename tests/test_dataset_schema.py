import csv
from pathlib import Path


def test_example_schema_columns():
    with Path("examples/local_captions_schema.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["image_id", "image_path", "caption"]
        row = next(reader)
        assert row["image_id"] == "img_001"

