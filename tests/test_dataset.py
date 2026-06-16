from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_dataset import read_manifest, split_by_image


def test_read_manifest_csv(tmp_path):
    path = tmp_path / "captions.csv"
    pd.DataFrame(
        [
            {"image_id": "a", "image_path": "a.jpg", "caption": "cap a"},
            {"image_id": "b", "image_path": "b.jpg", "caption": "cap b"},
        ]
    ).to_csv(path, index=False)
    df = read_manifest(path)
    assert list(df.columns) == ["image_id", "image_path", "caption"]
    assert len(df) == 2


def test_read_manifest_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"image_id": "a", "caption": "cap"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        read_manifest(path)


def test_split_by_image_is_reproducible():
    df = pd.DataFrame(
        [
            {"image_id": "a", "image_path": "a.jpg", "caption": "cap a"},
            {"image_id": "b", "image_path": "b.jpg", "caption": "cap b"},
            {"image_id": "c", "image_path": "c.jpg", "caption": "cap c"},
            {"image_id": "d", "image_path": "d.jpg", "caption": "cap d"},
        ]
    )
    left = split_by_image(df, [0.5, 0.25, 0.25], 42)
    right = split_by_image(df, [0.5, 0.25, 0.25], 42)
    assert left["split"].tolist() == right["split"].tolist()


def test_split_by_image_covers_all_rows():
    df = pd.DataFrame(
        [
            {"image_id": "a", "image_path": "a.jpg", "caption": "cap a"},
            {"image_id": "b", "image_path": "b.jpg", "caption": "cap b"},
            {"image_id": "c", "image_path": "c.jpg", "caption": "cap c"},
            {"image_id": "d", "image_path": "d.jpg", "caption": "cap d"},
            {"image_id": "e", "image_path": "e.jpg", "caption": "cap e"},
        ]
    )
    split = split_by_image(df, [0.6, 0.2, 0.2], 7)
    assert set(split["split"]) == {"train", "val", "test"}
