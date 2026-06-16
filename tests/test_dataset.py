from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from scripts.prepare_dataset import (
    build_canonical_rows_from_hf,
    build_report_paths,
    extract_captions_from_record,
    parse_caption_value,
    read_metadata_table,
    read_manifest,
    split_by_image,
    write_dataset_stats,
)


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


def test_parse_caption_value_handles_caption_list_strings():
    assert parse_caption_value(["one", "two"]) == ["one", "two"]
    assert parse_caption_value("['one', 'two']") == ["one", "two"]
    assert parse_caption_value("") == []


def test_extract_captions_from_record_flickr8k_style():
    record = {
        "caption_0": "a dog",
        "caption_1": "a puppy",
        "caption_2": "dog on grass",
        "caption_3": "playful dog",
        "caption_4": "dog outdoors",
    }
    assert extract_captions_from_record(record) == [
        "a dog",
        "a puppy",
        "dog on grass",
        "playful dog",
        "dog outdoors",
    ]


def test_read_metadata_table_jsonl(tmp_path):
    path = tmp_path / "meta.jsonl"
    path.write_text(
        '{"image_id":"1","image_path":"a.jpg","captions":["cap"],"split":"train","source":"flickr8k","metadata":{}}\n'
    )
    df = read_metadata_table(path)
    assert df.iloc[0]["image_id"] == 1
    assert df.iloc[0]["captions"] == ["cap"]


def test_read_metadata_table_csv(tmp_path):
    path = tmp_path / "meta.csv"
    pd.DataFrame(
        [
            {
                "image_id": "1",
                "image_path": "a.jpg",
                "captions": '["cap"]',
                "split": "train",
                "source": "flickr8k",
                "metadata": "{}",
            }
        ]
    ).to_csv(path, index=False)
    df = read_metadata_table(path)
    assert df.iloc[0]["image_id"] == 1


def test_build_report_paths_aliases():
    paths = build_report_paths(Path("reports"), "dataset_stats", "flickr8k_", "flickr8k")
    names = {path.name for path in paths}
    assert "flickr8k_dataset_stats.json" in names
    assert "dataset_stats_flickr8k.json" in names


def test_write_dataset_stats_alias_files(tmp_path):
    rows = [
        {"split": "train", "captions": ["cap1", "cap2"]},
        {"split": "val", "captions": ["cap3"]},
    ]
    reports_dir = tmp_path / "reports"
    image_dir = tmp_path / "images"
    metadata_path = tmp_path / "meta.jsonl"
    metadata_path.write_text("")
    write_dataset_stats(rows, reports_dir, "flickr8k_", "flickr8k", 2, image_dir, metadata_path, {"project": {"name": "Retina"}})
    assert (reports_dir / "flickr8k_dataset_stats.json").exists()
    assert (reports_dir / "dataset_stats_flickr8k.json").exists()


def test_build_canonical_rows_from_hf_materializes_images(tmp_path):
    image = Image.new("RGB", (32, 32), "white")
    rows = build_canonical_rows_from_hf(
        [
            {
                "image": image,
                "file_name": "sample.jpg",
                "caption_0": "cap 1",
                "caption_1": "cap 2",
                "caption_2": "cap 3",
                "caption_3": "cap 4",
                "caption_4": "cap 5",
            }
        ],
        source="flickr8k",
        image_dir=tmp_path / "artifacts",
        sample_size=1,
        seed=42,
    )
    assert rows[0]["captions"] == ["cap 1", "cap 2", "cap 3", "cap 4", "cap 5"]
    assert (tmp_path / "artifacts" / "flickr8k" / "train" / "sample.jpg").exists()
