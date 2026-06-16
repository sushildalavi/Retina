from pathlib import Path

from scripts.create_sample_dataset import main as create_sample_main


def test_create_sample_dataset_writes_manifest_and_images(tmp_path, monkeypatch):
    output_dir = tmp_path / "sample"
    manifest = tmp_path / "captions.csv"
    create_sample_main_args = [
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(manifest),
        "--count",
        "4",
    ]
    create_sample_main(create_sample_main_args)
    assert manifest.exists()
    assert len(list((output_dir / "images").glob("*.png"))) == 4

