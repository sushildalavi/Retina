import json

from scripts.evaluate_random_baseline import main as random_baseline_main


def test_random_baseline_writes_report(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    metadata_path = processed_dir / "retina_metadata.jsonl"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image_id": "img_1",
                        "image_path": "a.jpg",
                        "captions": ["caption one"],
                        "split": "train",
                        "source": "flickr8k",
                        "metadata": {},
                    }
                ),
                json.dumps(
                    {
                        "image_id": "img_2",
                        "image_path": "b.jpg",
                        "captions": ["caption two"],
                        "split": "val",
                        "source": "flickr8k",
                        "metadata": {},
                    }
                ),
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(
        "scripts.evaluate_random_baseline.load_config",
        lambda path: {
            "project": {"name": "Retina"},
            "artifacts": {"reports_dir": str(reports_dir), "metadata_path": str(metadata_path)},
        },
    )
    random_baseline_main([])
    payload = json.loads((reports_dir / "random_baseline.json").read_text())
    assert payload["method"] == "random"
    assert payload["candidate_images"] == 2
    assert "ndcg_at_10" in payload
