from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import yaml

from app.gradio_app import build_demo
from retrieval.search import RetinaSearchEngine


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    engine = RetinaSearchEngine.load(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        metadata_path=config["artifacts"]["metadata_path"],
        index_path=config["artifacts"]["index_path"],
        index_meta_path=config["artifacts"]["index_meta_path"],
    )
    demo = build_demo(engine)
    demo.launch()


if __name__ == "__main__":
    main()

