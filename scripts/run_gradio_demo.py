from __future__ import annotations

import argparse

from app.gradio_app import build_demo
from serving.dependencies import load_service


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    args = parser.parse_args()
    service = load_service(args.config)
    demo = build_demo(service.engine)
    demo.launch()


if __name__ == "__main__":
    main()
