from __future__ import annotations

import argparse
import logging

import uvicorn

try:
    from _bootstrap import bootstrap, resolve_repo_path
except ImportError:  # pragma: no cover - module execution path
    from scripts._bootstrap import bootstrap, resolve_repo_path

bootstrap()

from serving.api import create_app
from serving.dependencies import load_service


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config_path = resolve_repo_path(args.config)
    service = load_service(config_path)
    app = create_app(service=service, config_path=str(config_path))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
