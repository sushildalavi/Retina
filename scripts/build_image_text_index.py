from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from retrieval.vector_index import FaissVectorIndex


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    embeddings_dir = Path(config["artifacts"]["embeddings_dir"])
    image_embeddings = np.load(embeddings_dir / "retina_image_embeddings.npy")
    image_metadata = pd.read_csv(embeddings_dir / "retina_image_metadata.csv")
    index = FaissVectorIndex.build(image_embeddings, image_metadata["image_id"].tolist())
    index.save(config["artifacts"]["index_path"], config["artifacts"]["index_meta_path"])


if __name__ == "__main__":
    main()

