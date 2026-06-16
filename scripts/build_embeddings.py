from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from retrieval.clip_encoder import RetinaClipEncoder


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["artifacts"]["embeddings_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(config["artifacts"]["metadata_path"])

    encoder = RetinaClipEncoder(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
    )

    image_rows = metadata.drop_duplicates("image_id").reset_index(drop=True)
    text_rows = metadata.reset_index(drop=True)

    start = time.perf_counter()
    image_embeddings = encoder.encode_image_paths(
        image_rows["image_path"].tolist(),
        batch_size=int(config["model"]["batch_size"]),
    )
    text_embeddings = encoder.encode_texts(
        text_rows["caption"].tolist(),
        batch_size=int(config["model"]["batch_size"]),
    )
    elapsed = time.perf_counter() - start

    np.save(output_dir / "retina_image_embeddings.npy", image_embeddings)
    np.save(output_dir / "retina_text_embeddings.npy", text_embeddings)
    image_rows.to_csv(output_dir / "retina_image_metadata.csv", index=False)
    text_rows.to_csv(output_dir / "retina_text_metadata.csv", index=False)
    payload = {
        "model_name": encoder.model_name,
        "device": encoder.device,
        "image_embeddings_shape": list(image_embeddings.shape),
        "text_embeddings_shape": list(text_embeddings.shape),
        "seconds": elapsed,
    }
    (output_dir / "retina_embeddings.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

