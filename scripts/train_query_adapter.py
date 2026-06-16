from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from evaluation.retrieval_metrics import evaluate_ranks
from training.query_adapter import QueryAdapter, save_query_adapter


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _parse_split_frame(text_metadata: pd.DataFrame) -> pd.DataFrame:
    frame = text_metadata.copy().reset_index(drop=True)
    if "split" not in frame.columns:
        frame["split"] = "train"
    return frame


def _resolve_image_lookup(image_metadata: pd.DataFrame) -> dict[str, int]:
    if "image_id" not in image_metadata.columns:
        raise ValueError("image metadata must include image_id")
    return {str(image_id): idx for idx, image_id in enumerate(image_metadata["image_id"].astype(str).tolist())}


def _select_indices(frame: pd.DataFrame, split: str) -> np.ndarray:
    if "split" not in frame.columns:
        return np.arange(len(frame), dtype=np.int64)
    return frame.index[frame["split"].astype(str) == split].to_numpy(dtype=np.int64)


def _compute_metrics(
    model: QueryAdapter,
    query_embeddings: np.ndarray,
    positive_image_indices: np.ndarray,
    image_embeddings: np.ndarray,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    ranks: list[int] = []
    with torch.no_grad():
        projected_batches = []
        for start in range(0, len(query_embeddings), batch_size):
            batch = torch.as_tensor(query_embeddings[start : start + batch_size], dtype=torch.float32, device=device)
            projected_batches.append(model(batch).detach().cpu().numpy())
    if not projected_batches:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "median_rank": 0.0}
    projected = np.concatenate(projected_batches, axis=0).astype(np.float32)
    image_vectors = np.asarray(image_embeddings, dtype=np.float32)
    similarities = projected @ image_vectors.T
    for row_index, scores in enumerate(similarities):
        order = np.argsort(-scores)
        rank = int(np.where(order == positive_image_indices[row_index])[0][0]) + 1
        ranks.append(rank)
    return evaluate_ranks(ranks, [0.0] * len(ranks)).to_dict()


def _format_metrics(title: str, metrics: dict[str, float]) -> str:
    lines = [f"# {title}", ""]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retina.yaml")
    parser.add_argument("--report-prefix", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = config["artifacts"]
    training = dict(config.get("training", {}) or {})
    embeddings_dir = Path(artifacts["embeddings_dir"])

    image_embeddings = np.load(embeddings_dir / "retina_image_embeddings.npy")
    text_embeddings = np.load(embeddings_dir / "retina_text_embeddings.npy")
    image_metadata = _load_frame(embeddings_dir / "retina_image_metadata.csv").reset_index(drop=True)
    text_metadata = _parse_split_frame(_load_frame(embeddings_dir / "retina_text_metadata.csv"))

    image_lookup = _resolve_image_lookup(image_metadata)
    text_metadata["image_index"] = text_metadata["image_id"].astype(str).map(image_lookup)
    text_metadata = text_metadata.dropna(subset=["image_index"]).reset_index(drop=True)
    text_metadata["image_index"] = text_metadata["image_index"].astype(int)

    train_rows = text_metadata[text_metadata["split"].astype(str) == "train"]
    val_rows = text_metadata[text_metadata["split"].astype(str) == "val"]
    if val_rows.empty:
        val_rows = text_metadata[text_metadata["split"].astype(str) != "train"]
    if val_rows.empty:
        val_rows = train_rows.sample(n=min(len(train_rows), max(1, len(train_rows) // 5)), random_state=42)

    if train_rows.empty:
        raise RuntimeError("No training rows were found in the text metadata")

    device = str(training.get("device", config["model"].get("device", "cpu")))
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    embedding_dim = int(image_embeddings.shape[1])
    hidden_dim = int(training.get("hidden_dim", max(256, embedding_dim * 2)))
    dropout = float(training.get("dropout", 0.1))
    epochs = int(training.get("epochs", 6))
    batch_size = int(training.get("batch_size", 128))
    learning_rate = float(training.get("learning_rate", 1e-3))
    weight_decay = float(training.get("weight_decay", 1e-2))
    temperature = float(training.get("temperature", 0.07))
    seed = int(training.get("seed", 42))

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = QueryAdapter(embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_query_vectors = torch.tensor(text_embeddings[train_rows.index.to_numpy()], dtype=torch.float32)
    train_image_vectors = torch.tensor(image_embeddings[train_rows["image_index"].to_numpy()], dtype=torch.float32)
    train_dataset = TensorDataset(train_query_vectors, train_image_vectors)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=len(train_dataset) >= batch_size)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for query_batch, image_batch in train_loader:
            query_batch = query_batch.to(device)
            image_batch = image_batch.to(device)
            projected = model(query_batch)
            logits = projected @ image_batch.T / temperature
            labels = torch.arange(logits.shape[0], device=device)
            loss = 0.5 * (criterion(logits, labels) + criterion(logits.T, labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            steps += 1
        train_loss = running_loss / max(steps, 1)
        val_metrics = _compute_metrics(
            model=model,
            query_embeddings=text_embeddings[val_rows.index.to_numpy()],
            positive_image_indices=val_rows["image_index"].to_numpy(),
            image_embeddings=image_embeddings,
            batch_size=batch_size,
            device=device,
        )
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), **val_metrics})

    final_val = history[-1] if history else {}
    checkpoint_path = Path(artifacts.get("query_adapter_path", "models/retina_query_adapter.pt"))
    checkpoint_metadata = {
        "dataset_name": str(text_metadata["source"].iloc[0]) if "source" in text_metadata.columns and len(text_metadata) else "synthetic",
        "model_name": config["model"]["name"],
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "history": history,
    }
    save_query_adapter(model.cpu(), checkpoint_path, metadata=checkpoint_metadata)

    report_dir = Path(artifacts["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_name": checkpoint_metadata["dataset_name"],
        "model_name": checkpoint_metadata["model_name"],
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "final_train_loss": float(history[-1]["train_loss"]) if history else 0.0,
        "final_val_recall_at_1": float(final_val.get("recall_at_1", 0.0)),
        "final_val_recall_at_5": float(final_val.get("recall_at_5", 0.0)),
        "final_val_recall_at_10": float(final_val.get("recall_at_10", 0.0)),
        "final_val_mrr": float(final_val.get("mrr", 0.0)),
        "history": history,
    }
    (report_dir / f"{args.report_prefix}query_adapter_training.json").write_text(json.dumps(payload, indent=2))
    (report_dir / f"{args.report_prefix}query_adapter_training.md").write_text(
        _format_metrics("Retina Query Adapter Training", {k: v for k, v in payload.items() if k != "history"})
    )


if __name__ == "__main__":
    main()
