from __future__ import annotations

import numpy as np
import torch

from training.query_adapter import QueryAdapter, apply_query_adapter, load_query_adapter, save_query_adapter


def test_query_adapter_roundtrip(tmp_path):
    model = QueryAdapter(embedding_dim=4, hidden_dim=8, dropout=0.0)
    vectors = torch.randn(2, 4)
    output = model(vectors)
    assert output.shape == (2, 4)

    checkpoint_path = tmp_path / "adapter.pt"
    save_query_adapter(model, checkpoint_path, metadata={"hello": "world"})
    loaded, metadata = load_query_adapter(checkpoint_path, device="cpu")
    assert metadata["hello"] == "world"

    adapted = apply_query_adapter(np.ones((3, 4), dtype=np.float32), loaded, device="cpu")
    assert adapted.shape == (3, 4)
