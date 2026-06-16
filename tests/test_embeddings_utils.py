import numpy as np

from retrieval.clip_encoder import normalize_embeddings, select_device


def test_normalize_embeddings_unit_length():
    arr = np.array([[3.0, 4.0]], dtype=np.float32)
    out = normalize_embeddings(arr)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_normalize_embeddings_handles_zero_vectors():
    arr = np.array([[0.0, 0.0]], dtype=np.float32)
    out = normalize_embeddings(arr)
    assert np.allclose(out, [[0.0, 0.0]])


def test_select_device_auto_returns_supported_device():
    device = select_device("auto")
    assert device in {"cpu", "mps"}

