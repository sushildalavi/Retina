import numpy as np

from retrieval.vector_index import FaissVectorIndex


def test_faiss_index_returns_nearest_neighbor_first():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.95, 0.05]], dtype=np.float32)
    index = FaissVectorIndex.build(embeddings, ["a", "b", "c"])
    positions, scores = index.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=2)
    assert index.ids[positions[0][0]] == "a"
    assert scores[0][0] >= scores[0][1]

