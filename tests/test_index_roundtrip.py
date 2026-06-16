import numpy as np

from retrieval.vector_index import FaissVectorIndex


def test_faiss_index_save_and_load_roundtrip(tmp_path):
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = FaissVectorIndex.build(embeddings, ["a", "b"])
    index_path = tmp_path / "index.faiss"
    meta_path = tmp_path / "index.json"
    index.save(index_path, meta_path)
    loaded = FaissVectorIndex.load(index_path, meta_path)
    positions, _ = loaded.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=1)
    assert loaded.ids[positions[0][0]] == "a"

