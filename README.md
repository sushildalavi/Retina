# Retina

**Visual intelligence search engine for image-text retrieval, semantic recommendations, CLIP embeddings, FAISS indexing, evaluation, and local serving.**

Retina is a Mac-local multimodal visual search and content-based recommendation engine. It uses CLIP embeddings and a CPU FAISS index to retrieve images from text queries and similar images from image queries, then evaluates the system with Recall@K, MRR, latency, throughput, and failure analysis.

## Architecture

```text
Image-caption dataset
  -> CLIP image/text encoders
  -> normalized embeddings
  -> FAISS CPU index
  -> visual search / semantic recommendations
  -> Recall@K / MRR evaluation
  -> failure analysis
  -> FastAPI + Gradio serving
```

## Features

- CLIP image/text embeddings
- FAISS CPU vector search
- text-to-image retrieval
- image-to-image recommendations
- Recall@K and MRR evaluation
- latency and throughput benchmarks
- retrieval failure analysis
- FastAPI API
- Gradio UI
- Mac / MPS support

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make prepare-data
make embeddings
make index
make eval
make benchmark
make demo
```

## Dataset Format

`data/raw/captions.csv`

```csv
image_id,image_path,caption
img_001,images/img_001.jpg,A dog running through grass
```

## Reports

- `reports/dataset_stats.md`
- `reports/retina_retrieval_eval.md`
- `reports/retina_runtime_benchmark.md`
- `reports/retina_retrieval_failures.md`

## Limitations

- retrieval-first MVP
- no CUDA
- no supervised fine-tuning yet
- no phrase grounding yet
- no captioning yet unless explicitly added later
- first run may need CLIP weights downloaded

## Resume-Safe Summary

Built Retina, a visual intelligence search engine using CLIP embeddings, FAISS CPU indexing, and FastAPI/Gradio serving for semantic image recommendations, with evaluation across Recall@K, MRR, embedding throughput, search latency, and retrieval failure analysis on an image-caption dataset.

