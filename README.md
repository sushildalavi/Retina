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
python -m scripts.create_sample_dataset --count 50
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
- [reports/embedding_benchmark.md](reports/embedding_benchmark.md)
- [reports/retrieval_eval.md](reports/retrieval_eval.md)
- [reports/runtime_benchmark.md](reports/runtime_benchmark.md)
- [reports/retrieval_failures.md](reports/retrieval_failures.md)

## Measured Results

Dataset used:

- 50 locally generated image-caption pairs
- each image is a simple colored shape on a light background
- manifest generated with `python -m scripts.create_sample_dataset --count 50`

Model used:

- `openai/clip-vit-base-patch32`
- device: `mps`

Retrieval:

| Metric | Value |
| --- | ---: |
| Recall@1 | 0.30 |
| Recall@5 | 0.82 |
| Recall@10 | 0.98 |
| MRR | 0.5166 |
| Median rank | 2.0 |
| Query latency p50 | 12.08 ms |
| Query latency p95 | 13.97 ms |

Runtime:

| Metric | Value |
| --- | ---: |
| Image embeddings/sec | 51.93 |
| Text embeddings/sec | 51.93 |
| Search queries/sec | 77.31 |
| Search latency p50 | 13.10 ms |
| Search latency p95 | 15.16 ms |

Failure analysis:

- 1 / 50 queries missed the exact image in the top-10 set
- the failure was a color/shape confusion between similar green shapes

## Limitations

- retrieval-first MVP
- no CUDA
- no supervised fine-tuning yet
- no phrase grounding yet
- no captioning yet unless explicitly added later
- sample dataset is synthetic and intentionally small for Mac-local validation
- first run may need CLIP weights downloaded

## Resume-Safe Summary

Built Retina, a visual intelligence search engine using CLIP embeddings, FAISS CPU indexing, and FastAPI/Gradio serving for semantic image recommendations, with evaluation across Recall@K, MRR, embedding throughput, search latency, and retrieval failure analysis on an image-caption dataset.
