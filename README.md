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

For the synthetic smoke test only:

```bash
python -m scripts.create_sample_dataset --count 50
make prepare-data DATASET=synthetic SAMPLE_SIZE=50
```

## Dataset Format

`data/raw/captions.csv`

```csv
image_id,image_path,caption
img_001,images/img_001.jpg,A dog running through grass
```

## Reports

- `reports/flickr8k_dataset_stats.md`
- `reports/flickr8k_embedding_benchmark.md`
- `reports/flickr8k_retrieval_eval.md`
- `reports/flickr8k_runtime_benchmark.md`
- `reports/flickr8k_random_baseline.md`
- `reports/flickr8k_retrieval_failures.md`

## Measured Results

Dataset used:

- Flickr8k
- sample size: 500 images / 2500 captions
- split: 400 train, 50 val, 50 test
- materialized images stored under `data/artifacts/images/flickr8k/`

Model used:

- `openai/clip-vit-base-patch32`
- device: `mps`

Retrieval:

| Metric | Value |
| --- | ---: |
| Recall@1 | 0.6532 |
| Recall@5 | 0.8856 |
| Recall@10 | 0.9432 |
| MRR | 0.7529 |
| Median rank | 1.0 |
| Query latency p50 | 11.46 ms |
| Query latency p95 | 17.85 ms |

Random baseline:

| Metric | Value |
| --- | ---: |
| Recall@1 | 0.0028 |
| Recall@5 | 0.0128 |
| Recall@10 | 0.0200 |
| MRR | 0.0144 |
| Median rank | 248.0 |

Runtime:

| Metric | Value |
| --- | ---: |
| Image embeddings/sec | 56.16 |
| Text embeddings/sec | 280.78 |
| Search queries/sec | 88.28 |
| Search latency p50 | 10.92 ms |
| Search latency p95 | 14.22 ms |

Failure analysis:

- 142 / 2500 caption queries missed the exact image in the top-10 set
- all summarized failures fell into `visually_similar_negative`
- the failure report focuses on semantically close images, mostly dog/action/scene confusion

Report paths:

- `reports/flickr8k_dataset_stats.json`
- `reports/flickr8k_embedding_benchmark.json`
- `reports/flickr8k_retrieval_eval.json`
- `reports/flickr8k_random_baseline.json`
- `reports/flickr8k_runtime_benchmark.json`
- `reports/flickr8k_retrieval_failures.jsonl`

## Limitations

- retrieval-first MVP
- no CUDA
- no supervised fine-tuning yet
- no phrase grounding yet
- no captioning yet unless explicitly added later
- the synthetic 50-sample dataset remains the fastest smoke test
- the measured benchmark uses a capped Flickr8k subset to stay Mac-local
- first run may need CLIP weights downloaded

## Resume-Safe Summary

Built Retina, a visual intelligence search engine using CLIP embeddings, FAISS CPU indexing, and FastAPI/Gradio serving for semantic image recommendations, with evaluation across Recall@K, MRR, embedding throughput, search latency, and retrieval failure analysis on an image-caption dataset.
