# Retina

**Content-based visual recommendation system using CLIP embeddings, FAISS indexing, full Flickr8k evaluation, failure analysis, FastAPI, and Gradio.**

Retina is a Mac-local multimodal visual recommendation engine. It uses CLIP embeddings and a CPU FAISS index to recommend semantically similar images from text, image, and content-profile queries, then evaluates the system with Recall@K, MRR, nDCG@10, latency, throughput, and failure analysis.

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
- text-to-image recommendations
- image-to-image recommendations
- content-profile recommendations
- Recall@K and MRR evaluation
- nDCG@10 and coverage metrics
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
make recommendations
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

- `reports/flickr8k_full_dataset_stats.md`
- `reports/flickr8k_full_embedding_benchmark.md`
- `reports/flickr8k_full_retrieval_eval.md`
- `reports/flickr8k_full_recommendation_eval.md`
- `reports/flickr8k_full_runtime_benchmark.md`
- `reports/flickr8k_full_random_baseline.md`
- `reports/flickr8k_full_retrieval_failures.md`

## Full Benchmark

Dataset used:

- Flickr8k full
- 8000 images / 40000 captions
- splits: 6000 train, 1000 dev, 1000 test
- materialized images stored under `data/artifacts/images/flickr8k/`

Model used:

- `openai/clip-vit-base-patch32`
- device: `mps`

Benchmark table:

| Dataset | Images | Captions | Model | Device | R@1 | R@5 | R@10 | MRR | nDCG@10 | FAISS-only search p95 |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flickr8k | 8000 | 40000 | `openai/clip-vit-base-patch32` | `mps` | 0.3078 | 0.5424 | 0.6419 | 0.4082 | 0.4639 | 0.30 ms |

Baseline table:

| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.0002 | 0.0006 | 0.0012 | 0.0012 | 0.0006 |
| CLIP ViT-B/32 | 0.3078 | 0.5424 | 0.6419 | 0.4082 | 0.4639 |

Recommendation modes implemented:

- text-to-image recommendations for caption queries
- image-to-image recommendations for visually similar items
- content-profile recommendations from multiple text interests and liked image IDs
- caption-to-image is covered by the text recommendation path

Recommendation metrics:

| Mode | Precision@1 | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | FAISS-only search p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-to-image | 0.3078 | 0.3078 | 0.5424 | 0.6419 | 0.4082 | 0.4639 | 0.30 ms |
| Profile recommendations | 0.4435 | 0.4435 | 0.6932 | 0.7811 | 0.5494 | 0.6050 | 0.28 ms |

Image-to-image recommendations are reported as qualitative/latency-only behavior:

| Mode | p50 latency | p95 latency | average similarity |
| --- | ---: | ---: | ---: |
| Similar images | 0.25 ms | 0.31 ms | 0.8091 |

Runtime:

The embedding throughput and embedding latency numbers below come from the smaller embedding benchmark sample, while the full-benchmark search latency is shown in the table above.

| Metric | Value |
| --- | ---: |
| Image embeddings/sec | 31.92 |
| Text embeddings/sec | 56.14 |
| Search queries/sec | 83.05 |
| End-to-end search latency p50 | 11.59 ms |
| End-to-end search latency p95 | 14.70 ms |
| Image embedding p50 | 17.88 ms |
| Image embedding p95 | 75.47 ms |
| Text embedding p50 | 21.65 ms |
| Text embedding p95 | 27.90 ms |

Failure analysis:

- 14,323 / 40,000 caption queries missed the exact image in the top-10 set
- dominant failure categories: `multiple_valid_matches` and `visually_similar_negative`
- failure examples are heavily concentrated in action-heavy bicycle and motocross scenes

API and demo commands:

- `make api`
- `make demo`

Report paths:

- `reports/flickr8k_full_dataset_stats.json`
- `reports/flickr8k_full_embedding_benchmark.json`
- `reports/flickr8k_full_retrieval_eval.json`
- `reports/flickr8k_full_recommendation_eval.json`
- `reports/flickr8k_full_random_baseline.json`
- `reports/flickr8k_full_runtime_benchmark.json`
- `reports/flickr8k_full_retrieval_failures.jsonl`
- `reports/final_retina_metrics_summary.json`
- `reports/model_baseline_comparison.json`
- `reports/scaling_experiment.json`

## Limitations

- content-based, not collaborative filtering
- no CUDA
- no supervised fine-tuning yet
- no phrase grounding yet
- no captioning yet unless explicitly added later
- the synthetic 50-sample dataset remains the fastest smoke test
- the full Flickr8k benchmark was completed locally on MPS
- first run may need CLIP weights downloaded

## Additional Experiments

- second-model baseline on 500 images: `openai/clip-vit-large-patch14` reached Recall@10 0.9552, but end-to-end search p95 rose to 114.11 ms
- scaling experiment with CLIP ViT-B/32:
  - 500 images: Recall@10 0.9376, MRR 0.7417, p95 14.54 ms
  - 1000 images: Recall@10 0.8836, MRR 0.6593, p95 16.88 ms
  - 8000 images: Recall@10 0.6419, MRR 0.4082, p95 14.70 ms

## Resume-Safe Summary

Built Retina, a content-based visual recommendation system using CLIP embeddings, FAISS CPU indexing, and FastAPI/Gradio serving for text-to-image, image-to-image, and content-profile recommendations, with full Flickr8k evaluation across Recall@K, MRR, nDCG@10, embedding throughput, search latency, and failure analysis.
