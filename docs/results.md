# Retina Results

Measured local run:

- dataset: Flickr8k full
- sample size: 8000 images / 40000 captions
- model: `openai/clip-vit-base-patch32`
- device: `mps`

Retrieval metrics:

- Recall@1: 0.3078
- Recall@5: 0.5424
- Recall@10: 0.6419
- MRR: 0.4082
- nDCG@10: 0.4639
- median rank: 1.0
- query latency p50: 0.25 ms
- query latency p95: 0.31 ms

Random baseline:

- Recall@1: 0.0002
- Recall@5: 0.0006
- Recall@10: 0.0012
- MRR: 0.0012
- nDCG@10: 0.0006
- median rank: 4002.0

Recommendation modes:

- text-to-image recommendations
- image-to-image recommendations
- content-profile recommendations

Profile recommendation metrics:

- Precision@1: 0.4435
- Recall@1: 0.4435
- Recall@5: 0.6932
- Recall@10: 0.7811
- MRR: 0.5494
- nDCG@10: 0.6050
- p95 latency: 0.28 ms

Runtime metrics:

- image embeddings/sec: 31.92
- text embeddings/sec: 56.14
- search queries/sec: 83.05
- search latency p50: 11.59 ms
- search latency p95: 14.70 ms
- image embedding p50: 17.88 ms
- image embedding p95: 75.47 ms
- text embedding p50: 21.65 ms
- text embedding p95: 27.90 ms

Results are written to:

- `reports/flickr8k_full_dataset_stats.json`
- `reports/flickr8k_full_dataset_stats.md`
- `reports/flickr8k_full_embedding_benchmark.json`
- `reports/flickr8k_full_embedding_benchmark.md`
- `reports/flickr8k_full_retrieval_eval.json`
- `reports/flickr8k_full_retrieval_eval.md`
- `reports/flickr8k_full_recommendation_eval.json`
- `reports/flickr8k_full_recommendation_eval.md`
- `reports/flickr8k_full_random_baseline.json`
- `reports/flickr8k_full_random_baseline.md`
- `reports/flickr8k_full_runtime_benchmark.json`
- `reports/flickr8k_full_runtime_benchmark.md`
- `reports/flickr8k_full_retrieval_failures.jsonl`
- `reports/flickr8k_full_retrieval_failures.md`

The dominant failure modes in this run were visually similar negatives and multiple valid matches, with most mistakes concentrated in bicycle and motocross action scenes.
