# Retina Results

Measured local run:

- dataset: Flickr8k
- sample size: 500 images / 2500 captions
- model: `openai/clip-vit-base-patch32`
- device: `mps`

Retrieval metrics:

- Recall@1: 0.6532
- Recall@5: 0.8856
- Recall@10: 0.9432
- MRR: 0.7529
- median rank: 1.0
- query latency p50: 11.46 ms
- query latency p95: 17.85 ms

Random baseline:

- Recall@1: 0.0028
- Recall@5: 0.0128
- Recall@10: 0.0200
- MRR: 0.0144
- median rank: 248.0

Runtime metrics:

- image embeddings/sec: 56.16
- text embeddings/sec: 280.78
- search queries/sec: 88.28
- search latency p50: 10.92 ms
- search latency p95: 14.22 ms

Results are written to:

- `reports/flickr8k_dataset_stats.json`
- `reports/flickr8k_dataset_stats.md`
- `reports/flickr8k_embedding_benchmark.json`
- `reports/flickr8k_embedding_benchmark.md`
- `reports/flickr8k_retrieval_eval.json`
- `reports/flickr8k_retrieval_eval.md`
- `reports/flickr8k_random_baseline.json`
- `reports/flickr8k_random_baseline.md`
- `reports/flickr8k_runtime_benchmark.json`
- `reports/flickr8k_runtime_benchmark.md`
- `reports/flickr8k_retrieval_failures.jsonl`
- `reports/flickr8k_retrieval_failures.md`

The dominant failure mode in this run was visually similar negatives: semantically close images, mostly dogs, action shots, and scene-level confusion.
