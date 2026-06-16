# Retina Results

Measured local run:

- dataset: 50 locally generated image-caption pairs
- model: `openai/clip-vit-base-patch32`
- device: `mps`

Retrieval metrics:

- Recall@1: 0.30
- Recall@5: 0.82
- Recall@10: 0.98
- MRR: 0.5166
- median rank: 2.0
- query latency p50: 12.08 ms
- query latency p95: 13.97 ms

Runtime metrics:

- image embeddings/sec: 51.93
- text embeddings/sec: 51.93
- search queries/sec: 77.31
- search latency p50: 13.10 ms
- search latency p95: 15.16 ms

Results are written to:

- `reports/dataset_stats.json`
- `reports/dataset_stats.md`
- `reports/embedding_benchmark.json`
- `reports/embedding_benchmark.md`
- `reports/retrieval_eval.json`
- `reports/retrieval_eval.md`
- `reports/runtime_benchmark.json`
- `reports/runtime_benchmark.md`
- `reports/retrieval_failures.jsonl`
- `reports/retrieval_failures.md`

The only miss in this run was a visually similar green shape. That is a normal failure mode for a small synthetic dataset with many near-duplicate captions.
