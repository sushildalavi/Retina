# Retina Runtime Benchmarks

Measured local benchmark:

- dataset: Flickr8k
- model: `openai/clip-vit-base-patch32`
- device: `mps`
- sample size: 16
- image embeddings/sec: 56.16
- text embeddings/sec: 280.78
- search queries/sec: 88.28
- image latency p50: 16.66 ms
- image latency p95: 63.59 ms
- text latency p50: 35.44 ms
- text latency p95: 306.10 ms
- search latency p50: 10.92 ms
- search latency p95: 14.22 ms

The image/text latency numbers come from the embedding benchmark, while the search latency numbers come from the retrieval benchmark.

Benchmarks are expected to run on CPU or MPS only. CUDA-specific claims are out of scope for this repository.
