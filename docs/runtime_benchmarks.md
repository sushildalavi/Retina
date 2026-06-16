# Retina Runtime Benchmarks

Measured local benchmark:

- dataset: Flickr8k
- model: `openai/clip-vit-base-patch32`
- device: `mps`
- sample size: 16
- image embeddings/sec: 31.92
- text embeddings/sec: 56.14
- search queries/sec: 83.05
- image latency p50: 17.88 ms
- image latency p95: 75.47 ms
- text latency p50: 21.65 ms
- text latency p95: 27.90 ms
- FAISS-only search p50: 0.25 ms
- FAISS-only search p95: 0.31 ms
- end-to-end text recommendation p50: 11.59 ms
- end-to-end text recommendation p95: 14.70 ms

Full-benchmark recommendation latencies:

- text-to-image p50: 0.25 ms
- text-to-image p95: 0.31 ms
- profile recommendation p50: 0.25 ms
- profile recommendation p95: 0.28 ms
- image-to-image p50: 0.25 ms
- image-to-image p95: 0.31 ms

The image/text latency numbers come from the embedding benchmark, while the search latency numbers come from the retrieval benchmark.

Benchmarks are expected to run on CPU or MPS only. CUDA-specific claims are out of scope for this repository.
