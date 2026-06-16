# Retina Runtime Benchmarks

Measured local benchmark:

- model: `openai/clip-vit-base-patch32`
- device: `mps`
- sample size: 16
- image embeddings/sec: 51.93
- text embeddings/sec: 51.93
- search queries/sec: 77.31
- image latency p50: 17.42 ms
- image latency p95: 142.56 ms
- text latency p50: 13.31 ms
- text latency p95: 104.12 ms
- search latency p50: 13.10 ms
- search latency p95: 15.16 ms

Benchmarks are expected to run on CPU or MPS only. CUDA-specific claims are out of scope for this repository.
