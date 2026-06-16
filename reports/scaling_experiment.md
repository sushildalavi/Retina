# Scaling Experiment

| Sample | Images | Captions | R@10 | MRR | End-to-end search p95 | Image embeddings/sec | Total runtime | Index size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 500 | 2500 | 0.9376 | 0.7417 | 14.54 ms | 58.07 | 8.61 s | 1024045 |
| 1000 | 1000 | 5000 | 0.8836 | 0.6593 | 16.88 ms | 48.02 | 20.82 s | 2048045 |
| 8000 | 8000 | 40000 | 0.6419 | 0.4082 | 14.70 ms | 38.82 | 206.09 s | 16384045 |

- The 500 and 1000 sample runs use the same CLIP ViT-B/32 configuration as the full benchmark.
- Total runtime is reported as embedding runtime, which is the dominant setup cost in this pipeline.