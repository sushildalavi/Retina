# Model Baseline Comparison

| Model | Sample | Device | R@1 | R@5 | R@10 | MRR | nDCG@10 | Image embeddings/sec | End-to-end search p95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openai/clip-vit-base-patch32 | 500 | mps | 0.6432 | 0.8720 | 0.9376 | 0.7417 | 0.7891 | 58.07 | 14.54 ms |
| openai/clip-vit-large-patch14 | 500 | mps | 0.6624 | 0.9096 | 0.9552 | 0.7674 | 0.8135 | 2.43 | 114.11 ms |

- The larger CLIP model is feasible on MPS for a 500-image sample, but it is far slower and consumes more memory.
- This comparison is capped at 500 images to keep the run practical on the local Mac.