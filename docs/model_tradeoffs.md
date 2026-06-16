# Retina Model Tradeoffs

Retina keeps `openai/clip-vit-base-patch32` as the default local model because it is the best balance of quality and latency for Mac-local serving.

## Measured comparison on 500 images

| Model | Recall@10 | MRR | nDCG@10 | Image embeddings/sec | End-to-end search p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `openai/clip-vit-base-patch32` | 0.9376 | 0.7417 | 0.7891 | 58.07 | 14.54 ms |
| `openai/clip-vit-large-patch14` | 0.9552 | 0.7674 | 0.8135 | 2.43 | 114.11 ms |

## Interpretation

- ViT-L/14 improves retrieval slightly on the 500-image sample.
- The accuracy gain does not justify the latency and throughput cost for local interactive use.
- ViT-B/32 remains the default because it is much faster and still reaches the full Flickr8k benchmark numbers reported in the README.

## Practical conclusion

Use ViT-B/32 for local serving and recruiting demos.
Use ViT-L/14 only as an exploratory baseline when you can tolerate much slower runtime.
