# Retina Scaling Experiment

CLIP ViT-B/32 was evaluated at three scales to check how retrieval and runtime change as the corpus grows.

| Sample | Images | Captions | Recall@10 | MRR | End-to-end search p95 | Image embeddings/sec | Total runtime | Index size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 500 | 2500 | 0.9376 | 0.7417 | 14.54 ms | 58.07 | 8.61 s | 1024045 bytes |
| 1000 | 1000 | 5000 | 0.8836 | 0.6593 | 16.88 ms | 48.02 | 20.82 s | 2048045 bytes |
| full | 8000 | 40000 | 0.6419 | 0.4082 | 14.70 ms | 38.82 | 206.09 s | 16384045 bytes |

## Readout

- Retrieval quality drops as the task gets harder and the candidate pool grows.
- Search latency stays low because FAISS CPU lookup remains fast.
- Total runtime is dominated by embedding generation rather than the index itself.

## Why this matters

The experiment supports a local-first design: the backend can stay interactive while still evaluating a full Flickr8k corpus on a Mac.
