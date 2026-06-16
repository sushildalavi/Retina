# Retina Limitations

- Why frozen CLIP?

Retina uses frozen CLIP embeddings because the project is a content-based retrieval system, not a CLIP training project. Keeping CLIP frozen makes the benchmark reproducible, avoids overfitting on the small Flickr8k corpus, and fits the Mac-local compute budget. The current full Flickr8k results already validate the retrieval stack, so fine-tuning is better treated as future work rather than an implemented claim.

- retrieval-first MVP
- content-based, not collaborative filtering
- no CUDA dependency
- no fine-tuning claim unless a local run is measured
- grounding is optional
- captioning is optional
- dataset scale is intentionally capped for local development
- the measured run uses the full Flickr8k corpus, but not real user interaction logs
- the synthetic 50-sample manifest remains a smoke test, not the primary benchmark
- profile recommendations are simulated from text interests and liked image IDs
