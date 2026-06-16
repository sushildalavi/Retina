# Retina Limitations

- Why frozen CLIP backbone?

Retina freezes the CLIP backbone and trains only a lightweight query adapter. That keeps the embedding space stable, avoids overfitting the small Flickr8k corpus, and fits the Mac-local compute budget. Full backbone fine-tuning is still out of scope for this repo.

- full-stack discovery MVP
- content-based, not collaborative filtering
- no CUDA dependency
- no CLIP backbone fine-tuning
- the trainable part is the query adapter, not the foundation model
- grounding is optional
- captioning is optional
- dataset scale is intentionally capped for local development
- the measured run uses the full Flickr8k corpus, but not real user interaction logs
- the synthetic 50-sample manifest remains a smoke test, not the primary benchmark
- profile recommendations are simulated from text interests and liked image IDs
- the current product is optimized for visual search and review, not enterprise workflow orchestration
