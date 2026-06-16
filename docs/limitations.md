# Retina Limitations

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
