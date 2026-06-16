# Retina Recommendation System

Retina is a full-stack visual discovery system for catalog search and asset review. It does not use collaborative filtering or real user history.

## Why Frozen CLIP?

Retina is a retrieval system first, not a CLIP training project. Freezing CLIP keeps the benchmark reproducible, avoids overfitting on the small Flickr8k corpus, and works better with the Mac-local compute budget. The existing full Flickr8k results already validate the retrieval pipeline; fine-tuning remains future work, not a claimed implementation.

## Recommendation modes

- `GET /recommend/text?query=...&top_k=10`
- `GET /recommend/image?image_id=...&top_k=10`
- `POST /recommend/profile`

## What each mode means

- text-to-image: a caption or query string is embedded with CLIP and compared against the image index
- image-to-image: a seed image is embedded and nearest neighbors are returned, excluding the seed image itself
- profile recommendations: text interests and liked image IDs are averaged into a content profile embedding

## Honest framing

- recommendations are similarity-based, not personalized from user behavior
- profile recommendations are simulated from grouped captions and liked image IDs
- image-to-image recommendations are qualitative/latency-focused because the benchmark does not define strong real-world positives
- the product value is faster asset discovery, not model training

## Result fields

Each recommendation result includes:

- rank
- image_id
- image_path
- image_url when the backend can safely expose a browser-accessible local artifact path
- captions
- score
- recommendation_reason

## Backend contract

The production-style backend exposes the same modes as JSON endpoints:

- `GET /recommend/text`
- `GET /recommend/image`
- `POST /recommend/profile`
- `POST /search/text` as the compatibility alias

The React dashboard and Gradio demo are both thin clients over the same CLIP/FAISS recommendation engine.
