# Retina Recommendation System

Retina is a full-stack visual discovery system for catalog search and asset review. It does not use collaborative filtering or real user history.

## Why Frozen CLIP Backbone?

Retina keeps the CLIP backbone frozen and trains a small query adapter on Flickr8k caption-image pairs. That gives the project a real training stage without destabilizing the embedding space or overfitting the small dataset. Freezing the backbone keeps the benchmark reproducible and fits the Mac-local compute budget.

## Recommendation modes

- `GET /recommend/text?query=...&top_k=10`
- `GET /recommend/image?image_id=...&top_k=10`
- `POST /recommend/profile`

## What each mode means

- text-to-image: a caption or query string is embedded with CLIP and compared against the image index
- image-to-image: a seed image is embedded and nearest neighbors are returned, excluding the seed image itself
- profile recommendations: text interests and liked image IDs are averaged into a content profile embedding
- query adaptation: text and profile embeddings pass through a trained adapter before retrieval

## Honest framing

- recommendations are similarity-based, not personalized from user behavior
- profile recommendations are simulated from grouped captions and liked image IDs
- image-to-image recommendations are qualitative/latency-focused because the benchmark does not define strong real-world positives
- the product value is faster asset discovery, not uncontrolled end-to-end fine-tuning

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

The React dashboard and Gradio demo are both thin clients over the same CLIP/adapter/FAISS recommendation engine.
