# Retina Recommendation System

Retina is a content-based visual recommendation system. It does not use collaborative filtering or real user history.

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

## Result fields

Each recommendation result includes:

- rank
- image_id
- image_path
- captions
- score
- recommendation_reason

