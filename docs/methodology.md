# Retina Methodology

Retina is a retrieval-first visual intelligence system.

Workflow:

1. prepare an image-caption manifest
2. encode images and captions with CLIP
3. build a CPU FAISS index over image embeddings
4. retrieve images for text queries
5. evaluate using Recall@K, MRR, and rank statistics
6. save failure cases and runtime measurements

Current validation run used Flickr8k with a capped 500-image sample so the benchmark stayed local, measurable, and reproducible on the Mac.

The synthetic 50-image sample still exists as the fastest smoke test, but it is not the measured benchmark for this pass.

This keeps the project Mac-friendly while still exercising the full MLE loop.
