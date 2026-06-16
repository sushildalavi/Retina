# Retina Methodology

Retina is a retrieval-first visual intelligence system.

Workflow:

1. prepare an image-caption manifest
2. encode images and captions with CLIP
3. build a CPU FAISS index over image embeddings
4. retrieve images for text queries
5. build content-profile recommendations from text interests and liked images
6. evaluate using Recall@K, MRR, nDCG@10, and rank statistics
7. save failure cases and runtime measurements

Current validation run used the full Flickr8k dataset so the benchmark stayed public, measurable, and reproducible on the Mac.

The synthetic 50-image sample still exists as the fastest smoke test, but it is not the measured benchmark for this pass.

For recommendation evaluation, content-profile queries are simulated from grouped captions and seeded likes rather than real user behavior data.

This keeps the project Mac-friendly while still exercising the full MLE loop.
