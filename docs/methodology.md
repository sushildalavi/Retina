# Retina Methodology

Retina is a retrieval-first visual intelligence system.

Workflow:

1. prepare an image-caption manifest
2. encode images and captions with CLIP
3. build a CPU FAISS index over image embeddings
4. retrieve images for text queries
5. evaluate using Recall@K, MRR, and rank statistics
6. save failure cases and runtime measurements

Current validation run used a synthetic local sample dataset of 50 image-caption pairs generated on the Mac to keep the pipeline fully local and reproducible.

This keeps the project Mac-friendly while still exercising the full MLE loop.
