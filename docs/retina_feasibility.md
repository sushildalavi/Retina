# Retina Feasibility Audit

Retina is feasible as a Mac-local multimodal retrieval project if the MVP focuses on:

- CLIP/OpenCLIP embeddings
- CPU FAISS indexing
- text-to-image retrieval
- image-to-image retrieval
- Recall@K / MRR evaluation
- runtime benchmarks
- error analysis
- FastAPI + Gradio serving

Optional only:

- captioning
- grounding
- local fine-tuning

The right default dataset is a small image-caption set such as Flickr8k or a capped Flickr30k subset. The right default model is `openai/clip-vit-base-patch32` through `transformers`.

