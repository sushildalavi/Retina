# Retina Scaffold Audit

## What Works

- Project structure is in place for retrieval, evaluation, serving, docs, and tests.
- `RetinaClipEncoder` wraps Hugging Face CLIP inference on CPU or MPS.
- `FaissVectorIndex` builds and searches a CPU FAISS index.
- `RetinaSearchEngine` supports text-to-image and image-to-image retrieval against metadata.
- Dataset prep supports CSV or JSONL manifests with `image_id`, `image_path`, and `caption`.
- Basic retrieval metrics, API, and Gradio smoke tests already pass.

## What Is Untested

- Real dataset ingestion with images on disk.
- End-to-end embedding generation on a real sample dataset.
- FAISS index persistence against generated embeddings.
- Evaluation and failure-analysis scripts against real outputs.
- Runtime benchmark reporting with measured numbers.

## What Is Scaffold-Only

- README and docs currently describe the target system, but not measured outcomes yet.
- Report files exist as destinations, but no real benchmark or evaluation artifacts have been generated.
- Captioning, grounding, and fine-tuning are still out of scope for the current code.

## What Must Run To Produce Real Metrics

1. Create or download a small image-caption dataset with local image paths.
2. Run `make prepare-data`.
3. Run `make embeddings`.
4. Run `make index`.
5. Run `make eval`.
6. Run `make failures`.
7. Run `make benchmark`.

