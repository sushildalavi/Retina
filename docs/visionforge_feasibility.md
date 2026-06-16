# VisionForge / OmniVision Feasibility Audit

Date: 2026-06-15

## Final Verdict

**Feasible locally, with a constrained scope.**

The core project is realistic on this Mac if the MVP is centered on:

- CLIP/OpenCLIP-style embeddings
- CPU FAISS indexing
- text-to-image retrieval
- image-to-image retrieval
- offline evaluation and error analysis
- FastAPI + Gradio demo

The project becomes **partially feasible** when you add:

- local captioning
- local phrase grounding
- local LoRA-style fine-tuning

Those stretch goals are possible only with small models, strict batch sizing, and honest performance expectations. They should not be treated as required for the flagship MVP.

## Environment Snapshot

Observed local environment:

- macOS on Apple Silicon (`arm64`)
- Python: `3.13.5` in `/opt/anaconda3/bin/python`
- RAM: `16 GiB`
- CPU: `8 physical / 8 logical`
- Free disk on system volume: about `54 GiB`
- PyTorch: `2.11.0`
- MPS: built and available
- CUDA: unavailable

Installed packages relevant to the project:

- `torch` installed
- `transformers` installed
- `datasets` installed
- `sentence_transformers` installed
- `faiss` installed
- `fastapi` installed
- `gradio` installed
- `onnxruntime` installed
- `mlx` installed
- `hnswlib` not installed
- `open_clip` not installed

Local cache inspection:

- Hugging Face cache exists
- No obvious image-caption dataset cache was found during inspection
- No obvious CLIP / OpenCLIP / SigLIP cache was found during inspection
- Existing cache is dominated by unrelated text/audio/NLP assets

## Recommended Local-First Scope

Build the project around a retrieval system first.

Recommended MVP:

1. Prepare a small image-caption dataset locally.
2. Encode images and captions with CLIP/OpenCLIP.
3. Build a CPU FAISS index.
4. Support text query -> image retrieval.
5. Support image query -> similar image retrieval.
6. Evaluate Recall@1 / Recall@5 / Recall@10, MRR, median rank, latency, and throughput.
7. Store failure cases and metrics as JSON + Markdown artifacts.
8. Ship a FastAPI endpoint and Gradio demo.
9. Add tests for the data, embeddings, indexing, metrics, and smoke paths.

This is enough to be a strong MLE project if the implementation is clean and the evaluation is honest.

## Components That Can Run on This Mac

### Strongly feasible

- CLIP/OpenCLIP inference on CPU or MPS
- image and text embedding generation
- FAISS CPU indexing and search
- retrieval evaluation
- runtime benchmarking
- failure analysis generation
- FastAPI serving
- Gradio demo
- dataset sampling and split creation
- report generation in Markdown and JSON

### Feasible with caution

- image captioning with a small local model
- ONNX Runtime CPU export/benchmarking
- MLX-based experiments for smaller models
- limited local LoRA fine-tuning on small models

## Components To Skip Or Mark Optional

These should not be core requirements for the Mac-local MVP.

- CUDA-only training workflows
- vLLM
- Triton CUDA benchmarks
- QLoRA claims unless an actual local Mac-compatible run is demonstrated
- large VLM fine-tuning on full-size models
- heavy grounding pipelines that depend on large detector stacks
- any claim of speedups without measured results on this Mac

## Recommended Dataset

Best practical options:

1. **Flickr8k** for the fastest local iteration and a small demo.
2. **Flickr30k** for a stronger resume story and more credible evaluation.
3. **COCO captions subset** if you need more scale, but only a carefully limited subset.

Recommended choice:

- **Primary MVP**: Flickr8k or a small capped Flickr30k subset
- **Strong resume version**: Flickr30k

Why:

- paired image-caption data is enough to prove retrieval quality
- the dataset is small enough to run locally
- evaluation is straightforward
- the project can be made reproducible without giant downloads

## Recommended Model Stack

### MVP stack

- **Encoder**: CLIP ViT-B/32 or a comparable OpenCLIP variant
- **Index**: FAISS CPU
- **Serving**: FastAPI + Gradio
- **Metrics**: custom retrieval metrics module

### Optional additions

- **Captioning**: a small BLIP-style or similarly compact local captioning model if it runs acceptably
- **Grounding**: OWL-ViT-class model only for a small, clearly labeled demo
- **Fine-tuning**: MLX or small HF LoRA only if a tiny experiment completes locally

If `open_clip` is not installed, use `transformers` CLIP models first. OpenCLIP is preferred only if installation is smooth and the model weights are accessible.

## Runtime Risks

Main risks on this machine:

- 16 GiB RAM limits batch size and model choice
- MPS can be faster than CPU, but it is not a guarantee and can be unstable for some ops
- large captioning or grounding models may be slow or memory-hungry
- dataset download and preprocessing can dominate early setup time
- evaluation on too large a dataset may become sluggish

Practical mitigation:

- keep the MVP dataset small
- cache embeddings and metadata to disk
- use small batch sizes
- benchmark both CPU and MPS paths only where supported
- avoid forcing a big model just to look impressive

## Storage And Download Risks

Observed free space:

- about `54 GiB` free on the system volume

Implications:

- enough for a small retrieval project and cached artifacts
- not enough to casually download multiple large datasets and many large checkpoints
- the project should explicitly cache only what it uses

Recommended storage strategy:

- save dataset samples, embeddings, and indices under `data/`
- keep generated reports under `reports/`
- avoid downloading multiple redundant captioning or grounding checkpoints
- prefer one primary encoder and one optional captioning model

## Fine-Tuning Realism

### Is LoRA / fine-tuning realistic locally?

**Partially, but only in a narrow form.**

What is realistic:

- tiny adapter experiments
- limited-step LoRA on a small vision-language model
- using MLX for compact local experiments if a suitable model exists

What is not realistic to promise:

- full VLM fine-tuning
- QLoRA claims without a working Mac-compatible setup
- anything that requires CUDA-specific tooling

Recommendation:

- treat fine-tuning as an optional stretch goal
- include a clear “not attempted” or “not feasible locally” outcome if it does not run cleanly

## Grounding Realism

### Is phrase grounding realistic locally?

**Only as a limited optional demo.**

Possible:

- small-scale image-region grounding with a compact model
- qualitative examples on a small subset

Not recommended for MVP:

- large-scale grounding training
- strong quantitative grounding claims unless the dataset and labels are fully prepared locally

Recommendation:

- keep grounding as an optional add-on
- do not make it part of the core success criteria

## Minimum Viable Version

The minimum viable version that is still worth building:

- dataset prep for Flickr8k or a capped Flickr30k subset
- CLIP/OpenCLIP embeddings
- FAISS CPU search
- text-to-image retrieval
- image-to-image retrieval
- Recall@K, MRR, median rank
- latency and throughput benchmarks
- error analysis artifacts
- FastAPI + Gradio demo
- tests for core logic

This is already a legitimate MLE project.

## Strong Resume Version

The strong resume version adds:

- cleaned dataset splits and reproducible sampling
- cached embeddings and index artifacts
- failure analysis with real examples
- a polished Gradio UI
- a clean API
- a compact benchmark table in the README
- explicit limitations and honest tradeoffs

This version is credible for an MLE portfolio because it demonstrates:

- data handling
- retrieval systems
- evaluation
- serving
- benchmarking
- product polish

## 9/10 Version

To reach a 9/10 project:

- build the retrieval MVP well
- add one optional captioning path if it runs locally
- add a small, clearly labeled grounding demo only if it is stable
- add a tiny local adapter experiment only if it is repeatable
- publish a concise, evidence-backed writeup with measured results
- include ablations or comparisons, even if small:
  - CLIP vs OpenCLIP
  - FAISS index settings
  - CPU vs MPS inference for embedding generation if supported

The biggest upgrade is not more model complexity. It is stronger evaluation, cleaner artifact management, and better presentation.

## What Not To Claim

Do not claim:

- CUDA training on this Mac
- vLLM support locally
- Triton benchmarks locally
- QLoRA unless you actually run it on this machine
- large-model fine-tuning unless it is actually completed
- grounding accuracy unless labels and evaluation are actually implemented
- speedups without recorded measurements
- “production-ready” if the demo is only a local prototype

## Recommended Decision

Proceed with the project, but keep the MVP focused on retrieval.

That is the right balance of:

- feasible on Mac
- technically interesting
- resume-worthy
- measurable
- honest about limitations

## Implementation Recommendation

Build in this order:

1. dataset preparation
2. embedding pipeline
3. vector index
4. retrieval evaluation
5. failure analysis
6. benchmarking
7. Gradio demo
8. FastAPI endpoint
9. tests
10. optional captioning / grounding / fine-tuning only if the baseline is stable

