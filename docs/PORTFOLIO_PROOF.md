# Portfolio Proof

## What the project does

Retina is a local visual search app built on CLIP embeddings and FAISS, with a trained query adapter and a browser UI.

## Why it is technically impressive

- It unifies text, image, and profile search over a local index.
- The repo already tracks retrieval metrics and low-latency performance.
- The UI and backend make the retrieval pipeline easy to demo.

## Architecture summary

- Query -> embedding / adapter -> FAISS retrieval -> ranked results -> frontend detail view.

## How to run locally

- `make install`
- `make prepare-data`
- `make embeddings`
- `make train-query-adapter`
- `make index`
- `make eval`
- `make api`
- `make frontend`

## How to test

- `pytest`
- `npm run build` in `frontend/`

## How to benchmark or evaluate

- Review repo evaluation outputs and notes around CLIP/FAISS performance

## Verified metrics only

- Recall@10: 0.641925
- MRR: 0.408194
- p95 end-to-end latency: 14.701875 ms
- FAISS p95 latency: 0.307793 ms

## Current limitations

- Failure analysis is documented separately, but it still relies on repo artifacts rather than a live workflow.
- The new model comparison runner reads the checked-in comparison artifact; it does not launch heavy model downloads by default.
- Some UI polish can still be improved for portfolio presentation.

## Future improvements

- Add a failure gallery with clearly labeled hard cases.
- Add duplicate clustering tooling and a tighter result-detail explanation panel.

## Resume bullets

- Built a CLIP + FAISS visual search app with a trained query adapter and browser UI.
- Achieved verified low-latency retrieval with documented recall and ranking metrics.
- Delivered multi-modal search modes for text, image, and profile-based discovery.

## Verification Log

- `npm run build` in `frontend/` - pass - 2026-06-17 - Verified the redesigned UI compiles for production.
- `python3 scripts/compare_models.py --input reports/model_baseline_comparison.json --output-md /tmp/retina_model_comparison.md --output-json /tmp/retina_model_comparison.json` - pass - 2026-06-17 - Rendered the local model comparison artifact from the checked-in report.
