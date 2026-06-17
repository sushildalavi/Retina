# Model Comparison

## Status

- Measured comparison available from repo artifacts.

## Verified comparison

| Model | Device | Sample | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Image embeddings/sec | End-to-end p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openai/clip-vit-base-patch32 | mps | 500 | 0.6432 | 0.8720 | 0.9376 | 0.7417 | 0.7891 | 58.07 | 14.54 ms |
| openai/clip-vit-large-patch14 | mps | 500 | 0.6624 | 0.9096 | 0.9552 | 0.7674 | 0.8135 | 2.43 | 114.11 ms |

## Takeaways

- The larger CLIP model is slightly stronger on ranking quality.
- The smaller CLIP model is dramatically faster for local interactive use.
- The comparison runner now reads the repo artifact and can render the same summary locally.

## How to regenerate

```bash
python scripts/compare_models.py --input reports/model_baseline_comparison.json --output-md docs/MODEL_COMPARISON.md
```
