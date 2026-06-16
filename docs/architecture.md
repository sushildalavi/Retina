# Retina Architecture

```text
Image-caption dataset
  -> CLIP image/text encoders
  -> normalized embeddings
  -> FAISS CPU index
  -> visual search / semantic recommendations
  -> Recall@K / MRR evaluation
  -> failure analysis
  -> FastAPI + Gradio serving
```

Core design points:

- image and text embeddings live in the same vector space
- cosine similarity is implemented with normalized vectors and inner-product search
- evaluation is offline and reproducible
- artifacts are written to `data/processed/` and `reports/`

