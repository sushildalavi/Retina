# Retina Dataset Setup

Retina expects a local manifest at `data/raw/captions.csv` or a custom path passed to `scripts/prepare_dataset.py`.

Required columns:

```csv
image_id,image_path,caption
img_001,images/img_001.jpg,A dog running through grass
```

Recommended local sample path:

- `data/raw/images/`
- `data/raw/captions.csv`

For this pass, a small synthetic sample dataset may be generated locally to validate the pipeline without downloading a large public corpus.

