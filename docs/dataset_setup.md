# Retina Dataset Setup

Retina supports either:

- a local synthetic manifest at `data/raw/captions.csv`
- a Hugging Face image-caption dataset such as `intro/flickr8k`

For local CSV manifests, the required columns are:

```csv
image_id,image_path,caption
img_001,images/img_001.jpg,A dog running through grass
```

Recommended local sample path:

- `data/raw/images/`
- `data/raw/captions.csv`

For the measured benchmark pass, `scripts/prepare_dataset.py` can materialize Flickr8k images locally and write the canonical metadata to `data/processed/retina_metadata.jsonl`.
