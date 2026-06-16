# Retina Final Results

Retina is a content-based visual recommendation system built around CLIP embeddings and FAISS similarity search.

## Full Flickr8k benchmark

- 8,000 images
- 40,000 captions
- CLIP model: `openai/clip-vit-base-patch32`
- device: `mps`

Measured retrieval:

- Recall@1: `0.307825`
- Recall@5: `0.542425`
- Recall@10: `0.641925`
- MRR: `0.4081944525241852`
- MAP@10: `0.4081944543650794`
- nDCG@10: `0.46386424661409603`
- FAISS-only search p95: `0.30779314041137695 ms`
- end-to-end search p95: `14.701874752063304 ms`

## Random baseline

- Recall@10: `0.0012`
- MRR: `0.0012134587159380317`
- nDCG@10: `0.0005510213844842314`

CLIP retrieval is far above random because the embeddings capture semantic structure, not just index position.

## Recommendation modes

- text-to-image
- image-to-image
- profile recommendations

## Second model baseline

On a 500-image sample, `openai/clip-vit-large-patch14` improved retrieval slightly:

- Recall@10: `0.9552`
- nDCG@10: `0.8134588312468383`

But the tradeoff is steep:

- image embeddings/sec: `2.4347797323306373`
- end-to-end search p95: `114.11076041986234 ms`

`openai/clip-vit-base-patch32` stays the default because it is much faster and still strong enough for local serving.

## Scaling experiment

- 500 images: Recall@10 `0.9376`, MRR `0.7416639924049377`, p95 `14.53702148864977 ms`
- 1000 images: Recall@10 `0.8836`, MRR `0.6593074202537537`, p95 `16.88386473688297 ms`
- 8000 images: Recall@10 `0.641925`, MRR `0.4081944525241852`, p95 `14.701874752063304 ms`

## Failure analysis

- 14,323 misses out of 40,000 caption queries
- dominant categories: `multiple_valid_matches`, `visually_similar_negative`
- common scenes: bicycle racing, motocross jumps, dogs, and other action-heavy images

## Limitations

- content-based, not collaborative filtering
- no user-behavior personalization
- no fine-tuned CLIP
- no production deployment claim
- Mac-local benchmark

## Source of truth

- `reports/final_retina_metrics_summary.json`
- `reports/final_retina_metrics_summary.md`
- `reports/flickr8k_full_*`
- `reports/model_baseline_comparison.json`
- `reports/scaling_experiment.json`
