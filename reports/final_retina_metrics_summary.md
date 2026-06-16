# Final Retina Metrics Summary

## Dataset
- name: flickr8k
- images: 8000
- captions: 40000
- splits: {"dev": 1000, "test": 1000, "train": 6000}
- sample size: full
- requested sample size: full

## Model
- model: openai/clip-vit-base-patch32
- device: mps

## Retrieval
- recall_at_1: 0.307825
- recall_at_5: 0.542425
- recall_at_10: 0.641925
- mrr: 0.4081944525241852
- median_rank: 2.0
- faiss_search_latency_p95_ms: 0.30779314041137695
- end_to_end_text_search_latency_p95_ms: 14.701874752063304

## Recommendation
- text-to-image Recall@10: 0.641925
- profile Recall@10: 0.781125
- profile nDCG@10: 0.6049739582224133
- image-to-image latency p95: 0.3050503730773926

## Random Baseline
- Recall@10: 0.0012
- MRR: 0.0012134587159380317
- nDCG@10: 0.0005510213844842314

## Runtime
- image embeddings/sec: 38.817491318933634
- text embeddings/sec: 194.08745659466817
- search queries/sec: 83.05392785111384
- end-to-end search p95: 14.701874752063304 ms

## Failure Analysis
- failures: 14323 / 40000
- dominant categories: multiple_valid_matches, visually_similar_negative
- summary: reports/flickr8k_full_retrieval_failures.md

## API And Demo
- API command: `make api`
- demo command: `make demo`

## Resume Safe Claim
Built Retina, a Mac-local content-based visual recommendation system using CLIP ViT-B/32 embeddings, FAISS CPU indexing, and FastAPI/Gradio serving; evaluated on full Flickr8k (8K images / 40K captions) with 0.642 Recall@10, 0.408 MRR, 0.464 nDCG@10, and 14.7 ms p95 end-to-end search latency, outperforming random Recall@10 by 535x.

## Must Not Claim
- collaborative filtering
- personalized ranking from user behavior
- production deployment
- fine-tuned CLIP
- state-of-the-art performance

## Second Model Baseline
- status: measured
- model: openai/clip-vit-large-patch14
- sample size: 500
- Recall@10: 0.9552
- nDCG@10: 0.7891130309776321
- end-to-end search p95: 114.11076041986234 ms

## Scaling Experiment
- 500 images: Recall@10=0.9376, MRR=0.7416639924049377, search p95=14.53702148864977 ms, embeddings/sec=58.07424187912521, runtime=8.609668999910355 s, index bytes=1024045
- 1000 images: Recall@10=0.8836, MRR=0.6593074202537537, search p95=16.88386473688297 ms, embeddings/sec=48.02379886678884, runtime=20.823009082931094 s, index bytes=2048045
- full images: Recall@10=0.641925, MRR=0.4081944525241852, search p95=14.701874752063304 ms, embeddings/sec=38.817491318933634, runtime=206.09265895804856 s, index bytes=16384045
