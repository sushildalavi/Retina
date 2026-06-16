# Final Results

Retina is a content-based visual recommendation system built around CLIP embeddings and FAISS similarity search.

## What is measured
Built Retina, a Mac-local content-based visual recommendation system using CLIP ViT-B/32 embeddings, FAISS CPU indexing, and FastAPI/Gradio serving; evaluated on full Flickr8k (8K images / 40K captions) with 0.642 Recall@10, 0.408 MRR, 0.464 nDCG@10, and 14.7 ms p95 end-to-end search latency, outperforming random Recall@10 by 535x.

## What must not be claimed
- collaborative filtering
- personalized ranking from user behavior
- production deployment
- fine-tuned CLIP
- state-of-the-art performance

## Source Of Truth
- Final summary report: `reports/final_retina_metrics_summary.json`
- Final summary markdown: `reports/final_retina_metrics_summary.md`
- Full Flickr8k benchmark reports: `reports/flickr8k_full_*`

## Notes
- Text-to-image and profile recommendations are content-based CLIP similarity, not collaborative filtering.
- Image-to-image recommendation accuracy is qualitative; the benchmark reports latency and similarity only.
- The full Flickr8k benchmark completed locally on MPS.
