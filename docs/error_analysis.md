# Retina Error Analysis

Failure analysis records:

- query caption
- expected image
- retrieved image ids, paths, captions, and scores
- rank of the correct image
- conservative failure category

Current run summary:

- 14,323 failures out of 40,000 caption queries
- summarized failures in `reports/flickr8k_full_retrieval_failures.md`
- raw failure rows in `reports/flickr8k_full_retrieval_failures.jsonl`
- dominant failure categories: `multiple_valid_matches` and `visually_similar_negative`
- common confusion clusters: bicycle racing, motocross jumps, dogs, and similar action-heavy scenes

The failure classifier recognizes `object_mismatch`, `action_mismatch`, `scene_mismatch`, `attribute_mismatch`, `visually_similar_negative`, `generic_caption`, `multiple_valid_matches`, `exact_target_missing_from_top_k`, `ambiguous_caption`, and `unknown`.

The current default failure category is `exact_target_missing_from_top_k` when the correct image is not in the top-k retrieval set.
