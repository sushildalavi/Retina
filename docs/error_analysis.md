# Retina Error Analysis

Failure analysis records:

- query caption
- expected image
- retrieved results
- similarity scores
- rank of the correct image
- conservative failure category

Current run summary:

- 142 failures out of 2500 caption queries
- summarized failures in `reports/flickr8k_retrieval_failures.md`
- raw failure rows in `reports/flickr8k_retrieval_failures.jsonl`
- dominant failure category: `visually_similar_negative`
- example confusion: dog and action-heavy captions often retrieved another semantically close Flickr8k image

The failure classifier also recognizes `object_mismatch`, `action_mismatch`, `scene_mismatch`, `color_attribute_mismatch`, `generic_caption`, `ambiguous_caption`, and `correct_image_not_in_top_k`.

The current default failure category is `no_hit` when the correct image is not in the top-k retrieval set.
