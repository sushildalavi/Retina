# Retina Error Analysis

Failure analysis records:

- query caption
- expected image
- retrieved results
- similarity scores
- rank of the correct image
- conservative failure category

Current run summary:

- 1 failure out of 50 queries
- default failure category: `no_hit`
- failure example: a green square caption retrieved other green shapes first

The current default failure category is `no_hit` when the correct image is not in the top-k retrieval set.
