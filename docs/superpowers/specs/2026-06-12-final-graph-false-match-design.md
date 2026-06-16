# Final Graph False-Match Mining Design

## Goal

Add optional online supervision that mines high-confidence wrong matches from the final GraphMatcher output, rather than only feeding descriptor-mined false endpoints as extra no-match points.

## Problem

The existing `graph_matcher_online_false_no_match` path uses descriptor-level mining. It finds descriptor mutual false matches, samples their endpoints, and adds those endpoints to the GraphMatcher no-match pool. That helps rejection, but it does not directly penalize the final GraphMatcher edge that would be returned by inference.

The next training run needs to target the actual failure mode: a final off-diagonal matcher edge receives high assignment and accept confidence even though the sampled training correspondences already define a diagonal ground truth.

## Scope

This change only affects Python training. It does not change C++ inference, pair generation, extractor architecture, or default training behavior.

The new loss is disabled by default and activates only when its weight is positive.

## Design

Inside `graph_matcher_correspondence_loss()`, the first `count` keypoints in A and B are true one-to-one correspondences. Therefore, within the positive block `logits[:count, :count]`, diagonal entries are true matches and off-diagonal entries are false matches.

The new helper mines only high-confidence off-diagonal entries from this positive block:

- compute the same row and column softmax probabilities used by GraphMatcher assignment;
- form a dual score for each positive-block pair;
- remove diagonal entries;
- optionally remove spatially-near B candidates to avoid punishing near-duplicate samples;
- keep at most `topk` wrong entries above `min_score`;
- penalize their `accept_logits` as negative labels when the accept head exists;
- penalize their pair logits when they are too close to the corresponding true diagonal logit.

The helper returns both the scalar loss and diagnostics:

- `graph_matcher_final_false_match_loss`
- `graph_matcher_final_false_match_edges`
- `graph_matcher_final_false_match_score_mean`
- `graph_matcher_final_false_match_accept_mean`

The positive dustbin guard also disables this loss. If true matches are already being overpowered by dustbin, adding a false-match penalty can make the model more conservative, so the guard should keep rejection pressure low until positive-vs-dustbin margins recover.

## CLI

Add these optional arguments to `pfm_pytorch_training.py`:

- `--graph-matcher-final-false-match-weight`
- `--graph-matcher-final-false-match-topk`
- `--graph-matcher-final-false-match-min-score`
- `--graph-matcher-final-false-match-margin`
- `--graph-matcher-final-false-match-spatial-min-distance`

Defaults keep old behavior:

- weight `0.0`
- topk `8`
- min score `0.0`
- margin `0.25`
- spatial min distance `0.0`

## Testing

Add tests before production code:

1. a direct helper test where an off-diagonal edge has high final probability and produces positive loss plus one mined edge;
2. a direct helper test where no off-diagonal edge passes `min_score` and the helper returns zero loss and zero edges;
3. a `graph_matcher_correspondence_loss()` component test that verifies the weighted loss appears in returned components;
4. a parse-args test that verifies the new CLI options.

Run the targeted Python test module after implementation.

## Validation Run

After tests pass, launch a short fov76 training run from the current safe baseline checkpoint with low final-false-match weight. The first validation target is not best final performance; it is whether the model reduces hard-view zero-match and low-precision cases without increasing true-match dustbin rejection.
