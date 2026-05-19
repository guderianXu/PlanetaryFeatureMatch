# Keypoint graph matching loss optimization design

## Goal

Make `graph_matching_loss` train the same task that inference uses: matching sparse keypoints between a synthetic pair. The current descriptor and dense losses can decrease, but the graph matcher loss stays around 3-5 because its supervision is built from fixed descriptor-grid samples and the matcher ignores keypoint geometry.

## Evidence

A short reproduction of the user command with CSV logging showed:

- `graph_matching_loss`: first-window mean about 3.92, last-window mean about 3.94.
- `descriptor_loss`: first-window mean about 2.72, last-window mean about 0.30.
- `descriptor_accuracy`: first-window mean about 0.40, last-window mean about 0.93.
- `dense_loss`: first-window mean about 0.83, last-window mean about 0.065.

Code inspection found two training mismatch points:

1. `PlanetaryGraphMatcherImpl::forward()` accepts `keypoints_a` and `keypoints_b` but discards them.
2. `make_graph_matching_loss()` trains on 256 fixed descriptor-grid samples and builds B from positive warped targets only, so the target is mostly a positional identity label rather than a realistic sparse matching decision with negatives and dustbin cases.

## Recommended design

Train graph matching on decoded sparse keypoints instead of fixed descriptor-grid samples.

For each pair in a training batch:

1. Decode sparse features for A and B using the same `FeatureDecodeConfig` used by training visualization and inference.
2. For each A keypoint, use `warp_a_to_b` to find its expected B location.
3. Assign a positive B keypoint when the nearest B keypoint is within a configurable pixel threshold and both source/target pixels are valid under the training mask.
4. Build a compact B candidate set containing:
   - all positives needed by the sampled A queries,
   - local hard negatives near the warped location,
   - random negatives from the decoded B keypoints,
   - a dustbin column for unmatched/invalid A queries.
5. Train `PlanetaryGraphMatcher` with cross entropy over that candidate set.
6. Normalize keypoint coordinates and add `_keypoint_projection(normalized_keypoints)` to descriptor embeddings before graph attention.

This aligns the loss with inference behavior while keeping the implementation small enough to validate with existing C++ tests.

## Component changes

### Graph matcher

Use the existing `_keypoint_projection` module. Normalize keypoints to a stable coordinate range before projection. The training caller should pass keypoints in feature-map or image coordinates plus the relevant width/height normalization path; the matcher itself should require already-normalized or consistently scaled keypoints, not infer image size from values.

### Training loss construction

Add a graph-matching training helper that consumes decoded `ImageFeatures` for A/B, dense descriptors, `warp_a_to_b`, and `valid_mask`. It should return:

- scalar graph loss,
- candidate accuracy,
- query count,
- positive match count,
- dustbin target count.

The existing CSV/progress metrics can keep `graph_matching_loss` and add optional diagnostic columns later if needed. For the first implementation, tests should verify the helper behavior; logging extra counts is useful but not required for the core fix.

### Sampling policy

Cap graph queries per image to keep training bounded. Prefer high-score decoded A keypoints, but include only those with valid source/target masks. Candidate sets should be deterministic for the same pair variant so tests and cached training are reproducible.

### Loss target semantics

- Matched A query: target is the candidate column containing its assigned B keypoint.
- Unmatched or invalid A query: target is the dustbin column.
- Empty query set: return zero graph loss and zero accuracy without throwing.

## Testing strategy

Add tests before implementation:

1. Graph matcher keypoint embedding affects logits: same descriptors with different keypoints should produce different logits.
2. Keypoint graph target assignment maps a warped A point to the nearest valid B keypoint within threshold.
3. Unmatched A keypoints target dustbin when no B candidate is close enough.
4. Candidate construction includes positives and does not duplicate the positive as a negative.
5. Training graph loss is lower when logits favor the assigned positive/dustbin targets.
6. Full test suite still passes after the change.

## Verification plan

After implementation:

1. Build and run `pfm_tests`.
2. Run the short reproduction command with `--log-csv metrics_debug.csv` and compare the first/last-window `graph_matching_loss` means.
3. Run the user's longer command and inspect `graph_matching_loss`, `descriptor_accuracy`, and visualization correctness.

Expected outcome: `graph_matching_loss` should show a downward trend on the short run, while descriptor and dense metrics remain stable or improve. If graph loss still does not fall, the next hypothesis is architectural: replace row-wise CE with a bidirectional/Sinkhorn-style objective.

## Non-goals

- Do not redesign the backbone, dense head, or augmentation pipeline in this change.
- Do not add a large external matching dependency.
- Do not expose low-level graph-matching knobs through CLI until the training signal is proven useful.
