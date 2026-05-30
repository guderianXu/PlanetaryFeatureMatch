# Pose Metadata Training Design

## Goal
Use simulated camera parameters during training without requiring camera parameters at inference time.

## Approach
Keep the existing pair archive format unchanged: `view_a`, `view_b`, `warp_a_to_b`, and `valid_mask` remain the model-facing tensors. Add a training-only metadata sidecar loader that reads `manifests/*.csv` and `tsai_tracks/*/tsai/Camera*/NNNNN.tsai`, then attaches pair-level geometry such as baseline, viewing-angle difference, focal ratio, overlap fraction, and a coarse difficulty label.

## Training Use
The metadata is used only by the training loop:

- pose-balanced sampling across easy, medium, hard, and unknown buckets;
- optional geometry difficulty loss weighting for synthetic warp supervision;
- metrics that report how many easy, medium, hard, and unknown pose pairs were used.

The PFM feature extractor and matcher do not receive camera parameters as inference inputs.

## Compatibility
Existing `.pt` archives are reused directly. Existing training commands keep old behavior unless pose metadata options are enabled.
