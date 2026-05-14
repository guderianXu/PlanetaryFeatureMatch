# Planetary Sparse and Semi-Dense Feature Matching Design

## Goal

Build a C++/LibTorch deep-learning framework for planetary image feature extraction and matching. The first version targets offline research processing for Mars, Moon, and asteroid imagery, prioritizing matching quality over speed. It must use ordinary 8-bit/16-bit grayscale or RGB images and must not depend on camera metadata, PDS, or ISIS products.

The system addresses weak texture, multi-view geometry, imaging distortion, camera tilt, scale/rotation change, and large illumination variation. It combines sparse local feature extraction with semi-dense matching because traditional nearest-neighbor matching plus RANSAC is not robust enough for the target planetary cases.

## Design Direction

Use a dual-branch unified framework:

1. A shared multi-scale feature backbone.
2. A sparse local-feature branch for keypoints, descriptors, scale, orientation, and affine shape.
3. A semi-dense coarse-to-fine matching branch.
4. A learned matching and consistency module that refines sparse and semi-dense correspondences.

This direction is more complex than a SuperPoint-style sparse-only model or a LoFTR-style dense-only model, but it is the best fit because the project requires both sparse features and reliable semi-dense correspondences.

## Inputs

The first version supports:

- 8-bit grayscale images.
- 16-bit grayscale images.
- 8-bit RGB images.
- 16-bit RGB images.

Preprocessing:

- Convert images to LibTorch tensors.
- Normalize 8-bit and 16-bit inputs to `[0, 1]`.
- Keep grayscale as one channel and RGB as three channels.
- Optionally apply local contrast normalization during training and inference to improve robustness under weak texture and shadows.

## Architecture

### Shared Backbone

The backbone extracts multi-scale features from each image:

- CNN encoder stages at `1/2`, `1/4`, `1/8`, and `1/16` resolution.
- FPN-style fusion to combine shallow localization detail with deeper contextual features.
- Local/window attention blocks at `1/8` and `1/16` resolution.
- A global context gate from global pooling or a context token to help handle image-wide illumination changes.

This keeps the model practical in C++/LibTorch while giving it more context than a pure CNN descriptor.

### Sparse Branch

The sparse branch predicts:

- Keypoint heatmap.
- Keypoint confidence.
- Dense descriptor map.
- Scale map.
- Orientation map.
- Affine-shape map.

Inference converts maps into local features using differentiable-style decoding during training and standard NMS during inference. Each feature contains:

- `x, y` keypoint coordinates.
- `score`.
- Descriptor vector.
- `scale`.
- `orientation`.
- `2x2 affine shape`.

The affine shape is included because camera tilt, local perspective distortion, and planetary terrain relief can break purely rotation/scale-normalized features.

### Semi-Dense Branch

The semi-dense branch performs coarse-to-fine matching:

1. Build a coarse correlation volume or attention-based matching grid at low resolution.
2. Select confident coarse candidate matches.
3. Refine candidates in local windows at higher resolution.
4. Output reliable semi-dense correspondences rather than forcing every pixel to match.

The final output is a set of semi-dense point correspondences with confidence values. This is better suited than dense optical flow for planetary images because large shadows, low texture, and occlusion can create invalid regions.

### Learned Matcher

The learned matcher refines and filters matches from both branches:

- Sparse graph-attention or lightweight Transformer module over keypoints and descriptors.
- Semi-dense confidence refinement using local correlation, geometry consistency, and feature confidence.
- Optional cross-branch consistency, where semi-dense evidence can support sparse matches and sparse anchors can regularize semi-dense matches.

RANSAC may remain as an evaluation baseline or optional post-processing step, but it is not the primary source of robustness.

## Training Data Generation

Training starts from single planetary images and generates synthetic view pairs. For each source image, create two augmented views and store the dense warp field plus valid mask.

Synthetic transformations include:

- Rotation.
- Scale change.
- Affine tilt.
- Perspective transformation.
- Radial and tangential distortion.
- Nonlinear local distortion.
- Brightness, contrast, and gamma changes.
- Directional illumination changes.
- Local shadows.
- Albedo-like intensity changes.
- Blur.
- Noise.
- Compression artifacts.
- Low-resolution degradation.
- Weak-texture suppression.
- Random invalid or occluded regions.

The generated warp field provides supervision for sparse keypoint consistency, descriptor matching, affine-parameter consistency, and semi-dense correspondence training.

## Losses

### Sparse Losses

- Repeatability loss for keypoint heatmap consistency under the known synthetic warp.
- Descriptor contrastive or InfoNCE loss using positive pairs from the warp field and hard negatives from nearby confusing regions.
- Scale consistency loss.
- Orientation consistency loss.
- Affine-shape consistency loss.
- Reliability loss to down-weight invalid, occluded, or unmatchable regions.

### Semi-Dense Losses

- Coarse grid match classification loss.
- Fine offset regression loss.
- Confidence calibration loss.
- Valid-mask loss to suppress predictions in occluded, shadowed, or invalid regions.

### Matcher Losses

- Sparse match cross-entropy or optimal-transport-style assignment loss.
- Match confidence loss.
- Cycle-consistency loss.
- Geometry-aware ranking loss for hard negatives that look visually similar but are geometrically incorrect.

## Outputs

### Single-Image Feature Extraction

A feature extraction command outputs:

- `keypoints: Nx2`.
- `scores: N`.
- `descriptors: NxD`.
- `scale: N`.
- `orientation: N`.
- `affine: Nx2x2`.

### Pairwise Matching

A matching command outputs sparse matches:

- `matches: Mx2`, storing indices into image A and image B feature arrays.
- `match_scores: M`.
- Optional local geometry hints such as relative affine transform or local displacement.

It also outputs semi-dense matches:

- `points_a: Kx2`.
- `points_b: Kx2`.
- `confidence: K`.
- `local_offsets`.
- Optional valid/reliable-region mask.

## C++/LibTorch Project Modules

Use C++ throughout training, evaluation, and inference.

Modules:

- `data`: image reading, tensor conversion, normalization, augmentation, synthetic warp generation, valid-mask generation.
- `models`: shared backbone, sparse head, semi-dense head, learned matcher.
- `losses`: sparse losses, dense losses, matcher losses, shared geometry utilities.
- `train`: training loop, checkpointing, validation, logging.
- `eval`: offline metrics and benchmark reports.
- `infer`: single-image extraction and pairwise matching.
- `cli`: CLI11-based command definitions.
- `tests`: TDD unit and integration tests.

## CLI Design

Use `/home/xjw/code/deeplearning/Feature Extraction/CLI11.hpp` for command-line parsing.

Commands:

- `train`: train from an image directory using synthetic view generation.
- `extract`: run single-image feature extraction.
- `match`: run sparse and semi-dense matching for an image pair.
- `eval`: evaluate on synthetic or real image-pair benchmarks.
- `export`: export model checkpoints or inference artifacts.

Representative options:

- `--image-dir`.
- `--pairs`.
- `--checkpoint`.
- `--config`.
- `--output`.
- `--max-keypoints`.
- `--semi-dense-threshold`.
- `--device`.
- `--epochs`.
- `--batch-size`.

## Evaluation

The offline evaluation suite reports:

- Keypoint repeatability.
- Matching precision.
- Matching recall.
- Localization error.
- Sparse match count and spatial distribution.
- Semi-dense coverage.
- Semi-dense endpoint error on synthetic pairs.
- Registration success rate.
- Homography or geometric transform error when ground truth is available.

Reports should separate results by stress condition:

- Weak texture.
- Strong shadow.
- Large illumination shift.
- Large rotation/scale.
- Affine tilt.
- Perspective distortion.
- Nonlinear distortion.
- Noise/blur/compression.

## Error Handling

The program should fail clearly for unsupported inputs and invalid configuration.

Required behavior:

- Reject unsupported image formats with a clear message.
- Reject images that are too small for the configured model stride.
- Warn when 16-bit dynamic range is degenerate, then normalize safely if possible.
- Skip synthetic samples whose valid mask becomes empty.
- Reject checkpoint/config mismatches.
- Reject inference commands when required model files are missing.

## Test-Driven Development Plan

Implementation must follow TDD:

1. Write failing tests for image normalization and tensor shape conversion.
2. Write failing tests for synthetic geometry and warp-field consistency.
3. Write failing tests for valid-mask behavior.
4. Write failing tests for model tensor shapes at each output head.
5. Write failing tests for loss functions on controlled toy examples.
6. Write failing tests for CLI11 command parsing.
7. Write failing integration tests for `extract`, `match`, and `eval` on tiny fixture images.
8. Implement the minimum code required to pass each test group.
9. Refactor only after tests pass.

## Non-Goals for Version 1

- No Python training pipeline.
- No dependency on PDS, ISIS, SPICE, or camera metadata.
- No embedded or real-time deployment optimization.
- No full-pixel dense optical flow output requirement.
- No reliance on RANSAC as the main matching robustness mechanism.

## Open Implementation Choices

These can be finalized during implementation planning:

- Exact descriptor dimension.
- Exact backbone width/depth.
- Whether the sparse matcher uses optimal transport or simpler attention-based classification.
- Exact checkpoint serialization format.
- Exact image I/O library choice.
