# Training Plan

This document describes the intended C++/LibTorch training workflow for PlanetaryFeatureMatch. The current `train` command is a validation stub: it checks required CLI arguments and exits successfully, but it does not yet run optimization or write checkpoints.

## Training goal

Train a planetary local feature and matching model that supports both sparse keypoint matching and semi-dense correspondences. The model should be robust to:

- weak texture
- large illumination changes
- camera tilt
- multi-view geometry
- local affine deformation
- imaging distortion
- shadows, invalid regions, and occlusion

## Input data

The first training stage should use single planetary images and generate synthetic paired views online or through a cached preprocessing step.

Expected source images:

- 8-bit grayscale
- 16-bit grayscale
- 8-bit RGB
- 16-bit RGB

Preprocessing should convert each image to a LibTorch tensor with shape `C x H x W`, normalize it to `[0, 1]`, and optionally apply local contrast normalization.

## Synthetic pair generation

Each source image should produce two related views plus supervision:

- `view_a`
- `view_b`
- dense warp field from `view_a` to `view_b`
- valid correspondence mask

The current foundation implements deterministic translation-based synthetic pairs in `modules/data/synthetic_pair.cpp`. Future training should extend this module with:

- rotation and scale
- affine tilt
- perspective transforms
- radial and tangential distortion
- nonlinear local deformation
- brightness, contrast, and gamma shifts
- directional illumination changes
- shadow masks
- blur, noise, compression, and low-resolution degradation
- random invalid or occluded regions

## Model components

Training should optimize these modules together:

- `Backbone`: multi-scale feature extraction
- `SparseHead`: keypoint heatmap, descriptors, scale, orientation, affine shape
- `DenseHead`: semi-dense confidence and local offsets
- `Matcher`: descriptor similarity and matching foundation

## Losses

Already implemented foundation losses:

- `repeatability_loss`
- `descriptor_cross_entropy_loss`
- `masked_l1_loss`
- `confidence_bce_loss`

Future training should combine them into a full objective:

```text
total_loss =
  sparse_repeatability_weight * repeatability_loss +
  descriptor_weight * descriptor_loss +
  semi_dense_offset_weight * offset_loss +
  confidence_weight * confidence_loss +
  geometry_consistency_weight * consistency_loss
```

The geometry consistency loss is not implemented yet.

## Proposed `train` command

Current accepted command:

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1 \
  --device cpu
```

Planned behavior:

1. Load images from `--image-dir`.
2. Normalize images into LibTorch tensors.
3. Generate synthetic pairs and valid masks.
4. Run backbone, sparse branch, dense branch, and matcher.
5. Compute sparse, semi-dense, confidence, and matching losses.
6. Backpropagate with an AdamW optimizer.
7. Periodically evaluate repeatability, matching precision, and semi-dense coverage.
8. Save a LibTorch checkpoint to `--checkpoint` or `--output`.

## Minimal implementation milestones

1. Add an image dataset module and tests.
2. Extend synthetic pair generation beyond integer translation.
3. Add a training configuration structure and parser.
4. Implement checkpoint save/load tests.
5. Add a one-batch overfit test to prove gradients update model parameters.
6. Add the full training loop.
7. Add extraction and matching integration tests using a tiny synthetic checkpoint.

## Verification commands

Before considering training work complete, run:

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

For the current CLI stub, also run:

```bash
./build/pfm_cli train --image-dir images --checkpoint model.pt --epochs 1 --batch-size 1
```
