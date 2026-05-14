# PlanetaryFeatureMatch

PlanetaryFeatureMatch is a C++17 and LibTorch foundation for deep-learning planetary image feature extraction and matching. It targets Mars, Moon, and asteroid imagery where weak texture, viewpoint change, imaging distortion, camera tilt, and illumination variation make traditional local features and RANSAC-only pipelines unreliable.

The current implementation is a tested foundation rather than a complete training system. It provides module-level tensor utilities, normalization, synthetic pair generation, affine warp helpers, model shape contracts, losses, metrics, CLI parsing, and validation stubs for the main command flow.

## Design

The intended model is a dual sparse and semi-dense matching system:

- shared multi-scale backbone
- sparse feature branch for keypoints, descriptors, scale, orientation, and affine shape
- semi-dense branch for confident point correspondences
- learned matcher foundation for descriptor similarity and match scoring

## Repository layout

```text
modules/
  cli/        CLI11 command parsing and tests
  core/       tensor validation and grid helpers
  data/       normalization and synthetic pair generation
  eval/       matching and semi-dense metrics
  geometry/   affine warping helpers
  infer/      command validation stubs
  losses/     repeatability, descriptor, offset, and confidence losses
  models/     backbone, sparse head, dense head, matcher
src/
  main.cpp    CLI entry point
tests/
  test_main.cpp
  test_harness.h
```

The project intentionally keeps code organized by module with each module paired with its own `*_test.cpp` file.

## Requirements

- CMake 3.18+
- C++17 compiler
- LibTorch / PyTorch C++ package discoverable by CMake
- `CLI11.hpp` in the repository root

## Build and test

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

## CLI

```bash
./build/pfm_cli --help
```

Implemented command validation stubs:

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1

./build/pfm_cli extract \
  --image a.png \
  --checkpoint model.pt \
  --output a.pfm \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5

./build/pfm_cli match \
  --image-a a.png \
  --image-b b.png \
  --checkpoint model.pt \
  --output matches.json \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5

./build/pfm_cli eval \
  --pairs pairs.txt \
  --checkpoint model.pt \
  --output report.json

./build/pfm_cli export \
  --checkpoint model.pt \
  --output exported.pt
```

At this stage these commands validate arguments and print `command accepted`; they do not yet train, infer, serialize features, or export a real model.

## Current status

Implemented and tested:

- CHW tensor validation and XY grid creation
- 8-bit and 16-bit normalization
- local contrast normalization
- affine warp field and valid mask helpers
- deterministic synthetic pair generation
- backbone, sparse head, dense head, and matcher tensor contracts
- repeatability, descriptor, masked L1, and confidence losses
- matching precision and semi-dense coverage metrics
- CLI11 command parsing

Not yet implemented:

- image file loading
- dataset iteration
- full training loop
- checkpoint save/load
- real feature extraction output
- real pairwise matching output
- model export backend
