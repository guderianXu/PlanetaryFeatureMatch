# Rotation Matching Handoff

Date: 2026-05-20

## Current Status

The project now has a repeatable rotation sweep evaluator and several feature extractor / matcher changes aimed at 0-360 degree sparse matching. The latest code builds and the test suite passed locally:

```bash
cmake --build build-pfm-cf -j$(nproc) --target pfm_tests pfm_cli pfm_rotation_sweep_eval
./build-pfm-cf/pfm_tests
# 317 test(s) passed
```

The model is still not solved for 180 degree matching. The latest 180 degree visualization still shows mostly parallel match lines, not the expected X-shaped center-symmetric pattern.

## Important Code Changes

- `tools/rotation_sweep_eval.cpp`
  - Adds `pfm_rotation_sweep_eval`.
  - Rotates one image through fixed angle steps, runs `pfm_cli extract/match`, and writes a `summary.csv`.
  - Reports sparse match pass rate, mean error, descriptor mutual nearest diagnostics, and keypoint repeatability.

- `src/augment/transform_sampler.cpp`
  - Mixed augmentation now injects clean deterministic rotation anchors.
  - `variant % 8 == 3`: pure +/-90 degree anchor.
  - `variant % 8 == 7`: pure +/-180 degree anchor.
  - These clean anchors no longer stack random translation, scale, gamma, or shadow on top.

- `src/models/sparse_head.cpp`
  - Descriptor head uses C4 cyclic descriptor slots.
  - Orientation head is used to soft-canonicalize descriptor slots.

- `src/train/trainer.cpp`
  - Adds explicit orientation supervision from the known synthetic warp.
  - Adds full-map / keypoint descriptor hard-negative supervision.
  - Training sparse decode now uses the user's full `TrainConfig`, including `min_keypoints`.
  - For non-cached single-image training, variant indices now advance by epoch instead of repeating the same 8 pairs forever.

- `src/models/planetary_graph_matcher.cpp`
  - Keypoint embedding no longer uses absolute normalized x/y directly.
  - It now uses rotation-insensitive radius features `(r, r^2)` to reduce the "same screen position" shortcut that produces parallel 180 degree matches.

- `src/infer/matching_pipeline.cpp` and `src/losses/losses.cpp`
  - Descriptor matching/losses support cyclic descriptor similarity.
  - Fallback descriptor mutual matching was added for cases where the learned graph matcher returns no sparse matches.

## Latest Experiments

All generated outputs were kept under `runs/` and are intentionally ignored by git.

### Clean Anchor Training

Checkpoint:

```text
runs/rotation_cleananchors_2026-05-20/train_rotation100_cleananchors.pt
```

Sweep:

```text
runs/rotation_cleananchors_2026-05-20/rotation_sweep_100_cleananchors_step90/summary.csv
```

Result:

| angle | matches | pass_rate | mean_error_px |
| --- | ---: | ---: | ---: |
| 0 | 1024 | 1.0000 | 0.000 |
| 90 | 132 | 0.0152 | 184.225 |
| 180 | 124 | 0.0242 | 142.631 |
| 270 | 108 | 0.0093 | 156.057 |

180 degree visualization:

```text
runs/rotation_cleananchors_2026-05-20/rotation_sweep_100_cleananchors_step90/rot_180_vis/100__rot_180_matches.png
```

Conclusion: not acceptable. 180 degree lines are still mostly parallel.

### Epoch-Variant + Radial Matcher Training

Checkpoint:

```text
runs/rotation_epochvariants_2026-05-20/train_rotation100_epochvariants_radialmatcher.pt
```

Training command:

```bash
./build-pfm-cf/pfm_cli train \
  --image-dir runs/rotation_epochvariants_2026-05-20/input \
  --checkpoint runs/rotation_epochvariants_2026-05-20/train_rotation100_epochvariants_radialmatcher.pt \
  --epochs 60 --batch-size 1 --resize 512 --pairs-per-image 8 \
  --augmentation-profile mixed --min-keypoint-intensity 0.05 \
  --max-keypoints 1024 --min-keypoints 1024 \
  --device cuda \
  --log-csv runs/rotation_epochvariants_2026-05-20/metrics_rotation100_epochvariants_radialmatcher.csv \
  --visualization-dir runs/rotation_epochvariants_2026-05-20/vis_rotation100_epochvariants_radialmatcher \
  --visualization-samples 0
```

Final training loss:

```text
final_loss=4.36732
```

This is not a low loss. Descriptor accuracy remained unstable by batch.

Sweep:

```text
runs/rotation_epochvariants_2026-05-20/rotation_sweep_100_epochvariants_radialmatcher_step90/summary.csv
```

Result:

| angle | matches | pass_rate | mean_error_px |
| --- | ---: | ---: | ---: |
| 0 | 1024 | 1.0000 | 0.000 |
| 90 | 114 | 0.0175 | 162.879 |
| 180 | 165 | 0.0364 | 133.322 |
| 270 | 137 | 0.0146 | 146.974 |

180 degree visualization:

```text
runs/rotation_epochvariants_2026-05-20/rotation_sweep_100_epochvariants_radialmatcher_step90/rot_180_vis/100__rot_180_matches.png
```

Conclusion: slightly better numerically than clean-anchor-only training, but still wrong. The 180 degree match lines are still mostly parallel instead of X-shaped.

## What Is Known

- The training data warp for pure 180 degree rotation is correct. Ground-truth 180 degree visualization is X-shaped:

```text
runs/debug_halfturn_training_data_clean/pure_180_ground_truth_x.png
```

- The current learned descriptor/matcher still chooses visually similar horizontal structures after 180 degree rotation.
- The issue is no longer only "missing 180 degree augmentation".
- The likely bottleneck is local descriptor distinctiveness under large rotation. False mutual nearest descriptors still score higher than true rotated correspondences.
- The graph matcher can still be influenced by repeated terrain texture, even after removing absolute x/y position from keypoint embedding.

## Recommended Next Steps

1. Do not keep blindly extending training time on the current architecture.
2. On the faster machine, first run a classical baseline on `img/100.tif` rotated 180 degrees:
   - SIFT if OpenCV contrib is available.
   - ORB if SIFT is unavailable.
   - Check whether classical local descriptors produce the expected X-shaped matches.
3. If SIFT/ORB gives the correct X shape, use it as the reference behavior and redesign `SparseHead` toward true local rotation-normalized descriptors:
   - extract descriptor around orientation-normalized local patches, or
   - explicitly train a rotation-bin classifier and use hard canonical rotation, or
   - compare descriptors over multiple rotated local patch samples rather than only channel slot shifts.
4. Add a rotation-sweep hard-negative objective where the true warped keypoint descriptor must beat all same-image and rotated-image negatives by a margin.
5. Keep all experiment outputs under `runs/<experiment_name>/`; do not write new test folders into the repo root.

## Useful Commands

Build and test:

```bash
cmake -S . -B build-pfm-cf
cmake --build build-pfm-cf -j$(nproc) --target pfm_tests pfm_cli pfm_rotation_sweep_eval
./build-pfm-cf/pfm_tests
```

Run the latest style of rotation sweep:

```bash
./build-pfm-cf/pfm_rotation_sweep_eval \
  --image img/100.tif \
  --checkpoint runs/rotation_epochvariants_2026-05-20/train_rotation100_epochvariants_radialmatcher.pt \
  --output-dir runs/rotation_epochvariants_2026-05-20/rotation_sweep_100_epochvariants_radialmatcher_step90 \
  --pfm-cli ./build-pfm-cf/pfm_cli \
  --device cuda \
  --angle-step 90 \
  --max-keypoints 1024 \
  --min-keypoints 1024 \
  --min-keypoint-intensity 0.05 \
  --threshold-px 3
```

## Git Notes

- `runs/`, `img/`, `img.zip`, `build-*`, and `*.pre-merge-backup` are ignored.
- Checkpoints and visualizations are not committed.
- The source code and this handoff document are the important parts to pull on the other machine.
