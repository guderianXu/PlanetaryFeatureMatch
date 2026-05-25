# Cross-View Matching Handoff, 2026-05-25

## Current Git State

- Repository: `https://github.com/guderianXu/PlanetaryFeatureMatch.git`
- Branch: `main`
- Code baseline before this handoff document: `f396587 Default cache eval to adaptive geometry`
- Local worktree status when this file was written: clean before this document edit.

Important: `img/` and `runs/` are ignored by git. GitHub has code and docs only, not generated datasets, checkpoints, visualizations, or evaluation CSV files. When moving to the RTX 5090 machine, copy the needed `img/` and `runs/` artifacts manually, or regenerate them there.

## What Is Done

### Rotation Matching Baseline

The rotation-only matching pipeline is already implemented and documented in `docs/rotation_matching_final.md`.

- Rotation synthetic data uses exact `warp_a_to_b` pixel correspondences.
- The training objective uses true generated correspondence instead of SIFT/ORB.
- Rotation-only data is organized by source image folder.
- High-density extraction can produce enough matches for 0-360 degree in-plane rotation.
- Rotation remains the baseline that must not regress.

### Cross-View Synthetic Datasets

Current local generated datasets:

| dataset | local path | approximate size | role |
|---|---:|---:|---|
| rotation 512 | `img/Rotate` | 36G | rotation base training/eval |
| rotation 1024 | `img/Rotate_1024` | 143G | high-res rotation cache |
| viewpoint 512 | `img/Viewpoint` | 18G | cross-view training/eval |
| viewpoint 1024 | `img/Viewpoint_1024` | 72G | high-res viewpoint cache |
| compound 512 | `img/CompoundViewpoint` | 18G | rotation + viewpoint compound training/eval |
| compound 1024 | `img/CompoundViewpoint_1024` | 75G | high-res compound cache |

Approximate file counts from current machine:

- `img/Rotate`: 11089 `.pt` files and 11319 image files.
- `img/Rotate_1024`: 11089 `.pt` files and 11319 image files.
- `img/Viewpoint`: 5545 `.pt` files and 5775 image files.
- `img/Viewpoint_1024`: 5545 `.pt` files and 5775 image files.
- `img/CompoundViewpoint`: 5545 `.pt` files and 5775 image files.
- `img/CompoundViewpoint_1024`: 5546 `.pt` files and 5775 image files.

### PyTorch Iteration Tools

Implemented under `python/`:

- `python/pfm_model.py`: PyTorch version of the current PFM model for faster iteration.
- `python/pfm_pytorch_training.py`: multi-cache training and fine-tuning loop.
- `python/pytorch_cache_match_eval.py`: synthetic-cache evaluator using true `warp_a_to_b`.
- `python/cache_match_eval.py`: C++ `pfm_cli` cache evaluator.
- `python/interpolate_pytorch_states.py`: checkpoint interpolation utility.

Training/eval additions already implemented:

- warp-aware hard negative loss;
- scheduled teacher/hard-negative/diversity weights;
- stable descriptor normalization with larger epsilon;
- nonfinite loss guard and optional bad-step skip;
- texture-aware keypoint selection;
- local displacement consistency filtering for non-global geometry;
- Python tests for these pieces.

### C++ Inference Updates

Recent pushed commits:

- `c2c3f6b Add local displacement geometry filter`
- `4ff45b5 Enable adaptive local geometry filtering`
- `5577542 Expose adaptive sparse geometry CLI modes`
- `f396587 Default cache eval to adaptive geometry`

The C++ sparse geometry filter now supports:

```bash
--sparse-geometry-filter adaptive
--sparse-geometry-filter projective
--sparse-geometry-filter local
--sparse-geometry-filter rotation-only
```

Default behavior is now adaptive. It keeps projective geometry when it is already strong, and switches to local displacement consistency only when local produces a substantial match-count gain. This avoids forcing RANSAC/projective assumptions on non-global cross-view deformation while preserving easy rotation/projective cases.

### Verification

Latest verified commands on current machine:

```bash
ctest --test-dir build-pfm-verify-mamba --output-on-failure
PYTHONPATH=python /home/xjw/anaconda3/envs/llm-learn/bin/python -m unittest discover -s python -p 'test_*.py'
```

Observed results:

- C++ tests: passed.
- Python tests: 65 tests passed.

## Current Best Results

### PyTorch Synthetic Cache Eval

Best current Python-side setting:

- checkpoint: `runs/pytorch_pfm_finetune_2026-05-25_all512_context_300step/pytorch_pfm_state.pt`
- keypoints: texture-aware
- matching: mutual top-k 32
- geometry: local displacement consistency
- texture blend: 1

Summary:

| set | pairs | matches | correct | wrong | precision |
|---|---:|---:|---:|---:|---:|
| Rotate hard11 | 11 | 121 | 50 | 71 | 0.413223 |
| Viewpoint64 | 64 | 945 | 305 | 640 | 0.322751 |
| Compound64 | 64 | 560 | 108 | 452 | 0.192857 |

Compared with affine/projective filtering, local consistency clearly improves viewpoint and compound recall:

| set | affine/projective correct/matches | local correct/matches |
|---|---:|---:|
| Rotate hard11 | 44/87 | 50/121 |
| Viewpoint64 | 62/320 | 305/945 |
| Compound64 | 26/291 | 108/560 |

Interpretation: local consistency is useful, but precision is still too low for extreme cross-view. The remaining bottleneck is descriptor/keypoint quality, not only geometric filtering.

### C++ `pfm_cli` Smoke Eval

Checkpoint used:

```text
runs/rotation_clean_topk4_hard_lowlr_2026-05-25/rotation_clean_topk4_hard_lowlr_e1_b2.pt
```

Adaptive smoke output:

```text
runs/cache_match_eval_2026-05-25_cpp_adaptive_smoke/summary.csv
```

Four-pair 1024-cache smoke result:

| mode | matches | correct | wrong | precision |
|---|---:|---:|---:|---:|
| adaptive | 798 | 418 | 380 | 0.523810 |
| projective | 454 | 402 | 52 | 0.885463 |
| local | 821 | 421 | 400 | 0.512789 |

This smoke result is dominated by one easy pair (`400/400`). On hard tail samples, local/adaptive gets more correct matches but also many wrong matches. This confirms the current issue: extreme viewpoint matching still needs a stronger descriptor/keypoint extractor.

## Current Bottlenecks

1. Extreme cross-view descriptors are not invariant enough.
   The real hard case is weak texture, large viewpoint change, changing illumination, and non-global deformation. Current descriptors still create many plausible but wrong high-score matches.

2. Local geometry improves recall but not precision enough.
   RANSAC/projective is too strict for compound deformation, while local displacement consistency allows more true matches but also admits repeated-texture or low-texture false matches.

3. Keypoint selection still needs better valid-region and texture control.
   Texture-aware keypoints help, but hard samples still include unstable points near weak/invalid regions.

4. PyTorch and C++ checkpoints are not yet fully unified.
   PyTorch iteration is faster for model design. Final deployment still needs C++/LibTorch checkpoint compatibility or explicit export path.

5. Data/checkpoints are not in GitHub.
   The 5090 machine needs copied artifacts or regeneration. The full 1024 caches are hundreds of GB.

## Move To RTX 5090 Machine

### Minimum Code Setup

```bash
git clone https://github.com/guderianXu/PlanetaryFeatureMatch.git
cd PlanetaryFeatureMatch
```

Use a fresh conda environment instead of `base`. Install PyTorch with CUDA support matching the 5090 driver/runtime, plus OpenCV, NumPy, and CMake toolchain dependencies.

Then build:

```bash
cmake -S . -B build-pfm-verify-mamba -DCMAKE_BUILD_TYPE=Release
cmake --build build-pfm-verify-mamba --target pfm_cli pfm_tests -j
ctest --test-dir build-pfm-verify-mamba --output-on-failure
```

### Artifact Transfer Choices

Fastest path: copy only the essential artifacts first:

```text
img/Rotate
img/Viewpoint
img/CompoundViewpoint
runs/pytorch_pfm_finetune_2026-05-25_all512_context_300step/pytorch_pfm_state.pt
runs/rotation_clean_topk4_hard_lowlr_2026-05-25/rotation_clean_topk4_hard_lowlr_e1_b2.pt
```

If disk allows, copy the 1024 caches too:

```text
img/Rotate_1024
img/Viewpoint_1024
img/CompoundViewpoint_1024
```

The 1024 caches are much more suitable for the 5090, but total transfer size is large:

```text
img/Rotate_1024              143G
img/Viewpoint_1024            72G
img/CompoundViewpoint_1024    75G
```

### Regeneration Option

If copying is too slow, regenerate caches on the 5090 using the existing cache generation/training commands. Prefer 1024 resize for the 5090 after the 512 baseline is reproduced.

## Recommended Next Work On 5090

1. Reproduce current metrics first.
   Run the Python synthetic-cache evaluator on `Rotate`, `Viewpoint`, and `CompoundViewpoint` with the current best PyTorch state. Confirm the 512-cache numbers match before changing training.

2. Train at 1024 after reproduction.
   Use `Rotate_1024`, `Viewpoint_1024`, and `CompoundViewpoint_1024` together. Keep rotation data in every run so rotation does not regress.

3. Increase model capacity in PyTorch first.
   Priority changes:
   - deeper descriptor tower;
   - multi-scale descriptor head;
   - local patch-level descriptor supervision;
   - correlation/attention layer in descriptor head;
   - stronger valid-mask and low-gray suppression.

4. Improve loss for hard positives and false negatives.
   Current hard-negative mining helps but still accepts many wrong high-score matches. Add local-neighborhood positive supervision and explicit repeated-region hard negatives.

5. Add a stronger evaluation gate.
   Keep separate gates for:
   - rotation;
   - viewpoint;
   - compound viewpoint;
   - real extreme pair.
   Do not optimize only for the real extreme pair until synthetic-cache matching is solid.

6. Export the best PyTorch result back to C++.
   Once PyTorch beats current metrics, either port the architecture into C++/LibTorch or add a robust export path. The final deliverable still needs C++.

## Useful Commands

Python full tests:

```bash
PYTHONPATH=python /path/to/conda/env/bin/python -m unittest discover -s python -p 'test_*.py'
```

C++ full tests:

```bash
ctest --test-dir build-pfm-verify-mamba --output-on-failure
```

C++ adaptive smoke eval:

```bash
PYTHONPATH=python /path/to/python python/cache_match_eval.py \
  --cache-dir img/Viewpoint_1024 \
  --cache-dir img/CompoundViewpoint_1024 \
  --checkpoint runs/rotation_clean_topk4_hard_lowlr_2026-05-25/rotation_clean_topk4_hard_lowlr_e1_b2.pt \
  --pfm-cli ./build-pfm-verify-mamba/pfm_cli \
  --output-dir runs/cache_match_eval_5090_adaptive_smoke \
  --device cuda \
  --sparse-geometry-filter adaptive \
  --descriptor-topk 32 \
  --texture-blend-weight 1 \
  --limit-per-cache 2 \
  --max-keypoints 2048 \
  --keypoint-grid-rows 16 \
  --keypoint-grid-cols 16 \
  --keypoints-per-cell 4 \
  --nms-radius 2
```

## Decision Summary

- Do not go back to SIFT/ORB.
- Do not rely on global RANSAC/projective geometry for extreme compound deformation.
- Keep adaptive geometry as C++ default.
- Use PyTorch for rapid model iteration on the 5090.
- Keep rotation cache in all training mixes.
- Treat current extreme real-pair failure as descriptor/keypoint weakness, not only a post-filter problem.
