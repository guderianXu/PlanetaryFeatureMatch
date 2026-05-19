# Training Diagnostics Visualization Design

## Goal

Add training-time diagnostic visualization so we can inspect whether bad inference matches come from synthetic training data, descriptor learning, feature distribution, or match filtering.

## User-facing behavior

`train` gains two options:

```bash
--visualization-dir <dir>
--visualization-samples <N|all>
```

Behavior:

- If `--visualization-dir` is empty, training behaves exactly as today.
- If `--visualization-dir` is set and `--visualization-samples` is omitted, save diagnostics for the first 4 training pairs.
- If `--visualization-samples N` is set, save diagnostics for the first `N` training pairs.
- If `--visualization-samples all` is set, save diagnostics for every training pair.
- Use a larger async writer queue by default because the target machine has enough memory. Start with capacity 256 tasks.

## Diagnostics to write

For each selected training pair, write files under:

```text
<visualization-dir>/
  pair_000000_view_a.png
  pair_000000_view_b.png
  pair_000000_valid_mask.png
  pair_000000_warp_matches.png
  pair_000000_features_a.png
  pair_000000_features_b.png
  pair_000000_model_matches.png
```

Each output image should contain a text overlay in the upper-left corner:

- feature visualizations: `features=<count>`
- warp supervision visualization: `matches=<sampled_correspondence_count> valid=<valid_pixel_count>/<total_pixel_count>`
- model match visualization: `features_a=<count> features_b=<count> sparse_matches=<count> dense_matches=<count>`

## Data diagnostics

The data-side diagnostics inspect the actual synthetic pairs used for training:

- `view_a` and `view_b`: grayscale synthetic training views.
- `valid_mask`: valid supervision mask.
- `warp_matches`: sampled point correspondences from `warp_a_to_b` where `valid_mask` is true.

This directly answers whether the synthetic dataset is reasonable and whether training supervision covers the useful object area.

## Model diagnostics

The model-side diagnostics run the current model on selected training pairs and decode features using the same feature decode settings used by inference defaults unless train later exposes more decode knobs.

For each selected pair:

- extract features from `view_a` and `view_b` using the current model state;
- save feature overlays for both views;
- match the extracted features with the same matching pipeline used by `match`;
- save sparse/dense match visualization.

These images show whether the model is learning useful feature distributions and whether descriptor matches are becoming geometrically consistent.

## Async image writer

Visualization must not synchronously block the training loop on drawing and PNG writing.

Design:

- Add a small `AsyncVisualizationWriter` module used by training diagnostics.
- Training thread creates diagnostic jobs with CPU tensors / metadata and pushes them into a bounded queue.
- One writer thread drains the queue and performs OpenCV drawing + `imwrite`.
- Queue capacity defaults to 256 jobs.
- If the queue is full, the training thread waits until space is available. This bounds memory while still allowing substantial buffering.
- On train completion or error, the writer flushes pending jobs and joins before returning from `train_model`.
- Exceptions in the writer thread are captured and rethrown on join so failed visualization does not silently pass.

## Training progress output

When diagnostics are enabled, print a short note at training start:

```text
training visualization: dir=<dir> samples=<N|all> async_queue=256
```

Existing per-batch progress remains unchanged.

## Scope boundaries

This first version is diagnostic, not a full training redesign.

In scope:

- training data visualizations;
- model feature/match visualizations on selected training pairs;
- count overlays;
- async image writing;
- tests and README/docs updates.

Out of scope for this spec:

- changing model architecture;
- adding validation/test dataset split;
- adding new loss functions;
- adding RANSAC or homography verification to inference matching;
- GUI or HTML reports.

## Testing requirements

Add tests for:

- CLI parses `--visualization-dir` and default `--visualization-samples=4`.
- CLI parses `--visualization-samples all`.
- invalid sample values throw parse errors or validation errors.
- training with visualization writes the expected diagnostic PNG files for sampled pairs.
- `--visualization-samples all` writes diagnostics for every training pair in a tiny dataset.
- async writer flushes all queued files before `train_model` returns.
- count overlays are present enough to detect non-background text pixels in the expected upper-left area.

## README update

Document examples:

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --visualization-dir build/train_vis
```

Full diagnostic output:

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --visualization-dir build/train_vis \
  --visualization-samples all
```
