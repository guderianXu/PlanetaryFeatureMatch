# Cross-Domain Lazy Pair Sampling Design

## Goal

Expand lazy pose-pair training so it can sample useful pairs across viewpoint, camera position, and field-of-view domains without materializing a full pair cache.

## Scope

This change targets the Python lazy training path in `scripts/benchmark_lazy_pose_pairs.py`. It does not change C++ training, the simulator, or existing render manifest generation. Existing same-position behavior remains the default-compatible building block and should still be available explicitly.

## Current Behavior

The current pair builder groups manifest rows by `base_id` and pairs one `reference_variant` with each configured `target_variant`. With the current manifests this means `nadir -> small/mid/extreme` inside the same camera position and same source manifest. When multiple render manifests are passed, `base_id` receives a manifest prefix, which prevents accidental fov090-to-fov110 pairing.

This is useful for viewpoint changes but does not train different camera positions or cross-FOV pairs.

## Target Pair Families

The trainer should support three pair families:

- `same_position_view`: current same-`base_id` viewpoint pairs, such as `b000115/nadir -> b000115/extreme_02`.
- `cross_camera`: same manifest group, different `base_id`, any configured reference/target variants, such as `fov090/b000115/nadir -> fov090/b000116/mid_01`.
- `cross_fov`: different manifest groups, different or same `base_id`, any configured reference/target variants, such as `fov090/b000115/small_01 -> fov110/b000058/extreme_03`.

`cross_camera` and `cross_fov` rely on existing TSAI and depth projection. Low-overlap candidates are filtered later by the existing lazy pair generation path using `min_valid_fraction`, `absolute_depth_tolerance_m`, and `relative_depth_tolerance`.

## Sampling Strategy

The implementation should avoid full Cartesian products. Candidate construction should use controlled offsets and optional caps:

- `--pair-mode same-position`, `cross-camera`, `cross-fov`, or `mixed`.
- `--cross-camera-offsets`, defaulting to a small balanced set such as `1,2,4,8`.
- `--cross-fov-offsets`, defaulting to `0,1,2,4`, pairing sorted records between manifest groups by index offset.
- `--cross-pair-variants`, defaulting to reference plus configured target variants, to allow same-view and different-view cross-domain pairs.
- `--pair-type-weights`, defaulting to a mixed distribution such as `same_position_view=0.40,cross_camera=0.35,cross_fov=0.25`.

For `mixed`, the trainer should interleave or weighted-sample from each family so the training stream is not dominated by whichever family has the most candidates.

## Manifest Identity

The reader should keep a stable `dataset_id` for each render manifest. `dataset_id` should come from the manifest parent directory name when multiple manifests are passed, matching the existing base-prefix behavior. Pair specs should record a `pair_type` so summaries, CSV rows, and run HTML distinguish same-position, cross-camera, and cross-FOV samples.

## Reporting

Training artifacts should record pair construction choices:

- `input_summary.json`: `pair_mode`, offsets, variant list, type weights, per-type spec counts.
- `train_metrics.csv`: per-step counts for consumed pair types.
- `run.html`: existing argument dump plus summary fields are enough if the metadata contains the fields above.

## Error Handling

Invalid offsets, empty weight entries, negative weights, or a pair mode with zero generated specs should fail early with a clear message. If `cross-fov` is requested with fewer than two render manifests, the trainer should fail before training starts.

## Tests

Focused Python unit tests should cover:

- existing same-position behavior still works;
- cross-camera specs pair different `base_id` values within one dataset;
- cross-FOV specs pair different `dataset_id` values;
- mixed specs report per-type counts and respect weights enough to sample from all non-empty requested families;
- invalid cross-FOV mode with one manifest raises a clear error.

## Non-Goals

This design does not generate new images, re-render depth maps, train a new model immediately, or change validation/evaluation metrics. It only makes the lazy training data stream capable of using broader pair domains.
