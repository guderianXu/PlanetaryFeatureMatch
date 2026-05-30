# Weak Texture Coverage Sampling Implementation Plan

Goal: add a coverage-aware keypoint selection mode that improves weak-texture and spatially uniform match support without changing the trained backbone first.

Tasks:
1. Add tests for weak-texture quota and per-cell cap behavior in `python/test_pytorch_cache_match_eval.py`.
2. Extend `select_descriptor_keypoints()` with `weak_texture_fraction` and `per_cell_cap`, keeping old defaults unchanged.
3. Expose the parameters through `pytorch_cache_match_eval.py` and `training_visual_report.py` CLI.
4. Add coverage metrics/heatmaps to the training report so weak texture and spatial distribution are visible.
5. Run tests, regenerate reports with the new mode, sync latest generated data, then start a new full-256 training pass using the updated reporting configuration.
