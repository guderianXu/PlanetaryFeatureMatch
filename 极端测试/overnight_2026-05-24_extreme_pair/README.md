# Extreme Pair Sweep 2026-05-24

Target pair:

- `20260510T173954657_NAS_PAN_L2b.tif`
- `20260510T191252977_NAS_PAN_L2b.tif`

Current best-count run:

- `iter_042_max8192_grid32_kpc8_nms1_rotation_only_topk64_fastpath`
- sparse matches: 290
- elapsed: 32.037 s

Recommended balanced visual check:

- `iter_040_max8192_grid32_kpc8_rotation_only_topk64_fastpath`
- sparse matches: 282
- elapsed: 29.100 s
- less locally clustered than the nms=1 variant.

Key inference settings for the latest runs:

```bash
PFM_DESCRIPTOR_TOPK_CANDIDATES=64
./build-pfm-verify-mamba/pfm_cli match \
  --sparse-geometry-filter rotation-only \
  --max-keypoints 8192 \
  --keypoint-grid-rows 32 \
  --keypoint-grid-cols 32 \
  --keypoints-per-cell 8 \
  --nms-radius 2
```

Summary files:

- `summary_latest.csv`: all preserved iterations with counts and paths.
- `summary_best_by_matches.csv`: top iterations sorted by sparse match count.

Default projective/RANSAC inference remains unchanged unless `--sparse-geometry-filter rotation-only`
or `PFM_SPARSE_GEOMETRY_FILTER=rotation-only` is set.
