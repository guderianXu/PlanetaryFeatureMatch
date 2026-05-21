# Rotation Matching Final State

## Goal

Build a deep-learning rotation matching pipeline for planetary imagery. The final pipeline uses synthetic rotation-only pairs with known pixel correspondences from `warp_a_to_b`; it does not use SIFT or ORB.

## Final Approach

- Rotation-only synthetic data is generated from source images with exact pixel correspondence.
- Training and inference use a rotation-invariant texture descriptor target.
- The descriptor includes center-radius features, which are invariant for in-plane rotations around the image center.
- Sparse matching uses mutual normalized descriptor similarity.
- A global rotation-consistency filter keeps matches whose polar-angle rotation and radius are consistent.
- Final rotation filter radius tolerance: `2.0` feature-map pixels.
- Recommended high-density extraction settings:

```bash
--max-keypoints 8192 --min-keypoints 8192 --min-keypoint-intensity 0.05
```

## Final Local Artifacts

Final validated 15-degree sweep:

```text
runs/rotation_only_allimg_2026-05-21/rotation_sweep_100_step15_skip_e1_radiusdesc_r2_k4096
```

High-density 8192-keypoint sweep:

```text
runs/rotation_only_allimg_2026-05-21/rotation_sweep_100_step30_skip_e1_radiusdesc_r2_k8192
```

Final checkpoint:

```text
runs/rotation_only_allimg_2026-05-21/train_rotation_only_allimg_skip_e1.pt
```

Rotation-only training cache:

```text
img/traindata
```

## Final 4096-Keypoint Validation

Validated with `img/100.tif`, every 15 degrees from 0 to 345 degrees.

Representative sparse match counts and pass rates:

| angle | matches | pass_rate |
|---:|---:|---:|
| 15 | 704 | 0.961648 |
| 30 | 513 | 0.935673 |
| 60 | 451 | 0.942350 |
| 90 | 331 | 0.924471 |
| 180 | 2830 | 0.998587 |
| 225 | 438 | 0.922374 |
| 270 | 332 | 0.915663 |
| 315 | 468 | 0.940171 |
| 345 | 668 | 0.973054 |

## 8192-Keypoint Match Count Check

The 8192-keypoint setting roughly doubles the number of sparse matches. The final full summary was intentionally not kept because descriptor-mutual statistics at `8192 x 8192` are slow on CPU, but the match outputs and visualizations are generated.

Observed 30-degree sweep match counts:

| angle | sparse matches |
|---:|---:|
| 30 | 1011 |
| 60 | 978 |
| 90 | 676 |
| 180 | 6615 |
| 240 | 947 |
| 270 | 667 |
| 300 | 938 |
| 330 | 1026 |

## Verification

The final code state passed:

```text
330 test(s) passed
```
