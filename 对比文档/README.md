# 六组固定匹配对的算法匹配效果对比

## 数据口径

- 使用之前生成的 6 组固定样本：numeric/timestamp 两种影像风格 x rotate/viewpoint/compound 三个 gate，每组 2 个匹配对，共 12 个 `.pt` pair。
- rotate 组的两个样本固定为 RotationOnly cache 的 90 度和 180 度 pair；`fixed_pairs.csv` 的 `rotation_deg` 列记录角度。
- PFM 使用 `runs/cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234` 的当前 route 参数现场重算匹配点。
- 所有算法都在同一批 pair 上直接匹配 `view_a` 与 `view_b`，没有重新随机采样，也没有额外旋转。
- 外部算法展示原始 matcher 输出；未执行 RANSAC/Homography 几何筛选或修复。
- PFM 展示当前 postselected route 参数下的匹配点，但关闭额外 local/affine geometry filter。
- 可视化统一使用人工合成数据的 GT warp 判定：绿色为正确匹配，红色为错误匹配。
- 正确匹配阈值为 5 px。

## 原始匹配汇总表

| style | gate | algorithm | pairs | matches | correct | wrong | precision |
|---|---|---|---:|---:|---:|---:|---:|
| numeric | compound | AKAZE-cross-Ht3 | 2 | 332 | 128 | 204 | 0.385542 |
| numeric | compound | LightGlue-SIFT-Ht3 | 2 | 377 | 364 | 13 | 0.965517 |
| numeric | compound | ORB-cross-Ht3 | 2 | 512 | 266 | 246 | 0.519531 |
| numeric | compound | PlanetaryFeatureMatch-current | 2 | 233 | 20 | 213 | 0.085837 |
| numeric | compound | RootSIFT-r0.80-Ht2 | 2 | 207 | 179 | 28 | 0.864734 |
| numeric | compound | RootSIFT-r0.90-Ht2 | 2 | 512 | 342 | 170 | 0.667969 |
| numeric | compound | SIFT-r0.80-Ht2 | 2 | 221 | 168 | 53 | 0.760181 |
| numeric | rotate | AKAZE-cross-Ht3 | 2 | 512 | 511 | 1 | 0.998047 |
| numeric | rotate | LightGlue-SIFT-Ht3 | 2 | 658 | 655 | 3 | 0.995441 |
| numeric | rotate | ORB-cross-Ht3 | 2 | 512 | 512 | 0 | 1.000000 |
| numeric | rotate | PlanetaryFeatureMatch-current | 2 | 1024 | 1022 | 2 | 0.998047 |
| numeric | rotate | RootSIFT-r0.80-Ht2 | 2 | 512 | 512 | 0 | 1.000000 |
| numeric | rotate | RootSIFT-r0.90-Ht2 | 2 | 512 | 512 | 0 | 1.000000 |
| numeric | rotate | SIFT-r0.80-Ht2 | 2 | 512 | 512 | 0 | 1.000000 |
| numeric | viewpoint | AKAZE-cross-Ht3 | 2 | 308 | 145 | 163 | 0.470779 |
| numeric | viewpoint | LightGlue-SIFT-Ht3 | 2 | 531 | 516 | 15 | 0.971751 |
| numeric | viewpoint | ORB-cross-Ht3 | 2 | 512 | 306 | 206 | 0.597656 |
| numeric | viewpoint | PlanetaryFeatureMatch-current | 2 | 139 | 22 | 117 | 0.158273 |
| numeric | viewpoint | RootSIFT-r0.80-Ht2 | 2 | 261 | 241 | 20 | 0.923372 |
| numeric | viewpoint | RootSIFT-r0.90-Ht2 | 2 | 477 | 375 | 102 | 0.786164 |
| numeric | viewpoint | SIFT-r0.80-Ht2 | 2 | 259 | 225 | 34 | 0.868726 |
| timestamp | compound | AKAZE-cross-Ht3 | 2 | 143 | 44 | 99 | 0.307692 |
| timestamp | compound | LightGlue-SIFT-Ht3 | 2 | 342 | 315 | 27 | 0.921053 |
| timestamp | compound | ORB-cross-Ht3 | 2 | 447 | 134 | 313 | 0.299776 |
| timestamp | compound | PlanetaryFeatureMatch-current | 2 | 27 | 4 | 23 | 0.148148 |
| timestamp | compound | RootSIFT-r0.80-Ht2 | 2 | 233 | 204 | 29 | 0.875536 |
| timestamp | compound | RootSIFT-r0.90-Ht2 | 2 | 512 | 326 | 186 | 0.636719 |
| timestamp | compound | SIFT-r0.80-Ht2 | 2 | 209 | 176 | 33 | 0.842105 |
| timestamp | rotate | AKAZE-cross-Ht3 | 2 | 427 | 403 | 24 | 0.943794 |
| timestamp | rotate | LightGlue-SIFT-Ht3 | 2 | 472 | 471 | 1 | 0.997881 |
| timestamp | rotate | ORB-cross-Ht3 | 2 | 512 | 512 | 0 | 1.000000 |
| timestamp | rotate | PlanetaryFeatureMatch-current | 2 | 583 | 519 | 64 | 0.890223 |
| timestamp | rotate | RootSIFT-r0.80-Ht2 | 2 | 512 | 512 | 0 | 1.000000 |
| timestamp | rotate | RootSIFT-r0.90-Ht2 | 2 | 512 | 512 | 0 | 1.000000 |
| timestamp | rotate | SIFT-r0.80-Ht2 | 2 | 512 | 512 | 0 | 1.000000 |
| timestamp | viewpoint | AKAZE-cross-Ht3 | 2 | 70 | 51 | 19 | 0.728571 |
| timestamp | viewpoint | LightGlue-SIFT-Ht3 | 2 | 163 | 157 | 6 | 0.963190 |
| timestamp | viewpoint | ORB-cross-Ht3 | 2 | 333 | 175 | 158 | 0.525526 |
| timestamp | viewpoint | PlanetaryFeatureMatch-current | 2 | 127 | 5 | 122 | 0.039370 |
| timestamp | viewpoint | RootSIFT-r0.80-Ht2 | 2 | 257 | 254 | 3 | 0.988327 |
| timestamp | viewpoint | RootSIFT-r0.90-Ht2 | 2 | 314 | 254 | 60 | 0.808917 |
| timestamp | viewpoint | SIFT-r0.80-Ht2 | 2 | 259 | 248 | 11 | 0.957529 |

## 匹配图

- numeric/rotate/rot90 `PlanetaryFeatureMatch-current`: [figures/pfm/numeric/rotate/rot90/rot90_source_000201_72_pair_001587.png](figures/pfm/numeric/rotate/rot90/rot90_source_000201_72_pair_001587.png)
- numeric/rotate/rot90 `SIFT-r0.80-Ht2`: [figures/other_models/numeric/rotate/rot90/SIFT-r0.80-Ht2.png](figures/other_models/numeric/rotate/rot90/SIFT-r0.80-Ht2.png)
- numeric/rotate/rot90 `RootSIFT-r0.80-Ht2`: [figures/other_models/numeric/rotate/rot90/RootSIFT-r0.80-Ht2.png](figures/other_models/numeric/rotate/rot90/RootSIFT-r0.80-Ht2.png)
- numeric/rotate/rot90 `RootSIFT-r0.90-Ht2`: [figures/other_models/numeric/rotate/rot90/RootSIFT-r0.90-Ht2.png](figures/other_models/numeric/rotate/rot90/RootSIFT-r0.90-Ht2.png)
- numeric/rotate/rot90 `ORB-cross-Ht3`: [figures/other_models/numeric/rotate/rot90/ORB-cross-Ht3.png](figures/other_models/numeric/rotate/rot90/ORB-cross-Ht3.png)
- numeric/rotate/rot90 `AKAZE-cross-Ht3`: [figures/other_models/numeric/rotate/rot90/AKAZE-cross-Ht3.png](figures/other_models/numeric/rotate/rot90/AKAZE-cross-Ht3.png)
- numeric/rotate/rot90 `LightGlue-SIFT-Ht3`: [figures/other_models/numeric/rotate/rot90/LightGlue-SIFT-Ht3.png](figures/other_models/numeric/rotate/rot90/LightGlue-SIFT-Ht3.png)
- numeric/rotate/rot180 `PlanetaryFeatureMatch-current`: [figures/pfm/numeric/rotate/rot180/rot180_source_000007_105_pair_002779.png](figures/pfm/numeric/rotate/rot180/rot180_source_000007_105_pair_002779.png)
- numeric/rotate/rot180 `SIFT-r0.80-Ht2`: [figures/other_models/numeric/rotate/rot180/SIFT-r0.80-Ht2.png](figures/other_models/numeric/rotate/rot180/SIFT-r0.80-Ht2.png)
- numeric/rotate/rot180 `RootSIFT-r0.80-Ht2`: [figures/other_models/numeric/rotate/rot180/RootSIFT-r0.80-Ht2.png](figures/other_models/numeric/rotate/rot180/RootSIFT-r0.80-Ht2.png)
- numeric/rotate/rot180 `RootSIFT-r0.90-Ht2`: [figures/other_models/numeric/rotate/rot180/RootSIFT-r0.90-Ht2.png](figures/other_models/numeric/rotate/rot180/RootSIFT-r0.90-Ht2.png)
- numeric/rotate/rot180 `ORB-cross-Ht3`: [figures/other_models/numeric/rotate/rot180/ORB-cross-Ht3.png](figures/other_models/numeric/rotate/rot180/ORB-cross-Ht3.png)
- numeric/rotate/rot180 `AKAZE-cross-Ht3`: [figures/other_models/numeric/rotate/rot180/AKAZE-cross-Ht3.png](figures/other_models/numeric/rotate/rot180/AKAZE-cross-Ht3.png)
- numeric/rotate/rot180 `LightGlue-SIFT-Ht3`: [figures/other_models/numeric/rotate/rot180/LightGlue-SIFT-Ht3.png](figures/other_models/numeric/rotate/rot180/LightGlue-SIFT-Ht3.png)
- timestamp/rotate/rot90 `PlanetaryFeatureMatch-current`: [figures/pfm/timestamp/rotate/rot90/rot90_source_000123_20260514T144405909_NAS_PAN_L2b_pair_001509.png](figures/pfm/timestamp/rotate/rot90/rot90_source_000123_20260514T144405909_NAS_PAN_L2b_pair_001509.png)
- timestamp/rotate/rot90 `SIFT-r0.80-Ht2`: [figures/other_models/timestamp/rotate/rot90/SIFT-r0.80-Ht2.png](figures/other_models/timestamp/rotate/rot90/SIFT-r0.80-Ht2.png)
- timestamp/rotate/rot90 `RootSIFT-r0.80-Ht2`: [figures/other_models/timestamp/rotate/rot90/RootSIFT-r0.80-Ht2.png](figures/other_models/timestamp/rotate/rot90/RootSIFT-r0.80-Ht2.png)
- timestamp/rotate/rot90 `RootSIFT-r0.90-Ht2`: [figures/other_models/timestamp/rotate/rot90/RootSIFT-r0.90-Ht2.png](figures/other_models/timestamp/rotate/rot90/RootSIFT-r0.90-Ht2.png)
- timestamp/rotate/rot90 `ORB-cross-Ht3`: [figures/other_models/timestamp/rotate/rot90/ORB-cross-Ht3.png](figures/other_models/timestamp/rotate/rot90/ORB-cross-Ht3.png)
- timestamp/rotate/rot90 `AKAZE-cross-Ht3`: [figures/other_models/timestamp/rotate/rot90/AKAZE-cross-Ht3.png](figures/other_models/timestamp/rotate/rot90/AKAZE-cross-Ht3.png)
- timestamp/rotate/rot90 `LightGlue-SIFT-Ht3`: [figures/other_models/timestamp/rotate/rot90/LightGlue-SIFT-Ht3.png](figures/other_models/timestamp/rotate/rot90/LightGlue-SIFT-Ht3.png)
- timestamp/rotate/rot180 `PlanetaryFeatureMatch-current`: [figures/pfm/timestamp/rotate/rot180/rot180_source_000088_20260514T070226673_NAS_PAN_L2b_pair_002860.png](figures/pfm/timestamp/rotate/rot180/rot180_source_000088_20260514T070226673_NAS_PAN_L2b_pair_002860.png)
- timestamp/rotate/rot180 `SIFT-r0.80-Ht2`: [figures/other_models/timestamp/rotate/rot180/SIFT-r0.80-Ht2.png](figures/other_models/timestamp/rotate/rot180/SIFT-r0.80-Ht2.png)
- timestamp/rotate/rot180 `RootSIFT-r0.80-Ht2`: [figures/other_models/timestamp/rotate/rot180/RootSIFT-r0.80-Ht2.png](figures/other_models/timestamp/rotate/rot180/RootSIFT-r0.80-Ht2.png)
- timestamp/rotate/rot180 `RootSIFT-r0.90-Ht2`: [figures/other_models/timestamp/rotate/rot180/RootSIFT-r0.90-Ht2.png](figures/other_models/timestamp/rotate/rot180/RootSIFT-r0.90-Ht2.png)
- timestamp/rotate/rot180 `ORB-cross-Ht3`: [figures/other_models/timestamp/rotate/rot180/ORB-cross-Ht3.png](figures/other_models/timestamp/rotate/rot180/ORB-cross-Ht3.png)
- timestamp/rotate/rot180 `AKAZE-cross-Ht3`: [figures/other_models/timestamp/rotate/rot180/AKAZE-cross-Ht3.png](figures/other_models/timestamp/rotate/rot180/AKAZE-cross-Ht3.png)
- timestamp/rotate/rot180 `LightGlue-SIFT-Ht3`: [figures/other_models/timestamp/rotate/rot180/LightGlue-SIFT-Ht3.png](figures/other_models/timestamp/rotate/rot180/LightGlue-SIFT-Ht3.png)
- numeric/viewpoint/sample01 `PlanetaryFeatureMatch-current`: [figures/pfm/numeric/viewpoint/01/01_source_000201_72_pair_002049.png](figures/pfm/numeric/viewpoint/01/01_source_000201_72_pair_002049.png)
- numeric/viewpoint/sample01 `SIFT-r0.80-Ht2`: [figures/other_models/numeric/viewpoint/01/SIFT-r0.80-Ht2.png](figures/other_models/numeric/viewpoint/01/SIFT-r0.80-Ht2.png)
- numeric/viewpoint/sample01 `RootSIFT-r0.80-Ht2`: [figures/other_models/numeric/viewpoint/01/RootSIFT-r0.80-Ht2.png](figures/other_models/numeric/viewpoint/01/RootSIFT-r0.80-Ht2.png)
- numeric/viewpoint/sample01 `RootSIFT-r0.90-Ht2`: [figures/other_models/numeric/viewpoint/01/RootSIFT-r0.90-Ht2.png](figures/other_models/numeric/viewpoint/01/RootSIFT-r0.90-Ht2.png)
- numeric/viewpoint/sample01 `ORB-cross-Ht3`: [figures/other_models/numeric/viewpoint/01/ORB-cross-Ht3.png](figures/other_models/numeric/viewpoint/01/ORB-cross-Ht3.png)
- numeric/viewpoint/sample01 `AKAZE-cross-Ht3`: [figures/other_models/numeric/viewpoint/01/AKAZE-cross-Ht3.png](figures/other_models/numeric/viewpoint/01/AKAZE-cross-Ht3.png)
- numeric/viewpoint/sample01 `LightGlue-SIFT-Ht3`: [figures/other_models/numeric/viewpoint/01/LightGlue-SIFT-Ht3.png](figures/other_models/numeric/viewpoint/01/LightGlue-SIFT-Ht3.png)
- numeric/viewpoint/sample02 `PlanetaryFeatureMatch-current`: [figures/pfm/numeric/viewpoint/02/02_source_000007_105_pair_000238.png](figures/pfm/numeric/viewpoint/02/02_source_000007_105_pair_000238.png)
- numeric/viewpoint/sample02 `SIFT-r0.80-Ht2`: [figures/other_models/numeric/viewpoint/02/SIFT-r0.80-Ht2.png](figures/other_models/numeric/viewpoint/02/SIFT-r0.80-Ht2.png)
- numeric/viewpoint/sample02 `RootSIFT-r0.80-Ht2`: [figures/other_models/numeric/viewpoint/02/RootSIFT-r0.80-Ht2.png](figures/other_models/numeric/viewpoint/02/RootSIFT-r0.80-Ht2.png)
- numeric/viewpoint/sample02 `RootSIFT-r0.90-Ht2`: [figures/other_models/numeric/viewpoint/02/RootSIFT-r0.90-Ht2.png](figures/other_models/numeric/viewpoint/02/RootSIFT-r0.90-Ht2.png)
- numeric/viewpoint/sample02 `ORB-cross-Ht3`: [figures/other_models/numeric/viewpoint/02/ORB-cross-Ht3.png](figures/other_models/numeric/viewpoint/02/ORB-cross-Ht3.png)
- numeric/viewpoint/sample02 `AKAZE-cross-Ht3`: [figures/other_models/numeric/viewpoint/02/AKAZE-cross-Ht3.png](figures/other_models/numeric/viewpoint/02/AKAZE-cross-Ht3.png)
- numeric/viewpoint/sample02 `LightGlue-SIFT-Ht3`: [figures/other_models/numeric/viewpoint/02/LightGlue-SIFT-Ht3.png](figures/other_models/numeric/viewpoint/02/LightGlue-SIFT-Ht3.png)
- timestamp/viewpoint/sample01 `PlanetaryFeatureMatch-current`: [figures/pfm/timestamp/viewpoint/01/01_source_000123_20260514T144405909_NAS_PAN_L2b_pair_003819.png](figures/pfm/timestamp/viewpoint/01/01_source_000123_20260514T144405909_NAS_PAN_L2b_pair_003819.png)
- timestamp/viewpoint/sample01 `SIFT-r0.80-Ht2`: [figures/other_models/timestamp/viewpoint/01/SIFT-r0.80-Ht2.png](figures/other_models/timestamp/viewpoint/01/SIFT-r0.80-Ht2.png)
- timestamp/viewpoint/sample01 `RootSIFT-r0.80-Ht2`: [figures/other_models/timestamp/viewpoint/01/RootSIFT-r0.80-Ht2.png](figures/other_models/timestamp/viewpoint/01/RootSIFT-r0.80-Ht2.png)
- timestamp/viewpoint/sample01 `RootSIFT-r0.90-Ht2`: [figures/other_models/timestamp/viewpoint/01/RootSIFT-r0.90-Ht2.png](figures/other_models/timestamp/viewpoint/01/RootSIFT-r0.90-Ht2.png)
- timestamp/viewpoint/sample01 `ORB-cross-Ht3`: [figures/other_models/timestamp/viewpoint/01/ORB-cross-Ht3.png](figures/other_models/timestamp/viewpoint/01/ORB-cross-Ht3.png)
- timestamp/viewpoint/sample01 `AKAZE-cross-Ht3`: [figures/other_models/timestamp/viewpoint/01/AKAZE-cross-Ht3.png](figures/other_models/timestamp/viewpoint/01/AKAZE-cross-Ht3.png)
- timestamp/viewpoint/sample01 `LightGlue-SIFT-Ht3`: [figures/other_models/timestamp/viewpoint/01/LightGlue-SIFT-Ht3.png](figures/other_models/timestamp/viewpoint/01/LightGlue-SIFT-Ht3.png)
- timestamp/viewpoint/sample02 `PlanetaryFeatureMatch-current`: [figures/pfm/timestamp/viewpoint/02/02_source_000088_20260514T070226673_NAS_PAN_L2b_pair_004708.png](figures/pfm/timestamp/viewpoint/02/02_source_000088_20260514T070226673_NAS_PAN_L2b_pair_004708.png)
- timestamp/viewpoint/sample02 `SIFT-r0.80-Ht2`: [figures/other_models/timestamp/viewpoint/02/SIFT-r0.80-Ht2.png](figures/other_models/timestamp/viewpoint/02/SIFT-r0.80-Ht2.png)
- timestamp/viewpoint/sample02 `RootSIFT-r0.80-Ht2`: [figures/other_models/timestamp/viewpoint/02/RootSIFT-r0.80-Ht2.png](figures/other_models/timestamp/viewpoint/02/RootSIFT-r0.80-Ht2.png)
- timestamp/viewpoint/sample02 `RootSIFT-r0.90-Ht2`: [figures/other_models/timestamp/viewpoint/02/RootSIFT-r0.90-Ht2.png](figures/other_models/timestamp/viewpoint/02/RootSIFT-r0.90-Ht2.png)
- timestamp/viewpoint/sample02 `ORB-cross-Ht3`: [figures/other_models/timestamp/viewpoint/02/ORB-cross-Ht3.png](figures/other_models/timestamp/viewpoint/02/ORB-cross-Ht3.png)
- timestamp/viewpoint/sample02 `AKAZE-cross-Ht3`: [figures/other_models/timestamp/viewpoint/02/AKAZE-cross-Ht3.png](figures/other_models/timestamp/viewpoint/02/AKAZE-cross-Ht3.png)
- timestamp/viewpoint/sample02 `LightGlue-SIFT-Ht3`: [figures/other_models/timestamp/viewpoint/02/LightGlue-SIFT-Ht3.png](figures/other_models/timestamp/viewpoint/02/LightGlue-SIFT-Ht3.png)
- numeric/compound/sample01 `PlanetaryFeatureMatch-current`: [figures/pfm/numeric/compound/01/01_source_000201_72_pair_002049.png](figures/pfm/numeric/compound/01/01_source_000201_72_pair_002049.png)
- numeric/compound/sample01 `SIFT-r0.80-Ht2`: [figures/other_models/numeric/compound/01/SIFT-r0.80-Ht2.png](figures/other_models/numeric/compound/01/SIFT-r0.80-Ht2.png)
- numeric/compound/sample01 `RootSIFT-r0.80-Ht2`: [figures/other_models/numeric/compound/01/RootSIFT-r0.80-Ht2.png](figures/other_models/numeric/compound/01/RootSIFT-r0.80-Ht2.png)
- numeric/compound/sample01 `RootSIFT-r0.90-Ht2`: [figures/other_models/numeric/compound/01/RootSIFT-r0.90-Ht2.png](figures/other_models/numeric/compound/01/RootSIFT-r0.90-Ht2.png)
- numeric/compound/sample01 `ORB-cross-Ht3`: [figures/other_models/numeric/compound/01/ORB-cross-Ht3.png](figures/other_models/numeric/compound/01/ORB-cross-Ht3.png)
- numeric/compound/sample01 `AKAZE-cross-Ht3`: [figures/other_models/numeric/compound/01/AKAZE-cross-Ht3.png](figures/other_models/numeric/compound/01/AKAZE-cross-Ht3.png)
- numeric/compound/sample01 `LightGlue-SIFT-Ht3`: [figures/other_models/numeric/compound/01/LightGlue-SIFT-Ht3.png](figures/other_models/numeric/compound/01/LightGlue-SIFT-Ht3.png)
- numeric/compound/sample02 `PlanetaryFeatureMatch-current`: [figures/pfm/numeric/compound/02/02_source_000007_105_pair_000238.png](figures/pfm/numeric/compound/02/02_source_000007_105_pair_000238.png)
- numeric/compound/sample02 `SIFT-r0.80-Ht2`: [figures/other_models/numeric/compound/02/SIFT-r0.80-Ht2.png](figures/other_models/numeric/compound/02/SIFT-r0.80-Ht2.png)
- numeric/compound/sample02 `RootSIFT-r0.80-Ht2`: [figures/other_models/numeric/compound/02/RootSIFT-r0.80-Ht2.png](figures/other_models/numeric/compound/02/RootSIFT-r0.80-Ht2.png)
- numeric/compound/sample02 `RootSIFT-r0.90-Ht2`: [figures/other_models/numeric/compound/02/RootSIFT-r0.90-Ht2.png](figures/other_models/numeric/compound/02/RootSIFT-r0.90-Ht2.png)
- numeric/compound/sample02 `ORB-cross-Ht3`: [figures/other_models/numeric/compound/02/ORB-cross-Ht3.png](figures/other_models/numeric/compound/02/ORB-cross-Ht3.png)
- numeric/compound/sample02 `AKAZE-cross-Ht3`: [figures/other_models/numeric/compound/02/AKAZE-cross-Ht3.png](figures/other_models/numeric/compound/02/AKAZE-cross-Ht3.png)
- numeric/compound/sample02 `LightGlue-SIFT-Ht3`: [figures/other_models/numeric/compound/02/LightGlue-SIFT-Ht3.png](figures/other_models/numeric/compound/02/LightGlue-SIFT-Ht3.png)
- timestamp/compound/sample01 `PlanetaryFeatureMatch-current`: [figures/pfm/timestamp/compound/01/01_source_000123_20260514T144405909_NAS_PAN_L2b_pair_003819.png](figures/pfm/timestamp/compound/01/01_source_000123_20260514T144405909_NAS_PAN_L2b_pair_003819.png)
- timestamp/compound/sample01 `SIFT-r0.80-Ht2`: [figures/other_models/timestamp/compound/01/SIFT-r0.80-Ht2.png](figures/other_models/timestamp/compound/01/SIFT-r0.80-Ht2.png)
- timestamp/compound/sample01 `RootSIFT-r0.80-Ht2`: [figures/other_models/timestamp/compound/01/RootSIFT-r0.80-Ht2.png](figures/other_models/timestamp/compound/01/RootSIFT-r0.80-Ht2.png)
- timestamp/compound/sample01 `RootSIFT-r0.90-Ht2`: [figures/other_models/timestamp/compound/01/RootSIFT-r0.90-Ht2.png](figures/other_models/timestamp/compound/01/RootSIFT-r0.90-Ht2.png)
- timestamp/compound/sample01 `ORB-cross-Ht3`: [figures/other_models/timestamp/compound/01/ORB-cross-Ht3.png](figures/other_models/timestamp/compound/01/ORB-cross-Ht3.png)
- timestamp/compound/sample01 `AKAZE-cross-Ht3`: [figures/other_models/timestamp/compound/01/AKAZE-cross-Ht3.png](figures/other_models/timestamp/compound/01/AKAZE-cross-Ht3.png)
- timestamp/compound/sample01 `LightGlue-SIFT-Ht3`: [figures/other_models/timestamp/compound/01/LightGlue-SIFT-Ht3.png](figures/other_models/timestamp/compound/01/LightGlue-SIFT-Ht3.png)
- timestamp/compound/sample02 `PlanetaryFeatureMatch-current`: [figures/pfm/timestamp/compound/02/02_source_000088_20260514T070226673_NAS_PAN_L2b_pair_004708.png](figures/pfm/timestamp/compound/02/02_source_000088_20260514T070226673_NAS_PAN_L2b_pair_004708.png)
- timestamp/compound/sample02 `SIFT-r0.80-Ht2`: [figures/other_models/timestamp/compound/02/SIFT-r0.80-Ht2.png](figures/other_models/timestamp/compound/02/SIFT-r0.80-Ht2.png)
- timestamp/compound/sample02 `RootSIFT-r0.80-Ht2`: [figures/other_models/timestamp/compound/02/RootSIFT-r0.80-Ht2.png](figures/other_models/timestamp/compound/02/RootSIFT-r0.80-Ht2.png)
- timestamp/compound/sample02 `RootSIFT-r0.90-Ht2`: [figures/other_models/timestamp/compound/02/RootSIFT-r0.90-Ht2.png](figures/other_models/timestamp/compound/02/RootSIFT-r0.90-Ht2.png)
- timestamp/compound/sample02 `ORB-cross-Ht3`: [figures/other_models/timestamp/compound/02/ORB-cross-Ht3.png](figures/other_models/timestamp/compound/02/ORB-cross-Ht3.png)
- timestamp/compound/sample02 `AKAZE-cross-Ht3`: [figures/other_models/timestamp/compound/02/AKAZE-cross-Ht3.png](figures/other_models/timestamp/compound/02/AKAZE-cross-Ht3.png)
- timestamp/compound/sample02 `LightGlue-SIFT-Ht3`: [figures/other_models/timestamp/compound/02/LightGlue-SIFT-Ht3.png](figures/other_models/timestamp/compound/02/LightGlue-SIFT-Ht3.png)

## 不可用项

- SuperGlue: modules 'match_pairs' and 'superglue' unavailable

## 原始文件

- `fixed_pairs.csv`: 固定 12 个 pair 的路径。
- `metrics.csv`: 每个算法原始匹配的匹配数、正确数、precision 和可视化路径。
- `summary.csv`: 按 style/gate/algorithm 聚合的原始匹配结果。
- `figures/`: 原始匹配可视化图。
