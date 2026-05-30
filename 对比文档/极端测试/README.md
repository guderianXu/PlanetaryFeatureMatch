# 极端测试匹配算法对比

## 数据口径

- 图像 A：`/home/xjw/code/deeplearning/PlanetaryFeatureMatch/对比文档/极端测试/20260510T173954657_NAS_PAN_L2b.tif`。
- 图像 B：`/home/xjw/code/deeplearning/PlanetaryFeatureMatch/对比文档/极端测试/20260510T191252977_NAS_PAN_L2b.tif`。
- 这是一对真实 TIFF 影像，不是 synthetic cache pair；当前目录没有对应的人工/合成 GT warp。
- 因此本页不使用绿色/红色表示正确/错误，也不计算 precision/correct/wrong。
- 所有算法均展示原始 matcher 输出，未执行 RANSAC、Homography、USAC 或其他几何筛选/修复。
- 为控制显存和运行时间，匹配前将长边缩放到 `1600`；CSV 中记录了原始尺寸和缩放后尺寸。
- 可视化只画前若干条原始匹配线，颜色仅为中性显示，不代表对错。

## 运行命令

```bash
PYTHONPATH=python MKL_THREADING_LAYER=GNU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/xjw/anaconda3/envs/pfm-train/bin/python scripts/extreme_case_matcher_comparison.py --device cuda --resize-max 1600
```

## 原始匹配数量

| algorithm | family | status | keypoints A | keypoints B | raw matches | drawn | figure |
|---|---|---|---:|---:|---:|---:|---|
| AKAZE-cross-raw | classical | ok | 1085 | 980 | 256 | 160 | [figures/AKAZE-cross-raw.png](figures/AKAZE-cross-raw.png) |
| ORB-cross-raw | classical | ok | 2048 | 2048 | 256 | 160 | [figures/ORB-cross-raw.png](figures/ORB-cross-raw.png) |
| RootSIFT-r0.80-raw | classical | ok | 2049 | 2048 | 30 | 30 | [figures/RootSIFT-r0.80-raw.png](figures/RootSIFT-r0.80-raw.png) |
| RootSIFT-r0.90-raw | classical | ok | 2049 | 2048 | 256 | 160 | [figures/RootSIFT-r0.90-raw.png](figures/RootSIFT-r0.90-raw.png) |
| SIFT-r0.80-raw | classical | ok | 2049 | 2048 | 49 | 49 | [figures/SIFT-r0.80-raw.png](figures/SIFT-r0.80-raw.png) |
| LightGlue-SIFT-raw | learned | ok | 864 | 847 | 56 | 56 | [figures/LightGlue-SIFT-raw.png](figures/LightGlue-SIFT-raw.png) |
| PFM-current-raw | pfm | ok | 1024 | 1024 | 256 | 160 | [figures/PFM-current-raw.png](figures/PFM-current-raw.png) |
| PFM-latest-p1-viewpoint-raw | pfm | ok | 1024 | 1024 | 256 | 160 | [figures/PFM-latest-p1-viewpoint-raw.png](figures/PFM-latest-p1-viewpoint-raw.png) |

## 不可用项

- SuperGlue: modules 'match_pairs' and 'superglue' unavailable

## 输出文件

- `metrics.csv`: 每个算法的原始匹配数量、关键点数量和可视化路径。
- `summary.csv`: 与 `metrics.csv` 相同口径的简表。
- `skipped_algorithms.csv`: 依赖缺失或初始化失败的算法。
- `figures/`: 原始匹配线可视化。
