# 使用文档

本文档提供 PlanetaryFeatureMatch 的常用命令示例。更详细的训练说明见 `docs/training.md`。

## 构建

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

## 查看帮助

```bash
./build/pfm_cli --help
./build/pfm_cli train --help
./build/pfm_cli extract --help
./build/pfm_cli match --help
./build/pfm_cli eval --help
./build/pfm_cli export --help
```

## 训练

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1 \
  --device cpu \
  --resize 512 \
  --pairs-per-image 4 \
  --augmentation-profile mixed \
  --extreme-pair-ratio 0.2 \
  --synthetic-pair-cache-dir build/pair_cache \
  --log-csv build/train_metrics.csv \
  --visualization-dir build/train_vis \
  --visualization-samples 4 \
  --min-keypoint-intensity 0.0
```

指定 `--synthetic-pair-cache-dir` 后会先生成合成训练对缓存；后续缓存完整且配置匹配时直接复用。未指定缓存且设置 `--dataloader-workers` 大于 0 时，训练会用异步 DataLoader 在线生成 pair；`--prefetch-batches` 控制预取深度，`--pin-memory` 可配合 CUDA 数据搬运。默认训练使用更大的 deep matcher 模型，联合优化 feature extraction、graph matching 和 dense offset refinement，但不通过 CLI 暴露低层结构参数。`--pairs-per-image` 可让每张真实图像生成多组不同变换的合成匹配对，用于增加训练样本。`--min-keypoint-intensity` 会把低于阈值的灰度区域从训练监督中排除，适合屏蔽行星边缘低灰度伪影。`--log-csv` 会写出逐 iteration 指标，便于观察 matcher/dense loss 和 GPU 指标。`--augmentation-profile mixed` 会混合 mild/medium/hard/extreme 强度；想检查极端现象时可用 `--augmentation-profile extreme`。缓存中 PNG 用于查看变换后的图，`.pt` 文件保存训练用的 `view_a`、`view_b`、`warp_a_to_b` 和 `valid_mask`。需要强制重建时添加 `--synthetic-pair-cache-rebuild`。

指定 `--visualization-dir` 后会写训练诊断图，默认前 4 个训练 pair；`--visualization-samples all` 写出全部 pair。每个 pair 输出原始合成视图、有效 mask、warp 监督采样、模型特征点和模型匹配结果，图像左上角标注特征点数、匹配数或有效像素数。

## 特征提取

```bash
./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --device cpu \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5 \
  --visualization-dir vis \
  --min-keypoint-intensity 0.0 \
  --keypoint-grid-rows 8 \
  --keypoint-grid-cols 8 \
  --keypoints-per-cell 0 \
  --nms-radius 4
```

推理阶段的稀疏特征点默认会先应用低灰度过滤，再做局部 NMS，随后按网格分块选点，最后用全局高分候选补足 `--max-keypoints`。`--keypoints-per-cell 0` 表示根据 `max_keypoints` 和网格数量自动推导，采用向上取整且每个 cell 至少 1 个候选。

## 图像匹配

```bash
./build/pfm_cli match \
  --image-a images/a.tif \
  --image-b images/b.tif \
  --checkpoint model.pt \
  --output matches.pt \
  --device cpu \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5 \
  --visualization-dir vis \
  --min-keypoint-intensity 0.0
```

`--visualization-dir` 会自动创建目录并保存 PNG：特征提取保存特征点覆盖图，图像匹配保存左右拼接的匹配连线图。`--min-keypoint-intensity` 会按归一化灰度过滤低灰度区域，减少行星边缘暗背景伪影上的特征点。两个参数都不改变 `.pt` 文件格式。

`extract` 输出 `elapsed`、`image_load`、`model_forward`、`decode`、`save`、`visualization`。`match` 输出两张图的 `extract_a`、`extract_b`、`match_time`、`save`、`visualization`。`eval` 输出 `pairs`、`elapsed`、`avg_pair_time`。`export` 输出 `elapsed`。

## 批量评估

`pairs.txt` 每行包含一对图像路径：

```text
images/a.tif images/b.tif
"/path/with spaces/a.tif" "/path/with spaces/b.tif"
```

运行：

```bash
./build/pfm_cli eval \
  --pairs pairs.txt \
  --checkpoint model.pt \
  --output report.pt \
  --device cpu \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5
```

## 导出

```bash
./build/pfm_cli export \
  --checkpoint model.pt \
  --output exported.pt
```

## CUDA 设备

所有训练/推理命令默认使用 `--device cpu`。如果 LibTorch 是 CUDA 版本，可以使用：

```bash
--device cuda
--device cuda:0
```

`cuda` 等价于 `cuda:0`。CUDA 不可用、索引越界或格式错误时会明确失败，不会静默退回 CPU。当前 CUDA 覆盖训练 forward/backward/loss 和推理模型 forward；图像读取、特征解码、匹配后处理、评估汇总和 `.pt` 写出仍在 CPU。

GPU 训练时可调大 `--resize`、`--batch-size`、`--dataloader-workers` 或增加训练图像数量，以增加每轮计算量并减少 CPU 数据准备等待，例如：

```bash
./build/pfm_cli train \
  --image-dir build/img \
  --checkpoint train.pt \
  --epochs 100 \
  --batch-size 16 \
  --device cuda \
  --resize 512 \
  --pairs-per-image 4 \
  --augmentation-profile mixed \
  --extreme-pair-ratio 0.2 \
  --dataloader-workers 4 \
  --prefetch-batches 4 \
  --log-csv build/train_metrics.csv
```

## 测试程序 `pfm_tests`

`./build/pfm_tests` 是项目的 C++ 单元测试运行器。它会逐个运行模块测试并输出 `PASS <test_name>`，所以看到很多 `PASS` 是正常的。最后一行 `N test(s) passed` 且退出码为 0 表示全部通过；如果失败，会输出 `FAIL <test_name>: <reason>` 并返回非 0。

也可以运行：

```bash
ctest --test-dir build --output-on-failure
```

## 输出文件

- `model.pt`：训练 checkpoint。
- `features.pt`：单图特征。
- `matches.pt`：双图匹配结果。
- `report.pt`：评估报告。
- `exported.pt`：导出后的推理 checkpoint。
