# 训练说明

本文档说明 PlanetaryFeatureMatch 当前第一阶段 C++/LibTorch 训练流程。`train` 命令已经不再是参数校验桩：它会读取真实图像，生成自监督合成图像对，运行最小训练循环，并写出 checkpoint。

## 训练目标

训练一个面向行星影像的局部特征与匹配模型，支持稀疏关键点匹配和半稠密对应关系。当前阶段重点是打通真实图像输入、训练、checkpoint、提取、匹配、评估和导出链路；尚不宣称生产级精度。

需要逐步增强的鲁棒性包括：

- 弱纹理
- 大光照变化
- 相机倾斜
- 多视角几何
- 局部仿射形变
- 成像畸变
- 阴影、无效区域和遮挡

## 输入数据

当前训练从 `--image-dir` 指定目录读取真实图像，并支持常见 OpenCV 可读格式，包括：

- 8 位灰度
- 16 位灰度
- 8 位 RGB/BGR
- 16 位 RGB/BGR

预处理会将图像转换为 `C x H x W` 的 LibTorch 张量，并归一化到 `[0, 1]`。

## 自监督图像对

每张源图像会生成两张相关视图和监督信号：

- `view_a`
- `view_b`
- 从 `view_a` 到 `view_b` 的 dense warp field
- 有效对应 mask

当前第一阶段使用平移和光度扰动生成自监督合成图像对。后续可继续加入旋转、尺度、仿射倾斜、透视变换、畸变、局部形变、阴影、模糊、噪声和遮挡等增强。

## 模型组件

训练流程会运行以下 LibTorch 模块：

- `Backbone`：多尺度特征提取
- `SparseHead`：关键点热力图、描述子、尺度、方向和仿射形状
- `DenseHead`：半稠密置信度和局部偏移
- `Matcher`：描述子相似度与匹配基础

## 损失

当前基础损失包括：

- `repeatability_loss`
- `descriptor_cross_entropy_loss`
- `masked_l1_loss`
- `confidence_bce_loss`

第一阶段训练使用这些损失打通优化流程。几何一致性等更完整目标仍属于后续增强方向。

## 最小训练命令

```bash
./build/pfm_cli train --image-dir images --checkpoint model.pt --epochs 1 --batch-size 1
```

该命令会：

1. 从 `images` 目录读取真实图像。
2. 将图像归一化为 LibTorch 张量。
3. 在线生成平移/光度自监督图像对和有效 mask。
4. 运行 LibTorch 模型前向与反向传播。
5. 保存 checkpoint 到 `model.pt`。

第一阶段 MVP 会对训练图像做 CPU 友好的尺寸限幅，并限制每轮样本数，避免大幅面 TIFF 在本地 smoke 中占用过多内存和时间。

## 相关推理命令

训练完成后，可继续执行：

```bash
./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5

./build/pfm_cli match \
  --image-a images/a.tif \
  --image-b images/b.tif \
  --checkpoint model.pt \
  --output matches.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5

./build/pfm_cli eval \
  --pairs pairs.txt \
  --checkpoint model.pt \
  --output report.pt \
  --max-keypoints 1024

./build/pfm_cli export \
  --checkpoint model.pt \
  --output exported.pt
```

`extract` 输出 `.pt` 特征文件，`match` 输出 `.pt` 匹配结果，`eval` 输出 `.pt` 评估报告，`export` 会校验并复制/重存推理 checkpoint。

## 验证命令

修改训练或推理流程后运行：

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

建议同时使用真实 TIFF 图像执行一次 `train`、`extract`、`match`、`eval` 和 `export` CLI smoke，确认 checkpoint 与 `.pt` 输出文件可以生成。
