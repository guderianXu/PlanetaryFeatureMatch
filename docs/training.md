# 训练与使用说明

本文档说明 PlanetaryFeatureMatch 当前第一阶段 C++/LibTorch 训练、推理和评估流程。当前实现已经不是参数校验桩，而是可以读取真实图像、执行最小训练循环、写出 checkpoint，并继续完成特征提取、匹配、评估和导出。

## 目标

PlanetaryFeatureMatch 面向火星、月球和小行星影像，目标是训练一个同时支持稀疏关键点匹配和半稠密对应的局部特征模型。第一阶段的目标是打通真实图像闭环：

1. 从真实影像目录读取数据。
2. 生成自监督图像对。
3. 训练最小 LibTorch 模型。
4. 保存可加载 checkpoint。
5. 使用 checkpoint 提取特征。
6. 导出匹配结果和评估报告。

该阶段重点是工程闭环和测试覆盖，后续还需要继续提升匹配精度和几何鲁棒性。

## 输入图像

训练和推理通过 OpenCV 读取图像。当前支持 OpenCV 可读取的常见格式，主要包括：

- `.png`
- `.jpg` / `.jpeg`
- `.tif` / `.tiff`

支持的像素类型包括：

- 8 位灰度
- 16 位灰度
- 8 位 RGB/BGR
- 16 位 RGB/BGR

图像会转换为 `C x H x W` 的 LibTorch float tensor，并归一化到 `[0, 1]`。彩色图像会按通道处理；训练时当前会转为单通道灰度输入。

## 自监督训练数据

当前训练不依赖人工标注匹配点，而是从单张真实图像在线生成一对相关视图：

- `view_a`
- `view_b`
- 从 `view_a` 到 `view_b` 的 dense warp field
- 有效对应区域 mask

第一阶段使用平移和光度扰动构造监督信号。后续可以继续扩展：

- 旋转和尺度变化
- 仿射倾斜
- 透视变化
- 径向/切向畸变
- 局部非刚性形变
- 强光照、阴影和低对比度扰动
- 模糊、噪声、压缩退化和遮挡

## 训练模块

训练会联合运行以下模块：

- `Backbone`：共享多尺度特征提取。
- `SparseHead`：输出关键点 heatmap、描述子、尺度、方向和仿射形状。
- `DenseHead`：输出半稠密置信度和局部偏移。

当前基础损失包括：

- `repeatability_loss`
- `descriptor_cross_entropy_loss`
- `masked_l1_loss`
- `confidence_bce_loss`

为了让真实大幅面 TIFF 能在 CPU 本地 smoke 中稳定运行，第一阶段训练会限制输入图像尺寸、限制每轮参与训练的图像数量，并对 descriptor loss 的空间位置做采样。

## 构建

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

如果 CMake 找不到 LibTorch 或 OpenCV，可以显式指定路径：

```bash
cmake -S . -B build -DBUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH="/path/to/torch/share/cmake" \
  -DOpenCV_DIR="/path/to/opencv/lib/cmake/opencv4"
```

## 训练

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1 \
  --device cpu
```

参数说明：

- `--image-dir`：训练图像目录，目录中必须至少包含一张支持格式图像。
- `--checkpoint`：输出 checkpoint 路径。
- `--epochs`：训练轮数，必须为正数。
- `--batch-size`：batch 大小，必须为正数。
- `--device`：当前仅支持 `cpu`。

训练成功后会输出类似：

```text
training complete: epochs=1 final_loss=...
```

并生成 `model.pt`。

## 提取特征

```bash
./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5
```

输出 `features.pt` 是 LibTorch archive，包含：

- `keypoints`：稀疏关键点坐标。
- `scores`：关键点分数。
- `descriptors`：稀疏描述子。
- `scale`：尺度估计。
- `orientation`：方向向量。
- `affine`：局部仿射形状。
- `dense_points`：半稠密点。
- `dense_confidence`：半稠密置信度。

## 双图匹配

```bash
./build/pfm_cli match \
  --image-a images/a.tif \
  --image-b images/b.tif \
  --checkpoint model.pt \
  --output matches.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5
```

输出 `matches.pt` 包含：

- `sparse_matches`：稀疏匹配索引对。
- `sparse_scores`：稀疏匹配分数。
- `points_a`：图像 A 的半稠密点。
- `points_b`：图像 B 的半稠密对应点。
- `confidence`：半稠密匹配置信度。

## 批量评估

先准备 `pairs.txt`：

```text
images/a.tif images/b.tif
images/c.tif images/d.tif
```

如果路径包含空格，必须使用英文双引号：

```text
"/home/user/data/Feature Extraction/a.tif" "/home/user/data/Feature Extraction/b.tif"
```

执行评估：

```bash
./build/pfm_cli eval \
  --pairs pairs.txt \
  --checkpoint model.pt \
  --output report.pt \
  --max-keypoints 1024
```

输出 `report.pt` 包含：

- `average_matches`：每对图像的平均稀疏匹配数量。
- `average_sparse_score`：平均稀疏匹配分数。
- `average_dense_confidence`：平均半稠密置信度。
- `semi_dense_coverage`：半稠密覆盖率。

## 导出模型

```bash
./build/pfm_cli export \
  --checkpoint model.pt \
  --output exported.pt
```

导出前会校验 checkpoint 是否包含推理需要的配置和权重。只有 config 而没有模型权重的 checkpoint 会被拒绝。

## 真实 TIFF smoke 示例

下面是一套最小真实图像闭环示例：

```bash
mkdir -p /tmp/pfm_smoke/images
cp images/a.tif /tmp/pfm_smoke/images/a.tif
printf '"%s" "%s"\n' "$(pwd)/images/a.tif" "$(pwd)/images/b.tif" > /tmp/pfm_smoke/pairs.txt

./build/pfm_cli train \
  --image-dir /tmp/pfm_smoke/images \
  --checkpoint /tmp/pfm_smoke/model.pt \
  --epochs 1 \
  --batch-size 1

./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint /tmp/pfm_smoke/model.pt \
  --output /tmp/pfm_smoke/features.pt \
  --max-keypoints 128 \
  --semi-dense-threshold 0.5

./build/pfm_cli match \
  --image-a images/a.tif \
  --image-b images/b.tif \
  --checkpoint /tmp/pfm_smoke/model.pt \
  --output /tmp/pfm_smoke/matches.pt \
  --max-keypoints 128 \
  --semi-dense-threshold 0.5

./build/pfm_cli eval \
  --pairs /tmp/pfm_smoke/pairs.txt \
  --checkpoint /tmp/pfm_smoke/model.pt \
  --output /tmp/pfm_smoke/report.pt \
  --max-keypoints 128

./build/pfm_cli export \
  --checkpoint /tmp/pfm_smoke/model.pt \
  --output /tmp/pfm_smoke/exported.pt
```

检查输出：

```bash
ls -lh /tmp/pfm_smoke/*.pt
```

## 验证命令

修改训练、推理、图像 IO、特征编解码或匹配逻辑后，至少运行：

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

## 当前限制与下一步

当前限制：

- 仅支持 CPU 训练和推理。
- 当前训练是第一阶段 MVP，偏重链路正确性和可测试性。
- 大图会被缩小，训练样本数也会被限制，因此不能代表完整训练效果。
- 几何增强仍较简单，对强旋转、强透视、严重畸变和大尺度变化的鲁棒性还需要继续提升。

建议下一步：

1. 增强自监督图像对生成，加入旋转、尺度、仿射、透视和畸变。
2. 增加多尺度训练和更真实的光照/阴影扰动。
3. 完善 matcher 训练目标，提高稀疏匹配精度。
4. 加入几何一致性评估和真实标注/伪标注 benchmark。
5. 在 CPU smoke 之外增加可选 GPU 训练路径。
