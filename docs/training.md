# 训练与使用说明

本文档说明 PlanetaryFeatureMatch 当前第一阶段 C++/LibTorch 训练、推理和评估流程。当前实现已经不是参数校验桩，而是可以读取真实图像、执行最小训练循环、写出 checkpoint，并继续完成特征提取、匹配、评估和导出。

## 目标

PlanetaryFeatureMatch 面向火星、月球和小行星影像，目标是训练一个同时支持稀疏关键点匹配和半稠密对应的局部特征模型。第一阶段的目标是打通真实图像闭环：

1. 从真实影像目录读取数据。
2. 生成自监督图像对。
3. 训练前可选择先生成合成训练对缓存。
4. 训练最小 LibTorch 模型。
5. 保存可加载 checkpoint。
6. 使用 checkpoint 提取特征。
7. 导出匹配结果和评估报告。

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

当前训练不依赖人工标注匹配点，而是从单张真实图像在线或离线缓存生成一对相关视图：

- `view_a`
- `view_b`
- 从 `view_a` 到 `view_b` 的 dense warp field
- 有效对应区域 mask

当前合成增强支持 `mixed`、`mild`、`medium`、`hard`、`extreme` 五种 profile。默认 `mixed` 会在每张图的多组 pair 中混合轻度、中等、困难和极端样本；极端样本包含更大的平移、旋转、尺度变化、对比度/亮度变化、gamma、梯度阴影和噪声。后续还可以继续扩展：

- 透视变化
- 径向/切向畸变
- 局部非刚性形变
- 模糊、压缩退化和遮挡

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

为了让真实大幅面 TIFF 能在 CPU 本地 smoke 中稳定运行，第一阶段训练会限制输入图像尺寸，并对 descriptor loss 的空间位置做采样；默认每个 epoch 使用目录中的全部训练图像，样本总数为 `图像数 × pairs_per_image`。

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
  --device cpu \
  --resize 512 \
  --pairs-per-image 1 \
  --augmentation-profile mixed \
  --extreme-pair-ratio 0.2 \
  --min-keypoint-intensity 0.0
```

参数说明：

- `--image-dir`：训练图像目录，目录中必须至少包含一张支持格式图像。
- `--checkpoint`：输出 checkpoint 路径。
- `--epochs`：训练轮数，必须为正数。
- `--batch-size`：batch 大小，必须为正数。
- `--device`：计算设备，默认 `cpu`；可写 `cuda` 或 `cuda:0`，其中 `cuda` 等价于 `cuda:0`。
- `--resize`：训练前将图像最大边缩放到该值以内；默认 512，传 0 才保持原图尺寸，必须为非负数。
- `--pairs-per-image`：每张真实图像生成多少组自监督合成匹配对，默认 1；增大后每轮训练样本数变为 `图像数 × pairs_per_image`。
- `--augmentation-profile`：合成增强强度，支持 `mixed`、`mild`、`medium`、`hard`、`extreme`；默认 `mixed`。
- `--extreme-pair-ratio`：`mixed` 中极端样本比例控制入口，默认 0.2，取值范围 `[0, 1]`。
- `--synthetic-pair-cache-dir`：合成训练对缓存目录；未指定时保持在线生成。
- `--synthetic-pair-cache-rebuild`：强制重建缓存，忽略已有文件。
- `--min-keypoint-intensity`：关键点监督和输出的最低归一化灰度阈值，默认 0.0，取值范围 `[0, 1]`。

训练时 `--min-keypoint-intensity` 会从归一化图像生成灰度掩码，并与合成 pair 的几何 `valid_mask` 相交；只有源视图和 warp 后目标视图都达到阈值的位置才参与 repeatability、descriptor、offset 和 confidence 监督。显式请求 CUDA 时不会静默回退到 CPU；CUDA 不可用、索引越界或格式错误会直接失败。CUDA 训练结束保存 checkpoint 前会把权重移回 CPU，因此同一个 checkpoint 可以被 CPU 或 GPU 推理加载。推理侧 `extract` 和 `match` 也支持 `--device cuda`，模型 forward 会在 GPU 上执行；特征解码、匹配后处理、PNG 可视化和 `.pt` 写出仍在 CPU。训练或推理功耗没有接近显卡 TDP 时，通常是 batch、输入分辨率、模型规模或 CPU 数据准备限制导致 GPU 等待，并不等同于没有使用 CUDA。

指定 `--synthetic-pair-cache-dir` 后，训练开始前会先生成 `manifest.pt`、`pair_000000.pt`、`source_000000_view_a.png`、`pair_000000_view_b.png` 等文件。同一源图的 A 视图只保存一个 `source_XXXXXX_view_a.png`，每个增强 pair 保存自己的 `pair_XXXXXX_view_b.png`。后续运行如果缓存完整且训练缩放参数、每图 pair 数、增强 profile、极端比例和合成变换参数一致，就直接读取 `.pt` 监督文件训练，不再重复生成。PNG 用于人工检查变换效果，训练实际使用 `.pt` 中的 `view_a`、`view_b`、`warp_a_to_b` 和 `valid_mask`。

默认 `--resize 512` 会把训练图像最大边限制到 512，每轮使用目录中的全部训练图像。`--pairs-per-image` 可以让每张图生成多组不同平移、旋转、尺度和光照扰动的匹配对。`--augmentation-profile extreme` 会显著增大变换幅度，适合检查缓存图像是否出现极端现象；正式训练默认推荐 `mixed`。想提高 GPU 利用率或增强数据量时，可以调大 `--resize`、`--batch-size`、`--pairs-per-image` 或增加训练图像数量，例如：

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
  --synthetic-pair-cache-dir build/pair_cache
```

调大这些值会增加 GPU 计算量和显存占用，也会增加 CPU 图像读取与预处理压力。训练过程中会按 batch 输出进度，`batch-size` 只决定每个 batch 包含多少合成 pair，不限制每轮使用的样本总数。

训练成功后会输出类似：

```text
train progress: epoch=1/1 batch=1/4 images=16/64 loss=...
train epoch summary: epoch=1/1 epoch_time=...s
training complete: epochs=1 final_loss=... total_time=...s avg_batch_time=...s
```

训练命令每个 epoch 会输出 `epoch_time=<seconds>s`，训练结束输出 `total_time=<seconds>s` 和 `avg_batch_time=<seconds>s`，用于判断整体耗时和 batch 级吞吐。

并生成 `model.pt`。

## 提取特征

```bash
./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5 \
  --min-keypoint-intensity 0.0
```

`--min-keypoint-intensity` 会过滤原图中低于阈值的区域，稀疏关键点和半稠密点都不会从这些位置输出。输出 `features.pt` 是 LibTorch archive，包含：

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
  --semi-dense-threshold 0.5 \
  --min-keypoint-intensity 0.0
```

`--min-keypoint-intensity` 会分别过滤两张图中的低灰度特征点，再执行匹配。输出 `matches.pt` 包含：

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
  --batch-size 1 \
  --resize 64 \
  --synthetic-pair-cache-dir /tmp/pfm_smoke/pair_cache

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

## CUDA 运行范围

当前 CUDA 接入范围：

- 训练：模型 forward、backward、loss 和 optimizer step 在指定设备上运行。
- 推理：`extract`、`match`、`eval` 的模型 forward 在指定设备上运行。
- 仍在 CPU 的部分：OpenCV 图像读取、特征解码、匹配后处理、评估汇总和 `.pt` 结果写出。

可用设备写法：

```bash
--device cpu
--device cuda
--device cuda:0
```

`cuda` 等价于 `cuda:0`。如果当前 LibTorch 没有 CUDA、设备索引不存在或字符串如 `cuda:abc` 格式错误，命令会失败，不会伪装成 CPU 运行。

## `pfm_tests` 测试程序

`build/pfm_tests` 是项目自己的 C++ 单元测试运行器，不是训练程序。CMake 会把 `tests/test_main.cpp`、`tests/test_harness.h` 和所有模块的 `*_test.cpp` 编译进这个可执行文件。

直接运行：

```bash
./build/pfm_tests
```

也可以通过 CTest 运行：

```bash
ctest --test-dir build --output-on-failure
```

每一行 `PASS <test_name>` 表示一个测试通过；测试多时看到大量 `PASS` 是正常的。最后输出 `N test(s) passed` 并且退出码为 0，表示全部通过。如果某个测试失败，会输出 `FAIL <test_name>: <reason>`，程序返回非 0。

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

- CUDA 目前只覆盖训练和推理模型 forward 相关计算，后处理仍在 CPU。
- 当前训练是第一阶段 MVP，偏重链路正确性和可测试性。
- 大图缩放、每轮训练样本数和合成训练对缓存已有可配置参数；默认值仍偏小，因此不能代表完整训练效果。
- 合成增强已支持强旋转、尺度和光照 profile，但强透视、严重畸变和真实遮挡的鲁棒性还需要继续提升。

建议下一步：

1. 继续增强自监督图像对生成，加入透视、相机畸变和遮挡。
2. 增加多尺度训练和更真实的局部光照退化。
3. 完善 matcher 训练目标，提高稀疏匹配精度。
4. 加入几何一致性评估和真实标注/伪标注 benchmark。
5. 继续把匹配后处理、评估和更大规模数据管线逐步 GPU 化。
