# PlanetaryFeatureMatch

PlanetaryFeatureMatch 是一个基于 C++17、LibTorch 和 OpenCV 的行星影像局部特征提取与匹配项目，面向火星、月球和小行星等场景。行星影像常见弱纹理、光照变化大、视角差异、成像畸变、相机倾斜和局部形变，传统局部特征与仅依赖 RANSAC 的流程在这些场景下容易失效。

当前版本已经打通第一阶段真实闭环：读取真实图像、训练最小模型、保存 checkpoint、提取 `.pt` 特征、执行双图匹配、按 pairs 文件评估，并导出可用于推理的 checkpoint。该阶段重点是端到端可运行和可测试，尚不代表最终匹配精度。

## 已实现能力

- OpenCV 图像读取：支持常见 8/16 位灰度图和 RGB/BGR 图像。
- LibTorch 训练：使用真实图像生成自监督合成图像对并执行最小训练循环。
- 合成对缓存：可在训练前把变换后的图像和 `.pt` 监督文件生成到指定目录，后续训练直接复用。
- CUDA 设备选择：训练和推理 forward 支持 `cpu`、`cuda`、`cuda:N`。
- Checkpoint：使用 LibTorch `.pt` archive 保存和加载模型，CUDA 训练后仍保存为 CPU 权重。
- 特征提取：输出稀疏关键点、描述子、尺度、方向、仿射形状和半稠密点。
- 双图匹配：稀疏描述子 mutual nearest-neighbor 匹配与半稠密点对应导出。
- 评估：对 pairs 文件中的图像对聚合平均匹配数、稀疏分数、半稠密置信度和覆盖率。
- 模型导出：校验 checkpoint 完整性后导出推理 checkpoint。
- 模块化测试：每个主要模块配套 `*_test.cpp`。

## 模型结构

当前模型采用稀疏与半稠密结合的第一阶段结构：

- `Backbone`：共享多尺度特征提取。
- `SparseHead`：输出关键点 heatmap、描述子、尺度、方向和仿射形状。
- `DenseHead`：输出半稠密置信度和局部偏移。
- `Matcher`：提供描述子相似度与匹配评分基础。

这个方向不是只做 SuperPoint 风格的稀疏特征，也不是只做 LoFTR 风格的稠密匹配，而是同时服务于后续稀疏匹配和半稠密匹配。训练使用自监督合成图像对，基础损失包括 repeatability、descriptor cross entropy、masked L1 offset 和 confidence BCE。

## 仓库结构

```text
modules/
  cli/        CLI11 命令解析与测试
  core/       张量校验和网格工具
  data/       图像 IO、ImageDataset、归一化、自监督合成图像对和缓存
  eval/       匹配指标和半稠密覆盖率指标
  geometry/   仿射 warp 辅助函数
  infer/      特征/匹配编解码、特征解码、匹配与评估流水线
  losses/     repeatability、descriptor、offset 和 confidence 损失
  models/     backbone、sparse head、dense head、matcher
  train/      训练配置、trainer 和 checkpoint 保存/加载
src/
  main.cpp    CLI 入口
tests/
  test_main.cpp
  test_harness.h
```

合成训练对缓存由 `modules/data/synthetic_pair_cache.*` 管理：PNG 文件用于人工检查变换结果，`.pt` 文件保存训练实际需要的 `view_a`、`view_b`、`warp_a_to_b` 和 `valid_mask`。

项目按模块组织代码，不使用 `include/` 与 `src/` 分离的库式布局。

## 依赖

- CMake 3.18+
- C++17 编译器
- LibTorch / PyTorch C++ CMake 包
- OpenCV CMake 包
- 仓库根目录下的 `CLI11.hpp`

如果 CMake 无法自动找到 LibTorch 或 OpenCV，可以显式指定：

```bash
cmake -S . -B build -DBUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH="/path/to/torch/share/cmake" \
  -DOpenCV_DIR="/path/to/opencv/lib/cmake/opencv4"
```

## 构建与测试

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

查看命令行帮助：

```bash
./build/pfm_cli --help
```

## 快速开始

假设图像放在 `images/`，至少包含一张 OpenCV 可读取的图像：

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1
```

如果希望先离线生成变换后的训练对，再从缓存训练，可以加缓存目录：

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1 \
  --pairs-per-image 4 \
  --synthetic-pair-cache-dir build/pair_cache
```

第一次运行会生成 `manifest.pt`、`pair_000000.pt`、`source_000000_view_a.png` 和 `pair_000000_view_b.png` 等文件；`--pairs-per-image` 可让每张原图生成多组不同平移、旋转、尺度和光照扰动的匹配对。同一源图的 A 视图只保存一个 `source_XXXXXX_view_a.png`，每个增强 pair 保存自己的 `pair_XXXXXX_view_b.png`。默认 `--augmentation-profile mixed` 会混合 mild/medium/hard/extreme 强度；如果想明显检查极端变换，可临时使用 `--augmentation-profile extreme`。后续配置匹配时直接复用。需要强制重建时添加 `--synthetic-pair-cache-rebuild`。

训练完成后提取单张图像特征：

```bash
./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5 \
  --visualization-dir vis \
  --keypoint-grid-rows 8 \
  --keypoint-grid-cols 8 \
  --keypoints-per-cell 0 \
  --nms-radius 4
```

推理阶段的稀疏特征点默认会先应用低灰度过滤，再做局部 NMS，随后按网格分块选点，最后用全局高分候选补足 `--max-keypoints`。`--keypoints-per-cell 0` 表示根据 `max_keypoints` 和网格数量自动推导，采用向上取整且每个 cell 至少 1 个候选。

匹配两张图像：

```bash
./build/pfm_cli match \
  --image-a images/a.tif \
  --image-b images/b.tif \
  --checkpoint model.pt \
  --output matches.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5 \
  --visualization-dir vis
```

推理命令可以加 `--device cuda` 使用 GPU 跑模型 forward。当前 CUDA 覆盖模型前向；图像读取、特征解码、匹配后处理、PNG 可视化和 `.pt` 写出仍在 CPU。若要肉眼观察效果，可以给 `extract` 或 `match` 添加 `--visualization-dir vis`：`extract` 会生成 `<image_stem>_features.png`，`match` 会生成 `<image_a_stem>__<image_b_stem>_matches.png`。

准备评估 pairs 文件：

```text
images/a.tif images/b.tif
"/path/with spaces/a.tif" "/path/with spaces/b.tif"
```

运行评估：

```bash
./build/pfm_cli eval \
  --pairs pairs.txt \
  --checkpoint model.pt \
  --output report.pt \
  --max-keypoints 1024
```

导出 checkpoint：

```bash
./build/pfm_cli export \
  --checkpoint model.pt \
  --output exported.pt
```

## CLI 命令说明

### `train`

```bash
./build/pfm_cli train --image-dir images --checkpoint model.pt [--epochs 1] [--batch-size 1] [--device cpu] [--resize 512] [--pairs-per-image 1] [--augmentation-profile mixed] [--extreme-pair-ratio 0.2] [--synthetic-pair-cache-dir build/pair_cache] [--synthetic-pair-cache-rebuild] [--min-keypoint-intensity 0.0]
```

- `--image-dir`：训练图像目录。
- `--checkpoint`：输出 checkpoint 路径。
- `--epochs`：训练轮数，默认 1。
- `--batch-size`：batch 大小，默认 1。
- `--device`：计算设备，默认 `cpu`；可写 `cuda` 或 `cuda:0`，其中 `cuda` 等价于 `cuda:0`。
- `--resize`：训练前将图像最大边缩放到该值以内；默认 512，传 0 才保持原图尺寸，必须为非负数。
- `--pairs-per-image`：每张真实图像生成多少组自监督合成匹配对，默认 1；增大后每轮训练样本数变为 `图像数 × pairs_per_image`。
- `--augmentation-profile`：合成增强强度，支持 `mixed`、`mild`、`medium`、`hard`、`extreme`；默认 `mixed`。
- `--extreme-pair-ratio`：`mixed` 中极端样本比例控制入口，默认 0.2，取值范围 `[0, 1]`。
- `--synthetic-pair-cache-dir`：合成训练对缓存目录；未指定时仍在训练循环中在线生成。
- `--synthetic-pair-cache-rebuild`：忽略已有缓存并强制重新生成。
- `--min-keypoint-intensity`：关键点监督和输出的最低归一化灰度阈值，默认 0.0，取值范围 `[0, 1]`。

默认每轮使用目录中的全部训练图像；设置 `--pairs-per-image N` 后，每张图会派生 N 个不同合成 pair。训练时 `--min-keypoint-intensity` 会同时要求源视图和目标视图对应位置达到阈值，低灰度区域不参与 repeatability、descriptor、offset 和 confidence 监督。`--batch-size` 只控制每次反向传播的样本分组大小。CUDA 训练时可以调大 `--resize`、`--batch-size`、`--pairs-per-image` 或增加训练图像数量，以增加 GPU 计算量和显存占用。显式请求 CUDA 时不会静默回退到 CPU；CUDA 不可用、索引越界或格式错误会直接失败。

指定缓存目录后，训练开始前会先生成合成对缓存。缓存完整且配置匹配时会复用；缓存缺失、数量不匹配、每图 pair 数、缩放尺寸、profile、极端比例或合成参数变化时会自动重建。PNG 只用于查看变换效果，训练读取 `.pt` 中的监督张量。

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

### `extract`

```bash
./build/pfm_cli extract --image a.tif --checkpoint model.pt --output features.pt [--device cpu] [--max-keypoints 1024] [--semi-dense-threshold 0.5] [--visualization-dir vis] [--min-keypoint-intensity 0.0]
```

- 输出为 LibTorch `.pt` archive。
- 包含稀疏关键点、分数、描述子、尺度、方向、仿射形状、半稠密点和半稠密置信度。
- `--min-keypoint-intensity` 会按原图归一化灰度生成掩码，低于阈值的位置不会输出为稀疏关键点或半稠密点。

### `match`

```bash
./build/pfm_cli match --image-a a.tif --image-b b.tif --checkpoint model.pt --output matches.pt [--device cpu] [--max-keypoints 1024] [--semi-dense-threshold 0.5] [--visualization-dir vis] [--min-keypoint-intensity 0.0]
```

- 输出为 LibTorch `.pt` archive。
- 包含稀疏匹配索引、稀疏匹配分数、半稠密点对和置信度。
- `--min-keypoint-intensity` 会分别过滤两张图中的低灰度特征点，再进行匹配。

### `eval`

```bash
./build/pfm_cli eval --pairs pairs.txt --checkpoint model.pt --output report.pt [--device cpu] [--max-keypoints 1024] [--semi-dense-threshold 0.5]
```

- `pairs.txt` 每行写一对图像路径。
- 路径包含空格时必须使用英文双引号。
- 输出报告字段包括 `average_matches`、`average_sparse_score`、`average_dense_confidence` 和 `semi_dense_coverage`。

### `export`

```bash
./build/pfm_cli export --checkpoint model.pt --output exported.pt
```

`export` 会检查 checkpoint 是否包含推理所需的配置和模型权重；配置不完整或权重缺失时会失败。

## 输出文件

所有中间结果都使用 LibTorch `.pt` archive，便于 C++ 侧继续读取：

- `model.pt`：训练 checkpoint。
- `features.pt`：单图特征。
- `matches.pt`：双图匹配结果。
- `report.pt`：评估聚合指标。
- `exported.pt`：导出的推理 checkpoint。

## CUDA 说明

- `--device cpu` 是默认值。
- `--device cuda` 等价于 `--device cuda:0`。
- `--device cuda:N` 会使用第 `N` 张 CUDA 设备。
- CUDA 不可用、索引越界或设备字符串格式错误时命令会明确失败，不会静默退回 CPU。
- 当前 CUDA 范围是训练 forward/backward/loss 和推理模型 forward；OpenCV 图像读取、特征解码、匹配后处理、评估汇总和 `.pt` 输出仍在 CPU。
- 训练 checkpoint 保存为 CPU 权重，便于在 CPU/GPU 之间迁移。

## `pfm_tests` 是什么

`build/pfm_tests` 是 CMake 构建出的自定义 C++ 单元测试运行器，由 `tests/test_main.cpp` 和 `tests/test_harness.h` 组织。它会把各模块的 `*_test.cpp` 注册到同一个测试程序中。

运行：

```bash
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

输出里的 `PASS <test_name>` 表示一个测试用例通过；出现很多 `PASS` 是正常现象。最后一行类似 `127 test(s) passed` 且退出码为 0，表示全部通过。如果失败，会输出 `FAIL <test_name>: <reason>` 并返回非 0。

## 当前限制

- 当前合成增强已支持混合强度和极端旋转/尺度/光照扰动，但仍属于自监督近似，不保证最终匹配效果。
- 真实大图训练默认会被缩小到较小边长以保证本地 smoke 可运行；正式 CUDA 训练应显式调大训练规模参数。
- `train` 中的 `--pairs`、`--config`、`--output` 已保留在 CLI 中，但当前主要训练输出使用 `--checkpoint`。
- 后续还需要继续加入透视、相机畸变、遮挡和多尺度几何监督。

## 开发验证建议

修改代码后至少运行：

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

修改训练、推理或 IO 后，建议再用真实 TIFF 执行一次 `train`、`extract`、`match`、`eval` 和 `export` smoke。
