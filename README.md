# PlanetaryFeatureMatch

PlanetaryFeatureMatch 是一个基于 C++17、LibTorch 和 OpenCV 的行星影像局部特征提取与匹配项目，面向火星、月球和小行星等场景。行星影像常见弱纹理、光照变化大、视角差异、成像畸变、相机倾斜和局部形变，传统局部特征与仅依赖 RANSAC 的流程在这些场景下容易失效。

当前版本已经打通第一阶段真实闭环：读取真实图像、训练最小模型、保存 checkpoint、提取 `.pt` 特征、执行双图匹配、按 pairs 文件评估，并导出可用于推理的 checkpoint。该阶段重点是端到端可运行和可测试，尚不代表最终匹配精度。

## 已实现能力

- OpenCV 图像读取：支持常见 8/16 位灰度图和 RGB/BGR 图像。
- LibTorch 训练：使用真实图像生成自监督合成图像对并执行最小训练循环。
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
  data/       图像 IO、ImageDataset、归一化与自监督合成图像对
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

训练完成后提取单张图像特征：

```bash
./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5
```

匹配两张图像：

```bash
./build/pfm_cli match \
  --image-a images/a.tif \
  --image-b images/b.tif \
  --checkpoint model.pt \
  --output matches.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5
```

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
./build/pfm_cli train --image-dir images --checkpoint model.pt [--epochs 1] [--batch-size 1] [--device cpu]
```

- `--image-dir`：训练图像目录。
- `--checkpoint`：输出 checkpoint 路径。
- `--epochs`：训练轮数，默认 1。
- `--batch-size`：batch 大小，默认 1。
- `--device`：计算设备，默认 `cpu`；可写 `cuda` 或 `cuda:0`，其中 `cuda` 等价于 `cuda:0`。

第一阶段训练会对大图做 CPU 友好的尺寸限幅，并限制每轮样本数，避免真实大幅面 TIFF 在本地 smoke 中占用过多内存和时间。显式请求 CUDA 时不会静默回退到 CPU；CUDA 不可用、索引越界或格式错误会直接失败。

### `extract`

```bash
./build/pfm_cli extract --image a.tif --checkpoint model.pt --output features.pt [--device cpu] [--max-keypoints 1024] [--semi-dense-threshold 0.5]
```

- 输出为 LibTorch `.pt` archive。
- 包含稀疏关键点、分数、描述子、尺度、方向、仿射形状、半稠密点和半稠密置信度。

### `match`

```bash
./build/pfm_cli match --image-a a.tif --image-b b.tif --checkpoint model.pt --output matches.pt [--device cpu] [--max-keypoints 1024] [--semi-dense-threshold 0.5]
```

- 输出为 LibTorch `.pt` archive。
- 包含稀疏匹配索引、稀疏匹配分数、半稠密点对和置信度。

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

- 第一阶段训练使用轻量自监督扰动，主要验证链路，不保证最终匹配效果。
- 真实大图训练会被缩小到较小边长以保证本地 smoke 可运行。
- `train` 中的 `--pairs`、`--config`、`--output` 已保留在 CLI 中，但当前主要训练输出使用 `--checkpoint`。
- 后续需要继续增强仿射、透视、畸变、阴影、遮挡和多尺度几何监督。

## 开发验证建议

修改代码后至少运行：

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

修改训练、推理或 IO 后，建议再用真实 TIFF 执行一次 `train`、`extract`、`match`、`eval` 和 `export` smoke。
