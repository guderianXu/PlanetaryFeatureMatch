# PlanetaryFeatureMatch

PlanetaryFeatureMatch 是一个基于 C++17、LibTorch 和 OpenCV 的行星影像局部特征提取与匹配项目。目标场景包括火星、月球和小行星影像；这些影像常见弱纹理、视角变化、成像畸变、相机倾斜和光照变化，传统局部特征与仅依赖 RANSAC 的流程容易失效。

当前实现提供第一阶段真实可执行流程：从真实图像目录训练最小模型，保存/加载 checkpoint，提取特征，执行双图匹配，按图像对列表评估，并导出可用于推理的 checkpoint。该阶段用于打通端到端训练与推理链路，尚不代表生产级精度或完整鲁棒性。

## 设计

模型采用稀疏与半稠密结合的匹配结构：

- 共享多尺度 backbone
- 稀疏分支输出关键点、描述子、尺度、方向和仿射形状
- 半稠密分支输出置信点对应关系
- matcher 模块提供描述子相似度与匹配评分基础

## 仓库结构

```text
modules/
  cli/        CLI11 命令解析与测试
  core/       张量校验和网格工具
  data/       图像 IO、ImageDataset、归一化与自监督合成图像对
  eval/       匹配指标、半稠密指标和评估流水线辅助函数
  geometry/   仿射 warp 辅助函数
  infer/      特征/匹配编解码、特征解码、双图匹配与评估流水线辅助函数
  losses/     repeatability、descriptor、offset 和 confidence 损失
  models/     backbone、sparse head、dense head、matcher
  train/      训练配置、trainer 和 checkpoint 保存/加载
src/
  main.cpp    CLI 入口
```

项目按模块组织代码，每个模块配套对应的 `*_test.cpp` 测试文件。

## 依赖

- CMake 3.18+
- C++17 编译器
- LibTorch / PyTorch C++ CMake 包
- OpenCV CMake 包
- `CLI11.hpp` 位于仓库根目录

## 构建与测试

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

## CLI

```bash
./build/pfm_cli --help
```

第一阶段命令已经执行真实训练/推理行为：

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1
```

`train` 会读取真实图像目录，在线生成平移和光度扰动的自监督合成图像对，运行 LibTorch 最小模型训练，并保存 checkpoint。

```bash
./build/pfm_cli extract \
  --image a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5
```

`extract` 会读取图像和 checkpoint，输出 `.pt` 特征文件。

```bash
./build/pfm_cli match \
  --image-a a.tif \
  --image-b b.tif \
  --checkpoint model.pt \
  --output matches.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5
```

`match` 会读取两张图像和 checkpoint，输出 `.pt` 匹配结果。

```bash
./build/pfm_cli eval \
  --pairs pairs.txt \
  --checkpoint model.pt \
  --output report.pt \
  --max-keypoints 1024
```

`eval` 会读取图像对列表，逐对提取与匹配，并输出 `.pt` 评估报告。

```bash
./build/pfm_cli export \
  --checkpoint model.pt \
  --output exported.pt
```

`export` 会校验输入 checkpoint，并复制/重存为推理 checkpoint。

## 当前状态

已实现并测试：

- OpenCV 图像读取和 8/16 位灰度、RGB 归一化
- ImageDataset 图像枚举与加载
- 局部对比度归一化
- 自监督合成图像对生成
- 仿射 warp field 与 valid mask 工具
- backbone、sparse head、dense head、matcher 张量契约
- repeatability、descriptor、masked L1、confidence 损失
- trainer、checkpoint 保存/加载
- 特征与匹配结果 `.pt` 编解码
- 提取、匹配、评估和导出 CLI 流程
