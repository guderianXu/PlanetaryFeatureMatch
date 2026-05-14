# PlanetaryFeatureMatch

PlanetaryFeatureMatch 是一个基于 C++17 和 LibTorch 的行星影像深度学习特征提取与匹配项目。项目主要面向火星、月球、小行星影像，目标是在弱纹理、多视角、成像畸变、相机倾角和大光照变化等条件下，提供比传统局部特征与单纯 RANSAC 流程更稳健的特征匹配基础。

当前代码是一个经过测试的算法基础框架，还不是完整训练系统。现在已经实现了模块级张量工具、归一化、合成影像对生成、仿射几何辅助、模型张量形状约束、损失函数、评估指标、CLI11 命令行解析，以及主要命令流程的参数校验 stub。

## 设计方向

目标模型采用“稀疏特征 + 半稠密匹配”的双分支结构：

- 共享多尺度 backbone
- 稀疏特征分支：预测关键点、描述子、尺度、方向和仿射形状
- 半稠密分支：预测高置信度点对应关系
- 学习式 matcher：用于描述子相似度与匹配打分

这个方向不是只做 SuperPoint 风格的稀疏特征，也不是只做 LoFTR 风格的稠密匹配，而是同时服务于后续稀疏匹配和半稠密匹配。

## 目录结构

```text
modules/
  cli/        CLI11 命令行解析与测试
  core/       张量校验与坐标网格工具
  data/       归一化与合成影像对生成
  eval/       匹配精度与半稠密覆盖率指标
  geometry/   仿射 warp 辅助函数
  infer/      命令参数校验 stub
  losses/     重复性、描述子、偏移和置信度损失
  models/     backbone、稀疏 head、稠密 head、matcher
src/
  main.cpp    CLI 入口
tests/
  test_main.cpp
  test_harness.h
```

本项目按模块组织代码，每个模块配套自己的 `*_test.cpp` 测试文件，不采用 `include/` 和 `src/` 分离的库式结构。

## 环境要求

- CMake 3.18+
- 支持 C++17 的编译器
- CMake 能找到 LibTorch / PyTorch C++ 包
- 仓库根目录存在 `CLI11.hpp`

## 编译与测试

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

## 命令行

查看帮助：

```bash
./build/pfm_cli --help
```

当前已经实现命令参数校验 stub：

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1

./build/pfm_cli extract \
  --image a.png \
  --checkpoint model.pt \
  --output a.pfm \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5

./build/pfm_cli match \
  --image-a a.png \
  --image-b b.png \
  --checkpoint model.pt \
  --output matches.json \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5

./build/pfm_cli eval \
  --pairs pairs.txt \
  --checkpoint model.pt \
  --output report.json

./build/pfm_cli export \
  --checkpoint model.pt \
  --output exported.pt
```

注意：当前这些命令只做参数校验并输出 `command accepted`，还没有真正执行训练、推理、特征序列化、匹配结果导出或模型导出。

## 当前已实现内容

已实现并测试：

- CHW 图像张量校验与 XY 坐标网格生成
- 8 位和 16 位图像归一化
- 局部对比度归一化
- 仿射 warp field 与 valid mask 辅助函数
- 确定性的合成影像对生成
- backbone、稀疏 head、稠密 head、matcher 的张量接口约束
- repeatability、descriptor、masked L1、confidence 损失函数
- matching precision 与 semi-dense coverage 评估指标
- CLI11 命令行解析

尚未实现：

- 图像文件读取
- 数据集迭代器
- 完整训练循环
- checkpoint 保存与加载
- 真实特征提取输出
- 真实双图匹配输出
- 模型导出后端
