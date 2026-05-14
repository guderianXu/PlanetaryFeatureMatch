# 训练说明

本文档说明 PlanetaryFeatureMatch 计划采用的 C++/LibTorch 训练流程。当前 `train` 命令仍是参数校验 stub：它会检查必要 CLI 参数并正常退出，但还不会执行优化、保存 checkpoint 或写出训练日志。

## 训练目标

训练一个同时支持稀疏关键点匹配和半稠密对应关系的行星影像局部特征与匹配模型。模型需要尽量适应：

- 弱纹理
- 大光照变化
- 相机倾角
- 多视角几何变化
- 局部仿射形变
- 成像畸变
- 阴影、无效区域和遮挡

## 输入数据

第一阶段训练建议从单张行星影像出发，在线生成或缓存生成合成影像对。

计划支持的源影像类型：

- 8 位灰度图
- 16 位灰度图
- 8 位 RGB 图
- 16 位 RGB 图

预处理流程应将每张图像转换为 `C x H x W` 形状的 LibTorch tensor，归一化到 `[0, 1]`，并可选使用局部对比度归一化增强弱纹理和阴影区域的稳定性。

## 合成影像对生成

每张源影像应生成两张相关视图及其监督信息：

- `view_a`
- `view_b`
- 从 `view_a` 到 `view_b` 的 dense warp field
- 有效对应区域 mask

当前基础实现位于 `modules/data/synthetic_pair.cpp`，已支持确定性的整数平移合成影像对。后续训练需要继续扩展该模块，加入：

- 旋转和尺度变化
- 仿射倾斜
- 透视变换
- 径向和切向畸变
- 非线性局部形变
- 亮度、对比度和 gamma 变化
- 方向性光照变化
- 阴影 mask
- 模糊、噪声、压缩和低分辨率退化
- 随机无效区域或遮挡区域

## 模型组件

训练时应联合优化以下模块：

- `Backbone`：多尺度特征提取
- `SparseHead`：关键点 heatmap、描述子、尺度、方向、仿射形状
- `DenseHead`：半稠密置信度和局部偏移
- `Matcher`：描述子相似度与匹配基础模块

## 损失函数

当前已经实现的基础损失：

- `repeatability_loss`
- `descriptor_cross_entropy_loss`
- `masked_l1_loss`
- `confidence_bce_loss`

后续完整训练目标可以组合为：

```text
total_loss =
  sparse_repeatability_weight * repeatability_loss +
  descriptor_weight * descriptor_loss +
  semi_dense_offset_weight * offset_loss +
  confidence_weight * confidence_loss +
  geometry_consistency_weight * consistency_loss
```

其中 geometry consistency loss 还没有实现。

## 当前 `train` 命令

当前可接受的命令格式：

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1 \
  --device cpu
```

计划中的真实训练行为：

1. 从 `--image-dir` 读取图像。
2. 将图像归一化为 LibTorch tensor。
3. 生成合成影像对和 valid mask。
4. 前向运行 backbone、稀疏分支、半稠密分支和 matcher。
5. 计算稀疏、半稠密、置信度和匹配损失。
6. 使用 AdamW 优化器反向传播。
7. 定期评估 repeatability、matching precision 和 semi-dense coverage。
8. 保存 LibTorch checkpoint 到 `--checkpoint` 或 `--output`。

## 最小实现里程碑

1. 添加图像数据集模块和测试。
2. 将合成影像对生成从整数平移扩展到仿射、透视和光照扰动。
3. 添加训练配置结构与配置解析。
4. 实现 checkpoint 保存/加载测试。
5. 添加 one-batch overfit 测试，证明梯度能更新模型参数。
6. 实现完整训练循环。
7. 基于微型合成 checkpoint 添加 extract 和 match 集成测试。

## 验证命令

训练功能完成前，每次修改后至少运行：

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

当前 CLI stub 还应验证：

```bash
./build/pfm_cli train --image-dir images --checkpoint model.pt --epochs 1 --batch-size 1
```
