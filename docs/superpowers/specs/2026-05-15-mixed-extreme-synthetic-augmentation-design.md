# 混合强度合成增强设计

## 背景

当前 `--pairs-per-image` 已经能让每张真实图生成多个 synthetic pair，但额外 pair 的变换强度太弱，肉眼看起来和默认 pair 接近。默认训练增强只有固定平移、12° 旋转、0.92 缩放和轻微亮度/噪声，不能充分覆盖行星影像里的极端视角、拍摄倾角、弱纹理和强光照变化。

目标是在不破坏训练监督正确性的前提下，让离线缓存和在线训练都能生成更有差异的自监督匹配对。增强必须确定性可复现，缓存参数变化必须触发重建。

## 目标

- 每张真实图的多个 pair 应该产生明显不同的 `view_b` 和 `warp_a_to_b`。
- 默认使用混合强度增强，兼顾训练稳定性和极端场景覆盖。
- 提供少量 CLI 可调入口，不暴露大量底层范围参数。
- 缓存 manifest 记录增强配置，避免误复用旧缓存。
- 保证 `warp_a_to_b` 和 `valid_mask` 与图像变换一致。

## 非目标

- 本阶段不实现完整透视投影或真实相机模型。
- 本阶段不引入随机不可复现增强。
- 本阶段不把所有增强范围都变成 CLI 参数。
- 本阶段不改变模型结构或 loss 权重。

## 推荐方案

采用“方案 1 + 少量可调入口”：新增增强 profile，默认 `mixed`，并支持控制 mixed 中 extreme 样本比例。

新增训练参数：

- `--augmentation-profile mixed|mild|medium|hard|extreme`，默认 `mixed`。
- `--extreme-pair-ratio <float>`，默认 `0.2`，只在 `mixed` profile 下决定 extreme 样本占比。
- 保留 `--pairs-per-image`，每张图生成多个不同 variant。

示例：

```bash
./build/pfm_cli train \
  --image-dir build/img \
  --checkpoint train.pt \
  --epochs 100 \
  --batch-size 16 \
  --device cuda \
  --resize 512 \
  --pairs-per-image 6 \
  --augmentation-profile mixed \
  --extreme-pair-ratio 0.2 \
  --synthetic-pair-cache-dir build/pair_cache
```

## 增强 profile

### mild

用于稳定训练，变化轻微但不完全相同：

- 旋转：小角度。
- 缩放：接近 1。
- 平移：较小。
- 光照：轻微亮度/对比度变化。
- 噪声：轻微。

### medium

作为常规训练主力，比当前默认更明显：

- 旋转：中等角度。
- 缩放：明显放大或缩小。
- 平移：中等。
- 仿射：轻微 shear / 各向异性缩放。
- 光照：中等亮度、对比度和 gamma 变化。

### hard

模拟困难行星影像匹配：

- 旋转：大角度。
- 缩放：较强尺度变化。
- 平移：较大，但保留有效重叠。
- 仿射：明显 shear 和各向异性缩放，用于近似相机倾角。
- 光照：强亮度、对比、gamma 和局部渐变阴影。

### extreme

少量极端样本，覆盖模型最容易失败的情况：

- 旋转：很大角度。
- 缩放：强尺度变化。
- 仿射：强 shear / 各向异性缩放。
- 光照：强局部阴影、亮度偏移、对比变化和 gamma。
- valid mask 必须保留足够有效区域；如果某个极端参数导致 mask 过小，应在确定性范围内降低强度或重采样参数。

### mixed

默认 profile。每张图的 variant 根据 `source_index + variant_index` 确定性选择强度：

- 大多数样本来自 medium。
- 一部分样本来自 hard。
- `extreme_pair_ratio` 控制 extreme 样本比例。
- variant 0 可以保持一个较稳定的 baseline pair，后续 variant 必须产生可见差异。

## 数据流

1. CLI 解析 `augmentation_profile` 和 `extreme_pair_ratio`。
2. `run_train_command()` 写入 `TrainConfig`。
3. `TrainConfig` 构造 `SyntheticPairConfig` 或新的增强配置结构。
4. 在线训练路径根据 synthetic pair 全局 index 计算：
   - `source_index = index % dataset.size()`
   - `variant_index = index / dataset.size()`
5. 缓存路径使用同样的 source/variant 映射生成 `.pt` 和 PNG。
6. `make_synthetic_pair()` 使用 deterministic variant 参数生成图像、warp 和 valid mask。
7. cache manifest 写入 profile、ratio、pair count、resize 和基础增强参数。

## 实现边界

推荐把当前 `resolve_variant_config()` 扩展/替换为独立 helper，例如：

- `SyntheticPairAugmentationProfile`
- `SyntheticPairAugmentationConfig`
- `resolve_synthetic_pair_variant_config()`

该 helper 只负责从 profile、source index 和 variant index 得到确定性几何/光照参数。`make_synthetic_pair()` 仍负责实际 warp、photometric transform 和 mask 构造。

如果需要 affine shear / 各向异性缩放，应扩展 `AffineTransform` 构造或新增组合 helper，不能只改图像不改 warp。

## 测试验收

必须按 TDD 实现以下行为：

- 同一 source image 的多个 cached pair 中，`view_b` 和 `warp_a_to_b` 不相同。
- `hard` 或 `extreme` profile 的平均 warp 位移大于 `mild`。
- `extreme` pair 的 `valid_mask` 不是全空，并保留可训练重叠区域。
- `mixed` profile 在多 pair 下至少产生两种强度档位。
- cache manifest 记录 `augmentation_profile` 和 `extreme_pair_ratio`。
- 修改 profile 或 ratio 会触发缓存重建。
- CLI 能解析 `--augmentation-profile` 和 `--extreme-pair-ratio`。
- pipeline 能把 CLI 参数传到 trainer 和 cache。
- 文档说明新参数、推荐值和增强过强可能增加训练难度。

## 风险与取舍

增强变强后，训练 loss 初期可能升高，descriptor accuracy 可能短期下降。这不是失败，而是数据难度提高。后续评估应同时看 cached PNG 是否足够多样、valid mask 是否合理、训练曲线是否能下降，以及真实匹配输出是否改善。

先用 affine 近似极端视角是有意取舍：它能保持监督 warp 简洁可靠，也能覆盖旋转、尺度、倾角近似和光照变化。完整 perspective 可以作为下一阶段增强。