# 灰度阈值特征点过滤设计

## 背景

当前 `extract` 的特征点可视化显示，部分稀疏点落在行星主体外的暗背景或低灰度边缘伪影上。现有稀疏点解码从模型 heatmap 全图取 top-k，没有结合原始图像亮度；训练侧已有几何 `valid_mask`，但没有把低灰度区域从监督中排除。

行星影像中主体外背景通常接近黑色，主体边缘也可能存在拍摄、投影或分割问题。低灰度区域如果参与特征提取和训练，会让模型把不可用区域当成高分候选，影响可视化、匹配和后续训练质量。

## 目标

- 新增固定归一化灰度阈值参数 `--min-keypoint-intensity`。
- 在 `train`、`extract`、`match`、`eval` 中使用同一阈值语义。
- 推理时稀疏 keypoint 和半稠密点都避开低灰度区域。
- 训练时低灰度区域不参与 repeatability、descriptor、offset 和 confidence 监督。
- 默认值为 `0.0`，保持旧行为。

## 非目标

- 本次不实现自动阈值估计。
- 本次不实现边缘膨胀、腐蚀或连通域过滤。
- 本次不改变模型结构、checkpoint 格式或 `.pt` 输出字段。
- 本次不把图像预分割成行星主体 mask，只基于输入图像灰度阈值生成 mask。

## CLI 设计

新增参数：

```bash
--min-keypoint-intensity <value>
```

适用命令：

- `train`
- `extract`
- `match`
- `eval`

语义：

- 输入图像已归一化到 `[0, 1]`。
- 像素灰度 `< value` 的位置视为无效。
- `value` 必须在 `[0, 1]` 内。
- 默认 `0.0` 表示不过滤，兼容现有行为。

示例：

```bash
./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --visualization-dir vis \
  --min-keypoint-intensity 0.08
```

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 10 \
  --batch-size 4 \
  --min-keypoint-intensity 0.08
```

## 数据流设计

### 亮度 mask 生成

新增一个小型公共 helper，输入 `C x H x W` float 图像 tensor，输出 `H x W` float mask：

- 单通道图像直接使用该通道。
- 多通道图像对通道求均值。
- `intensity >= threshold` 的像素为 `1.0`。
- 低于阈值的像素为 `0.0`。
- threshold 为 `0.0` 时，仍可返回全 1 mask 或跳过过滤；行为必须与旧逻辑等价。

该 helper 放在 `modules/data` 或 `modules/core` 中，避免训练和推理重复实现。

### 推理过滤

`modules/infer/pipeline.cpp` 在读取图像后生成 intensity mask，并传给特征解码。

`decode_feature_maps` 增加可选 mask 输入：

- 稀疏 heatmap top-k 前，先把无效区域的分数置为一个很小值。
- 如果 mask 内没有有效像素，返回空稀疏 keypoints、scores、descriptors、scale、orientation、affine。
- 半稠密 confidence 同时满足：`confidence >= semi_dense_threshold` 且 `intensity_mask` 有效。
- mask 空间尺寸以输入图像尺寸为准；如果 feature map 分辨率不同，使用 nearest resize 到 heatmap/confidence 尺寸。

`extract`、`match`、`eval` 都通过同一 `extract_feature_set` 路径获得过滤后的 `FeatureSet`。因此可视化和 `.pt` 输出自然一致。

### 训练过滤

`TrainConfig` 新增：

```cpp
double min_keypoint_intensity = 0.0;
```

训练中每个 `SyntheticPair` 进入 loss 前：

- 从 `view_a` 生成 `mask_a`。
- 从 `view_b` 生成 `mask_b`。
- 将 `mask_a` 与现有 `valid_mask` 相乘，限制源位置监督。
- 对需要目标位置有效性的 descriptor/repeatability，可通过 warp 后对应位置或现有目标采样逻辑保证目标落在 `mask_b` 内；最小实现中至少将 `mask_a` 合并进现有 `valid_mask`，并在 descriptor target 采样时排除目标低灰度位置。
- dense offset/confidence 使用合并后的 mask，低灰度区域不贡献损失。

缓存训练路径和在线训练路径必须行为一致：缓存中的 `view_a/view_b` 已保存为 tensor，加载后按阈值即时生成 mask，不需要改变缓存格式。

## 错误处理

- CLI 解析阶段拒绝小于 `0.0` 或大于 `1.0` 的阈值。
- 如果 mask 尺寸和图像/feature map 不兼容，抛出 `std::invalid_argument`。
- 推理时 mask 全空不报错，输出空特征集合；match/eval 按现有空特征逻辑继续。

## 测试计划

### 单元测试

- intensity mask：单通道图像按阈值生成正确 mask。
- intensity mask：多通道图像按通道均值生成 mask。
- decode sparse：低灰度位置即使 heatmap 分数最高，也不会被选为 keypoint。
- decode sparse：mask 全空时返回空稀疏特征。
- decode dense：半稠密点同时受 confidence 阈值和 intensity mask 限制。
- trainer：低灰度位置不会进入 descriptor sample indices。
- trainer：阈值非法时抛出参数错误。
- CLI：四个命令都能解析 `--min-keypoint-intensity`，非法值失败。

### 集成测试

- pipeline extract：构造一张暗区高 heatmap 倾向的测试图，设置阈值后输出 keypoint 不在暗区。
- pipeline match：两张图设置阈值后仍能写出 `.pt` 和匹配可视化。
- train：指定阈值时训练成功并写出可加载 checkpoint。

### 验证命令

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

## 文档更新

更新：

- `README.md`
- `docs/training.md`
- `docs/usage.md`

说明内容：

- 行星影像暗背景和低灰度边缘伪影可以用 `--min-keypoint-intensity` 排除。
- 默认 `0.0` 保持兼容。
- 推荐从 `0.05` 到 `0.1` 之间试起，结合 `--visualization-dir` 查看过滤效果。
- 训练和推理使用同一阈值，避免训练学到低灰度问题区域。

## 验收标准

- 设置 `--min-keypoint-intensity 0.08` 后，明显暗背景上的特征点不再出现在 `extract` 可视化中。
- `match` 的稀疏和半稠密结果不包含低灰度无效区域点。
- 训练指定该参数后仍能完成，且低灰度区域不参与主要损失监督。
- 默认不指定该参数时，现有测试和输出行为保持兼容。
