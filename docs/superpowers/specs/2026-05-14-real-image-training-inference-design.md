# 真实图片训练与推理闭环设计

## 目标

第一阶段把当前 CLI validation stub 扩展为真实可运行的 C++/LibTorch 闭环：读取真实行星图片，训练一个可保存 checkpoint 的最小模型，执行特征提取，执行双图匹配，导出 `.pt` 特征和匹配结果，并提供基础效果评估。

本阶段采用“效果优先 MVP”，不是只验证命令能跑通。验收目标是：小数据训练后 loss 能下降，`extract` 能输出可加载特征，`match` 能输出可加载匹配结果，`eval` 能报告匹配精度、半稠密覆盖率和平均匹配数。

## 范围

### 本阶段实现

- OpenCV 读取真实图片：`png`、`jpg`、`jpeg`、`tif`、`tiff`。
- 支持 8 位和 16 位灰度/RGB 输入。
- 图像转为 LibTorch `C x H x W` float tensor，并归一化到 `[0, 1]`。
- 从真实图片生成自监督合成视图对。
- 训练循环、loss 聚合、checkpoint 保存。
- checkpoint 加载后执行 `extract`、`match`、`eval`、`export`。
- `.pt` 特征文件与 `.pt` 匹配结果文件。
- 每个新模块配套就近 `*_test.cpp`。

### 本阶段不实现

- Python 训练或推理。
- 复杂日志系统、可视化面板、分布式训练。
- 完整生产级模型导出后端。
- 高级 resume 策略；可在 checkpoint 可加载基础上后续扩展。
- JSON 大体积特征导出；第一阶段统一使用 `.pt`。

## 模块设计

```text
modules/data/
  image_io.h/.cpp        # OpenCV 读取和 tensor 转换
  image_io_test.cpp
  image_dataset.h/.cpp   # 图片目录遍历与样本读取
  image_dataset_test.cpp
  synthetic_pair.h/.cpp  # 扩展真实图像合成监督
  synthetic_pair_test.cpp

modules/train/
  trainer.h/.cpp         # 训练配置、模型封装、训练循环、checkpoint
  trainer_test.cpp

modules/infer/
  feature_codec.h/.cpp   # .pt 特征保存/读取
  feature_codec_test.cpp
  match_codec.h/.cpp     # .pt 匹配结果保存/读取
  match_codec_test.cpp
  pipeline.h/.cpp        # CLI 命令真实执行

modules/models/
  保留 Backbone、SparseHead、DenseHead、Matcher
```

继续保留当前 module-first 组织方式，不引入 `include/` 与 `src/` 分离结构。

## 数据流

### train

1. `train --image-dir images --checkpoint model.pt ...`。
2. `ImageDataset` 遍历真实图片。
3. `image_io` 用 OpenCV 读取图像并转为 `C x H x W` tensor。
4. `synthetic_pair` 从真实图像生成 `view_a`、`view_b`、warp field 和 valid mask。
5. 模型前向：backbone、sparse head、dense head、matcher。
6. 计算并聚合：
   - `repeatability_loss`
   - `descriptor_cross_entropy_loss`
   - `masked_l1_loss`
   - `confidence_bce_loss`
7. 使用 AdamW 更新参数。
8. 每个 epoch 输出平均 loss。
9. 保存 checkpoint 到 `--checkpoint`。

### extract

1. 加载 checkpoint。
2. 读取 `--image`。
3. 前向模型。
4. 从 heatmap 解码 top-k keypoints。
5. 在 keypoint 位置采样 descriptor、scale、orientation、affine。
6. 生成半稠密候选点和 confidence。
7. 保存 `.pt` 特征文件。

`extract` 输出字段：

```text
keypoints           [N,2]
scores              [N]
descriptors         [N,D]
scale               [N]
orientation         [N]
affine              [N,2,2]
dense_points        [K,2]
dense_confidence    [K]
```

### match

1. 加载 checkpoint。
2. 分别读取 `--image-a` 和 `--image-b`。
3. 对两张图执行特征提取。
4. 稀疏匹配使用 descriptor similarity。
5. 过滤策略：top-k、互检 cross-check、可选 ratio-like margin。
6. 半稠密匹配使用 dense head 置信度与 offset 生成点对应。
7. 保存 `.pt` 匹配结果。

`match` 输出字段：

```text
sparse_matches  [M,2]
sparse_scores   [M]
points_a        [K,2]
points_b        [K,2]
confidence      [K]
```

### eval

第一阶段支持两种来源之一：

- 读取 `--pairs` 中的真实图片对路径，运行 match 并报告平均匹配数和平均置信度。
- 或对图片生成合成验证对，使用已知 warp 计算 `matching_precision` 和 `semi_dense_coverage`。

优先实现合成验证对指标，因为它有可量化真值。

### export

第一阶段 `export` 加载 checkpoint，并重新保存为 inference `.pt`。它的验收标准是：导出的文件可被 `extract` 和 `match` 加载。

## 图像 IO

使用 OpenCV 的 `imread(..., IMREAD_UNCHANGED)` 保留位深和通道。转换规则：

- 8-bit：除以 `255.0`。
- 16-bit：除以 `65535.0`。
- 灰度：输出 `1 x H x W`。
- RGB/BGR：OpenCV 读取后转换为 RGB，输出 `3 x H x W`。
- 不支持的通道数、位深或空图像抛异常。

## 训练效果策略

第一阶段增强包括：

- 随机亮度、对比度、gamma。
- 高斯噪声。
- 整数平移。
- 小角度仿射和尺度变化；如果 OpenCV warp 与监督实现风险过高，先落地平移加光照扰动，再扩展仿射。

训练成功标准：

- 使用小数据集可以完成至少 1 个 epoch。
- 训练 loss 在短跑中有下降趋势。
- checkpoint 文件存在且可加载。
- 同一 checkpoint 能被 `extract`、`match`、`eval`、`export` 使用。

## 错误处理

- CLI 层保留 CLI11 参数错误。
- 系统边界做严格校验：文件不存在、图片读取失败、不支持格式、checkpoint 缺失、输出目录不可写均返回非 0。
- 模块内部继续使用异常表达非法输入。
- 不做静默 fallback；训练或推理无法继续时直接失败并输出明确错误。

## 测试策略

遵循 TDD。每个模块先加失败测试，再实现。

必须覆盖：

- `image_io`：读取 8-bit/16-bit、灰度/RGB、空路径、非法路径。
- `image_dataset`：目录遍历、扩展名过滤、空目录错误。
- `trainer`：小 batch 训练一步、loss finite、checkpoint 可保存加载。
- `feature_codec`：保存/读取 `.pt` 字段和形状。
- `match_codec`：保存/读取 `.pt` 字段和形状。
- `pipeline`：train/extract/match/eval/export 命令真实写出文件。
- 回归测试：`match --semi-dense-threshold` 必须继续可用。

最终验证命令：

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
./build/pfm_cli train --image-dir <images> --checkpoint model.pt --epochs 1 --batch-size 1
./build/pfm_cli extract --image <image> --checkpoint model.pt --output features.pt
./build/pfm_cli match --image-a <a> --image-b <b> --checkpoint model.pt --output matches.pt
./build/pfm_cli eval --pairs <pairs> --checkpoint model.pt --output report.pt
./build/pfm_cli export --checkpoint model.pt --output exported.pt
```
