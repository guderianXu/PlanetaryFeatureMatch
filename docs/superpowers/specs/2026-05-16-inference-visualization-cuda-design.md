# 推理 CUDA 与可视化输出设计

## 背景

训练时 GPU 功耗稳定在约 150W，不一定表示没有使用 CUDA。当前项目的训练 forward/backward/loss 已经可以在 CUDA 上执行，但图像读取、合成数据生成/缓存准备、训练样本搬运、推理解码、匹配后处理和 `.pt` 写出仍有 CPU 工作。如果 batch 较小、输入分辨率较低、模型较轻，GPU 会等待 CPU 或数据搬运，功耗不会接近 360W 满载。

推理侧现有 `--device cuda` 已经把模型 forward 放到 CUDA：extract/match 会解析设备、把模型模块移动到 device，并把输入 tensor 放到 device。模型输出随后会回到 CPU 做特征解码、匹配后处理和文件写出。本次设计不重写匹配后处理为 GPU kernel，只补齐文档说明，并新增 PNG 可视化输出，方便人工观察模型效果。

## 目标

- 为 `extract` 和 `match` 增加统一参数 `--visualization-dir DIR`。
- `extract` 在目录中保存带特征点覆盖层的 PNG。
- `match` 在目录中保存左右拼接、带匹配线的 PNG。
- 不改变现有 `.pt` 输出格式，不影响未指定可视化目录时的行为和性能。
- 文档明确说明推理 CUDA 范围：模型 forward 在 GPU，解码、匹配后处理和 PNG/`.pt` 写出在 CPU。

## 非目标

- 不新增 GUI。
- 不把 sparse descriptor matching 改成 CUDA 实现。
- 不改变训练 loss 或模型结构。
- 不为 eval 批量可视化做完整实现；本次只覆盖 extract/match 命令。

## CLI 设计

### extract

```bash
./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --device cuda \
  --visualization-dir vis
```

输出：

```text
vis/a_features.png
```

图片内容：原图灰度或 RGB 显示，叠加 sparse keypoints。默认最多绘制已经由 `--max-keypoints` 选出的 sparse keypoints；如果 dense points 已存在，可用较浅颜色额外绘制 dense points，避免遮挡 sparse keypoints。

### match

```bash
./build/pfm_cli match \
  --image-a images/a.tif \
  --image-b images/b.tif \
  --checkpoint model.pt \
  --output matches.pt \
  --device cuda \
  --visualization-dir vis
```

输出：

```text
vis/a__b_matches.png
```

图片内容：左右拼接 `image_a` 和 `image_b`，按 `MatchSet` 中的 matched points 画连线。只画 sparse matches；置信度越高线越亮。为了避免图像太乱，默认最多绘制 200 条匹配，按 confidence 从高到低选取。

## 文件命名

- 使用输入图片的 stem 作为基础名。
- `extract`：`<image_stem>_features.png`。
- `match`：`<image_a_stem>__<image_b_stem>_matches.png`。
- 如果 stem 为空或包含不适合文件名的字符，替换为下划线。
- `--visualization-dir` 指定的目录不存在时自动创建。

## 模块设计

新增模块：

- `modules/infer/visualization.h`
- `modules/infer/visualization.cpp`
- `modules/infer/visualization_test.cpp`

职责：

- 从输入图片路径读取可视化底图。
- 将 `FeatureSet` 的点绘制到 PNG。
- 将 `MatchSet` 的 matched points 绘制到左右拼接 PNG。
- 只接收已经生成的 `FeatureSet` / `MatchSet`，不调用模型，不参与训练。

修改模块：

- `modules/cli/commands.h/.cpp`：`CliOptions` 新增 `visualization_dir`，extract/match parser 增加 `--visualization-dir`。
- `modules/infer/pipeline.cpp`：extract/match 保存 `.pt` 后，如果 `visualization_dir` 非空，调用可视化模块写 PNG。
- `CMakeLists.txt` 和 `tests/test_main.cpp`：加入新模块和测试。
- `README.md`、`docs/usage.md`、`docs/training.md`：补充 CUDA 范围和可视化参数说明。

## 数据流

### extract

1. CLI 解析 `--visualization-dir`。
2. pipeline 运行现有 extract 流程：读图、模型 forward、decode、保存 `.pt`。
3. 如果 `visualization_dir` 非空：
   - 创建目录。
   - 读取原图作为 OpenCV Mat。
   - 绘制 `feature_set.keypoints`。
   - 写出 `<image_stem>_features.png`。

### match

1. CLI 解析 `--visualization-dir`。
2. pipeline 运行现有 match 流程：读两张图、模型 forward、decode、match、保存 `.pt`。
3. 如果 `visualization_dir` 非空：
   - 创建目录。
   - 读取两张原图。
   - 左右拼接。
   - 根据 `match_set.points_a`、`match_set.points_b` 和 `match_set.confidence` 绘制最多 200 条匹配线。
   - 写出 `<image_a_stem>__<image_b_stem>_matches.png`。

## 错误处理

- `--visualization-dir` 为空：不生成 PNG，不报错。
- 图片读取失败：抛出 `std::invalid_argument`，错误信息包含图片路径。
- 目录创建失败或 PNG 写出失败：抛出 `std::invalid_argument`，错误信息包含输出路径。
- 没有 keypoints 或 matches：仍写出原图/拼接图，图上不画点或线。

## 测试计划

- CLI parser 测试：extract/match 能解析 `--visualization-dir`。
- visualization 单元测试：
  - feature visualization 写出 PNG 文件。
  - match visualization 写出 PNG 文件。
  - 空 keypoints/matches 仍写出 PNG。
- pipeline 测试：
  - extract 指定 `visualization_dir` 后生成 `.pt` 和 features PNG。
  - match 指定 `visualization_dir` 后生成 `.pt` 和 matches PNG。
- 回归测试：未指定 `visualization_dir` 时现有 extract/match 行为不变。

## 验证命令

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
./build/pfm_cli extract --help
./build/pfm_cli match --help
```
