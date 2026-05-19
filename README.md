# PlanetaryFeatureMatch

PlanetaryFeatureMatch 是一个基于 C++17、LibTorch 和 OpenCV 的行星影像局部特征提取与匹配项目，面向火星、月球和小行星等场景。行星影像常见弱纹理、光照变化大、视角差异、成像畸变、相机倾斜和局部形变，传统局部特征与仅依赖 RANSAC 的流程在这些场景下容易失效。

当前版本已经打通第一阶段真实闭环：读取真实图像、训练最小模型、保存 checkpoint、提取 `.pt` 特征、执行双图匹配、按 pairs 文件评估，并导出可用于推理的 checkpoint。该阶段重点是端到端可运行和可测试，尚不代表最终匹配精度。

## 已实现能力

- OpenCV 图像读取：支持常见 8/16 位灰度图和 RGB/BGR 图像。
- LibTorch 训练：使用真实图像生成自监督合成图像对，并以较大的默认模型联合训练特征提取、图匹配和半稠密偏移。
- 合成对缓存：可在训练前把变换后的图像和 `.pt` 监督文件生成到指定目录，后续训练直接复用。
- CUDA 设备选择：训练和推理 forward 支持 `cpu`、`cuda`、`cuda:N`。
- Checkpoint：使用 LibTorch `.pt` archive 保存和加载模型，CUDA 训练后仍保存为 CPU 权重。
- 特征提取：输出稀疏关键点、描述子、尺度、方向、仿射形状和半稠密点。
- 双图匹配：稀疏描述子 mutual nearest-neighbor 匹配与半稠密点对应导出。
- 评估：对 pairs 文件中的图像对聚合平均匹配数、稀疏分数、半稠密置信度和覆盖率。
- 模型导出：校验 checkpoint 完整性后导出推理 checkpoint。
- 模块化测试：每个主要模块配套 `*_test.cpp`。

## 模型结构

当前模型采用稀疏与半稠密结合的行星影像 deep matcher 结构，默认容量已经从早期轻量 smoke 模型升级为 `base_channels=32`、`descriptor_dim=128`、`graph_hidden_dim=256`、`graph_attention_layers=4`：

- `Backbone`：共享多尺度特征提取，每个 stage 包含下采样卷积和 refinement 卷积。
- `SparseHead`：通过共享 context tower 输出关键点 heatmap、描述子、尺度、方向和仿射形状。
- `DenseHead`：融合两图特征、差分、坐标和小半径局部相关性，输出半稠密置信度和局部偏移。
- `PlanetaryGraphMatcher`：学习式双图匹配模块，基于关键点描述子和位置编码做多层自注意力、交叉注意力和 FFN 图推理，输出匹配 logits、置信度和未匹配 dustbin，用于替代旧的最近邻描述子匹配流程。

这个方向不是只做 SuperPoint 风格的稀疏特征，也不是只做 LoFTR 风格的稠密匹配，而是同时服务于后续稀疏匹配和半稠密匹配。匹配阶段默认使用 checkpoint 中训练好的 `PlanetaryGraphMatcher` 学习式匹配器，不再把最近邻描述子匹配作为正式推理流程。训练使用自监督合成图像对，基础损失包括 repeatability、descriptor cross entropy、graph matching cross entropy、Smooth L1 offset 和 confidence BCE；checkpoint 会把 backbone、sparse head、dense head、graph matcher 权重以及结构 metadata 保存到同一个 `.pt` 文件，便于训练、推理、评估和导出保持一致。

模型文件大小不能直接和 YOLO 对比：YOLO 是多类别目标检测器，包含更大的检测 backbone、neck、多尺度 detection heads 和类别输出层；本项目是灰度行星影像局部特征匹配模型，目标是匹配准确性和几何稳定性，不是用文件大小衡量能力。

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

如果要检查训练生成数据、有效监督区域、当前模型特征点和训练 pair 上的匹配情况，可以给训练命令加诊断目录：

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --visualization-dir build/train_vis \
  --min-keypoint-intensity 0.05 \
  --max-keypoints 2048 \
  --min-keypoints 512 \
  --keypoints-per-cell 8 \
  --nms-radius 2
```

默认每个 epoch 输出前 4 个训练 pair 的模型相关诊断到 `epoch_000001/`、`epoch_000002/` 等子目录；需要每个 epoch 输出全部训练 pair 时使用 `--visualization-samples all`，需要临时关闭诊断写图时使用 `--visualization-samples 0`。`view_a`、`view_b`、`valid_mask` 和 `warp_matches` 是合成 pair 的静态检查图，只会写到 `static/` 一次；每个 epoch 只重复写会随模型变化的 `features_a`、`features_b` 和 `model_matches`。诊断图的特征解码、匹配计算、画图、PNG 编码与写入由 4 个后台线程异步完成，训练结束前会 flush；每个 pair 的静态图、特征图和匹配图会拆成多个后台任务，较轻的 `features_a/features_b` 不需要等待较慢的 `model_matches` 渲染完成。`features_a/features_b/model_matches` 复用当前 batch 的模型 forward 结果，避免为诊断图额外跑一遍 backbone/head。`model_matches` 为避免半稠密匹配过多拖慢训练，最多绘制 2048 条稀疏线和 2048 条半稠密线，但左上角统计仍显示完整匹配数量。训练诊断里的 `features_a/features_b/model_matches` 使用 `--max-keypoints`、`--min-keypoints`、`--keypoint-grid-rows`、`--keypoint-grid-cols`、`--keypoints-per-cell` 和 `--nms-radius` 解码特征点；低灰度阈值导致有效区域变小时，可以提高 `--max-keypoints`、设置 `--min-keypoints`，或显式设置 `--keypoints-per-cell` 增加亮区内的候选点数量。

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

推理阶段的稀疏特征点默认会先应用低灰度过滤，再做局部 NMS，随后按网格分块选点，最后用全局高分候选补足到 `--max-keypoints` 以内。`--min-keypoints` 是软下限：如果 NMS 后点太少，会在强度掩码允许的区域内放松局部抑制继续补点，但不会超过 `--max-keypoints`，也不会从低灰度无效区域取点。`--keypoints-per-cell 0` 表示根据 `max_keypoints` 和网格数量自动推导，采用向上取整且每个 cell 至少 1 个候选。

匹配两张图像：

```bash
./build/pfm_cli match \
  --image-a images/a.tif \
  --image-b images/b.tif \
  --checkpoint model.pt \
  --output matches.pt \
  --match-mode both \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5 \
  --visualization-dir vis
```

如果已经用 `extract` 得到两张图的特征文件，可以直接复用，避免重新跑模型提取：

```bash
./build/pfm_cli match \
  --feature-a a_features.pt \
  --feature-b b_features.pt \
  --output matches.pt \
  --match-mode sparse
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
./build/pfm_cli train --image-dir images --checkpoint model.pt [--epochs 1] [--batch-size 1] [--device cpu] [--resize 512] [--pairs-per-image 1] [--augmentation-profile mixed] [--extreme-pair-ratio 0.2] [--synthetic-pair-cache-dir build/pair_cache] [--synthetic-pair-cache-rebuild] [--log-csv metrics.csv] [--dataloader-workers 0] [--prefetch-batches 2] [--pin-memory] [--visualization-dir build/train_vis] [--visualization-samples 4|all] [--min-keypoint-intensity 0.0] [--max-keypoints 1024] [--min-keypoints 0] [--keypoint-grid-rows 8] [--keypoint-grid-cols 8] [--keypoints-per-cell 0] [--nms-radius 4]
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
- `--log-csv`：逐 iteration 写出训练指标 CSV，包含 `loss_total`、`feature_loss`、`graph_matching_loss`、`dense_loss`、`offset_error_px`，以及可用时的 GPU 利用率和功耗。
- `--dataloader-workers`：在线合成 pair 的异步 DataLoader worker 数，默认 0 表示沿用同步生成；大数据训练可设为 2、4 或更高以提前准备 batch。指定 `--synthetic-pair-cache-dir` 时优先读取缓存，不走在线 DataLoader。
- `--prefetch-batches`：异步 DataLoader 预取 batch 数，默认 2，必须为正数；worker 较多或图像预处理较慢时可适当调大。
- `--pin-memory`：让 DataLoader 在 CPU batch 上尝试 pinned memory，配合 CUDA 训练可减少拷贝开销；不支持 pinned memory 的环境会报告明确错误。
- `--visualization-dir`：训练诊断 PNG 输出目录；未指定时不写诊断图。
- `--visualization-samples`：每个 epoch 的诊断输出样本数，默认 4；传 `all` 时每个 epoch 输出每个训练 pair，传 0 时即使设置了 `--visualization-dir` 也不生成 PNG。
- `--min-keypoint-intensity`：关键点监督和输出的最低归一化灰度阈值，默认 0.0，取值范围 `[0, 1]`。
- `--max-keypoints`：训练诊断图中每张图最多解码多少个稀疏特征点，默认 1024。
- `--min-keypoints`：训练诊断图中希望尽量达到的稀疏特征点软下限，默认 0 表示不启用。它只会在 `--min-keypoint-intensity` 允许的区域内补点；如果有效候选不足，则输出实际可用数量。该值不能大于 `--max-keypoints`。
- `--keypoint-grid-rows` / `--keypoint-grid-cols`：训练诊断特征点的空间均匀分布控制。程序会把特征图划分成 `rows × cols` 个网格 cell，先在每个 cell 内按分数选点，避免所有点都集中在高纹理的小区域。默认 `8x8`；行列数越大，分布约束越细，但每个 cell 面积更小。
- `--keypoints-per-cell`：每个网格 cell 最多优先保留多少个候选点。默认 0 表示自动按 `ceil(max_keypoints / (rows × cols))` 推导，并保证每个 cell 至少 1 个候选。低灰度阈值过滤后有效区域变小时，可以手动调大这个值，让亮区 cell 内保留更多点。
- `--nms-radius`：训练诊断特征点局部非极大值抑制半径，单位是特征图像素。半径越大，相邻特征点会被压得越稀疏；半径越小，允许更密集的点。默认 4。

默认每轮使用目录中的全部训练图像；设置 `--pairs-per-image N` 后，每张图会派生 N 个不同合成 pair。未指定缓存且 `--dataloader-workers > 0` 时，训练会用 `SyntheticPairTensorDataset + AsyncDataLoader` 在线异步生成 pair；指定缓存目录时则先生成/复用 `.pt` 缓存，减少重复增强开销。训练时 `--min-keypoint-intensity` 会同时要求源视图和目标视图对应位置达到阈值，低灰度区域不参与 repeatability、descriptor、offset 和 confidence 监督，也不会作为训练诊断特征点输出。训练日志中 `feature_loss` 表示特征提取相关损失，`matcher_loss` / `graph_matching_loss` 表示图匹配候选分类损失，`dense_loss` / `offset_error_px` 表示半稠密偏移 refinement 质量。`--log-csv` 可把这些指标保存成表格，方便观察不同增强 profile 或 dataloader 设置下的 loss 波动。`--max-keypoints` 等解码参数只影响训练诊断 PNG 中当前模型特征点和匹配的可视化数量，不改变训练 loss。`--batch-size` 只控制每次反向传播的样本分组大小。默认模型比早期版本更大，训练会占用更多显存和时间；CUDA 训练时可以调大或调小 `--resize`、`--batch-size`、`--pairs-per-image`、`--dataloader-workers` 或训练图像数量，以控制 GPU 计算量和显存占用。显式请求 CUDA 时不会静默回退到 CPU；CUDA 不可用、索引越界或格式错误会直接失败。

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
  --synthetic-pair-cache-dir build/pair_cache \
  --log-csv build/train_metrics.csv \
  --visualization-dir build/train_vis \
  --min-keypoint-intensity 0.05 \
  --max-keypoints 2048 \
  --min-keypoints 512 \
  --keypoints-per-cell 8 \
  --nms-radius 2
```

### `extract`

```bash
./build/pfm_cli extract --image a.tif --checkpoint model.pt --output features.pt [--device cpu] [--max-keypoints 1024] [--min-keypoints 0] [--semi-dense-threshold 0.5] [--visualization-dir vis] [--min-keypoint-intensity 0.0] [--keypoint-grid-rows 8] [--keypoint-grid-cols 8] [--keypoints-per-cell 0] [--nms-radius 4]
```

- `--image`：输入图像路径，支持 OpenCV 能读取的常见格式，包括 TIFF/PNG/JPEG。
- `--checkpoint`：训练得到的模型 checkpoint。
- `--output`：输出特征文件路径，格式为 LibTorch `.pt` archive。
- `--device`：推理设备，默认 `cpu`；可写 `cuda` 或 `cuda:0`。
- `--max-keypoints`：最多输出多少个稀疏关键点，默认 1024。增大后可提供更多稀疏匹配候选，但后处理和匹配开销也会增加。
- `--min-keypoints`：希望尽量输出的稀疏关键点软下限，默认 0 表示不启用。它会在低灰度掩码允许的区域内放松 NMS 补点；如果有效区域候选不足，则输出实际数量，不会为了凑数使用低灰度无效区域。
- `--semi-dense-threshold`：半稠密点输出的置信度阈值，默认 0.5。阈值越低，半稠密点越多但可能更噪；阈值越高，点更少但置信度更高。
- `--visualization-dir`：特征点 PNG 可视化输出目录；未指定时只写 `.pt` 特征文件。
- `--min-keypoint-intensity`：按原图归一化灰度生成掩码，低于阈值的位置不会输出为稀疏关键点或半稠密点，适合过滤行星边缘低灰度噪声。
- `--keypoint-grid-rows` / `--keypoint-grid-cols`：把特征图划分成 `rows × cols` 个网格，先按 cell 分配关键点名额，让点尽量覆盖整幅图，而不是集中在少数高响应区域。
- `--keypoints-per-cell`：每个网格 cell 优先保留的候选点数；0 表示根据 `max_keypoints` 和网格数量自动推导。
- `--nms-radius`：稀疏关键点 NMS 半径，单位是特征图像素；增大可减少扎堆点，减小可保留更密集点。
- 输出内容包含稀疏关键点、分数、描述子、尺度、方向、仿射形状、半稠密点和半稠密置信度。

### `match`

```bash
./build/pfm_cli match --image-a a.tif --image-b b.tif --checkpoint model.pt --output matches.pt [--device cpu] [--feature-a a_features.pt] [--feature-b b_features.pt] [--match-mode both] [--max-keypoints 1024] [--min-keypoints 0] [--semi-dense-threshold 0.5] [--visualization-dir vis] [--min-keypoint-intensity 0.0] [--keypoint-grid-rows 8] [--keypoint-grid-cols 8] [--keypoints-per-cell 0] [--nms-radius 4]
```

- `--image-a` / `--image-b`：需要现场提取并匹配的两张输入图像。
- `--feature-a` / `--feature-b`：已经由 `extract` 生成的 `.pt` 特征文件；两者同时指定时直接复用特征，不再要求 `--checkpoint`，也不会重新提取。
- `--checkpoint`：现场从图像提取特征时使用的模型 checkpoint；复用两个特征文件时可不传。
- `--output`：输出匹配文件路径，格式为 LibTorch `.pt` archive。
- `--device`：现场提取特征时使用的设备，默认 `cpu`；复用特征文件时主要读取 CPU 后处理。
- `--match-mode`：匹配输出模式，默认 `both`；`sparse` 只保留稀疏描述子匹配，`dense` 只保留半稠密点对，`both` 两者都输出。
- `--max-keypoints`：现场提取特征时每张图最多输出的稀疏关键点数；复用特征文件时使用文件里已有点数。
- `--min-keypoints`：现场提取特征时尽量达到的稀疏关键点软下限；复用特征文件时不重新补点。它不会突破 `--max-keypoints`，也不会从低灰度无效区域取点。
- `--semi-dense-threshold`：现场提取特征时的半稠密置信度阈值；复用特征文件时使用文件里已有半稠密点。
- `--visualization-dir`：匹配 PNG 可视化输出目录；未指定时只写 `.pt` 匹配文件。
- `--min-keypoint-intensity`：只在现场从图像提取特征时生效，低灰度区域不会输出为特征点；复用特征文件时使用文件中已保存的特征点。
- `--keypoint-grid-rows` / `--keypoint-grid-cols`：现场提取稀疏关键点时的网格分布控制，让关键点更均匀覆盖图像。
- `--keypoints-per-cell`：现场提取时每个网格 cell 优先保留的候选点数；0 表示自动推导。
- `--nms-radius`：现场提取时的稀疏关键点 NMS 半径，单位是特征图像素。
- 稀疏匹配默认使用 checkpoint 中的 `PlanetaryGraphMatcher` 学习式匹配器，根据关键点描述子和位置通过自注意力/交叉注意力输出匹配 logits、候选匹配和置信度，不再使用旧的最近邻描述子匹配作为正式流程。即使复用 `extract` 生成的特征文件，`match` 仍需要 checkpoint 来加载训练好的匹配器权重。
- 半稠密匹配当前仍按两张图的半稠密点序列一一配对，置信度取两侧 dense confidence 的较小值；后续会继续升级为 coarse-to-fine 学习式半稠密匹配。

### `eval`

```bash
./build/pfm_cli eval --pairs pairs.txt --checkpoint model.pt --output report.pt [--device cpu] [--max-keypoints 1024] [--min-keypoints 0] [--semi-dense-threshold 0.5] [--min-keypoint-intensity 0.0] [--keypoint-grid-rows 8] [--keypoint-grid-cols 8] [--keypoints-per-cell 0] [--nms-radius 4]
```

- `--pairs`：评估图像对列表，每行写一对图像路径；路径包含空格时必须使用英文双引号。
- `--checkpoint`：评估时用于提取特征的模型 checkpoint。
- `--output`：输出评估报告路径，格式为 LibTorch `.pt` archive。
- `--device`：特征提取设备，默认 `cpu`；可写 `cuda` 或 `cuda:0`。
- `--max-keypoints`：每张评估图像最多输出的稀疏关键点数，影响稀疏匹配候选规模。
- `--min-keypoints`：每张评估图像尽量达到的稀疏关键点软下限；有效候选不足时输出实际数量。
- `--semi-dense-threshold`：半稠密点置信度阈值，影响半稠密覆盖率和匹配数量。
- `--min-keypoint-intensity`：低灰度过滤阈值；低于该归一化灰度的位置不参与特征输出。
- `--keypoint-grid-rows` / `--keypoint-grid-cols`：稀疏关键点网格分布控制，避免评估时关键点集中在局部区域。
- `--keypoints-per-cell`：每个网格 cell 优先保留的候选点数；0 表示自动推导。
- `--nms-radius`：稀疏关键点 NMS 半径，单位是特征图像素。
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
