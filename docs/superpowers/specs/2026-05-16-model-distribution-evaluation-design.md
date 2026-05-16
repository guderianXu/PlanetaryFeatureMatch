# 模型、特征分布与实验闭环优化设计

## 背景

当前模型已经可以完成真实图像训练、特征提取、匹配、评估、缓存训练对、CUDA 训练/推理、低灰度过滤和可视化。但用户在真实行星/小天体影像上观察到几个问题：

1. 模型规模偏小，GPU 训练功耗只能到满功耗约一半，可能说明计算量不足或训练流水线不够高效。
2. 特征点可视化分布不好，常集中在某些纹理、边缘或高响应区域，主体大面积区域没有点。
3. 命令行训练和推理缺少总耗时/阶段耗时输出，不方便判断进度和瓶颈。
4. 当前训练没有明确训练集、验证集、测试集划分，缺少可复现实验闭环。

第一阶段优化主判断指标以用户选择的“可视分布”为准：特征点应覆盖行星/小天体主体的大部分有效区域，避免只聚集在少数局部区域。

## 目标

分阶段建立一个更可靠的行星影像特征训练与评估闭环：

1. 优先改善特征点空间分布，让可视化图更符合主体结构。
2. 增强模型表达能力和训练负载，但避免一次性引入过大架构导致调试困难。
3. 增加训练、验证、测试划分，让训练过程可复现、可比较。
4. 增加 CLI 耗时统计，能看到总耗时和关键阶段耗时。

## 非目标

- 不在第一阶段引入大型 Transformer、LoFTR 级别全局匹配架构。
- 不改变当前 `.pt` 特征和匹配结果的核心字段格式，除非后续单独设计版本迁移。
- 不把可视分布当成唯一长期指标；它只是第一阶段主观质量门槛，后续仍需要匹配指标和评估报告支撑。

## 推荐落地顺序

### 阶段 1：特征点分布优化

在 `modules/infer/feature_extractor.cpp` 的 sparse keypoint decode 路径中增加空间均匀化策略：

1. 保留已有低灰度 mask 过滤。
2. 对 heatmap 做局部 NMS，避免相邻位置重复出点。
3. 按网格分块做 top-k，每个网格最多取固定数量候选。
4. 如果网格均匀采样后数量不足，再从剩余全局高分候选中补足 `max_keypoints`。

建议新增 CLI 参数：

- `--keypoint-grid-rows`：默认 8。
- `--keypoint-grid-cols`：默认 8。
- `--keypoints-per-cell`：默认由 `max_keypoints / (rows * cols)` 推导，最少 1。
- `--nms-radius`：默认 4 个特征图像素。

推理、匹配、评估都复用同一 decode 配置。训练暂不直接依赖该 decode 策略；训练侧继续优化监督 mask 和模型输出。

验收：同一张真实图像输出的 feature PNG 中，特征点不再集中在左上、边缘或局部纹理块，而是覆盖主体大多数有效区域。

### 阶段 2：CLI 耗时统计

为 `train/extract/match/eval/export` 增加统一计时输出。

建议输出：

- 所有命令输出 `elapsed=<seconds>s`。
- `train` 每 epoch 输出 `epoch_time`，训练完成输出 `total_time` 和 `avg_batch_time`。
- `extract` 输出粗粒度阶段耗时：image load、model forward、decode、save、visualization。
- `match` 输出两张图特征提取、match、save、visualization 的阶段耗时。
- `eval` 输出 pair 数、总耗时、平均每 pair 耗时。

实现上新增小型 `Timer`/`ScopedTimer` 工具模块，避免每个命令手写重复 chrono 逻辑。

验收：命令行最后明确显示总耗时；训练过程可以看到 epoch 级进度和耗时。

### 阶段 3：训练/验证/测试划分

新增可复现实验划分，优先采用自动比例划分而不是要求用户手写三个目录。

建议新增训练参数：

- `--val-ratio`：默认 0.1。
- `--test-ratio`：默认 0.1。
- `--split-seed`：默认 42。
- `--split-manifest`：可选输出路径；未指定时写到 checkpoint 同目录。

行为：

1. 读取 `image_dir` 后按排序路径和 seed 洗牌。
2. 划分 train/val/test。
3. 训练只使用 train split。
4. 每个 epoch 后在 val split 上跑轻量评估：至少输出 val loss；如果成本允许，再输出平均稀疏匹配数和半稠密覆盖率。
5. test split 不参与训练过程，仅用于最终 `eval` 或后续独立测试命令。

验收：相同 `image_dir + split_seed + ratios` 得到稳定 manifest；训练日志包含 train loss 和 val 指标。

### 阶段 4：中等规模模型架构增强

在完成分布策略和实验闭环后，再增强模型，避免模型变大但缺少判断依据。

建议架构：

1. Backbone 从当前每层一个 stride-2 conv，升级为中等规模残差 backbone。
2. 每个 stage 使用 `Conv-BN-ReLU + residual block`。
3. 增加简单 FPN 融合：把深层语义上采样并与浅层/中层特征融合。
4. SparseHead 使用融合后的较高分辨率特征，提升点定位和纹理上下文。
5. DenseHead 使用浅层或融合层特征，保留位置通道。

建议默认参数：

- `base_channels=16`。
- `descriptor_dim=64`。
- 保留 CLI 或 config 可配置入口，方便在小显存机器上退回较小模型。

兼容策略：

- Checkpoint config 中增加 `model_variant` 或 `architecture_version`。
- 推理加载时根据 checkpoint 配置构造对应模型。
- 不要求旧 checkpoint 自动升级；旧 checkpoint 按旧架构加载，新 checkpoint 按新架构加载。

验收：新模型能完成训练、导出、提取、匹配和评估；默认训练负载高于旧模型；可视化分布不退化。

## 测试策略

1. 特征分布：构造 synthetic heatmap，验证 NMS 后相邻重复点被抑制，网格 top-k 后每个 cell 不超过限制。
2. CLI 参数：解析新增分布参数、split 参数和计时输出相关行为。
3. 数据划分：固定路径列表和 seed，验证 train/val/test 数量、互斥性和稳定性。
4. 计时输出：用命令 pipeline 测试验证输出包含 `elapsed=`、`epoch_time=` 或对应阶段字段。
5. 模型架构：验证新 backbone/FPN 输出 shape、checkpoint 可保存加载、旧架构 checkpoint 仍可加载。
6. 集成：`pfm_tests` 和 `ctest` 全通过。

## 风险与取舍

- 网格均匀化可能牺牲少数最高响应点，但能显著改善主体覆盖；第一阶段按可视分布优先。
- 模型变大可能提高 GPU 利用率，但也会提高训练时间和显存占用；因此放到实验闭环之后。
- 自动 split 对小数据集可能导致 val/test 太少；实现时需要保证每个非空 split 的数量可解释，极小数据集可以降级为仅 train。
- 计时输出中的 `.item()` 同步会影响训练性能；耗时统计应避免增加额外 GPU 同步，尽量复用已有同步点。

## 实施边界

第一轮实现建议只做阶段 1 和阶段 2：特征点分布优化与 CLI 耗时统计。阶段 3 和阶段 4 在第一轮效果确认后继续实现。这样可以最快验证用户最关心的可视分布问题，同时建立基础耗时观测能力。
