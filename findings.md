# 发现记录

## 模块边界整理 (2026-05-18)
- 最终目录结构：`modules/` 保留 runtime、logging、dataloader（基础部分）、core 四个通用模块；`src/` 包含 augment、cli、data、eval、geometry、infer、losses、models、train 九个项目特有模块。
- include 模式为模块相对路径（如 `"data/image_dataset.h"`），无需 `modules/` 或 `src/` 前缀，因此文件移动后只需在 CMake 中调整 include 目录，源文件基本无需修改。
- `synthetic_pair_dataset` 从 `modules/dataloader/` 移至 `src/data/`，因为它是项目特有的 online synthetic pair 生成逻辑，不属于通用 dataloader 基础设施。
- `pfm_augment` 保持为 OBJECT target 并入 `pfm`，链接关系不变。

## 模块化训练基础设施 / AsyncDataLoader
- 当前 reusable infrastructure 计划位于 `docs/superpowers/plans/2026-05-18-reusable-training-infrastructure.md`，设计位于 `docs/superpowers/specs/2026-05-18-reusable-training-infrastructure-design.md`。
- `runtime`、`dataloader` 的基础测试已存在并接入全量 `pfm_tests`。
- 全量测试曾在 `asyncDataLoaderFailedAsyncResetLeavesLoaderExhausted()` 崩溃；根因不是 BlockingQueue 本身，而是 `AsyncDataLoader::reset()` 在 `_sampler->indices()` 抛异常后提前退出，留下 `_queue == nullptr` 且 `_exhausted == false`，后续 `next()` 进入 async 分支调用 `_queue->pop()` 导致段错误。
- 最小修复是让 `reset()` 先进入安全 exhausted 状态并清空索引；只有 sampler 成功、async worker 启动成功后才把 `_exhausted` 置回 false。
- 修复后 `cmake --build build -j$(nproc) && ./build/pfm_tests` 通过，输出 `273 test(s) passed`。

## Training logging module
- 新增 `modules/logging/`，包括 `TrainingMetric`、`TrainingMetricLogger`、`ConsoleProgressLogger`、`CsvMetricLogger`、`NullGpuMetricProvider`、`MetricLoggerGroup`。
- `pfm_logging` 是独立 CMake target，并链接到 `pfm` / `pfm_tests`。
- 当前 logging module 还未集成进 trainer；下一阶段可用 `MetricLoggerGroup` 替换 trainer 里的直接 `std::cout`，并用 `CsvMetricLogger` 支持 `--log-csv`。
- logging module 实现后 `cmake --build build -j$(nproc) && ./build/pfm_tests` 通过，输出 `277 test(s) passed`。
- optional NVML provider 通过 `PFM_ENABLE_NVML` 控制；未找到 NVML headers/library 时仍编译 `NullGpuMetricProvider` fallback，避免普通开发环境被 GPU 指标依赖阻塞。

## Augmentation and synthetic pair dataset
- `modules/augment/` 已抽取 deterministic transform sampling 与 image-pair augmentation；`modules/data/synthetic_pair.cpp` 现在只负责旧配置到新配置的兼容转换。
- `pfm_augment` 需要作为 OBJECT target 并入 `pfm`，否则 augmentation 代码会反向依赖 `pfm` 中的 geometry/data 符号并产生静态库链接顺序问题。
- `SyntheticPairTensorDataset` 将内存中的 CHW float images 适配为 `TensorDataset`，按 `source_index = index / pairs_per_image`、`variant_index = index % pairs_per_image` 生成 `view_a/view_b/warp_a_to_b/valid_mask`。
- dataset adapter 实现后 `cmake --build build -j$(nproc) && ./build/pfm_tests` 通过，输出 `281 test(s) passed`。

## Trainer logging/DataLoader integration
- `TrainConfig::log_csv` 启用 per-iteration CSV metrics，列包含 feature/matcher/dense loss、descriptor accuracy/diversity、offset error，以及可选 GPU utilization/power。
- `--log-csv` 只增加一个高层日志输出参数，不暴露底层 logger 细节。
- `dataloader_workers > 0` 且未使用 synthetic pair cache 时，trainer 会预加载 source images 并通过 `SyntheticPairTensorDataset + AsyncDataLoader` 在线生成 pair；cache 路径保持原有同步读取。
- `pin_memory` 通过 DataLoader options 传递；无 CUDA/pinned-memory 支持环境下对应 dataloader 测试允许 runtime fallback 错误信息。
- trainer logging/DataLoader 集成后 `cmake --build build -j$(nproc) && ./build/pfm_tests` 通过，输出 `284 test(s) passed`。

## --min-keypoints 上下文
- `min_keypoints` 是软下限：只从有效强度掩码内、按分数补入原本被 NMS 抑制的候选，不突破 `max_keypoints`，候选不足时输出实际数量。

## 模型优化上下文 (2026-05-19)
- 当前模型由 Backbone、SparseHead、DenseHead、PlanetaryGraphMatcher 组成；默认结构参数文档为 base_channels=32、descriptor_dim=128、graph_hidden_dim=256、graph_attention_layers=4。
- 训练 loss 包括 repeatability、descriptor candidate CE、graph matching CE、dense smooth L1 offset、confidence BCE 和 descriptor diversity。
- 最近已完成 runtime/dataloader/logging 基础设施、online synthetic pair DataLoader、CSV 指标和训练可视化，适合先用指标驱动模型优化。
- 当前环境 `pfm-cf` 可编译并通过 `build-pfm-cf` 测试：284 test(s) passed，CTest 1/1 passed。

## Matching loss 不下降复现 (2026-05-19)
- 用用户命令的短版复现：epochs=3、batch=2、resize=512、pairs_per_image=15、mixed、min_keypoint_intensity=0.05、min_keypoints=1024、CUDA、CSV 指标。
- 指标证据：`graph_matching_loss` 首段均值约 3.9247，末段均值约 3.9361；epoch 1/2 均值均约 3.763，基本不降。
- 同一训练中 descriptor_loss 从首段均值约 2.717 降到末段约 0.295，descriptor_accuracy 从约 0.40 升到约 0.93；dense_loss 也从约 0.831 降到约 0.065。
- 代码证据：`PlanetaryGraphMatcherImpl::forward()` 接收 keypoints 但当前第 140-141 行显式 `(void)keypoints_a/b`，没有使用空间几何信息。
- 代码证据：`make_graph_matching_loss()` 用 256 个固定 descriptor 网格样本，B 侧只取 warp 后正样本集合，target 是 `0..sample_count-1`；监督问题更像“对一组正样本集合排序”，缺少真实关键点候选池和 dustbin/负样本结构。
- 初步根因假设：graph matcher 的输入/监督与实际 sparse matching 任务不一致，且未利用关键点几何，所以 loss 难以稳定下降；descriptor 分支能学会说明基础描述子监督不是主要瓶颈。

## Keypoint graph matching loss implementation findings (2026-05-19)
- 原 graph matcher 接口接收 keypoints 但实现忽略它们；加入 `_keypoint_projection` 后，same descriptors + different keypoints 会产生不同 logits。
- decoded `FeatureSet::keypoints` 是 sparse feature-map 坐标；`warp_a_to_b` 和 `valid_mask` 是 image-space。因此训练目标分配必须先把 keypoints 按 feature map 尺寸缩放到 image-space；graph matcher 本身仍应使用 inference 路径使用的 feature-map keypoints。
- `valid_mask` 由原始有效区域、A/B 强度 mask 共同构成；graph target assignment 必须同时检查 source 像素和 warped target 像素，否则会把无效目标监督为正样本。
- `decode_feature_maps()` 要求 dense confidence 与 sparse heatmap 空间尺寸一致；训练路径中 dense head confidence 需要 nearest resize 后才能作为 decode 输入。
- 完整 CUDA 短训练显示 graph loss 明显下降：4140 条 iteration 的 first_mean≈4.6967、last_mean≈0.0335，最后 10 条约 0.0010-0.0020；这与之前 fixed-grid graph loss 首末均值约 3.92/3.94 不下降形成对比。

## Full training results after keypoint graph supervision (2026-05-20)
- 完整 10 epoch CUDA 训练验证了 keypoint graph supervision 不只在短跑有效：13,800 条 iteration 中 graph_matching_loss 从首段均值≈2.17615 降到末段≈0.10262。
- 特征分支同步改善：descriptor_accuracy 从首段均值≈0.374187 升到末段≈0.95367；feature_loss 从≈2.78217 降到≈0.333686。
- dense 分支也稳定下降：dense_loss 从≈0.754952 降到≈0.0857591，offset_error_px 从≈156.817 降到≈16.3463。
- checkpoint `train_full.pt` 和训练诊断 `metrics_full.csv`/`vis_full/` 是本地训练产物，不应默认提交到 git。
