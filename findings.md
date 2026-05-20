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

## Inference filtering findings (2026-05-20)
- 训练 loss 收敛不等于推理 sparse match 可直接使用；如果推理忽略 dustbin 并强制每个 A keypoint 输出 argmax B，会在真实图像上产生大量低质量 sparse matches。
- Graph matcher 推理需要同时使用 dustbin 过滤和 mutual nearest 过滤：非 dustbin 说明模型认为该 A 有候选，mutual nearest 避免多个 A 单向挤到同一个 B。
- CUDA 推理路径必须按 matcher 参数设备搬运 feature tensors；features 本身由 decode 保持 CPU 是合理的，但 matcher 在 CUDA 时 forward 输入也必须到 CUDA。

## Trainer split integration finding (2026-05-20)
- 自动训练/验证划分的基础能力已存在于 `modules/dataloader/sampler`，但 trainer 曾重复实现 shuffle/slice；现在 trainer 复用 `make_train_validation_test_split()`，避免模块化 dataloader 能力闲置和两套划分逻辑漂移。
- `train_ratio + val_ratio` 隐含剩余比例作为 test split；当前 trainer 只使用 train/validation，test 保留给后续评估扩展。

## Extreme rotation matching finding (2026-05-20)
- 原 mixed/extreme augmentation 最大旋转约 ±55°，不足以训练 180° 旋转匹配；对 half-turn 图像对，模型倾向输出非交叉/同位置匹配是训练分布外行为。
- Graph matcher 的 keypoint projection 若使用原始像素/feature-map 坐标，会引入绝对位置偏置；对 180° 场景，绝对坐标相近反而通常是错误匹配。归一化坐标可减少尺寸尺度支配，但仍需重新训练来学习 half-turn 对应关系。
- 现有 eval half-turn 指标曾基于 `MatchSet.points_a/points_b`，但这些字段来自 semi-dense points，不是 graph matcher 的 `sparse_matches`；验证 sparse 关键点匹配必须结合 `FeatureSet.keypoints` 和 `MatchSet.sparse_matches`。
- 单图旋转 sweep 暴露更基础的问题：`train_full.pt` 在 `img/100.tif` 的 0° 自匹配上 sparse graph matcher 也不几何一致（60 条 sparse matches，3px 通过率 0，mean_error≈172.77 feature-map px）。因此当前问题不是只缺 180° 增强，而是 sparse matcher 推理/训练目标没有学到可靠的 identity/known-transform 对应。
- 30° step sweep 结果：0/30/60/90/150/180/210/240/270/300/330 度 3px 通过率均为 0；120 度仅 1/98 条通过。下一步应先修复 identity/known-rotation sparse matching，再做长训练。
- 训练 graph matcher 时裁剪 B 候选子集会改变 keypoint normalization 的上下文，和推理时完整 B set 不一致；改成 full B candidate 后，单图 0° 自匹配可达到 1024/1024 正确。
- 仅修 full B candidate、加入 ±90° augmentation、让 graph loss 回传到 sparse descriptor map，仍不足以让单图 90/180/270° 输出 sparse matches。当前证据指向更深的旋转不变描述子/关键点一致性问题，而不是单个 half-turn augmentation 缺口。
- descriptor 层诊断显示旋转后并非完全没有可重复关键点：`train_rotation100_graphgrad.pt` 在 90/180/270° 的 3px keypoint repeatability 约 54%-61%；但 raw descriptor mutual nearest 几何正确率只有约 0.8%-2.6%，且错误 mutual 的均值 cosine 约 0.976-0.978，高于真实几何重复点的均值约 0.86-0.88。这说明描述子在旋转后强混淆/近似塌缩。
- 网格 descriptor 全局对比和 sparse keypoint hard-negative CE 能让训练指标收敛，但单图 40 epoch 后 rotation sweep 仍失败；`train_rotation100_cfgdecode40.pt` 的 90/180/270° sparse matches 均为 0，descriptor mutual 几何正确率仍只有约 0.7%-3.7%。
- trainer 的 sparse decode 训练路径曾硬编码 `TrainConfig{max_keypoints=1024,min_keypoints=0,nms_radius=4}`，导致用户传入 `--min-keypoints 1024` 时 graph/sparse descriptor 训练看到的 keypoint 集合与推理不一致；已改为复用完整用户 `TrainConfig`。
- 当前更可信的根因：普通 CNN descriptor head 加旋转增强没有形成稳定旋转不变描述子；orientation/affine 头虽然存在，但未被监督也未参与 descriptor canonicalization/matching。下一步应把 orientation/affine 变为实用路径，而不是继续堆 descriptor CE。
- C4 descriptor pooling/statistics/harmonic bands 证明了“让真实重复点更相似”仍不够：`train_rotation100_harmonic.pt` 将 90/180/270° 真实重复点 descriptor cosine 提升到约 0.954-0.965，但错误 mutual nearest 的均值仍约 0.993，最终 sparse graph/fallback matching 仍全部拒配。
- 因此当前优化方向应从“更多旋转不变汇聚”转为“压低错误互近邻”：例如对 rotation sweep 生成的真实对应点做跨图 batch-hard margin/InfoNCE，或让 matcher 在多旋转/方向候选上显式比较并用几何一致性训练；仅靠单图内 descriptor diversity 指标不足以约束跨图旋转混淆。
- 扩大 sparse descriptor hard-negative 覆盖到 1024 queries 并把 margin weight 提到 5 后，`train_rotation100_margin5.pt` 仍无法匹配 90/180/270°：sparse matches 全为 0，descriptor mutual correct rate 约 0.6%-1.7%，错误 mutual score 仍约 0.990。这说明当前 invariant descriptor 空间里的 false nearest 问题不是简单加权可解。
- 接下来更值得做的结构性改动：不要过早把四个 C4 方向压成单个不变向量；应保留方向相关 descriptor 分量，让 matcher/descriptor similarity 在 4 个 cyclic shifts 上取最大或受监督选择正确方向。这样可以避免 pooling/harmonic magnitude 把不同地点的旋转纹理压到几乎相同的向量。
- 已验证上述 C4 cyclic slots + cyclic similarity 路线仍失败：训练收敛且 0° 自匹配正常，但 90/180/270° sparse matches 仍为 0，错误 mutual score 仍约 0.991。说明“方向槽位 + shift-invariant 比较”没有自动解决位置区分度，false nearest 仍是主瓶颈。
- 后续更可信的优化点应进入监督构造本身：用已知旋转生成的几何真值直接构造 all-keypoint/all-spatial batch-hard negatives，并约束 true positive score 必须高于同图/跨图最难负样本；或者显式监督 orientation，让 descriptor 在 canonical frame 中提取，而不是让网络从 CE 间接学会旋转规范化。
- keypoint-to-full-map descriptor 监督仍未解决极端旋转。重建 `pfm_cli` 后的可信结果是：`train_rotation100_keydense.pt` 的 90/180/270° sparse matches 分别为 92/108/87，但几何通过率只有约 1.1%/2.8%/2.3%，mean error 约 150-164 feature-map px。这说明“把 sparse keypoint descriptor 对齐到完整 B descriptor-map 真值位置”仍不足以压低跨旋转 false nearest。
- `train_rotation100_keydense.pt` 还让 repeatability 从 dense-hard 版本的约 55%-59% 降到约 40%-47%，提示当前强 descriptor 约束可能在和关键点稳定性/热力图目标竞争。后续若继续这条线，应同时加强 keypoint repeatability 几何监督，而不是只加 descriptor loss。
- rotation sweep 评估必须确保 `pfm_cli` 目标也被重建；仅构建 `pfm_tests` 和 `pfm_rotation_sweep_eval` 会留下旧 CLI 二进制，从而把已实现的 descriptor fallback/cyclic matching 误判为 0 sparse matches。
- orientation-supervised canonical descriptor 不是银弹：`train_rotation100_orientcanon.pt` 的 90/180/270° 几何通过率只有约 5.3%/3.1%/5.3%。它改善了 repeatability（约 53%-64%）并略微降低 false mutual score（约 0.977-0.983），但训练 feature loss 停在约 4.74，说明 canonicalization/orientation 监督和现有 descriptor CE 尚未协调好。
- 用户指出 180° 可视化仍接近平行是正确诊断：模型输出没有学到 half-turn 几何。训练数据的 warp 本身不是反的，但旧 mixed half-turn anchor 被 scale/translation 污染，且占比只有 1/8；这会让 descriptor mutual fallback 更容易保留近坐标/相似纹理匹配，而不是学中心对称 X 形对应。
- 改进后的 mixed profile 对 `variant % 8 == 3/7` 使用干净 ±90°/±180° 几何 anchor，避免极端旋转监督同时混入平移、缩放和强 photometric 变化。后续需要基于这个数据修复重新训练，旧 `train_rotation100_orientcanon.pt` 不代表修复后的训练分布。
- clean anchor 重新训练后仍失败，说明问题不只是 half-turn 样本被 scale/translation 污染：`train_rotation100_cleananchors.pt` 在 180° 上 pass rate≈2.42%，可视化仍是平行线。
- 单图训练原先每个 epoch 重复同一组 `pairs_per_image` variants，等价于在一个图上反复看同 8 个变换；这会高估训练进展且限制旋转/纹理覆盖。现在 variant index 随 epoch 前进。
- Graph matcher 使用 normalized x/y keypoint embedding 时，会给 180° 任务留下“同屏幕位置”捷径。改成 radius/radius^2 后能去掉明显绝对方向偏置，但 `train_rotation100_epochvariants_radialmatcher.pt` 的 180° pass rate 仍只有≈3.64%，说明主要瓶颈仍是 descriptor/局部纹理混淆。
- 当前最可信判断：继续盲训现有 CNN descriptor/head 收益很低。下一步应先在更快机器上用 SIFT/ORB 建立 180° baseline；如果传统局部描述子能给出 X 形匹配，就应把 learned feature extractor 改成真正的 orientation-normalized local patch descriptor 或 rotation-sweep hard-negative 监督。
