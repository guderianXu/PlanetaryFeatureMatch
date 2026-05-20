# 进度日志

## 2026-05-18
- 用户确认新增 `--min-keypoints`。
- 已完成 `--min-keypoints` TDD、实现、README 文档和相关模块验证；全量测试一度被无关 AsyncDataLoader 段错误阻塞。
- 切回继续“module 以及匹配 loss 优化”后，读取 reusable training infrastructure 设计/计划，确认下一步先稳定 AsyncDataLoader。
- 复现全量 `./build/pfm_tests` 段错误：退出码 139，崩溃测试为 `asyncDataLoaderFailedAsyncResetLeavesLoaderExhausted()`。
- 根因：`AsyncDataLoader::reset()` 中 sampler 抛异常后 `_queue` 被清空但 `_exhausted` 仍为 false，导致后续 `next()` 解引用空队列。
- 最小修复：`reset()` 先清空索引并置 `_exhausted=true`，sampler/async epoch 初始化成功后再置 false。
- 验证：`cmake --build build -j$(nproc) && ./build/pfm_tests` 通过，输出 `273 test(s) passed`。
- 继续实现 training logging module：先添加 logging RED 测试并接入 `pfm_logging` CMake target，RED 为缺少 `modules/logging/csv_metric_logger.cpp`。
- 最小实现 `TrainingMetric`、`ConsoleProgressLogger`、`CsvMetricLogger`、`NullGpuMetricProvider`、`MetricLoggerGroup` 后，验证 `cmake --build build -j$(nproc) && ./build/pfm_tests` 通过，输出 `277 test(s) passed`。
- 完成 optional NVML hook：`PFM_ENABLE_NVML` ON/OFF 均可构建，默认 GPU provider 在无 NVML 时回退为空 provider。
- 完成 augmentation module 抽取：`modules/augment/*` 接管 synthetic pair transform/augment 逻辑，旧 `make_synthetic_pair()` 保持兼容包装；修复 OBJECT target 链接和 Torch include 问题后，验证输出 `280 test(s) passed`。
- 完成 `SyntheticPairTensorDataset`：先用缺失源文件构建失败作为 RED，再实现 dataset adapter；验证 `cmake --build build -j$(nproc) && ./build/pfm_tests` 通过，输出 `281 test(s) passed`。
- 完成 trainer logging 集成：先让 `--log-csv` 和 trainer CSV 输出测试失败，再接入 `CsvMetricLogger` 与 default GPU metric provider；验证输出 `282 test(s) passed`。
- 完成 trainer AsyncDataLoader 集成：新增 `dataloader_workers/prefetch_batches/pin_memory` 配置、CLI 映射和 online synthetic pair loader 路径；验证输出 `284 test(s) passed`。
- 更新 README、docs/training.md、docs/usage.md，补充 `--log-csv`、DataLoader worker/prefetch/pin-memory、CSV/GPU 指标和在线/缓存训练路径说明。
- 完成模块边界整理：modules/ 只保留 runtime、logging、dataloader、core 通用基础设施；项目特有模块全部迁移到 src/（augment、cli、data、eval、geometry、infer、losses、models、train）。synthetic_pair_dataset 从 dataloader 移至 src/data/。验证输出 284 test(s) passed，CTest 100% tests passed。

## 2026-05-19 keypoint graph matching loss 优化
- 完成 spec 与实现计划：`docs/superpowers/specs/2026-05-19-keypoint-graph-matching-loss-design.md`、`docs/superpowers/plans/2026-05-19-keypoint-graph-matching-loss.md`。
- Task 1：新增 graph matcher keypoints affect logits 回归测试，并让 `_keypoint_projection` 参与 descriptor embedding；最终保留原始 feature-map keypoint 坐标，不做错误 clamp/归一化。
- Task 2：新增 `assign_graph_matching_targets`，用 image-space warp + valid mask 将 A keypoint 分配到最近 B keypoint；无匹配、source invalid、target invalid 均落入 dustbin。
- Task 3：新增 deterministic graph candidate construction，确保 positives 只出现一次，dustbin 固定在最后。
- Task 4：训练 graph loss 从固定 descriptor grid 改为 decoded sparse keypoints；candidate set 包含 positives/negatives/dustbin；target assignment 前将 feature-map keypoints 缩放到 image-space，matcher 仍使用 inference 同款 feature-map keypoints。
- 验证：重新配置并构建 `build-pfm-cf`，`pfm_tests` 输出 `290 test(s) passed`，`ctest` 输出 `100% tests passed, 0 tests failed out of 1`。
- CUDA 短训练复现已完成，命令包含 `--device cuda`；`nvidia-smi` 曾确认 `pfm_cli` 进程使用 GPU，显存约 2264 MiB。完整 4140 条 CSV iteration 的 `graph_matching_loss`：first_mean≈4.6967，last_mean≈0.0335，min≈0.00000227，max≈18.1822，最后 10 条约 0.0010-0.0020。

- 阶段 13 标记完成：graph matcher 使用 keypoint geometry，训练监督已切换到 decoded sparse keypoints；完整 CUDA 验证显示 graph loss 从 first_mean≈4.6967 降到 last_mean≈0.0335。下一步执行合并到 main 并推送 GitHub。

## 2026-05-20 full CUDA training
- 在 main 分支运行完整 10 epoch CUDA 训练，命令使用用户原始参数：image-dir=img、batch-size=2、resize=512、pairs-per-image=15、augmentation-profile=mixed、min-keypoint-intensity=0.05、min-keypoints=1024、device=cuda。
- 输出文件：`train_full.pt`（25M checkpoint）、`metrics_full.csv`（2.2M CSV）、`vis_full/`。这些是训练产物，未纳入 git 跟踪。
- 训练完成，exit code 0；CSV 有 13,800 条 iteration。
- 指标趋势：graph_matching_loss first_mean≈2.17615、last_mean≈0.10262；feature_loss first_mean≈2.78217、last_mean≈0.333686；dense_loss first_mean≈0.754952、last_mean≈0.0857591；offset_error_px first_mean≈156.817、last_mean≈16.3463；descriptor_accuracy first_mean≈0.374187、last_mean≈0.95367。
- 最后一条 iteration：graph_matching_loss≈0.000501843，feature_loss≈0.000998608，repeatability_loss≈0.000040554，dense_loss≈0.00360692，offset_error_px≈10.891，descriptor_accuracy=1。

## 2026-05-20 checkpoint inference evaluation
- 用 `train_full.pt` 跑真实图像 CUDA match 评估时发现推理路径问题：`pfm_cli match --device cuda` 最初因 graph matcher 在 CUDA、features 在 CPU 报 device mismatch。已修复 `matchSparseFeatures()`，按 matcher 参数所在设备搬运 sparse descriptors/keypoints，并新增 CUDA 回归测试。
- 评估还发现 graph matcher 训练有 dustbin，但推理原来仍对每个 A keypoint 强制输出一个 B match，真实图像上导致 1024 条 sparse match 全部输出且误匹配风险高。已修复为：推理只保留非 dustbin 且 B→A 互为最近的 sparse matches，并新增 dustbin/mutual tests。
- 修复后 `pfm_tests` 通过 `293 test(s) passed`。
- 真实图像示例：100-101 / 100-110 / 100-118 在 CUDA match 下 sparse_matches 从原来的 1024 分别降到 44 / 67 / 43；dense matches 仍为 74332 / 63506 / 74486。输出目录：`eval_matches_full_mutual/`。

## 2026-05-20 trainer dataset split module integration
- 检查发现 `modules/dataloader/sampler.{h,cpp}` 已有 `make_train_validation_test_split()`，但 trainer 仍在本地手写 `image_order` shuffle 和 train/val 切片。
- 已将 trainer 改为通过 `make_training_dataset_split()` 复用模块化 split 工具，训练和验证索引分别来自 `DatasetSplit::train` / `DatasetSplit::validation`；online dataloader、同步生成和 validation loop 共用同一 split 结果。
- 保留小数据集保护：当 total_images>0 但 split.train 为空时，从 validation/test 移一个样本到 train，避免训练集为空。
- 新增测试 `trainer_training_and_validation_indices_use_dataloader_split`，验证 trainer 的 train/validation indices 与 `make_train_validation_test_split()` 一致。
- 验证：`pfm_tests` 294 tests passed，`ctest` 100% passed。

## 2026-05-20 extreme rotation generalization pass
- 用户指出评估图像对存在 180° 极端旋转/视角差，正确匹配线应整体交叉；此前训练增强最大旋转约 ±55°，没有覆盖 half-turn。
- 已将 mixed augmentation 每 8 个 variant 注入一个 deterministic ±180° half-turn case，并新增 `transform sampler mixed includes half turn variants` 测试。
- 已将 graph matcher 内部 keypoint projection 输入归一化到每组 keypoints 的 [-1,1] 范围，避免像素绝对值/图像尺寸支配 descriptor matching；新增 keypoint logits scale-invariant 测试。
- 验证：`pfm_tests` 296 tests passed，`ctest` 100% passed。下一步需要重新训练 checkpoint 并重新评估 180° 图像对。
- 继续前先按用户要求做单图 0-360° 旋转测试；中止了正在跑的 3 epoch rot180 训练，避免在没有可靠评估判据时继续耗时。
- 新增 `tools/rotation_sweep_eval.cpp` 和 `pfm_rotation_sweep_eval` CMake target：生成同一图像的旋转版本，调用 CLI extract/match，读取 `FeatureSet.keypoints + MatchSet.sparse_matches` 计算 sparse 匹配到理论旋转位置的误差。
- 验证：`cmake -S . -B build-pfm-cf && cmake --build build-pfm-cf -j$(nproc) --target pfm_rotation_sweep_eval` 通过。
- 对 `img/100.tif` + `train_full.pt` 跑 30° step CUDA sweep：`rotation_sweep_100_step30/summary.csv`。结果显示 0° 自匹配也失败：60 条 sparse matches，3px 通过率 0，mean_error≈172.77 feature-map px；整圈除 120° 有 1/98 条通过外，其余角度通过率均为 0。
- 修复 graph loss 训练/推理分布差：keypoint graph loss 不再裁剪 B 候选子集，而是对完整 B keypoint set + dustbin 做 CE；新增测试确认第 69 个全局负候选也收到梯度。
- mixed augmentation 增加 deterministic ±90° quarter-turn 样本；新增测试确认 mixed 覆盖 quarter-turn。
- graph loss 选中的 sparse descriptors 改为从原始 descriptor map gather，保留到 sparse descriptor head 的梯度；验证 `pfm_tests` 302 tests passed。
- 单图 `img/100.tif` 过拟合实验：full B candidate 修复后 `0°` 自匹配从完全失败变为 1024/1024 正确；但 90/180/270° 仍输出 0 sparse matches。加入 ±90° 与 descriptor graph 梯度后，`rotation_sweep_100_graphgrad_step90/summary.csv` 仍显示 0°=1024/1024，90/180/270=0 matches。
- 尝试 graph 输出为空时 fallback 到 descriptor mutual nearest；`rotation_sweep_100_graphgrad_fallback_step90/summary.csv` 仍显示 90/180/270=0 sparse matches，说明当前 sparse descriptors 对大旋转仍不可匹配或存在全局拒配问题。
- 扩展 `pfm_rotation_sweep_eval`，新增 descriptor mutual nearest、descriptor finite rows、keypoint repeatability、repeatable descriptor score 等诊断列。
- 诊断 `train_rotation100_graphgrad.pt`：90/180/270° 的 repeatability 约 0.54-0.61，几何重复点 descriptor cosine 均值约 0.86-0.88；但 descriptor mutual nearest 几何正确率仅约 0.8%-2.6%，错误 mutual 的均值相似度约 0.976-0.978，说明 sparse descriptor 区分度不足而不是完全无重复点。
- 从特征提取器训练入手：descriptor loss 增加全局采样点对比 CE；decoded sparse keypoint descriptor 增加完整 B keypoint hard-negative CE；训练 decode 配置不再硬编码 `min_keypoints=0`，改为复用用户 `TrainConfig`。
- 验证：`cmake --build build-pfm-cf -j$(nproc) --target pfm_tests pfm_rotation_sweep_eval && ./build-pfm-cf/pfm_tests` 曾通过 304 tests passed；修复训练 decode 配置后 `pfm_tests` 仍为 304 tests passed。
- 单图 `img/100.tif` 训练/评估记录：
  - `train_rotation100_globaldesc.pt` + `rotation_sweep_100_globaldesc_step90/summary.csv`：0° 正确，90/180/270° 仍 0 sparse matches。
  - `train_rotation100_sparsekeydesc.pt` + `rotation_sweep_100_sparsekeydesc_step90/summary.csv`：0° 正确，180° 仅 1 条且错误，90/270° 仍 0。
  - `train_rotation100_cfgdecode40.pt` + `rotation_sweep_100_cfgdecode40_step90/summary.csv`：0° 1024/1024 正确；90/180/270° 仍 0 sparse matches，descriptor mutual 几何正确率约 0.7%-3.7%。
- 当前阶段结论：仅靠增强、全局 descriptor CE、sparse keypoint descriptor hard negatives 和训练/推理 keypoint 数量对齐，仍不能获得 0-360° 稳定匹配；下一步需要显式旋转等变/规范化的特征提取器设计（例如 orientation-supervised canonical descriptor 或旋转分组 descriptor pooling）。
- 特征提取器继续优化：`SparseHead` descriptor 分支改为 C4 旋转视图描述子，先尝试四方向均值池化，再尝试均值+方差统计投影，最后改为固定 C4 harmonic descriptor bands（DC、half-turn、quarter-turn magnitude），输出维度保持 `descriptor_dim`。
- 验证：`cmake --build build-pfm-cf -j$(nproc) --target pfm_tests pfm_rotation_sweep_eval && ./build-pfm-cf/pfm_tests` 通过，输出 `306 test(s) passed`。
- 单图 `img/100.tif` 训练/评估新增记录：
  - `train_rotation100_c4desc.pt` + `rotation_sweep_100_c4desc_step90/summary.csv`：0° 1024/1024 正确；90/180/270° sparse matches 仍为 0，repeatable descriptor score 约 0.92-0.94，错误 mutual score 约 0.989-0.993。
  - `train_rotation100_rotstats.pt` + `rotation_sweep_100_rotstats_step90/summary.csv`：0° 正确；90/180/270° sparse matches 仍为 0，descriptor mutual 几何正确率约 0%-1.4%。
  - `train_rotation100_harmonic.pt` + `rotation_sweep_100_harmonic_step90/summary.csv`：0° 正确；90/180/270° sparse matches 仍为 0；真实重复点 descriptor score 提升到约 0.954-0.965，但错误 mutual score 仍约 0.993。
- 当前结论：只在 `SparseHead` 内做 C4 pooling/statistics/harmonic invariant 还不足以解决 90/180/270° sparse matching；瓶颈已变为“错误互近邻相似度仍高于真实重复点”，下一步应引入更强的跨图 hard-negative/margin 约束，或显式 orientation-supervised canonical descriptor / rotation-aware matching。
- 继续增强 sparse keypoint descriptor hard-negative：descriptor loss query 覆盖从 graph matcher 的 256 扩到 1024，并加入 hardest-negative margin；margin weight=1 后错误 mutual score 有下降但 90/180/270° 仍失败，weight=5 后 `rotation_sweep_100_margin5_step90/summary.csv` 仍显示 90/180/270° sparse matches 均为 0。
- 最新 margin=5 诊断：0° 为 1024/1024 正确；90/180/270° descriptor mutual 几何正确率仅约 0.6%-1.7%，错误 mutual score 仍约 0.990，高于真实 repeatable descriptor score 约 0.946-0.957。
- 当前结论更新：继续加大当前 invariant descriptor 的 hard-negative 权重不能解决 0-360° 匹配；下一步应转向结构性方案，如保留方向分量并做 rotation-aware descriptor matching，或对 orientation/canonicalization 做显式监督。
- 实现并验证 rotation-aware cyclic descriptor 路线：`SparseHead` 改为保留 4 个 C4 方向槽位，descriptor CE / candidate CE / sparse keypoint margin / descriptor fallback matching 都允许 4-way cyclic shift 最大相似度；新增 cyclic descriptor loss 测试，`pfm_tests` 通过 309 tests passed。
- 单图 `img/100.tif` 训练/评估：`train_rotation100_cyclic.pt` + `rotation_sweep_100_cyclic_step90/summary.csv`。结果仍为 0° 1024/1024 正确，90/180/270° sparse matches 均为 0；descriptor mutual 几何正确率约 1.3%-2.1%，真实 repeatable descriptor score 约 0.919-0.936，错误 mutual score 仍约 0.991。
- 当前结论更新：C4 invariant、C4 cyclic slots、hard-negative margin 都未能让单图 90/180/270° 匹配成立；下一步需要更强的监督信号，例如直接以 rotation sweep 真实对应点做 dense/keypoint batch-hard loss，或者启用 orientation head 做显式 canonical patch/descriptor，而不是只改 descriptor aggregation。
- 继续实现 keypoint-to-full-map descriptor 监督：将 decoded A keypoint descriptor 直接拉向 warp 后的 B descriptor-map 真实位置，并对整张 B descriptor map 的最难错误位置做 margin；新增 dense/full-map 相关单测，`pfm_tests` 通过 311 tests passed。
- 单图 `img/100.tif` 训练/评估：`train_rotation100_keydense.pt`。最初 `rotation_sweep_100_keydense_step90/summary.csv` 显示 90/180° sparse matches 为 0，排查发现 `pfm_cli` 未随最近 fallback/cyclic matching 改动重建。
- 重建 `pfm_cli` 后重跑：`rotation_sweep_100_keydense_rebuilt_step90/summary.csv`。0° 为 1024/1024 正确；90/180/270° 分别输出 92/108/87 条 sparse matches，但几何通过率仅约 1.1%/2.8%/2.3%，mean error 约 150-164 feature-map px。
- 当前结论更新：0 matches 主要是旧 CLI 二进制导致的评估假象；真实瓶颈仍是 descriptor/keypoint 旋转鲁棒性差。keypoint-to-full-map batch-hard 监督没有实质提升 90/180/270° 匹配，且 keypoint repeatability 下降到约 40%-47%；下一步转向 orientation-supervised canonical descriptor 或直接改关键点 repeatability 监督。
- 实现 orientation-supervised canonical descriptor：`SparseHead` 用 orientation head 对 C4 cyclic descriptor slots 做 soft canonicalization；trainer 新增从 warp 估计旋转方向的 orientation loss，并把 `orientation_loss` 写入 CSV。验证 `pfm_tests` 312 tests passed。
- 单图 `img/100.tif` 训练/评估：`train_rotation100_orientcanon.pt` + `rotation_sweep_100_orientcanon_step90/summary.csv`。训练最终 feature loss 仍约 4.74，不低；0° 为 1024/1024 正确，90/180/270° 分别输出 113/129/95 条 sparse matches，几何通过率约 5.3%/3.1%/5.3%，mean error 约 166/138/128 feature-map px。
- 当前结论更新：orientation-canonical 版本相比 keydense 有小幅改善（90/270 通过率从约 1%-2% 提到约 5%），repeatability 也回升到约 53%-64%；但整体匹配效果仍明显不可用，不能称为收敛模型。
- 排查用户指出的 180° 匹配线仍接近平行问题：实际旧 mixed half-turn 样本几何方向是交叉的，但 `variant=7` 同时叠加了 scale≈0.912、tx=-9、ty=-8，并且每 8 个 pair 只有 1 个 half-turn；训练分布对纯 180° X 形匹配监督过弱。
- 已将 mixed 中 deterministic ±90°/±180° 样本改为干净旋转 anchor，不再叠加随机平移、缩放、gamma、shadow；新增单测验证 mixed half-turn/quarter-turn anchor 和 half-turn warp 交叉映射。验证 `pfm_tests` 315 tests passed。
- 新增训练数据检查输出：`runs/debug_halfturn_training_data_clean/pure_180_ground_truth_x.png` 展示纯 180° ground-truth X 形匹配；`runs/debug_halfturn_training_data_clean/vis/static/pair_000007_warp_matches.png` 是改后训练 variant 7 的静态 warp 可视化。
- 用 clean anchors 重新训练 `train_rotation100_cleananchors.pt` 后，0° 仍为 1024/1024 正确，但 180° 只有 124 matches、3px pass rate≈2.42%、mean_error≈142.63；可视化仍基本是平行线。
- 进一步修正训练 variant 调度：单图非 cache 训练现在随 epoch 推进 variant，避免 60 epoch 重复同 8 个 pair；graph matcher keypoint embedding 改为旋转不敏感的 radius/radius^2，减少绝对 x/y 坐标捷径。验证 `pfm_tests` 317 tests passed。
- 新训练 `train_rotation100_epochvariants_radialmatcher.pt`（60 epoch）最终 loss≈4.367，不低；90/180/270° sweep 分别为 114/165/137 matches，3px pass rate≈1.75%/3.64%/1.46%，180° mean_error≈133.32。180° 可视化仍主要是平行线，不是 X 形。
- 已新增交接文档 `docs/rotation_matching_handoff.md`，记录当前代码状态、实验结果、关键路径和换机器后的建议下一步。
