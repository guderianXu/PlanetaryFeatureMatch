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
