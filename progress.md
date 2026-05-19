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
- CUDA 短训练复现已启动，命令包含 `--device cuda`；`nvidia-smi` 确认 `pfm_cli` 进程使用 GPU，显存约 2264 MiB。训练仍在运行；当前 727 条 CSV iteration 的 `graph_matching_loss`：first_mean≈4.6967，current last_mean≈0.3339，min≈0.00000227，max≈18.1822。
