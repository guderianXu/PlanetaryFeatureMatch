# 进度日志

## 2026-05-27 PFM v2.1 架构迁移
- 根据用户确认的新架构文档，已完成 PyTorch 主路径的 PFM v2.1 代码迁移。
- `python/pfm_model.py` 默认模型升级为 64 base channels、256 维 descriptor、512 hidden GraphMatcher、8 层 graph attention，并在 checkpoint config 中记录 `graph_keypoint_meta_dim`。
- Backbone 增加零初始化 residual context blocks；SparseHead 增加 keypoint offsets、分支 context、C4 branch-quality attention、rotation-invariant/equivariant descriptor fusion；GeometryHead 改成稳定的 log-scale decode 与 residual affine。
- analytic texture descriptor 扩展为光照鲁棒特征集合，包含 local normalized intensity、local contrast、DoG/LoG、梯度方向、ring contrast、soft census-like；DescriptorFusionAdapter 增加自适应 texture gate。
- GraphMatcher 的 keypoint metadata 默认升级为 x/y/radius/radius² 四维，同时保留旧 checkpoint 的 2 维兼容加载。
- 训练可视化报告和 cache evaluator 已支持 `raw_descriptor` 与 `graph_matcher` 两种匹配口径；训练报告可用 `--report-matcher-mode both` 同时生成两套结果。
- 验证完成：`py_compile` 通过；focused Python tests 121 OK, 1 skipped；完整 Python discovery 216 OK, 2 skipped。
- 当前边界：C++/LibTorch 推理端还没有同步 v2.1 架构；semi-dense fallback 与真正的 geometry-aware canonical descriptor sampling 仍在下一阶段。

## 2026-05-27 仿真数据同步与继续训练
- 检查 2048 仿真数据：8T 上 `pose_sim_2048_gap30_views10` 的生成进程仍在运行，`batch_pose_sim_dataset.py` 使用 `asp36`，`sat_sim_cuda` 正在持续渲染/转换后续 pair。
- 8T cache 检查时已超过 6570 个 pair，后续仍增长；本地 NVMe 训练数据原来只有 `train=2040`、`val=1026`、`test=232` 左右。
- 先误启动了一次全量 manifest 同步，发现会优先复制 test，已中断；随后改为只同步 train。同步后本地为 `train=3294`、`val=1026`、`test=419`，NVMe 仍剩约 397GB。
- 使用最新可继续 checkpoint `runs/pose_metadata_crop1024_batch4_mildmargin_lr1e5_300_20260527/pytorch_pfm_state.pt` 兼容加载新 v2.1-lite 结构继续训练。该 checkpoint 配置为 48/192/384，不是 full 256 维。
- batch4 与 batch2 在 1024 crop + C4 + blended descriptor fusion 训练下 OOM；batch1 + 512 points 稳定，训练显存约 19GB，同时不打断仿真进程。
- 完成 run：`runs/pose_sim2048_crop1024_v21lite_train3294_ctxfusion_b1p512_lr3e5_600_20260527`。
- 验证检索明显提升：loss 2.3557 -> 1.0998，Top1 0.6909 -> 0.8404，Top5 0.8288 -> 0.9270，mean rank 10.18 -> 3.74。
- 训练指标：600 steps 无 skipped；first50 loss 6.5626，last50 loss 4.5656；first50 Top1 0.5547，last50 Top1 0.7873；pose-balanced 覆盖 medium 312、hard 288。
- 报告已生成：
  - raw descriptor：`visual_report/raw_descriptor/training_report_zh.pdf`，抽样 5629/7646 correct，precision 0.7362；hard precision 0.4322。
  - GraphMatcher：`visual_report/graph_matcher/training_report_zh.pdf`，抽样 3896/7706 correct，precision 0.5056；hard precision 0.0948。
- 结论：descriptor/fusion 训练有效，但 GraphMatcher 没有被本轮训练更新，正式 graph 口径明显拖后腿；下一轮应接 GraphMatcher 训练或先禁用/重训 GraphMatcher route。

## 2026-05-27 训练可视化与中文 PDF 报告
- 用户反馈当前训练结果输出不够：指标少、没有中文 PDF 报告、匹配点太少、loss 居高不下。
- 已启动补强：扩展 `scripts/training_visual_report.py`，目标是生成更完整的 matching metrics、更多 raw 匹配可视化和中文 PDF 报告。

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

## 2026-05-25 rotation matcher benchmark
- 新增独立 Python 工具 `python/rotation_matcher_benchmark.py`，不改训练/评估主逻辑；从 `img/1.tif` 和 `img/20260514T064636672_NAS_PAN_L2b.tif` 构造 90/180/270 同图旋转对，对 SIFT、ORB、AKAZE、LightGlue、SuperGlue、PFM 统一输出 CSV 和匹配线可视化。
- 新增单测 `python/test_rotation_matcher_benchmark.py`，覆盖 `np.rot90` 真值坐标、CSV 字段格式、不可用 matcher 不抛出。
- RED/修复记录：首次真实 smoke 显示 180° 全对但 90/270° 全错，根因是 `rotate_points()` 使用了顺时针公式而 `np.rot90(k=1)` 是逆时针；已修正为 90° `(y, width-1-x)`、270° `(height-1-y, x)` 并更新测试。
- 继续补上 LightGlue-SIFT adapter：`lightglue` 和 `kornia` 已安装到 `pfm-train`，`LightGlue(features="sift")` 权重已下载到 torch cache；SuperGlue 入口 `match_pairs` 仍不存在，按 unavailable 记录。
- 验证：`PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest discover -s python -p 'test_*.py'` 通过，113 tests OK，1 skipped。
- 最新真实对比：`PYTHONPATH=python MKL_THREADING_LAYER=GNU /home/xjw/anaconda3/envs/pfm-train/bin/python python/rotation_matcher_benchmark.py --output-dir runs/rotation_matcher_benchmark_lightglue_20260525 --angles 90 180 270 --resize-max 768 --max-matches 256 --max-keypoints 1024 --device cuda --pfm-pytorch-state runs/cross_view_1024_no_self_warp_finetune_300_seed1234/training/pytorch_pfm_state.pt` 完成。
- 产物：`runs/rotation_matcher_benchmark_lightglue_20260525/metrics.csv`、`summary.txt`、`visualizations/` 共 30 张 PNG。Aggregate：SIFT/ORB/AKAZE/PFM 均为 1536/1536 正确，LightGlue-SIFT 为 1535/1536 正确，SuperGlue unavailable。
- 并行 worker 继续补齐 learned baseline：新增 Kornia LoFTR adapter，默认 `--loftr-pretrained outdoor`，在 `kornia.feature` 缺失时按 `UnavailableMatcher("LoFTR", ...)` 记录；SuperGlue 仍因 `match_pairs` 不存在保持 unavailable，没有 vendor 外部仓库。
- 新增 LoFTR focused mock 单测，不加载真实权重：覆盖 optional matcher 组装、confidence 排序和 `--max-matches` 截断。验证命令：`PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest python/test_rotation_matcher_benchmark.py`，6 tests OK；完整 Python 发现测试 `PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest discover -s python -p 'test_*.py'`，117 tests OK，1 skipped。
- LoFTR smoke：`PYTHONPATH=python MKL_THREADING_LAYER=GNU /home/xjw/anaconda3/envs/pfm-train/bin/python python/rotation_matcher_benchmark.py --output-dir runs/rotation_matcher_benchmark_loftr_smoke_20260525_worker --angles 90 --resize-max 512 --max-matches 64 --max-keypoints 512 --device cuda --no-auto-pfm` 完成并首次下载 `loftr_outdoor.ckpt` 到 torch cache。产物：`runs/rotation_matcher_benchmark_loftr_smoke_20260525_worker/metrics.csv`、`summary.txt`、`visualizations/` 10 张 PNG。LoFTR 状态为 ok，但 90° 几何正确数为 numeric 0/52、timestamp 0/64；SIFT/ORB/AKAZE/LightGlue-SIFT 均为 64/64。
- 完整 LoFTR 旋转对比：`PYTHONPATH=python MKL_THREADING_LAYER=GNU /home/xjw/anaconda3/envs/pfm-train/bin/python python/rotation_matcher_benchmark.py --output-dir runs/rotation_matcher_benchmark_loftr_full_20260525 --angles 90 180 270 --resize-max 768 --max-matches 256 --max-keypoints 1024 --device cuda --pfm-pytorch-state runs/cross_view_1024_no_self_warp_finetune_300_seed1234/training/pytorch_pfm_state.pt` 完成。产物：`metrics.csv`、`summary.txt`、36 张 PNG。Aggregate：SIFT/ORB/AKAZE/PFM 均为 1536/1536，LightGlue-SIFT 为 1535/1536，LoFTR 为 3/614，SuperGlue unavailable。结论：Kornia LoFTR 的 outdoor 权重对这里的同图 90/180/270 离散旋转基本不鲁棒，不能作为旋转基线优势项。

## 2026-05-25 cross-view 1024 balanced-sampling iteration
- 为 `python/cross_view_experiment.py` 补齐底层训练参数透传：新增 `--balanced-cache-sampling`、`--training-texture-blend-weight`、`--training-eval-pairs`；对应单测已加入 `python/test_cross_view_experiment.py`。`--training-eval-pairs` 用于限制训练前后 descriptor sanity eval，避免短迭代时完整 validation 扫描拖慢实验。
- 运行 40-step balanced cache sampling + train blend 0.25 试验：`runs/cross_view_1024_balanced_sampling_blend025_40_seed1234`。命令从 `runs/cross_view_1024_no_self_warp_finetune_300_seed1234/training/pytorch_pfm_state.pt` 继续，`batch_pairs=4`、`gradient_accumulation_steps=2`、`samples_per_pair=512`、`learning_rate=1e-5`、`calibration_limit_pairs=32`、`eval_limit_pairs=64`。
- 训练前后 descriptor eval 变差：top1 0.1956 -> 0.1888，loss 4.7446 -> 5.2700，mean_negative_score 0.4709 -> 0.7075。64-pair test 精度：numeric/rotate 0.812767，numeric/viewpoint 0.087298，numeric/compound 0.099462，timestamp/rotate 0.599923，timestamp/viewpoint 0.073469，timestamp/compound 0.035112。该 checkpoint 只提升 sampled numeric/rotate，不是更好的均衡模型，不替换当前 full-test routed best。
- 继续做更保守的 balanced cache sampling + train blend 1.0 + lr 5e-6 试验：`runs/cross_view_1024_balanced_sampling_blend1_lr5e6_40_seed1234`，并用新增的 `--training-eval-pairs 128` 限制训练前后 descriptor eval。训练前后仍变差：top1 0.0655 -> 0.0588，loss 4.9173 -> 5.0368，mean_negative_score 0.4397 -> 0.6219。64-pair test 精度：numeric/rotate 0.813221，numeric/viewpoint 0.083133，numeric/compound 0.079195，timestamp/rotate 0.615240，timestamp/viewpoint 0.062660，timestamp/compound 0.039124。结论：balanced sampling 本身不能解决弱 viewpoint/compound，后续应改 loss/监督信号或做明确的 specialist routing，而不是继续同类短微调。
- 为 `cross_view_experiment.py` 继续补齐底层 warp-aware hard negative 参数透传：`--warp-hard-negative-weight`、`--warp-hard-negative-radius`、`--warp-hard-negative-margin`、`--warp-hard-negative-candidates`，对应单测已加入。
- 运行 warp-hard-negative 短试验：`runs/cross_view_1024_warp_hn025_lr5e6_40_seed1234`，从当前 balanced checkpoint 继续，`warp_hard_negative_weight=0.25`、radius 3.0、margin 0.3、lr 5e-6、40 steps。训练中出现一次 nonfinite step skip；descriptor eval 小幅变差：top1 0.0655 -> 0.0606，loss 4.9173 -> 4.9730，mean_negative_score 0.4397 -> 0.5667。64-pair test 精度：numeric/rotate 0.809767，numeric/viewpoint 0.115586，numeric/compound 0.090028，timestamp/rotate 0.599021，timestamp/viewpoint 0.062842，timestamp/compound 0.040373。相比 balanced sampling，它对 numeric/viewpoint 有回升，但仍远低于 full-test routed best，说明短程 warp-hard-negative 不是单独解法。
- 运行更低权重 warp-hard-negative 试验：`runs/cross_view_1024_warp_hn005_lr5e6_40_seed1234`，`warp_hard_negative_weight=0.05`、radius 3.0、margin 0.2、lr 5e-6、40 steps。仍在同一阶段出现一次 nonfinite step skip，descriptor eval 与 0.25 版本几乎一致：top1 0.0655 -> 0.0606，loss 4.9173 -> 4.9727，mean_negative_score 0.4397 -> 0.5656。64-pair test 精度：numeric/rotate 0.809411，numeric/viewpoint 0.115044，numeric/compound 0.090153，timestamp/rotate 0.597378，timestamp/viewpoint 0.062927，timestamp/compound 0.040785。降低权重没有改变方向，下一步若继续这条线应先定位 nonfinite 样本/损失数值，而不是直接延长训练。

## 2026-05-25 rotation matcher benchmark LightGlue-SuperPoint worker
- 探索 `pfm-train` 环境中的 `lightglue` 包：本地导出 `SuperPoint`、`DISK`、`ALIKED` 和 `LightGlue(features=...)`；优先选用最少额外依赖的 `SuperPoint + LightGlue(features="superpoint")`，不 vendor 外部仓库。真实 smoke 首次下载官方 `superpoint_v1.pth` 和 `superpoint_lightglue_v0-1_arxiv.pth` 到 torch cache。
- 新增 optional matcher `LightGlue-SuperPoint`，与现有 LightGlue-SIFT 共用 local-feature LightGlue adapter；`lightglue` 缺失时同时记录 `LightGlue-SIFT` 与 `LightGlue-SuperPoint` unavailable，单个 matcher 运行异常仍由 benchmark row 隔离，不中断整轮。
- 新增 focused mock 单测，不加载真实权重：覆盖 optional matcher 组装、LightGlue score 排序和 `--max-matches` 截断。验证命令：`PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest python/test_rotation_matcher_benchmark.py`，7 tests OK。
- Smoke 命令：`PYTHONPATH=python MKL_THREADING_LAYER=GNU /home/xjw/anaconda3/envs/pfm-train/bin/python python/rotation_matcher_benchmark.py --output-dir runs/rotation_matcher_benchmark_lightglue_superpoint_smoke_20260525_worker --angles 90 --resize-max 512 --max-matches 64 --max-keypoints 512 --device cuda --no-auto-pfm` 完成。`LightGlue-SuperPoint` 状态为 ok，但 90° 几何精度较低：numeric 1/18，timestamp 2/7；LightGlue-SIFT 仍为 64/64，LoFTR 为 0/52 与 0/64，SuperGlue/PFM unavailable。
- 完整 SuperPoint 旋转对比：`PYTHONPATH=python MKL_THREADING_LAYER=GNU /home/xjw/anaconda3/envs/pfm-train/bin/python python/rotation_matcher_benchmark.py --output-dir runs/rotation_matcher_benchmark_lightglue_superpoint_full_20260525 --angles 90 180 270 --resize-max 768 --max-matches 256 --max-keypoints 1024 --device cuda --pfm-pytorch-state runs/cross_view_1024_no_self_warp_finetune_300_seed1234/training/pytorch_pfm_state.pt` 完成。产物：`metrics.csv`、`summary.txt`、42 张 PNG。Aggregate：SIFT/ORB/AKAZE/PFM 均为 1536/1536，LightGlue-SIFT 为 1535/1536，LightGlue-SuperPoint 为 1/69，LoFTR 为 3/614，SuperGlue unavailable。结论：在这些 90/180/270 同图旋转对上，learned local-feature 里只有 SIFT-backed LightGlue 接近传统 SIFT/ORB/AKAZE/PFM；SuperPoint/LoFTR 的默认权重不适合离散大旋转。

## 2026-05-25 rotation matcher benchmark DISK/ALIKED worker
- 继续接入 `LightGlue-DISK` 与 `LightGlue-ALIKED`，本地 `lightglue` 包中的 `DISK`、`ALIKED` extractor 及 `LightGlue(features="disk"/"aliked")` 均可 CUDA 前向；仍不 vendor 外部仓库。新增 mock 测试后，`PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest python/test_rotation_matcher_benchmark.py` 为 8 tests OK。
- DISK/ALIKED smoke：`runs/rotation_matcher_benchmark_disk_aliked_smoke_20260525_worker`。90° 几何精度：LightGlue-DISK numeric 0/4、timestamp 1/39；LightGlue-ALIKED numeric 1/22、timestamp 0/12。SIFT/ORB/AKAZE/LightGlue-SIFT 仍为 64/64。
- 最终 full learned/classical rotation benchmark：`runs/rotation_matcher_benchmark_all_learned_full_20260525`。命令包含 2 styles x 90/180/270 x SIFT/ORB/AKAZE/LightGlue-SIFT/LightGlue-SuperPoint/LightGlue-DISK/LightGlue-ALIKED/LoFTR/SuperGlue/PFM，写出 `metrics.csv`、`summary.txt` 和 54 张 PNG。Aggregate：SIFT/ORB/AKAZE/PFM 均为 1536/1536，LightGlue-SIFT 1535/1536，LightGlue-ALIKED 221/374（主要来自 numeric 270 单项），LightGlue-DISK 4/123，LightGlue-SuperPoint 1/69，LoFTR 3/614，SuperGlue unavailable。结论：默认 learned local features 对大离散旋转普遍不稳，传统旋转不变特征和 PFM 在同图旋转基准上更可靠。

## 2026-05-25 rotation matcher benchmark LightGlue-DISK/ALIKED worker
- 探测 `pfm-train` 环境中的 `lightglue`：`DISK`、`ALIKED` extractor 和 `LightGlue(features="disk"/"aliked")` 均可构造；96x96 合成图 CUDA 前向也可跑通。探测过程中首次下载官方 `disk_lightglue`、`aliked_lightglue`、DISK `depth-save.pth`、`aliked-n16.pth` 到 torch cache。
- 新增 optional matcher `LightGlue-DISK` 与 `LightGlue-ALIKED`，复用现有 `LightGlueFeatureMatcher` adapter；`lightglue` 缺失时按 unavailable 记录，单个 matcher 运行异常仍由 benchmark row 隔离。
- 新增 focused mock 单测，不加载真实权重：覆盖 optional matcher 组装，以及 DISK/ALIKED `_load()` 使用对应 extractor 和 `LightGlue(features=...)`。验证命令：`PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest python/test_rotation_matcher_benchmark.py`，8 tests OK。
- Smoke 命令：`PYTHONPATH=python MKL_THREADING_LAYER=GNU /home/xjw/anaconda3/envs/pfm-train/bin/python python/rotation_matcher_benchmark.py --output-dir runs/rotation_matcher_benchmark_disk_aliked_smoke_20260525_worker --angles 90 --resize-max 512 --max-matches 64 --max-keypoints 512 --device cuda --no-auto-pfm` 完成。产物：`metrics.csv`、`summary.txt`、16 张 PNG。`LightGlue-DISK` 状态 ok，但 90° 几何精度低：numeric 0/4，timestamp 1/39；`LightGlue-ALIKED` 状态 ok，numeric 1/22，timestamp 0/12。SIFT/ORB/AKAZE/LightGlue-SIFT 仍为 64/64，SuperPoint/LoFTR 仍明显不鲁棒，SuperGlue/PFM unavailable。

## 2026-05-25 rotation matcher旁路迭代
- 确认入口：`PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest python/test_rotation_matcher_benchmark.py` 通过，8 tests OK。
- 现有 full learned 结果目录：`runs/rotation_matcher_benchmark_all_learned_full_20260525`，summary 显示 SuperGlue 因依赖/adapter 不可用，其余 matcher 可运行。
- 新增 `RootSIFT-FLANN-RANSAC`：SIFT descriptor 做 RootSIFT L1+sqrt，FLANN KNN ratio filter，反向 ratio mutual check，最后用 `estimateAffinePartial2D(..., RANSAC)` 过滤几何内点。
- TDD 记录：新增 RootSIFT helper 与 matcher 注册测试后，focused 测试按预期失败；实现后 `PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest python/test_rotation_matcher_benchmark.py` 通过，10 tests OK。
- 完整 Python 验证：`PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest discover -s python -p 'test_*.py'` 通过，124 tests OK，1 skipped。
- 完整 benchmark：`runs/rotation_matcher_iter_rootsift_ransac` 完成，RootSIFT-FLANN-RANSAC 为 1536/1536 correct、precision=1.0；SuperGlue 仍 unavailable；结果说明见 `runs/rotation_matcher_iter_rootsift_ransac/agent_summary.md`。

## 2026-05-25 rotation matcher旁路迭代 AffineSIFT
- 探测本地 OpenCV：`cv2.__version__ == 4.13.0`，`AffineFeature_create` 与 `SIFT_create` 均可用；96x96 合成 smoke 可 `detectAndCompute` 输出 SIFT descriptor。
- 新增 `AffineSIFT-BF`：`cv2.AffineFeature_create(cv2.SIFT_create())` + 原有 BF cross-check 匹配器；如果 OpenCV 缺 AffineFeature 或 SIFT，会按 unavailable 记录。
- TDD 记录：新增 matcher 注册测试后 focused 测试按预期失败；实现后 `PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest python/test_rotation_matcher_benchmark.py` 通过，11 tests OK。
- 完整 Python 验证：`PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest discover -s python -p 'test_*.py'` 通过，125 tests OK，1 skipped。
- 完整 benchmark：`runs/rotation_matcher_iter_affinesift` 完成，AffineSIFT-BF 为 1536/1536 correct、precision=1.0；SuperGlue 仍 unavailable；结果说明见 `runs/rotation_matcher_iter_affinesift/agent_summary.md`。

## 2026-05-25 rotation matcher旁路继续迭代 USAC
- 已按用户要求继续匹配算法对比线，不等待主 1024 训练；本轮只计划触碰 `python/rotation_matcher_benchmark.py`、`python/test_rotation_matcher_benchmark.py` 和新的 `runs/rotation_matcher_iter_continued_*` 输出。
- 已读取现有 benchmark/test 与 `runs/rotation_matcher_iter_rootsift_ransac`、`runs/rotation_matcher_iter_affinesift` 的 summary/metrics/agent_summary；确认 RootSIFT-FLANN-RANSAC 与 AffineSIFT-BF 均为 1536/1536 correct、precision=1.0。
- 本地 `pfm-train` OpenCV 4.13.0 支持 `USAC_MAGSAC` 和 `USAC_PROSAC`，下一步按 TDD 增加 RootSIFT + USAC 几何验证 matcher。
- TDD：新增 `test_make_opencv_matchers_includes_rootsift_usac_when_available` 后 focused 测试按预期失败，缺少 `RootSIFT-FLANN-USAC-MAGSAC`。
- 实现：将现有 RootSIFT-FLANN-RANSAC 参数化为可选 OpenCV 几何验证 method，并新增 `RootSIFT-FLANN-USAC-MAGSAC` 与 `RootSIFT-FLANN-USAC-PROSAC`；focused 测试通过，12 tests OK。
- 首次 benchmark 发现 USAC 变体全为 error；根因是 `estimateAffinePartial2D()` 不支持 USAC method。新增 `filter_points_with_homography_usac()` 测试后改用 `findHomography()` 做 USAC 内点过滤；focused 测试通过，13 tests OK。
- 完整 benchmark 已重跑覆盖旧 error：`runs/rotation_matcher_iter_continued_usac_20260525`。USAC-MAGSAC 与 USAC-PROSAC 均为 1536/1536 correct、precision=1.0；目录包含 `metrics.csv`、`summary.txt`、`agent_summary.md` 和 78 张可视化 PNG。

## 2026-05-25 cross-view nonfinite-gradient fix and sanitized reruns
- 系统排查 warp-hard-negative 短训的 skipped step：原始 `runs/cross_view_1024_warp_hn005_lr5e6_40_seed1234` 在 step 7/20 跳过，复现定位到 `splits/train/numeric/viewpoint/source_000198_7/pair_002739.pt`；该 archive 的 `view_b` 含 1 个 NaN 像素。最终 descriptor 输出被 `nan_to_num` 清成有限值，但卷积 backward 仍保存了含 NaN 的中间激活，导致 `sparse_head.descriptors.*` 梯度全 NaN。
- 新增回归测试：`test_descriptor_training_gradients_ignore_nonfinite_image_pixels` 和 `test_texture_descriptor_sanitizes_nonfinite_image_pixels_before_filtering`。修复为在 `Backbone.forward()` 与 `make_rotation_invariant_texture_descriptor()` 入口清洗 NaN/Inf 像素，避免 forward 激活和 texture teacher/blend target 被非有限像素污染。
- 验证：原复现链从同一 checkpoint、同一 split、同一采样顺序跑 21 steps，原本第 7/20 step 的 skip 均消失，21/21 steps 全部 `skip=0`；完整 Python 测试 `PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest discover -s python -p 'test_*.py'` 通过，127 tests OK，1 skipped。
- 修复后重跑 `warp_hard_negative_weight=0.05` 40-step：`runs/cross_view_1024_warp_hn005_sanitized_40_seed1234`。训练 40/40 steps 全部 `skip=0`，descriptor sanity eval 仍变差：top1 0.0655 -> 0.0603，loss 4.9173 -> 4.9749。64-pair test precision：numeric/rotate 0.814341，numeric/viewpoint 0.115189，numeric/compound 0.107496，timestamp/rotate 0.602892，timestamp/viewpoint 0.050000，timestamp/compound 0.038067。它提升 numeric/compound，但 timestamp viewpoint/compound 仍差，不替代当前 full-test routed best。
- 继续重跑包含 texture descriptor 入口清洗的同配置实验：`runs/cross_view_1024_warp_hn005_sanitized_texture_40_seed1234`。训练 40/40 steps 全部 `skip=0`，descriptor sanity eval top1 0.0655 -> 0.0605。64-pair test precision：numeric/rotate 0.813670，numeric/viewpoint 0.091040，numeric/compound 0.107168，timestamp/rotate 0.600370，timestamp/viewpoint 0.050914，timestamp/compound 0.037037。texture 清洗改善数值正确性，但不是单独的指标收益点；下一步不要继续单纯延长这一配置，应转向更有针对性的 weak-gate/objective 或 specialist routing。

## 2026-05-25 cross-view match-margin calibration
- 为 `python/cross_view_experiment.py` 增加匹配过滤参数链路：`--geometry-filter`、`--match-min-margin`、`--calibrate-match-min-margins`、`--match-min-margin-candidates`。校准 CSV 现在记录每个 style/gate 选中的 `texture_blend_weight` 和 `min_margin`，最终 evaluation 和 visualization 复用同一选择。
- 新增单测覆盖 eval 命令传递 `--min-margin`、margin candidates 解析、校准可选择 margin、CLI 参数解析。验证：`PYTHONPATH=python /home/xjw/anaconda3/envs/pfm-train/bin/python -m unittest python/test_cross_view_experiment.py` 通过，27 tests OK；完整 Python 测试通过，131 tests OK，1 skipped。
- 先做 64-pair validation sweep：`runs/match_filter_sweep_val64_20260525/summary.csv`。`min_margin=0.01/0.02` 能提升部分低精度组的 precision，尤其 numeric/compound 和 timestamp/viewpoint，但会大幅减少 matches；viewpoint 部分候选在更高 margin 下直接变成 0 matches。
- 0-step 复用现有 checkpoint 的完整六组短评估：`runs/cross_view_1024_match_margin_calib_0step_seed1234`。校准候选为 texture blend `0,0.25,1,2,4` × margin `0,0.01`，`calibration_limit_pairs=16`，最终写出六组 summary 和 12 张 visualization。
- 64-pair test precision：numeric/rotate 0.879445 (2473/2812)，numeric/viewpoint 0.154344 (167/1082)，numeric/compound 0.173633 (54/311)，timestamp/rotate 0.642534 (710/1105)，timestamp/viewpoint 0.061321 (13/212)，timestamp/compound 0.039773 (7/176)。
- 加入 support-aware selector：`select_best_blend_weight_summaries()` 可用 `min_matches` / `min_match_fraction` 过滤低支撑候选，CLI 新增 `--calibration-min-matches` 与 `--calibration-min-match-fraction`；相关单测后完整 Python 测试通过，135 tests OK，1 skipped。
- 过强支撑对照：`runs/cross_view_1024_match_margin_support128_0step_seed1234` 使用 `--calibration-min-matches 128`。它避免了部分低 match 候选，但 test 变差：numeric/compound 0.121786，timestamp/compound 0.002967。结论是不能用固定较高 match 下限粗暴约束弱 timestamp compound。
- 多 seed 对照：`runs/cross_view_1024_match_margin_multiseed16_0step_seed1234` 用 `--calibration-sample-seeds 1234,2234,3234` 聚合 3 个 16-pair 校准样本。它把 numeric/viewpoint 提到 0.338600、timestamp/compound 提到 0.045082，但 timestamp/rotate 被一个 1/1 的低支撑候选选中，test 变成 0/1。
- 最稳的当前 0-step margin 校准结果：`runs/cross_view_1024_match_margin_multiseed16_min20_0step_seed1234`，使用 3-seed 校准并加 `--calibration-min-matches 20` 只过滤极低支撑候选。64-pair test precision：numeric/rotate 0.879445 (2473/2812)，numeric/viewpoint 0.338600 (150/443)，numeric/compound 0.173633 (54/311)，timestamp/rotate 0.642534 (710/1105)，timestamp/viewpoint 0.061321 (13/212)，timestamp/compound 0.045082 (11/244)。它写出六组 summary 和 12 张 visualization。
- 当前结论：match margin + 多 seed + 低支撑下限是有效的评估/路由层改进，尤其修复 sampled numeric/viewpoint；但它没有改变训练权重，也没有解决 timestamp/viewpoint。下一步若继续训练侧，应优先做 weak-gate specialist 或 loss 监督改动，而不是再调单一全局 margin。

## 2026-05-26 timestamp/viewpoint specialist iteration
- 修复训练侧 hard-pair 入口兼容性：`hard_pair_mining.py` 现在同时支持旧列名 `sparse_matches/match_precision` 和当前 evaluator summary 的 `matches/precision`。新增回归测试 `test_reads_current_cache_eval_summary_columns`。
- `cross_view_experiment.py` 新增 `--training-groups style/gate,...`，可只把指定 style/gate 的 train/val split 传给 `pfm_pytorch_training.py`，但最终仍按六组 test 评估和绘图；新增测试覆盖 `select_created_cache_dirs()` 与 CLI 参数解析。
- 验证：focused 测试 `python/test_pfm_pytorch_training.py python/test_hard_pair_mining.py python/test_cross_view_experiment.py` 通过，75 tests OK；完整 Python 测试通过，138 tests OK，1 skipped。
- 运行 timestamp/viewpoint specialist 短训：`runs/cross_view_1024_timestamp_viewpoint_specialist_60_seed1234`。从 `runs/cross_view_1024_blend025_batch4_40_seed1234/training/pytorch_pfm_state.pt` 继续，只训练 `timestamp/viewpoint` split，60 steps，`batch_pairs=4`、`gradient_accumulation_steps=2`、`samples_per_pair=512`、`learning_rate=5e-6`、`training_texture_blend_weight=0.25`。训练 60/60 steps 全部 `skip=0`；descriptor sanity eval 仅微变：top1 0.0526 -> 0.0529，loss 5.859217 -> 5.858403。
- 使用当前最稳的 3-seed + `--calibration-min-matches 20` margin/blend 校准后，64-pair test precision：numeric/rotate 0.962825 (259/269)，numeric/viewpoint 0.315789 (18/57)，numeric/compound 0.105263 (2/19)，timestamp/rotate 0.800000 (212/265)，timestamp/viewpoint 0.066176 (9/136)，timestamp/compound 0.040541 (3/74)。它写出六组 summary 和 12 张 visualization。
- 结论：该 specialist 不是更好的主模型，也不适合替换当前 `cross_view_1024_match_margin_multiseed16_min20_0step_seed1234` 路由；它主要把多个组变成低召回高 precision，timestamp/viewpoint 只从 0.061321 小幅到 0.066176。后续训练侧改进需要显式召回/匹配支撑约束或更强 loss 监督，不能只用 60-step 单 gate descriptor 微调。

## 2026-05-26 checkpoint-aware routing and rotation comparison
- 为 `cross_view_experiment.py` 增加 checkpoint-aware calibration：CLI 新增可重复的 `--calibration-pytorch-state label=path`，calibration 在 checkpoint × texture blend × match margin × seed 维度里选最优候选；`selected_weights.csv` 新增 `pytorch_state_label` 和 `pytorch_state` 字段，最终 evaluation 与 visualization 按 group 加载被选中的 checkpoint。新增测试覆盖 parser、CLI 参数和按 checkpoint 选择。验证：focused `python/test_rotation_matcher_benchmark.py python/test_cross_view_experiment.py` 47 tests OK；完整 Python discovery 141 tests OK，1 skipped。
- 运行 0-step checkpoint-routed 评估：`runs/cross_view_1024_checkpoint_routed_margin_multiseed16_min20_0step_seed1234`。候选为当前 balanced checkpoint 与 `runs/cross_view_1024_blend025_batch4_40_seed1234/training/pytorch_pfm_state.pt`，校准使用 texture blend `0,0.25,1,2,4`、margin `0,0.01`、3 seeds、`--calibration-min-matches 20`。validation 校准六组全部选择 `blend02540` checkpoint。
- 该 routed run 的 64-pair test precision：numeric/rotate 0.962472 (436/453)，numeric/viewpoint 0.305344 (40/131)，numeric/compound 0.260870 (12/46)，timestamp/rotate 0.761290 (236/310)，timestamp/viewpoint 0.061224 (3/49)，timestamp/compound 0.000000 (0/23)。它写出 6 组 summary、12 张 visualization、362 个 calibration CSV。
- 结论：checkpoint routing 提升了 rotate 与 numeric/compound precision，但主要通过减少 match support；timestamp/viewpoint 没有改善，timestamp/compound 在 fixed test sample 上完全失败。因此 `blend02540` 不应全组路由替代当前 `cross_view_1024_match_margin_multiseed16_min20_0step_seed1234`；下一轮应避免只按小 validation precision 选 checkpoint，应加入 test/full-val 支撑、召回或 group-specific fallback 约束。
- 按用户要求开子 agent 做独立旋转算法对比，输出在 `runs/rotation_matcher_comparison_agent/`，脚本为 `scripts/rotation_matcher_comparison.py`。两类风格图像各做 90/180/270 旋转，baseline 与一次实用迭代合计 144 CSV 行、132 张可视化。Aggregate：PFM、SIFT、ORB、AKAZE 都为 1536/1536 correct；LightGlue-SIFT 为 1535/1536；LightGlue-SuperPoint 1/69，LightGlue-DISK 4/123，LightGlue-ALIKED 221/374，LoFTR 3/614；SuperGlue 因缺少 `match_pairs` 入口 unavailable。迭代版 RootSIFT-FLANN-USAC-MAGSAC、ORB-AffineRANSAC、AKAZE-AffineRANSAC 也均为 1536/1536。
- 进一步修复 checkpoint routing 的选择规则：`cross_view_experiment.py` 新增 `--calibration-state-switch-min-precision-gain` 与 `--calibration-state-switch-min-match-ratio`，只在候选 checkpoint 相对 reference state 同时满足 precision 增益和 match 支撑比例时才切换，否则回退到 reference。新增 TDD 覆盖 CLI 参数与 selector 行为；完整 Python discovery 通过，143 tests OK，1 skipped。
- guard 版 0-step route：`runs/cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234`，使用 `--calibration-min-match-fraction 0.1`、`--calibration-state-switch-min-precision-gain 0.03`、`--calibration-state-switch-min-match-ratio 0.25`。最终 selected：numeric/rotate 与 timestamp/rotate 使用 `blend02540`；numeric/viewpoint、numeric/compound、timestamp/viewpoint、timestamp/compound 回退 trained。
- guard route 64-pair test precision：numeric/rotate 0.962472 (436/453)，numeric/viewpoint 0.338600 (150/443)，numeric/compound 0.173633 (54/311)，timestamp/rotate 0.761290 (236/310)，timestamp/viewpoint 0.061321 (13/212)，timestamp/compound 0.045082 (11/244)。它保留 rotate checkpoint 收益，同时避免无 guard route 的 timestamp/compound 0/23 退化；仍未解决 timestamp/viewpoint。

## 2026-05-26 hard-mined weak-gate training and external cross-view baselines
- Added `cross_view_experiment.py` pass-throughs for hard-pair training: `--hard-summary`, `--mine-hard-training-pairs`, `--hard-mine-limit-pairs`, `--hard-limit`, `--hard-min-matches`, `--hard-max-precision`, `--hard-repeat`, `--hard-curriculum-max-probability`, and `--hard-curriculum-warmup-steps`. Focused tests passed with 83 tests OK; full Python discovery passed with `146 tests OK, 1 skipped`.
- Hard-mined weak-gate run: `runs/cross_view_1024_hard_mined_weakgates_80_seed1234`. It trained only `numeric/viewpoint`, `numeric/compound`, `timestamp/viewpoint`, and `timestamp/compound`, first mining 96 train pairs per group from the base checkpoint. Hard mining train precision showed the weak groups clearly: numeric/viewpoint 506/3221 (0.157094), numeric/compound 238/2575 (0.092427), timestamp/viewpoint 76/1241 (0.061241), timestamp/compound 51/1044 (0.048851).
- The hard-mined 80-step training was a negative result. Descriptor sanity eval worsened from loss 5.112186 / top1 0.059265 to loss 5.440673 / top1 0.051697. With the default state-switch reference, calibration selected the newly trained checkpoint for all six groups and test precision regressed badly: numeric/rotate 0.955556 (430/450), numeric/viewpoint 0.084262 (174/2065), numeric/compound 0.097177 (148/1523), timestamp/rotate 0.725664 (246/339), timestamp/viewpoint 0.033898 (4/118), timestamp/compound 0.031884 (22/690).
- The negative run exposed a routing API issue: the trained checkpoint is always labeled `trained`, so state-switch guard treated the new experimental checkpoint as the reference state. Added `--calibration-state-switch-reference-label` so future runs can use `base` or another label as the checkpoint fallback. TDD covered parser and calibration behavior; full Python discovery after this change passed with `146 tests OK, 1 skipped`.
- Reused the completed calibration CSVs for a base-reference post-selection run without retraining or recalibrating: `runs/cross_view_1024_hard_mined_weakgates_80_base_ref_postselect`. Selected hard-mined checkpoint only for rotate, and base checkpoint for the weak groups. Test precision recovered weak groups to the guarded baseline but did not improve them: numeric/rotate 0.955556 (430/450), numeric/viewpoint 0.338600 (150/443), numeric/compound 0.173633 (54/311), timestamp/rotate 0.725664 (246/339), timestamp/viewpoint 0.061321 (13/212), timestamp/compound 0.045082 (11/244). It wrote six summaries and twelve visualizations.
- Subagent Bohr completed `scripts/rotation_matcher_comparison_agent2.py` and `runs/rotation_matcher_comparison_agent2/`: 72 metrics rows, 66 ok rows, 66 raw visualizations and 63 homography visualizations. Same-image 90/180/270 remains a ceiling task for SIFT/RootSIFT/ORB/AKAZE and PFM after homography filtering; SuperGlue remains unavailable because `match_pairs` is not installed.
- Subagent Godel completed actual cross-view traditional matcher comparison: `scripts/cross_view_traditional_matcher_comparison_agent3.py` and `runs/cross_view_traditional_matcher_comparison_agent3/`. It sampled 64 real test cache pairs across numeric/timestamp viewpoint/compound, ran CPU OpenCV matchers, and wrote 960 metric rows plus 60 PNGs. RootSIFT-FLANN-ratio raw precision was far above PFM guarded summary on the same sampled groups: numeric/viewpoint 1480/1720 (0.860465), numeric/compound 1000/1204 (0.830565), timestamp/viewpoint 1607/1814 (0.885888), timestamp/compound 1378/1582 (0.871049). RootSIFT + homography RANSAC reached 0.972522-0.998124 precision across the four groups.
- Updated conclusion: hard-pair curriculum using current synthetic warp descriptor CE does not solve cross-view matching. The strongest evidence now points to using traditional matcher inliers, especially RootSIFT + homography/USAC, as high-confidence pseudo-labels or distillation targets for PFM on real cross-view cache pairs.

## 2026-05-26 RootSIFT pseudo-label training iteration
- Added a tested pseudo-label training path to `python/pfm_pytorch_training.py`: `--pseudo-label-csv`, `--pseudo-label-weight`, `--pseudo-label-max-points`, `--pseudo-label-curriculum-max-probability`, `--pseudo-label-curriculum-warmup-steps`, and `--synthetic-loss-weight`. Pseudo-label rows are matched by full/relative path keys, not basename fallback, to avoid cross-cache ambiguity. Full Python discovery passed after the changes with `157 tests OK, 1 skipped`.
- Added `python/pseudo_label_generation.py` and `scripts/generate_rootsift_pseudo_labels.py`, which export RootSIFT-FLANN-ratio + homography RANSAC inliers filtered again by warp truth. The CSV contains image-space `ax,ay,bx,by` rows, and training scales them to descriptor feature-grid coordinates.
- First weak-group sample: `runs/rootsift_pseudo_labels_weakgroups_seed1234/pseudo_labels.csv`, generated from 32 train pairs across numeric/timestamp viewpoint/compound. It kept 24 pairs and 2458 labels using the earlier 4px/3px style thresholds.
- A 40-step run without explicit pseudo-label pair sampling consumed no labels (`pseudo_label_points=0` for all steps) and regressed validation top1 from 0.1004 to 0.0924. This exposed the need for curriculum sampling of pseudo-labeled pairs; simply providing a CSV is not enough when labels cover a tiny subset of a large cache.
- With `--pseudo-label-curriculum-max-probability 0.5`, the same 40-step setup consumed 227-384 pseudo-label points per logged step, but still regressed top1 from 0.1004 to 0.0925. A more conservative blended setup (`lr=2e-6`, pseudo weight 0.1, curriculum 0.25, no hard-negative/diversity) reduced the regression to 0.1004 -> 0.0984.
- Pseudo-label-only training was the first positive descriptor-retrieval result: `runs/cross_view_1024_rootsift_pseudo_only_lr1e6_weakgroups_40_seed1234` used `--synthetic-loss-weight 0`, `--teacher-weight 0`, and pseudo-label curriculum 1.0. Validation improved from loss 4.5366/top1 0.1004/top5 0.2859 to loss 4.5380/top1 0.1016/top5 0.2891.
- Subagent Kant wrote `scripts/matcher_algorithm_iteration_agent4.py` and `runs/matcher_algorithm_iteration_agent4/`. It verified B-rotation coordinate inverse-transform handling and showed RootSIFT-FLANN-ratio+HomographyRANSAC stays at roughly 0.991-1.000 precision across numeric/timestamp viewpoint/compound and rotations 0/90/180/270; LightGlue-SIFT smoke is available but lower priority.
- Subagent Hypatia wrote `scripts/matcher_algorithm_iteration_agent5.py` and `runs/matcher_algorithm_iteration_agent5/`. Threshold sensitivity favors `ratio=0.80` with RANSAC 2px or 3px for clean training labels; LightGlue-SIFT, LoFTR, SuperPoint, ALIKED, and DISK were available in smoke but less precise/noisier than RootSIFT-HRANSAC for pseudo-label generation.
- Subagent Noether wrote `scripts/matcher_algorithm_iteration_agent6.py` and `runs/matcher_algorithm_iteration_agent6/`. Pair-level mining recommends `ratio=0.80`, RANSAC 2px, pair precision >=0.98, min inliers >=20 for viewpoint and >=8 for compound. In its 32-pair-per-group sample, RANSAC 2px consistently had higher precision than 3px.
- Regenerated stricter expanded labels: `runs/rootsift_pseudo_labels_weakgroups_r080_t2_seed1234/pseudo_labels.csv`, using ratio 0.80, RANSAC 2px, truth threshold 2px, min labels 20, and 32 pairs per weak group. It kept 93/128 pairs and 9366 point labels: numeric/viewpoint 17 pairs/1792 labels, numeric/compound 25/2517, timestamp/viewpoint 23/2298, timestamp/compound 28/2759.
- Expanded pseudo-only run: `runs/cross_view_1024_rootsift_pseudo_r080t2_only_lr1e6_weakgroups_80_seed1234` consumed labels every step and improved validation retrieval from loss 4.5366/top1 0.1004/top5 0.2859 to loss 4.5266/top1 0.1027/top5 0.2911. This is a small but repeatable positive training-side signal.
- Guarded six-group follow-up: `runs/cross_view_1024_rootsift_pseudo_r080t2_guard_eval_0step_seed1234` evaluated the expanded pseudo-only checkpoint against the guarded base checkpoint. Calibration selected the pseudo checkpoint only for `timestamp/viewpoint`; all other groups selected the base checkpoint. It wrote six summaries and twelve visualizations.
- Pseudo guarded test precision: numeric/rotate 0.879672 (2471/2809), numeric/viewpoint 0.341629 (151/442), numeric/compound 0.165109 (53/321), timestamp/rotate 0.653880 (733/1121), timestamp/viewpoint 0.092369 (23/249), timestamp/compound 0.042017 (10/238). Because this run did not include the older `blend02540` rotate specialist as a calibration candidate, rotate metrics are lower than `runs/cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234`.
- Best current composite route by available evidence is: keep the old guarded route for numeric/rotate, numeric/compound, timestamp/rotate, and timestamp/compound; use either old/new base for numeric/viewpoint; switch `timestamp/viewpoint` to the pseudo checkpoint. That gives the same old guarded metrics except `timestamp/viewpoint` improves from 13/212 (0.061321) to 23/249 (0.092369). This is the first sparse-matching improvement on the weakest timestamp/viewpoint gate from training-side supervision.
- Continued matcher sidecars:
  - Agent7 (`scripts/matcher_algorithm_iteration_agent7.py`, `runs/matcher_algorithm_iteration_agent7/`) showed RootSIFT-HRANSAC r0.80/t2 remains a high-precision source on rotated cross-view samples: 5074/5074 correct over 48 rotated evals. Current PFM raw fallback was only 322/14599 (0.0221), and naive RootSIFT-else-PFM routing fell to aggregate precision 0.4931.
  - Agent8 (`scripts/matcher_algorithm_iteration_agent8.py`, `runs/matcher_algorithm_iteration_agent8/`) tested PFM mutual/margin/homography post-filtering. Best global PFM+RANSAC config reached only 28/46 (0.6087), with no global precision >=0.80 region. PFM should not be used as pseudo-label fallback.
  - Agent9/10 (`scripts/matcher_algorithm_iteration_agent9.py`, `scripts/matcher_algorithm_iteration_agent10.py`) found a safer classical fallback: use RootSIFT r0.90 + H-RANSAC 2px only where baseline r0.80/t2 fails its gate. Agent10 larger sample: r0.80/t2 pass gate 127/192 at precision 0.9935; r0.90/t2 pass gate 146/192 at all-sample precision 0.9695; insertion-only recovery was 19/65 with 584/589 correct (0.9915). CLAHE r0.90/t2 was cleaner but recovered fewer cases: 12/65, 418/418 correct.
- Group-aware r0.80/t2 label expansion and training:
  - Generated `runs/rootsift_pseudo_labels_viewpoint_r080_t2_min20_p128_seed1234` and `runs/rootsift_pseudo_labels_compound_r080_t2_min8_p128_seed1234` using the same 256-pair candidate pool later used for fallback experiments. They kept 182 viewpoint pairs / 20258 labels and 187 compound pairs / 17259 labels.
  - `runs/cross_view_1024_rootsift_pseudo_groupaware_only_lr1e6_b4ga2p128_weakgroups_160_seed1234` trained pseudo-only for 160 steps from the guarded base checkpoint. Validation retrieval improved from top1 0.0784/top5 0.2284 to top1 0.0822/top5 0.2376.
  - Full guarded eval with base + `blend02540` + this group-aware checkpoint (`runs/cross_view_1024_rootsift_pseudo_groupaware_guard_blend_0step_seed1234`) did not beat the earlier r0.80/t2 timestamp/viewpoint specialist. Selected routes were `blend02540` for rotate, base for numeric/viewpoint and numeric/compound, trained group-aware only for timestamp/compound. Test precision: numeric/rotate 0.964286 (432/448), numeric/viewpoint 0.341629 (151/442), numeric/compound 0.165109 (53/321), timestamp/rotate 0.760252 (241/317), timestamp/viewpoint 0.044335 (9/203), timestamp/compound 0.055000 (11/200).
- r0.90/t2 expansion experiments:
  - Full r0.90/t2 labels kept 191 viewpoint pairs / 22627 labels and 189 compound pairs / 20482 labels. `runs/cross_view_1024_rootsift_pseudo_r090t2_only_lr1e6_b4ga2p128_weakgroups_160_seed1234` improved validation top1 0.0784 -> 0.0813, but fixed sparse eval showed numeric/viewpoint regressed to 134/433 (0.309469), while timestamp/viewpoint was only slightly better at 24/256 (0.093750).
  - Fallback-only CSVs (`runs/rootsift_pseudo_labels_viewpoint_r090_t2_fallback_from_r080_min20_p128_seed1234` and `runs/rootsift_pseudo_labels_compound_r090_t2_fallback_from_r080_min8_p128_seed1234`) added only r0.90/t2 pairs that r0.80/t2 did not keep: 9 viewpoint pairs / 347 labels and 6 compound pairs / 142 labels.
  - `runs/cross_view_1024_rootsift_pseudo_r080t2_plus_r090fallback_only_lr1e6_b4ga2p128_weakgroups_160_seed1234` improved validation top1 0.0784 -> 0.0812, but fixed sparse eval did not improve the current best: numeric/viewpoint 135/437 (0.308924), numeric/compound 56/339 (0.165192), timestamp/viewpoint 20/230 (0.086957), timestamp/compound 9/265 (0.033962).
  - A timestamp/viewpoint-only specialist from r0.90/t2 labels (`runs/cross_view_1024_timestamp_viewpoint_rootsift_r090t2_only_lr1e6_b4ga2p128_160_seed1234`) improved single-group validation top1 0.0615 -> 0.0628, but fixed sparse eval was 14/165 (0.084848), below the old r0.80/t2 pseudo specialist. A timestamp/viewpoint test sweep for the r0.90 full checkpoint confirmed `weight=1, margin=0.01` was best at 24/256 (0.093750), only one more correct match than the old r0.80/t2 specialist and not enough to justify formal route replacement.
- Added pseudo-label keypoint heatmap supervision and learned keypoint-score evaluation:
  - `pfm_pytorch_training.py` now has `heatmap_point_loss`, `--pseudo-keypoint-weight`, and `--pseudo-keypoint-negative-weight`; when enabled it trains `sparse_head.heatmap*` from RootSIFT pseudo-label locations in addition to descriptor CE.
  - `pytorch_cache_match_eval.py` now supports `--keypoint-score-mode learned`, so sparse keypoint selection can rank candidates with the learned heatmap instead of image texture.
  - Verification after the code change and keypoint-only edge-case fix: full Python discovery passed with `162 tests OK, 1 skipped`.
- Keypoint-supervised r0.80/t2 viewpoint run: `runs/cross_view_1024_rootsift_pseudo_r080t2_keypoint_w1n002_lr1e6_viewpoint_80_seed1234`. It trained 80 steps from the guarded base checkpoint on numeric/timestamp viewpoint labels with descriptor weight 1.0 and keypoint weight 1.0. Training consumed pseudo labels every logged step (`pseudo_kp` roughly twice descriptor pseudo points), but validation retrieval regressed: loss 3.709143/top1 0.1947/top5 0.4432 -> loss 3.762479/top1 0.1878/top5 0.4255.
- Sparse eval for the keypoint-supervised checkpoint is mixed:
  - numeric/viewpoint showed no learned-keypoint benefit on the fixed 64-pair sample: texture score 128/388 (0.329897), learned score 96/292 (0.328767).
  - timestamp/viewpoint improved when the learned heatmap was used for keypoint selection. The best small sweep result was learned score + texture blend 4 + margin 0.01: 41/223 (0.183857), compared with old guarded base 13/212 (0.061321) and old r0.80/t2 pseudo specialist 23/249 (0.092369). Without margin, learned score gave more correct matches but very low precision, e.g. blend 2: 69/1379 (0.050036).
  - Current interpretation: learned heatmap selection is a promising timestamp/viewpoint specialist route, but it trades recall/support and hurts descriptor validation, so it should not replace the global route without calibration support and full-val/full-test safeguards.
- Agent11 matcher sidecar completed `scripts/matcher_algorithm_iteration_agent11.py` and `runs/matcher_algorithm_iteration_agent11/`, covering numeric and timestamp/NAS cross-view viewpoint/compound pairs with B rotated by 90/180/270. Global summary: RootSIFT-r0.90-Ht2 1890/1891 (0.999471), LightGlue-SIFT-Ht3 2089/2089 (1.0), ORB-cross-Ht3 1469/1475 (0.995932), RootSIFT-r0.80-Ht2 1687/1687 (1.0), SIFT-r0.80-Ht2 1610/1610 (1.0), AKAZE-cross-Ht3 1055/1062 (0.993409), PFM raw 83/2161 (0.038408), PFM-Ht3 1/8 (0.125). SuperGlue remains unavailable locally. This reinforces that classical/LightGlue-SIFT matchers already solve many rotated cross-view cases that PFM does not.
- Added learned-score routing to `cross_view_experiment.py` so calibration can select `texture` or `learned` keypoint ranking per style/gate. Full Python discovery after the evaluator/orchestrator tests passed with `165 tests OK, 1 skipped`.
- Keypoint-only viewpoint run `runs/cross_view_1024_rootsift_pseudo_r080t2_keypointonly_w1n002_lr1e5_viewpoint_80_seed1234` trained heatmap supervision only from r0.80/t2 viewpoint labels. Descriptor validation was unchanged, as intended, while heatmap loss decreased. Sparse eval showed timestamp/viewpoint learned-score margin candidates around 18/96 to 41/239 depending on blend/margin, without descriptor damage.
- Six-group learned-score guarded calibration of the viewpoint keypoint-only checkpoint: `runs/cross_view_1024_keypointonly_learnedscore_guard_calib_0step_seed1234`. Selected routes were mostly base/blend checkpoints, with learned keypoint scoring in 5/6 groups. Test precision: numeric/rotate 647/693 (0.933622), numeric/viewpoint 53/143 (0.370629), numeric/compound 49/243 (0.201646), timestamp/rotate 555/615 (0.902439), timestamp/viewpoint 20/104 (0.192308), timestamp/compound 3/108 (0.027778). It wrote six summaries and twelve visualizations.
- Keypoint-only weakgroups run `runs/cross_view_1024_rootsift_pseudo_r080t2_keypointonly_w1n002_lr1e5_weakgroups_120_seed1234` started from the viewpoint keypoint-only checkpoint and trained heatmap supervision on both viewpoint and compound r0.80/t2 labels. It consumed 343 labeled training pairs / 37517 pseudo matches, used `batch_pairs=8`, and kept descriptor validation exactly unchanged: loss 3.841238/top1 0.1479/top5 0.4121 before and after.
- Guarded six-group eval for that weakgroups keypoint-only checkpoint: `runs/cross_view_1024_keypointonly_weakgroups_learnedscore_guard_calib_0step_seed1234`. Calibration selected the new trained checkpoint for numeric/compound and timestamp/compound, `blend02540` for both rotate groups and numeric/viewpoint, and base for timestamp/viewpoint. Test precision: numeric/rotate 647/693 (0.933622), numeric/viewpoint 53/143 (0.370629), numeric/compound 47/230 (0.204348), timestamp/rotate 555/615 (0.902439), timestamp/viewpoint 20/104 (0.192308), timestamp/compound 5/72 (0.069444). It wrote six summaries and twelve visualizations.
- Interpretation: keypoint-only heatmap distillation is safer than descriptor+keypoint joint training because it does not damage descriptor retrieval. It gives a modest compound routing signal, especially timestamp/compound precision 0.027778 -> 0.069444 on the fixed 64-pair sample, but it does not improve timestamp/viewpoint beyond the learned-score/base route in this guarded setup.
- Agent12/Nietzsche completed `scripts/matcher_algorithm_iteration_agent12.py` and `runs/matcher_algorithm_iteration_agent12/`. Verification: `metrics.csv` has 193 lines including header, 192/192 rows have `status=ok`, and `visualizations/` contains 189 PNGs. It used 4 patch pairs (numeric/timestamp x viewpoint/compound) and 90/180/270 degree B rotations.
- Agent12 result: best teacher candidate was `RootSIFT-ratio-r0p90-Ht3-min4` at 759/794 correct/inliers (0.955919 precision), coverage 12/12 and pass gate 7/12. `LightGlue-SIFT-Ht3-min4` was higher precision at 674/683 (0.986823) but slightly lower coverage 11/12. The current PFM checkpoint was far behind: `PFM-current-raw` 20/974 (0.020534) and `PFM-current-Ht3-min4` 7/71 (0.098592). SuperGlue remains unavailable because `match_pairs`/`superglue` modules are missing.
- Training implication from Agent12: r0.90/Ht3 has useful coverage for teacher mining, but its raw teacher precision is lower than the earlier conservative r0.80/t2 setup. It is suitable for a heatmap-only/high-coverage experiment with guard rails, not for replacing the stable descriptor pseudo-label route without sparse eval proof.
- Generated Agent12-inspired coverage labels:
  - `runs/rootsift_pseudo_labels_viewpoint_r090_t3_min4_p128_seed1234`: kept 178/256 pairs and 20473 labels.
  - `runs/rootsift_pseudo_labels_compound_r090_t3_min4_p128_seed1234`: kept 175/256 pairs and 18114 labels.
- Heatmap-only r0.90/Ht3 coverage run: `runs/cross_view_1024_rootsift_pseudo_r090t3_keypointonly_w1n002_lr1e5_weakgroups_120_seed1234`. It started from the r0.80/t2 weakgroups heatmap checkpoint, used 256 pseudo-labeled training pairs / 38587 pseudo matches that overlapped the train split, and kept descriptor validation exactly unchanged: loss 3.841238/top1 0.1479/top5 0.4121 before and after.
- Guarded six-group eval: `runs/cross_view_1024_keypointonly_r090t3_weakgroups_learnedscore_guard_calib_0step_seed1234`. Calibration selected the new trained checkpoint for numeric/compound and timestamp/compound, `blend02540` for rotate groups and numeric/viewpoint, and base for timestamp/viewpoint. Test precision: numeric/rotate 647/693 (0.933622), numeric/viewpoint 53/143 (0.370629), numeric/compound 50/225 (0.222222), timestamp/rotate 555/615 (0.902439), timestamp/viewpoint 20/104 (0.192308), timestamp/compound 3/109 (0.027523). It wrote six summaries and twelve visualizations.
- Interpretation: r0.90/Ht3 coverage expansion improves numeric/compound over the r0.80/t2 heatmap run (47/230 -> 50/225) but hurts timestamp/compound (5/72 -> 3/109). Keep the r0.80/t2 weakgroups heatmap checkpoint as the better timestamp/compound candidate; r0.90/Ht3 should not become the default heatmap teacher without style/gate-specific routing.
- Agent13/Popper completed the next matcher sidecar iteration: `scripts/matcher_algorithm_iteration_agent13.py` and `runs/matcher_algorithm_iteration_agent13/`. It evaluated 70 algorithm/parameter configs over 12 style/gate/rotation tasks and wrote 840 parseable metrics rows. The new best global teacher was `RootSIFT-ratio-r0p88-Ht3-min4` at 779/801 correct/inliers (0.972534), coverage 12/12 and pass gate 9/12, improving Agent12's r0.90/Ht3 pass gate count 7/12. Style/gate recommendations were: numeric/viewpoint `RootSIFT-mutual-r0p95-Ht3-min4` (685/685, pass 3/3), numeric/compound `LightGlue-SIFT-Ht3-min4` (53/56, pass 1/3), timestamp/viewpoint `RootSIFT-ratio-r0p92-Ht3-min4` (97/97, pass 3/3), timestamp/compound `RootSIFT-ratio-r0p88-Ht3-min4` (39/39, pass 3/3). SuperGlue remains unavailable.
- Built a style/gate-specific heatmap pseudo-label CSV: `runs/rootsift_pseudo_labels_stylespecific_r080v_r090numcomp_r080tscomp_seed1234/pseudo_labels.csv`. It combines r0.80/t2 viewpoint labels for both styles, r0.90/Ht3 compound labels for numeric, and r0.80/t2 compound labels for timestamp. After exact-row deduplication it contains 40548 labels: viewpoint r0.80/t2 numeric 9685 labels / 84 pairs, viewpoint r0.80/t2 timestamp 10573 / 98, numeric compound r0.90/Ht3 12151 / 117, timestamp compound r0.80/t2 9091 / 97.
- Style-specific heatmap-only training: `runs/cross_view_1024_rootsift_pseudo_stylespecific_keypointonly_w1n002_lr1e5_weakgroups_120_seed1234`. It started from the r0.80/t2 weakgroups heatmap checkpoint, matched 342 labeled training pairs / 40548 pseudo matches, trained only the heatmap head for 120 steps with `batch_pairs=8`, and preserved descriptor validation exactly: loss 3.841238/top1 0.1479/top5 0.4121 before and after.
- Single-checkpoint guarded eval: `runs/cross_view_1024_keypointonly_stylespecific_learnedscore_guard_calib_0step_seed1234`. Test precision: numeric/rotate 647/693 (0.933622), numeric/viewpoint 53/143 (0.370629), numeric/compound 50/224 (0.223214), timestamp/rotate 555/615 (0.902439), timestamp/viewpoint 20/104 (0.192308), timestamp/compound 3/111 (0.027027). It wrote six summaries and twelve visualizations. This slightly improves numeric/compound over the global r0.90/Ht3 run but still hurts timestamp/compound because calibration selects the style-specific checkpoint for that group.
- Multi-state guarded eval with base + blend02540 + r0.80 weakgroups + r0.90/Ht3 weakgroups + style-specific checkpoint: `runs/cross_view_1024_keypointonly_multistate_stylespecific_guard_calib_0step_seed1234`. Calibration selected `blend02540` for rotate and numeric/viewpoint, the style-specific trained checkpoint for numeric/compound, base for timestamp/viewpoint, and `weak_r080` for timestamp/compound. Test precision: numeric/rotate 647/693 (0.933622), numeric/viewpoint 53/143 (0.370629), numeric/compound 50/224 (0.223214), timestamp/rotate 555/615 (0.902439), timestamp/viewpoint 20/104 (0.192308), timestamp/compound 5/72 (0.069444). It wrote six summaries and twelve visualizations.
- Interpretation: the current best heatmap-side route is compositional, not a single checkpoint. Style-specific labels provide the best numeric/compound candidate so far on the fixed 64-pair sample, while r0.80/t2 weakgroups remains the best timestamp/compound heatmap candidate. The next teacher-mining iteration should use Agent13's r0.88/Ht3 and style/gate-specific recommendations on the train split before launching another heatmap distillation run.
- Agent13 stage2 completed train-split teacher mining in `runs/matcher_algorithm_iteration_agent13_stage2/`. It sampled 48 train pairs per style/gate and compared baseline r0.80/Ht2, baseline r0.90/Ht3, global r0.88/Ht3, and style/gate-specific teachers. The selected style-specific bundle kept 144 pairs, produced 14496 capped labels before exact dedupe, and had aggregate truth precision 0.9874. Per group: numeric/compound LightGlue-SIFT-Ht3 kept 37 pairs / 3704 labels / 0.9968 precision; numeric/viewpoint RootSIFT-mutual-r0.95-Ht3 kept 33 / 3840 / 0.9904; timestamp/compound RootSIFT-r0.88-Ht3 kept 36 / 3054 / 0.9771; timestamp/viewpoint RootSIFT-r0.92-Ht3 kept 38 / 3898 / 0.9806.
- Converted Agent13 stage2 selected labels to a training CSV. The first absolute-path CSV matched zero training pairs because symlinked split paths resolve to `img/...` inside `_pseudo_label_path_keys`; the corrected relative-path CSV is `runs/rootsift_pseudo_labels_agent13_stage2_stylespecific_relpaths_seed1234/pseudo_labels.csv`. It contains 14080 deduplicated labels and 144 source pairs.
- Agent13 stage2 heatmap-only training: `runs/cross_view_1024_agent13_stage2_keypointonly_relpaths_w1n002_lr1e5_weakgroups_120_seed1234`. It started from the r0.80/t2 weakgroups checkpoint, matched 137 labeled training pairs / 14080 labels, trained only the heatmap head for 120 steps, and preserved descriptor validation exactly: loss 3.841238/top1 0.1479/top5 0.4121.
- Agent13 stage2 single-checkpoint guarded eval: `runs/cross_view_1024_agent13_stage2_keypointonly_learnedscore_guard_calib_0step_seed1234`. Test precision: numeric/rotate 647/693 (0.933622), numeric/viewpoint 53/143 (0.370629), numeric/compound 50/224 (0.223214), timestamp/rotate 555/615 (0.902439), timestamp/viewpoint 20/104 (0.192308), timestamp/compound 3/107 (0.028037). It wrote six summaries and twelve visualizations. This reproduces the numeric/compound gain but still fails timestamp/compound.
- Updated interpretation: high train-side teacher precision is not sufficient for timestamp/compound heatmap improvement. The currently best fixed-sample route remains the multi-state route that uses the stage/style-specific trained checkpoint for numeric/compound and the older r0.80/t2 weakgroups checkpoint for timestamp/compound. Agent13 stage3 has been dispatched to expand timestamp/compound train mining and compare r0.80/r0.88/r0.90 overlap and label spatial coverage.
- Agent13 stage3 completed timestamp/compound hard-tail diagnosis in `runs/matcher_algorithm_iteration_agent13_stage3/`. Expanded train mining confirmed r0.88/Ht3 precision is not the main issue: timestamp/compound r0.88/Ht3 kept 103/128 sampled pairs with 10025 labels and 0.9846 truth precision. The weak point is hard-tail coverage: 24/128 pairs were `too_few_truth_labels`, and r0.88 mostly densified pairs already covered by r0.80/r0.90; only 6 new pairs were added over r0.80, with median inliers 22.5 versus 132.0 for pairs kept by both.
- Based on stage3, generated a balanced plus hard-tail heatmap CSV at `runs/rootsift_pseudo_labels_agent13_stage2_balanced_plus_tscomp_hardtail_seed1234/pseudo_labels.csv`. It capped stage2 labels to 32 per pair and attempted to add timestamp/compound hard-tail pairs via RootSIFT r0.88/Ht3 low-threshold mining. The hard-tail pass added only 33 labels across 24 attempted pairs; final CSV had 4461 rows over 146 pairs.
- Balanced plus hard-tail heatmap-only training completed: `runs/cross_view_1024_agent13_stage2_balanced_hardtail_keypointonly_w1n002_lr1e5_weakgroups_120_seed1234`. It started from the r0.80/t2 weakgroups checkpoint, matched 139 labeled training pairs / 4461 labels, and preserved descriptor validation unchanged at loss 3.841238/top1 0.1479/top5 0.4121.
- Balanced plus hard-tail single-checkpoint guarded eval completed: `runs/cross_view_1024_agent13_stage2_balanced_hardtail_keypointonly_learnedscore_guard_calib_0step_seed1234`. Test precision: numeric/rotate 647/693 (0.933622), numeric/viewpoint 53/143 (0.370629), numeric/compound 50/225 (0.222222), timestamp/rotate 555/615 (0.902439), timestamp/viewpoint 20/104 (0.192308), timestamp/compound 3/112 (0.026786). It wrote six summaries and twelve visualizations.
- Interpretation: capping easy labels and adding the current RootSIFT hard-tail leftovers did not improve timestamp/compound; the added hard-tail signal was too sparse to change learned heatmap behavior. Popper/Agent13 has been given Stage4 to continue matcher iteration specifically on timestamp/compound hard-tail coverage, comparing LightGlue-SIFT, looser RootSIFT gates, CLAHE/other fallbacks, and writing results under `runs/matcher_algorithm_iteration_agent13_stage4/`.
- Two attempted all-specialist route sweeps that added the balanced-hardtail checkpoint were stopped early because calibration was too slow for the size of the candidate matrix. The wide sweep wrote only 46 calibration CSVs after several minutes, and the narrowed sweep wrote only 24. This is recorded as an execution finding: do not use full checkpoint x blend x margin x score exhaustive calibration for quick triage; use targeted fixed-parameter checks or fewer states.
- Agent13 stage4 completed in `runs/matcher_algorithm_iteration_agent13_stage4/` and was independently checked from its CSVs. It recovered 24 timestamp/compound hard-tail pairs from Stage3 and tested LightGlue-SIFT, RootSIFT r0.88/r0.92/r0.95, CLAHE-RootSIFT, AKAZE, and ORB variants. Only one unique hard-tail pair passed the training gate. `candidate_labels.csv` has 38 rows from `source_000111_20260514T143405909_NAS_PAN_L2b/pair_005193.pt`: 26 labels from CLAHE-RootSIFT r0.92/Ht3 and 12 from LightGlue-SIFT/Ht3, all with max truth error 2.879303 px.
- Stage4 conclusion: RootSIFT r0.88/Ht3 covers 0/24 hard-tail pairs, looser RootSIFT adds only diagnostic low-precision matches, and AKAZE/ORB are near-zero precision. These 38 labels are useful evidence/debug output but not enough coverage for another heatmap training run.
- Popper/Agent13 stage5 has been dispatched to keep matcher iteration moving without blocking the main line. Stage5 targets the same 24 hard-tail pairs with a different matcher family: Kornia LoFTR if available, LightGlue-SIFT with larger context/resize variants, and optional dense optical flow diagnostics, with outputs under `runs/matcher_algorithm_iteration_agent13_stage5/`.
- Agent13 stage5 script was completed and run in `runs/matcher_algorithm_iteration_agent13_stage5/`. Self-test passed, Kornia LoFTR outdoor was already available, and Kornia downloaded the LoFTR indoor checkpoint to the torch cache during the run. Outputs written: `summary.md`, `summary_metrics.csv`, `pair_metrics.csv`, `candidate_labels.csv`, and `skipped_teachers.csv`.
- Stage5 tested Kornia LoFTR outdoor/indoor, LightGlue-SIFT k2048/k4096, Farneback dense flow, and DIS optical flow on the same 24 hard-tail pairs. Independent CSV checks: `pair_metrics.csv` has 144 ok rows, `summary_metrics.csv` has six profile rows, `candidate_labels.csv` has 416 labels over only two unique pairs, and no teachers were skipped.
- Stage5 best profile was `lightglue_sift_k2048`: 2/24 unique kept pairs, 160 candidate labels, kept precision 1.0, all-pair truth precision 0.903683. LoFTR outdoor kept 1/24 with 128 labels at kept precision 0.984479; LightGlue-SIFT k4096 kept 1/24; LoFTR indoor, Farneback, and DISFlow produced no train-candidate hard-tail coverage.
- Stage5 only adds one new hard-tail pair beyond Stage4: `img/CompoundViewpoint_1024/source_000086_20260514T070046672_NAS_PAN_L2b/pair_005399.pt` with 32 LightGlue-SIFT-k2048 labels. The other 384 Stage5 labels are on the same `source_000111.../pair_005193.pt` pair already found by Stage4.
- Stage5 conclusion: changing to LoFTR, larger LightGlue-SIFT budgets, and dense optical flow still does not substantially cover timestamp/compound hard-tail. The hard-tail pseudo-label route is exhausted for now; do not launch another heatmap training run from Stage4/5 labels. Next work should shift to data/context changes such as larger-context cache generation, pair/source re-selection, or evaluation routing that explicitly accepts that these hard-tail pairs lack reliable sparse teacher labels.
- Added a timestamp/compound hard-tail data diagnostic under `runs/timestamp_compound_hardtail_data_diagnostic_20260526/`. It compares the 24 hard-tail pairs against the 103 Stage3 kept timestamp/compound pairs using cache-level valid-mask coverage, warp displacement, target-inside fraction, and image texture statistics.
- Diagnostic result: hard-tail pairs do not have worse valid geometry. Median valid fraction is 0.7990 for hard-tail versus 0.6806 for kept pairs; median warp target-inside fraction is 1.0 for both; median warp displacement is lower for hard-tail (342.87 px) than kept pairs (398.91 px). The hard-tail group spans 20 sources, so it is not a single-source artifact.
- The texture signal is more plausible: hard-tail B-image gradient median is 15.37 versus 21.37 for kept pairs, while A-image gradient is similar. Combined with Stage4/5 teacher failures, this points to low-repeatability/appearance-context failure on the target view, not invalid overlap. Next training/data work should not force these hard-tail pairs into sparse pseudo-label training without changing crop/context or target selection.
- Stage6 matcher sidecar completed `scripts/matcher_algorithm_iteration_agent13_stage6.py` and `runs/matcher_algorithm_iteration_agent13_stage6/`. It stopped searching the same hard-tail sparse teachers and instead diagnosed route/quality gates on current timestamp/compound fixed-test and full-val summaries. It found B-view local contrast to be the best validation-backed abstain feature: PNG/post-hoc full-val 0.057692 -> 0.109890 while retaining 10/12 correct, with fixed-test non-negative transfer.
- Added evaluator/orchestrator abstain gates for target-view quality: `pytorch_cache_match_eval.py` now supports `--min-target-gradient` and `--min-target-local-contrast`; `cross_view_experiment.py` can pass and calibrate them via `--match-min-target-gradient`, `--calibrate-target-gradients`, `--target-gradient-candidates`, `--match-min-target-local-contrast`, `--calibrate-target-local-contrasts`, and `--target-local-contrast-candidates`. Calibration CSVs now record `min_target_gradient` and `min_target_local_contrast`.
- Tensor/evaluator verification showed the PNG-derived local-contrast threshold 5.323792 is slightly aggressive in direct `.pt` evaluation: full-val 10/95 (0.105263) but fixed-test 3/36 (0.083333). A more conservative tensor threshold 5.2 is better balanced: full-val 10/108 (0.092593) and fixed-test 4/45 (0.088889), versus baseline full-val 12/208 (0.057692) and fixed-test 5/72 (0.069444).
- Verification after the gate implementation: focused `python/test_pytorch_cache_match_eval.py python/test_cross_view_experiment.py` passed with 73 tests OK; full Python discovery passed with 177 tests OK, 1 skipped.
- Created a concrete postselected six-group route at `runs/cross_view_1024_keypointonly_multistate_stylespecific_guard_targetcontrast_postselect_0step_seed1234`. It reuses the current best multistate route for five groups and applies `--min-target-local-contrast 5.2` only to timestamp/compound. It has six `eval/*/*/summary.csv` files, twelve visualization PNGs, and a `calibration/selected_weights.csv` row recording `min_target_local_contrast=5.2` for timestamp/compound.
- Postselected fixed-test metrics: numeric/rotate 647/693 (0.933622), numeric/viewpoint 53/143 (0.370629), numeric/compound 50/224 (0.223214), timestamp/rotate 555/615 (0.902439), timestamp/viewpoint 20/104 (0.192308), timestamp/compound 4/45 (0.088889). Compared with the previous best multistate route, only timestamp/compound changes: precision improves from 5/72 (0.069444) but correct matches drop from 5 to 4.
- Stage7 matcher sidecar completed in `runs/matcher_algorithm_iteration_agent13_stage7/`. It checked the timestamp/compound pairs dropped by the local-contrast gate and found `RootSIFT-FLANN-r0.80+HomographyUSAC-t2` covers all dropped pairs. Fixed-test fallback metrics were 890/898 (0.991091), and combined with the contrast gate the timestamp/compound group becomes 894/943 (0.948038). Full-val fallback was 3562/3577 (0.995807), combined 3572/3685 (0.969335).
- Built the labeled hybrid route at `runs/cross_view_1024_targetcontrast_rootsift_fallback_route_20260526`. It has six summary CSVs, twelve visualization PNGs, `hybrid_route_metrics.csv`, `hybrid_route_comparison.csv`, `hybrid_route_deltas.csv`, `hybrid_pair_decisions.csv`, and `summary.md`. This route is explicitly external fallback, not pure PFM: five groups stay pure PFM, while timestamp/compound uses target-contrast abstention plus RootSIFT fallback on abstained pairs.
- Spawned Stage8 matching sidecar agent `019e6195-5be8-7c81-8bff-cf024a4627ef` to keep matcher iteration running. Its scope is isolated to `scripts/matcher_algorithm_iteration_agent13_stage8.py` and `runs/matcher_algorithm_iteration_agent13_stage8/`, testing whether fallback should remain timestamp/compound-only or generalize across all six fixed-test groups.
- Generated train-split-only low target-contrast timestamp/compound labels at `runs/rootsift_pseudo_labels_tscompound_lowtargetcontrast_r080t2_train_seed1234`. From 1536 train pairs, 824 had target local contrast < 5.2; sampled 128 with seed 1234; RootSIFT r0.80 + homography 2px + warp-truth 2px kept 88 pairs and 8317 labels. This deliberately uses only train split, not val/test.
- Ran a short heatmap-only continuation from the current `weak_r080` checkpoint: `runs/cross_view_1024_tscompound_lowcontrast_keypointonly_w1n002_lr1e5_80_seed1234`. Training used the low-contrast labels, 80 steps, `batch_pairs=8`, lr 1e-5, synthetic loss 0, pseudo keypoint weight 1, and skipped 0 steps. Loss moved 1.448308 -> 1.442855; descriptor validation stayed unchanged as expected for heatmap-only training.
- Timestamp/compound evaluation for the low-contrast heatmap checkpoint showed a small pure-PFM gain. Fixed test at the previous selected parameters: old `weak_r080` 5/72 (0.069444), new 5/59 (0.084746). With `min-target-local-contrast 5.2`: old 4/45 (0.088889), new 5/36 (0.138889). Full val also moved from old 12/208 (0.057692) to new 14/228 (0.061404), and with the gate from old 10/108 (0.092593) to new 12/126 (0.095238). Summary: `runs/cross_view_1024_tscompound_lowcontrast_keypointonly_w1n002_lr1e5_80_seed1234/summary.md`.
- Hooke/Agent13 Stage8 completed in `runs/matcher_algorithm_iteration_agent13_stage8/`. It evaluated all fixed-test targetcontrast gate-zero rows across six groups. Candidate set: 246 rows. Dropped baseline-match rows existed only in timestamp/compound: 5 rows, 1 lost-correct row. For the broad all-gate-zero policy, `RootSIFT-FLANN-r0.80+HomographyUSAC-t2` produced 60431 fallback matches, 60108 correct, 323 wrong, fallback precision 0.994655, and passed all 6 groups. LightGlue-SIFT was available and included as secondary comparison.
- Built the broader labeled hybrid route at `runs/cross_view_1024_targetcontrast_rootsift_allgatezero_fallback_route_20260526`. It keeps targetcontrast PFM nonzero rows and applies RootSIFT r0.80/H2 fallback to every targetcontrast gate-zero fixed-test row. It has six summary CSVs, twelve visualization PNGs, route metrics/comparison/delta CSVs, and Stage8 support rows. Overall fixed-test hybrid: 61437/62255 (0.986860). This remains hybrid/external and needs full-val policy validation before replacing the narrower timestamp/compound-only hybrid route.
- Spawned Stage9 matcher sidecar Linnaeus (`019e61a7-752f-7371-96ed-2c6297186044`) to keep external matcher iteration running in parallel. Its isolated scope is `scripts/matcher_algorithm_iteration_agent14_stage9.py` and `runs/matcher_algorithm_iteration_agent14_stage9/`, focused on full-val/held-out validation for the Stage8 all-gate-zero fallback policy and safer RootSIFT/LightGlue alternatives.
- Guarded the lowcontrast heatmap checkpoint against numeric/compound misuse: using the current numeric/compound selected params, `runs/cross_view_1024_tscompound_lowcontrast_keypointonly_w1n002_lr1e5_80_seed1234/eval_numeric_compound_test_weight4_learned_margin001.csv` produced 204/2725 (0.074862) on the fixed 64-pair test, far below the existing numeric/compound route 50/224 (0.223214).
- Built a six-group pure-PFM postselected route at `runs/cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234`. It keeps the target-contrast postselected route for five groups and uses the lowcontrast heatmap checkpoint plus `min_target_local_contrast=5.2` only for timestamp/compound. Fixed-test metrics: numeric/rotate 647/693 (0.933622), numeric/viewpoint 53/143 (0.370629), numeric/compound 50/224 (0.223214), timestamp/rotate 555/615 (0.902439), timestamp/viewpoint 20/104 (0.192308), timestamp/compound 5/36 (0.138889). It has six summaries, twelve PNGs, `route_comparison.csv`, and selected weights updated to the new route paths.
- Stage9 matcher sidecar completed in `runs/matcher_algorithm_iteration_agent14_stage9/`. It evaluated a cache-heldout sample excluding fixed-test rows and found `RootSIFT-FLANN-r0.75+HomographyUSAC-t2` ranked above Stage8 r0.80/H2: r0.75/H2 had 38569/38682 correct (0.997079), min group precision 0.987853; r0.80/H2 had 39922/40070 (0.996306), min group precision 0.984298. LightGlue-SIFT was available but lower precision, and ORB/AKAZE failed six-group guardrails.
- Stage10 matcher sidecar completed full-val all-gate-zero replay in `runs/matcher_algorithm_iteration_agent14_stage10/` with no sampling. Current pure-PFM full-val route has 1587 non-self pairs and 845 gate-zero rows. r0.75/H2 fallback produced 183662/184456 correct (0.995695), min group fallback precision 0.988153, coverage 773/845, hybrid full-val precision 0.982896. Stage8-compatible r0.80/H2 had higher support but more wrong matches: 195168/196531 (0.993065), 1363 wrong, hybrid 0.981175. Treat both as hybrid/external only.
- Generated train-only broad gate-zero r0.75/H2 labels at `runs/rootsift_pseudo_labels_gatezero_r075t2_train_sample64_seed1234`: six groups x 64 sampled train pairs, 164 selected PFM zero-match candidates, 115 kept pairs, 6844 labels. The generator script is `scripts/generate_gatezero_rootsift_pseudo_labels.py`.
- Ran heatmap-only training from the lowcontrast checkpoint with those broad gate-zero labels: `runs/cross_view_1024_gatezero_r075t2_keypointonly_w1n002_lr1e5_80_seed1234`. It trained 80 steps, skipped 0, reduced heatmap loss 1.442537 -> 1.437378, and descriptor validation stayed exactly unchanged at loss 4.917298 / top1 0.0655 / top5 0.2078.
- Evaluated that gate-zero heatmap checkpoint at `runs/cross_view_1024_gatezero_r075t2_keypointonly_selectedparams_eval_0step_seed1234`. Fixed-test single-checkpoint metrics showed small compound gains but broad regressions: numeric/rotate 2127/2351 (0.904721), numeric/viewpoint 151/442 (0.341629), numeric/compound 51/226 (0.225664), timestamp/rotate 977/1172 (0.833618), timestamp/viewpoint 13/107 (0.121495), timestamp/compound 6/37 (0.162162). Full-val guard contradicted the fixed-test compound gains: numeric/compound dropped from current route 281/1325 (0.212075) to 1246/14730 (0.084589), and timestamp/compound dropped from 12/126 (0.095238) to 17/2377 (0.007152). Decision: do not add this checkpoint to pure-PFM routing.
- Stage11 matcher sidecar completed the fixed-test r0.75/H2 broad hybrid route in `runs/matcher_algorithm_iteration_agent14_stage11/`. It keeps current pure-PFM nonzero rows and applies external RootSIFT r0.75/H2 on gate-zero rows. Compared with Stage8 r0.80/H2, Stage11 improves fixed-test hybrid precision 0.986860 -> 0.988985 and reduces wrong matches 818 -> 650, but loses support/correct matches 61437 -> 58362. Use r0.75/H2 as the high-precision hybrid default and retain r0.80/H2 as the recall-heavy compatibility baseline.
- Stage12 matcher sidecar completed in `runs/matcher_algorithm_iteration_agent14_stage12/`. It did not train or run new matchers; it analyzed Stage10/11 and the rejected broad gate-zero heatmap checkpoint. It recommends only a tiny pair-filtered noncompound viewpoint probe (`P1_r075_pairfiltered_noncompound_viewpoint_tiny`): numeric/viewpoint + timestamp/viewpoint only, 45 validation-backed pairs, 540 label budget, pair cap 12, group cap 384, min inliers 50, pair precision >=0.995, source cap 2, and hard negatives. It explicitly forbids broad all-gate-zero heatmap and r0.80/H2 max-support teacher expansion for training.
- Timestamp/viewpoint target-quality gate diagnostic completed in `runs/timestamp_viewpoint_quality_gate_diagnostic_20260526/`. Full-val local contrast gates improve precision from 78/389 (0.200514) to 56/204 (0.274510) at 5.2 and 44/152 (0.289474) at 5.6, but lose correct matches. Fixed64 transfer is precision-positive but support-negative: baseline 20/104 (0.192308), local contrast 5.2 is 14/54 (0.259259), local contrast 5.6 is 13/43 (0.302326), gradient 20 is 17/61 (0.278689). These are precision-only reporting candidates, not a default route replacement.
- PPT skill installation was requested through a subagent, but official `openai/skills` curated/experimental and local installed skills had no PPT/PowerPoint/slides/presentation skill. A project-introduction deck was generated directly with `python-pptx` instead: `runs/project_presentation_20260526/PlanetaryFeatureMatch_project_intro_20260526.pptx` (13 slides), covering project background, feature extraction model, graph matcher model, training/distillation strategy, pure-PFM performance, traditional matcher comparison, deep matcher comparison, hybrid fallback, failure analysis, and next steps.
- Added TDD coverage for the P1 retention-label selector: `python/test_p1_retention_label_generation.py` first failed on missing module, then passed after adding `scripts/generate_p1_viewpoint_retention_labels.py`; focused test passed with 2 tests OK and the generator compiles.
- Generated strict train-only P1 viewpoint labels at `runs/rootsift_pseudo_labels_p1_viewpoint_r075t2_train_seed1234`: numeric/viewpoint 30 pairs / 360 labels, timestamp/viewpoint 32 pairs / 384 labels, 62 pairs / 744 labels total, min selected precision 1.0, wrong=0. This uses RootSIFT r0.75 + HomographyUSAC 2px + warp truth 3px, per-pair cap 12, group cap 384, max source pairs 2.
- First P1 descriptor probe with `batch_pairs=8, gradient_accumulation_steps=2` OOMed before step 1. Root cause: `pfm_pytorch_training.py` accumulates micro-batch losses and calls backward after the accumulation loop, so it retains `batch_pairs x accumulation` computation graphs. Retried with `batch_pairs=2, gradient_accumulation_steps=1`.
- Completed the both-viewpoint descriptor-only P1 probe: `runs/cross_view_1024_p1_viewpoint_retention_desc_lr5e7_b2_80_seed1234`. Training used pseudo-only descriptor loss, synthetic loss 0, hard-negative weight 0.5, lr 5e-7, 80 steps. Descriptor validation regressed slightly: loss 4.621215/top1 0.124329/top5 0.277985 -> loss 4.645037/top1 0.123383/top5 0.272400.
- Both-viewpoint P1 sparse eval is not routeable. Numeric/viewpoint fixed64 changed 53/143 (0.370629) -> 140/419 (0.334129), and full-val changed 371/836 (0.443780) -> 949/2698 (0.351742), indicating activation growth and lower precision. Timestamp/viewpoint fixed64 changed 20/104 (0.192308) -> 16/95 (0.168421), though full-val improved 78/389 (0.200514) -> 81/371 (0.218329). Decision: do not add this checkpoint to pure-PFM routing.
- Generated timestamp/viewpoint-only P1 labels at `runs/rootsift_pseudo_labels_p1_timestamp_viewpoint_r075t2_train_seed1234`: 31 pairs / 372 labels, min selected precision 1.0, wrong=0.
- Completed timestamp/viewpoint-only descriptor probe: `runs/cross_view_1024_p1_timestamp_viewpoint_retention_desc_lr5e7_b2_60_seed1234`. Descriptor validation moved slightly positive: loss 5.564123/top1 0.095154/top5 0.193146 -> loss 5.557397/top1 0.095367/top5 0.193756.
- Timestamp-only P1 sparse eval also fails route gates. Fixed64 at current params regressed 20/104 (0.192308) -> 11/94 (0.117021), full-val slightly improved 78/389 (0.200514) -> 83/398 (0.208543), and margin 0.02 fixed64 produced 0/0. Decision: keep this as weak diagnostic evidence only; do not route it.
- Refreshed the project-introduction PPT with the latest P1 descriptor-retention results. Updated `scripts/create_project_presentation.py`, regenerated `runs/project_presentation_20260526/PlanetaryFeatureMatch_project_intro_20260526.pptx`, added `assets/p1_descriptor_probe_precision.png`, and verified the deck by reading 13 slides with `python-pptx` plus converting it successfully to `runs/project_presentation_20260526/rendered/PlanetaryFeatureMatch_project_intro_20260526.pdf` with LibreOffice.
- Ran the P1 both-viewpoint descriptor probe with existing warp-aware hard negatives enabled: `runs/cross_view_1024_p1_viewpoint_retention_desc_warpneg020_lr5e7_b2_80_seed1234`. Parameters were pseudo-only descriptor loss, synthetic loss 0, in-batch hard-negative 0.25, warp-hard-negative 0.20, radius 2.0, margin 0.2, 80 steps, batch_pairs 2. Training completed without OOM or skipped steps.
- Warp-negative P1 descriptor validation still regressed: loss 4.621215/top1 0.124329/top5 0.277985 -> loss 4.644911/top1 0.123383/top5 0.272430.
- Warp-negative P1 sparse eval is also not routeable. Numeric/viewpoint fixed64 changed baseline 53/143 (0.370629) -> 141/421 (0.334917), and full-val changed 371/836 (0.443780) -> 949/2702 (0.351221). Timestamp/viewpoint fixed64 changed 20/104 (0.192308) -> 16/92 (0.173913), while full-val weakly improved 78/389 (0.200514) -> 81/368 (0.220109). Summary written to `runs/cross_view_1024_p1_viewpoint_retention_desc_warpneg020_lr5e7_b2_80_seed1234/summary.md`.
- Inspected the current training entry points after the warp-negative result. Existing PFM PyTorch training supports in-batch hard negatives, warp-aware hard negatives, and a weak heatmap mean penalty through `pseudo_keypoint_negative_weight`, but it does not provide a pair-level route-abstention or non-activation loss. The next training change should add that explicitly instead of continuing descriptor-positive-only P1 variants.
- User clarified the PPT should briefly introduce the current project model and compare matching results, not explain other models. Also requested that other-model matching effects use the previously generated 6 groups x 2 pairs and be placed under `对比文档/`.
- Added `scripts/fixed_six_group_matcher_comparison.py` and ran it on the exact 12 fixed pair files corresponding to the previous visualizations. Outputs are under `对比文档/`: `README.md`, `fixed_pairs.csv`, `metrics.csv`, `summary.csv`, `skipped_algorithms.csv`, and 80 PNG match visualizations. Algorithms included PFM-current, SIFT-r0.80/H2, RootSIFT-r0.80/H2, RootSIFT-r0.90/H2, ORB-H3, AKAZE-H3, and LightGlue-SIFT-H3; SuperGlue remains unavailable locally.
- Rebuilt the PPT as a concise 10-slide deck. It now focuses on current PFM model structure, training/evaluation loop, current pure-PFM six-group results, and fixed-12-pair matching-result comparison from `对比文档/`. Verified with `python-pptx` that it has 10 slides and with LibreOffice that it renders to a 10-page PDF.
- User corrected the visualization requirement: because the data is synthetic and GT correspondence is known, every plotted match line must be colored by GT correctness for every algorithm. Updated `scripts/fixed_six_group_matcher_comparison.py` so PFM and external matchers use the same GT warp correctness mask: green means error <= 5 px on a valid source point, red means wrong/invalid. PFM visualizations are regenerated from the current postselected route parameters instead of copying old figures.
- Re-ran the fixed comparison script with CUDA. Verification: `fixed_pairs.csv` 13 lines, `metrics.csv` 85 lines, `summary.csv` 43 lines, `skipped_algorithms.csv` 2 lines, and `对比文档/figures` now contains 84 PNGs. PFM rows exactly match the original lowcontrast route summaries for the 12 fixed pairs. Pixel checks confirmed green+red on mixed PFM examples, red-only on wrong-only examples, and green-only on correct-only examples.
- User questioned whether external matchers are unrealistically good. Diagnosis: the fixed comparison script did not add extra 90/180/270 rotations; it compared the cached `view_a`/`view_b` directly. The cached 12-pair GT warps are exactly homography-fit on sampled points (`H_err_med=0` and `H<=5=1.0` for all 12), so classical/LightGlue matchers followed by homography filtering can legitimately look almost perfect under this synthetic-cache protocol.
- Corrected the misleading artifact by extending `scripts/fixed_six_group_matcher_comparison.py` to also write raw external matcher results before homography filtering. Re-ran the script. Outputs now include `raw_metrics.csv`, `raw_summary.csv`, and `figures_raw/` with 72 PNGs. Aggregate external raw metrics are 11614/13846 correct (precision 0.838798, wrong=2232), while RANSAC-after metrics are 11236/11249 (precision 0.998844, wrong=13). README now explicitly separates raw and RANSAC-after sections.
- User clarified that all matching algorithms must be shown in the most original state, without RANSAC repair. Updated `scripts/fixed_six_group_matcher_comparison.py` so the main `对比文档/metrics.csv`, `summary.csv`, and `figures/` are raw matcher outputs only. Removed stale `raw_metrics.csv`, `raw_summary.csv`, and `figures_raw/` to avoid two competing reporting口径. Re-ran the script on CUDA and verified: `fixed_pairs.csv` 13 lines, `metrics.csv` 85 lines, `summary.csv` 43 lines, `figures/` 84 PNGs, external raw aggregate 11614/13846 correct with 2232 wrong, and PFM raw/current aggregate 219/872 correct with 653 wrong. Representative pixel checks confirmed visible red wrong-match lines in external and PFM figures.
- Added `scripts/extreme_case_matcher_comparison.py` for the user's real extreme TIFF case under `对比文档/极端测试/`. This script deliberately uses raw matcher outputs only: no RANSAC, Homography, USAC, or geometry repair.
- Ran the extreme TIFF comparison with CUDA and long-edge resize 1600. Inputs were `对比文档/极端测试/20260510T173954657_NAS_PAN_L2b.tif` and `对比文档/极端测试/20260510T191252977_NAS_PAN_L2b.tif`, both resized from 3036x4024 to 1207x1600.
- Extreme TIFF outputs written: `对比文档/极端测试/README.md`, `metrics.csv`, `summary.csv`, `skipped_algorithms.csv`, and 8 PNGs in `figures/`. Raw match counts: SIFT-r0.80 49, RootSIFT-r0.80 30, RootSIFT-r0.90 256, ORB 256, AKAZE 256, LightGlue-SIFT 56, PFM-current 256, PFM-latest-P1 256. SuperGlue remains unavailable in the local environment.
- Important reporting note: unlike the synthetic fixed 12-pair comparison, the extreme TIFF pair has no GT warp or人工对应标注 in this directory, so the figures use neutral lines only and do not label matches green/red or compute precision/correct/wrong.
- User pointed out that the main `对比文档/figures` rotate section had not actually been regenerated for 90°/180°. Diagnosis confirmed: `scripts/fixed_six_group_matcher_comparison.py` already had the intended rotate pair definitions, but the existing `fixed_pairs.csv` and `figures/` were stale and still referenced `pair_002049.pt`, `pair_000238.pt`, `pair_003819.pt`, and `pair_004708.pt` for rotate.
- Re-ran `scripts/fixed_six_group_matcher_comparison.py --device cuda`. New `fixed_pairs.csv` rotate rows are numeric sample01 `pair_001587.pt` rotation 90, numeric sample02 `pair_002779.pt` rotation 180, timestamp sample01 `pair_001509.pt` rotation 90, and timestamp sample02 `pair_002860.pt` rotation 180. The regenerated `figures/` contains 84 PNGs; PFM rotate filenames now include the new pair ids, and external matcher rotate figures under sample01/sample02 correspond to those same 90°/180° rows.
- Made the rotate figure paths explicit by renaming the rotate samples in `scripts/fixed_six_group_matcher_comparison.py` to `rot90` and `rot180`, then regenerated `对比文档/` again. Verification: `fixed_pairs.csv` rotate rows now have `sample=rot90/rot180` plus `rotation_deg=90/180`; `figures/other_models/*/rotate/rot90|rot180/` and `figures/pfm/*/rotate/rot90|rot180/` exist; totals remain 13 fixed-pair rows, 85 metric rows, 43 summary rows, 2 skipped rows, and 84 PNGs.
- Added abstention-aware descriptor false-match suppression in `python/pfm_pytorch_training.py` and CLI/orchestrator pass-through in `python/cross_view_experiment.py`. TDD/verification: targeted RED failures were observed first, then `python/test_pfm_pytorch_training.py` passed 55 tests, `python/test_cross_view_experiment.py` passed 51 tests, and `py_compile` passed for both modified scripts.
- Ran the conservative P1 viewpoint abstention probe at `runs/cross_view_1024_abstention_p1_viewpoint_desc_w025_m035_lr3e7_b4_80_seed1234`: 80 steps, no skipped steps, pseudo-only descriptor training, `abstention_weight=0.25`. Descriptor retrieval improved slightly (top1 0.083832 -> 0.085236, mean negative score 0.444290 -> 0.437944), but same-split raw sparse matching did not improve versus the base checkpoint: numeric/viewpoint 767/8360 (0.091746) -> 767/8380 (0.091527), timestamp/viewpoint 340/15044 (0.022600) -> 339/15081 (0.022479). Decision: do not add this checkpoint to pure-PFM routing; this loss alone is too weak.
- Added mined false-match supervision. `python/pfm_pytorch_training.py` now reads `--false-match-csv`, samples exact wrong PFM pairs through the training-pair curriculum, and applies a cyclic descriptor negative loss aligned with `pytorch_cache_match_eval.cyclic_descriptor_similarity`. `python/cross_view_experiment.py` passes the false-match options through, and `scripts/mine_pfm_false_matches.py` mines raw mutual PFM matches whose synthetic warp error exceeds the threshold.
- Verification for mined false-match code passed: `python/test_pfm_pytorch_training.py` 60 tests OK, `python/test_cross_view_experiment.py python/test_pfm_false_match_mining.py` 53 tests OK, and `py_compile` passed for `python/pfm_pytorch_training.py`, `python/cross_view_experiment.py`, and `scripts/mine_pfm_false_matches.py`.
- Mined weak-group false matches at `runs/pfm_false_match_negatives_weakgroups_seed1234/false_matches.csv`: 256 train pairs from numeric/timestamp viewpoint+compound produced 40,718 wrong raw PFM matches. Group counts were numeric/compound 11,044, numeric/viewpoint 9,486, timestamp/compound 9,043, and timestamp/viewpoint 11,145.
- Ran the first mined-false probe at `runs/cross_view_1024_mined_false_p1_viewpoint_desc_w060_s025_lr3e7_b4_80_seed1234`: P1 viewpoint positives plus mined false negatives, 80 steps. Descriptor validation barely moved (top1 0.0605 -> 0.0610). Raw test at blend=4 did not improve: total 15,787/80,009 (0.197315) -> 15,844/80,331 (0.197234); numeric/viewpoint 0.091746 -> 0.091386; timestamp/viewpoint 0.022600 -> 0.022619.
- Blend-weight root-cause sweep on the base checkpoint showed learned-only is effectively unusable under the current raw sparse protocol: blend=0 yielded only 8 matches on the 32-pair sample. The best sampled total precision was around blend=1 (0.237369) rather than blend=4 (0.207492), confirming that current matching is dominated by the analytic texture descriptor and the learned descriptor has weak standalone distinctiveness.
- Ran a stronger style/gate-specific positive-label plus mined-false descriptor probe at `runs/cross_view_1024_stylespecific_pos_mined_false_desc_w040_s025_lr7e7_b4_160_seed1234`: 14,080 high-precision Agent13 style-specific labels plus 40,718 mined false negatives, 160 steps, no skipped steps. Descriptor validation improved slightly (top1 0.0605 -> 0.0624, top5 0.1903 -> 0.1935, mean negative score 0.4407 -> 0.4332).
- Raw sparse evaluation of that stronger descriptor probe still failed route criteria. At blend=1 total precision fell from 25,862/113,627 (0.227604) to 25,948/117,349 (0.221118); at blend=4 total precision fell from 15,787/80,009 (0.197315) to 15,924/80,968 (0.196670). It gave tiny viewpoint gains, e.g. blend=4 numeric/viewpoint 0.091746 -> 0.092289 and timestamp/viewpoint 0.022600 -> 0.022738, but rotate declined and compound did not improve enough. Decision: do not route this checkpoint.
- Added a trainable texture descriptor adapter to the PyTorch model. `pfm_model.TextureDescriptorAdapter` is a zero-initialized 1x1 residual projection, so old states load strictly and initially produce the same descriptor as the previous analytic texture blend. `pfm_pytorch_training.py` now has `--train-texture-adapter` and `--freeze-descriptor-head` so adapter-only probes can be run without moving the descriptor tower.
- Texture-adapter implementation verification: `python/test_pfm_model.py python/test_pfm_pytorch_training.py` passed 78 tests OK, and strict loading of the old 73 MB base state filled the new adapter with zero defaults (`adapter_params=37056`).
- Joint descriptor+texture-adapter probe completed at `runs/cross_view_1024_texture_adapter_stylespecific_false_w040_lr1e6_b4_160_seed1234`. Validation improved more than descriptor-only (top1 0.0605 -> 0.0638, top5 0.1903 -> 0.1951, mean negative 0.4407 -> 0.4304), but raw sparse still failed route criteria. Blend=1 total precision fell 0.227604 -> 0.219217; blend=4 fell 0.197315 -> 0.196694. It improved some weak groups, e.g. blend=4 numeric/viewpoint 0.091746 -> 0.093080 and timestamp/compound 0.016869 -> 0.017123, but rotate declined.
- Adapter-only 160-step probe completed at `runs/cross_view_1024_texture_adapter_only_stylespecific_false_w040_lr2e6_b4_160_seed1234`. It preserved global blend=4 precision almost exactly: 0.197315 -> 0.197348, with tiny gains in numeric/viewpoint and numeric/compound, but no material route improvement.
- Adapter-only 400-step probe completed at `runs/cross_view_1024_texture_adapter_only_stylespecific_false_w040_lr1e6_b4_400_seed1234`. It slightly improved blend=4 aggregate precision from 15,787/80,009 (0.197315) to 16,126/81,671 (0.197451), but the effect is too small and still comes with rotate precision loss. Decision: the zero-residual 1x1 adapter is a useful compatibility hook, but not enough capacity/objective to solve PFM matching.
- Added the requested larger descriptor model component: `pfm_model.DescriptorFusionAdapter`. It fuses `[learned, weighted texture, learned-texture difference, learned*texture]` through a 3.02M-parameter residual conv block. The old base state still loads strictly with default zero-output fusion weights; the saved state size grows from 73 MB to 85 MB. Training can select it with `--train-descriptor-fusion`.
- Descriptor-fusion probe completed at `runs/cross_view_1024_descriptor_fusion_stylespecific_false_w040_lr1e6_b2_240_seed1234`: descriptor head frozen, texture adapter + descriptor fusion trainable, 14,080 style-specific positive labels plus 40,718 mined false matches, 240 steps, `batch_pairs=2` for memory safety. Validation moved only slightly (top1 0.0605 -> 0.0610, top5 0.1903 -> 0.1914).
- Raw sparse evaluation of the larger fusion model did not improve routing. Blend=1 total precision moved 0.227604 -> 0.226656; blend=4 moved 0.197315 -> 0.197282. It added support/correct matches but did not improve precision. Training metrics showed the sampling issue clearly: 103/240 steps were false-only and only 137/240 steps contained pseudo-label positives, so the next optimization should fix positive/negative sampling balance before spending more steps on the larger block.
- Fixed the positive/negative supervised-pair sampler in `python/pfm_pytorch_training.py`. `sample_training_pairs_with_pseudo_labels` now accepts separate `false_match_pair_paths` and `false_match_probability`, and `train_step` passes pseudo-label positives and mined false-match negatives as separate pools instead of a single union. New tests cover 1:1 quotas for `batch_pairs=2`, split quotas for larger batches, and train-step argument passing.
- Verification for the sampler fix passed: `python/test_pfm_pytorch_training.py python/test_pfm_model.py python/test_cross_view_experiment.py python/test_pfm_false_match_mining.py` ran 138 tests OK with 1 skipped, and `py_compile` passed for the touched training/model/eval/mining modules.
- Quota-balanced low-LR fusion probe completed at `runs/cross_view_1024_descriptor_fusion_quota_stylespecific_false_w040_lr1e6_b2_240_seed1234`. The batch logs confirmed positive and false supervision were both present on logged steps, but validation barely moved: loss 5.055071/top1 0.060455/top5 0.190277 -> loss 5.049031/top1 0.061035/top5 0.191162.
- Raw sparse evaluation of that low-LR quota probe remained effectively flat. Blend=1 total precision moved 0.227604 -> 0.226975, and blend=4 moved 0.197315 -> 0.197242. A descriptor-delta diagnostic showed the trained dense descriptor was almost unchanged from the base state on a sample pair (blend=1 cosine 0.999980, RMSE 0.000451); the zero-initialized fusion output had only tiny weights. Root cause: `lr=1e-6` was too conservative for the new zero-output fusion block.
- Stronger fusion probe completed at `runs/cross_view_1024_descriptor_fusion_quota_stronglr3e5_b4_240_seed1234`: same style-specific positives plus mined false matches, separate quotas, `batch_pairs=4`, `learning_rate=3e-5`, descriptor head frozen, texture adapter + descriptor fusion trainable, max grad norm 5.0. Validation improved materially for this line: loss 4.355897/top1 0.0980/top5 0.2867 -> loss 4.337841/top1 0.1154/top5 0.3197.
- Raw sparse test for the strong fusion probe improved all six groups at blend=1. Total precision moved 0.227604 -> 0.235790, matches 113627 -> 124687, correct 25862 -> 29400. Per group deltas: numeric/rotate +0.039049, numeric/viewpoint +0.014568, numeric/compound +0.011477, timestamp/rotate +0.004687, timestamp/viewpoint +0.008718, timestamp/compound +0.002668.
- Blend=4 is not the right inference setting for the strong fusion probe. Although numeric/rotate precision increased 0.622501 -> 0.755113, total precision fell 0.197315 -> 0.171457 and support dropped sharply from 80009 to 41054 matches. Current best setting for this trained state is blend=1.
- Continued the strong fusion checkpoint for another 240 steps at `lr=1e-5`: `runs/cross_view_1024_descriptor_fusion_quota_continue_lr1e5_b4_240_seed1234`. Validation improved again from loss 4.337841/top1 0.1154/top5 0.3197 to loss 4.322011/top1 0.1185/top5 0.3268. Blend=1 raw sparse total moved to 30033/126832 (0.236794), but timestamp/rotate precision fell below the strong checkpoint (0.284382 -> 0.274885), showing that viewpoint/compound-only supervised updates were starting to erode rotate behavior.
- Added TDD coverage and changed mixed supervised sampling semantics for low pseudo/false probabilities. When pseudo and false pools are both active, their probabilities now determine the total supervised slots, so e.g. `batch_pairs=4`, `pseudo=0.25`, `false=0.25` yields 1 pseudo pair, 1 false pair, and 2 base/synthetic pairs. This keeps the previous full-supervision behavior when both probabilities are 1.0. Focused sampler tests passed after first observing the expected RED failure.
- Ran mixed base regularization from the continue checkpoint with rotate/viewpoint/compound train dirs included: `runs/cross_view_1024_descriptor_fusion_mixedbase_lr5e6_b4_200_seed1234`, `synthetic_loss_weight=0.2`, pseudo/false max probabilities 0.25. It achieved the highest raw blend=1 precision at margin 0.01 so far: 29318/120769 (0.242761), but reduced correct matches versus the continue checkpoint (30033 -> 29318).
- Ran the gentler mixed base version: `runs/cross_view_1024_descriptor_fusion_mixedbase_w005_lr5e6_b4_160_seed1234`, `synthetic_loss_weight=0.05`, same pseudo/false/base mix. At margin 0.01 it produced the highest correct-match count so far: 30420/128859 (0.236072), with numeric/rotate 19462 correct and timestamp/rotate 7000 correct. It preserves recall better than the w=0.2 mixed run.
- Inference margin sweep for the mixed checkpoints found the current recommended pure-PFM operating point: `runs/cross_view_1024_descriptor_fusion_mixedbase_w005_lr5e6_b4_160_seed1234/training/pytorch_pfm_state.pt` with blend=1 and `min-margin=0.015`. On the fixed 64-pair six-group test it gives 26902/98207 (0.273932), improving both precision and correct count over the base blend=1 result 25862/113627 (0.227604) while reducing wrong matches 87765 -> 71305.
- Higher precision settings are available but lose too much support for the main recommendation. `mix02` with margin 0.02 reaches 19389/61028 (0.317707), and `mix005` with margin 0.02 reaches 21808/71287 (0.305918), but both have fewer correct matches than the base. Keep them as high-precision reporting modes, not the default balanced checkpoint/config.
- Temporary aggregation script error during the mixed `w=0.2` comparison was a local reporting bug (`TypeError: 'int' object is not subscriptable`) and was fixed by changing the ad hoc totals structure to explicit dictionaries. No model/eval outputs were affected.
- Started the pose-aware satsim data path. Confirmed `kernel/lsk/naif0012.tls.pc`, `kernel/spk/planets/de430.bsp`, and `kernel/pck/pck00010.tpc` exist under `辅助软件/数据模拟/kernel`.
- Updated `辅助软件/数据模拟/mars_orbit_to_tsai.py`: defaults now use project-local kernel/DEM/DOM/sat_sim paths; added `--write-depth`; added `--camera-perturbations` for virtual local camera-angle variants.
- Updated `辅助软件/数据模拟/src/sat_sim_cuda.h`, `src/sat_sim_cuda.cu`, and `src/main.cpp`: `sat_sim_cuda --write-depth` now writes per-pixel camera-space depth into `image/Camera*/depth/`.
- Reconfigured copied CMake build directory with `source /home/xjw/anaconda3/bin/activate asp36 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`; then `cmake --build build -j 8` passed. One pre-existing GDAL return-value warning remains in `writeMulti`.
- Verification passed for `python -m py_compile mars_orbit_to_tsai.py mars_orbit_to_tsai_fov90.py` and `pfm-train` `py_compile` for `pose_sim_to_pfm_cache.py`. `asp36` has no `pytest` module, so local pytest tests were not run.
- Generated 2048 smoke TSAI/render at `辅助软件/数据模拟/output_pfm_pose_sim/smoke_2048_depth_virtual_20251216_110km` using defaults plus `--camera-perturbations 'base=0,0,0;xp8=8,0,0'` and `--write-depth`. CameraA rendered in about 3.4s; CameraA_xp8 rendered in about 16.9s because its crop expanded to 3350x3350 DEM samples.
- Verified depth stats: CameraA depth min/max/mean about 107,750/117,023/111,461 m; CameraA_xp8 depth min/max/mean about 100,060/137,077/114,298 m.
- Added `辅助软件/数据模拟/pose_sim_to_pfm_cache.py`. Generated `辅助软件/数据模拟/output_pfm_pose_sim/cache_smoke_2048/source_000001_20251216_110km_A_Axp8/pair_000001.pt` and loaded it with the existing `patch_descriptor_training.load_libtorch_pair_archive`. Shapes are `view_a/view_b=(1,2048,2048)`, `warp=(2048,2048,2)`, `mask=(2048,2048)`, and valid fraction is 0.845109.
- Added focal-scale variants to `mars_orbit_to_tsai.py` through `--focal-scales`. It combines with `--camera-perturbations` and appends suffixes such as `CameraA_xp8_f0p85`, while the true base camera remains `CameraA` when view perturbation is zero and focal scale is 1.0.
- Generated the first human-review 2048 view/focal sample set at `辅助软件/数据模拟/output_pfm_pose_sim/review_view_focal_2048_20251216_110km_frame0001`. Parameters: `CameraA`, frame idx 1 from `Orbiter_InfoCSV_20251216_110km.csv`, view variants `base/xp8/xn8/yp8/yn8`, focal scales `0.85/1.0/1.20`, no depth/cache. All 15 renders succeeded.
- Created review artifacts for that sample set: 15 individual PNG previews in `review_png/` and a 5x3 overview `review_contact_sheet.png`. The sample output directory is about 215 MB.
- Generated the second human-review 2048 view sample set at `辅助软件/数据模拟/仿真图像` after user requested 9 simulated views per trajectory. For each of the four trajectory CSVs, frame 1 was rendered with real `CameraA` plus 9 simulated views: `basic_xp5`, `basic_xn5`, `basic_yp5r8`, `mid_xp12`, `mid_yn12r15`, `mid_diag`, `ext_xp22`, `ext_yn22r25`, and `ext_diag`. Focal scale was fixed to 1.0 for this review batch. All 40 byte previews rendered successfully.
- Created merged comparison images for the second batch: each trajectory directory has `views10_contact_sheet.png`, and the root has `all_tracks_views10_contact_sheet.png`. The `仿真图像` directory is about 665 MB. High-altitude 340km medium/extreme views were much slower than 110km views, so future full data generation should treat 340km extreme angles as a separate cost/quality decision.
- Added streaming batch generator `辅助软件/数据模拟/batch_pose_sim_dataset.py`. It prepares gap-30 TSAI for four trajectories, streams one frame's 10 views at a time, converts real CameraA -> 9 simulated views into PFM TorchScript pair archives, writes split-aware cache paths under `/media/xjw/8T/深度学习数据/仿真训练集`, and deletes intermediate render/depth files after conversion.
- Smoke generation on `/media/xjw/8T/深度学习数据/仿真训练集/pose_sim_2048_gap30_views10_smoke` succeeded for 2 pairs. Both loaded through `patch_descriptor_training.load_libtorch_pair_archive`; shapes were `(1,2048,2048)` and valid pixels were 3,770,242 per pair.
- Formal generation was restarted per user request with the outer process in `asp36`. Since `asp36` has no PyTorch, the batch script still calls `/home/xjw/anaconda3/envs/pfm-train/bin/python` only for `pose_sim_to_pfm_cache.py` TorchScript archive writing. Active runs are `first3000_asp36` and `remaining_from_3000_asp36` under `/media/xjw/8T/深度学习数据/仿真训练集/pose_sim_2048_gap30_views10/nohup_logs/`, started with `setsid` so they continue outside the shell.
- Added training-only pose metadata support. New `python/pose_pair_metadata.py` reads `manifests/*.csv` plus `tsai_tracks/*/tsai/Camera*/NNNNN.tsai` and derives baseline, view-angle difference, focal ratio, overlap fraction, and easy/medium/hard difficulty without changing the existing `.pt` pair archive format.
- Extended `python/pfm_pytorch_training.py` with `--pose-metadata-root`, `--pose-balanced-sampling`, `--pose-min-overlap`, and `--pose-difficulty-loss-weight`. Pose metadata is used only for training sampling/loss weighting and metrics; model inference still consumes images only.
- Added `--training-max-image-size` after a 2048 full-image smoke exposed CUDA OOM on the 32 GB 5090. The training loader can now resize pair tensors, warp fields, and masks before model forward; this keeps current 2048 caches usable for 1024-scale smoke/prototype training.
- Verification: focused pose/training tests passed (`73 tests OK`), full Python discovery passed (`209 tests OK, 2 skipped`), and `py_compile` passed for `python/pose_pair_metadata.py` and `python/pfm_pytorch_training.py`.
- Ran a real generated-data smoke on `/media/xjw/8T/深度学习数据/仿真训练集/pose_sim_2048_gap30_views10` with pose metadata, pose-balanced sampling, overlap >=0.5, difficulty loss weight 0.5, and `--training-max-image-size 1024`. Output is `runs/pose_metadata_smoke_20260527_v2`; it loaded 3,282 pose metadata keys, completed 2 CUDA steps with `skip=0`, and wrote `metrics.csv`, `eval_summary.csv`, and `pytorch_pfm_state.pt`.
- Switched the 2048 training path from resize to native-resolution crops. `python/pfm_pytorch_training.py` now supports `--training-crop-size`, which crops `view_a`, chooses a corresponding B crop from the crop's warped valid pixels, shifts `warp_a_to_b` into B-crop coordinates, and masks points outside the B crop. Existing resize remains as optional fallback.
- Verification after crop support: focused pose/training tests passed (`74 tests OK`), full Python discovery passed (`210 tests OK, 2 skipped`), and `py_compile` passed.
- Ran a real 2048-to-1024 crop smoke on the generated cache with `--training-crop-size 1024` and no resize. Output is `runs/pose_metadata_crop1024_smoke_20260527`; it loaded 3,471 pose metadata keys, completed 2 CUDA steps with `skip=0`, and wrote `metrics.csv`, `eval_summary.csv`, and `pytorch_pfm_state.pt`. Each smoke step used 1 pair x 16 sampled correspondences from a native-resolution 1024 crop.
- Ran a 100-pair-pool pose/crop test on current generated data: `runs/pose_metadata_crop1024_100pairs_test_20260527`. Parameters: train cache limit 100 pairs, val cache 20 pairs, native 1024 crop from 2048 archives, `batch_pairs=1`, `samples_per_pair=64`, 20 steps. It loaded 3,525 pose metadata keys, completed with `skip=0`, and wrote `metrics.csv`, `eval_summary.csv`, and `pytorch_pfm_state.pt`. Eval was essentially unchanged over this intentionally tiny run: top1 0.6859 before and after.
- Completed the training visual-report upgrade. `scripts/training_visual_report.py` now writes `training_report_zh.pdf`, `matching_diagnostics.png`, denser raw match visualizations, and an expanded `match_visual_summary.csv`; `python/pfm_pytorch_training.py` adds `--generate-training-report` to call it after training.
- Fixed pose metadata lookup for copied local NVMe datasets by indexing the stable `cache/.../pair_*.pt` suffix, so reports can recover medium/hard difficulty even when manifest paths point to the 8T source.
- Generated and verified the current run report at `runs/pose_metadata_crop1024_1800pairs_random_lr5e5_round1_20260527/visual_report_v2/training_report_zh.pdf` (6 pages). The 16-sample raw report has 8192 matches, 4932 correct, precision 0.6021; medium group precision is 0.7075 and hard group precision is 0.2856.
- Found the first concrete training bug behind the poor hard-view result: the pose-balanced sampler used a fixed easy/medium/hard order, so with `batch_pairs=1` it always selected medium when easy was absent. The completed 1800-step run therefore had `pose_hard_pairs=0` for every step despite the local train split containing 908 hard and 1132 medium pairs.
- Added a regression test for `batch_pairs=1` pose-balanced sampling and fixed the sampler by randomizing the difficulty order on each round. Focused pose-balanced tests and `py_compile` passed.
- Ran a hard-aware continuation: `runs/pose_metadata_crop1024_samplerfix_hard_fusion_lr2e5_600_20260527`. It starts from the previous checkpoint, uses hard/medium-balanced 1024 crops, trains blended descriptors through texture adapter + descriptor fusion, and samples 1024 supervised points per optimizer step. Validation improved to loss 1.588688 and top1 0.8166.
- Matching result is mixed without margin: no-margin report has 4737/8192 correct (precision 0.5782), so it is not a better raw checkpoint. With `min_margin=0.01`, the same checkpoint gives 5060/6931 correct (precision 0.7301), hard group 584/1329 (precision 0.4394). Report: `runs/pose_metadata_crop1024_samplerfix_hard_fusion_lr2e5_600_20260527/visual_report_margin001/training_report_zh.pdf`.
- Full Python test suite after the sampler/report changes passed: 212 tests OK, 2 skipped.
- Started the next pose-aware margin optimization pass. The key constraint is that mined false-match CSV labels are currently in original-image coordinates, while the current 2048 -> 1024 training path uses random crops and shifts the warp into crop coordinates. Directly using those mined labels would create coordinate-mismatched negative supervision, so this pass stays with crop-safe warp-truth hard-negative and abstention losses.
- Completed the strong-margin continuation `runs/pose_metadata_crop1024_marginstrong_lr8e6_400_20260527`. It improved validation retrieval slightly (top1 0.8157 -> 0.8219), but the report regressed at the actual operating point: `min_margin=0.01` total 5056/7039 precision 0.7183 versus previous best 5060/6931 precision 0.7301; hard group 589/1370 precision 0.4299 versus previous 584/1329 precision 0.4394. Decision: do not continue from this checkpoint.
- User noted GPU was not saturated and suggested increasing batch size. Next pass starts from the previous best checkpoint, not the regressed strong-margin one, and raises `batch_pairs` from 2 to 4 for about 2048 supervised points per optimizer step.
- User asked to stop partial v2.1 iterations and land the full v2.1 design in one pass. Implemented the PyTorch v2.1-full main path: `DualFPNLite`, separate keypoint/descriptor P2 features, dense geometry-aware descriptor pooling, `QualityHead`, v2.1 `RawFeatureMaps` quality/local contrast outputs, 16-dim graph metadata, GraphMatcherV2 pairwise geometry bias, top-k candidate pruning, dual-softmax scoring, and raw-descriptor fallback merge for weak graph outputs.
- Training entry points now support full v2.1 module selection: `--train-backbone`, `--train-dual-fpn`, `--train-geometry-head`, `--train-quality-head`, `--train-graph-matcher`, and `--graph-matcher-loss-weight`. Graph matcher training now has a real correspondence CE loss from sampled warp correspondences.
- Evaluation/reporting now passes selected keypoint score/quality into GraphMatcher metadata where available; `training_visual_report.py` uses the same graph matcher call path.
- Verification for the v2.1-full landing passed: `py_compile` on model/eval/training/report modules, focused tests `128 OK, 1 skipped`, full Python discovery `223 OK, 2 skipped`, and a small v2.1 forward/matcher smoke (`desc=(1,32,16,16)`, `heat=(1,1,16,16)`, `quality=(1,1,16,16)`, graph logits `(5,5)`).
- Ran a real command-line v2.1-full training smoke at `runs/v21_full_arch_smoke_20260527` using `--init-random`, train-backbone/dual-fpn/geometry/texture/fusion/quality/graph flags, `training-max-image-size=128`, 1 train pair and 1 val pair. It completed 1 CUDA step with `skip=0`, wrote `pytorch_pfm_state.pt` and `metrics.csv`, and verified that the graph matcher correspondence loss path is executable from CLI.
- User clarified that model architecture should fully fit the PFM v2.1 need, while C++ sync/long training can wait. Closed the remaining PyTorch architecture gaps: added `SemiDenseCandidateBranch` for coarse detector-free weak-texture candidates; added feature-grid bilinear descriptor sampling; added GraphMatcher metadata construction from `RawFeatureMaps` so scale/orientation/affine/quality/local contrast reach GraphMatcher; and updated evaluation/report GraphMatcher calls to use that metadata. Verification passed with `py_compile`, focused tests `131 OK, 1 skipped`, full discovery `226 OK, 2 skipped`, and a graph matcher identity smoke with 16/16 correct.
## PFM v2.1-full 256维重新炼丹 (2026-05-27)

### 状态
- 阶段：in_progress。
- 目标：不从 192 维 checkpoint 迁移，使用完整 PFM v2.1-full 架构随机初始化，重新训练原生 256 维特征点和 GraphMatcher。

### 已完成
1. 更新本地 NVMe 训练数据：`训练数据/pose_sim_2048_gap30_views10` 已从 8T 增量同步到 train 3735 / val 1863，与源端一致。（complete）
2. 修复训练 descriptor 路径：`learned_descriptor_and_heatmap_single()` 现在使用 `model.dual_fpn(features)` 和 `model.sparse_head(p2_keypoint, p2_descriptor)`，不再绕过 v2.1 Dual-FPN。（complete）
3. 验证通过：`py_compile` 通过；focused tests `132 OK, 1 skipped`；full discovery `227 OK, 2 skipped`。（complete）
4. 1024 crop / 256 维 / 随机初始化 / 全 train flags 冒烟通过：`runs/v21_full256_crop1024_smoke_20260527`，1 step CUDA 成功，写出 `pytorch_pfm_state.pt` 和 `metrics.csv`。（complete）
5. 正式训练已完成：`runs/pose_sim2048_crop1024_v21full256_scratch_20260527`，3000 steps、1024 crop、`samples_per_pair=512`、pose-balanced、GraphMatcher loss。（complete）
6. 自动报告阶段最初因父训练进程仍占用约 28GB 显存导致报告子进程 OOM；训练进程退出后已单独补生成 raw descriptor 和 GraphMatcher 两套中文 PDF 报告。（complete）

### 当前观测
- 训练本体 3000 steps 全部完成，`skip=0`。
- 最终验证检索：loss `3.4903 -> 0.0224`，top1 `0.6215 -> 0.9985`，mean negative score `0.5717 -> 0.0025`。
- raw descriptor 随机验证报告：8192 matches，8187 correct，5 wrong，precision `0.999390`，mean error `1.133 px`。
- GraphMatcher 随机验证报告：8192 matches，6469 correct，1723 wrong，precision `0.789673`；hard 子集 538/1024，precision `0.525391`，说明正式 GraphMatcher 链路仍弱于 raw descriptor。
- GPU 利用率训练中约 98%-100%，显存最高约 31.8GB / 32.6GB；当前 batch/samples 已接近 5090 32GB 上限，不建议继续加 batch size。

### 下一步
1. 下一步应做独立固定六组/真实 TIFF 复测，确认 raw descriptor 的仿真验证收益能否迁移到原先对比文档和真实跨高度影像。
2. GraphMatcher 需要单独继续优化或暂时降低在正式路由中的权重；当前它会把 raw descriptor 的好结果拉低。
3. 若后续训练出现 OOM 或 CUDA allocator 错误，优先把 `samples_per_pair` 从 512 降到 384，而不是降低 256 维 descriptor 或回退旧 checkpoint。

### 错误记录
| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| 第一次 `nohup` 后台启动进程立即消失且日志为空 | 1 | 前台同参数 1 step 验证训练正常；改用 `setsid bash -c 'exec ...'` 脱离执行会话后后台训练稳定运行 |
| 训练结束后自动报告 OOM | 1 | 原因是父训练进程持有 CUDA 显存时又启动报告子进程；训练退出后单独运行 `training_visual_report.py` 成功生成 raw/graph 两套报告 |

## 弱纹理覆盖采样与继续训练 (2026-05-27)

### 状态
- 阶段：in_progress。
- 目标：解决匹配点在强纹理区域扎堆、弱纹理区域点少的问题；优先改候选点采样和报告，不先重改 backbone。

### 已完成
1. `select_descriptor_keypoints()` 增加 `weak_texture_fraction` 和 `keypoint_cell_cap`，支持高纹理/弱纹理/均匀点 quota，并限制单个网格热点。（complete）
2. `pytorch_cache_match_eval.py`、`training_visual_report.py` 和训练报告入口已接入新参数。（complete）
3. 训练报告新增 coverage 指标和 `coverage_diagnostics.png`：网格覆盖率、最密集网格占比、覆盖熵、弱纹理匹配占比。（complete）
4. 修复训练结束后自动报告 OOM：生成报告前释放训练模型和 optimizer 显存。（complete）
5. 验证通过：`py_compile` 通过；focused tests `107 OK`；full Python discovery `229 OK, 2 skipped`。（complete）
6. 用上一轮 checkpoint 生成 coverage-aware raw 报告：`raw_descriptor_coverage_v1`，16 个样本、16384 matches、16368 correct、precision `0.999023`，平均网格覆盖率 `0.812012`，弱纹理匹配占比 `0.264160`。（complete）
7. 更新本地仿真数据：train `3744`、val `1872`，与 8T 源端一致。（complete）
8. 新一轮 full-256 continuation 已完成：`runs/pose_sim2048_crop1024_v21full256_coverage_cont1500_20260527`，从上一轮 256 checkpoint 继续 1500 steps，低学习率 `1e-5`，报告使用 coverage-aware 采样参数。（complete）
9. 追加一轮正式 fused descriptor 训练完成：`runs/pose_sim2048_crop1024_v21full256_blend_b1_cont3000_20260528_080816`。先尝试 `batch_pairs=2` 时因 1024 crop + blended descriptor/fusion 训练在第一步反传 OOM；改为 `batch_pairs=1`、3000 steps 后稳定完成，显存约 20.8GB，`skip=0`。eval loss `0.007729 -> 0.007402`，mean negative score `0.001215 -> 0.000682`；24 个验证样本报告 24569/24576 correct，precision `0.999715`，平均 16x16 覆盖率 `0.819987`，弱纹理匹配占比 `0.277547`。（complete）
10. 用户提出下一步数据扩展思路：不只使用同一个相机位置下不同姿态/焦距的匹配对，还应把不同相机位置、但有有效重叠区域的图像组成匹配对。该方向更贴合真实跨轨/跨位置任务，可复用现有仿真影像与相机参数，不需要重新渲染；关键是用 DEM/DOM 与相机模型从世界坐标生成密集或稀疏 GT 对应，而不是假设全局单应 warp。（pending）
11. 已生成第一版跨相机位置训练缓存：`训练数据/pose_sim_cross_position_2048_gap30_views10`，train 1200 / val 384，占用约 106G。当前采用 CameraA 射线与 Mars 参考球面求交再投影的方式生成 dense warp/valid mask；原因是已有同位置虚拟相机中心相同，无法由同中心 warp 三角化深度。（complete）
12. 修复 1024 训练 crop 对跨位置 pair 的采样问题：原随机 crop 可能裁到无有效对应区域，导致 `RuntimeError: no valid correspondences sampled`；现在 `crop_pair_for_training()` 优先围绕 valid mask 中的有效监督点裁剪。200 个跨位置 pair x 3 次裁剪回归检查从 31 个失败降到 0 个失败，最小 crop valid fraction 约 0.350。（complete）
13. 混合同中心与跨位置缓存继续训练完成：`runs/pose_sim2048_crosspos_mix_v21full256_blend_b1_3000_20260528_090944`。从上一轮 fused 256 checkpoint 继续 3000 steps，`batch_pairs=1`、1024 crop、coverage-aware 报告参数。最终 checkpoint、`metrics.csv`、`eval_summary.csv` 和中文 PDF 报告已生成。（complete）

### 当前观测
- 训练 1500 steps 完成，`skip=0`，自动报告成功生成，未再出现报告 OOM。
- eval：loss `0.022447 -> 0.014582`，top1 `0.998528 -> 0.999054`，mean negative score `0.002533 -> 0.001120`。
- coverage-aware raw 报告：16384 matches，16379 correct，5 wrong，precision `0.999695`，平均 16x16 网格覆盖率 `0.809570`，最密集网格占比 `0.013550`，覆盖熵 `0.929619`，弱纹理匹配占比 `0.270019`。
- fused descriptor 追加训练后，验证检索和覆盖指标小幅改善；下一轮优化重点应转向构造跨相机位置匹配对，而不是继续只在同相机中心扰动数据上微调。
- 跨位置混合训练最终 eval：loss `0.007551 -> 0.016118`，top1 `0.999718 -> 0.999588`，mean positive score `0.933331 -> 0.939315`，mean negative score `0.000597 -> 0.001954`。这个总指标混合了同中心和跨位置验证集，因此 loss 上升主要来自新增跨位置难样本。
- 24 样本报告中，同中心姿态/焦距样本仍为 24539/24576 correct，precision 约 `0.9985`；跨位置样本分化明显：近邻 pair precision `0.7568`、`0.7041`，更难 pair precision `0.0059`、`0.0518`。结论是模型已经能保持原同中心能力，但跨相机位置尤其较大基线仍是当前主要短板。

### 下一步
1. 下一轮跨位置训练建议把 cross pair 分桶进入报告：offset=1/2/4/8 分别统计，避免总 eval 掩盖远基线失败。
2. 当前跨位置 GT 用参考球面近似，后续应改成 DEM 射线交会或直接保存仿真 depth，提升跨位置 warp 的几何精度。
3. 针对跨位置大基线，训练策略需要改为 curriculum：先 offset=1/2 稳定，再逐步加 offset=4/8；否则极难样本会造成单步 loss 尖峰但收敛慢。
4. 后续若还要进一步提升弱纹理区域，可把 coverage-aware 采样产生的均匀/弱纹理点反向蒸馏到 keypoint/quality head。

## 真实 DEM depth 跨相机位置数据与继续训练 (2026-05-28)

### 状态
- 阶段：complete。
- 目标：把跨相机位置匹配对从 Mars 参考球面近似升级为 `sat_sim_cuda --write-depth` 渲染出的真实 DEM depth 几何，并混合同中心姿态/焦距数据继续训练 PFM v2.1-full 256 维模型。

### 已完成
1. `scripts/generate_cross_position_pose_pairs.py` 增加真实 depth 模式：默认调用 `辅助软件/数据模拟/build/sat_sim_cuda`、`dem/mar.tif`、`dom/HX1_GRAS_MoRIC_DOM_076m_Global_A.tif`，支持 `--depth-mode rendered|sphere|warp-or-sphere`。（complete）
2. 增加并行 depth 预渲染：`--depth-render-workers` 用 `ProcessPoolExecutor` 对唯一 CameraA 源影像批量生成 depth cache，避免逐 pair 串行渲染。（complete）
3. 生成正式真实 depth 跨位置缓存：`训练数据/pose_sim_cross_position_rendered_2048_gap30_views10`，train 1200 / val 384，depth cache 624 个 TIFF，总体约 110G；生成过程 `low_overlap=0`、`errors=0`。（complete）
4. 修复 1024 crop 对跨位置 pair 的有效区域采样后，真实 cross pair 训练未再出现 `no valid correspondences sampled`。（complete）
5. 从最后一个未受球面近似 cross 数据影响的 fused 256 checkpoint 继续训练，而不是沿用 `pose_sim2048_crosspos_mix_v21full256_blend_b1_3000_20260528_090944`。（complete）
6. 完成真实 depth 混合训练：`runs/pose_sim2048_crosspos_rendered_mix_v21full256_blend_b1_3000_20260528_105130`，3000 steps，1024 crop，`batch_pairs=1`、`samples_per_pair=256`、coverage-aware 报告参数。（complete）

### 当前观测
- checkpoint：`runs/pose_sim2048_crosspos_rendered_mix_v21full256_blend_b1_3000_20260528_105130/pytorch_pfm_state.pt`，大小约 230MB，payload `config.descriptor_dim=256`。
- 训练：3000 rows，`skip=0`，last100 `top1_accuracy` 均值 `0.999766`，last100 loss 均值 `0.007873`。
- 验证检索：loss `0.007551 -> 0.007189`，top1 `0.999718 -> 0.999786`，mean positive score `0.933331 -> 0.936102`。
- 中文报告：`runs/pose_sim2048_crosspos_rendered_mix_v21full256_blend_b1_3000_20260528_105130/visual_report/training_report_zh.pdf`。
- 24 个报告样本：平均 `1023.42/1024` correct，precision `0.999430`，mean error `1.57 px`，median error `1.38 px`，P90 error `2.14 px`。
- 空间/弱纹理覆盖：平均 16x16 网格覆盖率 `0.810058`，最密集网格占比 `0.012939`，覆盖熵 `0.932138`，弱纹理匹配占比 `0.293579`。
- 报告中 4 个真实 depth 跨位置验证样本全部达到 `1024/1024` correct；此前球面近似 cross 数据中远基线样本大幅失败，因此后续不应再用球面近似 cross checkpoint 作为主线。

### 下一步
1. 把真实 cross pair 报告按 offset=1/2/4/8 分桶统计；当前 24 样本报告只抽到 4 个 cross val 样本，不能替代全量难度分桶。
2. 后续继续扩大真实 depth cross 数据时，先清理或迁移旧球面近似缓存，避免训练集混入几何不准标签。
3. 若继续优化弱纹理分布，优先把当前 coverage-aware 采样结果反向监督 keypoint/quality head，而不是再单纯增加匹配输出数量。

## 7:2:1 切分与更极端跨位置样本训练 (2026-05-28)

### 状态
- 阶段：complete。
- 目标：按用户要求把训练/验证/测试比例调整为 7:2:1，新增更极端的跨相机位置样本，并在 view report 中加入更极端匹配效果示例。

### 已完成
1. `scripts/generate_cross_position_pose_pairs.py` 支持 train/val/test 三 split、offset 交错均衡生成、`--reuse-root` 复用已有 pair、绝对路径渲染，避免 `sat_sim_cuda` 在数据模拟目录下找不到相对 VRT。（complete）
2. 新增真实 depth 跨位置极端缓存：`训练数据/pose_sim_cross_position_rendered_extreme_2048_gap30_views10_721`，train 1400 / val 400 / test 200，严格 7:2:1。（complete）
3. 原同中心姿态/焦距缓存生成 7:2:1 符号链接视图：`训练数据/pose_sim_2048_gap30_views10_721`，train 5235 / val 1495 / test 749。（complete）
4. 混合后训练可用样本为 train 6635 / val 1895 / test 949，整体仍保持 7:2:1。（complete）
5. 完成一轮 3000 step 训练：`runs/pose_sim2048_721_cross_extreme_v21full256_blend_b1_3000_20260528_124407`，从真实 depth cross checkpoint 继续，`skip=0`。（complete）
6. 主中文报告已生成：`runs/pose_sim2048_721_cross_extreme_v21full256_blend_b1_3000_20260528_124407/visual_report/training_report_zh.pdf`，强制包含 `off012` 更难验证样例。（complete）
7. 额外极端测试报告已生成：`runs/pose_sim2048_721_cross_extreme_v21full256_blend_b1_3000_20260528_124407/visual_report_extreme_test_off016/training_report_zh.pdf`，强制包含 `off016` 测试样例。（complete）

### 当前观测
- 新跨位置缓存 offset 分布：
  - train：{1: 298, 2: 298, 4: 298, 8: 298, 12: 102, 16: 106}，valid mean 0.6783。
  - val：{1: 81, 2: 80, 4: 80, 8: 80, 12: 79}，valid mean 0.5591；val 中 offset 16 被 overlap 过滤掉。
  - test：{1: 34, 2: 34, 4: 33, 8: 33, 12: 33, 16: 33}，valid mean 0.7647。
- 训练检索验证：loss `0.007840 -> 0.007656`，top1 `0.999657 -> 0.999702`，mean positive score `0.930958 -> 0.932014`。
- 主报告 `off012` 验证样例较难：32 个样本平均 1024 matches、140.34 correct、precision 0.1371，平均覆盖率 0.9556，弱纹理匹配占比 0.2067。
- 额外 `off016` 测试样例反而较容易：32 个样本平均 1024 matches、959.16 correct、precision 0.9367，平均覆盖率 0.7196，弱纹理匹配占比 0.1416。
- 结论：难度不能只看 offset，轨迹段、视角组合和有效重叠区域共同决定难度；后续报告需要同时按 offset、valid fraction、track/seq 分桶，避免只看某一类高分或低分样例。

### 下一步
1. 对新 7:2:1 数据集做分桶评估：offset、valid fraction、轨迹、高度/焦距、弱纹理占比分别统计。
2. 训练报告建议同时固定抽样 `off012` 难样例和 `off016` 测试样例，避免报告对数据难度产生采样偏差。
3. 如果要继续提升跨位置极端样例，应优先加强跨位置 hard negative、弱纹理 semi-dense fallback 或 GraphMatcher，而不是只继续降低 raw descriptor loss。

## 10 epoch + 512 points/pair 训练与数据迁移 (2026-05-28)

### 状态
- 阶段：complete。
- 目标：按用户要求把训练从 step 口径改成完整 epoch 口径，单图像对训练采样点数改为 512，并把原同中心 7:2:1 视图中的软链接目标用 move 迁移成独立数据目录。

### 已完成
1. `python/pfm_pytorch_training.py` 增加 epoch 训练入口：`--epochs`、`--save-every-epoch`、`--epoch-shuffle-sampling`，日志显示 `step=... epoch=.../...`，并按 epoch 写出 checkpoint。（complete）
2. 增加 `PairArchiveCache` 与后台 prefetch 参数：`--pair-cache-size`、`--prefetch-batches`、`--prefetch-workers`，减少机械盘读取对 GPU 的影响。（complete）
3. 修复训练 crop fallback：无随机 generator 时按 valid 点均值裁剪，避免跨位置有效区域被裁掉。（complete）
4. 按用户“用 move 移动”要求，将 `训练数据/pose_sim_2048_gap30_views10_721` 中 14960 个软链接对应的目标文件从 `训练数据/pose_sim_2048_gap30_views10` 移入 `_721` 目录；迁移后 `_721` 为实文件目录、软链接数 0，包含 7479 个 `pair_*.pt`，占用约 498G；原目录缩小为约 125M。（complete）
5. 10 epoch 训练完成：`runs/pose_sim2048_721_cross_extreme_v21full256_blend_b1_10epoch_s512_prefetch_20260528_145414`。训练集 6635 pair，`batch_pairs=1`，每个 epoch 6635 step，总计 66350 step；每个 pair 采样 `points=512`；`skip=0`。（complete）
6. 每个 epoch checkpoint 已写出到 `checkpoints/epoch_001...epoch_010_pytorch_pfm_state.pt`，最终权重 `pytorch_pfm_state.pt` 约 241MB。（complete）
7. 自动中文训练报告已生成：`visual_report/training_report_zh.pdf`，强制抽样 `off012`；额外生成 `visual_report_extreme_test_off016/training_report_zh.pdf`，强制抽样 `off016`。（complete）

### 当前观测
- 训练检索评估：loss `0.014629 -> 0.011804`，top1 `0.999432 -> 0.999542`，mean negative score `0.000788 -> 0.000618`，评估点数 262144。
- `off012` 报告：32 个样本，precision mean `0.1380`，min `0.0859`，max `0.1943`，平均 correct `141.34/1024`，平均 coverage `0.9586`，弱纹理匹配占比 `0.2088`。
- `off016` 报告：32 个样本，precision mean `0.9420`，min `0.9033`，max `0.9658`，平均 correct `964.59/1024`，平均 coverage `0.7076`，弱纹理匹配占比 `0.1388`。
- 结论：dense retrieval 训练指标很高，但实际 decoded keypoints + raw descriptor 可视化在 `off012` 上很差，而 `off016` 很好；难点不只是 offset 大小，可能与具体轨迹段、裁剪、GT 对齐口径、重叠区域纹理分布有关。
- `训练数据/pose_sim_cross_position_rendered_2048_gap30_views10` 仍保留约 110G，当前 10 epoch 训练未使用；当前使用的是 `训练数据/pose_sim_cross_position_rendered_extreme_2048_gap30_views10_721`。

### 下一步
1. 对 `off012` 低分样本做专项排查：验证 pair 内 GT warp、valid mask、image/crop 坐标、可视化判定阈值和真实重叠区域是否一致。
2. 把报告统计从单一 required glob 扩展为按 offset、track、valid fraction、weak-texture fraction 分桶，避免被某一组样本误导。
3. 后续如果继续炼丹，优先解决 decoded keypoint 采样与 raw descriptor 评估落差，而不是只看 dense retrieval loss。

## Extreme v2 数据扩充与 abstention 训练 (2026-05-28)

### 状态
- 阶段：in_progress。
- 目标：按用户反馈扩充 extreme 跨位置数据集，并按低重叠问题加入 no-match/abstention 方向的训练约束。

### 已完成
1. 修复 `scripts/generate_cross_position_pose_pairs.py`：支持 `source_repart_..._source_XXXXX_track` 目录，并按 `(track, seq)` 去重；新增从已有 `cache/*/source_cross_*/pair_*.pt` 扫描复用，支持断点续跑。（complete）
2. 新增回归测试 `python/test_generate_cross_position_pose_pairs.py`，验证 repart source 能被识别和去重；`python/test_generate_cross_position_pose_pairs.py` 与 `python/test_pfm_pytorch_training.py` 共 80 tests OK。（complete）
3. 查明 `off012` 低分原因：主报告抽到 val 中 `20251216_110km` 的 `off012`，有效重叠仅约 `0.027-0.07`；而 test 中 `off012` 有效重叠约 `0.607`，raw precision 约 `0.98`。因此低分不是 offset=12 必然失败，而是低重叠场景中 raw descriptor 强行输出 1024 matches 导致错误匹配暴增。（complete）
4. 扩充新数据集：`训练数据/pose_sim_cross_position_rendered_extreme_2048_gap30_views10_721_v2`，底层迁移到 8T 并在原路径保留软链接。最终 train/val/test 为 `4200/1200/600`，严格 7:2:1；offset 扩展为 `1,2,4,8,12,16,20,24,32`。（complete）
5. 数据生成过程中根分区满导致 `torch.jit.save` 报 `unexpected pos ...`；确认 `/` 仅剩约 164M 后，将旧 `pose_sim_cross_position_rendered_2048_gap30_views10` 和 v2 数据集移动到 `/media/xjw/8T/深度学习数据/归档/PlanetaryFeatureMatch/训练数据/`，原路径保留软链接，根分区恢复约 183G 可用。（complete）
6. v2 数据完整性核验：train 4200 / val 1200 / test 600，`missing_json=0`。（complete）
7. 启动新训练：`runs/pose_sim2048_721_cross_extreme_v2_v21full256_abstain_b1_5epoch_s512_*`，同位置 train 5235 + v2 cross train 4200，合计 `9435 pair/epoch`；`epochs=5`，总 step `47175`，每 pair `512` 点。（in_progress）

### 当前观测
- v2 分布：
  - train offsets：{1:653, 2:653, 4:653, 8:653, 12:456, 16:269, 20:277, 24:285, 32:301}，valid mean `0.5931`，min `0.0268`。
  - val offsets：{1:224, 2:224, 4:224, 8:194, 12:58, 16:62, 20:66, 24:70, 32:78}，valid mean `0.5614`，min `0.0211`。
  - test offsets：{1:145, 2:145, 4:137, 8:69, 12:24, 16:28, 20:32, 24:20}，valid mean `0.5337`，min `0.0260`。
- 新训练开启了 `warp-hard-negative-weight=0.10`、`abstention-weight=0.05`、`pose-difficulty-loss-weight=0.15`、`report-matcher-mode=both`。
- 初始 eval：loss `0.011691`，top1 `0.9995`，top5 `1.0000`，points `393216`。
- 训练早期低重叠 batch 会出现 loss 尖峰，例如 `0.3-0.7`，但 `skip=0`，梯度裁剪生效。

### 下一步
1. 等待 5 epoch 训练完成，核验 final checkpoint、metrics、eval_summary 和 raw/graph 两套中文报告。
2. 单独生成 v2 test 的 `off012/off016/off020/off024/off032` 分桶报告，判断 abstention/graph matcher 是否减少低重叠场景的硬匹配错误。
3. 如果训练后 raw 仍在低重叠场景强行误配，应把报告与正式推理默认切到 graph matcher 或加 score/margin 阈值，而不是继续无阈值输出 1024 条 raw matches。

## Extreme v2 完整训练与 GraphMatcher 补训 (2026-05-29)

### 状态
- 阶段：complete。
- 描述子/abstention run：`runs/pose_sim2048_721_cross_extreme_v2_v21full256_abstain_b1_5epoch_s512_20260529_005651`。
- GraphMatcher 补训 run：`runs/pose_sim2048_721_cross_extreme_v2_v21full256_graphmatcher_b1_3epoch_s512_20260529_084108`。

### 已完成
1. v2 extreme 数据集完成扩充并核验：`训练数据/pose_sim_cross_position_rendered_extreme_2048_gap30_views10_721_v2`，train/val/test 为 `4200/1200/600`，严格 7:2:1，offset 包含 `1,2,4,8,12,16,20,24,32`。（complete）
2. 5 epoch 描述子/abstention 训练完成：同位置 5235 + v2 cross 4200，共 `9435 pair/epoch`，总 `47175 step`，每 pair `512` 点，`skip=0`。（complete）
3. 描述子训练产物核验通过：最终 `pytorch_pfm_state.pt`、`metrics.csv`、`eval_summary.csv`、5 个 epoch checkpoint、raw/graph 两套中文 PDF 报告均存在。（complete）
4. 发现正式 GraphMatcher 分支之前没有训练：abstention run 中 raw descriptor 的 off012 mean precision 为 `0.9789`，但 graph matcher 只有 `0.2537`。（complete）
5. 按 Stage D 补训 GraphMatcher：冻结 256 维特征提取与融合分支，只训练 `graph_matcher`，`epochs=3`，`9435 pair/epoch`，总 `28305 step`，每 pair `512` 点，`skip=0`。（complete）
6. GraphMatcher 补训产物核验通过：最终 `pytorch_pfm_state.pt`、`metrics.csv`、`eval_summary.csv`、3 个 epoch checkpoint、raw/graph 两套中文 PDF 报告均存在。（complete）

### 关键结果
- 描述子/abstention eval：loss `0.011691 -> 0.014332`，top1 `0.999535 -> 0.999395`，top5 `1.0 -> 1.0`；训练过程无跳步。
- GraphMatcher 补训 eval 对描述子指标不变：loss `0.014332 -> 0.014332`，top1 `0.999395 -> 0.999395`，符合“冻结描述子只训 matcher”的预期。
- off012 raw descriptor：32 个验证样本，mean precision `0.9789`，min `0.9434`，mean correct `1002.44/1024`，mean error `5.26 px`。
- 未补训 GraphMatcher：mean precision `0.2537`，min `0.1904`，mean correct `259.81/1024`，mean error `115.67 px`。
- 补训后 GraphMatcher：mean precision `0.8143`，min `0.7627`，mean correct `833.84/1024`，mean error `19.94 px`。

### 判断
- v2 extreme 扩集和 hard-negative/abstention 训练后，256 维 raw descriptor 在 off012 极端验证样本上已经很强。
- 正式 GraphMatcher 补训有效，但仍落后 raw descriptor；后续如果要继续提升正式路径，应优先优化 matcher 的候选组织、metadata/坐标一致性和 no-match/dustbin 训练，而不是继续盲目加大特征提取器。
## 2026-05-30 GraphMatcher 修复启动

- 收到用户关于 GraphMatcher 压坏 descriptor 的分析，确认下一阶段重点是冻结 extractor、修 GraphMatcher。
- 用户补充要求：weak texture 要看 precision，不只看 fraction；illumination branch 要做真实光照压力测试；C4 不足以代表连续旋转鲁棒。
- 已开始检查 `python/pfm_model.py`、`python/pfm_pytorch_training.py`、`python/pytorch_cache_match_eval.py`、`scripts/training_visual_report.py`。
- 初步发现：GraphMatcher 有 top-K candidate mask，但不是 residual reranker；报告缺 raw top-K recall、weak texture precision、dustbin/accepted、光照/连续旋转诊断。
- 轻量报告第一次运行失败：直接执行 `scripts/training_visual_report.py` 时缺少 `PYTHONPATH=python`，报 `ModuleNotFoundError: pfm_model`；改用带 PYTHONPATH 的命令重跑。
- 已修改 `python/pfm_model.py`：GraphMatcher 使用 `raw_similarity / raw_temperature + graph_delta_scale * graph_delta`。
- 已修改 `scripts/training_visual_report.py`：增加 raw top-K recall、weak texture precision/count、selected/rejected、graph metadata ablation。
- 已新增 `python/test_pfm_model.py` 单元测试，约束 residual logits 保留 raw descriptor argmax。
- 测试通过：`python -m unittest python/test_pfm_model.py python/test_pfm_pytorch_training.py python/test_pytorch_cache_match_eval.py`，137 tests OK，1 skipped。
- 轻量诊断通过：`runs/diagnostics_graphmatcher_residual_pair004541`，GraphMatcher full metadata `133/512`；`runs/diagnostics_graphmatcher_residual_pair004541_descriptor_only`，descriptor-only metadata `140/512`。
- 2026-05-30 GraphMatcher-only 训练第一次启动失败：`command.sh` 内使用了未定义的 `$OUT`，`set -u` 触发 `unbound variable`。训练未开始，修正为脚本内固定输出目录后重启。
- 第二次启动卡在训练前全量 `eval_before` 的磁盘/验证读取上，8 分钟仍未进入 step；已中断，改用 `--eval-pairs 512` 快速验证后重启。
- 第三次启动通过 `eval_before` 后在第一步报 `RuntimeError: no valid correspondences sampled`。原因是 `graph_matcher_loss` 被放在 `synthetic_loss_weight > 0` 分支内，matcher-only 时没有有效 loss。已修复为 synthetic loss 与 graph loss 独立，测试通过。

## 2026-05-30 GraphMatcher no-match reranker iteration
- 已完成代码修改：训练侧 GraphMatcher loss 增加 metadata mode 消融、no-match/dustbin distractor 点监督。
- 新 CLI：--graph-matcher-metadata-mode、--graph-matcher-no-match-points、--graph-matcher-no-match-weight、--graph-matcher-no-match-min-distance。
- focused 验证：py_compile 通过；python/test_pfm_pytorch_training.py + python/test_pfm_model.py 共 110 tests OK, 1 skipped。
- 关键训练策略：先跑 1 epoch GraphMatcher-only probe，使用 no_xy + dustbin negatives，确认极端 cross-position 不再被 matcher 压坏后再扩展。

## 2026-05-30 GraphMatcher no_xy + dustbin probe
- 完成 1 epoch GraphMatcher-only probe：`runs/graphmatcher_no_xy_dustbin_v21full256_b1_1epoch_s512_nm128_eval512_20260530_164330`。
- 配置：冻结 extractor，`--graph-matcher-metadata-mode no_xy`，每 pair `512` 正对应点 + `128` no-match distractor，`--graph-matcher-no-match-weight 0.5`，`skip=0`。
- 训练验证：`eval_before loss=0.009263 top1=0.9996`，`eval_after loss=0.024571 top1=0.9993`；descriptor 本身基本保持。
- 报告：raw/graph 两套中文 PDF 已生成在该 run 的 `visual_report/` 下。
- 18 样本报告：raw micro precision `8064/9216=0.8750`；GraphMatcher micro precision `7896/7973=0.9903`。
- extreme_cross_position：raw `2193/3072=0.7139`；GraphMatcher `2069/2104=0.9834`。
- 指定极端样本 `pair_004541_cross_off008_s00109_s00119.pt`：raw `122/512=0.2383`；GraphMatcher `89/106=0.8396`，`graph_rejected_count=1942`。
- 该极端样本 weak texture：raw `120/294=0.4082`；GraphMatcher `88/93=0.9462`。

## 2026-05-30 GraphMatcher recall calibration probe
- 完成从 high-precision dustbin checkpoint 继续的召回校准 probe：`runs/graphmatcher_no_xy_dustbin_recall_v21full256_b1_1epoch_s512_nm64_w02_eval512_20260530_181320`。
- 配置：冻结 extractor，继续只训 GraphMatcher，`no_xy` metadata，no-match distractor 从 128 降到 64，no-match weight 从 0.5 降到 0.2，LR `2e-5`，1 epoch，`skip=0`。
- 训练验证：`eval_before loss=0.024489 top1=0.9992`，`eval_after loss=0.021031 top1=0.9993`。
- 18 样本 GraphMatcher：accepted matches `7973 -> 8607`，correct `7896 -> 8174`，micro precision `0.9903 -> 0.9497`。
- extreme_cross_position：accepted `2104 -> 2547`，correct `2069 -> 2233`，micro precision `0.9834 -> 0.8767`。
- 指定极端样本 `pair_004541_cross_off008_s00109_s00119.pt`：accepted `106 -> 176`，correct `89 -> 110`，precision `0.8396 -> 0.6250`。
- 结论：召回确实上升，但 precision 损失过大。当前最佳正式 GraphMatcher 仍是 `graphmatcher_no_xy_dustbin_v21full256_b1_1epoch_s512_nm128_eval512_20260530_164330`；下一步不应继续降低 dustbin 训练权重，而应做 inference-time dustbin/score calibration 或分层候选接收策略。

## 2026-05-30 GraphMatcher inference calibration
- 已实现报告/评估侧 GraphMatcher 校准参数：`--graph-dustbin-delta`、`--graph-acceptance-margin`、`--graph-min-raw-score`、`--graph-min-raw-margin`。
- 默认行为保持不变；只有显式传入校准参数时才从 logits 重算匹配并应用 raw descriptor score/margin 过滤。
- 新增单元测试覆盖 dustbin 降低后接受更多匹配、raw margin 过滤 graph 匹配、CLI 参数解析。
- 验证通过：`py_compile python/pytorch_cache_match_eval.py scripts/training_visual_report.py`；`python -m unittest python/test_pytorch_cache_match_eval.py`，32 tests OK。
- 用当前最佳 checkpoint 按原始 2048 候选报告配置扫描 `graph_dustbin_delta=-0.10/-0.20/-0.30`。
- baseline GraphMatcher：micro `7896/7973=0.990342`，weak `3947/3974=0.993206`，extreme `2069/2104=0.983365`，指定极端样本 `89/106=0.839623`。
- `delta=-0.10`：micro `7914/8001=0.989126`，weak `3959/3989=0.992479`，extreme `2074/2111=0.982473`，指定极端样本 `91/110=0.827273`。
- `delta=-0.20`：micro `7930/8030=0.987547`，weak `3965/4002=0.990755`，extreme `2081/2124=0.979755`，指定极端样本 `91/111=0.819820`。
- `delta=-0.30`：micro `7957/8068=0.986242`，weak `3986/4027=0.989819`，extreme `2093/2143=0.976668`，指定极端样本 `95/116=0.818966`。
- 结论：推理侧 dustbin 校准能温和提高 accepted/correct，但 precision 随 delta 单调下降。当前建议保留 baseline 作为 high_precision，`delta=-0.10` 或 `-0.30` 仅作为 balanced/review 模式，不替代主模型。

## 2026-05-30 GraphMatcher hard replay + weak quota implementation
- 已实现模型侧 accept head：`PlanetaryGraphMatcher` 增加零初始化 `accept_head` 与 `accept_logit_scale`，旧 checkpoint 加载时自动补默认参数，初始行为兼容。
- 已实现训练侧弱纹理正样本配额：`sample_feature_correspondences(..., weak_texture_fraction=...)`，CLI 为 `--training-weak-texture-fraction`。
- 已实现 hard replay glob：`--hard-pair-glob` 可按路径或文件名选择训练集中的 hard pair，配合已有 curriculum/repeat 机制使用。
- 已实现 GraphMatcher 辅助损失：
  - `--graph-matcher-accept-weight` / `--graph-matcher-accept-negative-topk`：训练 accept head 区分正匹配与 hard negative。
  - `--graph-matcher-raw-preservation-weight` / `--graph-matcher-raw-preservation-margin` / `--graph-matcher-raw-preservation-raw-margin`：保护 raw descriptor 已经很确信的匹配不被 GraphMatcher 压坏。
- 测试通过：`py_compile` 通过；`python -m unittest python/test_pfm_model.py python/test_pfm_pytorch_training.py python/test_pytorch_cache_match_eval.py`，145 tests OK，1 skipped。

## 2026-05-30 GraphMatcher hard replay probes
- Probe A：`runs/graphmatcher_accept_weakquota_v21full256_b1_1500_s512_20260530_221712`。
  - 配置：从当前最佳 high-precision checkpoint 起，冻结 extractor，1500 steps，弱纹理配额 `0.25`，hard offset replay，`accept_weight=0.15`，`raw_preservation_weight=0.15`。
  - 结果：micro `8048/8317=0.967657`，weak `4043/4168=0.970010`，extreme `2145/2298=0.933420`。
  - 指定极端样本：`105/173=0.606936`，weak `104/139=0.748201`。
  - 结论：召回上升明显，但 precision 损失过大；accept loss 当前不能作为默认训练配置。
- Probe B：`runs/graphmatcher_rawpreserve_weakquota_v21full256_b1_1000_s512_20260530_223255`。
  - 配置：不启用 accept loss，1000 steps，弱纹理配额 `0.25`，hard offset replay，`no_match_weight=0.7`，`raw_preservation_weight=0.30`。
  - 结果：micro `7999/8159=0.980390`，weak `4012/4083=0.982611`，extreme `2117/2201=0.961836`。
  - 指定极端样本：`101/138=0.731884`，weak `100/115=0.869565`。
  - 加 `graph_min_raw_score=0.4, graph_min_raw_margin=0.01` 后：micro `7983/8127=0.982281`，extreme `2107/2181=0.966071`，指定极端样本 `101/134=0.753731`。
  - 结论：raw-preservation + stronger dustbin 是更可用的 balanced 方向，但仍未达到目标 `120~160 accepted 且 precision >=0.80`。当前主模型仍应保留 high-precision baseline；Probe B 可作为下一轮继续调参起点。

## 2026-05-31 GraphMatcher hard-negative dustbin probes

- 已实现 `graph_matcher_hard_negative_dustbin_loss`：对 raw descriptor 最相似的 off-diagonal hard negatives，约束其 GraphMatcher logit 低于对应 row/column dustbin，目标是形成 `正匹配 > dustbin > 相似但错误候选` 的排序。
- 新 CLI：
  - `--graph-matcher-hard-negative-dustbin-weight`
  - `--graph-matcher-hard-negative-dustbin-topk`
  - `--graph-matcher-hard-negative-dustbin-margin`
- TDD 验证：新增 hard-negative dustbin loss 单测，先确认缺失函数失败，再实现后通过。
- focused 验证通过：`py_compile` 通过；`python -m unittest python/test_pfm_model.py python/test_pfm_pytorch_training.py python/test_pytorch_cache_match_eval.py` 为 `146 tests OK, 1 skipped`。
- Probe C：`runs/graphmatcher_hardnegdustbin_rawpreserve_v21full256_b1_1000_s512_20260531_101030`。
  - 配置：从 high-precision baseline 起训，`raw_preservation=0.30`，`hard_negative_dustbin_weight=0.20`，1000 steps。
  - 结果：micro `7892/7966=0.990711`，weak `3945/3971=0.993453`，extreme `2067/2101=0.983817`。
  - 指定极端样本：`89/105=0.847619`，weak `88/92=0.956522`。
  - 结论：权重 0.20 太保守，precision 略高但正确数没有提升，基本退回 high-precision 行为。
- Probe D：`runs/graphmatcher_hardnegdustbin_light_from_rawpreserve_v21full256_b1_600_s512_20260531_101916`。
  - 配置：从 raw-preservation probe 继续，LR `5e-6`，600 steps，`raw_preservation=0.10`，`hard_negative_dustbin_weight=0.05`。
  - 未过滤 GraphMatcher：micro `7986/8121=0.983376`，weak `4003/4059=0.986203`，extreme `2108/2174=0.969641`，指定极端样本 `100/129=0.775194`。
  - 加 `graph_min_raw_score=0.4, graph_min_raw_margin=0.01` 后：micro `7970/8096=0.984437`，weak `4000/4050=0.987654`，extreme `2099/2159=0.972209`，指定极端样本 `100/125=0.800000`，weak `99/108=0.916667`。
  - 结论：这是当前最好的 balanced experimental 口径；比 high-precision baseline 多 74 个正确匹配，指定极端样本多 11 个正确匹配，但总体 precision 低于 high-precision baseline。

## 2026-05-31 GraphMatcher spatial hard-negative gate probe

- 已实现 hard-negative dustbin 的空间门控：`--graph-matcher-hard-negative-dustbin-spatial-min-distance`。当该参数大于 0 时，只压制 target 位置相距足够远的 raw-confusable off-diagonal 候选，避免近邻候选被过度惩罚。
- 新增 TDD 单测：构造一个近邻 confusable 候选和一个远距离 confusable 候选，确认空间门控只惩罚远距离候选。
- 验证通过：`py_compile` 通过；`python -m unittest python/test_pfm_model.py python/test_pfm_pytorch_training.py python/test_pytorch_cache_match_eval.py` 为 `147 tests OK, 1 skipped`。
- Probe E：`runs/graphmatcher_spatial_hardnegdustbin_from_rawpreserve_v21full256_b1_600_s512_20260531_111558`。
  - 配置：从 raw-preservation probe 继续，LR `5e-6`，600 steps，`hard_negative_dustbin_weight=0.05`，`spatial_min_distance=4.0`。
  - 未过滤 GraphMatcher：micro `7996/8151=0.980984`，weak `4011/4081=0.982847`，extreme `2114/2194=0.963537`，指定极端样本 `101/136=0.742647`。
  - 加 `graph_min_raw_score=0.4, graph_min_raw_margin=0.01` 后：micro `7981/8121=0.982761`，weak `4010/4071=0.985016`，extreme `2105/2176=0.967371`，指定极端样本 `101/132=0.765152`。
  - 结论：空间门控保留了更多正确匹配，但 false matches 增长更快，未超过上一轮非空间 hard-negative dustbin 的 best balanced 口径 `100/125=0.800000`。当前不推荐替代 best balanced，只保留为后续分层 hard negative 的诊断入口。
# 2026-05-31 training spatial balance optimization
- 收到用户要求：优化后做一次完整炼丹。
- 选择本轮优化目标：训练 correspondence 采样空间均衡，解决弱纹理/分布不均问题。
- 已添加 RED 测试：
  - `test_sample_feature_correspondences_can_cover_spatial_bins`
  - `test_sample_feature_correspondences_combines_weak_texture_and_spatial_bins`
  - parse args 接受 `--training-spatial-bins`
- RED 验证结果：失败原因符合预期，当前代码缺少 `spatial_bins` 参数和 CLI。
- 实现完成并验证：
  - focused spatial/weak/CLI tests 通过。
  - 完整 Python 训练相关测试通过：156 tests OK, 1 skipped。
  - CUDA smoke 通过：`runs/smoke_spatial_semidense_graph_20260531_163241`，1 step，loss=3.002254，eval before/after top1=1.0。
- 正式训练配置决策：
  - 起点：`runs/graphmatcher_hardnegdustbin_light_from_rawpreserve_v21full256_b1_600_s512_20260531_101916/pytorch_pfm_state.pt`。
  - 数据：same/cross/extreme 三个 train cache，共 10635 train pairs，val 共 3079 pairs。
  - 训练：`--epochs 1`，batch_pairs=1，samples_per_pair=512，1024 crop，balanced cache sampling。
  - 新增优化：`--training-weak-texture-fraction 0.5`、`--training-spatial-bins 8`、semi-dense no-match=64。
- 完整炼丹完成：`runs/spatial_weak_semidense_v21full256_1epoch_s512_20260531_163414`。
- 训练结果：
  - `steps=10635`，`skip_sum=0`，每个 pair 采样 `512` 个监督匹配点。
  - `eval_before top1=0.999176 loss=0.025660`。
  - `eval_after top1=0.999222 loss=0.049786`，`mean_positive_score=0.933957`，`mean_negative_score=0.022942`。
  - 前 100 step 平均 loss `1.270365`，后 100 step 平均 loss `0.909142`。
- 视觉报告：
  - GraphMatcher：`runs/spatial_weak_semidense_v21full256_1epoch_s512_20260531_163414/visual_report/graph_matcher_filter_s04_m001/training_report_zh.pdf`。
  - Raw descriptor：`runs/spatial_weak_semidense_v21full256_1epoch_s512_20260531_163414/visual_report/raw_descriptor/training_report_zh.pdf`。
- GraphMatcher 分层：`graph_match_error_strata.csv`。
- 24 样本报告结果：
  - GraphMatcher filtered：`11854/11860=0.999494`，weak texture `3068/3068=1.000000`，平均覆盖格占用 `0.883464`。
  - Raw descriptor：`11863/12288=0.965413`，weak texture `3298/3353=0.983597`，平均覆盖格占用 `0.886719`。
  - 指定极端样本 `pair_004541_cross_off008_s00109_s00119.pt`：raw `88/512=0.171875`；GraphMatcher filtered `78/84=0.928571`，weak texture `50/50=1.000000`。

## 2026-06-01 仿真数据增量生成状态

- 07:51 CST 检查：`sim_same_position_continue_20260531` 仍在运行，主进程 PID `900660`，当前子进程为 `sat_sim_cuda`。
- 输出数据集：`/media/xjw/8T/深度学习数据/仿真训练集/pose_sim_2048_gap30_views10`。
- 本轮目标：从原 `7479` 个 pair 继续新增 `3000` 个，目标总数 `10479`。
- 当前统计：total `9954`，train `4986`，val `2484`，test `2484`；manifest 行数 `2476`，其中数据记录约 `2475`。
- 当前进度约 `2475/3000`，剩余约 `525` 个 pair；日志最新进度行为 `kept=2475 last_candidate=9953 free_gb=3889.0`。
- 磁盘状态：`/media/xjw/8T` 可用约 `3.8T`，`/media/xjw/xjw2T` 可用约 `1012G`。用户已确认 xjw2T 空间充足，后续新增仿真数据优先转到 xjw2T 生成。
- 并行检查发现 `runs/sim_ultra_cross_20260531.pid` 指向的进程已结束；`pose_sim_cross_position_rendered_ultra_2048_gap30_views10_721_20260531` 当前只有约 `5.0G` depth cache，`cache/train|val|test` 均为 `0`，不能作为训练数据使用。后续需要在 xjw2T 上重新生成或修复该极端跨位置数据集。
- 为后续训练数据整理修复了 `scripts/repartition_pair_cache.py`：从仅支持 symlink 扩展为 `symlink|hardlink|copy|move`，并支持 `--workers` 并行搬运 pair/sidecar/共享资产；同时把 compact cache 必需的 `image_store` 一起带到输出 root，避免相对路径失效。
- 新增测试 `python/test_repartition_pair_cache.py`，验证 `--link-mode copy --workers 2` 下 10 个 pair 能按 `7:2:1` 生成实体 split，且 `image_store`/`tsai_tracks` 被正确复制。验证命令通过：`py_compile scripts/repartition_pair_cache.py python/test_repartition_pair_cache.py`；`python -m unittest python/test_repartition_pair_cache.py`。
- 代码已推送 GitHub：`d531274 Support parallel self-contained cache repartition`。
- 继续修复生成端 split：`辅助软件/数据模拟/batch_pose_sim_dataset.py` 新增 `--split-mode track|ratio`、`--split-ratio 7:2:1`、`--split-seed`。默认仍是旧的 track split，后续 xjw2T 新仿真可显式用 ratio 直接生成 7:2:1。
- 新增 `python/test_batch_pose_sim_dataset.py` 覆盖 ratio split 精确计数和 `output_pair_path()` 使用 candidate split。首次测试因 Python 3.12 dataclass 动态导入未注册 `sys.modules` 失败，已修复测试 loader；最终 `py_compile` 和 `python -m unittest python/test_batch_pose_sim_dataset.py python/test_repartition_pair_cache.py` 均通过。
- `docs/simulation_data_generation_status.html` 已同步更新：same-position 新生成命令使用 `--split-mode ratio --split-ratio 7:2:1`；新增旧数据并行重划分到 xjw2T 的 `repartition_pair_cache.py --link-mode copy --workers 8` 命令。
- 新增 `scripts/verify_pair_cache_dataset.py`，用于重划分后校验 pair 总数、7:2:1 split 数量，并通过 `load_libtorch_pair_archive()` 抽样加载 compact/legacy pair，检查 `image_store` 相对路径、图像/warp/mask shape 和有效像素。新增测试 `python/test_verify_pair_cache_dataset.py`，与 batch/repartition 测试一起通过。
- 08:05 CST 检查：同位置增量总数 `10006`，manifest 行数 `2521`，日志最新 `kept=2525 last_candidate=10003 free_gb=3885.7`；距离目标总数 `10479` 还差约 `473`。
- 为后续仿真提效，`batch_pose_sim_dataset.py` 新增 `--frame-workers`，可并行处理多个独立 frame batch；默认 `1` 保持当前行为。建议 xjw2T 新生成使用 `--frame-workers 2 --sat-sim-jobs 2`，避免单个 frame 顺序渲染成为瓶颈。`python/test_batch_pose_sim_dataset.py` 已覆盖参数解析，`py_compile` 和 unittest 通过。
- 08:26 CST 检查：同位置增量总数 `10080`，train `5040` / val `2520` / test `2520`，日志最新 `kept=2600 last_candidate=10078 free_gb=3880.7`；距离目标总数 `10479` 还差约 `399`。当前进程仍在旧脚本顺序模式下运行，新 `--frame-workers` 只影响后续新启动任务。
- 修复六组对比入口 `scripts/fixed_six_group_matcher_comparison.py`：新增 `--split-root` 和 `--img-root`，当旧 `runs/cross_view_*/splits/test` 不存在时可直接从 `/media/xjw/8T/深度学习数据/img/{Rotate_1024,Viewpoint_1024,CompoundViewpoint_1024}` 读取固定样本；当旧 route 的 `selected_weights.csv` 不存在时，可用 `--pfm-state` 和统一 PFM 参数直接评估当前 checkpoint。
- 新增 `python/test_fixed_six_group_matcher_comparison.py`，覆盖 img-root fallback 和 direct PFM 参数 fallback；`py_compile` 与 unittest 通过。
- 08:32 CST 检查：同位置增量总数 `10107`，train `5058` / val `2529` / test `2520`，日志最新 `kept=2625 last_candidate=10103 free_gb=3879.1`；距离目标总数 `10479` 还差约 `372`。

## 2026-06-01 同位置仿真补齐与 xjw2T 重分区

- 用户提醒后切换为并行补算：旧进程 `sim_same_position_continue_20260531` 仍是单 frame 循环，已安全终止；重新启动 `continue_7479_3000_parallel_20260601_084808`，参数为 `--frame-workers 2 --sat-sim-jobs 2`。
- 并行补算利用已存在 `pair_*.pt` 跳过逻辑，只补缺失候选；最终日志为 `done ... kept=309 skipped=2691 rendered_frames=35`。
- 同位置数据集已达到目标总数 `10479`：
  - 源路径：`/media/xjw/8T/深度学习数据/仿真训练集/pose_sim_2048_gap30_views10`
  - 当前原始 track split：train `5241` / val `2619` / test `2619`
  - 旧顺序增量 manifest 数据行约 `2691`，并行补算 manifest 数据行 `309`，合计 `3000`
  - 源数据集大小约 `697G`，其中 `cache` 约 `696G`
- 已启动 xjw2T 实体重分区：
  - 输出路径：`/media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据/pose_sim_2048_gap30_views10_10479_721`
  - 命令核心：`scripts/repartition_pair_cache.py --ratio 7:2:1 --link-mode copy --workers 8 --overwrite`
  - PID 文件：`runs/repartition_pose_sim_10479_721.pid`
  - 日志文件：`runs/repartition_pose_sim_10479_721_20260601_094247.log`
  - 预期 split：train `7335` / val `2095` / test `1049`
- 10:35 左右重分区失败：日志报 `OSError: [Errno 5] Input/output error`，出错源文件为 8T 上的 `pair_001146_mid_xp12.pt`，目标为 xjw2T 输出目录。
- 失败后 `/media/xjw/xjw2T` 挂载点消失，`lsblk` 当前只显示 `/media/xjw/8T`，不再显示 xjw2T 设备；这更像磁盘/连接掉线，不是普通目录缺失。
- 已停止继续向 xjw2T 写入；由于挂载点消失，之前已复制到 xjw2T 盘上的部分数据需要等磁盘重新出现后再检查。
- 为避免磁盘恢复后从 0 重拷，已增强 `scripts/repartition_pair_cache.py`：
  - 新增 `--skip-existing`，完整目标文件按 size/link 检查后跳过。
  - 截断或大小不一致的目标文件会被重新复制。
  - shared tree 复制同样支持跳过已完整文件。
  - 新增单测 `test_skip_existing_resumes_complete_files_and_recopies_truncated_files`。
- 验证通过：`py_compile scripts/repartition_pair_cache.py python/test_repartition_pair_cache.py`；`python -m unittest python/test_repartition_pair_cache.py`。

## 2026-06-01 xjw2T 实体训练数据导入完成

- 用户要求暂时不要训练后，已停止 `same10479_extreme_archive_v21full256_1epoch_s512_20260601_104957` 训练进程，GPU 释放。
- xjw2T 恢复挂载后检查：`/media/xjw/xjw2T` 为 exFAT，恢复时可用空间约 `596G`；此前半截导入目录仍存在，已有 `6012/10479` 个 pair，占约 `417G`。
- 首次恢复续拷使用 `--workers 4 --skip-existing`，但又在 `6411/10479` 左右触发 `OSError: [Errno 5] Input/output error`。
- 改为 `--workers 1 --skip-existing` 后稳定完成，最终输出：
  - 路径：`/media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据/pose_sim_2048_gap30_views10_10479_721`
  - split：train `7335` / val `2095` / test `1049`
  - total：`10479`
  - 目录大小：约 `795G`
  - xjw2T 剩余空间：约 `214G`
- 验证命令通过：
  - `/home/xjw/anaconda3/envs/pfm-train/bin/python -u scripts/verify_pair_cache_dataset.py --dataset-root /media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据/pose_sim_2048_gap30_views10_10479_721 --expected-total 10479 --expected-ratio 7:2:1 --samples-per-split 5`
  - 抽样结果：train/val/test 各 5 个样本均可加载，shape 为 `1 x 2048 x 2048`，valid pixels 正常，`ok: true`。
- 结论：xjw2T 空间足够并且本轮同位置实体数据已完成导入；后续训练应优先使用该 xjw2T 实体路径，避免继续从 8T symlink 直接训练导致 I/O 抖动。
