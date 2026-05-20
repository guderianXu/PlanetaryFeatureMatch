# 模块化训练基础设施与匹配 loss 优化计划

## 目标
继续推进 reusable training infrastructure，先稳定 runtime/dataloader 基础设施和全量测试，再继续 logging、augment、trainer DataLoader 集成；匹配 loss 优化依赖更可靠的数据加载和训练指标。

## 当前阶段
- 阶段 1：runtime BlockingQueue/ThreadPool（complete）
- 阶段 2：dataloader sampler/split/collator/pinned memory（complete）
- 阶段 3：AsyncDataLoader 稳定性修复（complete）
- 阶段 4：最终验证当前基础设施（complete）
- 阶段 5：training logging module（complete）
- 阶段 6：optional NVML hook（complete）
- 阶段 7：augmentation module 抽取（complete）
- 阶段 8：trainer logging/DataLoader 集成（complete）
- 阶段 9：docs 与 matcher-specific data strategy/loss 后续优化（complete）
- 阶段 10：模块边界整理 — 项目代码从 modules/ 迁移到 src/（complete）
- 阶段 11：修复 graph_matching_loss + 接入 ConsoleProgressLogger 进度条（complete）
- 阶段 12：训练 loss 诊断修复 — B 侧关键点 warp 采样 + 进度条特征指标 + 相关性窗口扩容（complete）

- 阶段 13：keypoint graph matching loss 优化（complete）
  - 已完成：Task 1 graph matcher 使用 keypoint projection；Task 2 warp+mask 目标分配；Task 3 deterministic candidate set；Task 4 decoded sparse keypoint graph loss 接入训练。
  - 已验证：`cmake` 配置、完整 build、`pfm_tests` 290 tests passed、`ctest` 100% passed。
  - 已完成：Task 5 CUDA 短训练复现结束，完整 4140 条 CSV iteration 显示 `graph_matching_loss` first_mean≈4.6967、last_mean≈0.0335，趋势明显下降。

- 阶段 14：完整 CUDA 训练验证（complete）
  - 已完成：main 分支 10 epoch CUDA 完整训练。
  - 结果：13,800 条 iteration；graph_matching_loss first_mean≈2.17615、last_mean≈0.10262；descriptor_accuracy first_mean≈0.374187、last_mean≈0.95367。
  - 产物：`train_full.pt`、`metrics_full.csv`、`vis_full/` 保留为本地训练输出，不默认提交。

- 阶段 15：checkpoint 推理评估与 sparse match 过滤（complete）
  - 已完成：修复 CUDA match device mismatch；推理 sparse matches 增加 dustbin + mutual nearest 过滤。
  - 已验证：`pfm_tests` 293 tests passed；真实图像 100-101/100-110/100-118 sparse matches 降到 44/67/43。

- 阶段 16：trainer 训练/验证划分模块接入（complete）
  - 已完成：trainer 复用 `modules/dataloader/sampler` 的 `make_train_validation_test_split()` 生成 train/validation indices。
  - 已验证：`pfm_tests` 294 tests passed，`ctest` 100% passed。

- 阶段 17：极端旋转泛化修复（in_progress）
  - 已完成：mixed augmentation 增加 deterministic ±180° half-turn 样本；graph matcher keypoint projection 改为归一化坐标。
  - 已验证：`pfm_tests` 296 tests passed，`ctest` 100% passed。
  - 已完成：新增单图旋转 sweep 工具并用 `img/100.tif` + `train_full.pt` 跑 0..330° 每 30° sparse 匹配评估。
  - 关键结论：0° 自匹配 sparse 几何通过率也为 0，整圈旋转基本失败；先修 identity/known-transform sparse matching，再继续长训练。
  - 已完成：rotation sweep 增加 raw descriptor mutual nearest 与 keypoint repeatability 诊断列。
  - 已完成：descriptor loss 增加全局采样点对比、decoded sparse keypoint hard negatives，并修复训练 decode 配置复用用户 `max/min_keypoints`。
  - 关键结论：单图 40 epoch 过拟合后 0° 已稳定 1024/1024 正确，但 90/180/270° 仍没有可靠 sparse match；下一步应改特征提取器的显式旋转等变/规范化机制，而不是继续只调普通 descriptor CE。
  - 已完成：`SparseHead` descriptor 分支加入 C4 harmonic descriptor bands，并分别验证 C4 均值池化、统计投影、谐波幅值三种特征提取器改法。
  - 最新结论：三种特征提取器改法均未让单图 90/180/270° 产生有效 sparse matches；谐波版本提高了真实重复点 descriptor score，但错误 mutual score 仍更高。下一步应实现跨图 hard-negative/margin descriptor loss 或 orientation-supervised canonical descriptor，而不是继续只改 pooling。
  - 已完成：sparse keypoint descriptor hard-negative 覆盖扩到 1024 queries，并将 margin weight 提到 5 重新训练/评估。
  - 最新结论更新：`train_rotation100_margin5.pt` 在 0° 自匹配正常，但 90/180/270° 仍为 0 sparse matches；当前路线应转向 rotation-aware descriptor matching 或显式 orientation/canonicalization，而不是继续提高 hard-negative 权重。
  - 已完成：`SparseHead` 改为 C4 cyclic descriptor slots，descriptor loss 与 fallback matching 加入 4-way cyclic shift 相似度。
  - 最新结论更新：`train_rotation100_cyclic.pt` 仍未通过 90/180/270°；当前阶段应转向直接 rotation-sweep hard-negative 监督或 orientation-supervised canonical descriptor。
  - 已完成：keypoint-to-full-map descriptor hard-negative 监督与单图训练/评估。
  - 最新结论更新：重建 `pfm_cli` 后 `train_rotation100_keydense.pt` 在 90/180/270° 有 fallback sparse matches，但几何通过率仍只有约 1%-3%。下一步转向 orientation-supervised canonical descriptor 或加强关键点 repeatability 几何监督。
  - 已完成：orientation-supervised canonical descriptor 原型与单图训练/评估。
  - 最新结论更新：`train_rotation100_orientcanon.pt` 在 90/180/270° 几何通过率约 5.3%/3.1%/5.3%，比 keydense 略好但仍不可用；训练 feature loss 约 4.74，不能视为低 loss 或有效收敛。
  - 已完成：排查并修正 mixed 训练数据中的极端旋转 anchor。旧 half-turn 样本几何方向正确但叠加了 scale/translation，监督不够干净；现已改为纯 ±90°/±180° anchor 并用测试验证 half-turn warp 交叉。
  - 已完成：用 clean rotation anchors 重新训练；180° 仍只有约 2.42% 通过率，线型仍接近平行。
  - 已完成：训练 variant 随 epoch 推进，graph matcher keypoint embedding 改为 radius/radius^2，减少同屏幕位置捷径；验证 317 tests passed。
  - 最新结论：60 epoch 单图训练后 180° 通过率仅约 3.64%，可视化仍不是 X 形；当前 learned descriptor/matcher 架构仍不合格。
  - 下一步：在更快机器上先跑 SIFT/ORB 180° baseline，再决定是否改成 orientation-normalized local patch descriptor 或 rotation-sweep hard-negative 监督。

## 设计决策
- B 侧描述子和关键点从 warp 后目标位置采样，target 改为恒等映射（A[i]→B[i]），图匹配器可同时利用空间和描述子信号
- 进度条补充 feature_loss、repeatability_loss、descriptor_accuracy 等特征提取器指标
- CORRELATION_RADIUS 从 2 扩到 4：局部相关性从 25 通道扩展到 81 通道，扩大稠密匹配的搜索范围
- PyTorch intra-op 并行在 gather/stack 操作上触发 futex 死锁，test_main 中 setenv OMP_NUM_THREADS=1 规避
- 先修复 AsyncDataLoader 崩溃，解除全量测试阻塞，再继续 trainer/logging/augment 模块。
- `AsyncDataLoader::reset()` 必须具备强异常安全：sampler 抛异常后 loader 进入 exhausted 安全态，后续 `next()` 返回空而不是访问空队列。
- logging module 先提供可复用的 metric record、console progress、CSV sink、GPU provider 接口和 logger group；trainer 集成在后续阶段单独完成。
- matching loss 的进一步优化放在基础设施稳定之后，用 CSV/日志和 DataLoader sampler 支持诊断 `graph_matching_loss` 高波动。

## 文件计划
- `modules/dataloader/async_dataloader.cpp`：修复 reset 失败后的状态。
- `modules/logging/*`：训练日志、CSV、logger group 与 optional NVML provider 已完成。
- `modules/augment/*`：图像对增强模块已从旧 synthetic pair 逻辑抽取完成。
- `modules/dataloader/synthetic_pair_dataset.*`：online synthetic pair TensorDataset 已完成。
- `modules/train/trainer.*`：后续 trainer logging/DataLoader 集成。
- `README.md`、`docs/training.md`、`docs/usage.md`：最终文档。

## 错误记录
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 全量 `./build/pfm_tests` 段错误，gdb 指向 `AsyncDataLoader::next()` 中 `_queue->pop()` | 1 | 复现后修复 `reset()`：先清空并设置 `_exhausted=true`，仅在 sampler 和 async epoch 初始化成功后设置为 false |
| logging RED 构建失败：缺少 `modules/logging/csv_metric_logger.cpp` | 1 | 新增 logging headers/sources 并接入 `pfm_logging` target |
| augmentation module 抽取后静态库链接缺少 geometry/data 符号 | 1 | 将 `pfm_augment` 改为 OBJECT target 并把对象文件并入 `pfm` |
| `pfm_augment` OBJECT target 编译缺少 Torch include | 1 | 给 `pfm_augment` 链接 `${TORCH_LIBRARIES}` 和 `${OpenCV_LIBS}` 以继承 include/link usage |
| synthetic pair dataset RED 构建失败：缺少 `modules/dataloader/synthetic_pair_dataset.cpp` | 1 | 新增 `SyntheticPairTensorDataset` 并接入 `pfm_dataloader` target |

## Keypoint graph matching loss 优化 (2026-05-19)

### 已完成
- 分支：`feat/keypoint-graph-matching-loss`。
- 提交：
  - `e34a3ab Use keypoints in graph matcher`
  - `a2551fa Clamp graph matcher keypoint embeddings`
  - `b24ae6c Preserve graph matcher keypoint coordinates`
  - `23a2545 Clarify graph matcher keypoint preparation`
  - `37c8c64 Add keypoint graph target assignment`
  - `e3a2015 Build graph matching candidate sets`
  - `0bd060b Train graph matcher on decoded keypoints`
- 核心改动：graph matcher 不再忽略 keypoints；训练 graph loss 改为 decoded sparse keypoints + warp positives + deterministic negatives + dustbin。

### 完成状态
- CUDA 短训练已完成且 graph loss 明显下降。
- 该阶段优化目标已达成；如后续需要继续提升匹配质量，再考虑更强的双向/Sinkhorn-style objective。

### 错误记录
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| Task 4 初次接入 decoded features 后大量训练测试失败：`dense_confidence spatial size must match heatmap` | 1 | 复用可视化路径思路，将 dense confidence resize 到 sparse heatmap 尺寸后再 decode |
| decoded keypoints 是 feature-map 坐标而 warp/mask 是 image 坐标 | 1 | target assignment 前按 feature_map_width/height 缩放到 image 坐标，matcher 输入仍保留 inference 使用的 feature-map 坐标 |
