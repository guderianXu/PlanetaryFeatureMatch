# 模块化训练基础设施与匹配 loss 优化计划

## 目标
继续推进 reusable training infrastructure，先稳定 runtime/dataloader 基础设施和全量测试，再继续 logging、augment、trainer DataLoader 集成；匹配 loss 优化依赖更可靠的数据加载和训练指标。

## 真实 DEM depth 跨相机位置训练（2026-05-28）
- 阶段：complete。
- 目标：按用户确认的“全火 DEM/DOM + 相机参数可得到真实地理对应”路线，用真实渲染 depth 生成跨相机位置匹配对，并混合同中心姿态/焦距数据继续炼 PFM v2.1-full 256 维模型。
- 已完成：
  1. `scripts/generate_cross_position_pose_pairs.py` 支持 `--depth-mode rendered`，用 `sat_sim_cuda --write-depth` 生成 CameraA depth，并从真实 DEM depth 反投影/再投影得到 dense warp。（complete）
  2. depth cache 预渲染改为并行 worker，正式数据集生成完成：`训练数据/pose_sim_cross_position_rendered_2048_gap30_views10`，train 1200 / val 384，depth cache 624。（complete）
  3. 用同中心原始 cache + 真实 depth cross cache 混合训练 3000 steps：`runs/pose_sim2048_crosspos_rendered_mix_v21full256_blend_b1_3000_20260528_105130`。（complete）
  4. 自动生成中文 PDF 报告和 24 张匹配可视化，报告路径：`runs/pose_sim2048_crosspos_rendered_mix_v21full256_blend_b1_3000_20260528_105130/visual_report/training_report_zh.pdf`。（complete）
- 验证：
  - checkpoint 存在且 `config.descriptor_dim=256`，文件大小约 230MB。
  - `eval_summary.csv`：loss `0.007551 -> 0.007189`，top1 `0.999718 -> 0.999786`，top5/top10 均为 `1.0`。
  - `metrics.csv`：3000 steps，`skip=0`，last100 top1 均值 `0.999766`。
  - `match_visual_summary.csv`：24 样本平均 precision `0.999430`，平均正确数 `1023.42/1024`，平均覆盖率 `0.810058`，弱纹理匹配占比 `0.293579`。
- 结论：
  - 真实 DEM depth cross 数据没有复现球面近似 cross 数据的远基线崩坏，说明之前主要问题之一是 GT 几何近似不够准。
  - 当前报告抽样中的 4 个真实 cross val 样本均为 `1024/1024` correct；下一步应做 offset 分桶全量评估，而不是只看随机 24 样本。

## 训练可视化与中文 PDF 报告补强（2026-05-27）
- 阶段：complete。
- 目标：把当前 PyTorch PFM 训练 run 的输出升级为类似 YOLO 的可读训练结果包，包含更完整的训练指标、匹配点质量指标、更多匹配可视化，以及中文 PDF 报告。
- 用户反馈：
  1. 当前曲线和 8 张匹配图指标不足以判断训练质量。
  2. 希望训练完成后自动生成中文 PDF 报告。
  3. 当前训练效果仍差，匹配点太少，loss 居高不下。
- 计划：
  1. 扩展 `scripts/training_visual_report.py`，补充匹配质量指标：正确/错误数、precision、correct-per-sample、错误距离均值/中位数/P90、score 分布、正确/错误 score 分离、pose difficulty 分组。（complete）
  2. 增加更密集匹配图生成参数，默认提升 sample 数和匹配点上限，但保留 raw-only、不做 RANSAC/Homography 修复。（complete）
  3. 生成中文 PDF 报告，包含训练曲线、验证指标、匹配质量分布、样例图和问题诊断。（complete）
  4. 用当前 `pose_metadata_crop1024_1800pairs_random_lr5e5_round1_20260527` run 生成新版报告并校验 PDF/PNG/CSV 文件存在。（complete）
  5. 给出训练效果判断和下一轮训练建议。（complete）
- 结果：
  - 新版报告目录：`runs/pose_metadata_crop1024_1800pairs_random_lr5e5_round1_20260527/visual_report_v2`。
  - 中文 PDF：`training_report_zh.pdf`，6 页。
  - 抽样 16 对，每对最多 512 条 raw mutual match；总匹配 8192、正确 4932、precision 0.6021。
  - pose metadata 路径前缀修复完成，复制到本地 NVMe 后可按 `cache/.../pair_*.pt` 后缀查回相机难度元数据。

## Pose-balanced hard view 优化（2026-05-27）
- 阶段：in_progress。
- 目标：修复当前 pose-balanced sampler 在 `batch_pairs=1` 下长期不采 hard 样本的问题，并用当前 2048->1024 crop 仿真训练集跑一轮 hard-aware 训练和报告。
- 发现：
  1. 当前 `pose_metadata_crop1024_1800pairs_random_lr5e5_round1_20260527` 训练日志里 `pose_hard_pairs` 全程为 0，`pose_medium_pairs` 全程为 1。
  2. 本地 train split 实际包含 hard 908、medium 1132；val split 包含 hard 456、medium 570。数据有 hard，不是数据缺失。
  3. 根因是 `sample_pose_balanced_training_pairs()` 固定按 easy -> medium -> hard -> unknown 取桶；当 `batch_pairs=1` 且没有 easy 时，总是取 medium，hard 永远不会进入训练。
- 计划：
  1. 增加回归测试，证明 `batch_pairs=1` 时 pose-balanced sampler 会覆盖 hard/medium，而不是只取 medium。（complete）
  2. 修复 sampler：每轮随机 difficulty 顺序，使小 batch 也能跨步覆盖 non-empty difficulty buckets。（complete）
  3. 跑 focused/full Python 测试。（complete；212 tests OK, 2 skipped）
  4. 从当前 checkpoint 继续一轮 hard-aware 微调，降低 LR、增加每步样本数/梯度累积，并生成中文报告。（complete）
  5. 对比 v2 报告，重点看 hard precision、总体 precision、loss/grad clipping 和 score separation。（complete）
- 结果：
  - 新 run：`runs/pose_metadata_crop1024_samplerfix_hard_fusion_lr2e5_600_20260527`。
  - 训练点数提升到每步 1024；hard/medium 每步都被采到，旧 run hard steps 为 0/1800。
  - 验证 loss 2.9794 -> 1.5887，Top1 0.7206 -> 0.8166，Top5 0.8154 -> 0.9152。
  - 无 margin raw 输出不优：4737/8192 correct，precision 0.5782；hard 610/2048，precision 0.2979。
  - 推荐当前 checkpoint 配合 `min_margin=0.01`：5060/6931 correct，precision 0.7301；hard 584/1329，precision 0.4394。未使用 RANSAC/Homography。
  - 结论：本轮修复了 hard 数据没被训练到的问题，并提升了 margin 可过滤后的匹配质量；但模型本身 raw 输出仍会放出过多错误匹配，下一轮应继续优化 confidence/margin 或训练期 false-match 抑制。

## Pose-aware margin 第二轮优化（2026-05-27）
- 阶段：in_progress。
- 目标：从 `pose_metadata_crop1024_samplerfix_hard_fusion_lr2e5_600_20260527` checkpoint 继续微调，增强 warp 真值 hard-negative / abstention margin，使推荐 `min_margin=0.01` 输出在 hard 视角上进一步提升，同时监控 raw precision 是否继续退化。
- 计划：
  1. 维持原 1024 crop、pose-balanced sampling、texture adapter + descriptor fusion 训练路线。（complete）
  2. 降低学习率，增大 hard-negative / warp-hard-negative / abstention 权重，避免直接接入原图坐标的 mined false-match CSV。（complete）
  3. 训练后自动生成 `min_margin=0.01` 中文 PDF 报告，并和上一轮报告对比 total/hard precision、correct support 和验证 Top1/Top5。（complete）
  4. 根据结果决定是否转向 crop-aware false-match 标签变换或 pair-level no-match/score-gate loss。（in_progress）
- 结果：
  - run：`runs/pose_metadata_crop1024_marginstrong_lr8e6_400_20260527`。
  - validation retrieval 小幅提升：Top1 0.8157 -> 0.8219，Top5 0.9111 -> 0.9175。
  - 但 `min_margin=0.01` 报告退化：总体 5056/7039 precision 0.7183，低于上一轮 5060/6931 precision 0.7301；hard 组 589/1370 precision 0.4299，也低于上一轮 584/1329 precision 0.4394。
  - 结论：单纯加重 margin/abstention 权重会提升 retrieval loss 指标，但不等于更好的 raw/margin 匹配结果；不要沿用该 checkpoint 作为下一轮起点。

## Pose-aware 大 batch 温和 margin 优化（2026-05-27）
- 阶段：in_progress。
- 目标：按用户建议提高 GPU 利用率，从上一轮推荐 checkpoint 重新起训，把 `batch_pairs` 从 2 增到 4、每步监督点数从 1024 提到 2048，并使用较温和的 margin 约束验证是否能提升匹配报告。
- 计划：
  1. 从 `pose_metadata_crop1024_samplerfix_hard_fusion_lr2e5_600_20260527/pytorch_pfm_state.pt` 起训，而不是从退化的 marginstrong checkpoint 起训。（pending）
  2. 使用 `batch_pairs=4`、`samples_per_pair=256`、`gradient_accumulation_steps=2`，监控显存和 GPU 利用率。（pending）
  3. 训练完成生成 `min_margin=0.01` 中文报告，并与上一轮最佳和 marginstrong 结果对比。（pending）

## PFM v2.1 架构迁移（2026-05-27）
- 阶段：complete。
- 目标：按最新 PFM v2.1 文档，把当前 PyTorch 训练/评估主路径从“更大普通 CNN + 图匹配器”推进到更贴近行星影像的光照鲁棒、几何感知、纹理自适应架构。
- 已完成：
  1. `python/pfm_model.py` 默认模型容量升级为 `base_channels=64`、`descriptor_dim=256`、`graph_hidden_dim=512`、`graph_attention_layers=8`，checkpoint config 增加 `graph_keypoint_meta_dim`。（complete）
  2. Backbone 在原 stage 基础上增加零初始化 residual context blocks，默认初始行为兼容旧权重，同时提供更强局部表达能力。（complete）
  3. SparseHead 增加 keypoint subpixel offset 输出、keypoint/descriptor/geometry 分支 context、C4 branch-quality attention、rotation-invariant + rotation-sensitive descriptor fusion。（complete）
  4. GeometryHead 输出改为更稳定的 `exp(clamp(log_scale))` 与 `I + 0.1*tanh(delta_affine)`，为后续 descriptor canonical sampling 留接口。（complete）
  5. analytic texture descriptor 扩展为 illumination-robust channels，包含局部归一化强度、局部对比、DoG/LoG、梯度方向、ring contrast 和 soft census-like 特征。（complete）
  6. DescriptorFusionAdapter 从固定 texture weight 升级为局部自适应 texture gate，旧 checkpoint 缺失的新参数自动用默认 state 填充。（complete）
  7. GraphMatcher keypoint metadata 支持 x/y/radius/radius² 四维输入，默认新模型使用 `graph_keypoint_meta_dim=4`；旧 checkpoint 仍可回退到 2 维兼容路径。（complete）
  8. `pytorch_cache_match_eval.py` 与 `scripts/training_visual_report.py` 增加 `--matcher-mode raw_descriptor|graph_matcher`；训练报告支持 `--report-matcher-mode both`，可同时输出 raw descriptor 和正式 GraphMatcher 口径。（complete）
- 验证：
  - `py_compile` 通过：`python/pfm_model.py`、`python/pytorch_cache_match_eval.py`、`python/pfm_pytorch_training.py`、`scripts/training_visual_report.py`。
  - focused tests 通过：`python/test_pfm_model.py`、`python/test_pytorch_cache_match_eval.py`、`python/test_pfm_pytorch_training.py`，121 tests OK，1 skipped。
  - 完整 Python discovery 通过：216 tests OK，2 skipped。
  - 默认模型 smoke forward 通过，输出 descriptor 为 `[1, 256, 16, 16]`，keypoint offset 为 `[1, 2, 16, 16]`。
- 边界：
  - 本轮完成的是 PyTorch 训练/评估主路径；C++/LibTorch 正式推理端尚未同步 v2.1 架构。
  - semi-dense weak-texture fallback 与真正的 geometry-aware canonical descriptor sampling 仍是下一阶段实现项，目前已完成接口和稳定参数化基础。

## 仿真数据同步与 v2.1-lite 继续训练（2026-05-27）
- 阶段：complete。
- 目标：检查 2048 仿真数据生成进度，把已完成训练 cache 更新到本地 NVMe，然后继续训练并生成 raw descriptor / GraphMatcher 双报告。
- 数据状态：
  1. 8T 仿真生成仍在继续，`batch_pose_sim_dataset.py` 使用 `asp36` 环境，`sat_sim_cuda` 当前仍在跑。
  2. 检查时 8T cache 已有 `train=3294+`、`val=1638+`、`test=1638+`；同步后本地 NVMe 为 `train=3294`、`val=1026`、`test=419`。
  3. 本轮只补齐本地 train，避免全量复制 test 浪费时间；训练继续使用本地 NVMe cache，pose metadata 从 8T dataset root 读取。
- 训练结果：
  - run：`runs/pose_sim2048_crop1024_v21lite_train3294_ctxfusion_b1p512_lr3e5_600_20260527`。
  - 起点：`runs/pose_metadata_crop1024_batch4_mildmargin_lr1e5_300_20260527/pytorch_pfm_state.pt`，配置为 `base_channels=48`、`descriptor_dim=192`、`graph_hidden_dim=384`、`graph_keypoint_meta_dim=2`。
  - 本轮训练：600 steps，`batch_pairs=1`、`samples_per_pair=512`、1024 crop，训练 descriptor head、sparse context、texture adapter、descriptor fusion。
  - 验证检索：loss 2.3557 -> 1.0998，Top1 0.6909 -> 0.8404，Top5 0.8288 -> 0.9270，mean rank 10.18 -> 3.74。
  - 训练日志：first50 loss 6.5626 -> last50 loss 4.5656，first50 Top1 0.5547 -> last50 Top1 0.7873，无 skipped step；pose-balanced 采样 medium 312、hard 288。
  - raw descriptor 抽样报告：5629/7646 correct，precision 0.7362；medium precision 0.8181，hard precision 0.4322。
  - GraphMatcher 抽样报告：3896/7706 correct，precision 0.5056；medium precision 0.6100，hard precision 0.0948。
- 关键问题：
  - batch4 与 batch2 在 1024 + C4 + blended descriptor training 下 OOM；batch1 稳定，训练显存约 19GB。
  - 当前 PyTorch 训练只微调 descriptor/fusion/context，没有训练 GraphMatcher，所以 GraphMatcher 报告明显落后 raw descriptor。
  - 如果要上 full `256` 维新模型，不能直接 `--init-random` 用现有训练脚本，因为 backbone 会随机且默认不训练；需要先实现 full 模型的 backbone/GraphMatcher 训练阶段或迁移权重策略。

## PFM v2.1-full 一次性架构落地（2026-05-27）
- 阶段：complete。
- 用户要求：不要继续逐次小迭代，要按 `PFM_v2_model_architecture.md` 一次性把 v2.1 主架构落地到代码。
- 范围：
  1. PyTorch 主模型实现完整 v2.1 主干：Dual-FPN-Lite、分离的 keypoint/descriptor feature、Geometry-aware descriptor pooling、QualityHead、illumination robust texture fusion。
  2. GraphMatcher 升级到 V2：x/y/radius/score/scale/orientation/affine/quality/local contrast metadata、top-k candidate pruning、pairwise geometry compatibility bias、dual-softmax score。
  3. 推理/评估支持 weak-texture semi-dense fallback，并继续保留 raw_descriptor 和 graph_matcher 双口径报告。
  4. 训练脚本支持一次性选择 v2.1-full 可训练模块，包括 backbone、Dual-FPN、geometry/keypoint/quality、GraphMatcher。
  5. 补充 focused tests，保证新模块 shape、compat loading、matcher candidate pruning 和训练参数选择不破坏现有接口。
- 非范围：
  - 本轮不把 C++/LibTorch 端完整重写为 v2.1；当前炼丹和报告路径已经以 PyTorch 为主。
  - 本轮不承诺完成长时间训练，只完成架构、接口、短 smoke 验证，为后续正式训练提供完整模型。
- 当前计划：
  1. 更新 `python/pfm_model.py`：加入 DualFPNLite、QualityHead、几何规范化池化和 GraphMatcherV2。（complete）
  2. 更新 `python/pytorch_cache_match_eval.py`：GraphMatcher 使用 richer metadata；增加 raw descriptor semi-dense fallback 合并入口。（complete）
  3. 更新 `python/pfm_pytorch_training.py`：加入 `--train-backbone`、`--train-dual-fpn`、`--train-geometry-head`、`--train-quality-head`、`--train-graph-matcher` 等一次性训练开关，并接入 graph matcher correspondence loss。（complete）
  4. 更新测试并跑 focused/full Python 验证。（complete）
  5. 更新 `PFM_v2_model_architecture.md`、`progress.md`、`findings.md` 记录实际落地状态。（complete）
- 验证：
  - `py_compile` 通过：`python/pfm_model.py`、`python/pytorch_cache_match_eval.py`、`python/pfm_pytorch_training.py`、`scripts/training_visual_report.py`。
  - focused tests 通过：`python/test_pfm_model.py`、`python/test_pfm_pytorch_training.py`、`python/test_pytorch_cache_match_eval.py`，128 tests OK，1 skipped。
  - full Python discovery 通过：223 tests OK，2 skipped。
  - v2.1 smoke forward 通过：64x64 输入输出 descriptor/heatmap/quality 为 1/4 分辨率，GraphMatcherV2 输出 dustbin logits 和 matches。
  - v2.1-full 训练链路 smoke 通过：`runs/v21_full_arch_smoke_20260527`，`--init-random --train-backbone --train-dual-fpn --train-geometry-head --train-quality-head --train-graph-matcher` 等完整开关跑通 1 step，skip=0，并写出 checkpoint/metrics。
- 架构补齐更新：
  - GraphMatcher 评估路径已从 `RawFeatureMaps` 按关键点采样 `heatmap score / log scale / orientation / affine / quality / local contrast`，不再只传 x/y/score。
  - 新增 `SemiDenseCandidateBranch`，在 coarse descriptor grid 上用 dual-softmax 生成 detector-free 候选，并在 GraphMatcher 输入前与 sparse keypoint candidates 合并。
  - 新增 `sample_descriptor_rows_at_keypoints()`、`graph_metadata_from_raw_features()`，让 sparse 与 semi-dense 候选都能走统一 metadata 入口。
  - `training_visual_report.py` 的 GraphMatcher 可视化也改为使用完整 graph metadata。
  - 补充验证：focused tests 131 OK/1 skipped，full Python discovery 226 OK/2 skipped；graph matcher + semi-dense smoke 得到 identity pair `16/16` correct。
- 边界：
  - 本轮完成 PyTorch 主路径；C++/LibTorch 正式推理端仍未同步。
  - 新 v2.1-full 默认是 `base_channels=64`、`descriptor_dim=256`、`graph_hidden_dim=512`、`graph_keypoint_meta_dim=16`。旧 v2-lite checkpoint 仍可加载，但继续训练时会按 checkpoint 配置实例化，不能自动变成 full 256。

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
| rotation benchmark 90/270° 真实 smoke 全错但 180° 全对 | 1 | 查明 `np.rot90(k=1)` 方向为逆时针，修正 `rotate_points()` 90/270 真值公式并补测试 |

## Rotation matcher benchmark (2026-05-25)
- 阶段：complete。
- 范围：新增独立 `python/rotation_matcher_benchmark.py` 与 `python/test_rotation_matcher_benchmark.py`；不改训练/评估主流程。
- 输出：`runs/rotation_matcher_benchmark_20260525/metrics.csv`、`summary.txt`、`visualizations/`。
- 验证：base Python 与 `pfm-train` 环境单测通过；`pfm-train` 真实 smoke 跑完 SIFT/ORB/AKAZE/PFM，LightGlue/SuperGlue 按 unavailable 记录。

## Rotation matcher旁路迭代 (2026-05-25)
- 阶段：complete。
- 目标：不改训练主线，给 `python/rotation_matcher_benchmark.py` 增加至少一个传统/后处理增强 matcher，并在新 runs 目录完成两类图像 x 90/180/270 对比。
- 计划：
  1. 确认现有 benchmark 与测试入口。（complete）
  2. TDD 增加 RootSIFT/FLANN/ratio/mutual/RANSAC matcher 的最小测试。（complete）
  3. 实现 matcher 并接入 OpenCV matcher 列表。（complete）
  4. 跑 focused/full Python 测试。（complete）
  5. 跑 `runs/rotation_matcher_iter_rootsift_ransac` 对比并写 `agent_summary.md`。（complete）

## Rotation matcher旁路迭代 AffineSIFT (2026-05-25)
- 阶段：complete。
- 目标：不改训练主线，补充与 RootSIFT 不重复的 OpenCV AffineFeature(SIFT) baseline。
- 结果：新增 `AffineSIFT-BF`，在 `runs/rotation_matcher_iter_affinesift` 完成两风格 x 90/180/270 benchmark，并写入 `agent_summary.md`。

## Rotation matcher旁路继续迭代 USAC (2026-05-25)
- 阶段：complete。
- 目标：不改训练主线和主实验输出，补充 RootSIFT + OpenCV USAC 几何验证稳健性检查，并在 `runs/rotation_matcher_iter_continued_*` 新目录完成两风格 x 90/180/270 对比。
- 计划：
  1. 读取现有 benchmark/test 与 RootSIFT/AffineSIFT 结果，确认指标口径。（complete）
  2. TDD 增加 USAC RootSIFT matcher 注册/可用性测试。（complete）
  3. 实现 RootSIFT-FLANN-USAC-MAGSAC/PROSAC 或可用 fallback。（complete）
  4. 跑 focused 测试与完整 benchmark。（complete）
  5. 汇总 metrics、写入新 runs 目录说明。（complete）

## Cross-view 1024 match-margin 与 specialist 迭代 (2026-05-26)
- 阶段：in_progress。
- 已完成：
  1. `cross_view_experiment.py` 支持 match-margin 校准、support-aware selection、3-seed calibration 和 `--training-groups` 单 gate 训练过滤。（complete）
  2. `hard_pair_mining.py` 支持当前 evaluator summary 列名 `matches/precision`。（complete）
  3. 当前最好 0-step 路由结果为 `runs/cross_view_1024_match_margin_multiseed16_min20_0step_seed1234`。（complete）
  4. timestamp/viewpoint 60-step specialist 负结果已记录：`runs/cross_view_1024_timestamp_viewpoint_specialist_60_seed1234`。（complete）
  5. `cross_view_experiment.py` 支持 `--calibration-pytorch-state label=path`，calibration 可按 group 选择 checkpoint + texture blend + margin。（complete）
  6. checkpoint-routed 0-step 评估完成：`runs/cross_view_1024_checkpoint_routed_margin_multiseed16_min20_0step_seed1234`。（complete）
  7. 子 agent 独立完成 SIFT/ORB/AKAZE/LightGlue/LoFTR/SuperGlue/PFM 旋转对比和一次传统 matcher 迭代，输出 `runs/rotation_matcher_comparison_agent`。（complete）
  8. checkpoint state-switch guard 完成：`--calibration-state-switch-min-precision-gain` 与 `--calibration-state-switch-min-match-ratio`，并完成 `runs/cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234`。（complete）
  9. hard-pair curriculum pass-through、train-split hard mining、base-reference state-switch label 完成；`runs/cross_view_1024_hard_mined_weakgates_80_seed1234` 与 `runs/cross_view_1024_hard_mined_weakgates_80_base_ref_postselect` 均已评估。（complete）
  10. 子 agent 完成真实 cross-view 传统 matcher 对比，输出 `runs/cross_view_traditional_matcher_comparison_agent3`，RootSIFT/H-RANSAC 显著强于当前 PFM。（complete）
- 当前结论：评估/路由层的 margin + 多 seed 对 numeric/viewpoint 有明显帮助；带 guard 的 checkpoint routing 能安全吸收 `blend02540` 在 rotate 上的收益，避免 timestamp/compound 0/23 退化，但 timestamp/viewpoint 仍弱。短程单 gate descriptor-only 微调、hard-pair curriculum、无约束 checkpoint routing 都不是后续主方向。
- 下一步候选：把 RootSIFT/Homography-RANSAC 真实 cross-view inliers 转成伪标签或蒸馏目标，训练 PFM descriptor/keypoint 在实际 viewpoint/compound pair 上复现传统 matcher 的高置信对应；同时保留现有 base/guarded routing 作为回退，不再优先延长当前 synthetic warp CE 微调。（已进入下面的 pseudo-label 迭代。）

## RootSIFT pseudo-label 迭代 (2026-05-26)
- 阶段：in_progress。
- 已完成：
  1. `pfm_pytorch_training.py` 支持读取 RootSIFT/H-RANSAC pseudo-label CSV，并按完整路径匹配训练 pair。（complete）
  2. 增加 pseudo-label curriculum sampling，避免 labeled pair 在大训练池中长期采不到。（complete）
  3. 增加 `--synthetic-loss-weight`，允许 pseudo-only 或降低 synthetic warp CE 权重。（complete）
  4. 新增 `python/pseudo_label_generation.py` 与 `scripts/generate_rootsift_pseudo_labels.py`，可从 1024 cache 生成逐点伪标签 CSV。（complete）
  5. 子 agent 4/5/6 连续完成匹配算法 sidecar 迭代，确认 RootSIFT-HRANSAC 是当前最干净的伪标签来源，并推荐 `ratio=0.80, RANSAC=2px`。（complete）
  6. 小样本 pseudo-only 40-step 训练有轻微正向验证信号：top1 0.1004 -> 0.1016。（complete）
  7. r0.80/t2 扩样标签生成完成：93/128 pairs、9366 labels。（complete）
  8. 扩样 pseudo-only 80-step 训练完成，验证 retrieval 小幅改善：top1 0.1004 -> 0.1027，top5 0.2859 -> 0.2911。（complete）
  9. group-aware r0.80/t2 160-step 扩样训练完成：validation top1 0.0784 -> 0.0822，但 guarded sparse eval 未超过旧 r0.80/t2 timestamp/viewpoint specialist。（complete）
  10. Agent7/8 验证 PFM fallback 不可作为 pseudo-label 来源；Agent9/10 验证 RootSIFT r0.90/t2 可作为“r0.80/t2 未过 gate 时”的高精度插入 fallback。（complete）
  11. r0.90/t2 全量、r0.80+r0.90 fallback-only、timestamp/viewpoint-only r0.90 specialist 三条训练均完成，均未超过旧 r0.80/t2 timestamp/viewpoint specialist。（complete）
  12. pseudo-label keypoint heatmap 监督与 `--keypoint-score-mode learned` 评估链路完成；完整 Python discovery 通过，162 tests OK，1 skipped。（complete）
  13. `runs/cross_view_1024_rootsift_pseudo_r080t2_keypoint_w1n002_lr1e6_viewpoint_80_seed1234` 完成：descriptor validation 负向，但 timestamp/viewpoint learned-score sparse sweep 出现候选提升，best 41/223 (0.183857)。（complete）
  14. Agent11 完成旋转 cross-view matcher 扩展对比，`RootSIFT-r0.90-Ht2` 与 `LightGlue-SIFT-Ht3` 远强于 PFM，输出在 `runs/matcher_algorithm_iteration_agent11`。（complete）
  15. `cross_view_experiment.py` 已把 learned keypoint score 纳入 calibration/routing；完整 Python discovery 通过，165 tests OK，1 skipped。（complete）
  16. keypoint-only heatmap distillation 完成两轮：viewpoint-only 80 steps 与 weakgroups 120 steps。descriptor validation 均保持不变，weakgroups guarded eval 让 trained heatmap checkpoint 被选中到 numeric/compound 与 timestamp/compound。（complete）
  17. Agent12/Nietzsche 完成独立 matcher 迭代：`RootSIFT-ratio-r0p90-Ht3-min4` 为最高 coverage/correct teacher，759/794 (0.955919)；`LightGlue-SIFT-Ht3-min4` 为高 precision 外部 teacher，674/683 (0.986823)；PFM-current raw 仅 20/974。（complete）
  18. r0.90/Ht3/min4 coverage labels 生成并完成 heatmap-only 训练/六组 guarded eval；numeric/compound 提升到 50/225，但 timestamp/compound 退到 3/109，说明 r0.90/Ht3 不应全局替代 r0.80/t2 heatmap teacher。（complete）
  19. Agent13/Popper 完成外部 matcher 迭代：`RootSIFT-ratio-r0p88-Ht3-min4` 达到 779/801 (0.972534)、coverage 12/12、pass gate 9/12，优于 Agent12 的 r0.90/Ht3 pass gate 7/12；同时给出 style/gate-specific teacher 推荐。（complete）
  20. style/gate-specific heatmap pseudo-label CSV 已生成：viewpoint 使用 r0.80/t2，numeric/compound 使用 r0.90/Ht3，timestamp/compound 使用 r0.80/t2，共 40548 labels。（complete）
  21. style-specific heatmap-only 120-step 训练完成，descriptor validation 保持不变；单 checkpoint eval 将 numeric/compound 推到 50/224，但 timestamp/compound 仍为 3/111。（complete）
  22. 多 checkpoint guarded route 完成：`runs/cross_view_1024_keypointonly_multistate_stylespecific_guard_calib_0step_seed1234` 同时保留 numeric/compound 50/224 和 timestamp/compound 5/72，并写出六组 summary 与十二张可视化。（complete）
  23. Agent13 stage2 完成 train split teacher mining：style-specific bundle 为 144 kept pairs / 14496 capped labels / aggregate truth precision 0.9874，numeric/compound 明确推荐 LightGlue-SIFT，timestamp/compound 推荐 r0.88/Ht3。（complete）
  24. Agent13 stage2 selected labels 转成训练 CSV 并修正 split symlink 路径匹配问题；相对路径 CSV 为 `runs/rootsift_pseudo_labels_agent13_stage2_stylespecific_relpaths_seed1234/pseudo_labels.csv`，14080 dedup labels。（complete）
  25. Agent13 stage2 heatmap-only 120-step 训练与单 checkpoint eval 完成：descriptor validation 不变，numeric/compound 50/224，timestamp/compound 3/107，未超过多 checkpoint route。（complete）
  26. Agent13 stage3 完成 timestamp/compound hard-tail 诊断：r0.88/Ht3 kept 103/128、10025 labels、truth precision 0.9846，但主要覆盖 easy/common pairs，24/128 仍为 `too_few_truth_labels`。（complete）
  27. balanced + hard-tail heatmap CSV、120-step 训练和单 checkpoint eval 完成：`runs/cross_view_1024_agent13_stage2_balanced_hardtail_keypointonly_learnedscore_guard_calib_0step_seed1234` 的 timestamp/compound 为 3/112，未超过 r0.80/t2 weakgroups 的 5/72。（complete）
  28. all-specialist + balanced-hardtail route sweep 尝试过宽/窄两版，但 calibration 组合过慢，均中止；后续不再用全轴穷举做快速 triage。（complete）
  29. Popper/Agent13 stage4 完成 hard-tail sparse teacher 迭代：24 个 hard-tail 中只有 1 个 pair 被高精度覆盖，`candidate_labels.csv` 仅 38 labels，不能支撑下一轮训练。（complete）
  30. Agent13 stage5 完成：LoFTR、LightGlue-SIFT k2048/k4096、Farneback、DISFlow 仅覆盖 2/24 hard-tail unique pairs，新增 pair 只有 1 个；不进入训练。（complete）
  31. timestamp/compound hard-tail 数据几何诊断完成：hard-tail valid fraction 和 target-inside 不差，B 图梯度更低，问题更像 target-view 外观/纹理可重复性不足。（complete）
  32. Agent13 stage6 完成 route/quality gate 诊断：B-view local contrast 是当前 timestamp/compound 最强 abstain 特征，B-view gradient 次之，CLAHE SIFT 数量和 Laplacian variance 不建议作为主 gate。（complete）
  33. `pytorch_cache_match_eval.py` 与 `cross_view_experiment.py` 已支持 target-view quality gates：`--min-target-gradient`、`--min-target-local-contrast` 以及对应 calibration candidates。（complete）
  34. `.pt` tensor 口径验证完成：local contrast 5.2 在 full-val 上 10/108 (0.092593)，fixed-test 上 4/45 (0.088889)，比当前 timestamp/compound baseline 12/208 (0.057692) 与 5/72 (0.069444) 更高 precision 且保留 80% correct。（complete）
  35. 六组 target-contrast postselected route 已生成：`runs/cross_view_1024_keypointonly_multistate_stylespecific_guard_targetcontrast_postselect_0step_seed1234`，包含 6 个 summary、12 张可视化和 selected weights；仅 timestamp/compound 改为 4/45 (0.088889)。（complete）
  36. Hilbert/Agent13 stage7 完成 local-contrast dropped-pair fallback 检查：RootSIFT-FLANN-r0.80+HomographyUSAC-t2 在 fixed-test dropped set 上 890/898 (0.991091)，full-val dropped set 上 3562/3577 (0.995807)。（complete）
  37. 生成 hybrid route：`runs/cross_view_1024_targetcontrast_rootsift_fallback_route_20260526`，包含 6 个 summary、12 张可视化、hybrid metrics/delta/decision CSV；timestamp/compound 合并后为 894/943 (0.948038)，但明确标注为外部 fallback，不是纯 PFM。（complete）
  38. 生成 train-only low target-contrast timestamp/compound 伪标签：`runs/rootsift_pseudo_labels_tscompound_lowtargetcontrast_r080t2_train_seed1234`，128 sampled candidates / 88 kept pairs / 8317 labels。（complete）
  39. 完成 low-contrast heatmap-only 80-step continuation：`runs/cross_view_1024_tscompound_lowcontrast_keypointonly_w1n002_lr1e5_80_seed1234`。timestamp/compound fixed-test 从 5/72 提到 5/59；叠加 target-contrast gate 从旧 4/45 提到新 5/36；full-val 从 12/208 到 14/228，叠加 gate 从旧 10/108 到新 12/126。（complete）
  40. Hooke/Agent13 stage8 完成 all targetcontrast gate-zero fallback policy sweep：246 fixed-test candidates；RootSIFT r0.80/H2 all_gate_zero fallback 为 60108/60431 (0.994655)，6/6 groups pass。（complete）
  41. 生成 all-gate-zero broader hybrid route：`runs/cross_view_1024_targetcontrast_rootsift_allgatezero_fallback_route_20260526`，包含 6 个 summary、12 张可视化和 hybrid comparison/delta/support CSV；整体 fixed-test hybrid 为 61437/62255 (0.986860)，明确标注为外部 fallback。（complete）
  42. Stage9 sidecar Linnaeus 完成：`runs/matcher_algorithm_iteration_agent14_stage9/` 在 cache-heldout 上验证更严格的 `RootSIFT-FLANN-r0.75+HomographyUSAC-t2` 优于 r0.80/H2，r0.75/H2 为 38569/38682 (0.997079)，min group precision 0.987853。（complete）
  43. 生成 lowcontrast 纯 PFM 六组 postselected route：`runs/cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234`。五组沿用 target-contrast postselected route，timestamp/compound 使用 lowcontrast checkpoint + local contrast 5.2，fixed-test 为 5/36 (0.138889)；同时验证 lowcontrast 在 numeric/compound 为 204/2725 (0.074862)，因此不用于 numeric/compound。（complete）
  44. Stage10 sidecar Meitner 完成 full-val all-gate-zero replay：`runs/matcher_algorithm_iteration_agent14_stage10/`，无抽样，845 gate-zero rows；r0.75/H2 fallback 183662/184456 (0.995695)，hybrid full-val 0.982896，优于 r0.80/H2 的 fallback 0.993065 / hybrid 0.981175。（complete）
  45. train-only broad gate-zero r0.75/H2 heatmap labels 与训练完成：`runs/rootsift_pseudo_labels_gatezero_r075t2_train_sample64_seed1234` 产出 115 kept pairs / 6844 labels；`runs/cross_view_1024_gatezero_r075t2_keypointonly_w1n002_lr1e5_80_seed1234` 训练 80 steps，descriptor validation 不变。（complete）
  46. gate-zero heatmap checkpoint 的六组 fixed-test 与 compound full-val guard 完成：`runs/cross_view_1024_gatezero_r075t2_keypointonly_selectedparams_eval_0step_seed1234`。固定 test compound 小幅提升，但 full-val numeric/compound 退到 1246/14730 (0.084589)、timestamp/compound 退到 17/2377 (0.007152)，因此不加入 pure-PFM routing。（complete）
  47. Stage11 sidecar Sagan 完成 fixed-test r0.75/H2 broad hybrid route：`runs/matcher_algorithm_iteration_agent14_stage11/`。相对 Stage8 r0.80/H2，precision 0.986860 -> 0.988985，wrong 818 -> 650，但 correct 61437 -> 58362；推荐 r0.75/H2 作为 high-precision hybrid 默认，r0.80/H2 保留为 high-support baseline。（complete）
  48. Stage12 sidecar Poincare 完成训练安全策略诊断：`runs/matcher_algorithm_iteration_agent14_stage12/`。推荐 `P1_r075_pairfiltered_noncompound_viewpoint_tiny`，只保留 numeric/viewpoint 与 timestamp/viewpoint 的 45 个 validation-backed pairs / 540 label budget，使用 strict caps + hard negatives；禁止 broad all-gate-zero heatmap 和 r0.80/H2 max-support teacher 训练扩张。（complete）
  49. timestamp/viewpoint target-quality gate 诊断完成：`runs/timestamp_viewpoint_quality_gate_diagnostic_20260526/`。local contrast / target gradient gate 可提高 precision，但降低 correct/support；作为 precision-only reporting candidate，不替换当前 pure-PFM timestamp/viewpoint 默认路线。（complete）
  50. 项目介绍 PPT 已生成：`runs/project_presentation_20260526/PlanetaryFeatureMatch_project_intro_20260526.pptx`，13 页，覆盖项目背景、特征提取模型、特征匹配模型、性能、传统/深度 matcher 对比、hybrid/fallback、负结果和下一步。（complete）
  51. P1 tiny viewpoint retention 标签生成器与测试完成：`scripts/generate_p1_viewpoint_retention_labels.py`、`python/test_p1_retention_label_generation.py`；focused test 2 tests OK，py_compile OK。生成 `runs/rootsift_pseudo_labels_p1_viewpoint_r075t2_train_seed1234`，62 pairs / 744 labels，wrong=0。（complete）
  52. P1 both-viewpoint descriptor-only probe 完成：`runs/cross_view_1024_p1_viewpoint_retention_desc_lr5e7_b2_80_seed1234`。numeric/viewpoint full-val 371/836 -> 949/2698 precision 0.443780 -> 0.351742；timestamp/viewpoint full-val 78/389 -> 81/371 but fixed64 20/104 -> 16/95。判定不加入 pure-PFM route。（complete）
  53. P1 timestamp/viewpoint-only descriptor probe 完成：`runs/rootsift_pseudo_labels_p1_timestamp_viewpoint_r075t2_train_seed1234` 与 `runs/cross_view_1024_p1_timestamp_viewpoint_retention_desc_lr5e7_b2_60_seed1234`。full-val 78/389 -> 83/398 但 fixed64 20/104 -> 11/94，margin 0.02 为 0/0。判定不加入 pure-PFM route。（complete）
  54. 项目介绍 PPT 已刷新：`scripts/create_project_presentation.py` 重新生成 `runs/project_presentation_20260526/PlanetaryFeatureMatch_project_intro_20260526.pptx`，加入 P1 descriptor-retention 最新负/弱信号，并用 LibreOffice 验证可转 13 页 PDF。（complete）
  55. P1 warp-aware hard-negative descriptor probe 完成：`runs/cross_view_1024_p1_viewpoint_retention_desc_warpneg020_lr5e7_b2_80_seed1234`。训练 retrieval 仍轻微退化；numeric/viewpoint full-val 371/836 -> 949/2702 precision 0.443780 -> 0.351221，timestamp/viewpoint full-val 78/389 -> 81/368 但 fixed64 20/104 -> 16/92。判定不加入 pure-PFM route。（complete）
  56. 固定六组 x 两个 pair 的外部算法匹配效果已整理到 `对比文档/`：新增 `scripts/fixed_six_group_matcher_comparison.py`，复用之前 12 个可视化 pair，输出 `README.md`、`fixed_pairs.csv`、`metrics.csv`、`summary.csv` 和 80 张匹配图；包含 PFM-current、SIFT、RootSIFT、ORB、AKAZE、LightGlue-SIFT，SuperGlue 记录为 unavailable。（complete）
  57. 项目介绍 PPT 按用户反馈重做为 10 页：当前项目模型简介为主，只用 `对比文档/` 展示其他方法匹配效果，不再大段介绍其他模型；`python-pptx` 读取 10 页正常，LibreOffice 成功转 10 页 PDF。（complete）
  58. `对比文档/` 匹配图按用户反馈重绘：所有算法（含 PFM）统一用人工合成 GT warp 判定每条线，绿色表示正确匹配，红色表示错误匹配；PFM 不再复制旧随机色图，而是按 current postselected route 参数重算匹配点。输出 84 张 PNG，PFM 指标与原 route summary 逐 pair 完全一致。（complete）
  59. 外部算法“效果过好”质疑已核查并修正展示口径：固定 12 pair 未额外旋转 B 图，且原 `figures/` 展示的是 homography/RANSAC 后结果；raw 外部匹配实际为 11614/13846、wrong=2232、precision=0.838798。已新增 `raw_metrics.csv`、`raw_summary.csv` 和 `figures_raw/` 72 张 raw 图，README 明确区分 raw 与 RANSAC 后结果。（complete）
  60. 按用户要求将 `对比文档/` 主口径改为 raw-only：`metrics.csv`、`summary.csv` 和 `figures/` 全部为原始 matcher 输出，未执行 RANSAC/Homography 几何筛选或修复；旧 `raw_metrics.csv`、`raw_summary.csv`、`figures_raw/` 已移除避免双口径。验证结果：84 张 PNG、外部算法 11614/13846 correct、2232 wrong、precision 0.838798；PFM 219/872 correct、653 wrong、precision 0.251147。（complete）
  61. `对比文档/极端测试/` 已补充真实 TIFF 极端案例的 raw matcher 对比：新增 `scripts/extreme_case_matcher_comparison.py`，对两张 `20260510*.tif` 跑 SIFT、RootSIFT、ORB、AKAZE、LightGlue-SIFT、PFM-current 和最新 P1 checkpoint；输出 `README.md`、`metrics.csv`、`summary.csv`、`skipped_algorithms.csv` 和 8 张 PNG。该真实 TIFF 对无 GT warp，因此不做绿色/红色正确性判定，也不做 RANSAC/Homography 修复。（complete）
  62. `对比文档/` 固定六组对比已重新生成，rotate 组现在实际使用 90°/180° pair：numeric 为 `pair_001587.pt`/`pair_002779.pt`，timestamp 为 `pair_001509.pt`/`pair_002860.pt`；`fixed_pairs.csv` 已包含 `rotation_deg` 列，`figures/` 84 张 PNG 已按新 pair 重绘，rotate 子目录显式命名为 `rot90`/`rot180`。（complete）
  63. 添加并验证 descriptor false-match suppression / abstention-aware loss：`python/pfm_pytorch_training.py` 支持 `--abstention-*` 参数，`python/cross_view_experiment.py` 可透传；完整测试 `python/test_pfm_pytorch_training.py` 55 tests OK、`python/test_cross_view_experiment.py` 51 tests OK、`py_compile` OK。80-step P1 viewpoint probe `runs/cross_view_1024_abstention_p1_viewpoint_desc_w025_m035_lr3e7_b4_80_seed1234` 完成；same-split raw 对照显示 weak groups 未改善：numeric/viewpoint 0.091746 -> 0.091527，timestamp/viewpoint 0.022600 -> 0.022479，因此不加入 pure-PFM route。（complete）
  64. PFM v2.1-full 256 维重新炼丹已启动：本地 NVMe 数据同步到 train 3735 / val 1863；训练 descriptor 路径已修复为 Dual-FPN；验证 `py_compile`、focused tests 132 OK 和 full discovery 227 OK；1024 crop full-256 smoke 通过；正式 run `runs/pose_sim2048_crop1024_v21full256_scratch_20260527` 已运行到至少 step 100，GPU 98%-100%，显存约 31.8GB/32.6GB。（in_progress）
  65. 弱纹理覆盖采样已落地：新增 `weak_texture_fraction`、`keypoint_cell_cap`、coverage diagnostics 和报告参数；full discovery `229 OK, 2 skipped`。上一轮 checkpoint 的 coverage-aware raw 报告达到 16368/16384 correct、precision 0.999023、16x16 平均覆盖 0.812012、弱纹理占比 0.264160。仿真数据已同步到 train 3744 / val 1872，新 continuation run `runs/pose_sim2048_crop1024_v21full256_coverage_cont1500_20260527` 已完成：eval loss 0.022447 -> 0.014582，coverage-aware raw 16379/16384 correct，precision 0.999695，弱纹理占比 0.270019。（complete）
  66. 跨相机位置数据扩展与首轮训练完成：新增 `scripts/generate_cross_position_pose_pairs.py`，基于现有 2048 仿真影像和 CameraA TSAI 生成跨 seq pair。因同中心虚拟相机无法三角化深度，当前先用 Mars 参考球面射线求交生成 dense warp/valid mask。正式缓存 `训练数据/pose_sim_cross_position_2048_gap30_views10` 包含 train 1200 / val 384；混合同中心缓存训练完成于 `runs/pose_sim2048_crosspos_mix_v21full256_blend_b1_3000_20260528_090944`，中文报告已生成。跨位置样本暴露远基线仍弱：报告中 cross 近邻 precision 约 0.70-0.76，部分更难样本仅 0.006-0.052。（complete）
  67. 真实 DEM depth 跨位置数据与训练完成：`训练数据/pose_sim_cross_position_rendered_2048_gap30_views10`，train 1200 / val 384，基于 `sat_sim_cuda --write-depth` 生成真实 depth GT；训练 run `runs/pose_sim2048_crosspos_rendered_mix_v21full256_blend_b1_3000_20260528_105130` 完成，24 样本报告平均 precision 0.999430。（complete）
  68. 按用户要求完成 7:2:1 切分与更极端跨位置样本：`训练数据/pose_sim_cross_position_rendered_extreme_2048_gap30_views10_721` 为 train 1400 / val 400 / test 200；同中心数据建立 7:2:1 视图 `训练数据/pose_sim_2048_gap30_views10_721` 为 train 5235 / val 1495 / test 749；混合训练样本为 train 6635 / val 1895 / test 949。训练 run `runs/pose_sim2048_721_cross_extreme_v21full256_blend_b1_3000_20260528_124407` 完成，主报告强制包含 `off012` 难样例，额外报告 `visual_report_extreme_test_off016` 强制包含 `off016` 测试样例。（complete）
- 正在进行：
  1. 观察 7:2:1 极端跨位置数据在不同 offset/轨迹/重叠率下的分化表现。
- 下一步：
  1. 报告和评估中把跨位置 pair 按 offset、valid fraction、轨迹段和弱纹理占比分桶，避免总指标掩盖难样本失败。
  2. 下一轮训练若继续提升跨位置极端样例，优先加强 hard negative、弱纹理 semi-dense fallback 或 GraphMatcher，而不是只继续降低 raw descriptor loss。
  3. 报告固定保留 `off012` 难验证样例和 `off016` 测试样例，防止可视化抽样只显示高分或低分一侧。


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

## Pose-aware satsim 数据生成管线 (2026-05-26)

### 状态
- 阶段：in_progress。
- 目标：用真实轨迹 + DEM/DOM 仿真 2048x2048 火星影像，并从相机参数和深度图自动生成精确 PFM `warp_a_to_b`/`valid_mask` 训练 cache。

### 已完成
1. 修复 `辅助软件/数据模拟/mars_orbit_to_tsai.py` 默认路径：SPICE kernel、DEM、DOM、sat_sim 默认指向当前项目内 `kernel/`、`dem/`、`dom/`、`build/`。（complete）
2. `sat_sim_cuda` 增加 `--write-depth`，渲染影像时同步输出每像素相机空间 depth 图。（complete）
3. `mars_orbit_to_tsai.py` 增加 `--camera-perturbations`，支持基于 A/B/C/D 原始相机安装矩阵生成虚拟视角，例如 `A_xp8`。（complete）
4. 2048 smoke 渲染通过：`output_pfm_pose_sim/smoke_2048_depth_virtual_20251216_110km` 生成 CameraA 与 CameraA_xp8 的 Float32 图、byte 预览和 depth。（complete）
5. depth+TSAI 重投影一致性验证通过：自投影误差约 0.0002 px；A -> A_xp8 采样点 target inside 约 0.789，inside 深度一致性通过率 1.0。（complete）
6. 新增 `辅助软件/数据模拟/pose_sim_to_pfm_cache.py`，可把两幅仿真图、depth 和 TSAI 转成项目现有 LibTorch pair archive。（complete）
7. 生成并验证 2048 cache smoke pair：`辅助软件/数据模拟/output_pfm_pose_sim/cache_smoke_2048/source_000001_20251216_110km_A_Axp8/pair_000001.pt`，有效像素 3,544,642，valid fraction 0.845109，可被 `load_libtorch_pair_archive` 直接读取。（complete）

### 下一步
1. 写批量 pair 构建脚本：遍历 110km/340km、真实 A/B 和虚拟相机目录，按轨道段/地理块生成 train/val/test split。
2. 第一轮正式数据建议用 2048 图像、`render-gap 160` 或 `120`，先只跑 A + 2-4 个虚拟扰动；确认 cache 规模、训练吞吐和匹配收益后再扩到 B 与更多扰动。
3. 批量生成前先加 pair 选择规则：同高度相邻帧、110km vs 340km 近地面重叠帧、虚拟视角同帧，过滤 valid fraction 过低的 pair。

### 错误记录
| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| `git status` 报 `not a git repository` | 1 | 当前工作区无 `.git` 元数据，继续直接编辑和记录进度 |
| `cmake --build build` 报 `could not load cache` | 1 | 复制后的 build 目录只有旧二进制，无 CMake cache；用 `cmake -S . -B build` 重新配置后编译通过 |
| `asp36` 无 `pytest` / `osgeo` Python 模块 | 1 | 测试以 C++ 编译、脚本 `py_compile`、真实渲染和 `pfm-train` 环境读取 cache 为准 |
| base Python `tifffile` 读取 LZW TIFF 缺 `imagecodecs` | 1 | 改用 `asp36/pfm-train` 中的 OpenCV `cv2.imread(..., IMREAD_UNCHANGED)` 读取 Float32 TIFF |

## Epoch 训练、512 点采样与数据迁移收尾 (2026-05-28)

### 状态
- 阶段：complete。
- 当前主 run：`runs/pose_sim2048_721_cross_extreme_v21full256_blend_b1_10epoch_s512_prefetch_20260528_145414`。

### 已完成
1. 训练脚本已支持完整 epoch 训练、epoch shuffle、每 epoch checkpoint、后台预取和 pair archive cache。（complete）
2. 本轮训练使用 train 6635 pair，`batch_pairs=1`，`samples_per_pair=512`，`epochs=10`，总 step 66350，所有 epoch 完整跑完，`skip=0`。（complete）
3. 产物核验通过：最终权重、`metrics.csv`、`eval_summary.csv`、10 个 epoch checkpoint、主中文 PDF 报告均存在。（complete）
4. 额外 `off016` 极端测试报告已生成：`visual_report_extreme_test_off016/training_report_zh.pdf`。（complete）
5. `训练数据/pose_sim_2048_gap30_views10_721` 已由软链接视图转换为实文件目录：14960 个链接目标通过 move 迁入，迁移后软链接数 0，目录约 498G；原 `pose_sim_2048_gap30_views10` 约 125M。（complete）

### 关键结果
- eval summary：loss `0.014629 -> 0.011804`，top1 `0.999432 -> 0.999542`，top5 `1.0 -> 1.0`，mean negative score `0.000788 -> 0.000618`。
- 主报告 `off012`：precision mean `0.1380`，correct mean `141.34/1024`，coverage mean `0.9586`。
- 额外报告 `off016`：precision mean `0.9420`，correct mean `964.59/1024`，coverage mean `0.7076`。

### 后续判断
- 本轮训练完成，但 `off012` 与 `off016` 差异过大，后续应先排查 `off012` 的 GT/crop/坐标口径和样本难度分布，再决定是否继续扩大模型或继续训练。
- 未使用且未迁移的旧目录：`训练数据/pose_sim_cross_position_rendered_2048_gap30_views10`，约 110G，当前保留。

## Extreme v2 扩集与低重叠优化 (2026-05-28)

### 状态
- 阶段：in_progress。
- 当前训练 run 文件：`/tmp/pfm_current_extreme_v2_abstain_run.txt`。

### 已完成
1. `generate_cross_position_pose_pairs.py` 支持 repart source 目录、source 去重和 cache 扫描复用，解决 `_721` 目录不能直接生成 cross 数据的问题。（complete）
2. 新增测试 `python/test_generate_cross_position_pose_pairs.py`；相关 unittest 80 tests OK。（complete）
3. `off012` 低分原因定位：val 的 `off012` 多为 110km 低重叠样本，有效重叠约 `2.7%-7%`；test 的 `off012` 约 `60.7%` 重叠且 precision 约 `0.98`。结论是低重叠场景需要 rejection/no-match 机制，不能强制 raw descriptor 输出固定 1024 matches。（complete）
4. 新建 v2 数据集 `训练数据/pose_sim_cross_position_rendered_extreme_2048_gap30_views10_721_v2`，实际存放在 8T，原路径为软链接。最终为 train/val/test `4200/1200/600`，offset 包含 `1,2,4,8,12,16,20,24,32`。（complete）
5. 处理根分区满导致的数据生成崩溃：旧 non-extreme cross 与 v2 数据移动到 8T，清理崩溃时缺 json 的半写 pair，断点续跑完成。（complete）
6. 启动 5 epoch abstention 训练：同位置 5235 + v2 cross 4200，共 `9435 pair/epoch`，`samples_per_pair=512`，总 `47175 step`。（in_progress）

### 当前风险
- v2 在 8T 机械盘上，训练 I/O 可能比 NVMe 慢；已使用 `prefetch-batches=16`、`prefetch-workers=4` 缓解。
- 低重叠样本本身没有足够可匹配区域，评估必须区分“正确匹配数量”和“拒绝错误匹配能力”。

### 待完成
1. 等训练完成后验证 checkpoint/report。
2. 生成 v2 分桶报告，重点看 low-overlap `off012/off020/off024/off032`。
3. 根据 raw 与 graph matcher 对比决定默认报告/推理是否应启用 graph matcher 或 score/margin 阈值。

## Extreme v2 训练收尾与 GraphMatcher 补训 (2026-05-29)

### 状态
- 阶段：complete。

### 完成项
1. v2 extreme 数据集扩充完成：`训练数据/pose_sim_cross_position_rendered_extreme_2048_gap30_views10_721_v2`，train/val/test `4200/1200/600`，7:2:1，offset 覆盖 `1,2,4,8,12,16,20,24,32`。（complete）
2. 5 epoch 描述子/abstention 训练完成：`runs/pose_sim2048_721_cross_extreme_v2_v21full256_abstain_b1_5epoch_s512_20260529_005651`，总 `47175 step`，`512` 点/图像对，`skip=0`。（complete）
3. raw/graph 两套报告已生成，发现 raw descriptor 在 off012 上 mean precision `0.9789`，但未补训 graph matcher 只有 `0.2537`。（complete）
4. 执行 Stage D GraphMatcher 补训：`runs/pose_sim2048_721_cross_extreme_v2_v21full256_graphmatcher_b1_3epoch_s512_20260529_084108`，冻结描述子，只训练 `graph_matcher`，总 `28305 step`，`skip=0`。（complete）
5. GraphMatcher 补训后 off012 mean precision 提升到 `0.8143`，mean correct `833.84/1024`，mean error `19.94 px`；raw descriptor 保持 `0.9789`。（complete）

### 后续建议
1. 下一轮不要再优先扩大 backbone；当前主要短板是 GraphMatcher 推理/训练目标和 no-match 拒绝能力。
2. 对 low-overlap 样本应加入 dustbin/unmatched 监督或 score/margin 阈值，否则固定输出 1024 matches 会天然制造误匹配。
3. 报告继续保留 raw descriptor 与 graph matcher 双口径，防止 matcher 掩盖或拖累特征提取器真实能力。
## 2026-05-30 GraphMatcher/弱纹理/光照/旋转诊断修复计划

目标：冻结并保护已经可用的 256 维 descriptor，把 GraphMatcher 从“替代 raw descriptor 的最终匹配器”改成“基于 raw descriptor 的 residual reranker/filter”；同时补齐弱纹理 precision、raw top-K recall、光照压力测试、连续旋转压力测试等诊断。

阶段：
- [complete] 阶段 1：定位当前 GraphMatcher 训练和推理实现，确认 residual raw score、top-K、metadata/dustbin、弱纹理指标、光照/旋转诊断缺口。
- [complete] 阶段 2：实现 residual GraphMatcher：`final_score = raw_score / tau + alpha * graph_delta`，默认小 alpha，保留 candidate top-K。
- [complete] 阶段 3：加入 GraphMatcher metadata ablation 支持，至少支持 full/meta-disabled/xy-disabled/geometry-disabled 诊断入口。
- [complete] 阶段 4：报告增加 raw top-K recall、Graph accepted/rejected、worst pairs、按数据来源/极端跨位置分组统计。
- [complete] 阶段 5：报告增加 weak texture precision/recall/count，而不是只输出 weak texture fraction。
- [pending] 阶段 6：增加 illumination branch 压力测试入口：same geometry + different illumination / shadow / low contrast / brightness reversal-like。
- [pending] 阶段 7：增加连续旋转压力测试入口，不只测 0/90/180/270，至少覆盖 30/45/60/120/135/150 等非 C4 角度。
- [complete] 阶段 8：补测试，验证 residual 初始化不低于 raw 主信号，报告 CSV 包含新增诊断字段。
- [complete] 阶段 9：用最近完整训练权重跑一次轻量诊断，不先重训，确认失败 pair 的 raw top-K 上限、弱纹理 precision、GraphMatcher 破坏点。

当前决策：
- 暂不继续端到端混训 extractor。
- 优先让 GraphMatcher 不伤害 raw descriptor。
- 训练下一轮应只开 `--train-graph-matcher`，并通过 trainable selector 冻结 extractor。
- C4 分支保留，但不能把它当作完整旋转鲁棒性的证明；报告必须增加非 90 度旋转测试。

本轮已完成：
- GraphMatcher 改为 residual reranker：raw cosine 是主分数，graph/context/geometry 只作为 delta。
- 可视化报告 CSV 增加 raw top-1/5/16/32/64 recall、selected/rejected、weak texture precision/count。
- `--graph-metadata-mode` 支持 `full`、`descriptor_only`、`no_xy`、`no_geometry`、`no_quality`。
- 对 `pair_004541_cross_off008_s00109_s00119.pt` 的轻量诊断显示：旧权重在 residual 逻辑下 GraphMatcher 从原报告 `17/512` 恢复到 `133/512`，descriptor-only metadata 为 `140/512`。
- GraphMatcher 报告/评估入口新增 inference-time calibration：`--graph-dustbin-delta`、`--graph-acceptance-margin`、`--graph-min-raw-score`、`--graph-min-raw-margin`。
- 以当前最佳 `nm128/w0.5 no_xy dustbin` checkpoint 扫描后确认：小负 dustbin delta 可温和提高 accepted/correct，但会降低 precision；默认仍应使用 high_precision baseline。

仍待完成：
- 光照压力测试独立入口。
- 非 90 度连续旋转压力测试独立入口。
- 按数据来源/极端跨位置自动分组的汇总表。

## 2026-05-30 GraphMatcher hard replay 与弱纹理采样收尾

目标：在不破坏当前 256 维 descriptor 的前提下，继续修 GraphMatcher 对极端跨位置样本的过度拒配/误修正问题，并提高训练中弱纹理点和 hard pair 的出现比例。

已完成：
- [complete] 训练采样支持 `--training-weak-texture-fraction`，在每个 pair 的 512 个监督点中预留低局部纹理点，避免训练只集中在强纹理边缘区域。
- [complete] 训练采样支持 `--hard-pair-glob`，可把 `cross_off008/off012/off016/off020/off024/off032` 等困难样本加入 hard replay。
- [complete] GraphMatcher 新增 acceptance head，但最后一层零初始化，旧 checkpoint 初始行为兼容。
- [complete] GraphMatcher loss 新增 acceptance 辅助监督和 raw-preservation 约束，目标是让 matcher 优先保留 raw descriptor 已经高置信的正确匹配。
- [complete] 补充 focused unittest 和 `py_compile`，覆盖 weak texture quota、acceptance loss、raw-preservation loss、CLI 参数和 GraphMatcher 输出。
- [complete] 完成两组短训探针并写入 `progress.md` / `findings.md`。

实验结论：
- accept-head 探针提高了 accepted/correct 数，但 precision 明显下降；它让 matcher 变得过于宽松，暂不作为主模型。
- raw-preservation + 更强 dustbin 探针更稳，极端样本和弱纹理样本都有一定改善，但仍未超过当前 high-precision baseline 的总体可靠性。
- 当前默认模型仍建议使用 `graphmatcher_no_xy_dustbin_v21full256_b1_1epoch_s512_nm128_eval512_20260530_164330` 这一高精度基线；raw-preservation run 可作为下一轮 balanced tuning 起点。

下一步：
- 继续冻结 extractor，只训 GraphMatcher。
- 以 raw-preservation 探针为起点，调小接受范围、强化 hard negative 和 no-match，而不是直接增大 accept-head 权重。
- 下一轮报告必须同时看 micro precision、extreme precision、required hard pair correct/accepted、weak texture precision，不能只看平均值。

## 2026-05-31 GraphMatcher hard-negative dustbin 优化

目标：在 raw-preservation 增加召回的基础上，专门压制 raw descriptor 最容易混淆的 off-diagonal hard negatives，避免 GraphMatcher 把重复纹理/弱纹理候选误接受。

已完成：
- [complete] 增加 `graph_matcher_hard_negative_dustbin_loss`，训练排序目标为 `positive logit > dustbin logit > hard negative logit`。
- [complete] 增加 CLI 参数：`--graph-matcher-hard-negative-dustbin-weight/topk/margin`。
- [complete] 增加单元测试并跑 focused 验证：`146 tests OK, 1 skipped`。
- [complete] 运行强权重 probe：`graphmatcher_hardnegdustbin_rawpreserve_v21full256_b1_1000_s512_20260531_101030`。
- [complete] 运行轻权重 continuation probe：`graphmatcher_hardnegdustbin_light_from_rawpreserve_v21full256_b1_600_s512_20260531_101916`。
- [complete] 对轻权重 probe 增加 raw score/margin 过滤报告：`visual_report/graph_matcher_filter_s04_m001`。

结果：
- high_precision baseline：micro `7896/7973=0.990342`，extreme `2069/2104=0.983365`，指定极端样本 `89/106=0.839623`。
- raw-preservation probe：micro `7999/8159=0.980390`，extreme `2117/2201=0.961836`，指定极端样本 `101/138=0.731884`。
- hard-negative dustbin 强权重：micro `7892/7966=0.990711`，extreme `2067/2101=0.983817`，指定极端样本 `89/105=0.847619`。
- hard-negative dustbin 轻权重 + `raw_score>=0.4` + `raw_margin>=0.01`：micro `7970/8096=0.984437`，extreme `2099/2159=0.972209`，指定极端样本 `100/125=0.800000`。

当前决策：
- 默认高精度模型仍是 `graphmatcher_no_xy_dustbin_v21full256_b1_1epoch_s512_nm128_eval512_20260530_164330`。
- 当前最好的 balanced experimental 模型是 `graphmatcher_hardnegdustbin_light_from_rawpreserve_v21full256_b1_600_s512_20260531_101916`，推理/报告使用 `graph_min_raw_score=0.4` 和 `graph_min_raw_margin=0.01`。
- 下一阶段若继续优化，应围绕 hard negative 的类别分层做，不再简单调大/调小一个全局 dustbin 权重。
