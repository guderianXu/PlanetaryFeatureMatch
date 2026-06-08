# scripts 目录说明

这个目录只保留当前还可能复用的工程脚本。历史的 `matcher_algorithm_iteration_agent*`、旧 presentation 生成器、旧 fixed/extreme 对比入口已经移除；那些结论仍保留在历史 `runs/`、`progress.md` 和 `task_plan.md` 里，不再作为可执行维护入口。

运行这些脚本时通常使用：

```bash
PYTHONPATH=python:scripts /home/xjw/.local/share/mamba/envs/plascan/bin/python scripts/<script>.py ...
```

## 训练与 lazy 数据流

| 脚本 | 用途 | 保留原因 |
| --- | --- | --- |
| `benchmark_lazy_pose_pairs.py` | 从 pose render manifest 在线构造训练 pair；`--pair-mode spatial-index --spatial-index-height-km 100,250` 用 TSAI 相机 footprint 建空间索引并只纳入 100/250km 高度，`overlap-list --overlap-scan-all` 可生成固定重叠边 CSV，`train` 模式可用 `--pair-spec-manifest` 直接读取。 | 当前全量训练主入口，dashboard 训练也围绕它读取指标。 |
| `visualize_lazy_pose_matches.py` | 对 lazy pose pair 和 checkpoint 生成代表性匹配连线图。 | 训练结束自动报告和历史训练页依赖它。 |
| `watch_lazy_visual_report.py` | 等待训练 checkpoint 出现后自动触发 lazy 可视化报告。 | 长训练过程中自动补图使用。 |
| `training_visual_report.py` | 对已有 cache/checkpoint 生成训练曲线、直方图、匹配图和 HTML/PDF 报告。 | 历史训练可视化和诊断主入口。 |

## 数据 cache 与仿真后处理

| 脚本 | 用途 | 保留原因 |
| --- | --- | --- |
| `generate_pose_manifest_pair_cache.py` | 从 pose-sim `render_manifest.csv` 显式物化 `pair_*.pt` cache。 | 当不想 lazy 读取、需要固定 cache 训练或校验时使用。 |
| `generate_cross_position_pose_pairs.py` | 基于已有仿真 archive 构造跨位置 pair。 | 生成更难位姿变化数据时使用。 |
| `compact_pose_sim_pair_cache.py` | 将重复视图去重存储，降低 pair cache 体积。 | 老 cache 压缩和迁移时使用。 |
| `repartition_pair_cache.py` | 按 7:2:1 等比例重新划分 train/val/test，可复制或链接。 | 数据集重新分区仍会用。 |
| `verify_pair_cache_dataset.py` | 训练前检查 cache 总数、split、manifest 和 `.pt` 可加载性。 | 训练前必备校验工具。 |

## 伪标签、难例与错误挖掘

| 脚本 | 用途 | 保留原因 |
| --- | --- | --- |
| `generate_rootsift_pseudo_labels.py` | 用 RootSIFT/H-RANSAC 从 cache 中生成伪标签。 | 后续做 teacher label 或对照实验会用。 |
| `generate_gatezero_rootsift_pseudo_labels.py` | 只对当前 route 零匹配样本生成 RootSIFT 伪标签。 | 针对失败样本补充监督时使用。 |
| `generate_p1_viewpoint_retention_labels.py` | 生成 P1 viewpoint retention 标签。 | 之前 P1 retention 实验仍可复现。 |
| `mine_pfm_false_matches.py` | 从 PFM 原始匹配里挖错配点。 | hard negative 和错误诊断使用。 |

## 评估与诊断

| 脚本 | 用途 | 保留原因 |
| --- | --- | --- |
| `rotation_matcher_comparison.py` | 两阶段旋转 matcher 对比，不改训练代码。 | 旋转鲁棒性基线评估。 |
| `illumination_stress_eval.py` | 构造确定性光照变化，评估匹配鲁棒性。 | 分析光照不变性问题。 |
| `continuous_rotation_stress_eval.py` | 连续角度旋转压力测试。 | 检查非 90 度旋转适应性。 |
| `run_graph_matcher_mode_report.py` | 用命名 GraphMatcher profile 生成报告。 | 快速复现某个 graph matcher 配置。 |
| `sweep_graph_inference_configs.py` | 扫 graph inference 阈值并生成 HTML 汇总。 | 调整 graph 推理速度/精度折中。 |
| `stratify_graph_match_errors.py` | 按难度、失败类型汇总 GraphMatcher 报告 CSV。 | 训练后错误归因。 |

## 维护工具

| 脚本 | 用途 | 保留原因 |
| --- | --- | --- |
| `compare_pfm_v21_state_keys.py` | 对比 Python 模型参数名/shape 和 C++ v2.1 镜像。 | Python/C++ 对齐时使用。 |

如果新增脚本，必须在本 README 写明用途、输入、输出和是否属于当前主链；一次性探索脚本不要再长期留在 `scripts/` 根目录。
