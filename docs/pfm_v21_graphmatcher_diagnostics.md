# PFM v2.1 GraphMatcher 诊断与训练入口

本页记录当前 PFM v2.1 的 GraphMatcher 固化入口和新增诊断工具。目标是把 raw descriptor 已经做对的结果保住，再让 GraphMatcher 只在弱纹理、重复纹理和极端跨位置样本上做重排与拒配。

## 固化评估模式

命名配置在 `python/graph_matcher_modes.py`：

- `high_precision`：当前高精度基线，默认不加 raw score/margin 过滤。
- `balanced`：使用 raw preservation 后的平衡版权重，并启用 `graph_min_raw_score=0.4`、`graph_min_raw_margin=0.01`，更适合常规报告。

运行可视化报告：

```bash
/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/run_graph_matcher_mode_report.py balanced \
  --validation-cache-dir /media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据/pose_sim_2048_gap30_views10_721/cache/val \
  --validation-cache-dir /media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据/pose_sim_cross_position_rendered_2048_gap30_views10/cache/val \
  --validation-cache-dir /media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据/pose_sim_cross_position_rendered_extreme_2048_gap30_views10_721_v2/cache/val \
  --pose-metadata-root /media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据/pose_sim_2048_gap30_views10_721 \
  --pose-metadata-root /media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据/pose_sim_cross_position_rendered_2048_gap30_views10 \
  --pose-metadata-root /media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据/pose_sim_cross_position_rendered_extreme_2048_gap30_views10_721_v2 \
  --required-sample-glob '*pair_004541_cross_off008_s00109_s00119.pt'
```

## 分层错误统计

`scripts/stratify_graph_match_errors.py` 读取 `match_visual_summary.csv`，按数据组、难度、precision 区间和弱纹理 precision 区间聚合，方便快速定位 GraphMatcher 是在哪类样本上掉点。

```bash
/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/stratify_graph_match_errors.py \
  runs/.../visual_report/match_visual_summary.csv \
  --output runs/.../graph_match_error_strata.csv
```

## 光照压力测试

`scripts/illumination_stress_eval.py` 固定几何 warp，只对目标图做 gamma、对比度、带状阴影和侧向光照变化，用来验证 illumination/texture branch 是否真正有效。

```bash
/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/illumination_stress_eval.py \
  --pair /path/to/pair.pt \
  --pytorch-state runs/.../pytorch_pfm_state.pt \
  --output runs/.../illumination_stress.csv \
  --matcher-mode graph_matcher \
  --graph-min-raw-score 0.4 \
  --graph-min-raw-margin 0.01
```

## 连续旋转压力测试

`scripts/continuous_rotation_stress_eval.py` 从同一张源图构造已知角度旋转，不只测 0/90/180/270，而是支持任意角度列表，直接检查 C4 分支以外的连续旋转鲁棒性。

```bash
/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/continuous_rotation_stress_eval.py \
  --pair /path/to/pair.pt \
  --pytorch-state runs/.../pytorch_pfm_state.pt \
  --output runs/.../continuous_rotation_stress.csv \
  --angles 0,30,45,60,90,120,135,150,180,270 \
  --matcher-mode graph_matcher \
  --graph-min-raw-score 0.4 \
  --graph-min-raw-margin 0.01
```

## Semi-Dense 候选接入训练

GraphMatcher 训练现在支持把 `SemiDenseCandidateBranch` 产生的半稠密候选作为额外 no-match/dustbin 样本：

```bash
--graph-matcher-semi-dense-no-match-points 64
--graph-matcher-semi-dense-min-score 0.0
--graph-matcher-no-match-weight 0.5
```

这部分不是把 semi-dense 候选强行当作正确匹配，而是让 GraphMatcher 在训练时看到弱纹理候选和 detector-free 候选，并学会拒掉不可靠的候选。这样可以减少正式推理时 semi-dense fallback 对 GraphMatcher 的分布冲击。
