# Python 代码目录说明

这个目录现在同时承载模型、训练、评估、数据 cache 工具和历史实验入口。它们大多还在被使用，但不应该长期平铺在同一层。整理时优先“分层搬迁 + 保留兼容入口”，不要直接删除。

## 核心模型与共享数据结构

- `pfm_model.py`：PyTorch 主模型、checkpoint 读写、GraphMatcher 和 Python/C++ 对齐接口。训练、评估、可视化都会 import，不能删。
- `pfm_model_descriptors.py`：descriptor 几何池化与稳定归一化工具。它从 `pfm_model.py` 拆出，便于单独测试和继续优化旋转鲁棒性。
- `compact_pair_cache.py`：compact pair cache 的共享存储格式。仿真 pair 转换、训练读取和校验脚本都会用。
- `patch_descriptor_training.py`：早期 patch 训练脚本，同时还提供 `SyntheticPair`、pair archive 读取和 descriptor loss helper。短期不能删，后续应拆成 `data/pair_cache.py` 和 `training/descriptor_losses.py`。
- `pose_pair_metadata.py`：读取仿真相机位姿、baseline、视角等训练元数据。训练报告和评估会用。
- `graph_matcher_modes.py`：GraphMatcher 评估配置预设。报告脚本会按名字读取这些预设。

## 当前训练与评估入口

- `pfm_pytorch_training.py`：当前 Python 训练主入口。文件过大，后续应优先拆分为参数解析、数据读取、采样/缓存、loss、训练循环和报告调用。
- `pytorch_cache_match_eval.py`：当前 PyTorch checkpoint 的 cache 匹配评估入口。文件也偏大，后续应拆分为 feature 抽取、匹配过滤、指标统计和 CLI。
- `cross_view_experiment.py`：跨视角训练/校准/评估编排脚本，偏实验性质。还被测试覆盖，建议后续移动到 `scripts/experiments/`，根目录保留兼容 wrapper。
- `rotation_matcher_benchmark.py`：传统/外部 matcher 的旋转鲁棒性基线评测。属于实验工具，不是训练核心。

## 数据与实验辅助工具

- `cache_split.py`：按 source 级别拆分 train/val/test 的小工具，被 cross-view 实验使用。
- `cache_match_eval.py`：较早的 synthetic pair cache 评估 CLI，主要用于生成 cache 后快速验证。不是核心训练依赖。
- `hard_pair_mining.py`：从评估结果里选 hard pair。训练入口会 import，短期保留。
- `pseudo_label_generation.py`：用传统 matcher 生成高精度伪标签。伪标签脚本仍在使用。
- `export_pytorch_state_to_libtorch.py`：把 Python checkpoint 导出成 C++/LibTorch 可读格式。C++ 对齐阶段需要。
- `interpolate_pytorch_states.py`：checkpoint 插值小工具，目前引用很少，属于可迁移到 `scripts/tools/` 的低风险目标。

## 测试目录

所有 Python 测试统一放在 `python/tests/`：

```bash
PYTHONPATH=python:scripts /home/xjw/.local/share/mamba/envs/plascan/bin/python -m unittest discover -s python/tests -p 'test_*.py'
```

新增测试不要再放回 `python/` 根目录。
