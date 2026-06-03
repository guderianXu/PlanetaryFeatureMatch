# AGENTS.md

本文件给后续 agent 使用，优先级低于用户当前指令和系统/开发者指令。目标是在本项目里改代码、跑训练和生成仿真数据时保持一致做法。

## 项目定位

PlanetaryFeatureMatch 是面向火星、月球、小行星等行星影像的局部特征提取与匹配项目。主线包含：

- C++17 / LibTorch / OpenCV 的 CLI、模型、训练、推理与评估代码。
- Python / PyTorch 的训练、评估、缓存验证、报告和实验脚本。
- `辅助软件/数据模拟/` 下的 ASP + CUDA 仿真数据生成工具。

先读现有实现和测试，再改动。优先延续当前模块边界和命名风格，避免无关重构。

## 工作区约束

- 可能存在长时间运行的数据生成或训练任务。开始新任务前先检查：
  ```bash
  pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda' || true
  ```
- 不要随意删除、覆盖或重新分区已有训练数据、`runs/` 结果、`训练数据_regen/` 输出。需要清理磁盘时先确认路径、规模和用途。
- `runs/` 中的脚本、日志、pid 文件用于记录长任务。新增长任务应留下可复现命令和日志。
- 仓库当前可能不是 git worktree；不要假设一定能 commit。若存在 `.git` 且用户要求提交，先检查状态，保留用户未授权的改动。

## 代码规范

### C++

- 使用 C++17，4 空格缩进，不使用 tab，行宽尽量不超过 120 字符。
- C++ 代码必须使用 Allman brace style：左大括号单独成行，不使用 K&R 风格。
  ```cpp
  if (can_cast)
  {
      cast_ability();
  }
  else
  {
      recover_chakra(delta);
  }
  ```
- 类和结构体使用 PascalCase，例如 `TrainingStageConfig`。
- 函数和方法使用 camelCase，例如 `loadCheckpoint()`。
- 局部变量和普通成员变量使用 snake_case，例如 `pair_count`。
- 私有成员变量前加下划线，例如 `_worker_count`。
- 常量和宏使用大写下划线，例如 `MAX_CANDIDATES`。
- 命名空间使用项目现有的 `pfm` 及其子命名空间，例如 `pfm::training_schedule`。
- 头文件使用 `.h` 和 `#pragma once`。公共声明需要简短文档注释，说明参数、返回值和重要异常条件。
- include 顺序：
  1. 标准库
  2. CUDA / LibTorch / OpenCV / 第三方库
  3. 项目头文件
  每组之间空行分隔，组内尽量按字母序。
- CPU 侧参数非法、构造失败和 IO 失败优先抛出明确异常。CUDA 调用保持现有检查封装风格。
- 一个函数只承担清晰职责。文件超过 400 行或嵌套超过 4 层时优先拆小，但不要为了机械满足限制做无意义拆分。

### Python

- 使用 `pathlib.Path`、`argparse`、`dataclasses` 和结构化读写接口，不用脆弱的临时字符串解析替代 CSV/JSON/Path API。
- 函数、变量、文件名使用 snake_case。类和数据记录使用 PascalCase。
- 脚本入口保持 `parse_args()`、`main()` 结构，错误信息写清楚缺失路径、参数和建议修复方式。
- 大文件和训练缓存要流式处理；不要一次性读入整棵数据集。
- Python 脚本通常通过 `PYTHONPATH=python:scripts` 运行，保持相对 import 可测。

## 测试与验证

改 C++ 后优先跑相关单测，再视影响范围跑全量：

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
./build/pfm_tests
ctest --test-dir build --output-on-failure
```

改 Python 训练、缓存或仿真数据脚本后，优先使用包含 torch 和 cv2 的 `plascan` 环境：

```bash
PYTHONPATH=python:scripts /home/xjw/.local/share/mamba/envs/plascan/bin/python -m unittest python/test_batch_pose_sim_dataset.py
PYTHONPATH=python:scripts /home/xjw/.local/share/mamba/envs/plascan/bin/python -m unittest python/test_generate_cross_position_pose_pairs.py
```

按改动选择更具体的 `python/test_*.py`。涉及 pair cache 的改动至少跑一次 `scripts/verify_pair_cache_dataset.py` 或对应单测确认 `.pt` 可加载、split 比例和 manifest 一致。

## 仿真数据生成

当前数据源路径：

- DEM：`/media/xjw/xjw/xjw2Tdata/code/satsim/环火拍摄轨道数据/dem/mar.tif`
- 高分 DOM：`/media/xjw/xjw/xjw2Tdata/火星dom/HX1_GRAS_MoRIC_DOM_076m_Global_A.tif`
- 低分 DOM：`/media/xjw/xjw/xjw2Tdata/code/satsim/环火拍摄轨道数据/dom/mar_dom.tif`

生成数据时使用显式 `--dem` 和 `--dom`，不要依赖脚本里的旧默认路径。推荐输出到：

```text
/media/xjw/xjw2T/code/deeplearning/PlanetaryFeatureMatch/训练数据_regen
```

不要把大量输出写到 `/media/xjw/xjw`，该盘主要作为 DEM/DOM 数据源且空闲空间紧张。

环境约定：

- `sat_sim_cuda` 和 ASP 相关编译/运行使用 `asp36` conda 环境。
- pair 转换、数据校验和 PyTorch 训练使用 `/home/xjw/.local/share/mamba/envs/plascan/bin/python`。
- 如需给 `batch_pose_sim_dataset.py` 指定训练侧 Python，设置：
  ```bash
  export PFM_TRAIN_PYTHON=/home/xjw/.local/share/mamba/envs/plascan/bin/python
  ```

重新编译仿真程序：

```bash
source /home/xjw/anaconda3/etc/profile.d/conda.sh
conda activate asp36
CXX=/usr/bin/g++ CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -S 辅助软件/数据模拟 -B 辅助软件/数据模拟/build
cmake --build 辅助软件/数据模拟/build -j$(nproc)
```

`/media/xjw/xjw2T` 是 exFAT，不能依赖符号链接；代码应使用复制、硬拷贝或真实文件。大量小 TSAI 文件会有较高磁盘占用，生成时保守使用 `--sat-sim-jobs 1 --frame-workers 1`，确认温度、显存和磁盘后再提高并发。

长任务推荐写成 `runs/*.sh`，使用 `setsid` 或等价方式后台运行。原始 stdout/stderr 可写入 `runs/*.log` 方便实时查看；面向人工留档的任务日志和记录必须写成 `runs/*.html`，其中应包含 dataset、DEM/DOM、split、candidate 范围、free space 和进度。

## 训练与实验

- 小规模 smoke 数据生成后先做 1 轮短训练，确认 pair cache 能被 DataLoader 读取，loss/metrics/checkpoint 能正常落盘。
- 训练输出放在 `runs/<有语义的名称>_<日期时间>/`，不要覆盖已有实验。
- 训练命令里显式写 checkpoint、dataset root、split、resize、batch size、epoch 和 seed。
- 比较实验要保存 CSV 和 HTML 可视化报告，便于回查哪组参数产生了结果。面向人工阅读的日志、实验记录、阶段记录、诊断报告和总结统一使用 HTML 格式。
- 长任务 stdout/stderr 可保留原始 `.log` 方便 tail 和排错，但同一任务应补充 `.html` 汇总页，记录命令、环境、输入输出路径、关键进度、异常和结论。

## Dashboard

- 本项目提供本地工程控制台 `PFM Lab Dashboard`，用于启动 Python/C++ 训练、查看日志和对比 `runs/` 指标。
- 启动命令：
  ```bash
  setsid runs/dashboard_launch_20260603.sh > runs/dashboard_launch_20260603.log 2>&1 &
  ```
- 默认访问地址：`http://127.0.0.1:7860`。
- Dashboard 只管理项目本地 `runs/` 文件，不替代命令行训练；它生成的训练脚本、日志、PID 和 HTML 记录仍应保留在对应 `runs/<实验名>/` 目录。
- 从 Dashboard 启动训练时，要显式选择 Python、C++ 或 Python+C++ 对比后端，并检查 cache、checkpoint、crop、resize、batch、epoch、workers、prefetch 和 memory cache 参数。

## 文档与沟通

- 对用户用中文简洁说明做了什么、验证了什么、还有什么风险。
- 涉及磁盘、数据集规模、训练时间的结论要给出具体路径、数量和命令依据。
- 不要把临时 PID、一次性时间估计或 sudo 密码写进文档。
- 新增面向项目留档的记录时优先写 `.html`；只有机器消费或命令行实时查看需要时才额外写 `.csv`、`.json` 或 `.log`。
