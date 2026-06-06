from __future__ import annotations

import csv
import os
import re
import shutil
import signal
import time
from pathlib import Path
from typing import Any

from .models import DatasetSummary, MetricSeries, RunSummary


def _parse_value(value: str) -> Any:
    stripped = value.strip()
    if stripped == "":
        return ""
    try:
        return float(stripped)
    except ValueError:
        return stripped


def read_metrics_csv(path: Path) -> MetricSeries:
    if not path.exists():
        return MetricSeries(path=path, columns=[], rows=[], latest={})
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                continue
            rows.append({key: _parse_value(value or "") for key, value in row.items()})
        columns = list(reader.fieldnames or [])
    latest = rows[-1] if rows else {}
    return MetricSeries(path=path, columns=columns, rows=rows, latest=latest)


def run_metrics_path(run_path: Path) -> Path:
    """Return the metric CSV used by a run.

    训练脚本历史上写 `metrics.csv`，新懒加载训练写 `train_metrics.csv`。
    Dashboard 统一在这里兼容两种文件名，避免 UI 和训练脚本各自硬编码。
    """
    metrics_path = run_path / "metrics.csv"
    if metrics_path.exists():
        return metrics_path
    train_metrics_path = run_path / "train_metrics.csv"
    if train_metrics_path.exists():
        return train_metrics_path
    return metrics_path


def infer_backend(run_name: str, run_path: Path) -> str:
    lowered = run_name.lower()
    if "cpp" in lowered or (run_path / "model_final.pt").exists():
        return "cpp"
    if "python" in lowered or (run_path / "pytorch_pfm_state.pt").exists():
        return "python"
    return "unknown"


def _number(value: Any) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _script_option(script_path: Path, *names: str) -> float | None:
    if not script_path.exists():
        return None
    text = script_path.read_text(encoding="utf-8", errors="replace")
    for name in names:
        match = re.search(rf"{re.escape(name)}\s+([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return float(match.group(1))
    return None


def _run_script_paths(run_path: Path) -> list[Path]:
    """Return launch scripts associated with a run directory.

    Dashboard 直接启动的任务使用 runs/<name>/train.sh；命令行长任务历史上常用
    runs/<name>.sh。两种形式都参与进度总步数解析。
    """
    return [run_path / "train.sh", run_path.with_suffix(".sh")]


def _run_script_option(run_path: Path, *names: str) -> float | None:
    for script_path in _run_script_paths(run_path):
        value = _script_option(script_path, *names)
        if value is not None:
            return value
    return None


def _cache_generation_progress(run_path: Path) -> tuple[float, str] | None:
    log_path = run_path / "train.log"
    if not log_path.exists():
        return None
    text = tail_text(log_path, lines=80)
    matches = list(re.finditer(r"kept=(\d+)\s+done=(\d+)/(\d+)", text))
    if matches:
        match = matches[-1]
        kept = int(match.group(1))
        done = int(match.group(2))
        total = int(match.group(3))
        if total > 0:
            percent = min(99.0, max(0.0, done / float(total) * 100.0))
            return percent, f"cache {done}/{total} pair，保留 {kept}"
    candidate_match = re.search(r"candidate_tasks=(\d+)", text)
    if candidate_match:
        return 1.0, f"cache 0/{candidate_match.group(1)} pair"
    if "stage=cache_verify" in text:
        return 99.0, "cache 校验中"
    return None


def infer_progress(run_path: Path, metrics: MetricSeries, status: str, checkpoint_count: int) -> tuple[float, str]:
    latest = metrics.latest
    current_step = (
        _number(latest.get("step"))
        or _number(latest.get("global_step"))
        or _number(latest.get("batch"))
        or _number(latest.get("iteration"))
    )
    total_iterations = _number(latest.get("total_iterations"))
    if current_step is not None and total_iterations and total_iterations > 0:
        percent = min(100.0, max(0.0, current_step / total_iterations * 100.0))
        return percent, f"{int(current_step)}/{int(total_iterations)} 步"

    target_steps = _run_script_option(run_path, "--max-train-batches", "--steps")
    if current_step is not None and target_steps and target_steps > 0:
        percent = min(100.0, max(0.0, current_step / target_steps * 100.0))
        return percent, f"{int(current_step)}/{int(target_steps)} 步"

    current_epoch = _number(latest.get("epoch"))
    target_epochs = _run_script_option(run_path, "--epochs")
    if current_epoch is not None and target_epochs and target_epochs > 0:
        percent = min(100.0, max(0.0, current_epoch / target_epochs * 100.0))
        return percent, f"{int(current_epoch)}/{int(target_epochs)} 轮"

    if status == "running":
        cache_progress = _cache_generation_progress(run_path)
        if cache_progress is not None:
            return cache_progress
        metric_rows = len(metrics.rows)
        return min(95.0, max(6.0, float(metric_rows % 20) * 4.5)), f"{metric_rows} 条指标"
    if checkpoint_count > 0:
        return 100.0, "已写入检查点"
    if metrics.rows:
        return 100.0 if status in {"logged", "stopped"} else 0.0, f"{len(metrics.rows)} 条指标"
    return 0.0, "未开始"


def pid_status(pid_file: Path) -> str:
    if not pid_file.exists():
        return "missing"
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return "invalid"
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            stat_text = proc_stat.read_text(encoding="utf-8", errors="replace")
            if ") Z" in stat_text:
                return "stopped"
        except OSError:
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "stopped"
    except PermissionError:
        return "unknown"
    return "running"


def _file_mtimes(paths: list[Path]) -> list[float]:
    mtimes: list[float] = []
    for path in paths:
        if path.exists():
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                continue
    return mtimes


def _read_process_cmdline(pid_dir: Path) -> list[str]:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _read_process_cwd(pid_dir: Path) -> Path | None:
    try:
        return (pid_dir / "cwd").resolve()
    except OSError:
        return None


def _active_training_output_dirs() -> set[Path]:
    """Find output directories from currently running training commands.

    训练不一定由 Dashboard 启动，可能没有 train.pid。这里读取 /proc 的 cmdline，
    根据 --output-dir / --output_root / --output-root 映射到 run 目录。
    """
    script_names = {
        "benchmark_lazy_pose_pairs.py",
        "pfm_pytorch_training.py",
        "batch_pose_sim_dataset.py",
        "pfm_cli",
    }
    option_names = {"--output-dir", "--output_root", "--output-root", "--run-dir", "--run_dir"}
    active_dirs: set[Path] = set()
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        args = _read_process_cmdline(pid_dir)
        if not args:
            continue
        if not any(any(script_name in arg for script_name in script_names) for arg in args):
            continue
        cwd = _read_process_cwd(pid_dir)
        for index, arg in enumerate(args):
            value: str | None = None
            if arg in option_names and index + 1 < len(args):
                value = args[index + 1]
            else:
                for option_name in option_names:
                    prefix = f"{option_name}="
                    if arg.startswith(prefix):
                        value = arg[len(prefix):]
                        break
            if not value:
                continue
            output_path = Path(value)
            if not output_path.is_absolute():
                if cwd is None:
                    continue
                output_path = cwd / output_path
            try:
                active_dirs.add(output_path.resolve())
            except OSError:
                active_dirs.add(output_path.absolute())
    return active_dirs


def run_created_at(run_path: Path) -> float:
    candidates = [
        run_path / "run.html",
        run_path / "train.sh",
        run_path.with_suffix(".sh"),
        run_path / "metrics.csv",
        run_path / "train_metrics.csv",
        run_path / "train.log",
    ]
    mtimes = _file_mtimes(candidates)
    if mtimes:
        return min(mtimes)
    return run_path.stat().st_mtime


def run_completed_at(run_path: Path, status: str, checkpoint_count: int) -> float | None:
    if status == "running":
        return None
    candidates = [
        run_path / "metrics.csv",
        run_path / "train_metrics.csv",
        run_path / "train.log",
        run_path / "model_final.pt",
        run_path / "pytorch_pfm_state.pt",
    ]
    checkpoints_dir = run_path / "checkpoints"
    if checkpoints_dir.exists():
        candidates.extend(path for path in checkpoints_dir.glob("*.pt"))
    mtimes = _file_mtimes(candidates)
    if mtimes:
        return max(mtimes)
    if checkpoint_count > 0:
        return run_path.stat().st_mtime
    return None


def discover_runs(root: Path) -> list[RunSummary]:
    if not root.exists():
        return []
    active_output_dirs = _active_training_output_dirs()
    summaries: list[RunSummary] = []
    for run_path in sorted((path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")), key=lambda path: path.stat().st_mtime, reverse=True):
        metrics = read_metrics_csv(run_metrics_path(run_path))
        checkpoints = list(run_path.glob("*.pt")) + list((run_path / "checkpoints").glob("*.pt"))
        status = pid_status(run_path / "train.pid")
        try:
            active_external = run_path.resolve() in active_output_dirs
        except OSError:
            active_external = run_path.absolute() in active_output_dirs
        if status in {"missing", "invalid", "stopped"} and active_external:
            status = "running"
        if status == "missing" and (run_path / "train.log").exists():
            status = "logged"
        progress_percent, progress_label = infer_progress(run_path, metrics, status, len(checkpoints))
        summaries.append(
            RunSummary(
                name=run_path.name,
                path=run_path,
                backend=infer_backend(run_path.name, run_path),
                status=status,
                progress_percent=progress_percent,
                progress_label=progress_label,
                latest_metrics=metrics.latest,
                checkpoint_count=len(checkpoints),
                has_report=(run_path / "run.html").exists() or (run_path / "report").exists(),
                has_log=(run_path / "train.log").exists(),
                can_start=any(script_path.exists() for script_path in _run_script_paths(run_path)) and status != "running",
                can_stop=status == "running",
                can_delete=status != "running",
                created_at=run_created_at(run_path),
                completed_at=run_completed_at(run_path, status, len(checkpoints)),
                updated_at=run_path.stat().st_mtime,
            )
        )
    return summaries


def tail_text(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return "".join(content[-max(1, lines):])


def dataset_split_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    total = 0
    for split in ("train", "val", "test"):
        count = len(list((root / split).rglob("pair_*.pt"))) if (root / split).exists() else 0
        counts[split] = count
        total += count
    counts["total"] = total
    return counts


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def summarize_dataset(path: Path) -> DatasetSummary:
    return DatasetSummary(path=path, counts=dataset_split_counts(path), bytes_used=directory_size(path))


def active_training_processes() -> list[str]:
    patterns = "benchmark_lazy_pose_pairs.py|pfm_pytorch_training.py|pfm_cli train|batch_pose_sim_dataset.py|sat_sim_cuda"
    stream = os.popen(f"pgrep -af '{patterns}' || true")
    try:
        return [line.strip() for line in stream.readlines() if line.strip()]
    finally:
        stream.close()


def start_run_script(run_path: Path) -> int:
    script_path = run_path / "train.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"训练脚本缺失：{script_path}")
    status = pid_status(run_path / "train.pid")
    if status == "running":
        raise RuntimeError(f"任务正在运行：{run_path.name}")
    pid = os.fork()
    if pid == 0:
        os.setsid()
        log_path = run_path / "train.log"
        with log_path.open("ab", buffering=0) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            os.execv("/bin/bash", ["/bin/bash", str(script_path)])
    (run_path / "train.pid").write_text(str(pid), encoding="utf-8")
    return pid


def stop_run(run_path: Path) -> int:
    pid_file = run_path / "train.pid"
    if not pid_file.exists():
        raise FileNotFoundError(f"PID 文件缺失：{pid_file}")
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        os.kill(pid, signal.SIGTERM)
    except PermissionError:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        os.kill(pid, signal.SIGTERM)
    return pid


def delete_run(run_path: Path) -> Path:
    if pid_status(run_path / "train.pid") == "running":
        raise RuntimeError(f"任务正在运行，不能删除：{run_path.name}")
    trash_root = run_path.parent / ".trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = trash_root / f"{stamp}_{run_path.name}"
    suffix = 1
    while target.exists():
        target = trash_root / f"{stamp}_{suffix}_{run_path.name}"
        suffix += 1
    shutil.move(str(run_path), str(target))
    return target
