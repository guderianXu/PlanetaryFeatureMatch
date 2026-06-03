from __future__ import annotations

import csv
import os
import re
import signal
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
        rows = [{key: _parse_value(value or "") for key, value in row.items()} for row in reader]
        columns = list(reader.fieldnames or [])
    latest = rows[-1] if rows else {}
    return MetricSeries(path=path, columns=columns, rows=rows, latest=latest)


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


def infer_progress(run_path: Path, metrics: MetricSeries, status: str, checkpoint_count: int) -> tuple[float, str]:
    latest = metrics.latest
    script_path = run_path / "train.sh"
    current_step = _number(latest.get("step")) or _number(latest.get("global_step")) or _number(latest.get("batch"))
    target_steps = _script_option(script_path, "--max-train-batches", "--steps")
    if current_step is not None and target_steps and target_steps > 0:
        percent = min(100.0, max(0.0, current_step / target_steps * 100.0))
        return percent, f"{int(current_step)}/{int(target_steps)} 步"

    current_epoch = _number(latest.get("epoch"))
    target_epochs = _script_option(script_path, "--epochs")
    if current_epoch is not None and target_epochs and target_epochs > 0:
        percent = min(100.0, max(0.0, current_epoch / target_epochs * 100.0))
        return percent, f"{int(current_epoch)}/{int(target_epochs)} 轮"

    if status == "running":
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
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "stopped"
    except PermissionError:
        return "unknown"
    return "running"


def discover_runs(root: Path) -> list[RunSummary]:
    if not root.exists():
        return []
    summaries: list[RunSummary] = []
    for run_path in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True):
        metrics = read_metrics_csv(run_path / "metrics.csv")
        checkpoints = list(run_path.glob("*.pt")) + list((run_path / "checkpoints").glob("*.pt"))
        status = pid_status(run_path / "train.pid")
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
                can_start=(run_path / "train.sh").exists() and status != "running",
                can_stop=status == "running",
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
    patterns = "pfm_pytorch_training.py|pfm_cli train|batch_pose_sim_dataset.py|sat_sim_cuda"
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
