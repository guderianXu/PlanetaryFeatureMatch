from __future__ import annotations

import csv
import os
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
        summaries.append(
            RunSummary(
                name=run_path.name,
                path=run_path,
                backend=infer_backend(run_path.name, run_path),
                status=status,
                latest_metrics=metrics.latest,
                checkpoint_count=len(checkpoints),
                has_report=(run_path / "run.html").exists() or (run_path / "report").exists(),
                has_log=(run_path / "train.log").exists(),
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
