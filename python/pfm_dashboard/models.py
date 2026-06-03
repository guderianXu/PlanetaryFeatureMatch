from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricSeries:
    path: Path
    columns: list[str]
    rows: list[dict[str, Any]]
    latest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSummary:
    name: str
    path: Path
    backend: str
    status: str
    latest_metrics: dict[str, Any]
    checkpoint_count: int
    has_report: bool
    has_log: bool
    updated_at: float


@dataclass(frozen=True)
class DatasetSummary:
    path: Path
    counts: dict[str, int]
    bytes_used: int
