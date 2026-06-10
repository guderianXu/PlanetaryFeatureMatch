"""Training stability helpers for long-running PFM experiments."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import fmean
from typing import Deque


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class StabilityThresholds:
    min_steps_before_early_stop: int = 1000
    rolling_window: int = 200
    max_nan_in_window: int = 20
    max_loss_multiplier: float = 3.0
    min_top1_mean: float = 0.35
    min_match_score: float = -0.5


@dataclass(frozen=True)
class StabilityDecision:
    should_stop: bool
    should_save_latest: bool
    should_save_last_good: bool
    is_last_good: bool
    reason: str = ""


class RollingMetricWindow:
    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self._rows: Deque[dict[str, object]] = deque(maxlen=int(size))

    @property
    def count(self) -> int:
        return len(self._rows)

    def add(self, row: dict[str, object]) -> None:
        self._rows.append(dict(row))

    def values(self, key: str) -> list[float]:
        values: list[float] = []
        for row in self._rows:
            value = finite_float(row.get(key))
            if value is not None:
                values.append(value)
        return values

    def mean(self, key: str) -> float:
        values = self.values(key)
        return fmean(values) if values else float("nan")

    def nonfinite_count(self, key: str) -> int:
        return sum(1 for row in self._rows if finite_float(row.get(key)) is None)


class TrainingStabilityTracker:
    def __init__(self, *, thresholds: StabilityThresholds | None = None) -> None:
        self.thresholds = thresholds or StabilityThresholds()
        self.window = RollingMetricWindow(self.thresholds.rolling_window)
        self.best_recent_loss = float("inf")
        self.best_match_score = -float("inf")

    def match_score(self, metrics: dict[str, object]) -> float:
        top1 = finite_float(metrics.get("top1_accuracy")) or 0.0
        rejected = finite_float(metrics.get("true_match_rejected_by_dustbin_ratio")) or 0.0
        false_accept = finite_float(metrics.get("false_match_accepted_ratio")) or 0.0
        margin = finite_float(metrics.get("positive_vs_dustbin_margin_mean")) or 0.0
        loss = finite_float(metrics.get("loss"))
        nan_penalty = 1.0 if loss is None else 0.0
        return top1 + 0.25 * margin - 0.75 * rejected - 0.25 * false_accept - nan_penalty

    def update(self, step: int, metrics: dict[str, object]) -> StabilityDecision:
        self.window.add(metrics)
        loss = finite_float(metrics.get("loss"))
        top1 = finite_float(metrics.get("top1_accuracy"))
        score = self.match_score(metrics)
        self.best_match_score = max(self.best_match_score, score)
        recent_loss = self.window.mean("loss")
        if math.isfinite(recent_loss):
            self.best_recent_loss = min(self.best_recent_loss, recent_loss)
        is_last_good = loss is not None and top1 is not None and top1 >= self.thresholds.min_top1_mean
        should_stop = False
        reason = ""
        if step >= self.thresholds.min_steps_before_early_stop:
            top1_mean = self.window.mean("top1_accuracy")
            if self.window.nonfinite_count("loss") > self.thresholds.max_nan_in_window:
                should_stop = True
                reason = "too_many_nonfinite_loss_values"
            elif math.isfinite(top1_mean) and top1_mean < self.thresholds.min_top1_mean:
                should_stop = True
                reason = "top1_mean_below_threshold"
            elif (
                math.isfinite(recent_loss)
                and math.isfinite(self.best_recent_loss)
                and recent_loss > self.best_recent_loss * self.thresholds.max_loss_multiplier
            ):
                should_stop = True
                reason = "recent_loss_exceeded_best_window"
            elif score < self.thresholds.min_match_score:
                should_stop = True
                reason = "match_score_below_threshold"
        return StabilityDecision(
            should_stop=should_stop,
            should_save_latest=True,
            should_save_last_good=is_last_good,
            is_last_good=is_last_good,
            reason=reason,
        )
