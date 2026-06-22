#!/usr/bin/env python3
"""Sweep match-filter reject thresholds using selection and validation sources."""

from __future__ import annotations

import argparse
import bisect
import csv
import html
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SourceSpec:
    name: str
    predictions_csv: Path
    lightglue_correct: int
    lightglue_wrong: int


@dataclass(frozen=True)
class MatchPrediction:
    target_variant: str
    reject_probability: float
    correct: int


@dataclass(frozen=True)
class Metrics:
    matches: int
    correct: int
    wrong: int
    lightglue_correct: int = 0
    lightglue_wrong: int = 0

    @property
    def precision(self) -> float:
        return self.correct / self.matches if self.matches > 0 else 0.0

    @property
    def correct_delta_vs_lightglue(self) -> int:
        return self.correct - self.lightglue_correct

    @property
    def wrong_delta_vs_lightglue(self) -> int:
        return self.wrong - self.lightglue_wrong

    def to_json(self) -> dict[str, int | float]:
        return {
            "matches": self.matches,
            "correct": self.correct,
            "wrong": self.wrong,
            "precision": self.precision,
            "lightglue_correct": self.lightglue_correct,
            "lightglue_wrong": self.lightglue_wrong,
            "correct_delta_vs_lightglue": self.correct_delta_vs_lightglue,
            "wrong_delta_vs_lightglue": self.wrong_delta_vs_lightglue,
        }


class VariantPrefixStats:
    def __init__(self, rows: Sequence[MatchPrediction]) -> None:
        sorted_rows = sorted(rows, key=lambda row: row.reject_probability)
        self.probabilities = [row.reject_probability for row in sorted_rows]
        self.correct_prefix = [0]
        for row in sorted_rows:
            self.correct_prefix.append(self.correct_prefix[-1] + int(row.correct > 0))

    def score(self, threshold: float) -> tuple[int, int, int]:
        kept = bisect.bisect_left(self.probabilities, threshold)
        correct = self.correct_prefix[kept]
        wrong = kept - correct
        return kept, correct, wrong


class SourceData:
    def __init__(self, spec: SourceSpec, rows: Sequence[MatchPrediction]) -> None:
        self.spec = spec
        self.rows = list(rows)
        grouped: dict[str, list[MatchPrediction]] = {}
        for row in rows:
            grouped.setdefault(row.target_variant, []).append(row)
        self.by_variant = {variant: VariantPrefixStats(items) for variant, items in grouped.items()}

    def variants(self) -> set[str]:
        return set(self.by_variant)

    def evaluate(self, thresholds: dict[str, float], *, default_threshold: float) -> Metrics:
        matches = 0
        correct = 0
        wrong = 0
        for variant, stats in self.by_variant.items():
            kept, variant_correct, variant_wrong = stats.score(thresholds.get(variant, default_threshold))
            matches += kept
            correct += variant_correct
            wrong += variant_wrong
        return Metrics(
            matches=matches,
            correct=correct,
            wrong=wrong,
            lightglue_correct=self.spec.lightglue_correct,
            lightglue_wrong=self.spec.lightglue_wrong,
        )


@dataclass(frozen=True)
class ThresholdResult:
    thresholds: dict[str, float]
    select: Metrics
    validation: dict[str, Metrics]
    validation_aggregate: Metrics
    eligible: bool

    def ranking_key(self) -> tuple[int, int, float, int]:
        return (
            self.select.correct_delta_vs_lightglue,
            -self.select.wrong_delta_vs_lightglue,
            self.select.precision,
            self.select.correct,
        )


def _parse_source(value: str) -> SourceSpec:
    parts = [item.strip() for item in value.split(",", 3)]
    if len(parts) != 4 or any(part == "" for part in parts):
        raise ValueError(
            "source must be formatted as name,predictions_csv,lightglue_correct,lightglue_wrong"
        )
    name, path_text, correct_text, wrong_text = parts
    return SourceSpec(
        name=name,
        predictions_csv=Path(path_text),
        lightglue_correct=int(correct_text),
        lightglue_wrong=int(wrong_text),
    )


def _float_value(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing {key}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"invalid {key}: {value!r}")
    return parsed


def _read_predictions(path: Path) -> list[MatchPrediction]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                MatchPrediction(
                    target_variant=row.get("target_variant", ""),
                    reject_probability=_float_value(row, "reject_probability"),
                    correct=1 if _float_value(row, "correct") > 0.0 else 0,
                )
            )
    return rows


def _load_source(spec: SourceSpec) -> SourceData:
    if not spec.predictions_csv.is_file():
        raise FileNotFoundError(f"missing predictions CSV: {spec.predictions_csv}")
    return SourceData(spec, _read_predictions(spec.predictions_csv))


def _threshold_after(value: float) -> float:
    return math.nextafter(value, math.inf)


def _limited_thresholds(values: Sequence[float], max_count: int, extra: Sequence[float]) -> list[float]:
    candidates = {0.0, 1.000001}
    candidates.update(_threshold_after(value) for value in values)
    candidates.update(value for value in extra if math.isfinite(value))
    sorted_candidates = sorted(candidates)
    if max_count <= 0 or len(sorted_candidates) <= max_count:
        return sorted_candidates
    if max_count == 1:
        return [sorted_candidates[-1]]
    selected_indexes = {
        round(index * (len(sorted_candidates) - 1) / (max_count - 1))
        for index in range(max_count)
    }
    return [sorted_candidates[index] for index in sorted(selected_indexes)]


def _candidate_thresholds_by_variant(
    sources: Sequence[SourceData],
    *,
    max_thresholds_per_variant: int,
    extra_thresholds: Sequence[float],
) -> dict[str, list[float]]:
    values_by_variant: dict[str, list[float]] = {}
    for source in sources:
        for row in source.rows:
            values_by_variant.setdefault(row.target_variant, []).append(row.reject_probability)
    return {
        variant: _limited_thresholds(values, max_thresholds_per_variant, extra_thresholds)
        for variant, values in sorted(values_by_variant.items())
    }


def _aggregate(metrics: Sequence[Metrics]) -> Metrics:
    return Metrics(
        matches=sum(item.matches for item in metrics),
        correct=sum(item.correct for item in metrics),
        wrong=sum(item.wrong for item in metrics),
        lightglue_correct=sum(item.lightglue_correct for item in metrics),
        lightglue_wrong=sum(item.lightglue_wrong for item in metrics),
    )


def _evaluate_thresholds(
    thresholds: dict[str, float],
    *,
    select_sources: Sequence[SourceData],
    validation_sources: Sequence[SourceData],
    default_threshold: float,
    min_select_correct_delta: int,
    max_select_wrong_delta: int,
) -> ThresholdResult:
    select_metrics = _aggregate(
        [source.evaluate(thresholds, default_threshold=default_threshold) for source in select_sources]
    )
    validation = {
        source.spec.name: source.evaluate(thresholds, default_threshold=default_threshold)
        for source in validation_sources
    }
    validation_aggregate = _aggregate(list(validation.values()))
    eligible = (
        select_metrics.correct_delta_vs_lightglue >= min_select_correct_delta
        and select_metrics.wrong_delta_vs_lightglue <= max_select_wrong_delta
    )
    return ThresholdResult(
        thresholds=dict(thresholds),
        select=select_metrics,
        validation=validation,
        validation_aggregate=validation_aggregate,
        eligible=eligible,
    )


def sweep_thresholds(
    *,
    select_sources: Sequence[SourceData],
    validation_sources: Sequence[SourceData],
    mode: str,
    max_thresholds_per_variant: int,
    extra_thresholds: Sequence[float],
    default_threshold: float,
    min_select_correct_delta: int,
    max_select_wrong_delta: int,
    top_k: int,
) -> list[ThresholdResult]:
    thresholds_by_variant = _candidate_thresholds_by_variant(
        select_sources,
        max_thresholds_per_variant=max_thresholds_per_variant,
        extra_thresholds=extra_thresholds,
    )
    variants = sorted(thresholds_by_variant)
    results: list[ThresholdResult] = []
    if mode == "global":
        global_values = sorted({value for values in thresholds_by_variant.values() for value in values})
        for threshold in global_values:
            thresholds = {variant: threshold for variant in variants}
            results.append(
                _evaluate_thresholds(
                    thresholds,
                    select_sources=select_sources,
                    validation_sources=validation_sources,
                    default_threshold=default_threshold,
                    min_select_correct_delta=min_select_correct_delta,
                    max_select_wrong_delta=max_select_wrong_delta,
                )
            )
    elif mode == "per-target-variant":
        threshold_lists = [thresholds_by_variant[variant] for variant in variants]
        for values in itertools.product(*threshold_lists):
            thresholds = dict(zip(variants, values))
            results.append(
                _evaluate_thresholds(
                    thresholds,
                    select_sources=select_sources,
                    validation_sources=validation_sources,
                    default_threshold=default_threshold,
                    min_select_correct_delta=min_select_correct_delta,
                    max_select_wrong_delta=max_select_wrong_delta,
                )
            )
    else:
        raise ValueError(f"unsupported mode: {mode}")

    eligible = [result for result in results if result.eligible]
    ranked = sorted(eligible or results, key=lambda result: result.ranking_key(), reverse=True)
    return ranked[:top_k]


def _fmt_float(value: float) -> str:
    return f"{value:.9f}"


def _result_to_row(result: ThresholdResult, rank: int, validation_names: Sequence[str]) -> dict[str, str]:
    row: dict[str, str] = {
        "rank": str(rank),
        "eligible": "1" if result.eligible else "0",
        "thresholds_json": json.dumps(result.thresholds, sort_keys=True),
    }
    for variant, threshold in sorted(result.thresholds.items()):
        row[f"threshold_{variant}"] = _fmt_float(threshold)
    for prefix, metrics in [
        ("select", result.select),
        ("validation_aggregate", result.validation_aggregate),
    ]:
        for key, value in metrics.to_json().items():
            row[f"{prefix}_{key}"] = _fmt_float(value) if isinstance(value, float) else str(value)
    for name in validation_names:
        metrics = result.validation[name]
        for key, value in metrics.to_json().items():
            row[f"validation_{name}_{key}"] = _fmt_float(value) if isinstance(value, float) else str(value)
    return row


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, *, payload: dict[str, object], rows: Sequence[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fieldnames)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(row.get(field, ''))}</td>" for field in fieldnames) + "</tr>"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Match Filter Threshold Sweep</title>",
                "<h1>Match Filter Threshold Sweep</h1>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                f"<tr>{header}</tr>",
                body,
                "</table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-source", action="append", required=True)
    parser.add_argument("--validation-source", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["global", "per-target-variant"], default="per-target-variant")
    parser.add_argument("--min-select-correct-delta", type=int, default=1)
    parser.add_argument("--max-select-wrong-delta", type=int, default=0)
    parser.add_argument("--max-thresholds-per-variant", type=int, default=64)
    parser.add_argument("--default-threshold", type=float, default=0.16815922380501608)
    parser.add_argument("--extra-threshold", action="append", type=float, default=[])
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args(argv)
    if args.max_thresholds_per_variant <= 0:
        raise ValueError("--max-thresholds-per-variant must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not math.isfinite(args.default_threshold):
        raise ValueError("--default-threshold must be finite")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    select_specs = [_parse_source(value) for value in args.select_source]
    validation_specs = [_parse_source(value) for value in args.validation_source]
    select_sources = [_load_source(spec) for spec in select_specs]
    validation_sources = [_load_source(spec) for spec in validation_specs]
    results = sweep_thresholds(
        select_sources=select_sources,
        validation_sources=validation_sources,
        mode=args.mode,
        max_thresholds_per_variant=args.max_thresholds_per_variant,
        extra_thresholds=args.extra_threshold,
        default_threshold=args.default_threshold,
        min_select_correct_delta=args.min_select_correct_delta,
        max_select_wrong_delta=args.max_select_wrong_delta,
        top_k=args.top_k,
    )
    if not results:
        raise ValueError("threshold sweep produced no results")
    validation_names = [source.spec.name for source in validation_sources]
    rows = [_result_to_row(result, index, validation_names) for index, result in enumerate(results, 1)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "threshold_sweep.csv", rows)
    best = results[0]
    payload: dict[str, object] = {
        "mode": args.mode,
        "select_sources": [spec.__dict__ | {"predictions_csv": str(spec.predictions_csv)} for spec in select_specs],
        "validation_sources": [
            spec.__dict__ | {"predictions_csv": str(spec.predictions_csv)} for spec in validation_specs
        ],
        "min_select_correct_delta": args.min_select_correct_delta,
        "max_select_wrong_delta": args.max_select_wrong_delta,
        "max_thresholds_per_variant": args.max_thresholds_per_variant,
        "default_threshold": args.default_threshold,
        "top_k": args.top_k,
        "eligible_results": sum(1 for result in results if result.eligible),
        "best": {
            "eligible": best.eligible,
            "thresholds": best.thresholds,
            "select": best.select.to_json(),
            "validation": {name: metrics.to_json() for name, metrics in best.validation.items()},
            "validation_aggregate": best.validation_aggregate.to_json(),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_html(args.output_dir / "index.html", payload=payload, rows=rows)
    print(
        "match_filter_threshold_sweep "
        f"eligible={sum(1 for result in results if result.eligible)} "
        f"best_select_correct={best.select.correct} best_select_wrong={best.select.wrong} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
