#!/usr/bin/env python3
"""Sweep multi-source observable OR-of-AND PFM/LightGlue hybrid gates."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sweep_observable_pair_gate import (
    Rule,
    _SplitMetricCache,
    build_base_rules,
    build_pairwise_and_rules,
    read_csv_rows,
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_csv: Path


@dataclass(frozen=True)
class _GateState:
    clause_indexes: tuple[int, ...]
    gate: str
    masks_by_source: dict[str, int]


def _parse_source(value: str) -> SourceSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--source must have the form name,dataset_csv")
    name, dataset_csv = parts
    if not name:
        raise argparse.ArgumentTypeError("source name must not be empty")
    return SourceSpec(name=name, dataset_csv=Path(dataset_csv))


def _load_rows_by_source(sources: Sequence[SourceSpec]) -> dict[str, list[dict[str, str]]]:
    if not sources:
        raise ValueError("at least one source is required")
    rows_by_source: dict[str, list[dict[str, str]]] = {}
    for source in sources:
        if source.name in rows_by_source:
            raise ValueError(f"duplicate source name: {source.name}")
        if not source.dataset_csv.exists():
            raise FileNotFoundError(f"missing dataset CSV for {source.name}: {source.dataset_csv}")
        rows = read_csv_rows(source.dataset_csv)
        if not rows:
            raise ValueError(f"dataset is empty for {source.name}: {source.dataset_csv}")
        rows_by_source[source.name] = rows
    return rows_by_source


def _precision_delta(metrics: dict[str, int | float]) -> float:
    return float(metrics["hybrid_precision"]) - float(metrics["lightglue_precision"])


def _state_summary(
    state: _GateState,
    *,
    metric_cache_by_source: dict[str, _SplitMetricCache],
    allow_wrong_delta: int,
    min_correct_delta: int,
    min_precision_delta: float,
) -> dict[str, object]:
    item: dict[str, object] = {
        "gate": state.gate,
        "clause_count": len(state.clause_indexes),
    }
    aggregate = {
        "rows": 0,
        "use_pfm_pairs": 0,
        "fallback_lightglue_pairs": 0,
        "lightglue_matches": 0,
        "lightglue_correct": 0,
        "lightglue_wrong": 0,
        "hybrid_matches": 0,
        "hybrid_correct": 0,
        "hybrid_wrong": 0,
    }
    source_valid_count = 0
    valid_all_sources = True
    for source_name in sorted(metric_cache_by_source):
        metrics = metric_cache_by_source[source_name].summary_for_mask(state.masks_by_source.get(source_name, 0))
        precision_delta = _precision_delta(metrics)
        source_valid = (
            int(metrics["correct_delta_vs_lightglue"]) >= min_correct_delta
            and int(metrics["wrong_delta_vs_lightglue"]) <= allow_wrong_delta
            and precision_delta >= min_precision_delta
        )
        if source_valid:
            source_valid_count += 1
        else:
            valid_all_sources = False
        item[f"{source_name}_valid"] = source_valid
        item[f"{source_name}_use_pfm_pairs"] = metrics["use_pfm_pairs"]
        item[f"{source_name}_correct_delta_vs_lightglue"] = metrics["correct_delta_vs_lightglue"]
        item[f"{source_name}_wrong_delta_vs_lightglue"] = metrics["wrong_delta_vs_lightglue"]
        item[f"{source_name}_precision_delta_vs_lightglue"] = precision_delta
        item[f"{source_name}_hybrid_precision"] = metrics["hybrid_precision"]
        aggregate["rows"] += int(metrics["rows"])
        aggregate["use_pfm_pairs"] += int(metrics["use_pfm_pairs"])
        aggregate["fallback_lightglue_pairs"] += int(metrics["fallback_lightglue_pairs"])
        aggregate["lightglue_matches"] += int(metrics["lightglue_matches"])
        aggregate["lightglue_correct"] += int(metrics["lightglue_correct"])
        aggregate["lightglue_wrong"] += int(metrics["lightglue_wrong"])
        aggregate["hybrid_matches"] += int(metrics["hybrid_matches"])
        aggregate["hybrid_correct"] += int(metrics["hybrid_correct"])
        aggregate["hybrid_wrong"] += int(metrics["hybrid_wrong"])

    aggregate_lightglue_precision = (
        aggregate["lightglue_correct"] / aggregate["lightglue_matches"]
        if aggregate["lightglue_matches"]
        else 0.0
    )
    aggregate_hybrid_precision = (
        aggregate["hybrid_correct"] / aggregate["hybrid_matches"]
        if aggregate["hybrid_matches"]
        else 0.0
    )
    item.update(
        {
            "valid_all_sources": valid_all_sources,
            "source_valid_count": source_valid_count,
            "aggregate_rows": aggregate["rows"],
            "aggregate_use_pfm_pairs": aggregate["use_pfm_pairs"],
            "aggregate_fallback_lightglue_pairs": aggregate["fallback_lightglue_pairs"],
            "aggregate_lightglue_correct": aggregate["lightglue_correct"],
            "aggregate_lightglue_wrong": aggregate["lightglue_wrong"],
            "aggregate_lightglue_precision": aggregate_lightglue_precision,
            "aggregate_hybrid_correct": aggregate["hybrid_correct"],
            "aggregate_hybrid_wrong": aggregate["hybrid_wrong"],
            "aggregate_hybrid_precision": aggregate_hybrid_precision,
            "aggregate_correct_delta_vs_lightglue": aggregate["hybrid_correct"] - aggregate["lightglue_correct"],
            "aggregate_wrong_delta_vs_lightglue": aggregate["hybrid_wrong"] - aggregate["lightglue_wrong"],
            "aggregate_precision_delta_vs_lightglue": aggregate_hybrid_precision - aggregate_lightglue_precision,
        }
    )
    return item


def _summary_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        bool(row.get("valid_all_sources")),
        int(row.get("source_valid_count", 0)),
        int(row.get("aggregate_correct_delta_vs_lightglue", -10**9)),
        -int(row.get("aggregate_wrong_delta_vs_lightglue", 10**9)),
        float(row.get("aggregate_precision_delta_vs_lightglue", -1.0)),
        int(row.get("aggregate_use_pfm_pairs", 0)),
        -int(row.get("clause_count", 0)),
    )


def _add_top_row(rows: list[dict[str, object]], row: dict[str, object], *, limit: int) -> None:
    rows.append(row)
    rows.sort(key=_summary_sort_key, reverse=True)
    if len(rows) > limit:
        del rows[limit:]


def _state_signature(state: _GateState) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(state.masks_by_source.items()))


def _combine_state(state: _GateState, rule: Rule, rule_index: int) -> _GateState:
    masks = {
        source_name: state.masks_by_source.get(source_name, 0) | rule.masks_by_split.get(source_name, 0)
        for source_name in state.masks_by_source
    }
    return _GateState(
        clause_indexes=(*state.clause_indexes, rule_index),
        gate=f"{state.gate} OR {rule.name}",
        masks_by_source=masks,
    )


def _candidate_rules(
    rows_by_source: dict[str, list[dict[str, str]]],
    *,
    max_thresholds: int,
    max_candidate_clauses: int,
    metric_cache_by_source: dict[str, _SplitMetricCache],
    allow_wrong_delta: int,
    min_correct_delta: int,
    min_precision_delta: float,
) -> tuple[list[Rule], int, int]:
    base_rules = build_base_rules(rows_by_source, max_thresholds=max_thresholds)
    clause_rules = build_pairwise_and_rules(base_rules)
    scored: list[tuple[dict[str, object], Rule]] = []
    for index, rule in enumerate(clause_rules):
        state = _GateState(
            clause_indexes=(index,),
            gate=rule.name,
            masks_by_source=dict(rule.masks_by_split),
        )
        summary = _state_summary(
            state,
            metric_cache_by_source=metric_cache_by_source,
            allow_wrong_delta=allow_wrong_delta,
            min_correct_delta=min_correct_delta,
            min_precision_delta=min_precision_delta,
        )
        scored.append((summary, rule))
    scored.sort(key=lambda item: _summary_sort_key(item[0]), reverse=True)
    return [rule for _summary, rule in scored[:max_candidate_clauses]], len(base_rules), len(clause_rules)


def _output_fields(source_names: Sequence[str]) -> list[str]:
    fields = [
        "gate",
        "valid_all_sources",
        "source_valid_count",
        "clause_count",
        "aggregate_use_pfm_pairs",
        "aggregate_lightglue_correct",
        "aggregate_lightglue_wrong",
        "aggregate_hybrid_correct",
        "aggregate_hybrid_wrong",
        "aggregate_correct_delta_vs_lightglue",
        "aggregate_wrong_delta_vs_lightglue",
        "aggregate_precision_delta_vs_lightglue",
    ]
    for source_name in sorted(source_names):
        fields.extend(
            [
                f"{source_name}_valid",
                f"{source_name}_use_pfm_pairs",
                f"{source_name}_correct_delta_vs_lightglue",
                f"{source_name}_wrong_delta_vs_lightglue",
                f"{source_name}_precision_delta_vs_lightglue",
                f"{source_name}_hybrid_precision",
            ]
        )
    return fields


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_html(path: Path, *, summary: dict[str, object], rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows[:50]:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Observable Multi-Source Gate Sweep</title>",
                "<h1>Observable Multi-Source Gate Sweep</h1>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))}</pre>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                f"<tr>{header}</tr>",
                *body,
                "</table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def sweep_multi_source_rules(
    *,
    sources: Sequence[SourceSpec],
    output_dir: Path,
    max_thresholds: int = 36,
    max_candidate_clauses: int = 2000,
    beam_width: int = 128,
    max_clauses: int = 8,
    allow_wrong_delta: int = 0,
    min_correct_delta: int = 1,
    min_precision_delta: float = 0.0,
) -> dict[str, object]:
    rows_by_source = _load_rows_by_source(sources)
    metric_cache_by_source = {
        source_name: _SplitMetricCache.from_rows(rows)
        for source_name, rows in rows_by_source.items()
    }
    candidates, base_rule_count, clause_rule_count = _candidate_rules(
        rows_by_source,
        max_thresholds=max_thresholds,
        max_candidate_clauses=max_candidate_clauses,
        metric_cache_by_source=metric_cache_by_source,
        allow_wrong_delta=allow_wrong_delta,
        min_correct_delta=min_correct_delta,
        min_precision_delta=min_precision_delta,
    )
    if not candidates:
        raise ValueError("no observable candidate clauses were generated")

    source_names = sorted(rows_by_source)
    fields = _output_fields(source_names)
    seen_signatures: set[tuple[tuple[str, int], ...]] = set()
    top_rows: list[dict[str, object]] = []
    valid_rows: list[dict[str, object]] = []
    valid_count = 0
    scored_count = 0

    def score_state(state: _GateState) -> dict[str, object] | None:
        nonlocal scored_count, valid_count
        signature = _state_signature(state)
        if signature in seen_signatures:
            return None
        seen_signatures.add(signature)
        scored_count += 1
        summary = _state_summary(
            state,
            metric_cache_by_source=metric_cache_by_source,
            allow_wrong_delta=allow_wrong_delta,
            min_correct_delta=min_correct_delta,
            min_precision_delta=min_precision_delta,
        )
        _add_top_row(top_rows, summary, limit=500)
        if bool(summary["valid_all_sources"]):
            valid_count += 1
            _add_top_row(valid_rows, summary, limit=500)
        return summary

    beam_states: list[tuple[_GateState, dict[str, object]]] = []
    for index, rule in enumerate(candidates):
        state = _GateState(
            clause_indexes=(index,),
            gate=rule.name,
            masks_by_source=dict(rule.masks_by_split),
        )
        summary = score_state(state)
        if summary is not None:
            beam_states.append((state, summary))
    beam_states.sort(key=lambda item: _summary_sort_key(item[1]), reverse=True)
    beam_states = beam_states[:beam_width]

    for _clause_count in range(2, max_clauses + 1):
        next_states: list[tuple[_GateState, dict[str, object]]] = []
        for state, _summary in beam_states:
            last_index = state.clause_indexes[-1]
            for rule_index in range(last_index + 1, len(candidates)):
                combined = _combine_state(state, candidates[rule_index], rule_index)
                summary = score_state(combined)
                if summary is not None:
                    next_states.append((combined, summary))
        if not next_states:
            break
        next_states.sort(key=lambda item: _summary_sort_key(item[1]), reverse=True)
        beam_states = next_states[:beam_width]

    best_valid = valid_rows[0] if valid_rows else None
    summary: dict[str, object] = {
        "sources": [{"name": source.name, "dataset_csv": str(source.dataset_csv)} for source in sources],
        "max_thresholds": max_thresholds,
        "max_candidate_clauses": max_candidate_clauses,
        "beam_width": beam_width,
        "max_clauses": max_clauses,
        "allow_wrong_delta": allow_wrong_delta,
        "min_correct_delta": min_correct_delta,
        "min_precision_delta": min_precision_delta,
        "base_rule_count": base_rule_count,
        "clause_rule_count": clause_rule_count,
        "candidate_clause_count": len(candidates),
        "scored_gate_count": scored_count,
        "valid_gate_count": valid_count,
        "best_valid": best_valid,
        "best_any": top_rows[0] if top_rows else None,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "best_gates.csv", valid_rows, fields)
    _write_csv(output_dir / "top_gates.csv", top_rows, fields)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_html(output_dir / "index.html", summary=summary, rows=valid_rows or top_rows, fields=fields)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-thresholds", type=int, default=36)
    parser.add_argument("--max-candidate-clauses", type=int, default=2000)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-clauses", type=int, default=8)
    parser.add_argument("--allow-wrong-delta", type=int, default=0)
    parser.add_argument("--min-correct-delta", type=int, default=1)
    parser.add_argument("--min-precision-delta", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = sweep_multi_source_rules(
        sources=args.source,
        output_dir=args.output_dir,
        max_thresholds=args.max_thresholds,
        max_candidate_clauses=args.max_candidate_clauses,
        beam_width=args.beam_width,
        max_clauses=args.max_clauses,
        allow_wrong_delta=args.allow_wrong_delta,
        min_correct_delta=args.min_correct_delta,
        min_precision_delta=args.min_precision_delta,
    )
    best = summary.get("best_valid") or {}
    print(
        "observable_multi_source_gate_sweep "
        f"base_rules={summary['base_rule_count']} "
        f"candidate_clauses={summary['candidate_clause_count']} "
        f"scored_gates={summary['scored_gate_count']} "
        f"valid_gates={summary['valid_gate_count']} "
        f"best_correct_delta={best.get('aggregate_correct_delta_vs_lightglue', '')} "
        f"best_wrong_delta={best.get('aggregate_wrong_delta_vs_lightglue', '')} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
