#!/usr/bin/env python3
"""Apply an observable pair-level PFM/LightGlue hybrid gate."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Callable, Sequence


IDENTITY_FIELDS = [
    "source_name",
    "split",
    "pair_index",
    "pair_type",
    "base_id",
    "reference_variant",
    "target_variant",
]

OUTPUT_FIELDS = [
    *IDENTITY_FIELDS,
    "gate_selected_pfm",
    "chosen_source",
    "matches",
    "correct",
    "wrong",
    "precision",
    "pfm_matches",
    "pfm_correct",
    "pfm_wrong",
    "pfm_precision",
    "lightglue_matches",
    "lightglue_correct",
    "lightglue_wrong",
    "lightglue_precision",
]

FORBIDDEN_FIELD_FRAGMENTS = (
    "correct",
    "wrong",
    "label",
    "teacher",
    "oracle",
    "delta",
)

CONDITION_RE = re.compile(r"^([A-Za-z0-9_]+)\s*(>=|<=|==)\s*(.+?)\s*$")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _validate_gate_field(field: str) -> None:
    if field == "target_variant":
        return
    lowered = field.lower()
    if not field.startswith("feature_") or any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS):
        raise ValueError(f"gate field is not observable at inference time: {field}")


def _compile_condition(text: str) -> Callable[[dict[str, str]], bool]:
    match = CONDITION_RE.match(text.strip())
    if match is None:
        raise ValueError(f"unsupported gate condition: {text}")
    field, operator, raw_value = match.groups()
    _validate_gate_field(field)
    value = raw_value.strip().strip("'\"")
    if field == "target_variant":
        if operator != "==":
            raise ValueError("target_variant only supports ==")
        return lambda row, value=value: row.get("target_variant", "") == value

    try:
        threshold = float(value)
    except ValueError as exc:
        raise ValueError(f"numeric feature gate requires a float threshold: {text}") from exc
    if not math.isfinite(threshold):
        raise ValueError(f"numeric feature gate threshold must be finite: {text}")
    if operator == ">=":
        return lambda row, field=field, threshold=threshold: _float_value(row, field, float("-inf")) >= threshold
    if operator == "<=":
        return lambda row, field=field, threshold=threshold: _float_value(row, field, float("inf")) <= threshold
    if operator == "==":
        return lambda row, field=field, threshold=threshold: _float_value(row, field, float("nan")) == threshold
    raise ValueError(f"unsupported gate operator: {operator}")


def compile_gate(gate: str) -> Callable[[dict[str, str]], bool]:
    clauses = [clause.strip() for clause in gate.split(" OR ") if clause.strip()]
    if not clauses:
        raise ValueError("gate must contain at least one condition")
    compiled_clauses: list[list[Callable[[dict[str, str]], bool]]] = []
    for clause in clauses:
        parts = [part.strip() for part in clause.split(" AND ") if part.strip()]
        if not parts:
            raise ValueError(f"gate clause must contain at least one condition: {clause}")
        compiled_clauses.append([_compile_condition(part) for part in parts])
    return lambda row: any(all(condition(row) for condition in clause) for clause in compiled_clauses)


def _chosen_metrics(row: dict[str, str], *, selected_pfm: bool) -> tuple[str, int, int, int, float]:
    prefix = "pfm" if selected_pfm else "lightglue"
    matches = _int_value(row, f"{prefix}_matches")
    correct = _int_value(row, f"{prefix}_correct")
    wrong = _int_value(row, f"{prefix}_wrong")
    precision = correct / matches if matches else 0.0
    return prefix, matches, correct, wrong, precision


def build_hybrid_rows(
    rows: Sequence[dict[str, str]],
    gate_fn: Callable[[dict[str, str]], bool],
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for row in rows:
        selected_pfm = bool(gate_fn(row))
        chosen_source, matches, correct, wrong, precision = _chosen_metrics(row, selected_pfm=selected_pfm)
        output = {field: row.get(field, "") for field in IDENTITY_FIELDS}
        output.update(
            {
                "gate_selected_pfm": "1" if selected_pfm else "0",
                "chosen_source": chosen_source,
                "matches": str(matches),
                "correct": str(correct),
                "wrong": str(wrong),
                "precision": _fmt_float(precision),
                "pfm_matches": str(_int_value(row, "pfm_matches")),
                "pfm_correct": str(_int_value(row, "pfm_correct")),
                "pfm_wrong": str(_int_value(row, "pfm_wrong")),
                "pfm_precision": _fmt_float(_float_value(row, "pfm_precision")),
                "lightglue_matches": str(_int_value(row, "lightglue_matches")),
                "lightglue_correct": str(_int_value(row, "lightglue_correct")),
                "lightglue_wrong": str(_int_value(row, "lightglue_wrong")),
                "lightglue_precision": _fmt_float(_float_value(row, "lightglue_precision")),
            }
        )
        output_rows.append(output)
    return output_rows


def _summary_for_rows(rows: Sequence[dict[str, str]]) -> dict[str, int | float]:
    pfm_matches = sum(_int_value(row, "pfm_matches") for row in rows)
    pfm_correct = sum(_int_value(row, "pfm_correct") for row in rows)
    pfm_wrong = sum(_int_value(row, "pfm_wrong") for row in rows)
    lightglue_matches = sum(_int_value(row, "lightglue_matches") for row in rows)
    lightglue_correct = sum(_int_value(row, "lightglue_correct") for row in rows)
    lightglue_wrong = sum(_int_value(row, "lightglue_wrong") for row in rows)
    hybrid_matches = sum(_int_value(row, "matches") for row in rows)
    hybrid_correct = sum(_int_value(row, "correct") for row in rows)
    hybrid_wrong = sum(_int_value(row, "wrong") for row in rows)
    kept_pfm_rows = sum(1 for row in rows if row.get("chosen_source") == "pfm")
    fallback_lightglue_rows = sum(1 for row in rows if row.get("chosen_source") == "lightglue")
    return {
        "rows": len(rows),
        "kept_pfm_rows": kept_pfm_rows,
        "fallback_lightglue_rows": fallback_lightglue_rows,
        "pfm_matches": pfm_matches,
        "pfm_correct": pfm_correct,
        "pfm_wrong": pfm_wrong,
        "pfm_precision": pfm_correct / pfm_matches if pfm_matches else 0.0,
        "lightglue_matches": lightglue_matches,
        "lightglue_correct": lightglue_correct,
        "lightglue_wrong": lightglue_wrong,
        "lightglue_precision": lightglue_correct / lightglue_matches if lightglue_matches else 0.0,
        "hybrid_matches": hybrid_matches,
        "hybrid_correct": hybrid_correct,
        "hybrid_wrong": hybrid_wrong,
        "hybrid_precision": hybrid_correct / hybrid_matches if hybrid_matches else 0.0,
        "correct_delta_vs_lightglue": hybrid_correct - lightglue_correct,
        "wrong_delta_vs_lightglue": hybrid_wrong - lightglue_wrong,
        "correct_delta_vs_pfm": hybrid_correct - pfm_correct,
        "wrong_delta_vs_pfm": hybrid_wrong - pfm_wrong,
    }


def summarize_rows(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    summary: dict[str, object] = dict(_summary_for_rows(rows))
    by_split: dict[str, dict[str, int | float]] = {}
    by_variant: dict[str, dict[str, int | float]] = {}
    for split in sorted({row.get("split", "") for row in rows}):
        by_split[split] = _summary_for_rows([row for row in rows if row.get("split", "") == split])
    for variant in sorted({row.get("target_variant", "") for row in rows}):
        by_variant[variant] = _summary_for_rows([row for row in rows if row.get("target_variant", "") == variant])
    summary["by_split"] = by_split
    summary["by_variant"] = by_variant
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, *, dataset_csv: Path, gate: str, output_csv: Path, summary: dict[str, object]) -> None:
    keys = [
        "rows",
        "kept_pfm_rows",
        "fallback_lightglue_rows",
        "pfm_correct",
        "pfm_wrong",
        "pfm_precision",
        "lightglue_correct",
        "lightglue_wrong",
        "lightglue_precision",
        "hybrid_correct",
        "hybrid_wrong",
        "hybrid_precision",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
    ]
    metric_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(summary.get(key, '')))}</td></tr>"
        for key in keys
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Observable Pair Gate Apply</title>",
                "<h1>Observable Pair Gate Apply</h1>",
                f"<p>dataset_csv={html.escape(str(dataset_csv))}</p>",
                f"<p>gate={html.escape(gate)}</p>",
                f"<p>hybrid_rows={html.escape(str(output_csv))}</p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                metric_rows,
                "</table>",
                "<h2>Summary JSON</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def apply_observable_gate(*, dataset_csv: Path, gate: str, output_dir: Path) -> dict[str, object]:
    rows = read_csv_rows(dataset_csv)
    if not rows:
        raise ValueError(f"dataset is empty: {dataset_csv}")
    gate_fn = compile_gate(gate)
    hybrid_rows = build_hybrid_rows(rows, gate_fn)
    summary = summarize_rows(hybrid_rows)
    summary["dataset_csv"] = str(dataset_csv)
    summary["gate"] = gate

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "hybrid_rows.csv"
    _write_csv(output_csv, hybrid_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_html(output_dir / "index.html", dataset_csv=dataset_csv, gate=gate, output_csv=output_csv, summary=summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = apply_observable_gate(dataset_csv=args.dataset_csv, gate=args.gate, output_dir=args.output_dir)
    print(
        "observable_pair_gate_apply "
        f"kept_pfm={summary['kept_pfm_rows']} "
        f"fallback_lightglue={summary['fallback_lightglue_rows']} "
        f"correct_delta_vs_lightglue={summary['correct_delta_vs_lightglue']} "
        f"wrong_delta_vs_lightglue={summary['wrong_delta_vs_lightglue']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
