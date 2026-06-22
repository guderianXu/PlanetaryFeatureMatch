#!/usr/bin/env python3
"""Build pair-acceptance training manifests from observable gate hybrid rows."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ACCEPTANCE_FIELDS = [
    "pair_accept_label",
    "pair_accept_weight",
    "pair_accept_source_wrong",
    "pair_accept_source_precision",
]

GATE_ACCEPTANCE_FIELDS = [
    "gate_accept_source",
    "gate_accept_selected_pfm",
    "gate_accept_chosen_source",
    "gate_accept_correct_delta_vs_lightglue",
    "gate_accept_wrong_delta_vs_lightglue",
    "gate_accept_reason",
]

FORBIDDEN_SOURCE_TOKENS = ("fresh", "heldout", "holdout", "lockbox")

PairKey = tuple[str, str, str]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    pair_manifest: Path
    hybrid_rows_csv: Path


def _parse_source(value: str) -> SourceSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--source must have the form name,pair_manifest,hybrid_rows_csv")
    name, pair_manifest, hybrid_rows_csv = parts
    if not name:
        raise argparse.ArgumentTypeError("source name must not be empty")
    return SourceSpec(name=name, pair_manifest=Path(pair_manifest), hybrid_rows_csv=Path(hybrid_rows_csv))


def _clean(value: str | None) -> str:
    return "" if value is None else value.strip()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        rows = [
            {key: ("" if value is None else value) for key, value in row.items() if key is not None}
            for row in reader
        ]
    return list(reader.fieldnames), rows


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = _clean(row.get(key, ""))
    if value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _base_id(row: dict[str, str]) -> str:
    return (
        _clean(row.get("base_id", ""))
        or _clean(row.get("reference_base_id", ""))
        or _clean(row.get("target_base_id", ""))
    )


def _pair_key(row: dict[str, str]) -> PairKey:
    return (
        _clean(row.get("pair_index", "")),
        _base_id(row),
        _clean(row.get("target_variant", "")),
    )


def _assert_allowed_source(source: SourceSpec, rows: Sequence[dict[str, str]]) -> None:
    forbidden = tuple(token.lower() for token in FORBIDDEN_SOURCE_TOKENS)
    source_texts = [source.name]
    for row in rows:
        source_texts.extend([row.get("source_name", ""), row.get("split", ""), row.get("base_id", "")])
    joined = " ".join(source_texts).lower()
    for token in forbidden:
        if token and token in joined:
            raise ValueError(
                f"refusing to build training acceptance labels from forbidden source token '{token}'"
            )


def _hybrid_index(rows: Sequence[dict[str, str]]) -> dict[PairKey, dict[str, str]]:
    candidates: dict[PairKey, list[dict[str, str]]] = {}
    for row in rows:
        candidates.setdefault(_pair_key(row), []).append(row)
    duplicate_keys = [key for key, values in candidates.items() if len(values) > 1]
    if duplicate_keys:
        raise ValueError(f"hybrid rows contain duplicate pair keys: {duplicate_keys[:3]}")
    return {key: values[0] for key, values in candidates.items()}


def _label_for_hybrid_row(
    row: dict[str, str],
    *,
    min_accept_precision: float,
    max_accept_wrong: int,
) -> tuple[str, str]:
    selected = _clean(row.get("gate_selected_pfm", "")) == "1" and _clean(row.get("chosen_source", "")) == "pfm"
    if not selected:
        return "0", "gate_fallback_lightglue"
    wrong = _int_value(row, "wrong")
    precision = _float_value(row, "precision")
    if wrong > max_accept_wrong:
        return "0", "gate_selected_pfm_wrong_excess"
    if precision < min_accept_precision:
        return "0", "gate_selected_pfm_low_precision"
    return "1", "gate_selected_clean_pfm"


def _merged_row(
    *,
    source_name: str,
    pair_row: dict[str, str],
    hybrid_row: dict[str, str],
    accept_weight: float,
    reject_weight: float,
    min_accept_precision: float,
    max_accept_wrong: int,
) -> dict[str, str]:
    label, reason = _label_for_hybrid_row(
        hybrid_row,
        min_accept_precision=min_accept_precision,
        max_accept_wrong=max_accept_wrong,
    )
    correct_delta = _int_value(hybrid_row, "pfm_correct") - _int_value(hybrid_row, "lightglue_correct")
    wrong_delta = _int_value(hybrid_row, "pfm_wrong") - _int_value(hybrid_row, "lightglue_wrong")
    output = dict(pair_row)
    output.update(
        {
            "pair_accept_label": label,
            "pair_accept_weight": f"{accept_weight if label == '1' else reject_weight:.6f}",
            "pair_accept_source_wrong": str(_int_value(hybrid_row, "wrong")),
            "pair_accept_source_precision": f"{_float_value(hybrid_row, 'precision'):.6f}",
            "gate_accept_source": source_name,
            "gate_accept_selected_pfm": "1" if _clean(hybrid_row.get("gate_selected_pfm", "")) == "1" else "0",
            "gate_accept_chosen_source": _clean(hybrid_row.get("chosen_source", "")),
            "gate_accept_correct_delta_vs_lightglue": str(correct_delta),
            "gate_accept_wrong_delta_vs_lightglue": str(wrong_delta),
            "gate_accept_reason": reason,
        }
    )
    return output


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_html(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Gate acceptance training manifest</title>",
                "<h1>Gate acceptance training manifest</h1>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _reindex_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        copied = dict(row)
        copied["pair_index"] = str(index)
        output.append(copied)
    return output


def balance_acceptance_rows(
    rows: Sequence[dict[str, str]],
    *,
    target_accept_fraction: float,
) -> list[dict[str, str]]:
    if target_accept_fraction <= 0.0 or not rows:
        return list(rows)
    if target_accept_fraction >= 1.0:
        raise ValueError("target_accept_fraction must be less than 1")
    accept_rows = [dict(row) for row in rows if row.get("pair_accept_label", "") == "1"]
    reject_rows = [dict(row) for row in rows if row.get("pair_accept_label", "") != "1"]
    if not accept_rows or not reject_rows:
        return list(rows)
    current_fraction = len(accept_rows) / len(rows)
    if current_fraction >= target_accept_fraction:
        return list(rows)
    repeat = max(
        1,
        math.ceil(
            (target_accept_fraction * len(reject_rows))
            / (len(accept_rows) * (1.0 - target_accept_fraction))
        ),
    )
    accept_pool = [dict(row) for _ in range(repeat) for row in accept_rows]
    mixed: list[dict[str, str]] = []
    accept_index = 0
    accept_total = len(accept_pool)
    reject_total = len(reject_rows)
    for reject_index, row in enumerate(reject_rows, start=1):
        target_accept_seen = math.ceil((reject_index - 1) * accept_total / reject_total)
        while accept_index < target_accept_seen:
            mixed.append(dict(accept_pool[accept_index]))
            accept_index += 1
        mixed.append(dict(row))
    while accept_index < accept_total:
        mixed.append(dict(accept_pool[accept_index]))
        accept_index += 1
    return _reindex_rows(mixed)


def _summary(
    *,
    sources: Sequence[SourceSpec],
    rows: Sequence[dict[str, str]],
    source_rows: int,
    output_manifest: Path,
    accept_weight: float,
    reject_weight: float,
    min_accept_precision: float,
    max_accept_wrong: int,
    target_accept_fraction: float,
) -> dict[str, object]:
    label_counts = Counter(row.get("pair_accept_label", "") for row in rows)
    reason_counts = Counter(row.get("gate_accept_reason", "") for row in rows)
    source_counts = Counter(row.get("gate_accept_source", "") for row in rows)
    return {
        "sources": [
            {
                "name": source.name,
                "pair_manifest": str(source.pair_manifest),
                "hybrid_rows_csv": str(source.hybrid_rows_csv),
            }
            for source in sources
        ],
        "output_manifest": str(output_manifest),
        "rows": len(rows),
        "source_rows": int(source_rows),
        "balanced_rows": len(rows),
        "accept_rows": int(label_counts.get("1", 0)),
        "reject_rows": int(label_counts.get("0", 0)),
        "accept_fraction": int(label_counts.get("1", 0)) / len(rows) if rows else 0.0,
        "accept_weight": float(accept_weight),
        "reject_weight": float(reject_weight),
        "min_accept_precision": float(min_accept_precision),
        "max_accept_wrong": int(max_accept_wrong),
        "target_accept_fraction": float(target_accept_fraction),
        "reason_counts": dict(sorted(reason_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "forbidden_source_tokens": list(FORBIDDEN_SOURCE_TOKENS),
    }


def build_gate_acceptance_manifest(
    *,
    sources: Sequence[SourceSpec],
    output_manifest: Path,
    summary_json: Path,
    report_html: Path,
    accept_weight: float = 1.0,
    reject_weight: float = 3.0,
    min_accept_precision: float = 0.999,
    max_accept_wrong: int = 0,
    target_accept_fraction: float = 0.0,
    allow_forbidden_source_tokens: bool = False,
) -> dict[str, object]:
    if not sources:
        raise ValueError("at least one source is required")
    if accept_weight <= 0.0 or reject_weight <= 0.0:
        raise ValueError("accept_weight and reject_weight must be positive")
    if min_accept_precision < 0.0 or min_accept_precision > 1.0:
        raise ValueError("min_accept_precision must be in [0, 1]")
    if max_accept_wrong < 0:
        raise ValueError("max_accept_wrong must be nonnegative")
    if target_accept_fraction < 0.0 or target_accept_fraction >= 1.0:
        raise ValueError("target_accept_fraction must be in [0, 1)")

    output_fields: list[str] | None = None
    output_rows: list[dict[str, str]] = []
    seen_source_names: set[str] = set()
    for source in sources:
        if source.name in seen_source_names:
            raise ValueError(f"duplicate source name: {source.name}")
        seen_source_names.add(source.name)
        pair_fields, pair_rows = _read_csv(source.pair_manifest)
        _, hybrid_rows = _read_csv(source.hybrid_rows_csv)
        if not allow_forbidden_source_tokens:
            _assert_allowed_source(source, hybrid_rows)
        index = _hybrid_index(hybrid_rows)
        for pair_row in pair_rows:
            key = _pair_key(pair_row)
            hybrid_row = index.get(key)
            if hybrid_row is None:
                raise ValueError(f"missing hybrid row for source={source.name} key={key}")
            output_rows.append(
                _merged_row(
                    source_name=source.name,
                    pair_row=pair_row,
                    hybrid_row=hybrid_row,
                    accept_weight=accept_weight,
                    reject_weight=reject_weight,
                    min_accept_precision=min_accept_precision,
                    max_accept_wrong=max_accept_wrong,
                )
            )
        if output_fields is None:
            output_fields = list(pair_fields)
            for field in [*ACCEPTANCE_FIELDS, *GATE_ACCEPTANCE_FIELDS]:
                if field not in output_fields:
                    output_fields.append(field)

    source_rows = len(output_rows)
    output_rows = balance_acceptance_rows(output_rows, target_accept_fraction=target_accept_fraction)
    summary = _summary(
        sources=sources,
        rows=output_rows,
        source_rows=source_rows,
        output_manifest=output_manifest,
        accept_weight=accept_weight,
        reject_weight=reject_weight,
        min_accept_precision=min_accept_precision,
        max_accept_wrong=max_accept_wrong,
        target_accept_fraction=target_accept_fraction,
    )
    _write_csv(output_manifest, output_fields or [*ACCEPTANCE_FIELDS, *GATE_ACCEPTANCE_FIELDS], output_rows)
    _write_json(summary_json, summary)
    _write_html(report_html, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--report-html", type=Path, required=True)
    parser.add_argument("--accept-weight", type=float, default=1.0)
    parser.add_argument("--reject-weight", type=float, default=3.0)
    parser.add_argument("--min-accept-precision", type=float, default=0.999)
    parser.add_argument("--max-accept-wrong", type=int, default=0)
    parser.add_argument("--target-accept-fraction", type=float, default=0.0)
    parser.add_argument("--allow-forbidden-source-tokens", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_gate_acceptance_manifest(
        sources=args.source,
        output_manifest=args.output_manifest,
        summary_json=args.summary_json,
        report_html=args.report_html,
        accept_weight=float(args.accept_weight),
        reject_weight=float(args.reject_weight),
        min_accept_precision=float(args.min_accept_precision),
        max_accept_wrong=int(args.max_accept_wrong),
        target_accept_fraction=float(args.target_accept_fraction),
        allow_forbidden_source_tokens=bool(args.allow_forbidden_source_tokens),
    )
    print(
        "gate_acceptance_manifest "
        f"rows={summary['rows']} "
        f"accept_rows={summary['accept_rows']} "
        f"reject_rows={summary['reject_rows']} "
        f"output={args.output_manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
