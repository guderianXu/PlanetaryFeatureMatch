#!/usr/bin/env python3
"""Build hard pair manifests from adaptive-rescue gain reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path

from mine_hard_failure_pairs import PAIR_MANIFEST_FIELDS


RESCUE_GAIN_EXTRA_FIELDS = [
    "hard_reasons",
    "hard_score",
    "source_pair_index",
    "source_ordinal",
    "source_baseline_matches",
    "source_baseline_correct",
    "source_baseline_wrong",
    "source_baseline_precision",
    "source_candidate_matches",
    "source_candidate_correct",
    "source_candidate_wrong",
    "source_candidate_precision",
    "source_candidate_homography_p90_px",
    "source_candidate_homography_median_px",
    "source_candidate_score_mean",
    "delta_matches",
    "delta_correct",
    "delta_wrong",
    "match_delta",
    "correct_delta",
    "wrong_delta",
    "precision_delta",
]


@dataclass(frozen=True)
class RescueGainSource:
    split: str
    pair_manifest: Path
    baseline_summary: Path
    candidate_summary: Path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _format_float(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    if value == "":
        return ""
    return f"{_float_value(row, key):.6f}"


def _rescue_reasons(
    *,
    base_matches: int,
    delta_matches: int,
    delta_correct: int,
    delta_wrong: int,
) -> list[str]:
    reasons: list[str] = []
    if delta_correct > 0:
        reasons.append("rescue_correct_gain")
    if delta_matches > 0:
        reasons.append("rescue_match_gain")
    if base_matches <= 0 and delta_correct > 0:
        reasons.append("rescue_false_negative")
    if delta_wrong > 0:
        reasons.append("rescue_added_wrong")
    return reasons


def _hard_score(*, delta_matches: int, delta_correct: int, delta_wrong: int, base_matches: int) -> float:
    score = float(max(0, delta_correct)) * 10.0 + float(max(0, delta_matches))
    if base_matches <= 0 and delta_correct > 0:
        score += 20.0
    score -= float(max(0, delta_wrong)) * 2.0
    return score


def mine_rescue_gain_rows(
    pair_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    split: str,
    target_variants: tuple[str, ...] = ("extreme_02", "extreme_03"),
    min_correct_gain: int = 1,
) -> list[dict[str, str]]:
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("baseline and candidate summaries must have the same row count")
    if len(baseline_rows) > len(pair_rows):
        raise ValueError("pair manifest must contain at least as many rows as the summaries")

    target_set = set(target_variants)
    mined: list[dict[str, str]] = []
    for ordinal, (pair, baseline, candidate) in enumerate(zip(pair_rows, baseline_rows, candidate_rows)):
        target_variant = pair.get("target_variant", "")
        if target_variant not in target_set:
            continue
        base_matches = _int_value(baseline, "matches")
        base_correct = _int_value(baseline, "correct")
        base_wrong = _int_value(baseline, "wrong")
        base_precision = _float_value(baseline, "precision")
        candidate_matches = _int_value(candidate, "matches")
        candidate_correct = _int_value(candidate, "correct")
        candidate_wrong = _int_value(candidate, "wrong")
        candidate_precision = _float_value(candidate, "precision")
        delta_matches = candidate_matches - base_matches
        delta_correct = candidate_correct - base_correct
        delta_wrong = candidate_wrong - base_wrong
        precision_delta = candidate_precision - base_precision
        if delta_correct < min_correct_gain:
            continue

        reasons = _rescue_reasons(
            base_matches=base_matches,
            delta_matches=delta_matches,
            delta_correct=delta_correct,
            delta_wrong=delta_wrong,
        )
        row = {field: pair.get(field, "") for field in PAIR_MANIFEST_FIELDS}
        row.update(
            {
                "split": split,
                "hard_reasons": "|".join(reasons),
                "hard_score": f"{_hard_score(delta_matches=delta_matches, delta_correct=delta_correct, delta_wrong=delta_wrong, base_matches=base_matches):.6f}",
                "source_pair_index": pair.get("pair_index", ""),
                "source_ordinal": str(ordinal),
                "source_baseline_matches": str(base_matches),
                "source_baseline_correct": str(base_correct),
                "source_baseline_wrong": str(base_wrong),
                "source_baseline_precision": f"{base_precision:.6f}",
                "source_candidate_matches": str(candidate_matches),
                "source_candidate_correct": str(candidate_correct),
                "source_candidate_wrong": str(candidate_wrong),
                "source_candidate_precision": f"{candidate_precision:.6f}",
                "source_candidate_homography_p90_px": candidate.get("homography_residual_p90_px", ""),
                "source_candidate_homography_median_px": candidate.get("homography_residual_median_px", ""),
                "source_candidate_score_mean": candidate.get("score_mean", ""),
                "delta_matches": str(delta_matches),
                "delta_correct": str(delta_correct),
                "delta_wrong": str(delta_wrong),
                "match_delta": str(delta_matches),
                "correct_delta": str(delta_correct),
                "wrong_delta": str(delta_wrong),
                "precision_delta": f"{precision_delta:.6f}",
            }
        )
        mined.append(row)

    mined.sort(
        key=lambda row: (
            row.get("split", ""),
            int(row.get("source_ordinal") or 0),
        )
    )
    for index, row in enumerate(mined):
        row["pair_index"] = str(index)
    return mined


def build_rescue_gain_rows(
    sources: list[RescueGainSource],
    *,
    target_variants: tuple[str, ...],
    min_correct_gain: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in sources:
        rows.extend(
            mine_rescue_gain_rows(
                _read_csv_rows(source.pair_manifest),
                _read_csv_rows(source.baseline_summary),
                _read_csv_rows(source.candidate_summary),
                split=source.split,
                target_variants=target_variants,
                min_correct_gain=min_correct_gain,
            )
        )
    for index, row in enumerate(rows):
        row["pair_index"] = str(index)
    return rows


def write_rescue_gain_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*PAIR_MANIFEST_FIELDS, *RESCUE_GAIN_EXTRA_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rescue_gain_rows(rows: list[dict[str, str]], *, source_count: int) -> dict[str, object]:
    summary: dict[str, object] = {
        "source_count": source_count,
        "rows": len(rows),
        "delta_matches": sum(_int_value(row, "delta_matches") for row in rows),
        "delta_correct": sum(_int_value(row, "delta_correct") for row in rows),
        "delta_wrong": sum(_int_value(row, "delta_wrong") for row in rows),
        "by_split": {},
        "by_variant": {},
    }
    for key, bucket_name in (("split", "by_split"), ("target_variant", "by_variant")):
        bucket = summary[bucket_name]
        assert isinstance(bucket, dict)
        for row in rows:
            name = row.get(key, "")
            item = bucket.setdefault(name, {"rows": 0, "delta_matches": 0, "delta_correct": 0, "delta_wrong": 0})
            item["rows"] += 1
            item["delta_matches"] += _int_value(row, "delta_matches")
            item["delta_correct"] += _int_value(row, "delta_correct")
            item["delta_wrong"] += _int_value(row, "delta_wrong")
    return summary


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_index_html(path: Path, *, output_csv: Path, summary_json: Path, rows: list[dict[str, str]], summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split",
        "source_pair_index",
        "reference_pose_id",
        "target_pose_id",
        "target_variant",
        "delta_correct",
        "delta_wrong",
        "source_baseline_matches",
        "source_candidate_matches",
    ]
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>")
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-CN">',
                '<head><meta charset="utf-8"><title>rescue gain hard set</title></head>',
                "<body>",
                "<h1>rescue gain hard set</h1>",
                f"<p>Rows: <code>{summary['rows']}</code>; delta_correct: <code>{summary['delta_correct']}</code>; delta_wrong: <code>{summary['delta_wrong']}</code>.</p>",
                "<h2>Summary</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
                "<h2>Pairs</h2>",
                f'<table border="1" cellspacing="0" cellpadding="4"><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table>',
                "<ul>",
                f"<li>CSV: <code>{html.escape(str(output_csv))}</code></li>",
                f"<li>JSON: <code>{html.escape(str(summary_json))}</code></li>",
                "</ul>",
                "</body>",
                "</html>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parse_source(value: str) -> RescueGainSource:
    parts = [item.strip() for item in value.split(",", 3)]
    if len(parts) != 4 or not all(parts):
        raise argparse.ArgumentTypeError("source must be split,pair_manifest,baseline_summary,candidate_summary")
    return RescueGainSource(
        split=parts[0],
        pair_manifest=Path(parts[1]),
        baseline_summary=Path(parts[2]),
        candidate_summary=Path(parts[3]),
    )


def _parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    if not variants:
        raise argparse.ArgumentTypeError("at least one target variant is required")
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=_parse_source,
        required=True,
        help="Repeatable: split,pair_manifest,baseline_all_filtered_summary,candidate_all_filtered_summary.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--target-variants", type=_parse_variants, default=("extreme_02", "extreme_03"))
    parser.add_argument("--min-correct-gain", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rescue_gain_rows(
        args.source,
        target_variants=args.target_variants,
        min_correct_gain=args.min_correct_gain,
    )
    summary = summarize_rescue_gain_rows(rows, source_count=len(args.source))
    write_rescue_gain_csv(args.output_csv, rows)
    write_summary_json(args.summary_json, summary)
    write_index_html(args.output_html, output_csv=args.output_csv, summary_json=args.summary_json, rows=rows, summary=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
