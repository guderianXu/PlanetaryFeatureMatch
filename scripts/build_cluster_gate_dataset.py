#!/usr/bin/env python3
"""Build pair-level cluster-gate datasets from PFM failure buckets."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Sequence


OUTPUT_FIELDS = [
    "source_name",
    "split",
    "pair_index",
    "pair_type",
    "base_id",
    "reference_variant",
    "target_variant",
    "pfm_matches",
    "pfm_correct",
    "pfm_wrong",
    "pfm_precision",
    "lightglue_matches",
    "lightglue_correct",
    "lightglue_wrong",
    "lightglue_precision",
    "reject_label",
    "reject_reasons",
    "keep_label",
    "feature_valid_fraction",
    "feature_matches",
    "feature_score_min",
    "feature_score_mean",
    "feature_score_median",
    "feature_score_max",
    "feature_displacement_mad_px",
    "feature_homography_residual_valid",
    "feature_homography_residual_median_px",
    "feature_homography_residual_p90_px",
    "feature_target_is_extreme",
    "diagnostic_wrong",
    "diagnostic_wrong_ratio",
    "diagnostic_precision_gap",
    "diagnostic_false_cluster",
    "diagnostic_high_confidence_wrong",
    "diagnostic_near_miss_wrong",
    "diagnostic_far_wrong",
    "diagnostic_wrong_displacement_mad_px",
]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _float_value(row: dict[str, str] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_value(row: dict[str, str] | None, key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _fmt_int(value: int) -> str:
    return str(int(value))


def _summary_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("split", ""), row.get("base_id", ""), row.get("target_variant", ""))


def _lightglue_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("manifest_pair_index") or row.get("pair_index", ""),
        row.get("split", ""),
        row.get("base_id", ""),
        row.get("target_variant", ""),
    )


def _pfm_summary_index(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        indexed[_summary_key(row)] = row
    return indexed


def _lightglue_index(
    rows: Sequence[dict[str, str]],
    *,
    teacher_label: str,
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        if teacher_label and row.get("label", "") != teacher_label:
            continue
        indexed[_lightglue_key(row)] = row
    return indexed


def _select_lightglue(
    indexed: dict[tuple[str, str, str, str], dict[str, str]],
    pair: dict[str, str],
) -> dict[str, str] | None:
    keys = [
        (
            pair.get("pair_index", ""),
            pair.get("split", ""),
            pair.get("base_id", ""),
            pair.get("target_variant", ""),
        ),
        (
            pair.get("pair_index", ""),
            "",
            pair.get("base_id", ""),
            pair.get("target_variant", ""),
        ),
    ]
    for key in keys:
        if key in indexed:
            return indexed[key]
    return None


def _reject_reasons(pair: dict[str, str], *, reject_wrong_threshold: int) -> list[str]:
    reasons: list[str] = []
    if _int_value(pair, "false_cluster") > 0:
        reasons.append("false_cluster")
    if _int_value(pair, "high_confidence_wrong") > 0:
        reasons.append("high_confidence_wrong")
    if _int_value(pair, "wrong") >= reject_wrong_threshold:
        reasons.append("pfm_wrong")
    return reasons


def _target_is_extreme(target_variant: str) -> int:
    return 1 if target_variant.startswith("extreme_") else 0


def build_cluster_gate_rows(
    pair_failure_rows: Sequence[dict[str, str]],
    pfm_summary_rows: Sequence[dict[str, str]],
    lightglue_rows: Sequence[dict[str, str]] = (),
    *,
    source_name: str,
    split: str,
    teacher_label: str = "LightGlue-SIFT-MAGSAC-min16",
    reject_wrong_threshold: int = 8,
    keep_max_wrong: int = 1,
    keep_min_precision: float = 0.995,
) -> list[dict[str, str]]:
    pfm_by_key = _pfm_summary_index(pfm_summary_rows)
    lightglue_by_key = _lightglue_index(lightglue_rows, teacher_label=teacher_label)
    rows: list[dict[str, str]] = []
    for pair in pair_failure_rows:
        pfm = pfm_by_key.get(_summary_key(pair), {})
        lightglue = _select_lightglue(lightglue_by_key, pair)

        pfm_matches = _int_value(pair, "matches")
        pfm_correct = _int_value(pair, "correct")
        pfm_wrong = _int_value(pair, "wrong")
        pfm_precision = _float_value(pair, "precision", pfm_correct / pfm_matches if pfm_matches else 0.0)
        reasons = _reject_reasons(pair, reject_wrong_threshold=reject_wrong_threshold)
        keep_label = (
            not reasons
            and pfm_wrong <= keep_max_wrong
            and pfm_precision >= keep_min_precision
        )
        lightglue_matches = _int_value(lightglue, "matches")
        lightglue_correct = _int_value(lightglue, "correct")
        lightglue_wrong = _int_value(lightglue, "wrong")
        lightglue_precision = _float_value(
            lightglue,
            "precision",
            lightglue_correct / lightglue_matches if lightglue_matches else 0.0,
        )
        target_variant = pair.get("target_variant", "")
        wrong_ratio = pfm_wrong / pfm_matches if pfm_matches else 0.0
        row = {
            "source_name": source_name,
            "split": split or pair.get("split", ""),
            "pair_index": pair.get("pair_index", ""),
            "pair_type": pair.get("pair_type", "same_position_view"),
            "base_id": pair.get("base_id", ""),
            "reference_variant": pair.get("reference_variant", ""),
            "target_variant": target_variant,
            "pfm_matches": _fmt_int(pfm_matches),
            "pfm_correct": _fmt_int(pfm_correct),
            "pfm_wrong": _fmt_int(pfm_wrong),
            "pfm_precision": _fmt_float(pfm_precision),
            "lightglue_matches": _fmt_int(lightglue_matches),
            "lightglue_correct": _fmt_int(lightglue_correct),
            "lightglue_wrong": _fmt_int(lightglue_wrong),
            "lightglue_precision": _fmt_float(lightglue_precision),
            "reject_label": "1" if reasons else "0",
            "reject_reasons": ";".join(reasons),
            "keep_label": "1" if keep_label else "0",
            "feature_valid_fraction": _fmt_float(_float_value(pfm, "valid_fraction", _float_value(pair, "valid_fraction"))),
            "feature_matches": _fmt_int(pfm_matches),
            "feature_score_min": _fmt_float(_float_value(pfm, "score_min")),
            "feature_score_mean": _fmt_float(_float_value(pfm, "score_mean")),
            "feature_score_median": _fmt_float(_float_value(pfm, "score_median")),
            "feature_score_max": _fmt_float(_float_value(pfm, "score_max")),
            "feature_displacement_mad_px": _fmt_float(_float_value(pfm, "displacement_mad_px")),
            "feature_homography_residual_valid": _fmt_int(_int_value(pfm, "homography_residual_valid")),
            "feature_homography_residual_median_px": _fmt_float(_float_value(pfm, "homography_residual_median_px")),
            "feature_homography_residual_p90_px": _fmt_float(_float_value(pfm, "homography_residual_p90_px")),
            "feature_target_is_extreme": _fmt_int(_target_is_extreme(target_variant)),
            "diagnostic_wrong": _fmt_int(pfm_wrong),
            "diagnostic_wrong_ratio": _fmt_float(wrong_ratio),
            "diagnostic_precision_gap": _fmt_float(max(0.0, 1.0 - pfm_precision)),
            "diagnostic_false_cluster": _fmt_int(_int_value(pair, "false_cluster")),
            "diagnostic_high_confidence_wrong": _fmt_int(_int_value(pair, "high_confidence_wrong")),
            "diagnostic_near_miss_wrong": _fmt_int(_int_value(pair, "near_miss_wrong")),
            "diagnostic_far_wrong": _fmt_int(_int_value(pair, "far_wrong")),
            "diagnostic_wrong_displacement_mad_px": _fmt_float(_float_value(pair, "wrong_displacement_mad_px")),
        }
        rows.append(row)
    return rows


def _summarize(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    variants = Counter(row.get("target_variant", "") for row in rows)
    reject_rows = [row for row in rows if row.get("reject_label") == "1"]
    return {
        "rows": len(rows),
        "reject_rows": len(reject_rows),
        "keep_rows": sum(1 for row in rows if row.get("keep_label") == "1"),
        "false_cluster_rows": sum(_int_value(row, "diagnostic_false_cluster") for row in rows),
        "high_confidence_wrong_rows": sum(1 for row in rows if _int_value(row, "diagnostic_high_confidence_wrong") > 0),
        "pfm_matches": sum(_int_value(row, "pfm_matches") for row in rows),
        "pfm_correct": sum(_int_value(row, "pfm_correct") for row in rows),
        "pfm_wrong": sum(_int_value(row, "pfm_wrong") for row in rows),
        "lightglue_correct": sum(_int_value(row, "lightglue_correct") for row in rows),
        "lightglue_wrong": sum(_int_value(row, "lightglue_wrong") for row in rows),
        "target_variant_counts": dict(sorted(variants.items())),
    }


def _write_html(
    path: Path,
    *,
    output_csv: Path,
    pair_failure_summary: Path,
    pfm_summary: Path,
    lightglue_metrics: Path | None,
    summary: dict[str, object],
) -> None:
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Cluster gate dataset</title>",
                "<h1>Cluster gate dataset</h1>",
                f"<p>output_csv=<code>{html.escape(str(output_csv))}</code></p>",
                f"<p>pair_failure_summary=<code>{html.escape(str(pair_failure_summary))}</code></p>",
                f"<p>pfm_summary=<code>{html.escape(str(pfm_summary))}</code></p>",
                f"<p>lightglue_metrics=<code>{html.escape(str(lightglue_metrics or ''))}</code></p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>metric</th><th>value</th></tr>",
                rows,
                "</table>",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-failure-summary", type=Path, required=True)
    parser.add_argument("--pfm-summary", type=Path, required=True)
    parser.add_argument("--lightglue-metrics", type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--teacher-label", default="LightGlue-SIFT-MAGSAC-min16")
    parser.add_argument("--reject-wrong-threshold", type=int, default=8)
    parser.add_argument("--keep-max-wrong", type=int, default=1)
    parser.add_argument("--keep-min-precision", type=float, default=0.995)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lightglue_rows = _read_csv_rows(args.lightglue_metrics) if args.lightglue_metrics else []
    rows = build_cluster_gate_rows(
        _read_csv_rows(args.pair_failure_summary),
        _read_csv_rows(args.pfm_summary),
        lightglue_rows,
        source_name=str(args.source_name),
        split=str(args.split),
        teacher_label=str(args.teacher_label),
        reject_wrong_threshold=int(args.reject_wrong_threshold),
        keep_max_wrong=int(args.keep_max_wrong),
        keep_min_precision=float(args.keep_min_precision),
    )
    _write_csv(args.output_csv, rows)
    summary = _summarize(rows)
    summary.update(
        {
            "pair_failure_summary": str(args.pair_failure_summary),
            "pfm_summary": str(args.pfm_summary),
            "lightglue_metrics": str(args.lightglue_metrics or ""),
            "output_csv": str(args.output_csv),
            "source_name": str(args.source_name),
            "split": str(args.split),
            "reject_wrong_threshold": int(args.reject_wrong_threshold),
            "keep_max_wrong": int(args.keep_max_wrong),
            "keep_min_precision": float(args.keep_min_precision),
        }
    )
    _write_json(args.summary_json, summary)
    _write_html(
        args.output_html,
        output_csv=args.output_csv,
        pair_failure_summary=args.pair_failure_summary,
        pfm_summary=args.pfm_summary,
        lightglue_metrics=args.lightglue_metrics,
        summary=summary,
    )
    print(
        f"cluster_gate_rows={summary['rows']} reject_rows={summary['reject_rows']} output={args.output_csv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
