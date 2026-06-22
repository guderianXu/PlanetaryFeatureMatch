#!/usr/bin/env python3
"""Apply an observable pair gate, then filter selected PFM matches by homography residual."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from apply_observable_pair_gate import (
    IDENTITY_FIELDS,
    compile_gate,
    read_csv_rows,
    summarize_rows,
)


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


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return int(round(float(value)))
    except ValueError:
        return default


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def parse_variant_thresholds(value: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected variant=threshold, got: {item}")
        variant, raw_threshold = item.split("=", 1)
        variant = variant.strip()
        if not variant:
            raise argparse.ArgumentTypeError("variant name must not be empty")
        try:
            threshold = float(raw_threshold.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid threshold for {variant}: {raw_threshold}") from exc
        if threshold < 0.0:
            raise argparse.ArgumentTypeError("variant threshold must be nonnegative")
        thresholds[variant] = threshold
    return thresholds


def _detail_key(row_offset: int, detail: dict[str, str]) -> tuple[int, str, str]:
    return (
        row_offset + _int_value(detail, "pair_index"),
        detail.get("base_id", ""),
        detail.get("target_variant", ""),
    )


def _local_detail_key(row: dict[str, str]) -> tuple[int, str, str]:
    return (
        _int_value(row, "pair_index"),
        row.get("base_id", ""),
        row.get("target_variant", ""),
    )


def _base_variant_detail_key(row: dict[str, str]) -> tuple[int, str, str]:
    return (
        -1,
        row.get("base_id", ""),
        row.get("target_variant", ""),
    )


def _append_detail_once(
    groups: dict[tuple[int, str, str], list[dict[str, str]]],
    key: tuple[int, str, str],
    row: dict[str, str],
) -> None:
    bucket = groups.setdefault(key, [])
    if not bucket or bucket[-1] is not row:
        bucket.append(row)


def load_match_detail_groups(paths: Sequence[Path]) -> dict[tuple[int, str, str], list[dict[str, str]]]:
    groups: dict[tuple[int, str, str], list[dict[str, str]]] = {}
    row_offset = 0
    for path in paths:
        rows = read_csv_rows(path)
        max_pair_index = -1
        for row in rows:
            pair_index = _int_value(row, "pair_index")
            max_pair_index = max(max_pair_index, pair_index)
            for key in {_detail_key(row_offset, row), _local_detail_key(row), _base_variant_detail_key(row)}:
                _append_detail_once(groups, key, row)
        row_offset += max_pair_index + 1 if max_pair_index >= 0 else 0
    return groups


def _lookup_detail_rows(
    detail_groups: dict[tuple[int, str, str], list[dict[str, str]]],
    *,
    row_index: int,
    row: dict[str, str],
) -> list[dict[str, str]]:
    global_key = (row_index, row.get("base_id", ""), row.get("target_variant", ""))
    details = detail_groups.get(global_key, [])
    if details:
        return details
    details = detail_groups.get(_local_detail_key(row), [])
    if details:
        return details
    return detail_groups.get(_base_variant_detail_key(row), [])


def _homography_residual_keep_mask(
    details: Sequence[dict[str, str]],
    *,
    max_homography_residual_px: float | None,
) -> list[bool]:
    if max_homography_residual_px is None:
        return [True] * len(details)
    if len(details) < 4:
        return [False] * len(details)
    points_a = np.array(
        [[_float_value(row, "point_a_x_px"), _float_value(row, "point_a_y_px")] for row in details],
        dtype=np.float32,
    )
    points_b = np.array(
        [[_float_value(row, "point_b_x_px"), _float_value(row, "point_b_y_px")] for row in details],
        dtype=np.float32,
    )
    homography, _mask = cv2.findHomography(
        points_a,
        points_b,
        cv2.USAC_MAGSAC,
        5.0,
        maxIters=10000,
        confidence=0.999,
    )
    if homography is None:
        return [False] * len(details)
    homogeneous = np.concatenate([points_a, np.ones((points_a.shape[0], 1), dtype=np.float32)], axis=1)
    projected = (homography @ homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    residual = np.linalg.norm(projected - points_b, axis=1)
    return [bool(value <= max_homography_residual_px) for value in residual]


def _filtered_pfm_metrics(
    details: Sequence[dict[str, str]],
    *,
    max_homography_residual_px: float | None,
) -> tuple[int, int, int, float]:
    keep_mask = _homography_residual_keep_mask(
        details,
        max_homography_residual_px=max_homography_residual_px,
    )
    kept = [row for row, keep in zip(details, keep_mask) if keep]
    matches = len(kept)
    correct = sum(1 for row in kept if row.get("correct", "") == "1")
    wrong = matches - correct
    precision = correct / matches if matches else 0.0
    return matches, correct, wrong, precision


def build_hybrid_rows(
    dataset_rows: Sequence[dict[str, str]],
    detail_groups: dict[tuple[int, str, str], list[dict[str, str]]],
    *,
    gate: str,
    max_homography_residual_px: float | None,
    variant_homography_residual_px: dict[str, float] | None = None,
) -> list[dict[str, str]]:
    gate_fn = compile_gate(gate)
    variant_thresholds = variant_homography_residual_px or {}
    output_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(dataset_rows):
        selected_pfm = bool(gate_fn(row))
        chosen_source = "pfm" if selected_pfm else "lightglue"
        if selected_pfm:
            key = (row_index, row.get("base_id", ""), row.get("target_variant", ""))
            details = _lookup_detail_rows(detail_groups, row_index=row_index, row=row)
            if not details:
                if _int_value(row, "pfm_matches") == 0:
                    matches, correct, wrong, precision = 0, 0, 0, 0.0
                else:
                    raise ValueError(f"missing match details for selected row {key}")
            else:
                threshold_px = variant_thresholds.get(
                    row.get("target_variant", ""),
                    max_homography_residual_px,
                )
                matches, correct, wrong, precision = _filtered_pfm_metrics(
                    details,
                    max_homography_residual_px=threshold_px,
                )
        else:
            matches = _int_value(row, "lightglue_matches")
            correct = _int_value(row, "lightglue_correct")
            wrong = _int_value(row, "lightglue_wrong")
            precision = correct / matches if matches else 0.0
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


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, *, dataset_csv: Path, gate: str, summary: dict[str, object]) -> None:
    metric_keys = [
        "rows",
        "kept_pfm_rows",
        "fallback_lightglue_rows",
        "lightglue_correct",
        "lightglue_wrong",
        "hybrid_correct",
        "hybrid_wrong",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "max_homography_residual_px",
        "variant_homography_residual_px",
    ]
    rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(summary.get(key, '')))}</td></tr>"
        for key in metric_keys
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Observable Pair Gate Match Filter</title>",
                "<h1>Observable Pair Gate Match Filter</h1>",
                f"<p>dataset_csv={html.escape(str(dataset_csv))}</p>",
                f"<p>gate={html.escape(gate)}</p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                rows,
                "</table>",
                "<h2>Summary JSON</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def apply_gate_match_filter(
    *,
    dataset_csv: Path,
    match_details: Sequence[Path],
    gate: str,
    output_dir: Path,
    max_homography_residual_px: float | None,
    variant_homography_residual_px: dict[str, float] | None = None,
) -> dict[str, object]:
    rows = read_csv_rows(dataset_csv)
    if not rows:
        raise ValueError(f"dataset is empty: {dataset_csv}")
    if not match_details:
        raise ValueError("at least one match-details CSV is required")
    detail_groups = load_match_detail_groups(match_details)
    hybrid_rows = build_hybrid_rows(
        rows,
        detail_groups,
        gate=gate,
        max_homography_residual_px=max_homography_residual_px,
        variant_homography_residual_px=variant_homography_residual_px,
    )
    summary = summarize_rows(hybrid_rows)
    summary["dataset_csv"] = str(dataset_csv)
    summary["match_details"] = [str(path) for path in match_details]
    summary["gate"] = gate
    summary["max_homography_residual_px"] = max_homography_residual_px
    summary["variant_homography_residual_px"] = variant_homography_residual_px or {}

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "hybrid_rows.csv", hybrid_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_html(output_dir / "index.html", dataset_csv=dataset_csv, gate=gate, summary=summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--match-details", type=Path, action="append", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-homography-residual-px", type=float, default=None)
    parser.add_argument("--variant-homography-residual-px", type=parse_variant_thresholds, action="append", default=None)
    args = parser.parse_args(argv)
    variant_thresholds: dict[str, float] = {}
    for thresholds in args.variant_homography_residual_px or []:
        variant_thresholds.update(thresholds)
    args.variant_homography_residual_px = variant_thresholds
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = apply_gate_match_filter(
        dataset_csv=args.dataset_csv,
        match_details=args.match_details,
        gate=args.gate,
        output_dir=args.output_dir,
        max_homography_residual_px=args.max_homography_residual_px,
        variant_homography_residual_px=dict(args.variant_homography_residual_px),
    )
    print(
        "observable_pair_gate_match_filter "
        f"kept_pfm={summary['kept_pfm_rows']} "
        f"correct_delta_vs_lightglue={summary['correct_delta_vs_lightglue']} "
        f"wrong_delta_vs_lightglue={summary['wrong_delta_vs_lightglue']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
