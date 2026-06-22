#!/usr/bin/env python3
"""Sweep pair-acceptance thresholds for PFM/LightGlue hybrid evaluation."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Sequence


DEFAULT_LIGHTGLUE_LABEL = "LightGlue-SIFT-MAGSAC-min16"


def _clean(value: str | None) -> str:
    return "" if value is None else value.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        return [{key: ("" if value is None else value) for key, value in row.items() if key is not None} for row in reader]


def _pair_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        _clean(row.get("split", "")),
        _clean(row.get("base_id", "")),
        _clean(row.get("target_variant", "")),
    )


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = _clean(row.get(key, ""))
    if value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid float for {key}: {value!r}") from exc


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _safe_threshold_name(threshold: float) -> str:
    return f"{threshold:.6g}".replace("-", "neg_").replace(".", "_")


def _format_threshold(threshold: float) -> str:
    return f"{threshold:.6f}"


def _index_lightglue(rows: Sequence[dict[str, str]], label: str) -> dict[tuple[str, str, str], dict[str, str]]:
    index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if _clean(row.get("label", "")) != label:
            continue
        key = _pair_key(row)
        if key in index:
            raise ValueError(f"duplicate LightGlue row for key {key}")
        index[key] = row
    if not index:
        raise ValueError(f"no LightGlue rows found for label {label!r}")
    return index


def _summarize_rows(rows: Sequence[dict[str, str]], prefix: str = "") -> dict[str, int | float]:
    matches = sum(_int_value(row, f"{prefix}matches" if prefix else "matches") for row in rows)
    correct = sum(_int_value(row, f"{prefix}correct" if prefix else "correct") for row in rows)
    wrong = sum(_int_value(row, f"{prefix}wrong" if prefix else "wrong") for row in rows)
    return {
        "matches": matches,
        "correct": correct,
        "wrong": wrong,
        "precision": correct / matches if matches else 0.0,
    }


def hybrid_rows_for_threshold(
    pfm_rows: Sequence[dict[str, str]],
    lightglue_index: dict[tuple[str, str, str], dict[str, str]],
    threshold: float,
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for pfm_row in pfm_rows:
        key = _pair_key(pfm_row)
        lightglue_row = lightglue_index.get(key)
        if lightglue_row is None:
            raise ValueError(f"missing LightGlue row for key {key}")
        probability = _float_value(pfm_row, "pair_accept_probability", float("nan"))
        if not math.isfinite(probability):
            raise ValueError(f"missing pair_accept_probability for key {key}")
        use_pfm = probability >= threshold
        source_row = pfm_row if use_pfm else lightglue_row
        source = "pfm" if use_pfm else "lightglue"
        matches = _int_value(source_row, "matches")
        correct = _int_value(source_row, "correct")
        wrong = _int_value(source_row, "wrong")
        output_rows.append(
            {
                "threshold": _format_threshold(threshold),
                "split": key[0],
                "base_id": key[1],
                "target_variant": key[2],
                "source": source,
                "pair_accept_probability": f"{probability:.6f}",
                "pfm_matches": str(_int_value(pfm_row, "matches")),
                "pfm_correct": str(_int_value(pfm_row, "correct")),
                "pfm_wrong": str(_int_value(pfm_row, "wrong")),
                "lightglue_matches": str(_int_value(lightglue_row, "matches")),
                "lightglue_correct": str(_int_value(lightglue_row, "correct")),
                "lightglue_wrong": str(_int_value(lightglue_row, "wrong")),
                "hybrid_matches": str(matches),
                "hybrid_correct": str(correct),
                "hybrid_wrong": str(wrong),
                "hybrid_precision": f"{correct / matches if matches else 0.0:.6f}",
            }
        )
    return output_rows


def summarize_threshold(
    split_label: str,
    threshold: float,
    rows: Sequence[dict[str, str]],
    lightglue: dict[str, int | float],
    pfm_raw: dict[str, int | float],
) -> dict[str, object]:
    matches = sum(_int_value(row, "hybrid_matches") for row in rows)
    correct = sum(_int_value(row, "hybrid_correct") for row in rows)
    wrong = sum(_int_value(row, "hybrid_wrong") for row in rows)
    pfm_pairs = sum(1 for row in rows if row["source"] == "pfm")
    return {
        "split": split_label,
        "threshold": _format_threshold(threshold),
        "pairs": len(rows),
        "pfm_pairs": pfm_pairs,
        "lightglue_pairs": len(rows) - pfm_pairs,
        "matches": matches,
        "correct": correct,
        "wrong": wrong,
        "precision": correct / matches if matches else 0.0,
        "delta_correct_vs_lightglue": correct - int(lightglue["correct"]),
        "delta_wrong_vs_lightglue": wrong - int(lightglue["wrong"]),
        "delta_correct_vs_pfm_raw": correct - int(pfm_raw["correct"]),
        "delta_wrong_vs_pfm_raw": wrong - int(pfm_raw["wrong"]),
    }


def choose_best_threshold(rows: Sequence[dict[str, object]], lightglue_wrong: int) -> dict[str, object]:
    wrong_safe = [row for row in rows if int(row["wrong"]) <= lightglue_wrong]
    candidates = wrong_safe or list(rows)
    return max(candidates, key=lambda row: (int(row["correct"]), -int(row["wrong"]), int(row["pfm_pairs"])))


def write_csv(path: Path, rows: Sequence[dict[str, object | str]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary: dict[str, object], threshold_rows: Sequence[dict[str, object]]) -> None:
    fields = list(threshold_rows[0].keys()) if threshold_rows else []
    table_rows = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[field]))}</td>" for field in fields) + "</tr>"
        for row in threshold_rows
    )
    path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(str(summary['split']))} pair acceptance hybrid</title></head><body>"
        f"<h1>{html.escape(str(summary['split']))} pair acceptance hybrid</h1>"
        f"<p>Best threshold: <code>{html.escape(str(summary['best_threshold']['threshold']))}</code></p>"
        "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\"><tr>"
        + "".join(f"<th>{html.escape(field)}</th>" for field in fields)
        + f"</tr>{table_rows}</table></body></html>",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PFM/LightGlue hybrid threshold summaries.")
    parser.add_argument("--pfm-summary", type=Path, required=True)
    parser.add_argument("--lightglue-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-label", required=True)
    parser.add_argument("--lightglue-label", default=DEFAULT_LIGHTGLUE_LABEL)
    parser.add_argument("--threshold", type=float, action="append", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = sorted(set(float(value) for value in args.threshold))
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
            raise ValueError(f"threshold must be finite and in [-1, 1], got {threshold!r}")

    pfm_rows = _read_csv(args.pfm_summary)
    lightglue_rows = _read_csv(args.lightglue_metrics)
    lightglue_index = _index_lightglue(lightglue_rows, args.lightglue_label)

    if not pfm_rows:
        raise ValueError(f"{args.pfm_summary} has no rows")
    lightglue_for_pfm = [lightglue_index[_pair_key(row)] for row in pfm_rows]
    lightglue_summary = _summarize_rows(lightglue_for_pfm)
    pfm_raw_summary = _summarize_rows(pfm_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_rows: list[dict[str, object]] = []
    for threshold in thresholds:
        hybrid_rows = hybrid_rows_for_threshold(pfm_rows, lightglue_index, threshold)
        write_csv(args.output_dir / f"hybrid_rows_threshold_{_safe_threshold_name(threshold)}.csv", hybrid_rows)
        threshold_rows.append(
            summarize_threshold(args.split_label, threshold, hybrid_rows, lightglue_summary, pfm_raw_summary)
        )

    write_csv(args.output_dir / "threshold_summary.csv", threshold_rows)
    best = choose_best_threshold(threshold_rows, int(lightglue_summary["wrong"]))
    summary = {
        "split": args.split_label,
        "pfm_summary": str(args.pfm_summary),
        "lightglue_metrics": str(args.lightglue_metrics),
        "lightglue_label": args.lightglue_label,
        "threshold_policy": "best correct with hybrid wrong <= LightGlue wrong; if none, best correct diagnostic",
        "lightglue": lightglue_summary,
        "pfm_raw": pfm_raw_summary,
        "best_threshold": best,
        "threshold_summary_csv": str(args.output_dir / "threshold_summary.csv"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_html(args.output_dir / "index.html", summary, threshold_rows)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
