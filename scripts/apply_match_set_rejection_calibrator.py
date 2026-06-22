#!/usr/bin/env python3
"""Apply a match-set rejection calibrator and write PFM/LightGlue hybrid metrics."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Sequence

from train_match_set_rejection_calibrator import LogisticModel, score_rows


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
    "reject_probability",
    "predicted_reject",
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _load_model(path: Path) -> LogisticModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "standardized_logistic_regression":
        raise ValueError(f"unsupported model type: {payload.get('type', '')}")
    feature_columns = list(payload.get("feature_columns", []))
    means = [float(value) for value in payload.get("means", [])]
    scales = [float(value) for value in payload.get("scales", [])]
    weights = [float(value) for value in payload.get("weights", [])]
    if not (len(feature_columns) == len(means) == len(scales) == len(weights)):
        raise ValueError("model feature_columns, means, scales and weights must have the same length")
    return LogisticModel(
        feature_columns=feature_columns,
        means=means,
        scales=scales,
        weights=weights,
        bias=float(payload.get("bias", 0.0)),
        label_column=str(payload.get("label_column", "reject_label")),
        threshold=float(payload.get("threshold", 1.000001)),
    )


def _chosen_metrics(
    row: dict[str, str],
    *,
    predicted_reject: bool,
    reject_action: str,
) -> tuple[str, int, int, int, float]:
    if predicted_reject and reject_action == "zero":
        return "rejected", 0, 0, 0, 0.0
    prefix = "lightglue" if predicted_reject else "pfm"
    matches = _int_value(row, f"{prefix}_matches")
    correct = _int_value(row, f"{prefix}_correct")
    wrong = _int_value(row, f"{prefix}_wrong")
    precision = correct / matches if matches > 0 else 0.0
    return prefix, matches, correct, wrong, precision


def build_hybrid_rows(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    threshold: float,
    reject_action: str = "lightglue",
) -> list[dict[str, str]]:
    if len(rows) != len(scores):
        raise ValueError("rows and scores must have the same length")
    if reject_action not in {"lightglue", "zero"}:
        raise ValueError(f"unsupported reject_action: {reject_action}")
    output_rows: list[dict[str, str]] = []
    for row, score in zip(rows, scores):
        predicted_reject = score >= threshold
        chosen_source, matches, correct, wrong, precision = _chosen_metrics(
            row,
            predicted_reject=predicted_reject,
            reject_action=reject_action,
        )
        output = {field: row.get(field, "") for field in IDENTITY_FIELDS}
        output.update(
            {
                "reject_probability": _fmt_float(score),
                "predicted_reject": "1" if predicted_reject else "0",
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


def summarize_rows(rows: Sequence[dict[str, str]]) -> dict[str, object]:
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
    rejected_rows = sum(1 for row in rows if row.get("chosen_source") == "rejected")
    return {
        "rows": len(rows),
        "kept_pfm_rows": kept_pfm_rows,
        "fallback_lightglue_rows": fallback_lightglue_rows,
        "rejected_rows": rejected_rows,
        "pfm_matches": pfm_matches,
        "pfm_correct": pfm_correct,
        "pfm_wrong": pfm_wrong,
        "pfm_precision": pfm_correct / pfm_matches if pfm_matches > 0 else 0.0,
        "lightglue_matches": lightglue_matches,
        "lightglue_correct": lightglue_correct,
        "lightglue_wrong": lightglue_wrong,
        "lightglue_precision": lightglue_correct / lightglue_matches if lightglue_matches > 0 else 0.0,
        "hybrid_matches": hybrid_matches,
        "hybrid_correct": hybrid_correct,
        "hybrid_wrong": hybrid_wrong,
        "hybrid_precision": hybrid_correct / hybrid_matches if hybrid_matches > 0 else 0.0,
        "hybrid_correct_delta_vs_pfm": hybrid_correct - pfm_correct,
        "hybrid_wrong_delta_vs_pfm": hybrid_wrong - pfm_wrong,
        "hybrid_correct_delta_vs_lightglue": hybrid_correct - lightglue_correct,
        "hybrid_wrong_delta_vs_lightglue": hybrid_wrong - lightglue_wrong,
    }


def _write_output_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_html(path: Path, *, output_csv: Path, model_json: Path, dataset_csv: Path, summary: dict[str, object]) -> None:
    keys = [
        "rows",
        "kept_pfm_rows",
        "fallback_lightglue_rows",
        "rejected_rows",
        "pfm_correct",
        "pfm_wrong",
        "pfm_precision",
        "lightglue_correct",
        "lightglue_wrong",
        "lightglue_precision",
        "hybrid_correct",
        "hybrid_wrong",
        "hybrid_precision",
        "hybrid_correct_delta_vs_lightglue",
        "hybrid_wrong_delta_vs_lightglue",
    ]
    body = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(summary.get(key, '')))}</td></tr>"
        for key in keys
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Match-set rejection calibrator application</title>",
                "<h1>Match-set rejection calibrator application</h1>",
                f"<p>dataset_csv={html.escape(str(dataset_csv))}</p>",
                f"<p>model_json={html.escape(str(model_json))}</p>",
                f"<p>output_csv={html.escape(str(output_csv))}</p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>metric</th><th>value</th></tr>",
                body,
                "</table>",
                "<h2>Summary JSON</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument(
        "--reject-action",
        choices=["lightglue", "zero"],
        default="lightglue",
        help="Use LightGlue fallback for rejected PFM pairs, or zero the pair for PFM-only rejection.",
    )
    parser.add_argument(
        "--threshold-override",
        type=float,
        default=None,
        help="Override the model threshold for diagnostic sweeps without modifying model.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model = _load_model(args.model_json)
    rows = _read_csv_rows(args.dataset_csv)
    scores = score_rows(model, rows)
    threshold = float(args.threshold_override) if args.threshold_override is not None else model.threshold
    hybrid_rows = build_hybrid_rows(rows, scores, threshold=threshold, reject_action=str(args.reject_action))
    summary = summarize_rows(hybrid_rows)
    summary.update(
        {
            "dataset_csv": str(args.dataset_csv),
            "model_json": str(args.model_json),
            "threshold": threshold,
            "model_threshold": model.threshold,
            "feature_columns": model.feature_columns,
            "reject_action": str(args.reject_action),
        }
    )
    _write_output_csv(args.output_csv, hybrid_rows)
    _write_summary(args.summary_json, summary)
    _write_html(
        args.output_html,
        output_csv=args.output_csv,
        model_json=args.model_json,
        dataset_csv=args.dataset_csv,
        summary=summary,
    )
    print(
        f"applied rows={len(rows)} kept_pfm={summary['kept_pfm_rows']} "
        f"fallback_lightglue={summary['fallback_lightglue_rows']} output={args.output_csv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
