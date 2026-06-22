#!/usr/bin/env python3
"""Apply a trained per-match filter calibrator to PFM match detail CSVs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from train_match_detail_filter_calibrator import build_training_rows
from train_match_detail_mlp_filter_calibrator import MlpModel, score_rows as score_mlp_rows
from train_match_set_rejection_calibrator import LogisticModel, score_rows as score_logistic_rows


PAIR_FIELDS = [
    "label",
    "split",
    "pair_index",
    "base_id",
    "reference_variant",
    "target_variant",
]

PAIR_SUMMARY_FIELDS = [
    *PAIR_FIELDS,
    "matches",
    "correct",
    "wrong",
    "precision",
    "kept_matches",
    "kept_correct",
    "kept_wrong",
    "kept_precision",
    "rejected_matches",
    "rejected_correct",
    "rejected_wrong",
    "correct_retention",
    "wrong_reduction",
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


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _fmt_probability(value: float) -> str:
    return f"{value:.17g}"


FilterModel = LogisticModel | MlpModel


def _load_model(path: Path) -> tuple[FilterModel, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_type = payload.get("type")
    feature_columns = [str(value) for value in payload.get("feature_columns", [])]
    means = [float(value) for value in payload.get("means", [])]
    scales = [float(value) for value in payload.get("scales", [])]
    if not (len(feature_columns) == len(means) == len(scales)):
        raise ValueError("model feature_columns, means and scales must have the same length")
    if model_type == "standardized_mlp_match_filter":
        hidden_dim = int(payload.get("hidden_dim", 0))
        layer1_weight = [
            [float(value) for value in row]
            for row in payload.get("layer1_weight", [])
        ]
        layer1_bias = [float(value) for value in payload.get("layer1_bias", [])]
        layer2_weight = [float(value) for value in payload.get("layer2_weight", [])]
        if hidden_dim <= 0:
            raise ValueError("MLP model hidden_dim must be positive")
        if not (len(layer1_weight) == len(layer1_bias) == len(layer2_weight) == hidden_dim):
            raise ValueError("MLP model hidden layer dimensions do not match hidden_dim")
        if any(len(row) != len(feature_columns) for row in layer1_weight):
            raise ValueError("MLP model layer1_weight width must match feature_columns")
        return (
            MlpModel(
                model_type="standardized_mlp_match_filter",
                feature_columns=feature_columns,
                means=means,
                scales=scales,
                hidden_dim=hidden_dim,
                layer1_weight=layer1_weight,
                layer1_bias=layer1_bias,
                layer2_weight=layer2_weight,
                layer2_bias=float(payload.get("layer2_bias", 0.0)),
                label_column=str(payload.get("label_column", "reject_label")),
                threshold=float(payload.get("threshold", 1.000001)),
                include_true_geometry_features=bool(payload.get("include_true_geometry_features", False)),
            ),
            bool(payload.get("include_true_geometry_features", False)),
        )
    if model_type != "standardized_logistic_regression_match_filter":
        raise ValueError(f"unsupported model type: {payload.get('type', '')}")
    weights = [float(value) for value in payload.get("weights", [])]
    if len(weights) != len(feature_columns):
        raise ValueError("model weights must have the same length as feature_columns")
    return (
        LogisticModel(
            feature_columns=feature_columns,
            means=means,
            scales=scales,
            weights=weights,
            bias=float(payload.get("bias", 0.0)),
            label_column=str(payload.get("label_column", "reject_label")),
            threshold=float(payload.get("threshold", 1.000001)),
        ),
        bool(payload.get("include_true_geometry_features", False)),
    )


def _model_type(model: FilterModel) -> str:
    if isinstance(model, MlpModel):
        return model.model_type
    return "standardized_logistic_regression_match_filter"


def _score_model(model: FilterModel, feature_rows: Sequence[dict[str, str]]) -> list[float]:
    if isinstance(model, MlpModel):
        return score_mlp_rows(model, feature_rows)
    return score_logistic_rows(model, feature_rows)


def _pair_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return tuple(row.get(field, "") for field in PAIR_FIELDS)


def build_prediction_rows(
    original_rows: Sequence[dict[str, str]],
    feature_rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    threshold: float,
    variant_thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, str]]:
    if not (len(original_rows) == len(feature_rows) == len(scores)):
        raise ValueError("original rows, feature rows and scores must have the same length")
    thresholds = dict(variant_thresholds or {})
    prediction_rows: list[dict[str, str]] = []
    for original, feature_row, score in zip(original_rows, feature_rows, scores):
        row_threshold = thresholds.get(original.get("target_variant", ""), threshold)
        predicted_reject = score >= row_threshold
        row = dict(original)
        row["reject_probability"] = _fmt_probability(score)
        row["predicted_reject"] = "1" if predicted_reject else "0"
        row["reject_label"] = feature_row.get("reject_label", "")
        prediction_rows.append(row)
    return prediction_rows


def summarize_match_rows(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    matches = len(rows)
    correct = sum(1 for row in rows if _int_value(row, "correct") > 0)
    wrong = matches - correct
    kept_rows = [row for row in rows if row.get("predicted_reject") != "1"]
    kept_matches = len(kept_rows)
    kept_correct = sum(1 for row in kept_rows if _int_value(row, "correct") > 0)
    kept_wrong = kept_matches - kept_correct
    rejected_matches = matches - kept_matches
    rejected_correct = correct - kept_correct
    rejected_wrong = wrong - kept_wrong
    return {
        "matches": matches,
        "correct": correct,
        "wrong": wrong,
        "precision": correct / matches if matches > 0 else 0.0,
        "kept_matches": kept_matches,
        "kept_correct": kept_correct,
        "kept_wrong": kept_wrong,
        "kept_precision": kept_correct / kept_matches if kept_matches > 0 else 0.0,
        "rejected_matches": rejected_matches,
        "rejected_correct": rejected_correct,
        "rejected_wrong": rejected_wrong,
        "correct_retention": kept_correct / correct if correct > 0 else 0.0,
        "wrong_reduction": rejected_wrong / wrong if wrong > 0 else 0.0,
    }


def build_pair_summary_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[_pair_key(row)].append(row)

    output_rows: list[dict[str, str]] = []
    for key, group_rows in groups.items():
        metrics = summarize_match_rows(group_rows)
        row = {field: value for field, value in zip(PAIR_FIELDS, key)}
        for name in PAIR_SUMMARY_FIELDS:
            if name in PAIR_FIELDS:
                continue
            value = metrics[name]
            row[name] = _fmt_float(value) if isinstance(value, float) else str(value)
        output_rows.append(row)
    return output_rows


def _write_csv(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_html(path: Path, *, match_details: Path, model_json: Path, summary: dict[str, Any]) -> None:
    keys = [
        "matches",
        "correct",
        "wrong",
        "precision",
        "kept_matches",
        "kept_correct",
        "kept_wrong",
        "kept_precision",
        "rejected_correct",
        "rejected_wrong",
        "correct_retention",
        "wrong_reduction",
        "threshold",
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
                "<title>Match-detail filter application</title>",
                "<h1>Match-detail filter application</h1>",
                f"<p>match_details={html.escape(str(match_details))}</p>",
                f"<p>model_json={html.escape(str(model_json))}</p>",
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
    parser.add_argument("--match-details", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant-threshold",
        action="append",
        default=[],
        help="Override the model threshold for one target variant, formatted as target_variant=threshold.",
    )
    return parser.parse_args(argv)


def _parse_variant_thresholds(values: Sequence[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--variant-threshold must use target_variant=threshold: {value}")
        variant, threshold_text = value.split("=", 1)
        variant = variant.strip()
        if not variant:
            raise ValueError("--variant-threshold target_variant must be non-empty")
        try:
            threshold = float(threshold_text)
        except ValueError as exc:
            raise ValueError(f"invalid threshold for {variant}: {threshold_text}") from exc
        if not math.isfinite(threshold):
            raise ValueError(f"threshold for {variant} must be finite")
        thresholds[variant] = threshold
    return thresholds


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model, include_true_geometry_features = _load_model(args.model_json)
    variant_thresholds = _parse_variant_thresholds(args.variant_threshold)
    original_rows = _read_csv_rows(args.match_details)
    feature_rows = build_training_rows(
        original_rows,
        include_true_geometry_features=include_true_geometry_features,
    )
    scores = _score_model(model, feature_rows)
    prediction_rows = build_prediction_rows(
        original_rows,
        feature_rows,
        scores,
        threshold=model.threshold,
        variant_thresholds=variant_thresholds,
    )
    kept_rows = [row for row in prediction_rows if row.get("predicted_reject") != "1"]
    pair_summary_rows = build_pair_summary_rows(prediction_rows)
    summary = summarize_match_rows(prediction_rows)
    summary.update(
        {
            "match_details": str(args.match_details),
            "model_json": str(args.model_json),
            "threshold": model.threshold,
            "model_type": _model_type(model),
            "variant_thresholds": variant_thresholds,
            "feature_columns": model.feature_columns,
            "include_true_geometry_features": include_true_geometry_features,
            "pair_count": len(pair_summary_rows),
        }
    )

    output_dir = args.output_dir
    predictions_csv = output_dir / "match_predictions.csv"
    kept_csv = output_dir / "kept_match_details.csv"
    pair_summary_csv = output_dir / "pair_summary.csv"
    summary_json = output_dir / "summary.json"
    output_html = output_dir / "index.html"
    prediction_fields = list(original_rows[0].keys()) + ["reject_probability", "predicted_reject", "reject_label"] if original_rows else [
        "reject_probability",
        "predicted_reject",
        "reject_label",
    ]
    original_fields = list(original_rows[0].keys()) if original_rows else []

    _write_csv(predictions_csv, prediction_rows, prediction_fields)
    _write_csv(kept_csv, kept_rows, original_fields)
    _write_csv(pair_summary_csv, pair_summary_rows, PAIR_SUMMARY_FIELDS)
    _write_json(summary_json, summary)
    _write_html(output_html, match_details=args.match_details, model_json=args.model_json, summary=summary)
    print(
        f"match_detail_filter matches={summary['matches']} kept={summary['kept_matches']} "
        f"kept_correct={summary['kept_correct']} kept_wrong={summary['kept_wrong']} output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
