#!/usr/bin/env python3
"""Train a lightweight match-set rejection calibrator from feature-only rows."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


IDENTITY_FIELDS = [
    "source_name",
    "split",
    "pair_index",
    "pair_type",
    "base_id",
    "reference_variant",
    "target_variant",
]


PREDICTION_FIELDS = [
    *IDENTITY_FIELDS,
    "reject_label",
    "reject_probability",
    "predicted_reject",
    "pfm_matches",
    "pfm_correct",
    "pfm_wrong",
    "pfm_precision",
    "lightglue_matches",
    "lightglue_correct",
    "lightglue_wrong",
    "lightglue_precision",
]


SWEEP_FIELDS = [
    "threshold",
    "train_predicted_reject_rows",
    "train_kept_pfm_correct",
    "train_kept_pfm_wrong",
    "train_kept_precision",
    "train_correct_retention",
    "train_wrong_reduction",
    "train_hybrid_pfm_lightglue_correct",
    "train_hybrid_pfm_lightglue_wrong",
    "train_hybrid_pfm_lightglue_precision",
    "train_hybrid_correct_delta_vs_lightglue",
    "train_hybrid_wrong_delta_vs_lightglue",
    "eval_predicted_reject_rows",
    "eval_kept_pfm_correct",
    "eval_kept_pfm_wrong",
    "eval_kept_precision",
    "eval_correct_retention",
    "eval_wrong_reduction",
    "eval_hybrid_pfm_lightglue_correct",
    "eval_hybrid_pfm_lightglue_wrong",
    "eval_hybrid_pfm_lightglue_precision",
    "eval_hybrid_correct_delta_vs_lightglue",
    "eval_hybrid_wrong_delta_vs_lightglue",
]


@dataclass(frozen=True)
class LogisticModel:
    feature_columns: list[str]
    means: list[float]
    scales: list[float]
    weights: list[float]
    bias: float
    label_column: str
    threshold: float = 1.000001


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def select_feature_columns(rows: Sequence[dict[str, str]]) -> list[str]:
    feature_columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key.startswith("feature_") and key not in seen:
                feature_columns.append(key)
                seen.add(key)
    return feature_columns


def _validate_feature_columns(feature_columns: Sequence[str]) -> None:
    if not feature_columns:
        raise ValueError("no feature columns found; expected columns prefixed with feature_")
    leaked = [name for name in feature_columns if not name.startswith("feature_")]
    if leaked:
        raise ValueError(f"feature columns must be prefixed with feature_: {', '.join(leaked)}")


def _labels(rows: Sequence[dict[str, str]], label_column: str) -> list[float]:
    labels = [_float_value(row, label_column) for row in rows]
    if not labels:
        raise ValueError("training rows are empty")
    unique = {int(round(value)) for value in labels}
    if not unique.issubset({0, 1}):
        raise ValueError(f"{label_column} must contain binary 0/1 labels")
    if len(unique) < 2:
        raise ValueError(f"{label_column} must contain both positive and negative rows")
    return [1.0 if value >= 0.5 else 0.0 for value in labels]


def _fit_standardizer(rows: Sequence[dict[str, str]], feature_columns: Sequence[str]) -> tuple[list[float], list[float]]:
    means: list[float] = []
    scales: list[float] = []
    for name in feature_columns:
        values = [_float_value(row, name) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale > 1e-12 else 1.0)
    return means, scales


def _matrix(
    rows: Sequence[dict[str, str]],
    feature_columns: Sequence[str],
    means: Sequence[float],
    scales: Sequence[float],
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append(
            [
                (_float_value(row, name) - means[index]) / scales[index]
                for index, name in enumerate(feature_columns)
            ]
        )
    return matrix


def _sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def train_model(
    rows: Sequence[dict[str, str]],
    *,
    feature_columns: Sequence[str],
    label_column: str = "reject_label",
    epochs: int = 1000,
    learning_rate: float = 0.05,
    l2: float = 0.001,
) -> LogisticModel:
    _validate_feature_columns(feature_columns)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be a positive finite value")
    if l2 < 0.0 or not math.isfinite(l2):
        raise ValueError("l2 must be a nonnegative finite value")

    labels = _labels(rows, label_column)
    means, scales = _fit_standardizer(rows, feature_columns)
    features = _matrix(rows, feature_columns, means, scales)
    positive_count = sum(1 for value in labels if value >= 0.5)
    negative_count = len(labels) - positive_count
    positive_weight = len(labels) / (2.0 * positive_count)
    negative_weight = len(labels) / (2.0 * negative_count)
    sample_weights = [positive_weight if value >= 0.5 else negative_weight for value in labels]
    weight_total = sum(sample_weights)
    prior = min(max(sum(labels) / len(labels), 1e-6), 1.0 - 1e-6)
    weights = [0.0 for _ in feature_columns]
    bias = math.log(prior / (1.0 - prior))

    for _ in range(epochs):
        grad_weights = [0.0 for _ in feature_columns]
        grad_bias = 0.0
        for vector, label, sample_weight in zip(features, labels, sample_weights):
            logit = bias + sum(weight * value for weight, value in zip(weights, vector))
            error = (_sigmoid(logit) - label) * sample_weight
            grad_bias += error
            for index, value in enumerate(vector):
                grad_weights[index] += error * value
        bias -= learning_rate * grad_bias / weight_total
        for index in range(len(weights)):
            gradient = grad_weights[index] / weight_total + l2 * weights[index]
            weights[index] -= learning_rate * gradient

    return LogisticModel(
        feature_columns=list(feature_columns),
        means=means,
        scales=scales,
        weights=weights,
        bias=bias,
        label_column=label_column,
    )


def score_rows(model: LogisticModel, rows: Sequence[dict[str, str]]) -> list[float]:
    features = _matrix(rows, model.feature_columns, model.means, model.scales)
    scores: list[float] = []
    for vector in features:
        logit = model.bias + sum(weight * value for weight, value in zip(model.weights, vector))
        scores.append(_sigmoid(logit))
    return scores


def evaluate_threshold(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    threshold: float,
    label_column: str,
) -> dict[str, object]:
    if len(rows) != len(scores):
        raise ValueError("rows and scores must have the same length")

    total_matches = sum(_int_value(row, "pfm_matches") for row in rows)
    total_correct = sum(_int_value(row, "pfm_correct") for row in rows)
    total_wrong = sum(_int_value(row, "pfm_wrong") for row in rows)
    lightglue_matches = sum(_int_value(row, "lightglue_matches") for row in rows)
    lightglue_correct = sum(_int_value(row, "lightglue_correct") for row in rows)
    lightglue_wrong = sum(_int_value(row, "lightglue_wrong") for row in rows)
    kept_matches = 0
    kept_correct = 0
    kept_wrong = 0
    fallback_matches = 0
    fallback_correct = 0
    fallback_wrong = 0
    predicted_reject_rows = 0
    tp = fp = tn = fn = 0

    for row, score in zip(rows, scores):
        predicted_reject = score >= threshold
        label = _int_value(row, label_column)
        if predicted_reject:
            predicted_reject_rows += 1
            fallback_matches += _int_value(row, "lightglue_matches")
            fallback_correct += _int_value(row, "lightglue_correct")
            fallback_wrong += _int_value(row, "lightglue_wrong")
            if label == 1:
                tp += 1
            else:
                fp += 1
            continue
        kept_matches += _int_value(row, "pfm_matches")
        kept_correct += _int_value(row, "pfm_correct")
        kept_wrong += _int_value(row, "pfm_wrong")
        if label == 1:
            fn += 1
        else:
            tn += 1

    rejected_matches = total_matches - kept_matches
    rejected_correct = total_correct - kept_correct
    rejected_wrong = total_wrong - kept_wrong
    correct_retention = kept_correct / total_correct if total_correct > 0 else 1.0
    wrong_retention = kept_wrong / total_wrong if total_wrong > 0 else 0.0
    wrong_reduction = 1.0 - wrong_retention if total_wrong > 0 else 0.0
    kept_precision = kept_correct / kept_matches if kept_matches > 0 else 0.0
    total_precision = total_correct / total_matches if total_matches > 0 else 0.0
    lightglue_precision = lightglue_correct / lightglue_matches if lightglue_matches > 0 else 0.0
    hybrid_matches = kept_matches + fallback_matches
    hybrid_correct = kept_correct + fallback_correct
    hybrid_wrong = kept_wrong + fallback_wrong
    hybrid_precision = hybrid_correct / hybrid_matches if hybrid_matches > 0 else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0

    return {
        "rows": len(rows),
        "threshold": threshold,
        "predicted_reject_rows": predicted_reject_rows,
        "kept_rows": len(rows) - predicted_reject_rows,
        "pfm_matches": total_matches,
        "pfm_correct": total_correct,
        "pfm_wrong": total_wrong,
        "pfm_precision": total_precision,
        "lightglue_matches": lightglue_matches,
        "lightglue_correct": lightglue_correct,
        "lightglue_wrong": lightglue_wrong,
        "lightglue_precision": lightglue_precision,
        "kept_pfm_matches": kept_matches,
        "kept_pfm_correct": kept_correct,
        "kept_pfm_wrong": kept_wrong,
        "kept_precision": kept_precision,
        "rejected_pfm_matches": rejected_matches,
        "rejected_pfm_correct": rejected_correct,
        "rejected_pfm_wrong": rejected_wrong,
        "correct_retention": correct_retention,
        "wrong_reduction": wrong_reduction,
        "fallback_lightglue_matches": fallback_matches,
        "fallback_lightglue_correct": fallback_correct,
        "fallback_lightglue_wrong": fallback_wrong,
        "hybrid_pfm_lightglue_matches": hybrid_matches,
        "hybrid_pfm_lightglue_correct": hybrid_correct,
        "hybrid_pfm_lightglue_wrong": hybrid_wrong,
        "hybrid_pfm_lightglue_precision": hybrid_precision,
        "hybrid_correct_delta_vs_pfm": hybrid_correct - total_correct,
        "hybrid_wrong_delta_vs_pfm": hybrid_wrong - total_wrong,
        "hybrid_correct_delta_vs_lightglue": hybrid_correct - lightglue_correct,
        "hybrid_wrong_delta_vs_lightglue": hybrid_wrong - lightglue_wrong,
        "label_true_positive": tp,
        "label_false_positive": fp,
        "label_true_negative": tn,
        "label_false_negative": fn,
        "label_accuracy": accuracy,
    }


def _candidate_thresholds(scores: Sequence[float]) -> list[float]:
    if not scores:
        return [1.000001]
    values = sorted({max(0.0, min(1.0, score)) for score in scores})
    thresholds = {0.0}
    for value in values:
        thresholds.add(value)
        thresholds.add(min(1.000001, value + 1e-6))
    return sorted(thresholds)


def build_threshold_sweep(
    train_rows: Sequence[dict[str, str]],
    train_scores: Sequence[float],
    eval_rows: Sequence[dict[str, str]],
    eval_scores: Sequence[float],
    *,
    label_column: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    thresholds = _candidate_thresholds([*train_scores, *eval_scores])
    for threshold in thresholds:
        train_metrics = evaluate_threshold(train_rows, train_scores, threshold=threshold, label_column=label_column)
        eval_metrics = evaluate_threshold(eval_rows, eval_scores, threshold=threshold, label_column=label_column)
        rows.append(
            {
                "threshold": threshold,
                "train_predicted_reject_rows": train_metrics["predicted_reject_rows"],
                "train_kept_pfm_correct": train_metrics["kept_pfm_correct"],
                "train_kept_pfm_wrong": train_metrics["kept_pfm_wrong"],
                "train_kept_precision": train_metrics["kept_precision"],
                "train_correct_retention": train_metrics["correct_retention"],
                "train_wrong_reduction": train_metrics["wrong_reduction"],
                "train_hybrid_pfm_lightglue_correct": train_metrics["hybrid_pfm_lightglue_correct"],
                "train_hybrid_pfm_lightglue_wrong": train_metrics["hybrid_pfm_lightglue_wrong"],
                "train_hybrid_pfm_lightglue_precision": train_metrics["hybrid_pfm_lightglue_precision"],
                "train_hybrid_correct_delta_vs_lightglue": train_metrics["hybrid_correct_delta_vs_lightglue"],
                "train_hybrid_wrong_delta_vs_lightglue": train_metrics["hybrid_wrong_delta_vs_lightglue"],
                "eval_predicted_reject_rows": eval_metrics["predicted_reject_rows"],
                "eval_kept_pfm_correct": eval_metrics["kept_pfm_correct"],
                "eval_kept_pfm_wrong": eval_metrics["kept_pfm_wrong"],
                "eval_kept_precision": eval_metrics["kept_precision"],
                "eval_correct_retention": eval_metrics["correct_retention"],
                "eval_wrong_reduction": eval_metrics["wrong_reduction"],
                "eval_hybrid_pfm_lightglue_correct": eval_metrics["hybrid_pfm_lightglue_correct"],
                "eval_hybrid_pfm_lightglue_wrong": eval_metrics["hybrid_pfm_lightglue_wrong"],
                "eval_hybrid_pfm_lightglue_precision": eval_metrics["hybrid_pfm_lightglue_precision"],
                "eval_hybrid_correct_delta_vs_lightglue": eval_metrics["hybrid_correct_delta_vs_lightglue"],
                "eval_hybrid_wrong_delta_vs_lightglue": eval_metrics["hybrid_wrong_delta_vs_lightglue"],
            }
        )
    return rows


def choose_threshold(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    label_column: str,
    min_kept_correct_ratio: float,
) -> float:
    if min_kept_correct_ratio < 0.0 or min_kept_correct_ratio > 1.0:
        raise ValueError("min_kept_correct_ratio must be in [0, 1]")
    candidates: list[tuple[float, float, float, int, float]] = []
    for threshold in _candidate_thresholds(scores):
        metrics = evaluate_threshold(rows, scores, threshold=threshold, label_column=label_column)
        correct_retention = float(metrics["correct_retention"])
        if correct_retention < min_kept_correct_ratio:
            continue
        candidates.append(
            (
                float(metrics["wrong_reduction"]),
                float(metrics["kept_precision"]),
                -int(metrics["kept_pfm_wrong"]),
                correct_retention,
                threshold,
            )
        )
    if not candidates:
        return 1.000001
    return max(candidates)[-1]


def choose_hybrid_lightglue_wrong_cap_threshold(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    label_column: str,
    max_wrong_delta_vs_lightglue: int,
) -> float:
    candidates: list[tuple[int, float, int, int, float, int]] = []
    for threshold in _candidate_thresholds(scores):
        metrics = evaluate_threshold(rows, scores, threshold=threshold, label_column=label_column)
        wrong_delta = int(metrics["hybrid_wrong_delta_vs_lightglue"])
        if wrong_delta > max_wrong_delta_vs_lightglue:
            continue
        candidates.append(
            (
                int(metrics["hybrid_correct_delta_vs_lightglue"]),
                float(metrics["hybrid_pfm_lightglue_precision"]),
                -wrong_delta,
                -int(metrics["predicted_reject_rows"]),
                threshold,
                int(metrics["predicted_reject_rows"]),
            )
        )
    if not candidates:
        return 1.000001
    best = max(candidates)
    best_correct_delta = best[0]
    best_reject_rows = best[-1]
    if best_correct_delta <= 0 and best_reject_rows == len(rows):
        return 0.0
    return best[-2]


def choose_pfm_wrong_cap_threshold(
    rows: Sequence[dict[str, str]],
    scores: Sequence[float],
    *,
    label_column: str,
    max_kept_pfm_wrong: int,
    min_kept_correct_ratio: float,
) -> float:
    if max_kept_pfm_wrong < 0:
        raise ValueError("max_kept_pfm_wrong must be nonnegative")
    if min_kept_correct_ratio < 0.0 or min_kept_correct_ratio > 1.0:
        raise ValueError("min_kept_correct_ratio must be in [0, 1]")
    candidates: list[tuple[int, float, int, float]] = []
    for threshold in _candidate_thresholds(scores):
        metrics = evaluate_threshold(rows, scores, threshold=threshold, label_column=label_column)
        kept_wrong = int(metrics["kept_pfm_wrong"])
        if kept_wrong > max_kept_pfm_wrong:
            continue
        if float(metrics["correct_retention"]) < min_kept_correct_ratio:
            continue
        candidates.append(
            (
                int(metrics["kept_pfm_correct"]),
                float(metrics["kept_precision"]),
                -int(metrics["predicted_reject_rows"]),
                threshold,
            )
        )
    if not candidates:
        return 1.000001
    return max(candidates)[-1]


def _filter_split(rows: Sequence[dict[str, str]], splits: Sequence[str]) -> list[dict[str, str]]:
    wanted = {split.strip() for split in splits if split.strip()}
    return [row for row in rows if row.get("split", "") in wanted]


def _write_predictions(path: Path, rows: Sequence[dict[str, str]], scores: Sequence[float], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row, score in zip(rows, scores):
            output = {field: row.get(field, "") for field in PREDICTION_FIELDS}
            output["reject_probability"] = _fmt_float(score)
            output["predicted_reject"] = "1" if score >= threshold else "0"
            writer.writerow(output)


def _write_sweep(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key, value in list(formatted.items()):
                if isinstance(value, float):
                    formatted[key] = _fmt_float(value)
            writer.writerow(formatted)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _model_payload(model: LogisticModel) -> dict[str, object]:
    return {
        "type": "standardized_logistic_regression",
        "feature_columns": model.feature_columns,
        "means": model.means,
        "scales": model.scales,
        "weights": model.weights,
        "bias": model.bias,
        "label_column": model.label_column,
        "threshold": model.threshold,
    }


def _render_metrics_table(title: str, metrics: dict[str, object]) -> str:
    keys = [
        "rows",
        "predicted_reject_rows",
        "pfm_correct",
        "pfm_wrong",
        "pfm_precision",
        "lightglue_correct",
        "lightglue_wrong",
        "lightglue_precision",
        "kept_pfm_correct",
        "kept_pfm_wrong",
        "kept_precision",
        "correct_retention",
        "wrong_reduction",
        "fallback_lightglue_correct",
        "fallback_lightglue_wrong",
        "hybrid_pfm_lightglue_correct",
        "hybrid_pfm_lightglue_wrong",
        "hybrid_pfm_lightglue_precision",
        "hybrid_correct_delta_vs_lightglue",
        "hybrid_wrong_delta_vs_lightglue",
        "label_accuracy",
    ]
    body = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(metrics.get(key, '')))}</td></tr>"
        for key in keys
    )
    return (
        f"<h2>{html.escape(title)}</h2>"
        '<table border="1" cellspacing="0" cellpadding="4">'
        "<tr><th>metric</th><th>value</th></tr>"
        f"{body}</table>"
    )


def _write_report_html(
    path: Path,
    *,
    dataset_csv: Path,
    eval_dataset_csv: Path | None,
    model_json: Path,
    predictions_csv: Path,
    sweep_csv: Path,
    summary: dict[str, object],
) -> None:
    train_metrics = summary.get("train", {})
    eval_metrics = summary.get("eval", {})
    threshold_selection_metrics = summary.get("threshold_selection", {})
    assert isinstance(train_metrics, dict)
    assert isinstance(eval_metrics, dict)
    assert isinstance(threshold_selection_metrics, dict)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Match-set rejection calibrator</title>",
                "<h1>Match-set rejection calibrator</h1>",
                f"<p>dataset_csv={html.escape(str(dataset_csv))}</p>",
                f"<p>eval_dataset_csv={html.escape(str(eval_dataset_csv or dataset_csv))}</p>",
                f"<p>model_json={html.escape(str(model_json))}</p>",
                f"<p>predictions_csv={html.escape(str(predictions_csv))}</p>",
                f"<p>threshold_sweep_csv={html.escape(str(sweep_csv))}</p>",
                _render_metrics_table("Train", train_metrics),
                _render_metrics_table("Threshold selection", threshold_selection_metrics),
                _render_metrics_table("Eval", eval_metrics),
                "<h2>Summary JSON</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--eval-dataset-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", action="append", default=[])
    parser.add_argument("--eval-split", action="append", default=[])
    parser.add_argument("--label-column", default="reject_label")
    parser.add_argument("--feature-column", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--min-kept-correct-ratio", type=float, default=0.90)
    parser.add_argument(
        "--threshold-objective",
        choices=["kept_correct_ratio", "hybrid_lightglue_wrong_cap", "pfm_wrong_cap"],
        default="kept_correct_ratio",
    )
    parser.add_argument(
        "--threshold-selection-source",
        choices=["train", "eval"],
        default="train",
        help="Rows used to choose the stored threshold. Use eval for validation-selected deployment thresholds.",
    )
    parser.add_argument("--max-hybrid-wrong-delta-vs-lightglue", type=int, default=0)
    parser.add_argument(
        "--max-kept-pfm-wrong",
        type=int,
        default=None,
        help="Maximum kept PFM wrong matches for --threshold-objective pfm_wrong_cap. Defaults to the LightGlue wrong count in threshold-selection rows.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_source_rows = _read_csv_rows(args.dataset_csv)
    eval_source_rows = (
        _read_csv_rows(args.eval_dataset_csv)
        if args.eval_dataset_csv is not None
        else train_source_rows
    )
    train_splits = args.train_split or ["formal"]
    eval_splits = args.eval_split or ["validation"]
    train_rows = _filter_split(train_source_rows, train_splits)
    eval_rows = _filter_split(eval_source_rows, eval_splits)
    if not train_rows:
        raise ValueError(f"no training rows found for split(s): {', '.join(train_splits)}")
    if not eval_rows:
        raise ValueError(f"no eval rows found for split(s): {', '.join(eval_splits)}")

    feature_columns = (
        list(args.feature_column)
        if args.feature_column
        else select_feature_columns(train_source_rows)
    )
    model = train_model(
        train_rows,
        feature_columns=feature_columns,
        label_column=str(args.label_column),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
    )
    train_scores = score_rows(model, train_rows)
    eval_scores = score_rows(model, eval_rows)
    if args.threshold_selection_source == "eval":
        threshold_rows = eval_rows
        threshold_scores = eval_scores
    else:
        threshold_rows = train_rows
        threshold_scores = train_scores
    max_kept_pfm_wrong = (
        int(args.max_kept_pfm_wrong)
        if args.max_kept_pfm_wrong is not None
        else sum(_int_value(row, "lightglue_wrong") for row in threshold_rows)
    )
    if args.threshold_objective == "hybrid_lightglue_wrong_cap":
        threshold = choose_hybrid_lightglue_wrong_cap_threshold(
            threshold_rows,
            threshold_scores,
            label_column=str(args.label_column),
            max_wrong_delta_vs_lightglue=int(args.max_hybrid_wrong_delta_vs_lightglue),
        )
    elif args.threshold_objective == "pfm_wrong_cap":
        threshold = choose_pfm_wrong_cap_threshold(
            threshold_rows,
            threshold_scores,
            label_column=str(args.label_column),
            max_kept_pfm_wrong=max_kept_pfm_wrong,
            min_kept_correct_ratio=float(args.min_kept_correct_ratio),
        )
    else:
        threshold = choose_threshold(
            threshold_rows,
            threshold_scores,
            label_column=str(args.label_column),
            min_kept_correct_ratio=float(args.min_kept_correct_ratio),
        )
    model = LogisticModel(
        feature_columns=model.feature_columns,
        means=model.means,
        scales=model.scales,
        weights=model.weights,
        bias=model.bias,
        label_column=model.label_column,
        threshold=threshold,
    )
    train_metrics = evaluate_threshold(train_rows, train_scores, threshold=threshold, label_column=model.label_column)
    eval_metrics = evaluate_threshold(eval_rows, eval_scores, threshold=threshold, label_column=model.label_column)
    threshold_selection_metrics = evaluate_threshold(
        threshold_rows,
        threshold_scores,
        threshold=threshold,
        label_column=model.label_column,
    )
    sweep = build_threshold_sweep(
        train_rows,
        train_scores,
        eval_rows,
        eval_scores,
        label_column=model.label_column,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_json = output_dir / "model.json"
    summary_json = output_dir / "summary.json"
    predictions_csv = output_dir / f"{'_'.join(eval_splits)}_predictions.csv"
    sweep_csv = output_dir / "threshold_sweep.csv"
    output_html = output_dir / "index.html"

    model_payload = _model_payload(model)
    summary = {
        "dataset_csv": str(args.dataset_csv),
        "eval_dataset_csv": str(args.eval_dataset_csv or args.dataset_csv),
        "train_splits": train_splits,
        "eval_splits": eval_splits,
        "feature_columns": model.feature_columns,
        "label_column": model.label_column,
        "threshold_objective": str(args.threshold_objective),
        "threshold_selection_source": str(args.threshold_selection_source),
        "min_kept_correct_ratio": float(args.min_kept_correct_ratio),
        "max_hybrid_wrong_delta_vs_lightglue": int(args.max_hybrid_wrong_delta_vs_lightglue),
        "max_kept_pfm_wrong": max_kept_pfm_wrong,
        "threshold": threshold,
        "threshold_selection": threshold_selection_metrics,
        "train": train_metrics,
        "eval": eval_metrics,
    }
    _write_json(model_json, model_payload)
    _write_json(summary_json, summary)
    _write_predictions(predictions_csv, eval_rows, eval_scores, threshold)
    _write_sweep(sweep_csv, sweep)
    _write_report_html(
        output_html,
        dataset_csv=args.dataset_csv,
        eval_dataset_csv=args.eval_dataset_csv,
        model_json=model_json,
        predictions_csv=predictions_csv,
        sweep_csv=sweep_csv,
        summary=summary,
    )
    print(
        "calibrator "
        f"train_rows={len(train_rows)} eval_rows={len(eval_rows)} threshold={threshold:.6f} "
        f"eval_reject={eval_metrics['predicted_reject_rows']} "
        f"eval_kept_correct={eval_metrics['kept_pfm_correct']} eval_kept_wrong={eval_metrics['kept_pfm_wrong']} "
        f"output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
