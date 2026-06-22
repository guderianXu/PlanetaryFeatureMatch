#!/usr/bin/env python3
"""Train a per-match filter from true-geometry edge supervision rows."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Sequence

from train_match_detail_filter_calibrator import (
    build_threshold_sweep,
    choose_match_threshold,
    choose_match_wrong_cap_threshold,
    filter_feature_columns,
    limit_training_rows,
    summarize_threshold,
)
from train_match_set_rejection_calibrator import (
    score_rows,
    select_feature_columns,
    train_model,
)


IDENTITY_FIELDS = [
    "source_name",
    "label",
    "pair_index",
    "base_id",
    "reference_variant",
    "target_variant",
    "split",
    "match_index",
]

GEOMETRY_FIELDS = [
    "geometry_valid_label",
    "geometry_invalid_label",
    "geometry_hard_negative_label",
    "geometry_visibility_label",
    "geometry_supervision_weight",
    "geometry_reprojection_error_px",
    "geometry_valid_fraction",
    "geometry_reason",
]

PREDICTION_FIELDS = [
    *IDENTITY_FIELDS,
    *GEOMETRY_FIELDS,
    "reject_label",
    "reject_probability",
    "predicted_reject",
    "pfm_matches",
    "pfm_correct",
    "pfm_wrong",
    "pfm_precision",
]

SWEEP_FIELDS = [
    "threshold",
    "train_kept_correct",
    "train_kept_wrong",
    "train_precision",
    "train_correct_retention",
    "train_wrong_reduction",
    "eval_kept_correct",
    "eval_kept_wrong",
    "eval_precision",
    "eval_correct_retention",
    "eval_wrong_reduction",
]

FORBIDDEN_FEATURE_FIELDS = {
    "feature_correct",
    "feature_error_px",
    "feature_geometry_reprojection_error_px",
    "feature_geometry_valid_fraction",
    "feature_valid_fraction",
}

FORBIDDEN_FEATURE_PREFIXES = (
    "feature_true_geometry_",
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_csv_rows_many(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_csv_rows(path))
    return rows


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
    return f"{float(value):.6f}"


def _is_allowed_feature(name: str) -> bool:
    if not name.startswith("feature_"):
        return False
    if name in FORBIDDEN_FEATURE_FIELDS:
        return False
    return not any(name.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)


def build_filter_rows(edge_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for edge_row in edge_rows:
        valid = 1 if _int_value(edge_row, "geometry_valid_label") > 0 else 0
        invalid = 1 if _int_value(edge_row, "geometry_invalid_label", default=1 - valid) > 0 else 0
        hard_negative = 1 if _int_value(edge_row, "geometry_hard_negative_label") > 0 else 0
        output_row = {field: edge_row.get(field, "") for field in IDENTITY_FIELDS}
        output_row.update({field: edge_row.get(field, "") for field in GEOMETRY_FIELDS})
        output_row.update(
            {
                "reject_label": str(invalid),
                "hard_negative_label": str(hard_negative),
                "pfm_matches": "1",
                "pfm_correct": str(valid),
                "pfm_wrong": str(invalid),
                "pfm_precision": "1.0" if valid else "0.0",
            }
        )
        for field, value in edge_row.items():
            if _is_allowed_feature(field):
                output_row[field] = value
        rows.append(output_row)
    return rows


def repeat_hard_negative_rows(rows: Sequence[dict[str, str]], *, repeat: int) -> list[dict[str, str]]:
    if repeat <= 0:
        raise ValueError("hard_negative_repeat must be positive")
    if repeat == 1:
        return list(rows)
    augmented: list[dict[str, str]] = []
    for row in rows:
        copies = repeat if _int_value(row, "hard_negative_label") > 0 else 1
        for _ in range(copies):
            augmented.append(dict(row))
    return augmented


def select_geometry_feature_columns(rows: Sequence[dict[str, str]], pattern: str = "") -> list[str]:
    feature_columns = [name for name in select_feature_columns(rows) if _is_allowed_feature(name)]
    return filter_feature_columns(feature_columns, pattern)


def _candidate_feature_columns(rows: Sequence[dict[str, str]], pattern: str) -> list[str]:
    feature_columns = [name for name in select_feature_columns(rows) if _is_allowed_feature(name)]
    return filter_feature_columns(feature_columns, pattern)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_predictions(path: Path, rows: Sequence[dict[str, str]], scores: Sequence[float], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row, score in zip(rows, scores):
            payload = {field: row.get(field, "") for field in PREDICTION_FIELDS}
            payload["reject_probability"] = _fmt_float(score)
            payload["predicted_reject"] = "1" if score >= threshold else "0"
            writer.writerow(payload)


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


def _path_payload(paths: Sequence[Path]) -> str | list[str]:
    if len(paths) == 1:
        return str(paths[0])
    return [str(path) for path in paths]


def _write_html(
    path: Path,
    *,
    train_geometry_edges: Sequence[Path],
    eval_geometry_edges: Sequence[Path],
    predictions_csv: Path,
    sweep_csv: Path,
    summary: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Geometry edge filter calibrator</title>",
                "<h1>Geometry edge filter calibrator</h1>",
                "<p>source=<code>true depth/camera geometry labels; inference features only</code></p>",
                f"<p>train_geometry_edges={html.escape(json.dumps(_path_payload(train_geometry_edges), ensure_ascii=False))}</p>",
                f"<p>eval_geometry_edges={html.escape(json.dumps(_path_payload(eval_geometry_edges), ensure_ascii=False))}</p>",
                f"<p>predictions_csv={html.escape(str(predictions_csv))}</p>",
                f"<p>threshold_sweep_csv={html.escape(str(sweep_csv))}</p>",
                "<h2>Summary</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def train_geometry_edge_filter(
    *,
    train_geometry_edges: Sequence[Path],
    eval_geometry_edges: Sequence[Path],
    output_dir: Path,
    epochs: int = 120,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    min_kept_valid_ratio: float = 0.99,
    threshold_objective: str = "kept_valid_ratio",
    threshold_selection_source: str = "train",
    max_kept_wrong: int | None = None,
    max_train_rows: int = 0,
    balance_sampling_key: str = "",
    hard_negative_repeat: int = 1,
    max_thresholds: int = 500,
    feature_name_regex: str = "",
) -> dict[str, object]:
    train_source_rows = build_filter_rows(_read_csv_rows_many(train_geometry_edges))
    train_augmented_rows = repeat_hard_negative_rows(train_source_rows, repeat=hard_negative_repeat)
    train_rows = limit_training_rows(
        train_augmented_rows,
        max_train_rows,
        balance_key=balance_sampling_key,
    )
    eval_rows = build_filter_rows(_read_csv_rows_many(eval_geometry_edges))
    feature_columns = _candidate_feature_columns(train_rows, feature_name_regex)
    model = train_model(
        train_rows,
        feature_columns=feature_columns,
        label_column="reject_label",
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    train_scores = score_rows(model, train_rows)
    eval_scores = score_rows(model, eval_rows)
    if threshold_selection_source == "eval":
        threshold_rows = eval_rows
        threshold_scores = eval_scores
    else:
        threshold_rows = train_rows
        threshold_scores = train_scores
    wrong_cap = (
        int(max_kept_wrong)
        if max_kept_wrong is not None
        else sum(_int_value(row, "pfm_wrong") for row in threshold_rows)
    )
    if wrong_cap < 0:
        raise ValueError("max_kept_wrong must be nonnegative")
    if threshold_objective == "pfm_wrong_cap":
        threshold = choose_match_wrong_cap_threshold(
            threshold_rows,
            threshold_scores,
            max_kept_wrong=wrong_cap,
            max_thresholds=max_thresholds,
        )
    elif threshold_objective == "kept_valid_ratio":
        threshold = choose_match_threshold(
            threshold_rows,
            threshold_scores,
            min_kept_correct_ratio=min_kept_valid_ratio,
            max_thresholds=max_thresholds,
        )
    else:
        raise ValueError(f"unsupported threshold_objective: {threshold_objective}")

    train_summary = summarize_threshold(train_rows, train_scores, threshold=threshold)
    eval_summary = summarize_threshold(eval_rows, eval_scores, threshold=threshold)
    threshold_selection_summary = summarize_threshold(threshold_rows, threshold_scores, threshold=threshold)
    sweep = build_threshold_sweep(
        train_rows,
        train_scores,
        eval_rows,
        eval_scores,
        max_thresholds=max_thresholds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_json = output_dir / "model.json"
    summary_json = output_dir / "summary.json"
    predictions_csv = output_dir / "geometry_edge_predictions.csv"
    sweep_csv = output_dir / "threshold_sweep.csv"
    output_html = output_dir / "index.html"
    model_payload = {
        "type": "standardized_logistic_regression_match_filter",
        "feature_columns": model.feature_columns,
        "means": model.means,
        "scales": model.scales,
        "weights": model.weights,
        "bias": model.bias,
        "label_column": model.label_column,
        "threshold": threshold,
        "include_true_geometry_features": False,
        "training_source": "geometry_edge_supervision",
    }
    summary = {
        "train_geometry_edges": _path_payload(train_geometry_edges),
        "eval_geometry_edges": _path_payload(eval_geometry_edges),
        "train_source_matches": len(train_source_rows),
        "train_augmented_matches": len(train_augmented_rows),
        "hard_negative_repeat": int(hard_negative_repeat),
        "max_train_rows": int(max_train_rows),
        "balance_sampling_key": balance_sampling_key,
        "feature_name_regex": feature_name_regex,
        "max_thresholds": int(max_thresholds),
        "feature_columns": model.feature_columns,
        "include_true_geometry_features": False,
        "threshold_objective": threshold_objective,
        "threshold_selection_source": threshold_selection_source,
        "max_kept_wrong": wrong_cap,
        "threshold": threshold,
        "threshold_selection": threshold_selection_summary,
        "train": train_summary,
        "eval": eval_summary,
    }
    _write_json(model_json, model_payload)
    _write_json(summary_json, summary)
    _write_predictions(predictions_csv, eval_rows, eval_scores, threshold)
    _write_sweep(sweep_csv, sweep)
    _write_html(
        output_html,
        train_geometry_edges=train_geometry_edges,
        eval_geometry_edges=eval_geometry_edges,
        predictions_csv=predictions_csv,
        sweep_csv=sweep_csv,
        summary=summary,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-geometry-edges", type=Path, action="append", required=True)
    parser.add_argument("--eval-geometry-edges", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--min-kept-valid-ratio", type=float, default=0.99)
    parser.add_argument(
        "--threshold-objective",
        choices=["kept_valid_ratio", "pfm_wrong_cap"],
        default="kept_valid_ratio",
    )
    parser.add_argument(
        "--threshold-selection-source",
        choices=["train", "eval"],
        default="train",
    )
    parser.add_argument("--max-kept-wrong", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--balance-sampling-key", default="")
    parser.add_argument("--hard-negative-repeat", type=int, default=1)
    parser.add_argument("--max-thresholds", type=int, default=500)
    parser.add_argument(
        "--feature-name-regex",
        default="",
        help="Optional regular expression used to keep only matching feature columns.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = train_geometry_edge_filter(
        train_geometry_edges=list(args.train_geometry_edges),
        eval_geometry_edges=list(args.eval_geometry_edges),
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
        min_kept_valid_ratio=float(args.min_kept_valid_ratio),
        threshold_objective=str(args.threshold_objective),
        threshold_selection_source=str(args.threshold_selection_source),
        max_kept_wrong=args.max_kept_wrong,
        max_train_rows=int(args.max_train_rows),
        balance_sampling_key=str(args.balance_sampling_key),
        hard_negative_repeat=int(args.hard_negative_repeat),
        max_thresholds=int(args.max_thresholds),
        feature_name_regex=str(args.feature_name_regex),
    )
    print(
        "geometry_edge_filter "
        f"train_matches={summary['train']['matches']} eval_matches={summary['eval']['matches']} "
        f"threshold={float(summary['threshold']):.6f} "
        f"eval_kept_correct={summary['eval']['kept_correct']} "
        f"eval_kept_wrong={summary['eval']['kept_wrong']} output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
