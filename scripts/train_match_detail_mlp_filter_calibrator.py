#!/usr/bin/env python3
"""Train a nonlinear per-match filter from true-geometry match labels."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from train_match_detail_filter_calibrator import (
    PREDICTION_FIELDS,
    SWEEP_FIELDS,
    build_threshold_sweep,
    build_training_rows,
    choose_match_threshold,
    choose_match_wrong_cap_threshold,
    filter_feature_columns,
    limit_training_rows,
    select_match_feature_columns,
    summarize_threshold,
)
from train_match_set_rejection_calibrator import _float_value


@dataclass(frozen=True)
class MlpModel:
    model_type: str
    feature_columns: list[str]
    means: list[float]
    scales: list[float]
    hidden_dim: int
    layer1_weight: list[list[float]]
    layer1_bias: list[float]
    layer2_weight: list[float]
    layer2_bias: float
    label_column: str
    threshold: float = 1.000001
    include_true_geometry_features: bool = False


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_csv_rows_many(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_csv_rows(path))
    return rows


def _labels(rows: Sequence[dict[str, str]], label_column: str) -> list[float]:
    labels = [1.0 if _float_value(row, label_column) >= 0.5 else 0.0 for row in rows]
    if not labels:
        raise ValueError("training rows are empty")
    unique = {int(value) for value in labels}
    if not unique.issubset({0, 1}) or len(unique) < 2:
        raise ValueError(f"{label_column} must contain both binary classes")
    return labels


def _fit_standardizer(rows: Sequence[dict[str, str]], feature_columns: Sequence[str]) -> tuple[list[float], list[float]]:
    if not feature_columns:
        raise ValueError("no feature columns found")
    means: list[float] = []
    scales: list[float] = []
    for name in feature_columns:
        values = [_float_value(row, name) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale > 1.0e-12 else 1.0)
    return means, scales


def _matrix(
    rows: Sequence[dict[str, str]],
    feature_columns: Sequence[str],
    means: Sequence[float],
    scales: Sequence[float],
) -> torch.Tensor:
    values = [
        [
            (_float_value(row, name) - means[index]) / scales[index]
            for index, name in enumerate(feature_columns)
        ]
        for row in rows
    ]
    return torch.tensor(values, dtype=torch.float32)


class _TinyMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(1)


def train_mlp_model(
    rows: Sequence[dict[str, str]],
    *,
    feature_columns: Sequence[str],
    label_column: str = "reject_label",
    hidden_dim: int = 32,
    epochs: int = 240,
    learning_rate: float = 0.01,
    l2: float = 0.0005,
    seed: int = 20260621,
    include_true_geometry_features: bool = False,
) -> MlpModel:
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    if l2 < 0.0 or not math.isfinite(l2):
        raise ValueError("l2 must be finite and nonnegative")
    labels = _labels(rows, label_column)
    means, scales = _fit_standardizer(rows, feature_columns)
    features = _matrix(rows, feature_columns, means, scales)
    targets = torch.tensor(labels, dtype=torch.float32)
    positive_count = float(sum(1 for value in labels if value >= 0.5))
    negative_count = float(len(labels) - int(positive_count))
    weights = torch.tensor(
        [
            len(labels) / (2.0 * positive_count) if value >= 0.5 else len(labels) / (2.0 * negative_count)
            for value in labels
        ],
        dtype=torch.float32,
    )
    torch.manual_seed(int(seed))
    network = _TinyMlp(features.size(1), int(hidden_dim))
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(learning_rate), weight_decay=float(l2))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = network(features)
        losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = (losses * weights).mean()
        loss.backward()
        optimizer.step()

    layer1 = network.net[0]
    layer2 = network.net[2]
    assert isinstance(layer1, nn.Linear)
    assert isinstance(layer2, nn.Linear)
    return MlpModel(
        model_type="standardized_mlp_match_filter",
        feature_columns=list(feature_columns),
        means=means,
        scales=scales,
        hidden_dim=int(hidden_dim),
        layer1_weight=layer1.weight.detach().cpu().tolist(),
        layer1_bias=layer1.bias.detach().cpu().tolist(),
        layer2_weight=layer2.weight.detach().cpu().reshape(-1).tolist(),
        layer2_bias=float(layer2.bias.detach().cpu().reshape(())),
        label_column=label_column,
        include_true_geometry_features=bool(include_true_geometry_features),
    )


def score_rows(model: MlpModel, rows: Sequence[dict[str, str]]) -> list[float]:
    features = _matrix(rows, model.feature_columns, model.means, model.scales)
    layer1_weight = torch.tensor(model.layer1_weight, dtype=torch.float32)
    layer1_bias = torch.tensor(model.layer1_bias, dtype=torch.float32)
    layer2_weight = torch.tensor(model.layer2_weight, dtype=torch.float32)
    layer2_bias = torch.tensor(model.layer2_bias, dtype=torch.float32)
    hidden = torch.relu(features @ layer1_weight.t() + layer1_bias)
    logits = hidden @ layer2_weight + layer2_bias
    return torch.sigmoid(logits).detach().cpu().tolist()


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _path_payload(paths: Sequence[Path]) -> str | list[str]:
    if len(paths) == 1:
        return str(paths[0])
    return [str(path) for path in paths]


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
            payload["reject_probability"] = _fmt_float(float(score))
            payload["predicted_reject"] = "1" if float(score) >= threshold else "0"
            writer.writerow(payload)


def _write_sweep(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = {
                key: _fmt_float(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(payload)


def _model_payload(model: MlpModel, *, threshold: float) -> dict[str, object]:
    return {
        "type": model.model_type,
        "feature_columns": model.feature_columns,
        "means": model.means,
        "scales": model.scales,
        "hidden_dim": model.hidden_dim,
        "layer1_weight": model.layer1_weight,
        "layer1_bias": model.layer1_bias,
        "layer2_weight": model.layer2_weight,
        "layer2_bias": model.layer2_bias,
        "label_column": model.label_column,
        "threshold": threshold,
        "include_true_geometry_features": model.include_true_geometry_features,
    }


def _write_report_html(
    path: Path,
    *,
    train_match_details: Sequence[Path],
    eval_match_details: Sequence[Path],
    summary: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Match-detail MLP filter calibrator</title>",
                "<h1>Match-detail MLP filter calibrator</h1>",
                f"<p>train_match_details={html.escape(json.dumps(_path_payload(train_match_details), ensure_ascii=False))}</p>",
                f"<p>eval_match_details={html.escape(json.dumps(_path_payload(eval_match_details), ensure_ascii=False))}</p>",
                "<h2>Summary</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-match-details", type=Path, action="append", required=True)
    parser.add_argument("--eval-match-details", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--min-kept-correct-ratio", type=float, default=0.99)
    parser.add_argument(
        "--threshold-objective",
        choices=["kept_correct_ratio", "pfm_wrong_cap"],
        default="kept_correct_ratio",
    )
    parser.add_argument(
        "--threshold-selection-source",
        choices=["train", "eval"],
        default="train",
    )
    parser.add_argument("--max-kept-wrong", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--balance-sampling-key", default="")
    parser.add_argument("--max-thresholds", type=int, default=500)
    parser.add_argument("--include-true-geometry-features", action="store_true")
    parser.add_argument("--feature-name-regex", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_match_details = list(args.train_match_details)
    eval_match_details = list(args.eval_match_details)
    train_source_rows = build_training_rows(
        _read_csv_rows_many(train_match_details),
        include_true_geometry_features=bool(args.include_true_geometry_features),
    )
    train_rows = limit_training_rows(
        train_source_rows,
        int(args.max_train_rows),
        balance_key=str(args.balance_sampling_key),
    )
    eval_rows = build_training_rows(
        _read_csv_rows_many(eval_match_details),
        include_true_geometry_features=bool(args.include_true_geometry_features),
    )
    feature_columns = filter_feature_columns(
        select_match_feature_columns(train_rows),
        str(args.feature_name_regex),
    )
    model = train_mlp_model(
        train_rows,
        feature_columns=feature_columns,
        label_column="reject_label",
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
        seed=int(args.seed),
        include_true_geometry_features=bool(args.include_true_geometry_features),
    )
    train_scores = score_rows(model, train_rows)
    eval_scores = score_rows(model, eval_rows)
    threshold_rows = eval_rows if args.threshold_selection_source == "eval" else train_rows
    threshold_scores = eval_scores if args.threshold_selection_source == "eval" else train_scores
    max_kept_wrong = (
        int(args.max_kept_wrong)
        if args.max_kept_wrong is not None
        else sum(1 for row in threshold_rows if row.get("reject_label") == "1")
    )
    if args.threshold_objective == "pfm_wrong_cap":
        threshold = choose_match_wrong_cap_threshold(
            threshold_rows,
            threshold_scores,
            max_kept_wrong=max_kept_wrong,
            max_thresholds=int(args.max_thresholds),
        )
    else:
        threshold = choose_match_threshold(
            threshold_rows,
            threshold_scores,
            min_kept_correct_ratio=float(args.min_kept_correct_ratio),
            max_thresholds=int(args.max_thresholds),
        )
    train_summary = summarize_threshold(train_rows, train_scores, threshold=threshold)
    eval_summary = summarize_threshold(eval_rows, eval_scores, threshold=threshold)
    threshold_selection_summary = summarize_threshold(threshold_rows, threshold_scores, threshold=threshold)
    sweep = build_threshold_sweep(
        train_rows,
        train_scores,
        eval_rows,
        eval_scores,
        max_thresholds=int(args.max_thresholds),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_json = output_dir / "model.json"
    summary_json = output_dir / "summary.json"
    predictions_csv = output_dir / "all_match_predictions.csv"
    sweep_csv = output_dir / "threshold_sweep.csv"
    html_path = output_dir / "index.html"
    summary = {
        "train_match_details": _path_payload(train_match_details),
        "eval_match_details": _path_payload(eval_match_details),
        "train_source_matches": len(train_source_rows),
        "max_train_rows": int(args.max_train_rows),
        "balance_sampling_key": str(args.balance_sampling_key),
        "feature_name_regex": str(args.feature_name_regex),
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "l2": float(args.l2),
        "seed": int(args.seed),
        "max_thresholds": int(args.max_thresholds),
        "feature_columns": model.feature_columns,
        "include_true_geometry_features": bool(args.include_true_geometry_features),
        "threshold_objective": str(args.threshold_objective),
        "threshold_selection_source": str(args.threshold_selection_source),
        "max_kept_wrong": max_kept_wrong,
        "threshold": threshold,
        "threshold_selection": threshold_selection_summary,
        "train": train_summary,
        "eval": eval_summary,
    }
    _write_json(model_json, _model_payload(model, threshold=threshold))
    _write_json(summary_json, summary)
    _write_predictions(predictions_csv, eval_rows, eval_scores, threshold)
    _write_sweep(sweep_csv, sweep)
    _write_report_html(
        html_path,
        train_match_details=train_match_details,
        eval_match_details=eval_match_details,
        summary=summary,
    )
    print(
        "match_mlp_filter "
        f"train_matches={train_summary['matches']} eval_matches={eval_summary['matches']} "
        f"threshold={threshold:.6f} eval_kept_correct={eval_summary['kept_correct']} "
        f"eval_kept_wrong={eval_summary['kept_wrong']} output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
