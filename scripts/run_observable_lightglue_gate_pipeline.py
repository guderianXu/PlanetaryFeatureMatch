#!/usr/bin/env python3
"""Run observable PFM/LightGlue gate application, validation and audit."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import apply_observable_pair_gate_match_filter as observable_gate
import audit_pfm_optimization_goal as optimization_audit
import validate_hybrid_lightglue_gate as hybrid_gate


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_csv: Path
    match_details: Path


def _parse_source(value: str) -> SourceSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--source must have the form name,dataset_csv,match_details_csv"
        )
    name, dataset_csv, match_details = parts
    if not name:
        raise argparse.ArgumentTypeError("source name must not be empty")
    return SourceSpec(name=name, dataset_csv=Path(dataset_csv), match_details=Path(match_details))


def _int_value(data: dict[str, Any], key: str, default: int = 0) -> int:
    value = data.get(key, default)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _float_value(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary_path(output_dir: Path, source_name: str) -> Path:
    return output_dir / source_name / "summary.json"


def _validation_path(output_dir: Path, source_name: str) -> Path:
    return output_dir / f"{source_name}_validation.json"


def _validation_html_path(output_dir: Path, source_name: str) -> Path:
    return output_dir / f"{source_name}_validation.html"


def _validate_sources(sources: Sequence[SourceSpec]) -> None:
    if not sources:
        raise ValueError("at least one --source is required")
    seen: set[str] = set()
    for source in sources:
        if source.name in seen:
            raise ValueError(f"duplicate source name: {source.name}")
        seen.add(source.name)
        if not source.dataset_csv.exists():
            raise FileNotFoundError(f"missing dataset CSV for {source.name}: {source.dataset_csv}")
        if not source.match_details.exists():
            raise FileNotFoundError(f"missing match details CSV for {source.name}: {source.match_details}")


def _aggregate_summaries(summaries: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "rows": sum(_int_value(summary, "rows") for _name, summary in summaries),
        "kept_pfm_rows": sum(_int_value(summary, "kept_pfm_rows") for _name, summary in summaries),
        "fallback_lightglue_rows": sum(_int_value(summary, "fallback_lightglue_rows") for _name, summary in summaries),
        "rejected_rows": sum(_int_value(summary, "rejected_rows") for _name, summary in summaries),
        "pfm_matches": sum(_int_value(summary, "pfm_matches") for _name, summary in summaries),
        "pfm_correct": sum(_int_value(summary, "pfm_correct") for _name, summary in summaries),
        "pfm_wrong": sum(_int_value(summary, "pfm_wrong") for _name, summary in summaries),
        "lightglue_matches": sum(_int_value(summary, "lightglue_matches") for _name, summary in summaries),
        "lightglue_correct": sum(_int_value(summary, "lightglue_correct") for _name, summary in summaries),
        "lightglue_wrong": sum(_int_value(summary, "lightglue_wrong") for _name, summary in summaries),
        "hybrid_matches": sum(_int_value(summary, "hybrid_matches") for _name, summary in summaries),
        "hybrid_correct": sum(_int_value(summary, "hybrid_correct") for _name, summary in summaries),
        "hybrid_wrong": sum(_int_value(summary, "hybrid_wrong") for _name, summary in summaries),
        "by_source": {name: summary for name, summary in summaries},
    }
    aggregate["pfm_precision"] = (
        aggregate["pfm_correct"] / aggregate["pfm_matches"] if aggregate["pfm_matches"] else 0.0
    )
    aggregate["lightglue_precision"] = (
        aggregate["lightglue_correct"] / aggregate["lightglue_matches"]
        if aggregate["lightglue_matches"]
        else 0.0
    )
    aggregate["hybrid_precision"] = (
        aggregate["hybrid_correct"] / aggregate["hybrid_matches"] if aggregate["hybrid_matches"] else 0.0
    )
    aggregate["correct_delta_vs_lightglue"] = aggregate["hybrid_correct"] - aggregate["lightglue_correct"]
    aggregate["wrong_delta_vs_lightglue"] = aggregate["hybrid_wrong"] - aggregate["lightglue_wrong"]
    aggregate["precision_delta_vs_lightglue"] = aggregate["hybrid_precision"] - aggregate["lightglue_precision"]
    aggregate["hybrid_correct_delta_vs_lightglue"] = aggregate["correct_delta_vs_lightglue"]
    aggregate["hybrid_wrong_delta_vs_lightglue"] = aggregate["wrong_delta_vs_lightglue"]
    aggregate["hybrid_precision_delta_vs_lightglue"] = aggregate["precision_delta_vs_lightglue"]
    return aggregate


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_index_html(path: Path, payload: dict[str, Any]) -> None:
    validation_fields = [
        "source",
        "valid",
        "lightglue_correct",
        "lightglue_wrong",
        "hybrid_correct",
        "hybrid_wrong",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "precision_delta_vs_lightglue",
    ]
    rows = []
    for item in payload.get("split_validations", []):
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(item.get(field, '')))}</td>"
                for field in validation_fields
            )
            + "</tr>"
        )
    header = "".join(f"<th>{html.escape(field)}</th>" for field in validation_fields)
    keys = [
        "valid",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "precision_delta_vs_lightglue",
        "aggregate_summary_json",
        "aggregate_validation_json",
        "optimization_audit_json",
    ]
    summary_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(payload.get(key, '')))}</td></tr>"
        for key in keys
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Observable LightGlue gate pipeline</title>",
                "<h1>Observable LightGlue gate pipeline</h1>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                summary_rows,
                "</table>",
                "<h2>Per-source validation</h2>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                f"<tr>{header}</tr>",
                *rows,
                "</table>",
                "<h2>Pipeline JSON</h2>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_pipeline(
    *,
    sources: Sequence[SourceSpec],
    gate: str,
    output_dir: Path,
    project_root: Path,
    max_homography_residual_px: float | None,
    min_correct_delta_vs_lightglue: int,
    max_wrong_delta_vs_lightglue: int,
    min_precision_delta_vs_lightglue: float,
    active_mainline_validation_json: Path | None = None,
    selector_promotion_json: Path | None = None,
    train_metrics_csv: Path | None = None,
    candidate_checkpoint: str = "",
) -> dict[str, Any]:
    _validate_sources(sources)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[tuple[str, dict[str, Any]]] = []
    split_validations: list[dict[str, Any]] = []
    for source in sources:
        source_dir = output_dir / source.name
        summary = observable_gate.apply_gate_match_filter(
            dataset_csv=source.dataset_csv,
            match_details=[source.match_details],
            gate=gate,
            output_dir=source_dir,
            max_homography_residual_px=max_homography_residual_px,
        )
        summaries.append((source.name, summary))

        validation = hybrid_gate.validate_summary(
            summary,
            min_correct_delta_vs_lightglue=min_correct_delta_vs_lightglue,
            max_wrong_delta_vs_lightglue=max_wrong_delta_vs_lightglue,
            min_precision_delta_vs_lightglue=min_precision_delta_vs_lightglue,
        )
        validation["source"] = source.name
        validation["summary_json"] = str(_summary_path(output_dir, source.name))
        validation["validation_json"] = str(_validation_path(output_dir, source.name))
        validation["validation_html"] = str(_validation_html_path(output_dir, source.name))
        hybrid_gate._write_json(_validation_path(output_dir, source.name), validation)
        hybrid_gate._write_html(
            _validation_html_path(output_dir, source.name),
            summary_json=_summary_path(output_dir, source.name),
            payload=validation,
        )
        split_validations.append(validation)

    aggregate_summary = _aggregate_summaries(summaries)
    aggregate_summary["gate"] = gate
    aggregate_summary["max_homography_residual_px"] = max_homography_residual_px
    aggregate_summary["candidate_checkpoint"] = candidate_checkpoint
    aggregate_summary_json = output_dir / "aggregate_summary.json"
    _write_json(aggregate_summary_json, aggregate_summary)

    aggregate_validation = hybrid_gate.validate_summary(
        aggregate_summary,
        min_correct_delta_vs_lightglue=min_correct_delta_vs_lightglue,
        max_wrong_delta_vs_lightglue=max_wrong_delta_vs_lightglue,
        min_precision_delta_vs_lightglue=min_precision_delta_vs_lightglue,
    )
    aggregate_validation["type"] = "observable_lightglue_gate_aggregate_validation"
    aggregate_validation["splits"] = split_validations
    aggregate_validation["errors"] = list(aggregate_validation.get("errors", []))
    for item in split_validations:
        if not bool(item.get("valid")):
            aggregate_validation["errors"].append(f"{item['source']}:{','.join(item.get('errors', []))}")
    aggregate_validation["valid"] = bool(aggregate_validation.get("valid")) and not aggregate_validation["errors"]
    aggregate_validation["gate"] = gate
    aggregate_validation["candidate_checkpoint"] = candidate_checkpoint
    aggregate_validation_json = output_dir / "aggregate_validation.json"
    aggregate_validation_html = output_dir / "aggregate_validation.html"
    hybrid_gate._write_json(aggregate_validation_json, aggregate_validation)
    hybrid_gate._write_html(
        aggregate_validation_html,
        summary_json=aggregate_summary_json,
        payload=aggregate_validation,
    )

    validation_fields = [
        "source",
        "valid",
        "lightglue_correct",
        "lightglue_wrong",
        "hybrid_correct",
        "hybrid_wrong",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "precision_delta_vs_lightglue",
        "summary_json",
        "validation_json",
    ]
    _write_csv(output_dir / "per_split_validation.csv", split_validations, validation_fields)

    audit_items = optimization_audit.audit_goal(
        project_root=project_root,
        selector_promotion_json=selector_promotion_json,
        active_mainline_validation_json=active_mainline_validation_json,
        hybrid_lightglue_validation_json=aggregate_validation_json,
        train_metrics_csv=train_metrics_csv,
    )
    audit_json = output_dir / "optimization_audit.json"
    audit_html = output_dir / "optimization_audit.html"
    optimization_audit.write_json(audit_json, audit_items)
    optimization_audit.write_html(audit_html, audit_items)

    pipeline_summary = {
        "valid": bool(aggregate_validation.get("valid")),
        "source_count": len(sources),
        "gate": gate,
        "max_homography_residual_px": max_homography_residual_px,
        "candidate_checkpoint": candidate_checkpoint,
        "output_dir": str(output_dir),
        "hybrid_summary_json": str(aggregate_summary_json),
        "aggregate_summary_json": str(aggregate_summary_json),
        "validation_json": str(aggregate_validation_json),
        "aggregate_validation_json": str(aggregate_validation_json),
        "optimization_audit_json": str(audit_json),
        "optimization_audit_html": str(audit_html),
        "correct_delta_vs_lightglue": aggregate_validation.get("correct_delta_vs_lightglue"),
        "wrong_delta_vs_lightglue": aggregate_validation.get("wrong_delta_vs_lightglue"),
        "precision_delta_vs_lightglue": aggregate_validation.get("precision_delta_vs_lightglue"),
        "split_validations": split_validations,
    }
    _write_json(output_dir / "pipeline_summary.json", pipeline_summary)
    _write_index_html(output_dir / "index.html", pipeline_summary)
    return pipeline_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=_parse_source, action="append", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-homography-residual-px", type=float, default=None)
    parser.add_argument("--active-mainline-validation-json", type=Path, default=None)
    parser.add_argument("--selector-promotion-json", type=Path, default=None)
    parser.add_argument("--train-metrics-csv", type=Path, default=None)
    parser.add_argument("--candidate-checkpoint", default="")
    parser.add_argument("--min-correct-delta-vs-lightglue", type=int, default=1)
    parser.add_argument("--max-wrong-delta-vs-lightglue", type=int, default=0)
    parser.add_argument("--min-precision-delta-vs-lightglue", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_pipeline(
        sources=args.source,
        gate=str(args.gate),
        output_dir=args.output_dir,
        project_root=args.project_root,
        max_homography_residual_px=args.max_homography_residual_px,
        min_correct_delta_vs_lightglue=int(args.min_correct_delta_vs_lightglue),
        max_wrong_delta_vs_lightglue=int(args.max_wrong_delta_vs_lightglue),
        min_precision_delta_vs_lightglue=float(args.min_precision_delta_vs_lightglue),
        active_mainline_validation_json=args.active_mainline_validation_json,
        selector_promotion_json=args.selector_promotion_json,
        train_metrics_csv=args.train_metrics_csv,
        candidate_checkpoint=str(args.candidate_checkpoint),
    )
    print(
        f"observable_hybrid_pipeline_valid={summary['valid']} "
        f"correct_delta={summary['correct_delta_vs_lightglue']} "
        f"wrong_delta={summary['wrong_delta_vs_lightglue']} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
