#!/usr/bin/env python3
"""Run the PFM/LightGlue hybrid gate application, validation and audit."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence

import apply_match_set_rejection_calibrator as apply_calibrator
import audit_pfm_optimization_goal as optimization_audit
import validate_hybrid_lightglue_gate as hybrid_gate


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    keys = [
        "valid",
        "validation_exit_code",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "precision_delta_vs_lightglue",
        "hybrid_summary_csv",
        "hybrid_summary_json",
        "validation_json",
        "optimization_audit_json",
    ]
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(payload.get(key, '')))}</td></tr>"
        for key in keys
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Hybrid LightGlue gate pipeline</title>",
                "<h1>Hybrid LightGlue gate pipeline</h1>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>metric</th><th>value</th></tr>",
                rows,
                "</table>",
                "<h2>Pipeline JSON</h2>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--active-mainline-validation-json", type=Path, default=None)
    parser.add_argument("--selector-promotion-json", type=Path, default=None)
    parser.add_argument("--train-metrics-csv", type=Path, default=None)
    parser.add_argument("--min-correct-delta-vs-lightglue", type=int, default=1)
    parser.add_argument("--max-wrong-delta-vs-lightglue", type=int, default=0)
    parser.add_argument("--min-precision-delta-vs-lightglue", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hybrid_summary_csv = output_dir / "hybrid_summary.csv"
    hybrid_summary_json = output_dir / "summary.json"
    application_html = output_dir / "application.html"
    validation_json = output_dir / "validation.json"
    validation_html = output_dir / "validation.html"
    audit_json = output_dir / "optimization_audit.json"
    audit_html = output_dir / "optimization_audit.html"
    pipeline_summary_json = output_dir / "pipeline_summary.json"
    pipeline_html = output_dir / "index.html"

    apply_exit_code = apply_calibrator.main(
        [
            "--dataset-csv",
            str(args.dataset_csv),
            "--model-json",
            str(args.model_json),
            "--output-csv",
            str(hybrid_summary_csv),
            "--summary-json",
            str(hybrid_summary_json),
            "--output-html",
            str(application_html),
        ]
    )

    validation_exit_code = hybrid_gate.main(
        [
            "--summary-json",
            str(hybrid_summary_json),
            "--output-json",
            str(validation_json),
            "--output-html",
            str(validation_html),
            "--min-correct-delta-vs-lightglue",
            str(args.min_correct_delta_vs_lightglue),
            "--max-wrong-delta-vs-lightglue",
            str(args.max_wrong_delta_vs_lightglue),
            "--min-precision-delta-vs-lightglue",
            str(args.min_precision_delta_vs_lightglue),
        ]
    )

    audit_items = optimization_audit.audit_goal(
        project_root=args.project_root,
        selector_promotion_json=args.selector_promotion_json,
        active_mainline_validation_json=args.active_mainline_validation_json,
        hybrid_lightglue_validation_json=validation_json,
        train_metrics_csv=args.train_metrics_csv,
    )
    optimization_audit.write_json(audit_json, audit_items)
    optimization_audit.write_html(audit_html, audit_items)

    validation = _load_json_object(validation_json)
    summary = {
        "valid": bool(validation.get("valid")) and apply_exit_code == 0 and validation_exit_code == 0,
        "apply_exit_code": apply_exit_code,
        "validation_exit_code": validation_exit_code,
        "dataset_csv": str(args.dataset_csv),
        "model_json": str(args.model_json),
        "output_dir": str(output_dir),
        "hybrid_summary_csv": str(hybrid_summary_csv),
        "hybrid_summary_json": str(hybrid_summary_json),
        "application_html": str(application_html),
        "validation_json": str(validation_json),
        "validation_html": str(validation_html),
        "optimization_audit_json": str(audit_json),
        "optimization_audit_html": str(audit_html),
        "correct_delta_vs_lightglue": validation.get("correct_delta_vs_lightglue"),
        "wrong_delta_vs_lightglue": validation.get("wrong_delta_vs_lightglue"),
        "precision_delta_vs_lightglue": validation.get("precision_delta_vs_lightglue"),
    }
    _write_json(pipeline_summary_json, summary)
    _write_html(pipeline_html, summary)

    print(
        f"hybrid_pipeline_valid={summary['valid']} "
        f"correct_delta={summary['correct_delta_vs_lightglue']} "
        f"wrong_delta={summary['wrong_delta_vs_lightglue']} "
        f"output={output_dir}",
        flush=True,
    )
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
