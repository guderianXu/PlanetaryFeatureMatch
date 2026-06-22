#!/usr/bin/env python3
"""Run true-geometry selector, validation and optimization audit as one pipeline."""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import audit_pfm_optimization_goal as optimization_audit
import select_true_geometry_pair_reports as selector
import validate_true_geometry_selector as selector_validation


@dataclass(frozen=True)
class PlanStep:
    name: str
    command: list[str]
    inputs: list[str]
    outputs: list[str]


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
        "status",
        "valid",
        "error",
        "selector_summary_json",
        "selector_pair_selection_csv",
        "validation_json",
        "validation_html",
        "optimization_audit_json",
        "optimization_audit_html",
    ]
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(payload.get(key, '')))}</td></tr>"
        for key in keys
    )
    step_rows = []
    for step in payload.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_rows.append(
            "<tr>"
            f"<td>{html.escape(str(step.get('name', '')))}</td>"
            f"<td><code>{html.escape(' '.join(str(item) for item in step.get('command', [])))}</code></td>"
            f"<td>{html.escape(', '.join(str(item) for item in step.get('outputs', [])))}</td>"
            "</tr>"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-CN">',
                '<head><meta charset="utf-8"><title>True geometry selector pipeline</title></head>',
                "<body>",
                "<h1>True geometry selector pipeline</h1>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>metric</th><th>value</th></tr>",
                rows,
                "</table>",
                "<h2>Steps</h2>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>name</th><th>command</th><th>outputs</th></tr>",
                *step_rows,
                "</table>",
                "<h2>Pipeline JSON</h2>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
                "</body>",
                "</html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _source_arg(source: selector.SelectionSource) -> str:
    parts = [source.split, str(source.pair_manifest)]
    if source.lightglue_metrics is not None:
        parts.append(str(source.lightglue_metrics))
    return ",".join(parts)


def _candidate_arg(candidate: selector.CandidateReport) -> str:
    return ",".join([candidate.name, str(candidate.root), candidate.eval_subdir])


def _script(name: str, project_root: Path) -> str:
    return str(project_root / "scripts" / name)


def _selector_command(
    *,
    project_root: Path,
    sources: Sequence[selector.SelectionSource],
    candidates: Sequence[selector.CandidateReport],
    output_dir: Path,
    lightglue_label: str,
    selection_rank_profile: str,
) -> list[str]:
    command = [sys.executable, _script("select_true_geometry_pair_reports.py", project_root)]
    for source in sources:
        command.extend(["--source", _source_arg(source)])
    for candidate in candidates:
        command.extend(["--candidate", _candidate_arg(candidate)])
    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--lightglue-label",
            lightglue_label,
            "--selection-rank-profile",
            selection_rank_profile,
        ]
    )
    return command


def _validation_command(
    *,
    project_root: Path,
    summary_json: Path,
    manifest_validation_json: Path | None,
    output_json: Path,
    output_html: Path,
    min_rows: int,
    min_correct_delta_vs_lightglue: int,
    max_wrong_delta_vs_lightglue: int,
    min_precision_delta_vs_lightglue: float,
    required_splits: Sequence[str],
    required_variants: Sequence[str],
    require_base_disjoint: bool,
) -> list[str]:
    command = [
        sys.executable,
        _script("validate_true_geometry_selector.py", project_root),
        "--summary-json",
        str(summary_json),
        "--output-json",
        str(output_json),
        "--output-html",
        str(output_html),
        "--min-rows",
        str(min_rows),
        "--min-correct-delta-vs-lightglue",
        str(min_correct_delta_vs_lightglue),
        "--max-wrong-delta-vs-lightglue",
        str(max_wrong_delta_vs_lightglue),
        "--min-precision-delta-vs-lightglue",
        str(min_precision_delta_vs_lightglue),
    ]
    if manifest_validation_json is not None:
        command.extend(["--manifest-validation-json", str(manifest_validation_json)])
    for split in required_splits:
        command.extend(["--required-split", split])
    for variant in required_variants:
        command.extend(["--required-variant", variant])
    if not require_base_disjoint:
        command.append("--no-require-base-disjoint")
    return command


def _audit_command(
    *,
    project_root: Path,
    selector_summary_json: Path,
    manifest_validation_json: Path | None,
    output_json: Path,
    output_html: Path,
) -> list[str]:
    command = [
        sys.executable,
        _script("audit_pfm_optimization_goal.py", project_root),
        "--project-root",
        str(project_root),
        "--true-geometry-selector-summary-json",
        str(selector_summary_json),
        "--output-json",
        str(output_json),
        "--output-html",
        str(output_html),
    ]
    if manifest_validation_json is not None:
        command.extend(["--true-geometry-manifest-validation-json", str(manifest_validation_json)])
    return command


def _validate_pipeline_inputs(
    sources: Sequence[selector.SelectionSource],
    candidates: Sequence[selector.CandidateReport],
) -> None:
    if not sources:
        raise ValueError("at least one --source is required")
    if not candidates:
        raise ValueError("at least one --candidate is required")
    for source in sources:
        if source.lightglue_metrics is None:
            raise ValueError(
                "--source must include LightGlue metrics for validation: "
                "split,pair_manifest,lightglue_sift_metrics.csv"
            )
    selector.ensure_inputs_exist(sources, candidates)


def _visual_input_paths(
    sources: Sequence[selector.SelectionSource],
    candidates: Sequence[selector.CandidateReport],
) -> list[str]:
    paths: list[str] = []
    for source in sources:
        paths.append(str(source.pair_manifest))
        if source.lightglue_metrics is not None:
            paths.append(str(source.lightglue_metrics))
        for candidate in candidates:
            paths.append(str(candidate.summary_path(source.split)))
            paths.append(str(candidate.details_path(source.split)))
    return paths


def _build_steps(
    args: argparse.Namespace,
    required_splits: Sequence[str],
    required_variants: Sequence[str],
) -> list[PlanStep]:
    selector_dir = args.output_root / "selector"
    selector_summary_json = selector_dir / "summary.json"
    selector_pair_selection_csv = selector_dir / "pair_selection.csv"
    audit_json = args.audit_output_json or args.output_root / "optimization_audit.json"
    audit_html = args.audit_output_html or args.output_root / "optimization_audit.html"
    visual_inputs = _visual_input_paths(args.source, args.candidate)
    visual_command = [
        "check",
        "visualize_lazy_pose_matches.py",
        "outputs",
        "--requires",
        "all_filtered_summary.csv",
        "--requires",
        "all_filtered_match_details.csv",
    ]
    return [
        PlanStep(
            name="visual_eval_input_check",
            command=visual_command,
            inputs=visual_inputs,
            outputs=visual_inputs,
        ),
        PlanStep(
            name="select_true_geometry_pair_reports",
            command=_selector_command(
                project_root=args.project_root,
                sources=args.source,
                candidates=args.candidate,
                output_dir=selector_dir,
                lightglue_label=str(args.lightglue_label),
                selection_rank_profile=str(args.selection_rank_profile),
            ),
            inputs=visual_inputs,
            outputs=[str(selector_summary_json), str(selector_pair_selection_csv)],
        ),
        PlanStep(
            name="validate_true_geometry_selector",
            command=_validation_command(
                project_root=args.project_root,
                summary_json=selector_summary_json,
                manifest_validation_json=args.manifest_validation_json,
                output_json=args.validation_output_json,
                output_html=args.validation_output_html,
                min_rows=int(args.min_rows),
                min_correct_delta_vs_lightglue=int(args.min_correct_delta_vs_lightglue),
                max_wrong_delta_vs_lightglue=int(args.max_wrong_delta_vs_lightglue),
                min_precision_delta_vs_lightglue=float(args.min_precision_delta_vs_lightglue),
                required_splits=required_splits,
                required_variants=required_variants,
                require_base_disjoint=not bool(args.no_require_base_disjoint),
            ),
            inputs=[str(selector_summary_json)],
            outputs=[str(args.validation_output_json), str(args.validation_output_html)],
        ),
        PlanStep(
            name="audit_pfm_optimization_goal",
            command=_audit_command(
                project_root=args.project_root,
                selector_summary_json=selector_summary_json,
                manifest_validation_json=args.manifest_validation_json,
                output_json=audit_json,
                output_html=audit_html,
            ),
            inputs=[str(selector_summary_json), str(args.validation_output_json)],
            outputs=[str(audit_json), str(audit_html)],
        ),
    ]


def _manifest_payload(summary: dict[str, Any], manifest_validation_json: Path | None) -> dict[str, Any]:
    if manifest_validation_json is not None:
        return _load_json_object(manifest_validation_json)
    manifest = summary.get("manifest_validation")
    return manifest if isinstance(manifest, dict) else {}


def _write_pipeline_summary(
    *,
    path_json: Path,
    path_html: Path,
    status: str,
    steps: Sequence[PlanStep],
    selector_summary: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    audit_items: list[optimization_audit.AuditItem] | None = None,
    error: str = "",
    output_root: Path,
    selector_summary_json: Path,
    selector_pair_selection_csv: Path,
    validation_json: Path,
    validation_html: Path,
    audit_json: Path,
    audit_html: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "valid": status == "valid",
        "error": error,
        "output_root": str(output_root),
        "selector_summary_json": str(selector_summary_json),
        "selector_pair_selection_csv": str(selector_pair_selection_csv),
        "validation_json": str(validation_json),
        "validation_html": str(validation_html),
        "optimization_audit_json": str(audit_json),
        "optimization_audit_html": str(audit_html),
        "steps": [asdict(step) for step in steps],
    }
    if selector_summary is not None:
        payload["selector"] = selector_summary
        aggregate = selector_summary.get("aggregate")
        if isinstance(aggregate, dict):
            payload["selector_aggregate"] = aggregate
    if validation is not None:
        payload["validation"] = validation
    if audit_items is not None:
        payload["audit"] = [asdict(item) for item in audit_items]
    _write_json(path_json, payload)
    _write_html(path_html, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=selector.parse_source, action="append", required=True)
    parser.add_argument("--candidate", type=selector.parse_candidate, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-output-json", type=Path, required=True)
    parser.add_argument("--validation-output-html", type=Path, required=True)
    parser.add_argument("--audit-output-json", type=Path, default=None)
    parser.add_argument("--audit-output-html", type=Path, default=None)
    parser.add_argument("--manifest-validation-json", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--lightglue-label", default="LightGlue-SIFT-MAGSAC-min16")
    parser.add_argument(
        "--selection-rank-profile",
        choices=["inference_safe", "diagnostic_wrong_tiebreak"],
        default="inference_safe",
    )
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-correct-delta-vs-lightglue", type=int, default=1)
    parser.add_argument("--max-wrong-delta-vs-lightglue", type=int, default=0)
    parser.add_argument("--min-precision-delta-vs-lightglue", type=float, default=0.0)
    parser.add_argument("--required-split", action="append", default=None)
    parser.add_argument("--required-variant", action="append", default=None)
    parser.add_argument("--no-require-base-disjoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    selector_dir = args.output_root / "selector"
    selector_summary_json = selector_dir / "summary.json"
    selector_pair_selection_csv = selector_dir / "pair_selection.csv"
    summary_json = args.output_root / "summary.json"
    summary_html = args.output_root / "summary.html"
    audit_json = args.audit_output_json or args.output_root / "optimization_audit.json"
    audit_html = args.audit_output_html or args.output_root / "optimization_audit.html"
    required_splits = list(args.required_split) if args.required_split else sorted({source.split for source in args.source})
    required_variants = list(args.required_variant) if args.required_variant else []
    steps = _build_steps(args, required_splits, required_variants)

    try:
        _validate_pipeline_inputs(args.source, args.candidate)
    except Exception as exc:
        _write_pipeline_summary(
            path_json=summary_json,
            path_html=summary_html,
            status="failed",
            steps=steps,
            error=str(exc),
            output_root=args.output_root,
            selector_summary_json=selector_summary_json,
            selector_pair_selection_csv=selector_pair_selection_csv,
            validation_json=args.validation_output_json,
            validation_html=args.validation_output_html,
            audit_json=audit_json,
            audit_html=audit_html,
        )
        print(f"true_geometry_selector_pipeline_status=failed error={exc}", flush=True)
        return 2

    if args.dry_run:
        _write_pipeline_summary(
            path_json=summary_json,
            path_html=summary_html,
            status="dry_run",
            steps=steps,
            output_root=args.output_root,
            selector_summary_json=selector_summary_json,
            selector_pair_selection_csv=selector_pair_selection_csv,
            validation_json=args.validation_output_json,
            validation_html=args.validation_output_html,
            audit_json=audit_json,
            audit_html=audit_html,
        )
        print(f"true_geometry_selector_pipeline_status=dry_run output={summary_json}", flush=True)
        return 0

    try:
        selector_summary = selector.select_reports(
            sources=args.source,
            candidates=args.candidate,
            output_dir=selector_dir,
            lightglue_label=str(args.lightglue_label),
            selection_rank_profile=str(args.selection_rank_profile),
        )
        manifest = _manifest_payload(selector_summary, args.manifest_validation_json)
        validation = selector_validation.validate_summary(
            selector_summary,
            manifest=manifest,
            min_rows=int(args.min_rows),
            min_correct_delta_vs_lightglue=int(args.min_correct_delta_vs_lightglue),
            max_wrong_delta_vs_lightglue=int(args.max_wrong_delta_vs_lightglue),
            min_precision_delta_vs_lightglue=float(args.min_precision_delta_vs_lightglue),
            required_splits=required_splits,
            required_variants=required_variants,
            require_base_disjoint=not bool(args.no_require_base_disjoint),
        )
        validation["summary_json"] = str(selector_summary_json)
        validation["manifest_validation_json"] = (
            str(args.manifest_validation_json) if args.manifest_validation_json is not None else ""
        )
        selector_validation._write_json(args.validation_output_json, validation)
        selector_validation._write_html(args.validation_output_html, summary_json=selector_summary_json, payload=validation)
        if not bool(validation.get("valid")):
            _write_pipeline_summary(
                path_json=summary_json,
                path_html=summary_html,
                status="validation_failed",
                steps=steps,
                selector_summary=selector_summary,
                validation=validation,
                error=",".join(str(item) for item in validation.get("errors", [])),
                output_root=args.output_root,
                selector_summary_json=selector_summary_json,
                selector_pair_selection_csv=selector_pair_selection_csv,
                validation_json=args.validation_output_json,
                validation_html=args.validation_output_html,
                audit_json=audit_json,
                audit_html=audit_html,
            )
            print(
                "true_geometry_selector_pipeline_status=validation_failed "
                f"errors={','.join(str(item) for item in validation.get('errors', []))}",
                flush=True,
            )
            return 1

        audit_items = optimization_audit.audit_goal(
            project_root=args.project_root,
            true_geometry_selector_summary_json=selector_summary_json,
            true_geometry_manifest_validation_json=args.manifest_validation_json,
        )
        optimization_audit.write_json(audit_json, audit_items)
        optimization_audit.write_html(audit_html, audit_items)
        _write_pipeline_summary(
            path_json=summary_json,
            path_html=summary_html,
            status="valid",
            steps=steps,
            selector_summary=selector_summary,
            validation=validation,
            audit_items=audit_items,
            output_root=args.output_root,
            selector_summary_json=selector_summary_json,
            selector_pair_selection_csv=selector_pair_selection_csv,
            validation_json=args.validation_output_json,
            validation_html=args.validation_output_html,
            audit_json=audit_json,
            audit_html=audit_html,
        )
        print(
            "true_geometry_selector_pipeline_status=valid "
            f"correct_delta={validation['correct_delta_vs_lightglue']} "
            f"wrong_delta={validation['wrong_delta_vs_lightglue']} "
            f"output={summary_json}",
            flush=True,
        )
        return 0
    except Exception as exc:
        _write_pipeline_summary(
            path_json=summary_json,
            path_html=summary_html,
            status="failed",
            steps=steps,
            error=str(exc),
            output_root=args.output_root,
            selector_summary_json=selector_summary_json,
            selector_pair_selection_csv=selector_pair_selection_csv,
            validation_json=args.validation_output_json,
            validation_html=args.validation_output_html,
            audit_json=audit_json,
            audit_html=audit_html,
        )
        print(f"true_geometry_selector_pipeline_status=failed error={exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
