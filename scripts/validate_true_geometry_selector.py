#!/usr/bin/env python3
"""Validate that a true-geometry PFM selector beats the LightGlue baseline."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


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


def _multiseed_seed_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    totals = summary.get("totals")
    seed_results = summary.get("seed_results")
    if not isinstance(totals, dict) or not isinstance(seed_results, list):
        return []
    if "selector_correct" not in totals or "correct_delta_vs_lightglue" not in totals:
        return []
    return [row for row in seed_results if isinstance(row, dict)]


def _multiseed_selector_payload(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    seed_rows = _multiseed_seed_rows(summary)
    totals = summary.get("totals")
    if not seed_rows or not isinstance(totals, dict):
        return {}, {}, {}

    selected_correct = _int_value(totals, "selector_correct")
    selected_wrong = _int_value(totals, "selector_wrong")
    lightglue_correct = _int_value(totals, "lightglue_correct")
    lightglue_wrong = _int_value(totals, "lightglue_wrong")
    selected_matches = _int_value(totals, "selector_matches", _int_value(totals, "selected_matches"))
    if selected_matches <= 0:
        selected_matches = selected_correct + selected_wrong
    lightglue_matches = _int_value(totals, "lightglue_matches")
    if lightglue_matches <= 0:
        lightglue_matches = lightglue_correct + lightglue_wrong
    selector = {
        "rows": _int_value(totals, "rows"),
        "selected_matches": selected_matches,
        "selected_correct": selected_correct,
        "selected_wrong": selected_wrong,
        "selected_precision": selected_correct / selected_matches if selected_matches else 0.0,
        "lightglue_matches": lightglue_matches,
        "lightglue_correct": lightglue_correct,
        "lightglue_wrong": lightglue_wrong,
        "lightglue_precision": lightglue_correct / lightglue_matches if lightglue_matches else 0.0,
        "correct_delta_vs_lightglue": _int_value(totals, "correct_delta_vs_lightglue"),
        "wrong_delta_vs_lightglue": _int_value(totals, "wrong_delta_vs_lightglue"),
    }
    by_split: dict[str, dict[str, Any]] = {}
    by_variant: dict[str, dict[str, Any]] = {}
    for row in seed_rows:
        split_results = row.get("split_results")
        split_results = split_results if isinstance(split_results, dict) else {}
        for split, result in split_results.items():
            result = result if isinstance(result, dict) else {}
            aggregate = by_split.setdefault(
                str(split),
                {"rows": 0, "correct_delta_vs_lightglue": 0, "wrong_delta_vs_lightglue": 0},
            )
            aggregate["rows"] += _int_value(result, "rows")
            aggregate["correct_delta_vs_lightglue"] += _int_value(result, "correct_delta_vs_lightglue")
            aggregate["wrong_delta_vs_lightglue"] += _int_value(result, "wrong_delta_vs_lightglue")
        variant_results = row.get("variant_results")
        variant_results = variant_results if isinstance(variant_results, dict) else {}
        for variant, result in variant_results.items():
            result = result if isinstance(result, dict) else {}
            aggregate = by_variant.setdefault(
                str(variant),
                {"rows": 0, "correct_delta_vs_lightglue": 0, "wrong_delta_vs_lightglue": 0},
            )
            aggregate["rows"] += _int_value(result, "rows")
            aggregate["correct_delta_vs_lightglue"] += _int_value(result, "correct_delta_vs_lightglue")
            aggregate["wrong_delta_vs_lightglue"] += _int_value(result, "wrong_delta_vs_lightglue")
    return selector, by_split, by_variant


def _multiseed_manifest_payload(summary: dict[str, Any]) -> dict[str, Any]:
    seed_rows = _multiseed_seed_rows(summary)
    if not seed_rows:
        return {}
    counts: dict[str, int] = {}
    base_disjoint = True
    excluded_base_ids = 0
    for row in seed_rows:
        manifest_gate = row.get("manifest_gate")
        manifest_gate = manifest_gate if isinstance(manifest_gate, dict) else {}
        base_disjoint = base_disjoint and bool(row.get("base_disjoint", manifest_gate.get("base_disjoint")))
        excluded_base_ids = max(excluded_base_ids, _int_value(manifest_gate, "excluded_base_ids"))
        manifest_counts = row.get("manifest_counts")
        if not isinstance(manifest_counts, dict):
            manifest_counts = manifest_gate.get("counts")
        manifest_counts = manifest_counts if isinstance(manifest_counts, dict) else {}
        for split, value in manifest_counts.items():
            counts[str(split)] = counts.get(str(split), 0) + _int_value({str(split): value}, str(split))
    return {
        "counts": counts,
        "base_disjoint": base_disjoint,
        "excluded_base_ids": excluded_base_ids,
    }


def _selector_payload(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    multiseed_selector, multiseed_by_split, multiseed_by_variant = _multiseed_selector_payload(summary)
    if multiseed_selector:
        return multiseed_selector, multiseed_by_split, multiseed_by_variant

    aggregate = summary.get("aggregate")
    if isinstance(aggregate, dict) and (
        "selected_correct" in aggregate
        or "selected_matches" in aggregate
        or "pfm_correct" in aggregate
        or "pfm_matches" in aggregate
    ):
        by_split = summary.get("by_split")
        by_variant = summary.get("by_variant")
        return (
            aggregate,
            by_split if isinstance(by_split, dict) else {},
            by_variant if isinstance(by_variant, dict) else {},
        )

    comparison = summary.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    selector = comparison.get("selector")
    if isinstance(selector, dict):
        by_split = comparison.get("selector_by_split")
        by_variant = comparison.get("selector_by_variant")
        return (
            selector,
            by_split if isinstance(by_split, dict) else {},
            by_variant if isinstance(by_variant, dict) else {},
        )

    selector = summary.get("selector")
    if isinstance(selector, dict):
        by_split = summary.get("selector_by_split")
        by_variant = summary.get("selector_by_variant")
        return (
            selector,
            by_split if isinstance(by_split, dict) else {},
            by_variant if isinstance(by_variant, dict) else {},
        )

    return {}, {}, {}


def _manifest_payload(summary: dict[str, Any], manifest_validation_json: Path | None) -> dict[str, Any]:
    if manifest_validation_json is not None:
        return _load_json_object(manifest_validation_json)
    manifest = summary.get("manifest_validation")
    if isinstance(manifest, dict):
        return manifest
    return _multiseed_manifest_payload(summary)


def _correct_delta(metrics: dict[str, Any]) -> int:
    if "correct_delta_vs_lightglue" in metrics:
        return _int_value(metrics, "correct_delta_vs_lightglue")
    selected_correct = _int_value(metrics, "selected_correct", _int_value(metrics, "pfm_correct"))
    return selected_correct - _int_value(metrics, "lightglue_correct")


def _wrong_delta(metrics: dict[str, Any]) -> int:
    if "wrong_delta_vs_lightglue" in metrics:
        return _int_value(metrics, "wrong_delta_vs_lightglue")
    selected_wrong = _int_value(metrics, "selected_wrong", _int_value(metrics, "pfm_wrong"))
    return selected_wrong - _int_value(metrics, "lightglue_wrong")


def validate_summary(
    summary: dict[str, Any],
    *,
    manifest: dict[str, Any],
    min_rows: int,
    min_correct_delta_vs_lightglue: int,
    max_wrong_delta_vs_lightglue: int,
    min_precision_delta_vs_lightglue: float,
    required_splits: Sequence[str],
    required_variants: Sequence[str],
    require_base_disjoint: bool,
) -> dict[str, Any]:
    selector, by_split, by_variant = _selector_payload(summary)
    errors: list[str] = []
    if not selector:
        errors.append("missing_selector_summary")

    rows = _int_value(selector, "rows")
    selected_matches = _int_value(selector, "selected_matches", _int_value(selector, "pfm_matches"))
    selected_correct = _int_value(selector, "selected_correct", _int_value(selector, "pfm_correct"))
    selected_wrong = _int_value(selector, "selected_wrong", _int_value(selector, "pfm_wrong"))
    selected_precision = _float_value(
        selector,
        "selected_precision",
        _float_value(selector, "pfm_precision", selected_correct / selected_matches if selected_matches else 0.0),
    )
    lightglue_matches = _int_value(selector, "lightglue_matches")
    lightglue_correct = _int_value(selector, "lightglue_correct")
    lightglue_wrong = _int_value(selector, "lightglue_wrong")
    lightglue_precision = _float_value(
        selector,
        "lightglue_precision",
        lightglue_correct / lightglue_matches if lightglue_matches else 0.0,
    )
    correct_delta = _int_value(selector, "correct_delta_vs_lightglue", selected_correct - lightglue_correct)
    wrong_delta = _int_value(selector, "wrong_delta_vs_lightglue", selected_wrong - lightglue_wrong)
    precision_delta = selected_precision - lightglue_precision

    if rows < min_rows:
        errors.append("rows_below_minimum")
    if correct_delta < min_correct_delta_vs_lightglue:
        errors.append("correct_delta_below_minimum")
    if wrong_delta > max_wrong_delta_vs_lightglue:
        errors.append("wrong_delta_exceeds_limit")
    if precision_delta < min_precision_delta_vs_lightglue:
        errors.append("precision_delta_below_minimum")

    split_results: dict[str, dict[str, Any]] = {}
    for split in required_splits:
        split_metrics = by_split.get(split)
        split_metrics = split_metrics if isinstance(split_metrics, dict) else {}
        split_correct_delta = _correct_delta(split_metrics)
        split_wrong_delta = _wrong_delta(split_metrics)
        split_rows = _int_value(split_metrics, "rows")
        if not split_metrics:
            errors.append(f"{split}_missing")
        elif split_rows <= 0:
            errors.append(f"{split}_rows_missing")
        if split_correct_delta < min_correct_delta_vs_lightglue:
            errors.append(f"{split}_correct_delta_below_minimum")
        if split_wrong_delta > max_wrong_delta_vs_lightglue:
            errors.append(f"{split}_wrong_delta_exceeds_limit")
        split_results[split] = {
            "rows": split_rows,
            "correct_delta_vs_lightglue": split_correct_delta,
            "wrong_delta_vs_lightglue": split_wrong_delta,
        }

    variant_results: dict[str, dict[str, Any]] = {}
    for variant in required_variants:
        variant_metrics = by_variant.get(variant)
        variant_metrics = variant_metrics if isinstance(variant_metrics, dict) else {}
        variant_correct_delta = _correct_delta(variant_metrics)
        variant_wrong_delta = _wrong_delta(variant_metrics)
        variant_rows = _int_value(variant_metrics, "rows")
        if not variant_metrics:
            errors.append(f"{variant}_missing")
        elif variant_rows <= 0:
            errors.append(f"{variant}_rows_missing")
        if variant_correct_delta < min_correct_delta_vs_lightglue:
            errors.append(f"{variant}_correct_delta_below_minimum")
        if variant_wrong_delta > max_wrong_delta_vs_lightglue:
            errors.append(f"{variant}_wrong_delta_exceeds_limit")
        variant_results[variant] = {
            "rows": variant_rows,
            "correct_delta_vs_lightglue": variant_correct_delta,
            "wrong_delta_vs_lightglue": variant_wrong_delta,
        }

    counts = manifest.get("counts")
    counts = counts if isinstance(counts, dict) else {}
    base_disjoint = bool(manifest.get("base_disjoint"))
    excluded_base_ids = _int_value(manifest, "excluded_base_ids")
    if require_base_disjoint and not base_disjoint:
        errors.append("fresh_manifest_not_base_disjoint")
    if require_base_disjoint or counts:
        for split in required_splits:
            if _int_value(counts, split) <= 0:
                errors.append(f"fresh_manifest_missing_{split}")

    return {
        "valid": not errors,
        "errors": errors,
        "rows": rows,
        "selected_matches": selected_matches,
        "selected_correct": selected_correct,
        "selected_wrong": selected_wrong,
        "selected_precision": selected_precision,
        "lightglue_matches": lightglue_matches,
        "lightglue_correct": lightglue_correct,
        "lightglue_wrong": lightglue_wrong,
        "lightglue_precision": lightglue_precision,
        "correct_delta_vs_lightglue": correct_delta,
        "wrong_delta_vs_lightglue": wrong_delta,
        "precision_delta_vs_lightglue": precision_delta,
        "min_rows": min_rows,
        "min_correct_delta_vs_lightglue": min_correct_delta_vs_lightglue,
        "max_wrong_delta_vs_lightglue": max_wrong_delta_vs_lightglue,
        "min_precision_delta_vs_lightglue": min_precision_delta_vs_lightglue,
        "required_splits": list(required_splits),
        "required_variants": list(required_variants),
        "split_results": split_results,
        "variant_results": variant_results,
        "base_disjoint": base_disjoint,
        "excluded_base_ids": excluded_base_ids,
        "manifest_counts": counts,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_html(path: Path, *, summary_json: Path, payload: dict[str, Any]) -> None:
    keys = [
        "valid",
        "errors",
        "rows",
        "selected_correct",
        "selected_wrong",
        "selected_precision",
        "lightglue_correct",
        "lightglue_wrong",
        "lightglue_precision",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "precision_delta_vs_lightglue",
        "base_disjoint",
        "excluded_base_ids",
    ]
    body = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(payload.get(key, '')))}</td></tr>"
        for key in keys
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>True Geometry Selector validation</title>",
                "<h1>True Geometry Selector validation</h1>",
                f"<p>summary_json={html.escape(str(summary_json))}</p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>metric</th><th>value</th></tr>",
                body,
                "</table>",
                "<h2>Validation JSON</h2>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--manifest-validation-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-correct-delta-vs-lightglue", type=int, default=1)
    parser.add_argument("--max-wrong-delta-vs-lightglue", type=int, default=0)
    parser.add_argument("--min-precision-delta-vs-lightglue", type=float, default=0.0)
    parser.add_argument("--required-split", action="append", default=None)
    parser.add_argument("--required-variant", action="append", default=None)
    parser.add_argument("--no-require-base-disjoint", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = _load_json_object(args.summary_json)
    manifest = _manifest_payload(summary, args.manifest_validation_json)
    payload = validate_summary(
        summary,
        manifest=manifest,
        min_rows=int(args.min_rows),
        min_correct_delta_vs_lightglue=int(args.min_correct_delta_vs_lightglue),
        max_wrong_delta_vs_lightglue=int(args.max_wrong_delta_vs_lightglue),
        min_precision_delta_vs_lightglue=float(args.min_precision_delta_vs_lightglue),
        required_splits=list(args.required_split) if args.required_split else ["dev", "val", "lockbox"],
        required_variants=list(args.required_variant) if args.required_variant else [],
        require_base_disjoint=not bool(args.no_require_base_disjoint),
    )
    payload["summary_json"] = str(args.summary_json)
    payload["manifest_validation_json"] = str(args.manifest_validation_json) if args.manifest_validation_json else ""
    _write_json(args.output_json, payload)
    _write_html(args.output_html, summary_json=args.summary_json, payload=payload)
    print(
        f"true_geometry_selector_valid={payload['valid']} "
        f"correct_delta={payload['correct_delta_vs_lightglue']} "
        f"wrong_delta={payload['wrong_delta_vs_lightglue']} "
        f"output={args.output_json}",
        flush=True,
    )
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
