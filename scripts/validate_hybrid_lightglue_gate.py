#!/usr/bin/env python3
"""Validate that a PFM/LightGlue hybrid summary beats the LightGlue baseline."""

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


def validate_summary(
    summary: dict[str, Any],
    *,
    min_correct_delta_vs_lightglue: int,
    max_wrong_delta_vs_lightglue: int,
    min_precision_delta_vs_lightglue: float,
) -> dict[str, Any]:
    lightglue_correct = _int_value(summary, "lightglue_correct")
    lightglue_wrong = _int_value(summary, "lightglue_wrong")
    lightglue_precision = _float_value(summary, "lightglue_precision")
    hybrid_correct = _int_value(summary, "hybrid_correct")
    hybrid_wrong = _int_value(summary, "hybrid_wrong")
    hybrid_precision = _float_value(summary, "hybrid_precision")
    correct_delta = hybrid_correct - lightglue_correct
    wrong_delta = hybrid_wrong - lightglue_wrong
    precision_delta = hybrid_precision - lightglue_precision

    errors: list[str] = []
    if correct_delta < min_correct_delta_vs_lightglue:
        errors.append("correct_delta_below_minimum")
    if wrong_delta > max_wrong_delta_vs_lightglue:
        errors.append("wrong_delta_exceeds_limit")
    if precision_delta < min_precision_delta_vs_lightglue:
        errors.append("precision_delta_below_minimum")

    return {
        "valid": not errors,
        "errors": errors,
        "lightglue_correct": lightglue_correct,
        "lightglue_wrong": lightglue_wrong,
        "lightglue_precision": lightglue_precision,
        "hybrid_correct": hybrid_correct,
        "hybrid_wrong": hybrid_wrong,
        "hybrid_precision": hybrid_precision,
        "correct_delta_vs_lightglue": correct_delta,
        "wrong_delta_vs_lightglue": wrong_delta,
        "precision_delta_vs_lightglue": precision_delta,
        "min_correct_delta_vs_lightglue": min_correct_delta_vs_lightglue,
        "max_wrong_delta_vs_lightglue": max_wrong_delta_vs_lightglue,
        "min_precision_delta_vs_lightglue": min_precision_delta_vs_lightglue,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_html(path: Path, *, summary_json: Path, payload: dict[str, Any]) -> None:
    keys = [
        "valid",
        "errors",
        "lightglue_correct",
        "lightglue_wrong",
        "lightglue_precision",
        "hybrid_correct",
        "hybrid_wrong",
        "hybrid_precision",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "precision_delta_vs_lightglue",
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
                "<title>Hybrid LightGlue gate validation</title>",
                "<h1>Hybrid LightGlue gate validation</h1>",
                f"<p>summary_json={html.escape(str(summary_json))}</p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>metric</th><th>value</th></tr>",
                body,
                "</table>",
                "<h2>Validation JSON</h2>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--min-correct-delta-vs-lightglue", type=int, default=1)
    parser.add_argument("--max-wrong-delta-vs-lightglue", type=int, default=0)
    parser.add_argument("--min-precision-delta-vs-lightglue", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = _load_json_object(args.summary_json)
    payload = validate_summary(
        summary,
        min_correct_delta_vs_lightglue=int(args.min_correct_delta_vs_lightglue),
        max_wrong_delta_vs_lightglue=int(args.max_wrong_delta_vs_lightglue),
        min_precision_delta_vs_lightglue=float(args.min_precision_delta_vs_lightglue),
    )
    payload["summary_json"] = str(args.summary_json)
    _write_json(args.output_json, payload)
    _write_html(args.output_html, summary_json=args.summary_json, payload=payload)
    print(
        f"hybrid_gate_valid={payload['valid']} "
        f"correct_delta={payload['correct_delta_vs_lightglue']} "
        f"wrong_delta={payload['wrong_delta_vs_lightglue']} "
        f"output={args.output_json}",
        flush=True,
    )
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
