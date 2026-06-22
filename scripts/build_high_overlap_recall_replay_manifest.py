#!/usr/bin/env python3
"""Build high-overlap recall replay manifests from true-geometry train pairs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


EXTRA_FIELDS = [
    "phase43_recall_replay_reason",
    "phase43_recall_replay_source",
    "phase43_recall_replay_min_valid_fraction",
    "phase43_recall_replay_valid_bucket",
    "phase43_recall_replay_repeat_weight",
    "phase43_recall_replay_copy_index",
]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def valid_fraction_bucket(valid_fraction: float) -> str:
    if valid_fraction < 0.30:
        return "below_recall_floor"
    if valid_fraction < 0.50:
        return "mid_high"
    if valid_fraction < 0.75:
        return "high"
    return "very_high"


def select_high_overlap_rows(
    rows: Sequence[dict[str, str]],
    *,
    min_valid_fraction: float,
    target_variants: set[str],
    pair_types: set[str],
    max_per_bucket: int,
    seed: int,
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split", "train") != "train":
            continue
        pair_type = row.get("pair_type", "") or "cross_camera"
        if pair_type not in pair_types:
            continue
        target_variant = row.get("target_variant", "")
        if target_variant not in target_variants:
            continue
        valid_fraction = _float_value(row, "valid_fraction")
        if valid_fraction < min_valid_fraction:
            continue
        grouped[(pair_type, target_variant, valid_fraction_bucket(valid_fraction))].append(dict(row))

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        rng.shuffle(candidates)
        if max_per_bucket > 0:
            candidates = candidates[:max_per_bucket]
        selected.extend(candidates)
    return selected


def build_replay_rows(
    selected_rows: Sequence[dict[str, str]],
    *,
    min_valid_fraction: float,
    repeat: int,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in selected_rows:
        valid_fraction = _float_value(row, "valid_fraction")
        for copy_index in range(repeat):
            out = dict(row)
            out.update(
                {
                    "phase43_recall_replay_reason": "true_geometry_high_overlap_recall_replay",
                    "phase43_recall_replay_source": "train_manifest_valid_fraction",
                    "phase43_recall_replay_min_valid_fraction": f"{min_valid_fraction:.6f}",
                    "phase43_recall_replay_valid_bucket": valid_fraction_bucket(valid_fraction),
                    "phase43_recall_replay_repeat_weight": str(repeat),
                    "phase43_recall_replay_copy_index": str(copy_index),
                }
            )
            output.append(out)
    return output


def build_mixed_rows(base_rows: Sequence[dict[str, str]], replay_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    mixed_rows: list[dict[str, str]] = []
    for row in base_rows:
        out = dict(row)
        for field in EXTRA_FIELDS:
            out.setdefault(field, "")
        mixed_rows.append(out)
    mixed_rows.extend(dict(row) for row in replay_rows)
    return mixed_rows


def _write_csv(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_counts(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        valid_fraction = _float_value(row, "valid_fraction")
        counter[f"{row.get('target_variant', '')}:{valid_fraction_bucket(valid_fraction)}"] += 1
    return dict(sorted(counter.items()))


def _write_report(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count_rows = []
    selected_by_variant_bucket = summary.get("selected_by_variant_bucket", {})
    if isinstance(selected_by_variant_bucket, dict):
        for key, value in selected_by_variant_bucket.items():
            count_rows.append(
                f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
            )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Phase43 High-Overlap Recall Replay Manifest</title>",
                "<h1>Phase43 High-Overlap Recall Replay Manifest</h1>",
                "<p>source=<code>true geometry valid_fraction from train manifest only</code></p>",
                f"<p>input_rows=<code>{summary['input_rows']}</code></p>",
                f"<p>selected_input_rows=<code>{summary['selected_input_rows']}</code></p>",
                f"<p>output_rows=<code>{summary['output_rows']}</code></p>",
                f"<p>mixed_output_rows=<code>{summary['mixed_output_rows']}</code></p>",
                f"<p>min_valid_fraction=<code>{summary['min_valid_fraction']}</code></p>",
                f"<p>repeat=<code>{summary['repeat']}</code></p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>target_variant:bucket</th><th>selected rows</th></tr>",
                *count_rows,
                "</table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mixed-output-manifest", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--report-html", type=Path)
    parser.add_argument("--min-valid-fraction", type=float, default=0.30)
    parser.add_argument("--target-variant", action="append")
    parser.add_argument("--pair-type", action="append", default=["cross_camera"])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-per-bucket", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260621)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.max_per_bucket < 0:
        raise ValueError("--max-per-bucket must be non-negative")
    if not math.isfinite(args.min_valid_fraction):
        raise ValueError("--min-valid-fraction must be finite")
    target_variants = set(args.target_variant or ["extreme_01", "extreme_02", "extreme_03"])
    pair_types = set(args.pair_type or ["cross_camera"])

    input_rows = _read_csv_rows(args.input_manifest)
    input_fieldnames = _fieldnames(args.input_manifest)
    selected_rows = select_high_overlap_rows(
        input_rows,
        min_valid_fraction=args.min_valid_fraction,
        target_variants=target_variants,
        pair_types=pair_types,
        max_per_bucket=args.max_per_bucket,
        seed=args.seed,
    )
    replay_rows = build_replay_rows(selected_rows, min_valid_fraction=args.min_valid_fraction, repeat=args.repeat)
    output_fieldnames = [*input_fieldnames, *[field for field in EXTRA_FIELDS if field not in input_fieldnames]]
    _write_csv(args.output_manifest, replay_rows, output_fieldnames)

    mixed_output_rows = 0
    if args.mixed_output_manifest:
        mixed_rows = build_mixed_rows(input_rows, replay_rows)
        _write_csv(args.mixed_output_manifest, mixed_rows, output_fieldnames)
        mixed_output_rows = len(mixed_rows)

    summary = {
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(args.output_manifest),
        "mixed_output_manifest": str(args.mixed_output_manifest) if args.mixed_output_manifest else "",
        "input_rows": len(input_rows),
        "selected_input_rows": len(selected_rows),
        "output_rows": len(replay_rows),
        "mixed_output_rows": mixed_output_rows,
        "min_valid_fraction": args.min_valid_fraction,
        "target_variants": sorted(target_variants),
        "pair_types": sorted(pair_types),
        "repeat": args.repeat,
        "max_per_bucket": args.max_per_bucket,
        "seed": args.seed,
        "selected_by_variant_bucket": _summary_counts(selected_rows),
        "uses_lightglue_labels": False,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report_html:
        _write_report(args.report_html, summary)
    print(json.dumps({"selected_input_rows": len(selected_rows), "output_rows": len(replay_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
