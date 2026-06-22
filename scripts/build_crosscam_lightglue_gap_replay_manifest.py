#!/usr/bin/env python3
"""Build train-only replay manifests from cross-camera PFM/LightGlue recall gaps."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FORBIDDEN_SOURCE_TOKENS = ("fresh", "heldout", "holdout", "lockbox")
LIGHTGLUE_LABEL = "LightGlue-SIFT-MAGSAC-min16"
EXTRA_FIELDS = [
    "phase42_gap_replay_reason",
    "phase42_gap_replay_sources",
    "phase42_gap_replay_source_rows",
    "phase42_gap_replay_pfm_correct_sum",
    "phase42_gap_replay_lightglue_correct_sum",
    "phase42_gap_replay_correct_gap_sum",
    "phase42_gap_replay_valid_bucket",
    "phase42_gap_replay_repeat_weight",
    "phase42_gap_replay_copy_index",
]


@dataclass(frozen=True)
class GapSource:
    name: str
    pair_manifest: Path
    pfm_summary: Path
    lightglue_metrics: Path


@dataclass
class GapPattern:
    pair_type: str
    reference_variant: str
    target_variant: str
    valid_bucket: str
    source_ids: list[str]
    source_rows: int = 0
    pfm_correct_sum: int = 0
    lightglue_correct_sum: int = 0

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.pair_type, self.reference_variant, self.target_variant, self.valid_bucket

    @property
    def correct_gap_sum(self) -> int:
        return self.lightglue_correct_sum - self.pfm_correct_sum


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


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def valid_fraction_bucket(valid_fraction: float) -> str:
    if valid_fraction < 0.20:
        return "low"
    if valid_fraction < 0.50:
        return "mid"
    if valid_fraction < 0.75:
        return "high"
    return "very_high"


def parse_source(value: str) -> GapSource:
    parts = value.split(",", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--source must use name,pair_manifest,pfm_summary,lightglue_metrics")
    name, pair_manifest, pfm_summary, lightglue_metrics = parts
    if not name:
        raise argparse.ArgumentTypeError("source name must not be empty")
    return GapSource(
        name=name,
        pair_manifest=Path(pair_manifest),
        pfm_summary=Path(pfm_summary),
        lightglue_metrics=Path(lightglue_metrics),
    )


def assert_allowed_source(source: GapSource, *, allow_forbidden_source_tokens: bool) -> None:
    if allow_forbidden_source_tokens:
        return
    text = " ".join(
        [
            source.name.lower(),
            str(source.pair_manifest).lower(),
            str(source.pfm_summary).lower(),
            str(source.lightglue_metrics).lower(),
        ]
    )
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in text:
            raise ValueError(
                f"refusing to build training replay from forbidden source token '{token}'; "
                "lockbox/fresh/held-out sources are validation-only by default"
            )


def _lightglue_filtered_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in _read_csv_rows(path) if row.get("label") == LIGHTGLUE_LABEL]


def _source_row_id(source: GapSource, row_index: int, pair_row: dict[str, str]) -> str:
    return ":".join(
        item
        for item in (
            source.name,
            str(row_index),
            pair_row.get("reference_base_id", ""),
            pair_row.get("target_variant", ""),
        )
        if item
    )


def collect_gap_patterns(
    sources: Sequence[GapSource],
    *,
    min_valid_fraction: float,
    min_lightglue_correct: int,
    max_pfm_correct_fraction: float,
    allow_forbidden_source_tokens: bool = False,
) -> list[GapPattern]:
    grouped: dict[tuple[str, str, str, str], GapPattern] = {}
    for source in sources:
        assert_allowed_source(source, allow_forbidden_source_tokens=allow_forbidden_source_tokens)
        pair_rows = _read_csv_rows(source.pair_manifest)
        pfm_rows = _read_csv_rows(source.pfm_summary)
        lightglue_rows = _lightglue_filtered_rows(source.lightglue_metrics)
        if len(pair_rows) != len(pfm_rows) or len(pair_rows) != len(lightglue_rows):
            raise ValueError(
                f"row count mismatch for source {source.name}: "
                f"pairs={len(pair_rows)} PFM={len(pfm_rows)} LightGlue={len(lightglue_rows)}"
            )
        for row_index, (pair_row, pfm_row, lightglue_row) in enumerate(
            zip(pair_rows, pfm_rows, lightglue_rows)
        ):
            valid_fraction = _float_value(pfm_row, "valid_fraction", _float_value(pair_row, "valid_fraction"))
            pfm_correct = _int_value(pfm_row, "correct")
            lightglue_correct = _int_value(lightglue_row, "correct")
            if valid_fraction < min_valid_fraction:
                continue
            if lightglue_correct < min_lightglue_correct:
                continue
            if pfm_correct >= lightglue_correct * max_pfm_correct_fraction:
                continue
            pair_type = pair_row.get("pair_type", "") or "cross_camera"
            reference_variant = pair_row.get("reference_variant", "")
            target_variant = pair_row.get("target_variant", pfm_row.get("target_variant", ""))
            if not reference_variant or not target_variant:
                continue
            bucket = valid_fraction_bucket(valid_fraction)
            key = (pair_type, reference_variant, target_variant, bucket)
            pattern = grouped.setdefault(
                key,
                GapPattern(
                    pair_type=pair_type,
                    reference_variant=reference_variant,
                    target_variant=target_variant,
                    valid_bucket=bucket,
                    source_ids=[],
                ),
            )
            pattern.source_ids.append(_source_row_id(source, row_index, pair_row))
            pattern.source_rows += 1
            pattern.pfm_correct_sum += pfm_correct
            pattern.lightglue_correct_sum += lightglue_correct
    return sorted(grouped.values(), key=lambda item: (-item.correct_gap_sum, item.target_variant, item.valid_bucket))


def sample_train_rows(
    train_rows: Sequence[dict[str, str]],
    patterns: Sequence[GapPattern],
    *,
    max_per_pattern: int,
    repeat: int,
    seed: int,
) -> list[dict[str, str]]:
    pattern_by_key = {pattern.key: pattern for pattern in patterns}
    candidates_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        if row.get("split", "train") != "train":
            continue
        valid_fraction = _float_value(row, "valid_fraction")
        key = (
            row.get("pair_type", "") or "cross_camera",
            row.get("reference_variant", ""),
            row.get("target_variant", ""),
            valid_fraction_bucket(valid_fraction),
        )
        if key in pattern_by_key:
            candidates_by_key[key].append(dict(row))

    rng = random.Random(seed)
    output_rows: list[dict[str, str]] = []
    for key, pattern in pattern_by_key.items():
        candidates = list(candidates_by_key.get(key, []))
        if not candidates:
            continue
        rng.shuffle(candidates)
        if max_per_pattern > 0:
            candidates = candidates[:max_per_pattern]
        for row in candidates:
            for copy_index in range(repeat):
                out = dict(row)
                out.update(
                    {
                        "phase42_gap_replay_reason": "pfm_recall_gap_vs_lightglue_pattern",
                        "phase42_gap_replay_sources": ";".join(pattern.source_ids),
                        "phase42_gap_replay_source_rows": str(pattern.source_rows),
                        "phase42_gap_replay_pfm_correct_sum": str(pattern.pfm_correct_sum),
                        "phase42_gap_replay_lightglue_correct_sum": str(pattern.lightglue_correct_sum),
                        "phase42_gap_replay_correct_gap_sum": str(pattern.correct_gap_sum),
                        "phase42_gap_replay_valid_bucket": pattern.valid_bucket,
                        "phase42_gap_replay_repeat_weight": str(repeat),
                        "phase42_gap_replay_copy_index": str(copy_index),
                    }
                )
                output_rows.append(out)
    return output_rows


def _write_csv(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_mixed_rows(base_rows: Sequence[dict[str, str]], replay_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    mixed_rows: list[dict[str, str]] = []
    for row in base_rows:
        out = dict(row)
        for field in EXTRA_FIELDS:
            out.setdefault(field, "")
        mixed_rows.append(out)
    mixed_rows.extend(dict(row) for row in replay_rows)
    return mixed_rows


def _write_report(path: Path, summary: dict[str, object], patterns: Sequence[GapPattern]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for pattern in patterns:
        rows.append(
            "<tr>"
            f"<td>{html.escape(pattern.pair_type)}</td>"
            f"<td>{html.escape(pattern.reference_variant)}</td>"
            f"<td>{html.escape(pattern.target_variant)}</td>"
            f"<td>{html.escape(pattern.valid_bucket)}</td>"
            f"<td>{pattern.source_rows}</td>"
            f"<td>{pattern.pfm_correct_sum}</td>"
            f"<td>{pattern.lightglue_correct_sum}</td>"
            f"<td>{pattern.correct_gap_sum}</td>"
            "</tr>"
        )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Phase42 Cross-Camera LightGlue Gap Replay</title>",
                "<h1>Phase42 Cross-Camera LightGlue Gap Replay</h1>",
                "<p>note=<code>LightGlue is used only to choose train-pool replay patterns, not as labels.</code></p>",
                f"<p>selected_source_rows=<code>{summary['selected_source_rows']}</code></p>",
                f"<p>output_rows=<code>{summary['output_rows']}</code></p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>pair_type</th><th>reference_variant</th><th>target_variant</th><th>valid_bucket</th><th>source_rows</th><th>PFM correct</th><th>LightGlue correct</th><th>gap</th></tr>",
                *rows,
                "</table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mixed-base-manifest", type=Path)
    parser.add_argument("--mixed-output-manifest", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--report-html", type=Path)
    parser.add_argument("--min-valid-fraction", type=float, default=0.20)
    parser.add_argument("--min-lightglue-correct", type=int, default=30)
    parser.add_argument("--max-pfm-correct-fraction", type=float, default=0.50)
    parser.add_argument("--max-per-pattern", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--allow-forbidden-source-tokens", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.max_per_pattern < 0:
        raise ValueError("--max-per-pattern must be non-negative")
    if bool(args.mixed_base_manifest) != bool(args.mixed_output_manifest):
        raise ValueError("--mixed-base-manifest and --mixed-output-manifest must be provided together")
    patterns = collect_gap_patterns(
        args.source,
        min_valid_fraction=args.min_valid_fraction,
        min_lightglue_correct=args.min_lightglue_correct,
        max_pfm_correct_fraction=args.max_pfm_correct_fraction,
        allow_forbidden_source_tokens=args.allow_forbidden_source_tokens,
    )
    train_rows = _read_csv_rows(args.train_manifest)
    train_fieldnames = _fieldnames(args.train_manifest)
    output_rows = sample_train_rows(
        train_rows,
        patterns,
        max_per_pattern=args.max_per_pattern,
        repeat=args.repeat,
        seed=args.seed,
    )
    output_fieldnames = [*train_fieldnames, *[field for field in EXTRA_FIELDS if field not in train_fieldnames]]
    _write_csv(args.output_manifest, output_rows, output_fieldnames)
    mixed_output_rows = 0
    if args.mixed_base_manifest and args.mixed_output_manifest:
        mixed_base_rows = _read_csv_rows(args.mixed_base_manifest)
        mixed_base_fieldnames = _fieldnames(args.mixed_base_manifest)
        mixed_fieldnames = [*mixed_base_fieldnames, *[field for field in EXTRA_FIELDS if field not in mixed_base_fieldnames]]
        mixed_rows = build_mixed_rows(mixed_base_rows, output_rows)
        _write_csv(args.mixed_output_manifest, mixed_rows, mixed_fieldnames)
        mixed_output_rows = len(mixed_rows)

    summary = {
        "sources": [source.name for source in args.source],
        "train_manifest": str(args.train_manifest),
        "output_manifest": str(args.output_manifest),
        "patterns": [
            {
                "pair_type": pattern.pair_type,
                "reference_variant": pattern.reference_variant,
                "target_variant": pattern.target_variant,
                "valid_bucket": pattern.valid_bucket,
                "source_rows": pattern.source_rows,
                "pfm_correct_sum": pattern.pfm_correct_sum,
                "lightglue_correct_sum": pattern.lightglue_correct_sum,
                "correct_gap_sum": pattern.correct_gap_sum,
            }
            for pattern in patterns
        ],
        "selected_source_rows": sum(pattern.source_rows for pattern in patterns),
        "output_rows": len(output_rows),
        "mixed_output_manifest": str(args.mixed_output_manifest) if args.mixed_output_manifest else "",
        "mixed_output_rows": mixed_output_rows,
        "min_valid_fraction": args.min_valid_fraction,
        "min_lightglue_correct": args.min_lightglue_correct,
        "max_pfm_correct_fraction": args.max_pfm_correct_fraction,
        "repeat": args.repeat,
        "max_per_pattern": args.max_per_pattern,
        "seed": args.seed,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report_html:
        _write_report(args.report_html, summary, patterns)
    print(
        json.dumps(
            {"selected_source_rows": summary["selected_source_rows"], "output_rows": len(output_rows)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
