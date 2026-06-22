#!/usr/bin/env python3
"""Build stratified cross-camera extreme pair-spec manifests."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


EXTREME_VARIANTS = ("extreme_01", "extreme_02", "extreme_03")


@dataclass(frozen=True)
class OutputSpec:
    name: str
    per_bucket: int


def valid_fraction_bucket(value: float) -> str:
    if value < 0.02:
        return "reject"
    if value < 0.15:
        return "low"
    if value < 0.50:
        return "mid"
    return "high"


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_base_keys(row: dict[str, str]) -> set[str]:
    return {
        value
        for value in (row.get("reference_base_id", ""), row.get("target_base_id", ""))
        if value
    }


def row_dataset_base_keys(row: dict[str, str]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for dataset_field, base_field in (
        ("reference_dataset_id", "reference_base_id"),
        ("target_dataset_id", "target_base_id"),
    ):
        dataset_id = row.get(dataset_field, "")
        base_id = row.get(base_field, "")
        if dataset_id and base_id:
            keys.add((dataset_id, base_id))
    return keys


def row_bucket_key(row: dict[str, str]) -> tuple[str, str, str]:
    target_variant = row.get("target_variant", "")
    pattern = f"{row.get('reference_variant', '')}->{target_variant}"
    bucket = valid_fraction_bucket(float(row.get("valid_fraction", "0") or 0.0))
    return target_variant, pattern, bucket


def eligible_row(row: dict[str, str]) -> bool:
    if row.get("pair_type", "") != "cross_camera":
        return False
    if row.get("reference_variant", "") not in EXTREME_VARIANTS:
        return False
    if row.get("target_variant", "") not in EXTREME_VARIANTS:
        return False
    try:
        return valid_fraction_bucket(float(row.get("valid_fraction", "0") or 0.0)) != "reject"
    except ValueError:
        return False


def _reindexed_row(row: dict[str, str], pair_index: int, split: str) -> dict[str, str]:
    copied = dict(row)
    copied["pair_index"] = str(pair_index)
    copied["split"] = split
    return copied


def select_split_rows(
    rows: Sequence[dict[str, str]],
    *,
    output_specs: Sequence[OutputSpec],
    seed: int,
    drop_underfilled: bool = False,
    excluded_base_keys: set[tuple[str, str]] | None = None,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    if not output_specs:
        raise ValueError("at least one output spec is required")
    if any(spec.per_bucket <= 0 for spec in output_specs):
        raise ValueError("per_bucket must be positive for every output")

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    rejected_rows = 0
    excluded_rows = 0
    excluded_base_keys = excluded_base_keys or set()
    for row in rows:
        if not eligible_row(row):
            rejected_rows += 1
            continue
        if row_dataset_base_keys(row) & excluded_base_keys:
            excluded_rows += 1
            continue
        grouped[row_bucket_key(row)].append(dict(row))

    selected: dict[str, list[dict[str, str]]] = {spec.name: [] for spec in output_specs}
    used_bases_by_output: dict[str, set[str]] = {spec.name: set() for spec in output_specs}
    counts: dict[str, dict[str, int]] = {spec.name: {} for spec in output_specs}
    available_counts = {":".join(key): len(items) for key, items in sorted(grouped.items())}
    dropped_buckets: dict[str, int] = {}
    rng = random.Random(seed)

    for bucket_key in sorted(grouped):
        bucket_name = ":".join(bucket_key)
        candidates = list(grouped[bucket_key])
        required_rows = sum(spec.per_bucket for spec in output_specs)
        if drop_underfilled and len(candidates) < required_rows:
            dropped_buckets[bucket_name] = len(candidates)
            continue
        rng.shuffle(candidates)
        for spec in output_specs:
            other_output_bases = set().union(
                *(
                    bases
                    for output_name, bases in used_bases_by_output.items()
                    if output_name != spec.name
                )
            )
            chosen: list[dict[str, str]] = []
            for candidate in candidates:
                candidate_bases = row_base_keys(candidate)
                if candidate_bases & other_output_bases:
                    continue
                if candidate_bases & used_bases_by_output[spec.name]:
                    continue
                if candidate in chosen:
                    continue
                chosen.append(candidate)
                used_bases_by_output[spec.name].update(candidate_bases)
                if len(chosen) >= spec.per_bucket:
                    break
            if len(chosen) < spec.per_bucket:
                raise RuntimeError(
                    f"not enough base-disjoint rows for {spec.name}:{bucket_name}: "
                    f"{len(chosen)} < {spec.per_bucket}"
                )
            selected[spec.name].extend(chosen)
            counts[spec.name][bucket_name] = len(chosen)

    for name, output_rows in selected.items():
        output_rows.sort(
            key=lambda row: (
                row.get("target_variant", ""),
                row.get("reference_variant", ""),
                float(row.get("valid_fraction", "0") or 0.0),
                row.get("reference_base_id", ""),
                row.get("target_base_id", ""),
            )
        )
        selected[name] = [_reindexed_row(row, index, name) for index, row in enumerate(output_rows)]

    summary = {
        "eligible_rows": sum(len(items) for items in grouped.values()),
        "rejected_rows": rejected_rows,
        "excluded_rows": excluded_rows,
        "excluded_base_ids": len(excluded_base_keys),
        "bucket_count": len(grouped),
        "dropped_bucket_count": len(dropped_buckets),
        "dropped_buckets": dropped_buckets,
        "available_counts": available_counts,
        "selected_pairs": {name: len(items) for name, items in selected.items()},
        "counts": counts,
        "target_variant_counts": dict(Counter(row.get("target_variant", "") for row in rows if eligible_row(row))),
        "reference_to_target_counts": dict(
            Counter(
                f"{row.get('reference_variant', '')}->{row.get('target_variant', '')}"
                for row in rows
                if eligible_row(row)
            )
        ),
    }
    return selected, summary


def write_html_summary(path: Path, summary: dict[str, object]) -> None:
    counts = summary.get("counts", {})
    selected_pairs = summary.get("selected_pairs", {})
    available_counts = summary.get("available_counts", {})
    split_names = sorted(selected_pairs) if isinstance(selected_pairs, dict) else []
    bucket_names = sorted(available_counts) if isinstance(available_counts, dict) else []
    rows = []
    for bucket in bucket_names:
        cells = [bucket, str(available_counts.get(bucket, ""))]
        if isinstance(counts, dict):
            for split_name in split_names:
                split_counts = counts.get(split_name, {})
                value = split_counts.get(bucket, "") if isinstance(split_counts, dict) else ""
                cells.append(str(value))
        rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")

    header = ["bucket", "available", *split_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Cross-Camera Extreme Manifest</title>",
                "<h1>Cross-Camera Extreme Manifest</h1>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr>" + "".join(f"<th>{html.escape(item)}</th>" for item in header) + "</tr>",
                *rows,
                "</table>",
                "<h2>Summary JSON</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-bucket", type=int, default=12)
    parser.add_argument("--dev-per-bucket", type=int, default=4)
    parser.add_argument("--val-per-bucket", type=int, default=4)
    parser.add_argument("--lockbox-per-bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-html", type=Path, default=None)
    parser.add_argument("--drop-underfilled-buckets", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fieldnames, rows = read_csv_rows(args.candidate_manifest)
    excluded_base_keys: set[tuple[str, str]] = set()
    for path in args.exclude_manifest:
        _exclude_fieldnames, exclude_rows = read_csv_rows(path)
        for row in exclude_rows:
            excluded_base_keys.update(row_dataset_base_keys(row))
    output_specs = [
        OutputSpec("train", int(args.train_per_bucket)),
        OutputSpec("dev", int(args.dev_per_bucket)),
        OutputSpec("val", int(args.val_per_bucket)),
        OutputSpec("lockbox", int(args.lockbox_per_bucket)),
    ]
    selected, summary = select_split_rows(
        rows,
        output_specs=output_specs,
        seed=int(args.seed),
        drop_underfilled=bool(args.drop_underfilled_buckets),
        excluded_base_keys=excluded_base_keys,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in output_specs:
        write_csv_rows(args.output_dir / f"{spec.name}_pairs.csv", fieldnames, selected[spec.name])

    summary = {
        "candidate_manifest": str(args.candidate_manifest),
        "exclude_manifests": [str(path) for path in args.exclude_manifest],
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "per_bucket": {spec.name: spec.per_bucket for spec in output_specs},
        **summary,
    }
    summary_json = args.summary_json or (args.output_dir / "summary.json")
    summary_html = args.summary_html or (args.output_dir / "index.html")
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_html_summary(summary_html, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
