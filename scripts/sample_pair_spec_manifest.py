#!/usr/bin/env python3
"""Sample stratified, base-disjoint pair-spec manifests from existing manifests."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


BASE_KEY_FIELDS = (
    ("reference_dataset_id", "reference_base_id"),
    ("target_dataset_id", "target_base_id"),
)


@dataclass(frozen=True)
class OutputSpec:
    name: str
    path: Path
    per_bucket: int


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, [dict(row) for row in reader]


def _merge_fieldnames(groups: Sequence[Sequence[str]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for field in group:
            if field in seen:
                continue
            seen.add(field)
            fieldnames.append(field)
    return fieldnames


def _base_keys(row: dict[str, str]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for dataset_field, base_field in BASE_KEY_FIELDS:
        dataset_id = row.get(dataset_field, "")
        base_id = row.get(base_field, "")
        if dataset_id and base_id:
            keys.add((dataset_id, base_id))
    return keys


def _parse_output_spec(value: str) -> OutputSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--output-spec must have the form name,path,per_bucket"
        )
    name, path_text, per_bucket_text = parts
    if not name:
        raise argparse.ArgumentTypeError("output spec name must not be empty")
    try:
        per_bucket = int(per_bucket_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("output spec per_bucket must be an integer") from exc
    if per_bucket <= 0:
        raise argparse.ArgumentTypeError("output spec per_bucket must be positive")
    return OutputSpec(name=name, path=Path(path_text), per_bucket=per_bucket)


def _write_manifest(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_html(path: Path, payload: dict[str, object]) -> None:
    counts = payload.get("counts", {})
    available = payload.get("available_counts_after_exclusion", {})
    output_specs = payload.get("outputs", {})
    output_names = sorted(output_specs) if isinstance(output_specs, dict) else []
    buckets = sorted(available) if isinstance(available, dict) else []
    rows: list[str] = []
    for bucket in buckets:
        cells = [bucket, str(available.get(bucket, ""))]
        if isinstance(counts, dict):
            for output_name in output_names:
                output_counts = counts.get(output_name, {})
                value = output_counts.get(bucket, "") if isinstance(output_counts, dict) else ""
                cells.append(str(value))
        rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")
    header = ["bucket", "available after exclusion", *output_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Phase33 pair-spec sample</title>",
                "<h1>Phase33 pair-spec sample</h1>",
                f"<p>Seed: <code>{html.escape(str(payload.get('seed', '')))}</code></p>",
                f"<p>Excluded base ids: <strong>{html.escape(str(payload.get('excluded_base_ids', '')))}</strong></p>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr>" + "".join(f"<th>{html.escape(item)}</th>" for item in header) + "</tr>",
                *rows,
                "</table>",
                "<h2>Summary JSON</h2>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def sample_outputs(
    rows: Sequence[dict[str, str]],
    *,
    output_specs: Sequence[OutputSpec],
    splits: Sequence[str],
    target_variants: Sequence[str],
    excluded_base_ids: set[tuple[str, str]],
    seed: int,
    base_disjoint_across_outputs: bool = True,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, int]], dict[str, int]]:
    if not output_specs:
        raise ValueError("at least one output spec is required")
    if not splits:
        raise ValueError("at least one split is required")
    if not target_variants:
        raise ValueError("at least one target variant is required")

    split_set = set(splits)
    variant_set = set(target_variants)
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        split = row.get("split", "")
        variant = row.get("target_variant", "")
        if split not in split_set or variant not in variant_set:
            continue
        if _base_keys(row) & excluded_base_ids:
            continue
        buckets[(split, variant)].append(row)

    rng = random.Random(seed)
    selected: dict[str, list[dict[str, str]]] = {spec.name: [] for spec in output_specs}
    output_base_ids: dict[str, set[tuple[str, str]]] = {spec.name: set() for spec in output_specs}
    counts: dict[str, dict[str, int]] = {spec.name: {} for spec in output_specs}
    available_counts: dict[str, int] = {}

    for split in splits:
        for variant in target_variants:
            bucket_key = (split, variant)
            bucket_name = f"{split}:{variant}"
            candidates = list(buckets.get(bucket_key, []))
            rng.shuffle(candidates)
            available_counts[bucket_name] = len(candidates)
            for spec in output_specs:
                chosen: list[dict[str, str]] = []
                if base_disjoint_across_outputs:
                    unavailable_base_ids = set().union(
                        *(
                            base_ids
                            for output_name, base_ids in output_base_ids.items()
                            if output_name != spec.name
                        )
                    )
                else:
                    unavailable_base_ids = set()
                for candidate in candidates:
                    candidate_base_ids = _base_keys(candidate)
                    if candidate_base_ids & unavailable_base_ids:
                        continue
                    if candidate in chosen:
                        continue
                    chosen.append(candidate)
                    output_base_ids[spec.name].update(candidate_base_ids)
                    if len(chosen) >= spec.per_bucket:
                        break
                if len(chosen) < spec.per_bucket:
                    raise RuntimeError(
                        f"not enough candidates for {bucket_name}:{spec.name}: "
                        f"{len(chosen)} < {spec.per_bucket}"
                    )
                chosen.sort(key=lambda row: int(float(row.get("pair_index", "0") or 0)))
                selected[spec.name].extend(chosen)
                counts[spec.name][bucket_name] = len(chosen)
    return selected, counts, available_counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--target-variant", action="append", required=True)
    parser.add_argument("--output-spec", type=_parse_output_spec, action="append", required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-html", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-base-overlap-across-outputs",
        action="store_true",
        help="Allow the same dataset/base_id to appear in more than one output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_field_groups: list[list[str]] = []
    source_rows: list[dict[str, str]] = []
    for path in args.source_manifest:
        fieldnames, rows = _read_csv_rows(path)
        source_field_groups.append(fieldnames)
        source_rows.extend(rows)
    fieldnames = _merge_fieldnames(source_field_groups)

    excluded_base_ids: set[tuple[str, str]] = set()
    for path in args.exclude_manifest:
        _fieldnames, rows = _read_csv_rows(path)
        for row in rows:
            excluded_base_ids.update(_base_keys(row))

    selected, counts, available_counts = sample_outputs(
        source_rows,
        output_specs=args.output_spec,
        splits=args.split,
        target_variants=args.target_variant,
        excluded_base_ids=excluded_base_ids,
        seed=int(args.seed),
        base_disjoint_across_outputs=not bool(args.allow_base_overlap_across_outputs),
    )

    for spec in args.output_spec:
        _write_manifest(spec.path, selected[spec.name], fieldnames)

    summary = {
        "seed": int(args.seed),
        "source_manifests": [str(path) for path in args.source_manifest],
        "exclude_manifests": [str(path) for path in args.exclude_manifest],
        "excluded_base_ids": len(excluded_base_ids),
        "splits": list(args.split),
        "target_variants": list(args.target_variant),
        "base_disjoint_across_outputs": not bool(args.allow_base_overlap_across_outputs),
        "outputs": {spec.name: str(spec.path) for spec in args.output_spec},
        "per_bucket": {spec.name: spec.per_bucket for spec in args.output_spec},
        "selected_pairs": {name: len(rows) for name, rows in selected.items()},
        "available_counts_after_exclusion": available_counts,
        "counts": counts,
    }
    if args.summary_json is not None:
        _write_json(args.summary_json, summary)
    if args.summary_html is not None:
        _write_html(args.summary_html, summary)
    print(json.dumps({"selected_pairs": summary["selected_pairs"], "counts": counts}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
