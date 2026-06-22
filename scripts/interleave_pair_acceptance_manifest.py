#!/usr/bin/env python3
"""Interleave pair-acceptance manifest rows by accept/reject label."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter
from pathlib import Path
from typing import Sequence


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        rows = [
            {key: ("" if value is None else value) for key, value in row.items() if key is not None}
            for row in reader
        ]
    return list(reader.fieldnames), rows


def max_same_label_run(rows: Sequence[dict[str, str]]) -> int:
    max_run = 0
    current_label = None
    current_run = 0
    for row in rows:
        label = row.get("pair_accept_label", "")
        if label == current_label:
            current_run += 1
        else:
            current_label = label
            current_run = 1
        max_run = max(max_run, current_run)
    return max_run


def interleave_manifest_rows(
    rows: Sequence[dict[str, str]],
    *,
    seed: int,
    reject_repeat: int = 1,
) -> list[dict[str, str]]:
    if reject_repeat <= 0:
        raise ValueError("reject_repeat must be positive")
    accepts = [dict(row) for row in rows if row.get("pair_accept_label", "") == "1"]
    rejects = [dict(row) for row in rows if row.get("pair_accept_label", "") == "0"]
    if not accepts or not rejects:
        raise ValueError("input manifest must contain both pair_accept_label=1 and pair_accept_label=0 rows")

    rng = random.Random(seed)
    rng.shuffle(accepts)
    rng.shuffle(rejects)
    reject_pool = []
    for copy_index in range(reject_repeat):
        for row in rejects:
            item = dict(row)
            item["pair_accept_interleave_reject_copy_index"] = str(copy_index)
            reject_pool.append(item)
    rng.shuffle(reject_pool)

    label_rows = {
        "0": reject_pool,
        "1": accepts,
    }
    target_counts = {label: len(items) for label, items in label_rows.items()}
    consumed = {"0": 0, "1": 0}
    total_rows = target_counts["0"] + target_counts["1"]
    output: list[dict[str, str]] = []
    for position in range(total_rows):
        choices = [
            label
            for label in ("0", "1")
            if consumed[label] < target_counts[label]
        ]
        if len(choices) == 1:
            label = choices[0]
        else:
            deficits = {
                label: ((position + 1) * target_counts[label] / total_rows) - consumed[label]
                for label in choices
            }
            label = max(choices, key=lambda item: (deficits[item], item))
        item = dict(label_rows[label][consumed[label]])
        item.setdefault("pair_accept_interleave_reject_copy_index", "")
        output.append(item)
        consumed[label] += 1
    return output


def write_manifest(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    if "pair_accept_interleave_reject_copy_index" not in fields:
        fields.append("pair_accept_interleave_reject_copy_index")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_html(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Interleaved pair acceptance manifest</title>",
                "<h1>Interleaved pair acceptance manifest</h1>",
                "<p>source=<code>pair_accept_label from true-geometry supervision</code></p>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_summary(
    *,
    input_manifest: Path,
    output_manifest: Path,
    input_rows: Sequence[dict[str, str]],
    output_rows: Sequence[dict[str, str]],
    reject_repeat: int,
    seed: int,
) -> dict[str, object]:
    label_counts = Counter(row.get("pair_accept_label", "") for row in output_rows)
    input_label_counts = Counter(row.get("pair_accept_label", "") for row in input_rows)
    return {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "seed": int(seed),
        "reject_repeat": int(reject_repeat),
        "input_rows": len(input_rows),
        "output_rows": len(output_rows),
        "input_label_counts": dict(sorted(input_label_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "max_same_label_run": max_same_label_run(output_rows),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--report-html", type=Path, required=True)
    parser.add_argument("--reject-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fieldnames, rows = read_manifest(args.input_manifest)
    output_rows = interleave_manifest_rows(
        rows,
        seed=int(args.seed),
        reject_repeat=int(args.reject_repeat),
    )
    summary = build_summary(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        input_rows=rows,
        output_rows=output_rows,
        reject_repeat=int(args.reject_repeat),
        seed=int(args.seed),
    )
    write_manifest(args.output_manifest, fieldnames, output_rows)
    write_json(args.summary_json, summary)
    write_html(args.report_html, summary)
    print(
        "interleaved_pair_acceptance_manifest "
        f"input_rows={summary['input_rows']} "
        f"output_rows={summary['output_rows']} "
        f"labels={summary['label_counts']} "
        f"output={args.output_manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
