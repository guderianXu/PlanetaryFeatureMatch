#!/usr/bin/env python3
"""Build a train-only balanced/protected fov76 phase2j manifest."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from mine_hard_failure_pairs import PAIR_MANIFEST_FIELDS


EXTRA_FIELDS = [
    "phase2j_bucket",
    "phase2j_source",
    "phase2j_pattern",
    "phase2j_sample_rank",
]


@dataclass(frozen=True)
class Phase2JConfig:
    extreme_count: int = 4096
    protected_count: int = 1024
    residual_count: int = 512
    extreme_repeat: int = 1
    protected_repeat: int = 1
    residual_repeat: int = 1
    interleave_cycle: tuple[str, ...] = ()
    seed: int = 20260615
    extreme_variants: tuple[str, ...] = ("extreme_02", "extreme_03")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def pair_identity(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("split", ""),
        row.get("pair_type", ""),
        row.get("reference_pose_id", ""),
        row.get("target_pose_id", ""),
        row.get("crop_a_x0", ""),
        row.get("crop_a_y0", ""),
        row.get("crop_a_x1", ""),
        row.get("crop_a_y1", ""),
        row.get("crop_b_x0", ""),
        row.get("crop_b_y0", ""),
        row.get("crop_b_x1", ""),
        row.get("crop_b_y1", ""),
    )


def _pattern_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("reference_variant", ""), row.get("target_variant", "")


def _train_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("split") == "train"]


def _manifest_row(row: dict[str, str], *, bucket: str, source: str, sample_rank: int) -> dict[str, str]:
    copied = {field: row.get(field, "") for field in PAIR_MANIFEST_FIELDS}
    reference_variant, target_variant = _pattern_key(row)
    copied.update(
        {
            "phase2j_bucket": bucket,
            "phase2j_source": source,
            "phase2j_pattern": f"{reference_variant}->{target_variant}",
            "phase2j_sample_rank": str(sample_rank),
        }
    )
    return copied


def _repeat_rows(rows: list[dict[str, str]], *, repeat: int) -> list[dict[str, str]]:
    if repeat <= 0:
        return []
    repeated: list[dict[str, str]] = []
    for repeat_index in range(repeat):
        for row in rows:
            copied = dict(row)
            if repeat > 1:
                copied["phase2j_source"] = f"{copied.get('phase2j_source', '')}:repeat{repeat_index}"
            repeated.append(copied)
    return repeated


def _interleave_bucket_rows(
    buckets: dict[str, list[dict[str, str]]],
    *,
    cycle: tuple[str, ...],
) -> list[dict[str, str]]:
    if not cycle:
        return [row for bucket in ("residual_hard", "protected_replay", "extreme_main") for row in buckets.get(bucket, [])]
    unknown = sorted(set(cycle) - set(buckets))
    if unknown:
        raise ValueError(f"unknown bucket in interleave cycle: {', '.join(unknown)}")
    queues = {name: list(rows) for name, rows in buckets.items()}
    selected: list[dict[str, str]] = []
    while any(queues.values()):
        made_progress = False
        for name in cycle:
            queue = queues.get(name, [])
            if not queue:
                continue
            selected.append(queue.pop(0))
            made_progress = True
        if not made_progress:
            break
    return selected


def _balanced_sample(
    rows: list[dict[str, str]],
    *,
    count: int,
    seed: int,
    exclude: set[tuple[str, ...]],
) -> list[dict[str, str]]:
    if count <= 0:
        return []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _train_rows(rows):
        if pair_identity(row) in exclude:
            continue
        key = _pattern_key(row)
        if not key[0] or not key[1]:
            continue
        grouped[key].append(row)
    for key, items in grouped.items():
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(items)
    selected: list[dict[str, str]] = []
    pattern_keys = sorted(grouped)
    while len(selected) < count and pattern_keys:
        next_keys: list[tuple[str, str]] = []
        for key in pattern_keys:
            items = grouped[key]
            if not items:
                continue
            selected.append(items.pop())
            if len(selected) >= count:
                break
            if items:
                next_keys.append(key)
        else:
            pattern_keys = next_keys
            continue
        break
    return selected


def _rows_for_patterns(
    train_rows: list[dict[str, str]],
    pattern_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    patterns = {_pattern_key(row) for row in pattern_rows if row.get("reference_variant") and row.get("target_variant")}
    return [row for row in train_rows if _pattern_key(row) in patterns]


def build_phase2j_manifest_rows(
    *,
    extreme_rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    protected_pattern_rows: list[dict[str, str]],
    residual_rows: list[dict[str, str]],
    config: Phase2JConfig,
) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = {
        "residual_hard": [],
        "protected_replay": [],
        "extreme_main": [],
    }
    seen: set[tuple[str, ...]] = set()

    def append_bucket(rows: list[dict[str, str]], *, bucket: str, count: int, seed_offset: int) -> None:
        sampled = _balanced_sample(rows, count=count, seed=config.seed + seed_offset, exclude=seen)
        for row in sampled:
            identity = pair_identity(row)
            if identity in seen:
                continue
            seen.add(identity)
            buckets[bucket].append(_manifest_row(row, bucket=bucket, source=bucket, sample_rank=len(buckets[bucket])))

    append_bucket(residual_rows, bucket="residual_hard", count=config.residual_count, seed_offset=10)

    protected_candidates = _rows_for_patterns(train_rows, protected_pattern_rows)
    append_bucket(protected_candidates, bucket="protected_replay", count=config.protected_count, seed_offset=20)

    extreme_set = set(config.extreme_variants)
    extreme_candidates = [row for row in extreme_rows if row.get("target_variant") in extreme_set]
    append_bucket(extreme_candidates, bucket="extreme_main", count=config.extreme_count, seed_offset=30)

    buckets["residual_hard"] = _repeat_rows(buckets["residual_hard"], repeat=config.residual_repeat)
    buckets["protected_replay"] = _repeat_rows(buckets["protected_replay"], repeat=config.protected_repeat)
    buckets["extreme_main"] = _repeat_rows(buckets["extreme_main"], repeat=config.extreme_repeat)

    selected = _interleave_bucket_rows(buckets, cycle=config.interleave_cycle)
    for index, row in enumerate(selected):
        row["pair_index"] = str(index)
        row["phase2j_sample_rank"] = str(index)
    return selected


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*PAIR_MANIFEST_FIELDS, *EXTRA_FIELDS], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    rows: list[dict[str, str]],
    output_manifest: Path,
    args: argparse.Namespace,
) -> None:
    bucket_counts = Counter(row.get("phase2j_bucket", "") for row in rows)
    pattern_counts = Counter(row.get("phase2j_pattern", "") for row in rows)
    payload = {
        "output_manifest": str(output_manifest),
        "extreme_manifest": str(args.extreme_manifest),
        "train_manifest": str(args.train_manifest),
        "protected_pattern_manifest": [str(path) for path in args.protected_pattern_manifest],
        "residual_manifest": [str(path) for path in args.residual_manifest],
        "seed": int(args.seed),
        "counts": {
            "rows": len(rows),
            "buckets": dict(sorted(bucket_counts.items())),
            "patterns": dict(sorted(pattern_counts.items())),
        },
        "requested": {
            "extreme_count": int(args.extreme_count),
            "protected_count": int(args.protected_count),
            "residual_count": int(args.residual_count),
            "extreme_repeat": int(args.extreme_repeat),
            "protected_repeat": int(args.protected_repeat),
            "residual_repeat": int(args.residual_repeat),
            "interleave_cycle": list(args.interleave_cycle),
            "extreme_variants": list(args.extreme_variant),
        },
    }
    document = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>phase2j balanced manifest</title></head>
<body>
<h1>phase2j balanced/protected manifest</h1>
<pre>{html.escape(json.dumps(payload, indent=2, ensure_ascii=False))}</pre>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extreme-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--protected-pattern-manifest", type=Path, action="append", default=[])
    parser.add_argument("--residual-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report-html", type=Path, default=None)
    parser.add_argument("--extreme-count", type=int, default=4096)
    parser.add_argument("--protected-count", type=int, default=1024)
    parser.add_argument("--residual-count", type=int, default=512)
    parser.add_argument("--extreme-repeat", type=int, default=1)
    parser.add_argument("--protected-repeat", type=int, default=1)
    parser.add_argument("--residual-repeat", type=int, default=1)
    parser.add_argument(
        "--interleave-cycle",
        action="append",
        default=[],
        choices=["residual_hard", "protected_replay", "extreme_main"],
        help="Bucket name to append to the interleave cycle. Repeat flags to set weights.",
    )
    parser.add_argument("--extreme-variant", action="append", default=["extreme_02", "extreme_03"])
    parser.add_argument("--seed", type=int, default=20260615)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.extreme_count < 0 or args.protected_count < 0 or args.residual_count < 0:
        raise ValueError("requested bucket counts must be nonnegative")
    if args.extreme_repeat < 0 or args.protected_repeat < 0 or args.residual_repeat < 0:
        raise ValueError("bucket repeats must be nonnegative")
    if not args.extreme_variant:
        raise ValueError("at least one --extreme-variant is required")
    extreme_rows = read_csv_rows(args.extreme_manifest)
    train_rows = read_csv_rows(args.train_manifest)
    protected_pattern_rows: list[dict[str, str]] = []
    for path in args.protected_pattern_manifest:
        protected_pattern_rows.extend(read_csv_rows(path))
    residual_rows: list[dict[str, str]] = []
    for path in args.residual_manifest:
        residual_rows.extend(read_csv_rows(path))
    rows = build_phase2j_manifest_rows(
        extreme_rows=extreme_rows,
        train_rows=train_rows,
        protected_pattern_rows=protected_pattern_rows,
        residual_rows=residual_rows,
        config=Phase2JConfig(
            extreme_count=int(args.extreme_count),
            protected_count=int(args.protected_count),
            residual_count=int(args.residual_count),
            extreme_repeat=int(args.extreme_repeat),
            protected_repeat=int(args.protected_repeat),
            residual_repeat=int(args.residual_repeat),
            interleave_cycle=tuple(args.interleave_cycle),
            seed=int(args.seed),
            extreme_variants=tuple(args.extreme_variant),
        ),
    )
    write_manifest(args.output_manifest, rows)
    if args.report_html is not None:
        write_report(args.report_html, rows=rows, output_manifest=args.output_manifest, args=args)
    print(f"phase2j_rows={len(rows)} manifest={args.output_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
