#!/usr/bin/env python3
"""Create a split-ratio view of an existing PFM pair cache using symlinks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairRecord:
    pair_path: Path
    old_split: str
    source_dir: str
    row: dict[str, str] | None


def index_keys(path: Path) -> list[str]:
    keys = [path.resolve(strict=False).as_posix(), path.as_posix()]
    parts = path.parts
    for index, part in enumerate(parts):
        if part == "cache" and index + 3 < len(parts):
            keys.append(Path(*parts[index:]).as_posix())
    return list(dict.fromkeys(keys))


def load_manifest_rows(root: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for manifest_path in sorted((root / "manifests").glob("*.csv")):
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                pair_value = row.get("pair_path")
                if not pair_value:
                    continue
                pair_path = Path(pair_value)
                for key in index_keys(pair_path):
                    rows.setdefault(key, dict(row))
    return rows


def discover_pairs(root: Path) -> list[PairRecord]:
    manifest_rows = load_manifest_rows(root)
    records: list[PairRecord] = []
    for split in ("train", "val", "test"):
        for pair_path in sorted((root / "cache" / split).glob("source_*/pair_*.pt")):
            row = None
            for key in index_keys(pair_path):
                row = manifest_rows.get(key)
                if row is not None:
                    break
            records.append(
                PairRecord(
                    pair_path=pair_path,
                    old_split=split,
                    source_dir=pair_path.parent.name,
                    row=row,
                )
            )
    return records


def split_counts(total: int, ratio: tuple[int, int, int]) -> dict[str, int]:
    ratio_sum = sum(ratio)
    train = int(total * ratio[0] / ratio_sum)
    val = int(total * ratio[1] / ratio_sum)
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def safe_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def write_split(
    *,
    output_root: Path,
    split: str,
    records: list[PairRecord],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, record in enumerate(records):
        source_name = f"source_repart_{index:06d}_{record.old_split}_{record.source_dir}"
        target = output_root / "cache" / split / source_name / record.pair_path.name
        safe_symlink(record.pair_path, target)
        sidecar = record.pair_path.with_suffix(".json")
        if sidecar.exists():
            safe_symlink(sidecar, target.with_suffix(".json"))
        row = dict(record.row or {})
        row["split"] = split
        row["pair_path"] = str(target.resolve(strict=False))
        rows.append(row)
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_ratio(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(":")]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("ratio must look like 7:2:1")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repartition a PFM pair cache into train/val/test symlink splits.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ratio", type=parse_ratio, default=parse_ratio("7:2:1"))
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = discover_pairs(input_root)
    rng = random.Random(args.seed)
    records = sorted(records, key=lambda record: record.pair_path.as_posix())
    rng.shuffle(records)
    counts = split_counts(len(records), args.ratio)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    split_records = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    all_rows: list[dict[str, str]] = []
    for split, split_items in split_records.items():
        rows = write_split(output_root=output_root, split=split, records=split_items)
        write_manifest(output_root / "manifests" / f"repartition_{split}.csv", rows)
        all_rows.extend(rows)
    write_manifest(output_root / "manifests" / "repartition_all.csv", all_rows)

    for name in ("tsai_tracks", "dataset_metadata.json"):
        source = input_root / name
        target = output_root / name
        if source.exists():
            safe_symlink(source, target)

    metadata = {
        "source_dataset_root": str(input_root),
        "ratio": list(args.ratio),
        "seed": args.seed,
        "total_pairs": len(records),
        "splits": {split: len(items) for split, items in split_records.items()},
        "link_type": "symlink",
    }
    (output_root / "repartition_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
