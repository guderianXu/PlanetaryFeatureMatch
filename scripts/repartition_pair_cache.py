#!/usr/bin/env python3
"""Create a split-ratio PFM pair cache view using links, copies, or moves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def remove_existing(target: Path) -> None:
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)


def materialize_file(source: Path, target: Path, *, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    remove_existing(target)
    if mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    elif mode == "move":
        shutil.move(str(source), str(target))
    else:
        raise ValueError(f"unsupported link mode: {mode}")


def tree_files(source: Path) -> list[Path]:
    return [path for path in source.rglob("*") if path.is_file()]


def copy_tree_parallel(source: Path, target: Path, *, mode: str, workers: int) -> None:
    files = tree_files(source)
    for child in source.rglob("*"):
        if child.is_dir():
            (target / child.relative_to(source)).mkdir(parents=True, exist_ok=True)

    def materialize_child(child: Path) -> None:
        out = target / child.relative_to(source)
        if mode == "hardlink":
            out.parent.mkdir(parents=True, exist_ok=True)
            remove_existing(out)
            os.link(child, out)
        elif mode == "copy":
            out.parent.mkdir(parents=True, exist_ok=True)
            remove_existing(out)
            shutil.copy2(child, out)
        else:
            raise ValueError(f"unsupported parallel tree mode: {mode}")

    if workers <= 1 or len(files) <= 1:
        for child in files:
            materialize_child(child)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(materialize_child, child) for child in files]
        for future in as_completed(futures):
            future.result()


def materialize_tree(source: Path, target: Path, *, mode: str, workers: int) -> None:
    remove_existing(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target.symlink_to(source.resolve(), target_is_directory=True)
    elif mode == "hardlink":
        target.mkdir(parents=True, exist_ok=True)
        copy_tree_parallel(source, target, mode=mode, workers=workers)
    elif mode == "copy":
        target.mkdir(parents=True, exist_ok=True)
        copy_tree_parallel(source, target, mode=mode, workers=workers)
    elif mode == "move":
        shutil.move(str(source), str(target))
    else:
        raise ValueError(f"unsupported link mode: {mode}")


def materialize_path(source: Path, target: Path, *, mode: str, workers: int) -> None:
    if source.is_dir():
        materialize_tree(source, target, mode=mode, workers=workers)
    else:
        materialize_file(source, target, mode=mode)


def materialize_pair_record(
    *,
    output_root: Path,
    split: str,
    index: int,
    record: PairRecord,
    link_mode: str,
) -> dict[str, str]:
    source_name = f"source_repart_{index:06d}_{record.old_split}_{record.source_dir}"
    target = output_root / "cache" / split / source_name / record.pair_path.name
    materialize_file(record.pair_path, target, mode=link_mode)
    sidecar = record.pair_path.with_suffix(".json")
    if sidecar.exists():
        materialize_file(sidecar, target.with_suffix(".json"), mode=link_mode)
    row = dict(record.row or {})
    row["split"] = split
    row["pair_path"] = str(target.resolve(strict=False))
    return row


def write_split(
    *,
    output_root: Path,
    split: str,
    records: list[PairRecord],
    link_mode: str,
    workers: int,
) -> list[dict[str, str]]:
    if workers <= 1 or len(records) <= 1:
        return [
            materialize_pair_record(
                output_root=output_root,
                split=split,
                index=index,
                record=record,
                link_mode=link_mode,
            )
            for index, record in enumerate(records)
        ]
    rows: list[dict[str, str] | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                materialize_pair_record,
                output_root=output_root,
                split=split,
                index=index,
                record=record,
                link_mode=link_mode,
            ): index
            for index, record in enumerate(records)
        }
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    return [row for row in rows if row is not None]


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
    parser = argparse.ArgumentParser(description="Repartition a PFM pair cache into train/val/test splits.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ratio", type=parse_ratio, default=parse_ratio("7:2:1"))
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy", "move"),
        default="symlink",
        help=(
            "How to materialize pair files and shared assets. Use copy/move when "
            "the repartitioned cache must be self-contained on another disk."
        ),
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel file workers for pair files and copied trees.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
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
        rows = write_split(
            output_root=output_root,
            split=split,
            records=split_items,
            link_mode=args.link_mode,
            workers=args.workers,
        )
        write_manifest(output_root / "manifests" / f"repartition_{split}.csv", rows)
        all_rows.extend(rows)
    write_manifest(output_root / "manifests" / "repartition_all.csv", all_rows)

    # Compact pair payloads store image paths relative to each pair file, so the
    # shared image store must exist at the same relative dataset-root location.
    for name in ("tsai_tracks", "dataset_metadata.json", "image_store"):
        source = input_root / name
        target = output_root / name
        if source.exists():
            materialize_path(source, target, mode=args.link_mode, workers=args.workers)

    metadata = {
        "source_dataset_root": str(input_root),
        "ratio": list(args.ratio),
        "seed": args.seed,
        "total_pairs": len(records),
        "splits": {split: len(items) for split, items in split_records.items()},
        "link_type": args.link_mode,
        "workers": args.workers,
    }
    (output_root / "repartition_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
