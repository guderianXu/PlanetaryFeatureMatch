#!/usr/bin/env python3
"""Generate PFM pair caches from a pose-simulation render manifest.

The input render manifest contains image/depth/TSAI records for each simulated
camera variant. This script groups rows by base_id and creates supervised
pair_*.pt archives by projecting the reference camera into each target variant.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for candidate in (PYTHON_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from compact_pair_cache import make_compact_pair_payload, save_shared_image  # noqa: E402
from generate_cross_position_pose_pairs import (  # noqa: E402
    parse_tsai,
    project_warp,
    read_float_tif,
)


DEFAULT_TARGET_VARIANTS = (
    "small_01",
    "small_02",
    "small_03",
    "mid_01",
    "mid_02",
    "mid_03",
    "extreme_01",
    "extreme_02",
    "extreme_03",
)


@dataclass(frozen=True)
class RenderRecord:
    pose_id: str
    base_id: str
    variant: str
    split: str
    tsai_path: Path
    image_path: Path
    uint8_path: Path | None
    depth_path: Path


@dataclass(frozen=True)
class PairTask:
    pair_index: int
    split: str
    reference: RenderRecord
    target: RenderRecord
    output_root: Path
    min_overlap: float
    absolute_depth_tolerance_m: float
    relative_depth_tolerance: float
    image_source: str


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_uint8_manifest(path: Path | None) -> dict[str, Path]:
    if path is None or not path.exists():
        return {}
    mapping: dict[str, Path] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source_path", "")
            target = row.get("uint8_path", "")
            if not source or not target or source == "source_path":
                continue
            mapping[str(Path(source))] = Path(target)
    return mapping


def read_render_manifest(path: Path, uint8_paths: dict[str, Path]) -> list[RenderRecord]:
    records: list[RenderRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = Path(row["image_path"])
            uint8_path = uint8_paths.get(str(image_path))
            records.append(
                RenderRecord(
                    pose_id=row["pose_id"],
                    base_id=row["base_id"],
                    variant=row["variant"],
                    split=row["split"],
                    tsai_path=Path(row["tsai_path"]),
                    image_path=image_path,
                    uint8_path=uint8_path,
                    depth_path=Path(row["depth_path"]),
                )
            )
    return records


def selected_image_path(record: RenderRecord, image_source: str) -> Path:
    if image_source == "uint8" and record.uint8_path is not None:
        return record.uint8_path
    return record.image_path


def load_view(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(image).to(torch.float32)
    if image.dtype.name == "uint8":
        tensor = tensor / 255.0
    elif image.dtype.name == "uint16":
        tensor = tensor / 65535.0
    elif float(tensor.max().item()) > 1.5:
        tensor = tensor / float(tensor.max().clamp_min(1.0).item())
    return tensor.nan_to_num(0.0, 0.0, 0.0).clamp(0.0, 1.0).unsqueeze(0).contiguous()


def pair_output_path(output_root: Path, split: str, pair_index: int, reference: RenderRecord, target: RenderRecord) -> Path:
    source_name = (
        f"source_pose_{pair_index:06d}_{safe_name(reference.base_id)}_"
        f"{safe_name(reference.variant)}_to_{safe_name(target.variant)}"
    )
    pair_name = f"pair_{pair_index:06d}_{safe_name(reference.variant)}_to_{safe_name(target.variant)}.pt"
    return output_root / "cache" / split / source_name / pair_name


def existing_pair_metrics(path: Path) -> dict[str, float] | None:
    json_path = path.with_suffix(".json")
    if not path.exists() or not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def generate_pair(task: PairTask) -> dict[str, object]:
    out_path = pair_output_path(task.output_root, task.split, task.pair_index, task.reference, task.target)
    metrics = existing_pair_metrics(out_path)
    reused = metrics is not None
    if metrics is None:
        for record in (task.reference, task.target):
            for path in (
                selected_image_path(record, task.image_source),
                record.depth_path,
                record.tsai_path,
            ):
                if not path.exists():
                    raise FileNotFoundError(path)
        view_a = load_view(selected_image_path(task.reference, task.image_source))
        view_b = load_view(selected_image_path(task.target, task.image_source))
        depth_a = read_float_tif(task.reference.depth_path)
        depth_b = read_float_tif(task.target.depth_path)
        camera_a = parse_tsai(task.reference.tsai_path)
        camera_b = parse_tsai(task.target.tsai_path)
        warp, valid_mask, metrics = project_warp(
            depth_a,
            depth_b,
            camera_a,
            camera_b,
            absolute_depth_tolerance_m=task.absolute_depth_tolerance_m,
            relative_depth_tolerance=task.relative_depth_tolerance,
        )
        if float(metrics["valid_pair_fraction"]) < task.min_overlap:
            return {
                "status": "low_overlap",
                "pair_index": task.pair_index,
                "split": task.split,
                "base_id": task.reference.base_id,
                "reference_variant": task.reference.variant,
                "target_variant": task.target.variant,
                **metrics,
            }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image_store = task.output_root / "shared_images"
        image_a_path = save_shared_image(view_a, image_store)
        image_b_path = save_shared_image(view_b, image_store)
        payload = make_compact_pair_payload(
            pair_path=out_path,
            image_a_path=image_a_path,
            image_b_path=image_b_path,
            warp_a_to_b=warp,
            valid_mask=valid_mask,
        )
        tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
        out_path.with_suffix(".json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "status": "reused" if reused else "kept",
        "pair_index": task.pair_index,
        "split": task.split,
        "base_id": task.reference.base_id,
        "reference_pose_id": task.reference.pose_id,
        "target_pose_id": task.target.pose_id,
        "reference_variant": task.reference.variant,
        "target_variant": task.target.variant,
        "reference_image": str(selected_image_path(task.reference, task.image_source)),
        "target_image": str(selected_image_path(task.target, task.image_source)),
        "reference_depth": str(task.reference.depth_path),
        "target_depth": str(task.target.depth_path),
        "reference_tsai": str(task.reference.tsai_path),
        "target_tsai": str(task.target.tsai_path),
        "pair_path": str(out_path),
        **metrics,
    }


def build_tasks(
    records: list[RenderRecord],
    *,
    output_root: Path,
    reference_variant: str,
    target_variants: tuple[str, ...],
    bidirectional: bool,
    min_overlap: float,
    absolute_depth_tolerance_m: float,
    relative_depth_tolerance: float,
    image_source: str,
    max_pairs_per_split: int,
) -> list[PairTask]:
    by_base: dict[str, dict[str, RenderRecord]] = defaultdict(dict)
    for record in records:
        if selected_image_path(record, image_source).exists() and record.depth_path.exists() and record.tsai_path.exists():
            by_base[record.base_id][record.variant] = record
    tasks: list[PairTask] = []
    pair_index = 0
    kept_by_split: dict[str, int] = defaultdict(int)
    for base_id in sorted(by_base):
        variants = by_base[base_id]
        reference = variants.get(reference_variant)
        if reference is None:
            continue
        for variant in target_variants:
            target = variants.get(variant)
            if target is None:
                continue
            pairs = [(reference, target)]
            if bidirectional:
                pairs.append((target, reference))
            for source, destination in pairs:
                split = source.split
                if max_pairs_per_split > 0 and kept_by_split[split] >= max_pairs_per_split:
                    continue
                tasks.append(
                    PairTask(
                        pair_index=pair_index,
                        split=split,
                        reference=source,
                        target=destination,
                        output_root=output_root,
                        min_overlap=min_overlap,
                        absolute_depth_tolerance_m=absolute_depth_tolerance_m,
                        relative_depth_tolerance=relative_depth_tolerance,
                        image_source=image_source,
                    )
                )
                pair_index += 1
                kept_by_split[split] += 1
    return tasks


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_index",
        "split",
        "base_id",
        "reference_pose_id",
        "target_pose_id",
        "reference_variant",
        "target_variant",
        "reference_image",
        "target_image",
        "reference_depth",
        "target_depth",
        "reference_tsai",
        "target_tsai",
        "pair_path",
        "valid_pair_fraction",
        "valid_pixels",
        "target_inside_fraction",
        "valid_a_fraction",
        "height",
        "width",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_html_report(path: Path, metadata: dict[str, object], rows: list[dict[str, object]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["split"])] += 1
    items = "".join(
        f"<li>{html.escape(split)}: {count}</li>"
        for split, count in sorted(counts.items())
    )
    content = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Pose Manifest Pair Cache</title></head>
<body>
<h1>Pose Manifest Pair Cache</h1>
<h2>输入</h2>
<pre>{html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))}</pre>
<h2>输出统计</h2>
<ul>{items}</ul>
<p>保留 pair 数：{len(rows)}</p>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--uint8-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-variant", default="nadir")
    parser.add_argument("--target-variant", action="append", default=[])
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--min-overlap", type=float, default=0.05)
    parser.add_argument("--absolute-depth-tolerance-m", type=float, default=100.0)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.005)
    parser.add_argument("--image-source", choices=["uint8", "render"], default="uint8")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-pairs-per-split", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.min_overlap < 0.0 or args.min_overlap > 1.0:
        raise ValueError("--min-overlap must be in [0, 1]")
    args.output_root.mkdir(parents=True, exist_ok=True)
    uint8_paths = read_uint8_manifest(args.uint8_manifest)
    records = read_render_manifest(args.render_manifest, uint8_paths)
    target_variants = tuple(args.target_variant) if args.target_variant else DEFAULT_TARGET_VARIANTS
    tasks = build_tasks(
        records,
        output_root=args.output_root,
        reference_variant=args.reference_variant,
        target_variants=target_variants,
        bidirectional=args.bidirectional,
        min_overlap=args.min_overlap,
        absolute_depth_tolerance_m=args.absolute_depth_tolerance_m,
        relative_depth_tolerance=args.relative_depth_tolerance,
        image_source=args.image_source,
        max_pairs_per_split=args.max_pairs_per_split,
    )
    print(f"candidate_tasks={len(tasks)} workers={args.workers}", flush=True)
    rows: list[dict[str, object]] = []
    low_overlap = 0
    errors = 0
    if args.workers == 1:
        iterator = ((task, generate_pair(task)) for task in tasks)
        for task, result in iterator:
            if result["status"] == "low_overlap":
                low_overlap += 1
            else:
                rows.append(result)
            if len(rows) == 1 or len(rows) % 50 == 0:
                print(
                    f"kept={len(rows)} task={task.pair_index} split={task.split} "
                    f"{task.reference.variant}->{task.target.variant}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate_pair, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    errors += 1
                    print(
                        f"skip_error task={task.pair_index} split={task.split} "
                        f"{task.reference.base_id} {task.reference.variant}->{task.target.variant}: {exc}",
                        flush=True,
                    )
                    continue
                if result["status"] == "low_overlap":
                    low_overlap += 1
                else:
                    rows.append(result)
                if len(rows) == 1 or len(rows) % 50 == 0:
                    print(
                        f"kept={len(rows)} done={len(rows) + low_overlap + errors}/{len(tasks)} "
                        f"split={task.split} {task.reference.variant}->{task.target.variant}",
                        flush=True,
                    )
    rows.sort(key=lambda row: int(row["pair_index"]))
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        write_manifest(args.output_root / "manifests" / f"pose_manifest_{split}.csv", split_rows)
    write_manifest(args.output_root / "manifests" / "pose_manifest_all.csv", rows)
    metadata = {
        "render_manifest": str(args.render_manifest),
        "uint8_manifest": str(args.uint8_manifest) if args.uint8_manifest else None,
        "image_source": args.image_source,
        "reference_variant": args.reference_variant,
        "target_variants": list(target_variants),
        "bidirectional": args.bidirectional,
        "min_overlap": args.min_overlap,
        "absolute_depth_tolerance_m": args.absolute_depth_tolerance_m,
        "relative_depth_tolerance": args.relative_depth_tolerance,
        "candidate_tasks": len(tasks),
        "kept_pairs": len(rows),
        "low_overlap": low_overlap,
        "errors": errors,
    }
    (args.output_root / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_html_report(args.output_root / "dataset_report.html", metadata, rows)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
