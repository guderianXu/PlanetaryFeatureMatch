#!/usr/bin/env python3
"""Evaluate generated synthetic pair caches with warp_a_to_b match metrics."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


METRIC_RE = re.compile(
    r"sparse_matches=(?P<sparse>\d+).*?"
    r"correct_matches=(?P<correct>\d+).*?"
    r"wrong_matches=(?P<wrong>\d+).*?"
    r"match_precision=(?P<precision>[-+0-9.eE]+).*?"
    r"elapsed=(?P<elapsed>[-+0-9.eE]+)s"
)


@dataclass(frozen=True)
class PairEntry:
    cache_dir: Path
    source_dir: Path
    pair_pt: Path
    image_a: Path
    image_b: Path
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pfm_cli match on cached synthetic pairs and score against warp_a_to_b."
    )
    parser.add_argument("--cache-dir", action="append", required=True, help="Synthetic pair cache root; repeatable")
    parser.add_argument(
        "--pair-kind",
        default="cache",
        choices=["cache", "named-rotation"],
        help="Evaluate pair_*.pt cache entries or named rotation *_DDD.pt entries",
    )
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint to evaluate")
    parser.add_argument("--pfm-cli", default="./build-pfm-verify-mamba/pfm_cli", help="pfm_cli executable")
    parser.add_argument("--output-dir", default="runs/cache_match_eval", help="Directory for matches/logs/CSV")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--match-mode", default="sparse", choices=["sparse", "dense", "both"])
    parser.add_argument(
        "--sparse-geometry-filter",
        default="rotation-only",
        choices=["projective", "rotation-only"],
        help="Geometry filter passed to pfm_cli match",
    )
    parser.add_argument("--threshold-px", type=float, default=5.0, help="Warp correctness threshold")
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--min-keypoints", type=int, default=0)
    parser.add_argument("--keypoint-grid-rows", type=int, default=24)
    parser.add_argument("--keypoint-grid-cols", type=int, default=24)
    parser.add_argument("--keypoints-per-cell", type=int, default=6)
    parser.add_argument("--nms-radius", type=int, default=2)
    parser.add_argument("--descriptor-topk", type=int, default=4)
    parser.add_argument("--texture-blend-weight", default=None)
    parser.add_argument("--limit-per-cache", type=int, default=0, help="Deterministic evenly-spaced sample count")
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many sorted entries per cache")
    parser.add_argument("--visualize-failures", type=int, default=0, help="Write visualizations for first N low-precision pairs")
    parser.add_argument("--failure-precision", type=float, default=0.9)
    parser.add_argument("--resume", action="store_true", help="Reuse rows already present in summary.csv")
    return parser.parse_args()


def source_view_a(source_dir: Path) -> Path | None:
    matches = sorted(source_dir.glob("source_*_view_a.png"))
    return matches[0] if matches else None


def discover_cache_pairs(cache_dir: Path) -> list[PairEntry]:
    entries: list[PairEntry] = []
    for source_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        image_a = source_view_a(source_dir)
        if image_a is None:
            continue
        for pair_pt in sorted(source_dir.glob("pair_*.pt")):
            image_b = pair_pt.with_name(pair_pt.stem + "_view_b.png")
            if image_b.exists():
                entries.append(PairEntry(cache_dir, source_dir, pair_pt, image_a, image_b, "cache"))
    return entries


def discover_named_rotation_pairs(cache_dir: Path) -> list[PairEntry]:
    entries: list[PairEntry] = []
    pattern = re.compile(r".+_\d{3}\.pt$")
    for source_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        image_a = source_view_a(source_dir)
        if image_a is None:
            continue
        for pair_pt in sorted(source_dir.glob("*.pt")):
            if pair_pt.name.startswith("pair_") or pair_pt.name == "manifest.pt":
                continue
            if not pattern.match(pair_pt.name):
                continue
            image_b = pair_pt.with_suffix(".tif")
            if image_b.exists():
                entries.append(PairEntry(cache_dir, source_dir, pair_pt, image_a, image_b, "named-rotation"))
    return entries


def discover_pairs(cache_dir: Path, pair_kind: str) -> list[PairEntry]:
    if pair_kind == "cache":
        return discover_cache_pairs(cache_dir)
    if pair_kind == "named-rotation":
        return discover_named_rotation_pairs(cache_dir)
    raise ValueError(f"unsupported pair kind: {pair_kind}")


def deterministic_sample(entries: list[PairEntry], start_index: int, limit: int) -> list[PairEntry]:
    entries = entries[max(0, start_index) :]
    if limit <= 0 or len(entries) <= limit:
        return entries
    if limit == 1:
        return [entries[0]]
    step = (len(entries) - 1) / float(limit - 1)
    indices = sorted({round(i * step) for i in range(limit)})
    sampled = [entries[index] for index in indices]
    return sampled[:limit]


def stable_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_done_pairs(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["pair_pt"] for row in reader if row.get("returncode") == "0"}


def run_pair(
    args: argparse.Namespace,
    entry: PairEntry,
    output_dir: Path,
    workspace: Path,
    visualize: bool,
) -> dict[str, str]:
    cache_name = entry.cache_dir.name
    source_name = entry.source_dir.name
    pair_name = entry.pair_pt.stem
    pair_out = output_dir / cache_name / source_name / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)
    log_path = pair_out / "match.log"
    match_path = pair_out / "matches.pt"

    command = [
        args.pfm_cli,
        "match",
        "--image-a",
        str(entry.image_a),
        "--image-b",
        str(entry.image_b),
        "--checkpoint",
        args.checkpoint,
        "--output",
        str(match_path),
        "--device",
        args.device,
        "--match-mode",
        args.match_mode,
        "--sparse-geometry-filter",
        args.sparse_geometry_filter,
        "--warp-a-to-b",
        str(entry.pair_pt),
        "--match-correct-threshold-pixels",
        str(args.threshold_px),
        "--max-keypoints",
        str(args.max_keypoints),
        "--min-keypoints",
        str(args.min_keypoints),
        "--keypoint-grid-rows",
        str(args.keypoint_grid_rows),
        "--keypoint-grid-cols",
        str(args.keypoint_grid_cols),
        "--keypoints-per-cell",
        str(args.keypoints_per_cell),
        "--nms-radius",
        str(args.nms_radius),
    ]
    if visualize:
        command += ["--visualization-dir", str(pair_out)]

    env = os.environ.copy()
    env["PFM_DESCRIPTOR_TOPK_CANDIDATES"] = str(args.descriptor_topk)
    if args.texture_blend_weight is not None:
        env["PFM_TEXTURE_BLEND_WEIGHT"] = str(args.texture_blend_weight)

    process = subprocess.run(
        command,
        cwd=workspace,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(process.stdout, encoding="utf-8")

    sparse = correct = wrong = 0
    precision = elapsed = 0.0
    match = METRIC_RE.search(process.stdout)
    if match:
        sparse = int(match.group("sparse"))
        correct = int(match.group("correct"))
        wrong = int(match.group("wrong"))
        precision = float(match.group("precision"))
        elapsed = float(match.group("elapsed"))

    return {
        "cache_dir": stable_rel(entry.cache_dir, workspace),
        "source_dir": stable_rel(entry.source_dir, workspace),
        "pair_pt": stable_rel(entry.pair_pt, workspace),
        "image_a": stable_rel(entry.image_a, workspace),
        "image_b": stable_rel(entry.image_b, workspace),
        "pair_kind": entry.kind,
        "sparse_matches": str(sparse),
        "correct_matches": str(correct),
        "wrong_matches": str(wrong),
        "match_precision": f"{precision:.6f}",
        "elapsed_seconds": f"{elapsed:.3f}",
        "returncode": str(process.returncode),
        "output_dir": stable_rel(pair_out, workspace),
    }


def summarize(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "pairs=0"
    total = sum(int(row["sparse_matches"]) for row in rows)
    correct = sum(int(row["correct_matches"]) for row in rows)
    failed = sum(1 for row in rows if row["returncode"] != "0")
    precision = 0.0 if total == 0 else correct / total
    low = sum(1 for row in rows if float(row["match_precision"]) < 0.9)
    return (
        f"pairs={len(rows)} failed={failed} total_matches={total} "
        f"correct={correct} precision={precision:.6f} low_precision_pairs={low}"
    )


def main() -> int:
    args = parse_args()
    workspace = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    fields = [
        "cache_dir",
        "source_dir",
        "pair_pt",
        "image_a",
        "image_b",
        "pair_kind",
        "sparse_matches",
        "correct_matches",
        "wrong_matches",
        "match_precision",
        "elapsed_seconds",
        "returncode",
        "output_dir",
    ]
    done = load_done_pairs(summary_path) if args.resume else set()
    all_rows: list[dict[str, str]] = []
    visualize_remaining = max(0, args.visualize_failures)

    mode = "a" if args.resume and summary_path.exists() else "w"
    with summary_path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for cache_arg in args.cache_dir:
            cache_dir = Path(cache_arg)
            entries = discover_pairs(cache_dir, args.pair_kind)
            selected = deterministic_sample(entries, args.start_index, args.limit_per_cache)
            print(f"cache={cache_dir} discovered={len(entries)} selected={len(selected)}", flush=True)
            for index, entry in enumerate(selected, 1):
                pair_key = stable_rel(entry.pair_pt, workspace)
                if pair_key in done:
                    continue
                visualize = visualize_remaining > 0
                row = run_pair(args, entry, output_dir, workspace, visualize)
                if visualize and float(row["match_precision"]) < args.failure_precision:
                    visualize_remaining -= 1
                writer.writerow(row)
                handle.flush()
                all_rows.append(row)
                print(
                    f"[{index}/{len(selected)}] {pair_key} "
                    f"matches={row['sparse_matches']} correct={row['correct_matches']} "
                    f"precision={row['match_precision']} rc={row['returncode']}",
                    flush=True,
                )

    print(summarize(all_rows))
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
