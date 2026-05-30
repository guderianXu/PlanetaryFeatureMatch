#!/usr/bin/env python3
"""Export RootSIFT/H-RANSAC pseudo labels from 1024 cache pairs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import pseudo_label_generation as plg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--limit-pairs-per-cache", type=int, default=16)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--max-raw-matches", type=int, default=512)
    parser.add_argument("--max-labels-per-pair", type=int, default=128)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--truth-threshold-px", type=float, default=3.0)
    parser.add_argument("--min-labels-per-pair", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_csv = args.summary_csv or args.output_csv.with_name("pseudo_label_summary.csv")
    selected = plg.discover_pair_archives(args.cache_dir, limit_per_cache=args.limit_pairs_per_cache, seed=args.seed)
    if not selected:
        raise RuntimeError("no pair_*.pt archives found")

    all_rows: list[plg.PseudoLabelRow] = []
    summaries: list[plg.PairSummaryRow] = []
    kept_pairs = 0
    for index, (cache_dir, pair_path) in enumerate(selected, start=1):
        try:
            rows, summary = plg.generate_for_pair(
                cache_dir,
                pair_path,
                max_keypoints=args.max_keypoints,
                max_raw_matches=args.max_raw_matches,
                ratio=args.ratio,
                sift_contrast=args.sift_contrast,
                ransac_threshold_px=args.ransac_threshold_px,
                truth_threshold_px=args.truth_threshold_px,
                max_labels_per_pair=args.max_labels_per_pair,
                seed=args.seed + index,
            )
            if len(rows) >= args.min_labels_per_pair:
                all_rows.extend(rows)
                kept_pairs += 1
            else:
                summary = plg.PairSummaryRow(
                    cache_dir=summary.cache_dir,
                    pair_pt=summary.pair_pt,
                    status="too_few_labels",
                    keypoints_a=summary.keypoints_a,
                    keypoints_b=summary.keypoints_b,
                    raw_matches=summary.raw_matches,
                    ransac_matches=summary.ransac_matches,
                    truth_filtered_matches=summary.truth_filtered_matches,
                    mean_error_px=summary.mean_error_px,
                    message=f"min_labels_per_pair={args.min_labels_per_pair}",
                )
            summaries.append(summary)
            print(
                f"{index:04d}/{len(selected):04d} labels={summary.truth_filtered_matches:4d} "
                f"raw={summary.raw_matches:4d} ransac={summary.ransac_matches:4d} "
                f"status={summary.status} pair={pair_path}",
                flush=True,
            )
        except Exception as exc:
            summaries.append(
                plg.PairSummaryRow(
                    cache_dir=cache_dir.as_posix(),
                    pair_pt=pair_path.as_posix(),
                    status="error",
                    keypoints_a=0,
                    keypoints_b=0,
                    raw_matches=0,
                    ransac_matches=0,
                    truth_filtered_matches=0,
                    mean_error_px=math.nan,
                    message=str(exc),
                )
            )
            print(f"{index:04d}/{len(selected):04d} status=error pair={pair_path} message={exc}", flush=True)

    plg.write_pseudo_label_csv(args.output_csv, all_rows)
    plg.write_summary_csv(summary_csv, summaries)
    print(f"selected_pairs={len(selected)} kept_pairs={kept_pairs} labels={len(all_rows)}")
    print(f"pseudo_labels={args.output_csv}")
    print(f"summary={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
