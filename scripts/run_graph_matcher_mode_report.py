#!/usr/bin/env python3
"""Run a visual report with a named GraphMatcher v2.1 evaluation profile."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from graph_matcher_modes import graph_matcher_mode_config, graph_matcher_mode_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=graph_matcher_mode_names())
    parser.add_argument("--validation-cache-dir", action="append", type=Path, required=True)
    parser.add_argument("--pose-metadata-root", action="append", type=Path, default=[])
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=18)
    parser.add_argument("--sample-seed", type=int, default=20260527)
    parser.add_argument("--required-sample-glob", action="append", default=[])
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = graph_matcher_mode_config(args.mode)
    state_path = cfg.resolved_state_path(ROOT)
    run_dir = args.run_dir or state_path.parent
    command = [
        sys.executable,
        str(ROOT / "scripts" / "training_visual_report.py"),
        "--run-dir",
        str(run_dir),
        "--pytorch-state",
        str(state_path),
        "--device",
        args.device,
        "--sample-count",
        str(args.sample_count),
        "--sample-seed",
        str(args.sample_seed),
        *cfg.training_visual_report_args()[2:],
    ]
    for path in args.validation_cache_dir:
        command.extend(["--validation-cache-dir", str(path)])
    for path in args.pose_metadata_root:
        command.extend(["--pose-metadata-root", str(path)])
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    for pattern in args.required_sample_glob:
        command.extend(["--required-sample-glob", pattern])
    if args.no_pdf:
        command.append("--no-pdf")
    if args.print_command:
        print(" ".join(command))
        return 0
    return subprocess.run(command, check=False, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
