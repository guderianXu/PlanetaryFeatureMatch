#!/usr/bin/env python3
"""Compare Python pfm_model.py state keys/shapes with the C++ v2.1 mirror."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from pfm_model import PlanetaryFeatureMatcher  # noqa: E402


def shape_string(shape: tuple[int, ...]) -> str:
    return "[" + ",".join(str(value) for value in shape) + "]"


def python_state(args: argparse.Namespace) -> dict[str, str]:
    model = PlanetaryFeatureMatcher(
        input_channels=args.input_channels,
        base_channels=args.base_channels,
        descriptor_dim=args.descriptor_dim,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_attention_layers=args.graph_attention_layers,
        graph_keypoint_meta_dim=args.graph_keypoint_meta_dim,
    )
    return {key: shape_string(tuple(value.shape)) for key, value in model.state_dict().items()}


def cpp_state(args: argparse.Namespace) -> dict[str, str]:
    command = [
        str(args.cpp_tool),
        "--input-channels",
        str(args.input_channels),
        "--base-channels",
        str(args.base_channels),
        "--descriptor-dim",
        str(args.descriptor_dim),
        "--graph-hidden-dim",
        str(args.graph_hidden_dim),
        "--graph-attention-layers",
        str(args.graph_attention_layers),
        "--graph-keypoint-meta-dim",
        str(args.graph_keypoint_meta_dim),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"invalid C++ state row: {line}")
        _, key, shape = parts
        rows[key] = shape
    return rows


def report_mismatch(name: str, values: list[str], limit: int) -> None:
    print(f"{name}: {len(values)}")
    for value in values[:limit]:
        print(f"  {value}")
    if len(values) > limit:
        print(f"  ... {len(values) - limit} more")


def run(args: argparse.Namespace) -> int:
    py = python_state(args)
    cpp = cpp_state(args)
    py_keys = set(py)
    cpp_keys = set(cpp)
    missing_in_cpp = sorted(py_keys - cpp_keys)
    extra_in_cpp = sorted(cpp_keys - py_keys)
    shape_mismatches = sorted(
        f"{key}: python={py[key]} cpp={cpp[key]}"
        for key in py_keys & cpp_keys
        if py[key] != cpp[key]
    )
    if missing_in_cpp or extra_in_cpp or shape_mismatches:
        report_mismatch("missing_in_cpp", missing_in_cpp, args.limit)
        report_mismatch("extra_in_cpp", extra_in_cpp, args.limit)
        report_mismatch("shape_mismatches", shape_mismatches, args.limit)
        return 1
    print(f"matched {len(py)} state entries")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-tool", type=Path, default=REPO_ROOT / "build" / "pfm_print_v21_state_keys")
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--descriptor-dim", type=int, default=8)
    parser.add_argument("--graph-hidden-dim", type=int, default=16)
    parser.add_argument("--graph-attention-layers", type=int, default=1)
    parser.add_argument("--graph-keypoint-meta-dim", type=int, default=16)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
