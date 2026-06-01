#!/usr/bin/env python3
"""Verify a PFM pair cache dataset before training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from patch_descriptor_training import load_libtorch_pair_archive  # noqa: E402


SPLITS = ("train", "val", "test")


def discover_pairs(dataset_root: Path) -> dict[str, list[Path]]:
    return {
        split: sorted((dataset_root / "cache" / split).glob("**/pair_*.pt"))
        for split in SPLITS
    }


def parse_ratio(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(":")]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("ratio must look like 7:2:1")
    return parts[0], parts[1], parts[2]


def expected_split_counts(total: int, ratio: tuple[int, int, int]) -> dict[str, int]:
    ratio_sum = sum(ratio)
    train = int(total * ratio[0] / ratio_sum)
    val = int(total * ratio[1] / ratio_sum)
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def validate_pair(path: Path) -> dict[str, object]:
    pair = load_libtorch_pair_archive(path, device="cpu")
    if pair.view_a.dim() != 3 or pair.view_b.dim() != 3:
        raise ValueError(f"{path} image tensors must have shape CxHxW")
    if pair.warp_a_to_b.dim() != 3 or pair.warp_a_to_b.size(-1) != 2:
        raise ValueError(f"{path} warp_a_to_b must have shape HxWx2")
    if pair.valid_mask.dim() != 2:
        raise ValueError(f"{path} valid_mask must have shape HxW")
    height, width = pair.valid_mask.shape
    if tuple(pair.warp_a_to_b.shape[:2]) != (height, width):
        raise ValueError(f"{path} warp and valid mask shapes do not match")
    if tuple(pair.view_a.shape[-2:]) != (height, width):
        raise ValueError(f"{path} view_a and valid mask shapes do not match")
    valid_pixels = int(pair.valid_mask.sum().item())
    if valid_pixels <= 0:
        raise ValueError(f"{path} has no valid correspondence pixels")
    return {
        "path": str(path),
        "view_a_shape": list(pair.view_a.shape),
        "view_b_shape": list(pair.view_b.shape),
        "valid_pixels": valid_pixels,
    }


def sample_paths(paths: list[Path], *, count: int, rng: random.Random) -> list[Path]:
    if count <= 0 or not paths:
        return []
    if len(paths) <= count:
        return list(paths)
    return rng.sample(paths, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=0)
    parser.add_argument("--expected-ratio", type=parse_ratio, default=None)
    parser.add_argument("--samples-per-split", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_split < 0:
        raise ValueError("--samples-per-split must be nonnegative")
    dataset_root = args.dataset_root.resolve(strict=False)
    split_paths = discover_pairs(dataset_root)
    counts = {split: len(paths) for split, paths in split_paths.items()}
    total = sum(counts.values())
    errors: list[str] = []
    if args.expected_total > 0 and total != args.expected_total:
        errors.append(f"expected total {args.expected_total}, got {total}")
    ratio_counts = None
    if args.expected_ratio is not None:
        ratio_counts = expected_split_counts(total, args.expected_ratio)
        if counts != ratio_counts:
            errors.append(f"expected split counts {ratio_counts}, got {counts}")
    rng = random.Random(args.seed)
    loaded: dict[str, list[dict[str, object]]] = {}
    for split, paths in split_paths.items():
        loaded[split] = []
        for path in sample_paths(paths, count=args.samples_per_split, rng=rng):
            try:
                loaded[split].append(validate_pair(path))
            except Exception as exc:  # pragma: no cover - exact message is reported to CLI caller.
                errors.append(f"failed to load {path}: {exc}")
    report = {
        "dataset_root": str(dataset_root),
        "counts": counts,
        "total": total,
        "expected_total": args.expected_total or None,
        "expected_ratio_counts": ratio_counts,
        "samples_per_split": args.samples_per_split,
        "loaded_samples": loaded,
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
