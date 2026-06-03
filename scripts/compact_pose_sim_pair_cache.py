#!/usr/bin/env python3
"""Rewrite synthetic pair caches so repeated views are stored once."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from compact_pair_cache import (  # noqa: E402
    is_compact_pair_payload,
    make_compact_pair_payload,
    save_shared_image,
)


def discover_pair_paths(cache_root: Path) -> list[Path]:
    if cache_root.is_file():
        return [cache_root]
    return sorted(cache_root.glob("**/pair_*.pt"))


def is_compact_pair(path: Path) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="'torch.load' received a zip file")
            payload = torch.load(path, map_location="cpu")
    except RuntimeError:
        return False
    return is_compact_pair_payload(payload)


def load_legacy_pair(path: Path):
    module = torch.jit.load(str(path), map_location="cpu")
    return (
        module.view_a.detach().cpu().to(torch.float32).contiguous(),
        module.view_b.detach().cpu().to(torch.float32).contiguous(),
        module.warp_a_to_b.detach().cpu().to(torch.float32).contiguous(),
        module.valid_mask.detach().cpu().to(torch.bool).contiguous(),
    )


def compact_one(path: Path, *, image_store: Path, dry_run: bool) -> tuple[bool, int, int]:
    if is_compact_pair(path):
        return False, path.stat().st_size, path.stat().st_size
    old_size = path.stat().st_size
    try:
        view_a, view_b, warp, valid_mask = load_legacy_pair(path)
    except Exception as exc:
        raise RuntimeError(f"failed to load legacy pair archive {path}") from exc
    if dry_run:
        estimated_new_size = warp.numel() * warp.element_size() + valid_mask.numel() * valid_mask.element_size()
        return True, old_size, estimated_new_size
    image_a_path = save_shared_image(view_a, image_store)
    image_b_path = save_shared_image(view_b, image_store)
    payload = make_compact_pair_payload(
        pair_path=path,
        image_a_path=image_a_path,
        image_b_path=image_b_path,
        warp_a_to_b=warp,
        valid_mask=valid_mask,
    )
    tmp_path = path.with_suffix(path.suffix + f".compact_tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return True, old_size, path.stat().st_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache_root",
        type=Path,
        help="A dataset root, cache root, or a single pair_*.pt file.",
    )
    parser.add_argument(
        "--image-store",
        type=Path,
        default=None,
        help="Directory for shared view tensors. Defaults to <dataset>/image_store.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Convert at most this many legacy pair files.")
    parser.add_argument("--dry-run", action="store_true", help="Report candidates without rewriting files.")
    parser.add_argument("--skip-errors", action="store_true", help="Continue after unreadable pair archives.")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def default_image_store(cache_root: Path) -> Path:
    if cache_root.is_file():
        dataset_root = cache_root
        for parent in cache_root.parents:
            if parent.name == "cache":
                dataset_root = parent.parent
                break
        else:
            dataset_root = cache_root.parent
    else:
        dataset_root = cache_root
        if cache_root.name == "cache":
            dataset_root = cache_root.parent
    return dataset_root / "image_store"


def main() -> int:
    args = parse_args()
    cache_root = args.cache_root.resolve(strict=False)
    image_store = (args.image_store or default_image_store(cache_root)).resolve(strict=False)
    paths = discover_pair_paths(cache_root)
    if args.limit > 0:
        paths = paths[: args.limit]
    converted = 0
    skipped = 0
    errors = 0
    old_total = 0
    new_total = 0
    for index, path in enumerate(paths, start=1):
        try:
            changed, old_size, new_size = compact_one(path, image_store=image_store, dry_run=args.dry_run)
        except Exception as exc:
            if not args.skip_errors:
                raise
            errors += 1
            print(f"error path={path} message={exc}", flush=True)
            continue
        old_total += old_size
        new_total += new_size
        if changed:
            converted += 1
        else:
            skipped += 1
        if args.progress_every > 0 and index % args.progress_every == 0:
            saved = (old_total - new_total) / 1024 / 1024 / 1024
            print(
                f"processed={index} converted={converted} skipped={skipped} "
                f"errors={errors} pair_saved_gib={saved:.2f}",
                flush=True,
            )
    saved = (old_total - new_total) / 1024 / 1024 / 1024
    print(f"pairs={len(paths)} converted={converted} skipped={skipped} errors={errors}")
    print(f"pair_bytes_before={old_total} pair_bytes_after={new_total} pair_saved_gib={saved:.2f}")
    print(f"image_store={image_store}")
    if image_store.exists():
        print(f"image_store_files={sum(1 for _ in image_store.glob('*.pt'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
