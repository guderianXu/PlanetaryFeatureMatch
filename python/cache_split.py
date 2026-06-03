"""Source-level split helpers for synthetic PFM cache directories."""

from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE_DIR_RE = re.compile(r"^source_\d+_(.+)$")
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class CacheSource:
    cache_root: Path
    source_dir: Path
    source_name: str
    style: str
    gate: str
    pair_count: int


def source_name_from_dir(source_dir: Path | str) -> str:
    name = Path(source_dir).name
    match = SOURCE_DIR_RE.match(name)
    if not match:
        raise ValueError(f"source directory must look like source_000001_name: {name}")
    return match.group(1)


def source_style(source_name: str) -> str:
    return "numeric" if source_name.isdigit() else "timestamp"


def gate_name(cache_root: Path | str) -> str:
    name = Path(cache_root).name.lower()
    if name.startswith("compoundviewpoint"):
        return "compound"
    if name.startswith("viewpoint"):
        return "viewpoint"
    if name.startswith("rotate"):
        return "rotate"
    return name


def discover_cache_sources(cache_dirs: Iterable[Path | str]) -> list[CacheSource]:
    sources: list[CacheSource] = []
    for cache_dir in cache_dirs:
        root = Path(cache_dir)
        if not root.exists():
            continue
        gate = gate_name(root)
        for source_dir in sorted(path for path in root.glob("source_*") if path.is_dir()):
            pairs = sorted(source_dir.glob("pair_*.pt"))
            if not pairs:
                continue
            name = source_name_from_dir(source_dir)
            sources.append(
                CacheSource(
                    cache_root=root,
                    source_dir=source_dir,
                    source_name=name,
                    style=source_style(name),
                    gate=gate,
                    pair_count=len(pairs),
                )
            )
    return sources


def split_source_names(
    source_names: Iterable[str],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1234,
) -> dict[str, str]:
    if train_ratio <= 0.0 or val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("ratios must satisfy train_ratio > 0, val_ratio >= 0, and train + val < 1")
    names = sorted(set(source_names))
    if not names:
        return {}
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)
    if train_count == 0:
        train_count = 1
    if train_count + val_count >= len(shuffled):
        val_count = max(0, len(shuffled) - train_count - 1)
    assignments: dict[str, str] = {}
    for index, name in enumerate(shuffled):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        assignments[name] = split
    return assignments


def create_split_cache_dirs(
    sources: Iterable[CacheSource],
    assignments: dict[str, str],
    output_dir: Path | str,
) -> dict[tuple[str, str, str], Path]:
    root = Path(output_dir)
    created: dict[tuple[str, str, str], Path] = {}
    for source in sources:
        split = assignments.get(source.source_name)
        if split is None:
            continue
        if split not in VALID_SPLITS:
            raise ValueError(f"unsupported split for {source.source_name}: {split}")
        group_dir = root / split / source.style / source.gate
        group_dir.mkdir(parents=True, exist_ok=True)
        created[(split, source.style, source.gate)] = group_dir
        target = group_dir / source.source_dir.name
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                target.unlink()
            else:
                shutil.rmtree(target)
        target.symlink_to(source.source_dir.resolve(), target_is_directory=True)
    return created

