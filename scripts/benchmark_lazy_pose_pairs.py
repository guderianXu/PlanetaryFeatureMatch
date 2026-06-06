#!/usr/bin/env python3
"""Benchmark lazy pose-manifest pair generation without materializing pair cache."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for candidate in (PYTHON_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pfm_model  # noqa: E402
import pytorch_cache_match_eval as match_eval  # noqa: E402
from generate_cross_position_pose_pairs import parse_tsai, read_float_tif  # noqa: E402
from patch_descriptor_training import SyntheticPair  # noqa: E402
from pfm_data.photometric import (  # noqa: E402
    PhotometricAugmentConfig,
    apply_local_contrast_normalization,
    apply_photometric_augmentation,
    apply_training_transforms,
    make_illumination_consistency_pair,
    make_illumination_match_pair,
)
from pfm_pytorch_training import (  # noqa: E402
    FalseMatchLabels,
    descriptor_parameters,
    hard_pair_probability,
    read_false_match_labels,
    train_step,
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

DEFAULT_INIT_STATE = PROJECT_ROOT / "runs" / "python_diag_balanced_512_3epoch_20260603_2143" / "pytorch_pfm_state.pt"

_IMAGE_CACHE: "OrderedDict[str, torch.Tensor]" = OrderedDict()
_DEPTH_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_CAMERA_CACHE: "OrderedDict[str, object]" = OrderedDict()
_CACHE_MAX_ITEMS = 32


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
class LazyPairSpec:
    pair_index: int
    split: str
    reference: RenderRecord
    target: RenderRecord


@dataclass(frozen=True)
class LazyPairResult:
    spec: LazyPairSpec
    pair: SyntheticPair
    valid_fraction: float
    valid_pixels: int
    attempt_count: int
    elapsed_ms: float
    illumination_pair: SyntheticPair | None = None
    illumination_match_pair: SyntheticPair | None = None


def _worker_init(cache_max_items: int) -> None:
    global _CACHE_MAX_ITEMS
    _CACHE_MAX_ITEMS = max(0, int(cache_max_items))
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _cache_get(cache: "OrderedDict[str, object]", key: Path, loader):
    cache_key = str(key)
    if _CACHE_MAX_ITEMS > 0 and cache_key in cache:
        value = cache.pop(cache_key)
        cache[cache_key] = value
        return value
    value = loader(key)
    if _CACHE_MAX_ITEMS > 0:
        cache[cache_key] = value
        while len(cache) > _CACHE_MAX_ITEMS:
            cache.popitem(last=False)
    return value


def _path_list(value: Path | list[Path] | tuple[Path, ...] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, Path):
        return [value]
    return list(value)


def _first_path(value: Path | list[Path] | tuple[Path, ...] | None) -> Path | None:
    paths = _path_list(value)
    return paths[0] if paths else None


def _read_uint8_manifest(path: Path | None) -> dict[str, Path]:
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


def _read_uint8_manifests(paths: list[Path]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in paths:
        mapping.update(_read_uint8_manifest(path))
    return mapping


def _read_render_manifest(path: Path, uint8_paths: dict[str, Path], *, dataset_prefix: str | None = None) -> list[RenderRecord]:
    records: list[RenderRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pose_id = row.get("pose_id", "")
            if not pose_id or pose_id == "pose_id":
                continue
            image_path = Path(row["image_path"])
            uint8_path = uint8_paths.get(str(image_path))
            if uint8_path is None and image_path.is_file():
                uint8_path = image_path
            base_id = row["base_id"]
            if dataset_prefix:
                base_id = f"{dataset_prefix}:{base_id}"
            records.append(
                RenderRecord(
                    pose_id=pose_id,
                    base_id=base_id,
                    variant=row["variant"],
                    split=row["split"],
                    tsai_path=Path(row["tsai_path"]),
                    image_path=image_path,
                    uint8_path=uint8_path,
                    depth_path=Path(row["depth_path"]),
                )
            )
    return records


def _dataset_prefix_for_manifest(path: Path, index: int) -> str:
    parent_name = path.parent.name
    return parent_name if parent_name else f"dataset_{index:03d}"


def _read_all_render_records(render_manifests: list[Path], uint8_manifests: list[Path]) -> list[RenderRecord]:
    uint8_paths = _read_uint8_manifests(uint8_manifests)
    use_prefix = len(render_manifests) > 1
    records: list[RenderRecord] = []
    for index, render_manifest in enumerate(render_manifests):
        prefix = _dataset_prefix_for_manifest(render_manifest, index) if use_prefix else None
        records.extend(_read_render_manifest(render_manifest, uint8_paths, dataset_prefix=prefix))
    return records


def _selected_image_path(record: RenderRecord, image_source: str) -> Path:
    if image_source == "uint8":
        return record.uint8_path if record.uint8_path is not None else Path()
    return record.image_path


def _load_view(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(torch.float32)
    dtype_name = image.dtype.name
    if dtype_name == "uint8":
        tensor = tensor / 255.0
    elif dtype_name == "uint16":
        tensor = tensor / 65535.0
    elif bool(torch.isfinite(tensor).any()) and float(tensor.max().item()) > 1.5:
        tensor = tensor / float(tensor.max().clamp_min(1.0).item())
    return tensor.nan_to_num(0.0, 0.0, 0.0).clamp(0.0, 1.0).unsqueeze(0).contiguous()


def _cached_view(path: Path) -> torch.Tensor:
    return _cache_get(_IMAGE_CACHE, path, _load_view)


def _cached_depth(path: Path) -> np.ndarray:
    return _cache_get(_DEPTH_CACHE, path, read_float_tif)


def _cached_camera(path: Path):
    return _cache_get(_CAMERA_CACHE, path, parse_tsai)


def build_pair_specs(
    records: list[RenderRecord],
    *,
    split: str,
    reference_variant: str,
    target_variants: tuple[str, ...],
    image_source: str,
    limit_pairs: int,
    seed: int,
    shuffle: bool,
) -> list[LazyPairSpec]:
    by_base: dict[str, dict[str, RenderRecord]] = defaultdict(dict)
    for record in records:
        if split != "all" and record.split != split:
            continue
        if image_source == "uint8" and record.uint8_path is None:
            continue
        image_path = _selected_image_path(record, image_source)
        if image_path.is_file() and record.depth_path.is_file() and record.tsai_path.is_file():
            by_base[record.base_id][record.variant] = record

    specs: list[LazyPairSpec] = []
    for base_id in sorted(by_base):
        variants = by_base[base_id]
        reference = variants.get(reference_variant)
        if reference is None:
            continue
        for variant in target_variants:
            target = variants.get(variant)
            if target is None:
                continue
            specs.append(
                LazyPairSpec(
                    pair_index=len(specs),
                    split=reference.split,
                    reference=reference,
                    target=target,
                )
            )
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(specs)
    if limit_pairs > 0:
        return specs[:limit_pairs]
    return specs


def _clamp_origin(center: float, *, crop_size: int, full_size: int) -> int:
    if crop_size >= full_size:
        return 0
    return max(0, min(int(round(center - float(crop_size - 1) * 0.5)), full_size - crop_size))


def _random_crop_origin(rng: random.Random, *, crop_size: int, full_size: int) -> int:
    if crop_size >= full_size:
        return 0
    return rng.randint(0, full_size - crop_size)


def _project_crop_pair(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    camera_a,
    camera_b,
    *,
    crop_size: int,
    absolute_depth_tolerance_m: float,
    relative_depth_tolerance: float,
    rng: random.Random,
) -> tuple[SyntheticPair, float, int]:
    if depth_a.shape != depth_b.shape:
        raise ValueError(f"depth shape mismatch: {depth_a.shape} vs {depth_b.shape}")
    _, height_a, width_a = view_a.shape
    _, height_b, width_b = view_b.shape
    if depth_a.shape != (height_a, width_a) or depth_b.shape != (height_b, width_b):
        raise ValueError(
            f"image/depth shape mismatch: view_a={tuple(view_a.shape)} depth_a={depth_a.shape} "
            f"view_b={tuple(view_b.shape)} depth_b={depth_b.shape}"
        )

    crop_h_a = min(int(crop_size), height_a)
    crop_w_a = min(int(crop_size), width_a)
    crop_h_b = min(int(crop_size), height_b)
    crop_w_b = min(int(crop_size), width_b)
    ax0 = _random_crop_origin(rng, crop_size=crop_w_a, full_size=width_a)
    ay0 = _random_crop_origin(rng, crop_size=crop_h_a, full_size=height_a)
    ax1 = ax0 + crop_w_a
    ay1 = ay0 + crop_h_a

    yy, xx = np.indices((crop_h_a, crop_w_a), dtype=np.float64)
    gx = xx + float(ax0)
    gy = yy + float(ay0)
    z = depth_a[ay0:ay1, ax0:ax1].astype(np.float64, copy=False)
    valid_a = np.isfinite(z) & (z > 0.0)

    x_cam = (gx + 0.5 - camera_a.cu) / camera_a.fu * z
    y_cam = (gy + 0.5 - camera_a.cv) / camera_a.fv * z
    pts_cam = np.stack((x_cam, y_cam, z), axis=0).reshape(3, -1)
    world = camera_a.center[:, None] + camera_a.rotation_world_to_camera.T @ pts_cam
    projected_b = camera_b.rotation_world_to_camera @ (world - camera_b.center[:, None])
    pb_x = projected_b[0].reshape(crop_h_a, crop_w_a)
    pb_y = projected_b[1].reshape(crop_h_a, crop_w_a)
    pb_z = projected_b[2].reshape(crop_h_a, crop_w_a)

    with np.errstate(divide="ignore", invalid="ignore"):
        u_b = camera_b.fu * (pb_x / pb_z) + camera_b.cu - 0.5
        v_b = camera_b.fv * (pb_y / pb_z) + camera_b.cv - 0.5
    inside_b = (pb_z > 0.0) & (u_b >= 0.0) & (u_b <= width_b - 1.0) & (v_b >= 0.0) & (v_b <= height_b - 1.0)

    sampled_depth_b = cv2.remap(
        depth_b.astype(np.float32, copy=False),
        u_b.astype(np.float32, copy=False),
        v_b.astype(np.float32, copy=False),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1.0,
    ).astype(np.float64, copy=False)
    tolerance = np.maximum(float(absolute_depth_tolerance_m), float(relative_depth_tolerance) * np.abs(pb_z))
    valid_global = (
        valid_a
        & inside_b
        & np.isfinite(sampled_depth_b)
        & (sampled_depth_b > 0.0)
        & (np.abs(sampled_depth_b - pb_z) <= tolerance)
    )

    if bool(valid_global.any()):
        bx0 = _clamp_origin(float(np.median(u_b[valid_global])), crop_size=crop_w_b, full_size=width_b)
        by0 = _clamp_origin(float(np.median(v_b[valid_global])), crop_size=crop_h_b, full_size=height_b)
    else:
        bx0 = max(0, min(ax0, width_b - crop_w_b))
        by0 = max(0, min(ay0, height_b - crop_h_b))
    bx1 = bx0 + crop_w_b
    by1 = by0 + crop_h_b

    warp_np = np.stack((u_b - float(bx0), v_b - float(by0)), axis=-1).astype(np.float32, copy=False)
    finite_warp = np.isfinite(warp_np).all(axis=-1)
    valid_mask_np = (
        valid_global
        & finite_warp
        & (warp_np[..., 0] >= 0.0)
        & (warp_np[..., 0] <= float(crop_w_b - 1))
        & (warp_np[..., 1] >= 0.0)
        & (warp_np[..., 1] <= float(crop_h_b - 1))
    )
    warp_np[~finite_warp] = 0.0

    pair = SyntheticPair(
        view_a=view_a[:, ay0:ay1, ax0:ax1].contiguous(),
        view_b=view_b[:, by0:by1, bx0:bx1].contiguous(),
        warp_a_to_b=torch.from_numpy(warp_np.copy()).to(torch.float32).contiguous(),
        valid_mask=torch.from_numpy(valid_mask_np.copy()).to(torch.bool).contiguous(),
    )
    valid_pixels = int(valid_mask_np.sum())
    valid_fraction = float(valid_pixels) / float(max(1, crop_h_a * crop_w_a))
    return pair, valid_fraction, valid_pixels


def generate_lazy_pair(
    spec: LazyPairSpec,
    *,
    crop_size: int,
    image_source: str,
    max_attempts: int,
    min_valid_fraction: float,
    absolute_depth_tolerance_m: float,
    relative_depth_tolerance: float,
    seed: int,
    photometric_config: PhotometricAugmentConfig | None = None,
    transform_seed: int = 0,
    input_local_contrast: bool = False,
    local_contrast_strength: float = 0.0,
    local_contrast_kernel: int = 31,
    illumination_consistency_config: PhotometricAugmentConfig | None = None,
    illumination_consistency_probability: float = 1.0,
    illumination_match_config: PhotometricAugmentConfig | None = None,
    illumination_match_probability: float = 1.0,
    illumination_match_changed_view: str = "b",
) -> LazyPairResult:
    start = time.perf_counter()
    view_a = _cached_view(_selected_image_path(spec.reference, image_source))
    view_b = _cached_view(_selected_image_path(spec.target, image_source))
    depth_a = _cached_depth(spec.reference.depth_path)
    depth_b = _cached_depth(spec.target.depth_path)
    camera_a = _cached_camera(spec.reference.tsai_path)
    camera_b = _cached_camera(spec.target.tsai_path)

    best: tuple[SyntheticPair, float, int] | None = None
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        rng = random.Random(seed + spec.pair_index * 1009 + attempt * 9176)
        pair, valid_fraction, valid_pixels = _project_crop_pair(
            view_a,
            view_b,
            depth_a,
            depth_b,
            camera_a,
            camera_b,
            crop_size=crop_size,
            absolute_depth_tolerance_m=absolute_depth_tolerance_m,
            relative_depth_tolerance=relative_depth_tolerance,
            rng=rng,
        )
        if best is None or valid_fraction > best[1]:
            best = (pair, valid_fraction, valid_pixels)
        if valid_fraction >= min_valid_fraction:
            break
    if best is None:
        raise RuntimeError("failed to generate lazy pair")
    pair = best[0]
    transform_config = photometric_config if photometric_config is not None else PhotometricAugmentConfig()
    if transform_config.enabled or input_local_contrast:
        pair = apply_training_transforms(
            pair,
            photometric_config=transform_config,
            seed=transform_seed,
            input_local_contrast=input_local_contrast,
            local_contrast_strength=local_contrast_strength,
            local_contrast_kernel=local_contrast_kernel,
        )
    illumination_pair = None
    consistency_config = illumination_consistency_config or PhotometricAugmentConfig()
    consistency_probability = max(0.0, min(1.0, float(illumination_consistency_probability)))
    if consistency_config.enabled and random.Random(transform_seed + 0x5DEECE66D).random() <= consistency_probability:
        illumination_pair = make_illumination_consistency_pair(
            pair,
            consistency_config,
            seed=transform_seed + 0xD1B54A32D192ED03,
        )
    illumination_match_pair = None
    match_config = illumination_match_config or PhotometricAugmentConfig()
    match_probability = max(0.0, min(1.0, float(illumination_match_probability)))
    if match_config.enabled and random.Random(transform_seed + 0xC0FFEE).random() <= match_probability:
        illumination_match_pair = make_illumination_match_pair(
            pair,
            match_config,
            seed=transform_seed + 0xA24BAED4963EE407,
            changed_view=illumination_match_changed_view,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return LazyPairResult(
        spec=spec,
        pair=pair,
        valid_fraction=best[1],
        valid_pixels=best[2],
        attempt_count=attempt + 1,
        elapsed_ms=elapsed_ms,
        illumination_pair=illumination_pair,
        illumination_match_pair=illumination_match_pair,
    )


def iter_lazy_pairs(
    specs: list[LazyPairSpec],
    *,
    count: int,
    workers: int,
    prefetch: int,
    cache_max_items: int,
    crop_size: int,
    image_source: str,
    max_attempts: int,
    min_valid_fraction: float,
    absolute_depth_tolerance_m: float,
    relative_depth_tolerance: float,
    seed: int,
    skip_bad_pairs: bool,
    max_bad_pairs: int,
    photometric_config: PhotometricAugmentConfig | None = None,
    input_local_contrast: bool = False,
    local_contrast_strength: float = 0.0,
    local_contrast_kernel: int = 31,
    illumination_consistency_config: PhotometricAugmentConfig | None = None,
    illumination_consistency_probability: float = 1.0,
    illumination_match_config: PhotometricAugmentConfig | None = None,
    illumination_match_probability: float = 1.0,
    illumination_match_changed_view: str = "b",
) -> Iterator[LazyPairResult]:
    if not specs:
        raise RuntimeError("no lazy pair specs available")
    total = count if count > 0 else len(specs)
    cursor = 0
    skipped = 0
    max_skips = max_bad_pairs if max_bad_pairs > 0 else max(total, len(specs))

    def submit(executor: ProcessPoolExecutor, future_map: dict[Future[LazyPairResult], float]) -> None:
        nonlocal cursor
        spec = specs[cursor % len(specs)]
        future = executor.submit(
            generate_lazy_pair,
            spec,
            crop_size=crop_size,
            image_source=image_source,
            max_attempts=max_attempts,
            min_valid_fraction=min_valid_fraction,
            absolute_depth_tolerance_m=absolute_depth_tolerance_m,
            relative_depth_tolerance=relative_depth_tolerance,
            seed=seed + cursor * 31,
            photometric_config=photometric_config,
            transform_seed=seed + cursor * 1000003,
            input_local_contrast=input_local_contrast,
            local_contrast_strength=local_contrast_strength,
            local_contrast_kernel=local_contrast_kernel,
            illumination_consistency_config=illumination_consistency_config,
            illumination_consistency_probability=illumination_consistency_probability,
            illumination_match_config=illumination_match_config,
            illumination_match_probability=illumination_match_probability,
            illumination_match_changed_view=illumination_match_changed_view,
        )
        future_map[future] = time.perf_counter()
        cursor += 1

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(cache_max_items,)) as executor:
        pending: dict[Future[LazyPairResult], float] = {}
        target_prefetch = max(1, int(prefetch))
        while cursor < min(total + max_skips, target_prefetch):
            submit(executor, pending)
        yielded = 0
        while yielded < total:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    if not skip_bad_pairs:
                        raise
                    skipped += 1
                    print(f"skip_bad_pair count={skipped} error={exc}", flush=True)
                    if skipped > max_skips:
                        raise RuntimeError(f"too many lazy pair failures: {skipped}") from exc
                    while cursor < total + max_skips and len(pending) < target_prefetch:
                        submit(executor, pending)
                    continue
                yielded += 1
                while cursor < total + max_skips and len(pending) < target_prefetch:
                    submit(executor, pending)
                yield result
                if yielded >= total:
                    break


def _write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class StreamingCsvRows:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self._handle = None
        self._writer: csv.DictWriter | None = None

    def open(self) -> "StreamingCsvRows":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._handle.flush()
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None

    def __enter__(self) -> "StreamingCsvRows":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, row: dict[str, object]) -> None:
        if self._writer is None or self._handle is None:
            raise RuntimeError("streaming csv writer is not open")
        self._writer.writerow(row)
        self._handle.flush()


def _summarize_float(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, int(math.ceil(len(sorted_values) * 0.95)) - 1)
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": float(sorted_values[p95_index]),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
    }


def _gpu_snapshot() -> dict[str, str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    first = output.splitlines()[0].split(",")
    if len(first) < 3:
        return {}
    return {
        "gpu_util_percent": first[0].strip(),
        "gpu_mem_used_mib": first[1].strip(),
        "gpu_mem_total_mib": first[2].strip(),
    }


class GpuUsageMonitor:
    """后台采样 GPU 状态，避免训练 step 内同步调用 nvidia-smi。"""

    _FIELDS = ["elapsed_s", "gpu_util_percent", "gpu_mem_used_mib", "gpu_mem_total_mib"]

    def __init__(
        self,
        path: Path,
        *,
        sample_interval_s: float,
        snapshot_fn: Callable[[], dict[str, str]] = _gpu_snapshot,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if sample_interval_s <= 0.0:
            raise ValueError("sample_interval_s must be positive")
        self._path = path
        self._sample_interval_s = float(sample_interval_s)
        self._snapshot_fn = snapshot_fn
        self._clock = clock
        self._start_time = self._clock()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, str] = {}
        self._handle = None
        self._writer: csv.DictWriter | None = None

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self._FIELDS)
        self._writer.writeheader()
        self._handle.flush()

    def sample_once(self) -> None:
        snapshot = self._snapshot_fn()
        if not snapshot:
            return
        elapsed_s = max(0.0, self._clock() - self._start_time)
        row = {
            "elapsed_s": f"{elapsed_s:.3f}",
            "gpu_util_percent": snapshot.get("gpu_util_percent", ""),
            "gpu_mem_used_mib": snapshot.get("gpu_mem_used_mib", ""),
            "gpu_mem_total_mib": snapshot.get("gpu_mem_total_mib", ""),
        }
        with self._lock:
            self._latest = {
                "gpu_util_percent": row["gpu_util_percent"],
                "gpu_mem_used_mib": row["gpu_mem_used_mib"],
                "gpu_mem_total_mib": row["gpu_mem_total_mib"],
            }
        self._ensure_writer()
        assert self._writer is not None and self._handle is not None
        self._writer.writerow(row)
        self._handle.flush()

    def latest(self) -> dict[str, str]:
        with self._lock:
            return dict(self._latest)

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sample_once()
        self._thread = threading.Thread(target=self._run, name="gpu-usage-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self._sample_interval_s):
            try:
                self.sample_once()
            except Exception as exc:  # pragma: no cover - 后台监控失败不能中断训练。
                print(f"gpu_monitor_warning={exc}", flush=True)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._sample_interval_s * 2.0))
            self._thread = None
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None


def _should_collect_gpu_snapshot(step: int, interval: int) -> bool:
    if interval <= 0:
        raise ValueError("GPU snapshot interval must be positive")
    return step == 1 or step % interval == 0


def _photometric_config_from_args(args: argparse.Namespace) -> PhotometricAugmentConfig:
    return PhotometricAugmentConfig(
        enabled=bool(args.photometric_augment),
        probability=float(args.photometric_probability),
        brightness=float(args.photometric_brightness),
        contrast=float(args.photometric_contrast),
        gamma=float(args.photometric_gamma),
        shadow=float(args.photometric_shadow),
        noise=float(args.photometric_noise),
    )


def _illumination_consistency_config_from_args(args: argparse.Namespace) -> PhotometricAugmentConfig:
    return PhotometricAugmentConfig(
        enabled=float(args.illumination_consistency_weight) > 0.0,
        probability=1.0,
        brightness=float(args.illumination_consistency_brightness),
        contrast=float(args.illumination_consistency_contrast),
        gamma=float(args.illumination_consistency_gamma),
        shadow=float(args.illumination_consistency_shadow),
        noise=float(args.illumination_consistency_noise),
    )


def _illumination_match_config_from_args(args: argparse.Namespace) -> PhotometricAugmentConfig:
    return PhotometricAugmentConfig(
        enabled=float(args.illumination_match_weight) > 0.0,
        probability=1.0,
        brightness=float(args.illumination_match_brightness),
        contrast=float(args.illumination_match_contrast),
        gamma=float(args.illumination_match_gamma),
        shadow=float(args.illumination_match_shadow),
        noise=float(args.illumination_match_noise),
    )


def _write_html_report(
    path: Path,
    *,
    title: str,
    args: argparse.Namespace,
    spec_count: int,
    summary: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    sample_rows = rows[-20:]
    table = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row.values())
        + "</tr>"
        for row in sample_rows
    )
    headers = "".join(f"<th>{html.escape(str(key))}</th>" for key in sample_rows[0].keys()) if sample_rows else ""
    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #0b1117; color: #d9e6f2; }}
section {{ border: 1px solid #233646; border-radius: 8px; padding: 16px; margin: 16px 0; background: #111b24; }}
pre {{ white-space: pre-wrap; background: #071018; padding: 12px; border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #243746; padding: 6px 8px; text-align: left; }}
th {{ color: #7edce2; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<section>
<h2>配置</h2>
<pre>{html.escape(json.dumps(vars(args), indent=2, ensure_ascii=False, default=str))}</pre>
<p>可用 pair specs: {spec_count}</p>
</section>
<section>
<h2>结果摘要</h2>
<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False, default=str))}</pre>
</section>
<section>
<h2>最近记录</h2>
<table><thead><tr>{headers}</tr></thead><tbody>{table}</tbody></table>
</section>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def run_preprocess(args: argparse.Namespace, specs: list[LazyPairSpec]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for index, result in enumerate(
        iter_lazy_pairs(
            specs,
            count=args.pairs,
            workers=args.workers,
            prefetch=args.prefetch_batches,
            cache_max_items=args.worker_cache_items,
            crop_size=args.crop_size,
            image_source=args.image_source,
            max_attempts=args.max_attempts,
            min_valid_fraction=args.min_valid_fraction,
            absolute_depth_tolerance_m=args.absolute_depth_tolerance_m,
            relative_depth_tolerance=args.relative_depth_tolerance,
            seed=args.seed,
            skip_bad_pairs=args.skip_bad_pairs,
            max_bad_pairs=args.max_bad_pairs,
        ),
        1,
    ):
        rows.append(
            {
                "index": index,
                "pair_index": result.spec.pair_index,
                "split": result.spec.split,
                "base_id": result.spec.reference.base_id,
                "variant": result.spec.target.variant,
                "valid_fraction": f"{result.valid_fraction:.6f}",
                "valid_pixels": result.valid_pixels,
                "attempts": result.attempt_count,
                "worker_elapsed_ms": f"{result.elapsed_ms:.2f}",
            }
        )
        if index == 1 or index % max(1, args.progress_every) == 0:
            elapsed = time.perf_counter() - start
            print(
                f"preprocess {index}/{args.pairs} pairs "
                f"rate={index / max(elapsed, 1.0e-6):.2f} pairs/s "
                f"valid={result.valid_fraction:.4f}",
                flush=True,
            )
    elapsed = time.perf_counter() - start
    summary = {
        "mode": "preprocess",
        "pairs": len(rows),
        "elapsed_s": elapsed,
        "pairs_per_second": len(rows) / max(elapsed, 1.0e-6),
        "worker_elapsed_ms": _summarize_float([float(row["worker_elapsed_ms"]) for row in rows]),
        "valid_fraction": _summarize_float([float(row["valid_fraction"]) for row in rows]),
    }
    _write_rows(
        args.output_dir / "preprocess_metrics.csv",
        rows,
        ["index", "pair_index", "split", "base_id", "variant", "valid_fraction", "valid_pixels", "attempts", "worker_elapsed_ms"],
    )
    _write_html_report(
        args.output_dir / "run.html",
        title="Lazy Pose Pair Preprocess Benchmark",
        args=args,
        spec_count=len(specs),
        summary=summary,
        rows=rows,
    )
    return summary


def _load_model(args: argparse.Namespace, device: torch.device):
    checkpoint = Path(args.init_pytorch_state)
    if not checkpoint.exists():
        raise FileNotFoundError(f"--init-pytorch-state does not exist: {checkpoint}")
    model, _ = pfm_model.load_pytorch_state(checkpoint, device=device, strict=False)
    trainable = descriptor_parameters(
        model,
        train_backbone=args.train_backbone,
        train_dual_fpn=args.train_dual_fpn,
        train_descriptor_head=args.train_descriptor_head,
        train_sparse_context=args.train_sparse_context,
        train_keypoint_head=args.train_keypoint_head,
        train_geometry_head=args.train_geometry_head,
        train_texture_adapter=args.train_texture_adapter,
        train_descriptor_fusion=args.train_descriptor_fusion,
        train_quality_head=args.train_quality_head,
        train_graph_matcher=args.train_graph_matcher,
    )
    if not trainable:
        raise RuntimeError("no trainable parameters selected")
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, optimizer


def _write_visual_report_error(path: Path, *, command: list[str], error: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>训练后匹配可视化失败</title>
<style>
body {{ margin: 24px; background: #081017; color: #dcebf7; font-family: Arial, "Noto Sans CJK SC", sans-serif; }}
pre {{ white-space: pre-wrap; background: #101b24; border: 1px solid #26394a; border-radius: 8px; padding: 14px; }}
.error {{ color: #fb7185; }}
</style>
</head>
<body>
<h1>训练后匹配可视化失败</h1>
<p class="error">训练 checkpoint 已写入，但自动生成匹配连线图失败。</p>
<h2>命令</h2>
<pre>{html.escape(" ".join(command))}</pre>
<h2>错误</h2>
<pre>{html.escape(error)}</pre>
</body>
</html>
"""
    (path / "index.html").write_text(body, encoding="utf-8")


def _run_visual_report(args: argparse.Namespace, checkpoint_path: Path) -> Path | None:
    if not args.auto_visual_report:
        return None
    render_manifest = _first_path(args.render_manifest)
    uint8_manifest = _first_path(args.uint8_manifest)
    if render_manifest is None or uint8_manifest is None:
        return None
    report_dir = args.output_dir / "visual_report"
    script_path = PROJECT_ROOT / "scripts" / "visualize_lazy_pose_matches.py"
    command = [
        sys.executable,
        str(script_path),
        "--render-manifest",
        str(render_manifest),
        "--uint8-manifest",
        str(uint8_manifest),
        "--pytorch-state",
        str(checkpoint_path),
        "--output-dir",
        str(report_dir),
        "--run-dir",
        str(args.output_dir),
        "--metrics-csv",
        str(args.output_dir / "train_metrics.csv"),
        "--split",
        args.visual_split or args.split,
        "--reference-variant",
        args.reference_variant,
        "--candidate-pairs",
        str(args.visual_candidate_pairs),
        "--select-count",
        str(args.visual_select_count),
        "--seed",
        str(args.seed + 2026),
        "--crop-size",
        str(args.crop_size),
        "--max-image-size",
        str(args.visual_max_image_size),
        "--max-attempts",
        str(args.max_attempts),
        "--min-valid-fraction",
        str(args.min_valid_fraction),
        "--absolute-depth-tolerance-m",
        str(args.absolute_depth_tolerance_m),
        "--relative-depth-tolerance",
        str(args.relative_depth_tolerance),
        "--device",
        args.visual_device or args.device,
        "--descriptor-mode",
        args.visual_descriptor_mode,
        "--keypoint-score-mode",
        args.visual_keypoint_score_mode,
        "--matcher-mode",
        args.visual_matcher_mode,
        "--max-keypoints",
        str(args.visual_max_keypoints),
        "--max-matches",
        str(args.visual_max_matches),
        "--draw-matches",
        str(args.visual_draw_matches),
        "--threshold-px",
        str(args.visual_threshold_px),
        "--graph-width-prune-min-score",
        str(args.visual_graph_width_prune_min_score),
        "--graph-early-stop-min-confidence",
        str(args.visual_graph_early_stop_min_confidence),
    ]
    command.append("--filtered-report" if args.visual_filtered_report else "--no-filtered-report")
    command.extend(
        [
            "--filtered-geometry-filter",
            args.visual_filtered_geometry_filter,
            "--filtered-min-margin",
            str(args.visual_filtered_min_margin),
            "--filtered-min-score",
            str(args.visual_filtered_min_score),
            "--filtered-max-matches",
            str(args.visual_filtered_max_matches),
            "--filtered-draw-matches",
            str(args.visual_filtered_draw_matches),
        ]
    )
    if args.input_local_contrast:
        command.extend(
            [
                "--input-local-contrast",
                "--input-local-contrast-strength",
                str(args.input_local_contrast_strength),
                "--input-local-contrast-kernel",
                str(args.input_local_contrast_kernel),
            ]
        )
    if args.visual_filtered_mutual:
        command.append("--filtered-mutual")
    else:
        command.append("--no-filtered-mutual")
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except Exception as exc:
        _write_visual_report_error(report_dir, command=command, error=str(exc))
        return report_dir
    return report_dir


def select_hard_lazy_specs(specs: list[LazyPairSpec], hard_variants: list[str]) -> list[LazyPairSpec]:
    tokens = [item.strip().lower() for item in hard_variants if item.strip()]
    if not tokens:
        return []
    selected: list[LazyPairSpec] = []
    for spec in specs:
        variant = spec.target.variant.lower()
        if any(token in variant or token in spec.reference.base_id.lower() for token in tokens):
            selected.append(spec)
    return selected


def _valid_source_mask(valid_mask: torch.Tensor, points_a: torch.Tensor) -> torch.Tensor:
    if points_a.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=points_a.device)
    height, width = valid_mask.shape
    in_bounds = (
        (points_a[:, 0] >= 0.0)
        & (points_a[:, 0] <= float(width - 1))
        & (points_a[:, 1] >= 0.0)
        & (points_a[:, 1] <= float(height - 1))
    )
    rounded = points_a.round().to(torch.long)
    x = rounded[:, 0].clamp(0, width - 1)
    y = rounded[:, 1].clamp(0, height - 1)
    return in_bounds.to(valid_mask.device) & valid_mask.to(points_a.device)[y, x]


@torch.no_grad()
def mine_false_matches_for_lazy_pair(
    model: pfm_model.PlanetaryFeatureMatcher,
    pair: SyntheticPair,
    pair_path: Path,
    *,
    device: torch.device,
    descriptor_mode: str,
    texture_blend_weight: float,
    keypoint_score_mode: str,
    max_keypoints: int,
    max_matches: int,
    min_intensity: float,
    min_score: float,
    min_margin: float,
    threshold_px: float,
) -> tuple[dict[str, FalseMatchLabels], list[dict[str, object]]]:
    if max_matches < 0:
        raise ValueError("false match mining max_matches must be nonnegative")
    was_training = model.training
    model.eval()
    try:
        pair_device = SyntheticPair(
            view_a=pair.view_a.to(device=device, non_blocking=True),
            view_b=pair.view_b.to(device=device, non_blocking=True),
            warp_a_to_b=pair.warp_a_to_b.to(device=device, non_blocking=True),
            valid_mask=pair.valid_mask.to(device=device, non_blocking=True),
        )
        descriptors_a, descriptors_b, score_a, score_b, _, _ = match_eval.feature_maps_and_keypoint_scores_for_pair(
            model,
            pair_device,
            mode=descriptor_mode,
            texture_blend_weight=texture_blend_weight,
            keypoint_score_mode=keypoint_score_mode,
        )
        keypoints_a, selected_a = match_eval.select_descriptor_keypoints(
            pair_device.view_a,
            descriptors_a,
            max_keypoints=max_keypoints,
            min_intensity=min_intensity,
            texture_fraction=1.0,
            weak_texture_fraction=0.0,
            keypoint_scores=score_a,
        )
        keypoints_b, selected_b = match_eval.select_descriptor_keypoints(
            pair_device.view_b,
            descriptors_b,
            max_keypoints=max_keypoints,
            min_intensity=min_intensity,
            texture_fraction=1.0,
            weak_texture_fraction=0.0,
            keypoint_scores=score_b,
        )
        rows_a = match_eval.gather_descriptor_rows(descriptors_a, selected_a)
        rows_b = match_eval.gather_descriptor_rows(descriptors_b, selected_b)
        matches, scores = match_eval.mutual_nearest_matches(
            rows_a,
            rows_b,
            max_matches=max_matches,
            min_score=min_score,
            min_margin=min_margin,
        )
        if matches.numel() == 0:
            return {}, []
        _, image_height_a, image_width_a = pair_device.view_a.shape
        _, image_height_b, image_width_b = pair_device.view_b.shape
        points_a = match_eval._feature_to_image_points(
            keypoints_a.index_select(0, matches[:, 0].to(keypoints_a.device)),
            feature_height=descriptors_a.size(2),
            feature_width=descriptors_a.size(3),
            image_height=image_height_a,
            image_width=image_width_a,
        )
        points_b = match_eval._feature_to_image_points(
            keypoints_b.index_select(0, matches[:, 1].to(keypoints_b.device)),
            feature_height=descriptors_b.size(2),
            feature_width=descriptors_b.size(3),
            image_height=image_height_b,
            image_width=image_width_b,
        )
        target_b = match_eval.sample_warp(pair_device.warp_a_to_b, points_a)
        errors = (target_b.to(points_b.device) - points_b).norm(dim=1)
        valid = _valid_source_mask(pair_device.valid_mask, points_a)
        wrong = (~valid.to(errors.device)) | errors.gt(float(threshold_px))
        indices = torch.nonzero(wrong, as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            return {}, []
        false_a = points_a.index_select(0, indices).detach().cpu()
        false_b = points_b.index_select(0, indices).detach().cpu()
        path_key = pair_path.as_posix()
        labels = {
            path_key: FalseMatchLabels(
                points_a_xy=false_a.to(torch.float32),
                points_b_xy=false_b.to(torch.float32),
            )
        }
        rows = []
        for local_index in indices.detach().cpu().tolist():
            rows.append(
                {
                    "pair_pt": path_key,
                    "ax": f"{float(points_a[local_index, 0].detach().cpu()):.3f}",
                    "ay": f"{float(points_a[local_index, 1].detach().cpu()):.3f}",
                    "bx": f"{float(points_b[local_index, 0].detach().cpu()):.3f}",
                    "by": f"{float(points_b[local_index, 1].detach().cpu()):.3f}",
                    "error_px": f"{float(errors[local_index].detach().cpu()):.3f}",
                    "score": f"{float(scores[local_index].detach().cpu()):.6f}",
                }
            )
        return labels, rows
    finally:
        model.train(was_training)


def run_train(args: argparse.Namespace, specs: list[LazyPairSpec]) -> dict[str, object]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model, optimizer = _load_model(args, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    ref_dir = args.output_dir / "lazy_pair_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    photometric_config = _photometric_config_from_args(args)
    illumination_consistency_config = _illumination_consistency_config_from_args(args)
    illumination_match_config = _illumination_match_config_from_args(args)
    train_metric_fields = [
        "step",
        "loss",
        "top1_accuracy",
        "mean_positive_rank",
        "points",
        "false_match_points",
        "false_match_pairs",
        "online_false_match_points",
        "online_false_match_pairs",
        "illumination_consistency_points",
        "illumination_consistency_pairs",
        "illumination_match_points",
        "illumination_match_pairs",
        "hard_lazy_pairs",
        "data_wait_ms",
        "augment_ms",
        "false_mine_ms",
        "train_ms",
        "gpu_snapshot_ms",
        "step_ms",
        "valid_fraction_mean",
        "worker_elapsed_ms_mean",
        "photometric_augment",
        "photometric_probability",
        "input_local_contrast",
        "graph_matcher_loss_weight",
        "graph_matcher_assignment_weight",
        "graph_matcher_accept_weight",
        "graph_matcher_prune_ranking_weight",
        "graph_matcher_stop_confidence_weight",
        "graph_matcher_train_max_attention_layers",
        "graph_matcher_train_random_attention_layers",
        "graph_matcher_train_max_attention_work_fraction",
        "graph_matcher_train_width_keep_ratio",
        "graph_matcher_online_false_no_match",
        "graph_matcher_total_loss",
        "graph_matcher_ce_loss",
        "graph_matcher_assignment_loss",
        "graph_matcher_no_match_loss",
        "graph_matcher_accept_loss",
        "graph_matcher_prune_ranking_loss",
        "graph_matcher_stop_confidence_loss",
        "graph_matcher_raw_preservation_loss",
        "graph_matcher_hard_negative_dustbin_loss",
        "graph_matcher_executed_attention_layers",
        "graph_matcher_attention_work_fraction",
        "graph_matcher_positive_pairs",
        "graph_matcher_extra_no_match_points",
        "abstention_weight",
        "inline_false_match_mining",
        "illumination_consistency_weight",
        "illumination_consistency_probability",
        "illumination_match_weight",
        "illumination_match_probability",
        "gpu_util_percent",
        "gpu_mem_used_mib",
        "gpu_mem_total_mib",
    ]
    metrics_writer = StreamingCsvRows(args.output_dir / "train_metrics.csv", train_metric_fields).open()
    hard_specs = select_hard_lazy_specs(specs, args.hard_variant)
    base_iterator = iter_lazy_pairs(
        specs,
        count=args.steps * args.batch_pairs,
        workers=args.workers,
        prefetch=args.prefetch_batches,
        cache_max_items=args.worker_cache_items,
        crop_size=args.crop_size,
        image_source=args.image_source,
        max_attempts=args.max_attempts,
        min_valid_fraction=args.min_valid_fraction,
        absolute_depth_tolerance_m=args.absolute_depth_tolerance_m,
        relative_depth_tolerance=args.relative_depth_tolerance,
        seed=args.seed,
        skip_bad_pairs=args.skip_bad_pairs,
        max_bad_pairs=args.max_bad_pairs,
        photometric_config=photometric_config,
        input_local_contrast=args.input_local_contrast,
        local_contrast_strength=args.input_local_contrast_strength,
        local_contrast_kernel=args.input_local_contrast_kernel,
        illumination_consistency_config=illumination_consistency_config,
        illumination_consistency_probability=args.illumination_consistency_probability,
        illumination_match_config=illumination_match_config,
        illumination_match_probability=args.illumination_match_probability,
        illumination_match_changed_view=args.illumination_match_changed_view,
    )
    hard_iterator: Iterator[LazyPairResult] | None = None
    if hard_specs:
        hard_iterator = iter_lazy_pairs(
            hard_specs,
            count=args.steps * args.batch_pairs,
            workers=args.workers,
            prefetch=args.prefetch_batches,
            cache_max_items=args.worker_cache_items,
            crop_size=args.crop_size,
            image_source=args.image_source,
            max_attempts=args.max_attempts,
            min_valid_fraction=args.min_valid_fraction,
            absolute_depth_tolerance_m=args.absolute_depth_tolerance_m,
            relative_depth_tolerance=args.relative_depth_tolerance,
            seed=args.seed + 91013,
            skip_bad_pairs=args.skip_bad_pairs,
            max_bad_pairs=args.max_bad_pairs,
            photometric_config=photometric_config,
            input_local_contrast=args.input_local_contrast,
            local_contrast_strength=args.input_local_contrast_strength,
            local_contrast_kernel=args.input_local_contrast_kernel,
            illumination_consistency_config=illumination_consistency_config,
            illumination_consistency_probability=args.illumination_consistency_probability,
            illumination_match_config=illumination_match_config,
            illumination_match_probability=args.illumination_match_probability,
            illumination_match_changed_view=args.illumination_match_changed_view,
        )
        print(f"hard_lazy_specs={len(hard_specs)} hard_variants={','.join(args.hard_variant)}", flush=True)
    static_false_matches = read_false_match_labels(args.false_match_csv) if args.false_match_csv else {}
    false_match_handle = None
    false_match_writer: csv.DictWriter | None = None
    if args.mine_false_matches:
        false_output = args.false_match_output or (args.output_dir / "false_matches.csv")
        false_output.parent.mkdir(parents=True, exist_ok=True)
        false_match_handle = false_output.open("w", encoding="utf-8", newline="")
        false_match_writer = csv.DictWriter(
            false_match_handle,
            fieldnames=["pair_pt", "ax", "ay", "bx", "by", "error_px", "score"],
        )
        false_match_writer.writeheader()

    gpu_monitor: GpuUsageMonitor | None = None
    if device.type == "cuda" and args.gpu_monitor:
        gpu_monitor = GpuUsageMonitor(
            args.output_dir / "gpu_metrics.csv",
            sample_interval_s=args.gpu_sample_interval_s,
        )
        gpu_monitor.start()

    start = time.perf_counter()
    try:
        for step in range(1, args.steps + 1):
            step_start = time.perf_counter()
            fetch_start = time.perf_counter()
            hard_probability = hard_pair_probability(
                step,
                max_probability=args.hard_curriculum_max_probability,
                warmup_steps=args.hard_curriculum_warmup_steps,
            )
            use_hard_iterator = hard_iterator is not None and random.random() < hard_probability
            active_iterator = hard_iterator if use_hard_iterator and hard_iterator is not None else base_iterator
            results = [next(active_iterator) for _ in range(args.batch_pairs)]
            data_wait_ms = (time.perf_counter() - fetch_start) * 1000.0
            fake_paths = [ref_dir / f"step_{step:06d}_pair_{idx:02d}.pt" for idx in range(len(results))]
            augment_start = time.perf_counter()
            augmented_pairs = [result.pair for result in results]
            augment_ms = (time.perf_counter() - augment_start) * 1000.0
            prefetched = {path.resolve(strict=False): pair for path, pair in zip(fake_paths, augmented_pairs)}
            illumination_prefetched = {
                path.resolve(strict=False): result.illumination_pair
                for path, result in zip(fake_paths, results)
                if result.illumination_pair is not None
            }
            illumination_match_prefetched = {
                path.resolve(strict=False): result.illumination_match_pair
                for path, result in zip(fake_paths, results)
                if result.illumination_match_pair is not None
            }
            mined_false_matches = dict(static_false_matches)
            mined_false_rows: list[dict[str, object]] = []
            false_mine_ms = 0.0
            false_probability = hard_pair_probability(
                step,
                max_probability=args.false_match_curriculum_max_probability,
                warmup_steps=args.false_match_curriculum_warmup_steps,
            )
            should_mine_false_matches = (step - 1) % args.false_match_mine_every == 0
            if (
                args.mine_false_matches
                and args.false_match_weight > 0.0
                and should_mine_false_matches
                and random.random() < false_probability
            ):
                false_mine_start = time.perf_counter()
                for pair_path, pair in zip(fake_paths, augmented_pairs):
                    labels, label_rows = mine_false_matches_for_lazy_pair(
                        model,
                        pair,
                        pair_path,
                        device=device,
                        descriptor_mode=args.visual_descriptor_mode,
                        texture_blend_weight=args.texture_blend_weight,
                        keypoint_score_mode=args.visual_keypoint_score_mode,
                        max_keypoints=args.false_match_mine_max_keypoints,
                        max_matches=args.false_match_mine_max_matches,
                        min_intensity=args.min_intensity,
                        min_score=args.false_match_mine_min_score,
                        min_margin=args.false_match_mine_min_margin,
                        threshold_px=args.false_match_mine_threshold_px,
                    )
                    mined_false_matches.update(labels)
                    mined_false_rows.extend(label_rows)
                if false_match_writer is not None and false_match_handle is not None:
                    for mined_row in mined_false_rows:
                        false_match_writer.writerow(mined_row)
                    false_match_handle.flush()
                false_mine_ms = (time.perf_counter() - false_mine_start) * 1000.0
            train_start = time.perf_counter()
            metrics = train_step(
                model,
                optimizer,
                [],
                device=device,
                batch_pairs=args.batch_pairs,
                samples_per_pair=args.samples_per_pair,
                min_intensity=args.min_intensity,
                generator=generator,
                temperature=args.temperature,
                teacher_weight=args.teacher_weight,
                synthetic_loss_weight=args.synthetic_loss_weight,
                hard_negative_weight=args.hard_negative_weight,
                diversity_weight=args.diversity_weight,
                warp_hard_negative_weight=args.warp_hard_negative_weight,
                warp_hard_negative_radius=args.warp_hard_negative_radius,
                warp_hard_negative_margin=args.warp_hard_negative_margin,
                warp_hard_negative_candidates=args.warp_hard_negative_candidates,
                abstention_weight=args.abstention_weight,
                abstention_negative_radius=args.abstention_negative_radius,
                abstention_max_false_score=args.abstention_max_false_score,
                abstention_topk=args.abstention_topk,
                abstention_candidates=args.abstention_candidates,
                max_grad_norm=args.max_grad_norm,
                skip_nonfinite_steps=args.skip_nonfinite_steps,
                train_blended_descriptors=args.train_blended_descriptors,
                texture_blend_weight=args.texture_blend_weight,
                false_matches=mined_false_matches,
                false_match_weight=args.false_match_weight,
                false_match_max_points=args.false_match_max_points,
                false_match_max_score=args.false_match_max_score,
                false_match_pair_paths=fake_paths,
                false_match_probability=1.0 if mined_false_matches else 0.0,
                online_false_match_weight=args.false_match_weight if args.inline_false_match_mining else 0.0,
                online_false_match_max_points=args.false_match_max_points,
                online_false_match_max_score=args.false_match_max_score,
                online_false_match_max_keypoints=args.false_match_mine_max_keypoints,
                online_false_match_max_matches=args.false_match_mine_max_matches,
                online_false_match_min_score=args.false_match_mine_min_score,
                online_false_match_min_margin=args.false_match_mine_min_margin,
                online_false_match_threshold_px=args.false_match_mine_threshold_px,
                graph_matcher_loss_weight=args.graph_matcher_loss_weight if args.train_graph_matcher else 0.0,
                graph_matcher_metadata_mode=args.graph_matcher_metadata_mode,
                graph_matcher_no_match_points=args.graph_matcher_no_match_points,
                graph_matcher_no_match_weight=args.graph_matcher_no_match_weight,
                graph_matcher_no_match_min_distance=args.graph_matcher_no_match_min_distance,
                graph_matcher_assignment_weight=args.graph_matcher_assignment_weight,
                graph_matcher_accept_weight=args.graph_matcher_accept_weight,
                graph_matcher_accept_negative_topk=args.graph_matcher_accept_negative_topk,
                graph_matcher_prune_ranking_weight=args.graph_matcher_prune_ranking_weight,
                graph_matcher_prune_ranking_margin=args.graph_matcher_prune_ranking_margin,
                graph_matcher_stop_confidence_weight=args.graph_matcher_stop_confidence_weight,
                graph_matcher_stop_confidence_margin=args.graph_matcher_stop_confidence_margin,
                graph_matcher_raw_preservation_weight=args.graph_matcher_raw_preservation_weight,
                graph_matcher_raw_preservation_margin=args.graph_matcher_raw_preservation_margin,
                graph_matcher_raw_preservation_raw_margin=args.graph_matcher_raw_preservation_raw_margin,
                graph_matcher_hard_negative_dustbin_weight=args.graph_matcher_hard_negative_dustbin_weight,
                graph_matcher_hard_negative_dustbin_topk=args.graph_matcher_hard_negative_dustbin_topk,
                graph_matcher_hard_negative_dustbin_margin=args.graph_matcher_hard_negative_dustbin_margin,
                graph_matcher_hard_negative_dustbin_spatial_min_distance=(
                    args.graph_matcher_hard_negative_dustbin_spatial_min_distance
                ),
                graph_matcher_semi_dense_no_match_points=args.graph_matcher_semi_dense_no_match_points,
                graph_matcher_semi_dense_min_score=args.graph_matcher_semi_dense_min_score,
                graph_matcher_online_false_no_match=args.graph_matcher_online_false_no_match,
                graph_matcher_train_max_attention_layers=args.graph_matcher_train_max_attention_layers,
                graph_matcher_train_random_attention_layers=args.graph_matcher_train_random_attention_layers,
                graph_matcher_train_max_attention_work_fraction=args.graph_matcher_train_max_attention_work_fraction,
                graph_matcher_train_width_keep_ratio=args.graph_matcher_train_width_keep_ratio,
                training_spatial_bins=args.training_spatial_bins,
                training_crop_size=0,
                training_max_image_size=0,
                forced_pair_paths=fake_paths,
                prefetched_pairs=prefetched,
                illumination_consistency_pairs=illumination_prefetched,
                illumination_consistency_weight=args.illumination_consistency_weight,
                illumination_consistency_max_points=args.illumination_consistency_points,
                illumination_consistency_probability=1.0,
                illumination_match_pairs=illumination_match_prefetched,
                illumination_match_weight=args.illumination_match_weight,
                illumination_match_probability=1.0,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_ms = (time.perf_counter() - train_start) * 1000.0
            gpu_snapshot_start = time.perf_counter()
            if gpu_monitor is not None:
                gpu = gpu_monitor.latest()
            else:
                gpu = _gpu_snapshot() if _should_collect_gpu_snapshot(step, args.gpu_snapshot_every) else {}
            gpu_snapshot_ms = (time.perf_counter() - gpu_snapshot_start) * 1000.0
            hard_lazy_pairs = sum(
                1
                for result in results
                if use_hard_iterator
                or (
                    args.hard_valid_fraction_max > 0.0
                    and result.valid_fraction <= float(args.hard_valid_fraction_max)
                )
            )
            row = {
                "step": step,
                "loss": f"{metrics.get('loss', float('nan')):.6f}",
                "top1_accuracy": f"{metrics.get('top1_accuracy', float('nan')):.6f}",
                "mean_positive_rank": f"{metrics.get('mean_positive_rank', float('nan')):.3f}",
                "points": f"{metrics.get('points', 0.0):.0f}",
                "false_match_points": f"{metrics.get('false_match_points', 0.0):.0f}",
                "false_match_pairs": f"{metrics.get('false_match_pairs', 0.0):.0f}",
                "online_false_match_points": f"{metrics.get('online_false_match_points', 0.0):.0f}",
                "online_false_match_pairs": f"{metrics.get('online_false_match_pairs', 0.0):.0f}",
                "illumination_consistency_points": f"{metrics.get('illumination_consistency_points', 0.0):.0f}",
                "illumination_consistency_pairs": f"{metrics.get('illumination_consistency_pairs', 0.0):.0f}",
                "illumination_match_points": f"{metrics.get('illumination_match_points', 0.0):.0f}",
                "illumination_match_pairs": f"{metrics.get('illumination_match_pairs', 0.0):.0f}",
                "hard_lazy_pairs": hard_lazy_pairs,
                "data_wait_ms": f"{data_wait_ms:.2f}",
                "augment_ms": f"{augment_ms:.2f}",
                "false_mine_ms": f"{false_mine_ms:.2f}",
                "train_ms": f"{train_ms:.2f}",
                "gpu_snapshot_ms": f"{gpu_snapshot_ms:.2f}",
                "step_ms": f"{(time.perf_counter() - step_start) * 1000.0:.2f}",
                "valid_fraction_mean": f"{statistics.fmean(result.valid_fraction for result in results):.6f}",
                "worker_elapsed_ms_mean": f"{statistics.fmean(result.elapsed_ms for result in results):.2f}",
                "photometric_augment": int(photometric_config.enabled),
                "photometric_probability": f"{photometric_config.probability:.3f}",
                "input_local_contrast": int(args.input_local_contrast),
                "graph_matcher_loss_weight": f"{args.graph_matcher_loss_weight if args.train_graph_matcher else 0.0:.6f}",
                "graph_matcher_assignment_weight": (
                    f"{args.graph_matcher_assignment_weight if args.train_graph_matcher else 0.0:.6f}"
                ),
                "graph_matcher_accept_weight": f"{args.graph_matcher_accept_weight if args.train_graph_matcher else 0.0:.6f}",
                "graph_matcher_prune_ranking_weight": (
                    f"{args.graph_matcher_prune_ranking_weight if args.train_graph_matcher else 0.0:.6f}"
                ),
                "graph_matcher_stop_confidence_weight": (
                    f"{args.graph_matcher_stop_confidence_weight if args.train_graph_matcher else 0.0:.6f}"
                ),
                "graph_matcher_train_max_attention_layers": (
                    args.graph_matcher_train_max_attention_layers if args.train_graph_matcher else 0
                ),
                "graph_matcher_train_random_attention_layers": int(
                    bool(args.graph_matcher_train_random_attention_layers and args.train_graph_matcher)
                ),
                "graph_matcher_train_max_attention_work_fraction": (
                    f"{args.graph_matcher_train_max_attention_work_fraction if args.train_graph_matcher else 1.0:.6f}"
                ),
                "graph_matcher_train_width_keep_ratio": (
                    f"{args.graph_matcher_train_width_keep_ratio if args.train_graph_matcher else 1.0:.6f}"
                ),
                "graph_matcher_online_false_no_match": int(
                    bool(args.graph_matcher_online_false_no_match and args.train_graph_matcher)
                ),
                "graph_matcher_total_loss": f"{metrics.get('graph_matcher_total_loss', 0.0):.6f}",
                "graph_matcher_ce_loss": f"{metrics.get('graph_matcher_ce_loss', 0.0):.6f}",
                "graph_matcher_assignment_loss": f"{metrics.get('graph_matcher_assignment_loss', 0.0):.6f}",
                "graph_matcher_no_match_loss": f"{metrics.get('graph_matcher_no_match_loss', 0.0):.6f}",
                "graph_matcher_accept_loss": f"{metrics.get('graph_matcher_accept_loss', 0.0):.6f}",
                "graph_matcher_prune_ranking_loss": f"{metrics.get('graph_matcher_prune_ranking_loss', 0.0):.6f}",
                "graph_matcher_stop_confidence_loss": f"{metrics.get('graph_matcher_stop_confidence_loss', 0.0):.6f}",
                "graph_matcher_raw_preservation_loss": f"{metrics.get('graph_matcher_raw_preservation_loss', 0.0):.6f}",
                "graph_matcher_hard_negative_dustbin_loss": (
                    f"{metrics.get('graph_matcher_hard_negative_dustbin_loss', 0.0):.6f}"
                ),
                "graph_matcher_executed_attention_layers": (
                    f"{metrics.get('graph_matcher_executed_attention_layers', 0.0):.0f}"
                ),
                "graph_matcher_attention_work_fraction": (
                    f"{metrics.get('graph_matcher_attention_work_fraction', 0.0):.6f}"
                ),
                "graph_matcher_positive_pairs": f"{metrics.get('graph_matcher_positive_pairs', 0.0):.0f}",
                "graph_matcher_extra_no_match_points": (
                    f"{metrics.get('graph_matcher_extra_no_match_points', 0.0):.0f}"
                ),
                "abstention_weight": f"{args.abstention_weight:.6f}",
                "inline_false_match_mining": int(args.inline_false_match_mining),
                "illumination_consistency_weight": f"{args.illumination_consistency_weight:.6f}",
                "illumination_consistency_probability": f"{args.illumination_consistency_probability:.3f}",
                "illumination_match_weight": f"{args.illumination_match_weight:.6f}",
                "illumination_match_probability": f"{args.illumination_match_probability:.3f}",
                **gpu,
            }
            rows.append(row)
            metrics_writer.write(row)
            if step == 1 or step % max(1, args.progress_every) == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"train step={step}/{args.steps} loss={row['loss']} "
                    f"top1={row['top1_accuracy']} false={row['false_match_points']} "
                    f"gassign={row['graph_matcher_assignment_loss']} gnomatch={row['graph_matcher_no_match_loss']} "
                    f"illum={row['illumination_consistency_points']} "
                    f"illum_match={row['illumination_match_points']} hard={row['hard_lazy_pairs']} "
                    f"data_wait={data_wait_ms:.1f}ms "
                    f"train={train_ms:.1f}ms rate={step / max(elapsed, 1.0e-6):.2f} step/s",
                    flush=True,
                )
            if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                _save_training_state(args.output_dir / "checkpoints" / "latest_pytorch_pfm_state.pt", model, args, step)
    finally:
        metrics_writer.close()
        if false_match_handle is not None:
            false_match_handle.close()
        if gpu_monitor is not None:
            gpu_monitor.stop()
    elapsed = time.perf_counter() - start
    data_wait_values = [float(row["data_wait_ms"]) for row in rows]
    augment_values = [float(row["augment_ms"]) for row in rows]
    false_mine_values = [float(row["false_mine_ms"]) for row in rows]
    train_values = [float(row["train_ms"]) for row in rows]
    gpu_snapshot_values = [float(row["gpu_snapshot_ms"]) for row in rows]
    step_values = [float(row["step_ms"]) for row in rows]
    summary = {
        "mode": "train",
        "steps": len(rows),
        "elapsed_s": elapsed,
        "steps_per_second": len(rows) / max(elapsed, 1.0e-6),
        "data_wait_ms": _summarize_float(data_wait_values),
        "augment_ms": _summarize_float(augment_values),
        "false_mine_ms": _summarize_float(false_mine_values),
        "train_ms": _summarize_float(train_values),
        "gpu_snapshot_ms": _summarize_float(gpu_snapshot_values),
        "step_ms": _summarize_float(step_values),
        "data_wait_to_train_ratio_mean": statistics.fmean(data_wait_values) / max(statistics.fmean(train_values), 1.0e-6),
        "last_loss": rows[-1]["loss"] if rows else "-",
        "last_top1_accuracy": rows[-1]["top1_accuracy"] if rows else "-",
        "photometric_augmentation": photometric_config,
        "illumination_consistency": {
            "enabled": float(args.illumination_consistency_weight) > 0.0,
            "weight": float(args.illumination_consistency_weight),
            "probability": float(args.illumination_consistency_probability),
            "points": int(args.illumination_consistency_points),
            "config": vars(illumination_consistency_config),
        },
        "illumination_match": {
            "enabled": float(args.illumination_match_weight) > 0.0,
            "weight": float(args.illumination_match_weight),
            "probability": float(args.illumination_match_probability),
            "changed_view": args.illumination_match_changed_view,
            "config": vars(illumination_match_config),
        },
        "gpu_monitor_enabled": bool(gpu_monitor is not None),
        "gpu_sample_interval_s": float(args.gpu_sample_interval_s),
        "last_gpu": gpu_monitor.latest() if gpu_monitor is not None else _gpu_snapshot(),
    }
    _write_rows(
        args.output_dir / "train_metrics.csv",
        rows,
        train_metric_fields,
    )
    _write_html_report(
        args.output_dir / "run.html",
        title="Lazy Pose Pair GPU Feeding Benchmark",
        args=args,
        spec_count=len(specs),
        summary=summary,
        rows=rows,
    )
    checkpoint_path = args.output_dir / "pytorch_pfm_state.pt"
    _save_training_state(checkpoint_path, model, args, args.steps)
    summary["checkpoint"] = str(checkpoint_path)
    if device.type == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()
    report_dir = _run_visual_report(args, checkpoint_path)
    if report_dir is not None:
        summary["visual_report"] = str(report_dir / "index.html")
    return summary


def _save_training_state(
    path: Path,
    model: pfm_model.PlanetaryFeatureMatcher,
    args: argparse.Namespace,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "input_channels": model.config.input_channels,
                "base_channels": model.config.base_channels,
                "descriptor_dim": model.config.descriptor_dim,
                "graph_hidden_dim": model.config.graph_hidden_dim,
                "graph_attention_layers": model.config.graph_attention_layers,
                "graph_keypoint_meta_dim": model.config.graph_keypoint_meta_dim,
            },
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "training": {
                "script": "scripts/benchmark_lazy_pose_pairs.py",
                "step": int(step),
                "steps": int(args.steps),
                "crop_size": int(args.crop_size),
                "batch_pairs": int(args.batch_pairs),
                "samples_per_pair": int(args.samples_per_pair),
                "render_manifest": str(_first_path(args.render_manifest) or ""),
                "uint8_manifest": str(_first_path(args.uint8_manifest) or ""),
                "render_manifests": [str(path) for path in _path_list(args.render_manifest)],
                "uint8_manifests": [str(path) for path in _path_list(args.uint8_manifest)],
                "init_pytorch_state": str(args.init_pytorch_state),
                "photometric_augment": bool(args.photometric_augment),
                "photometric_probability": float(args.photometric_probability),
                "photometric_brightness": float(args.photometric_brightness),
                "photometric_contrast": float(args.photometric_contrast),
                "photometric_gamma": float(args.photometric_gamma),
                "photometric_shadow": float(args.photometric_shadow),
                "photometric_noise": float(args.photometric_noise),
                "illumination_consistency_weight": float(args.illumination_consistency_weight),
                "illumination_consistency_probability": float(args.illumination_consistency_probability),
                "illumination_consistency_points": int(args.illumination_consistency_points),
                "illumination_match_weight": float(args.illumination_match_weight),
                "illumination_match_probability": float(args.illumination_match_probability),
                "illumination_match_changed_view": str(args.illumination_match_changed_view),
                "input_local_contrast": bool(args.input_local_contrast),
                "input_local_contrast_strength": float(args.input_local_contrast_strength),
                "gpu_snapshot_every": int(args.gpu_snapshot_every),
                "hard_variant": list(args.hard_variant),
                "hard_curriculum_max_probability": float(args.hard_curriculum_max_probability),
                "false_match_csv": [str(path) for path in args.false_match_csv],
                "false_match_weight": float(args.false_match_weight),
                "mine_false_matches": bool(args.mine_false_matches),
                "false_match_mine_every": int(args.false_match_mine_every),
                "train_graph_matcher": bool(args.train_graph_matcher),
                "graph_matcher_loss_weight": float(args.graph_matcher_loss_weight),
                "graph_matcher_no_match_points": int(args.graph_matcher_no_match_points),
                "graph_matcher_no_match_weight": float(args.graph_matcher_no_match_weight),
                "graph_matcher_no_match_min_distance": float(args.graph_matcher_no_match_min_distance),
                "graph_matcher_assignment_weight": float(args.graph_matcher_assignment_weight),
                "graph_matcher_accept_weight": float(args.graph_matcher_accept_weight),
                "graph_matcher_prune_ranking_weight": float(args.graph_matcher_prune_ranking_weight),
                "graph_matcher_stop_confidence_weight": float(args.graph_matcher_stop_confidence_weight),
                "graph_matcher_train_max_attention_layers": int(args.graph_matcher_train_max_attention_layers),
                "graph_matcher_train_random_attention_layers": bool(args.graph_matcher_train_random_attention_layers),
                "graph_matcher_train_max_attention_work_fraction": float(
                    args.graph_matcher_train_max_attention_work_fraction
                ),
                "graph_matcher_train_width_keep_ratio": float(args.graph_matcher_train_width_keep_ratio),
                "graph_matcher_online_false_no_match": bool(args.graph_matcher_online_false_no_match),
                "abstention_weight": float(args.abstention_weight),
                "warp_hard_negative_weight": float(args.warp_hard_negative_weight),
            },
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--uint8-manifest", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["preprocess", "train"], default="preprocess")
    parser.add_argument("--split", default="train")
    parser.add_argument("--reference-variant", default="nadir")
    parser.add_argument("--target-variant", action="append", default=[])
    parser.add_argument("--image-source", choices=["uint8", "render"], default="uint8")
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=8)
    parser.add_argument("--worker-cache-items", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--min-valid-fraction", type=float, default=0.02)
    parser.add_argument("--absolute-depth-tolerance-m", type=float, default=100.0)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--gpu-snapshot-every", type=int, default=25)
    parser.add_argument("--gpu-monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-sample-interval-s", type=float, default=1.0)
    parser.add_argument("--skip-bad-pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-bad-pairs", type=int, default=0)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-pytorch-state", type=Path, default=DEFAULT_INIT_STATE)
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--samples-per-pair", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=0.10)
    parser.add_argument("--warp-hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--warp-hard-negative-radius", type=float, default=2.0)
    parser.add_argument("--warp-hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--warp-hard-negative-candidates", type=int, default=4096)
    parser.add_argument("--abstention-weight", type=float, default=0.0)
    parser.add_argument("--abstention-negative-radius", type=float, default=2.0)
    parser.add_argument("--abstention-max-false-score", type=float, default=0.35)
    parser.add_argument("--abstention-topk", type=int, default=8)
    parser.add_argument("--abstention-candidates", type=int, default=4096)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--skip-nonfinite-steps", action="store_true")
    parser.add_argument("--train-blended-descriptors", action="store_true")
    parser.add_argument("--texture-blend-weight", type=float, default=pfm_model.INFERENCE_TEXTURE_BLEND_WEIGHT)
    parser.add_argument("--graph-matcher-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--graph-matcher-metadata-mode",
        choices=["full", "descriptor_only", "no_xy", "no_geometry", "no_quality"],
        default="full",
    )
    parser.add_argument("--graph-matcher-no-match-points", type=int, default=0)
    parser.add_argument("--graph-matcher-no-match-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-no-match-min-distance", type=float, default=4.0)
    parser.add_argument("--graph-matcher-assignment-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-train-max-attention-layers", type=int, default=0)
    parser.add_argument("--graph-matcher-train-random-attention-layers", action="store_true")
    parser.add_argument("--graph-matcher-train-max-attention-work-fraction", type=float, default=1.0)
    parser.add_argument("--graph-matcher-train-width-keep-ratio", type=float, default=1.0)
    parser.add_argument("--graph-matcher-accept-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-accept-negative-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-prune-ranking-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-prune-ranking-margin", type=float, default=0.25)
    parser.add_argument("--graph-matcher-stop-confidence-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-stop-confidence-margin", type=float, default=0.5)
    parser.add_argument("--graph-matcher-raw-preservation-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-raw-preservation-margin", type=float, default=1.0)
    parser.add_argument("--graph-matcher-raw-preservation-raw-margin", type=float, default=0.05)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-weight", type=float, default=0.0)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-topk", type=int, default=8)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-margin", type=float, default=0.25)
    parser.add_argument("--graph-matcher-hard-negative-dustbin-spatial-min-distance", type=float, default=0.0)
    parser.add_argument("--graph-matcher-semi-dense-no-match-points", type=int, default=0)
    parser.add_argument("--graph-matcher-semi-dense-min-score", type=float, default=0.0)
    parser.add_argument("--graph-matcher-online-false-no-match", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--training-spatial-bins", type=int, default=0)
    parser.add_argument("--hard-variant", action="append", default=[])
    parser.add_argument("--hard-valid-fraction-max", type=float, default=0.0)
    parser.add_argument("--hard-curriculum-max-probability", type=float, default=0.0)
    parser.add_argument("--hard-curriculum-warmup-steps", type=int, default=100)
    parser.add_argument("--false-match-csv", action="append", type=Path, default=[])
    parser.add_argument("--false-match-weight", type=float, default=0.0)
    parser.add_argument("--false-match-max-points", type=int, default=128)
    parser.add_argument("--false-match-max-score", type=float, default=0.25)
    parser.add_argument("--false-match-curriculum-max-probability", type=float, default=0.0)
    parser.add_argument("--false-match-curriculum-warmup-steps", type=int, default=100)
    parser.add_argument("--mine-false-matches", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--false-match-output", type=Path, default=None)
    parser.add_argument("--false-match-mine-max-keypoints", type=int, default=384)
    parser.add_argument("--false-match-mine-max-matches", type=int, default=0)
    parser.add_argument("--false-match-mine-min-score", type=float, default=-1.0)
    parser.add_argument("--false-match-mine-min-margin", type=float, default=0.02)
    parser.add_argument("--false-match-mine-threshold-px", type=float, default=5.0)
    parser.add_argument("--false-match-mine-every", type=int, default=1)
    parser.add_argument("--inline-false-match-mining", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-local-contrast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-local-contrast-strength", type=float, default=0.0)
    parser.add_argument("--input-local-contrast-kernel", type=int, default=31)
    parser.add_argument("--photometric-augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--photometric-probability", type=float, default=0.85)
    parser.add_argument("--photometric-brightness", type=float, default=0.16)
    parser.add_argument("--photometric-contrast", type=float, default=0.35)
    parser.add_argument("--photometric-gamma", type=float, default=0.45)
    parser.add_argument("--photometric-shadow", type=float, default=0.45)
    parser.add_argument("--photometric-noise", type=float, default=0.015)
    parser.add_argument("--illumination-consistency-weight", type=float, default=0.0)
    parser.add_argument("--illumination-consistency-probability", type=float, default=1.0)
    parser.add_argument("--illumination-consistency-points", type=int, default=128)
    parser.add_argument("--illumination-consistency-brightness", type=float, default=0.22)
    parser.add_argument("--illumination-consistency-contrast", type=float, default=0.45)
    parser.add_argument("--illumination-consistency-gamma", type=float, default=0.75)
    parser.add_argument("--illumination-consistency-shadow", type=float, default=0.65)
    parser.add_argument("--illumination-consistency-noise", type=float, default=0.02)
    parser.add_argument("--illumination-match-weight", type=float, default=0.0)
    parser.add_argument("--illumination-match-probability", type=float, default=1.0)
    parser.add_argument("--illumination-match-changed-view", choices=["a", "b", "both"], default="b")
    parser.add_argument("--illumination-match-brightness", type=float, default=0.25)
    parser.add_argument("--illumination-match-contrast", type=float, default=0.55)
    parser.add_argument("--illumination-match-gamma", type=float, default=0.95)
    parser.add_argument("--illumination-match-shadow", type=float, default=0.75)
    parser.add_argument("--illumination-match-noise", type=float, default=0.02)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--train-dual-fpn", action="store_true")
    parser.add_argument("--train-descriptor-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sparse-context", action="store_true")
    parser.add_argument("--train-keypoint-head", action="store_true")
    parser.add_argument("--train-geometry-head", action="store_true")
    parser.add_argument("--train-texture-adapter", action="store_true")
    parser.add_argument("--train-descriptor-fusion", action="store_true")
    parser.add_argument("--train-quality-head", action="store_true")
    parser.add_argument("--train-graph-matcher", action="store_true")
    parser.add_argument("--auto-visual-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-split", default="")
    parser.add_argument("--visual-device", default="")
    parser.add_argument("--visual-candidate-pairs", type=int, default=24)
    parser.add_argument("--visual-select-count", type=int, default=6)
    parser.add_argument("--visual-max-image-size", type=int, default=768)
    parser.add_argument("--visual-descriptor-mode", choices=["learned", "texture", "blend"], default="learned")
    parser.add_argument("--visual-keypoint-score-mode", choices=["texture", "learned"], default="texture")
    parser.add_argument("--visual-matcher-mode", choices=["raw_descriptor", "graph_matcher"], default="raw_descriptor")
    parser.add_argument("--visual-max-keypoints", type=int, default=384)
    parser.add_argument("--visual-max-matches", type=int, default=0)
    parser.add_argument("--visual-draw-matches", type=int, default=0)
    parser.add_argument("--visual-threshold-px", type=float, default=5.0)
    parser.add_argument("--visual-graph-width-prune-min-score", type=float, default=-1.0)
    parser.add_argument("--visual-graph-early-stop-min-confidence", type=float, default=-1.0)
    parser.add_argument("--visual-filtered-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-filtered-mutual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-filtered-geometry-filter", choices=["none", "affine", "local"], default="local")
    parser.add_argument("--visual-filtered-min-score", type=float, default=-1.0)
    parser.add_argument("--visual-filtered-min-margin", type=float, default=0.02)
    parser.add_argument("--visual-filtered-max-matches", type=int, default=0)
    parser.add_argument("--visual-filtered-draw-matches", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.prefetch_batches <= 0:
        raise ValueError("--prefetch-batches must be positive")
    if args.crop_size <= 0:
        raise ValueError("--crop-size must be positive")
    if args.mode == "preprocess" and args.pairs <= 0:
        raise ValueError("--pairs must be positive in preprocess mode")
    if args.mode == "train" and args.steps <= 0:
        raise ValueError("--steps must be positive in train mode")
    if args.gpu_snapshot_every <= 0:
        raise ValueError("--gpu-snapshot-every must be positive")
    if args.gpu_sample_interval_s <= 0.0:
        raise ValueError("--gpu-sample-interval-s must be positive")
    if not 0.0 <= args.photometric_probability <= 1.0:
        raise ValueError("--photometric-probability must be in [0, 1]")
    if not 0.0 <= args.illumination_consistency_probability <= 1.0:
        raise ValueError("--illumination-consistency-probability must be in [0, 1]")
    if args.illumination_consistency_weight < 0.0:
        raise ValueError("--illumination-consistency-weight must be nonnegative")
    if args.illumination_consistency_points < 0:
        raise ValueError("--illumination-consistency-points must be nonnegative")
    if not 0.0 <= args.illumination_match_probability <= 1.0:
        raise ValueError("--illumination-match-probability must be in [0, 1]")
    if args.illumination_match_weight < 0.0:
        raise ValueError("--illumination-match-weight must be nonnegative")
    if args.visual_max_matches < 0:
        raise ValueError("--visual-max-matches must be nonnegative; use 0 to keep all matches")
    if args.visual_draw_matches < 0:
        raise ValueError("--visual-draw-matches must be nonnegative; use 0 to draw all matches")
    if args.visual_filtered_max_matches < 0:
        raise ValueError("--visual-filtered-max-matches must be nonnegative; use 0 to keep all matches")
    if args.visual_filtered_draw_matches < 0:
        raise ValueError("--visual-filtered-draw-matches must be nonnegative; use 0 to draw all matches")
    if args.visual_filtered_min_margin < 0.0:
        raise ValueError("--visual-filtered-min-margin must be non-negative")
    if args.visual_graph_width_prune_min_score < -1.0:
        raise ValueError("--visual-graph-width-prune-min-score must be at least -1.0; -1 disables pruning")
    if args.visual_graph_early_stop_min_confidence < -1.0:
        raise ValueError("--visual-graph-early-stop-min-confidence must be at least -1.0; -1 disables early stopping")
    if args.hard_curriculum_max_probability < 0.0 or args.hard_curriculum_max_probability > 1.0:
        raise ValueError("--hard-curriculum-max-probability must be in [0, 1]")
    if args.false_match_curriculum_max_probability < 0.0 or args.false_match_curriculum_max_probability > 1.0:
        raise ValueError("--false-match-curriculum-max-probability must be in [0, 1]")
    if args.false_match_weight < 0.0:
        raise ValueError("--false-match-weight must be non-negative")
    if args.false_match_max_points < 0:
        raise ValueError("--false-match-max-points must be non-negative")
    if args.false_match_mine_max_keypoints <= 0:
        raise ValueError("--false-match-mine-max-keypoints must be positive")
    if args.false_match_mine_max_matches < 0:
        raise ValueError("--false-match-mine-max-matches must be nonnegative; use 0 to keep all matches")
    if args.false_match_mine_min_margin < 0.0:
        raise ValueError("--false-match-mine-min-margin must be non-negative")
    if args.false_match_mine_every <= 0:
        raise ValueError("--false-match-mine-every must be positive")
    if args.input_local_contrast_strength < 0.0 or args.input_local_contrast_strength > 1.0:
        raise ValueError("--input-local-contrast-strength must be in [0, 1]")
    if args.graph_matcher_loss_weight < 0.0:
        raise ValueError("--graph-matcher-loss-weight must be non-negative")
    if args.graph_matcher_no_match_points < 0:
        raise ValueError("--graph-matcher-no-match-points must be non-negative")
    if args.graph_matcher_no_match_weight < 0.0:
        raise ValueError("--graph-matcher-no-match-weight must be non-negative")
    if args.graph_matcher_no_match_min_distance < 0.0:
        raise ValueError("--graph-matcher-no-match-min-distance must be non-negative")
    if args.graph_matcher_assignment_weight < 0.0:
        raise ValueError("--graph-matcher-assignment-weight must be non-negative")
    if args.graph_matcher_train_max_attention_layers < 0:
        raise ValueError("--graph-matcher-train-max-attention-layers must be non-negative")
    if (
        not math.isfinite(float(args.graph_matcher_train_max_attention_work_fraction))
        or args.graph_matcher_train_max_attention_work_fraction < 0.0
        or args.graph_matcher_train_max_attention_work_fraction > 1.0
    ):
        raise ValueError("--graph-matcher-train-max-attention-work-fraction must be in [0, 1]")
    if (
        not math.isfinite(float(args.graph_matcher_train_width_keep_ratio))
        or args.graph_matcher_train_width_keep_ratio <= 0.0
        or args.graph_matcher_train_width_keep_ratio > 1.0
    ):
        raise ValueError("--graph-matcher-train-width-keep-ratio must be in (0, 1]")
    if args.graph_matcher_accept_weight < 0.0:
        raise ValueError("--graph-matcher-accept-weight must be non-negative")
    if args.graph_matcher_accept_negative_topk < 0:
        raise ValueError("--graph-matcher-accept-negative-topk must be non-negative")
    if args.graph_matcher_prune_ranking_weight < 0.0:
        raise ValueError("--graph-matcher-prune-ranking-weight must be non-negative")
    if args.graph_matcher_prune_ranking_margin < 0.0:
        raise ValueError("--graph-matcher-prune-ranking-margin must be non-negative")
    if args.graph_matcher_stop_confidence_weight < 0.0:
        raise ValueError("--graph-matcher-stop-confidence-weight must be non-negative")
    if args.graph_matcher_stop_confidence_margin < 0.0:
        raise ValueError("--graph-matcher-stop-confidence-margin must be non-negative")
    if args.graph_matcher_raw_preservation_weight < 0.0:
        raise ValueError("--graph-matcher-raw-preservation-weight must be non-negative")
    if args.graph_matcher_raw_preservation_margin < 0.0:
        raise ValueError("--graph-matcher-raw-preservation-margin must be non-negative")
    if args.graph_matcher_raw_preservation_raw_margin < 0.0:
        raise ValueError("--graph-matcher-raw-preservation-raw-margin must be non-negative")
    if args.graph_matcher_hard_negative_dustbin_weight < 0.0:
        raise ValueError("--graph-matcher-hard-negative-dustbin-weight must be non-negative")
    if args.graph_matcher_hard_negative_dustbin_topk < 0:
        raise ValueError("--graph-matcher-hard-negative-dustbin-topk must be non-negative")
    if args.graph_matcher_hard_negative_dustbin_margin < 0.0:
        raise ValueError("--graph-matcher-hard-negative-dustbin-margin must be non-negative")
    if args.graph_matcher_hard_negative_dustbin_spatial_min_distance < 0.0:
        raise ValueError("--graph-matcher-hard-negative-dustbin-spatial-min-distance must be non-negative")
    if args.graph_matcher_semi_dense_no_match_points < 0:
        raise ValueError("--graph-matcher-semi-dense-no-match-points must be non-negative")
    if args.graph_matcher_semi_dense_min_score < 0.0:
        raise ValueError("--graph-matcher-semi-dense-min-score must be non-negative")
    for name in (
        "photometric_brightness",
        "photometric_contrast",
        "photometric_gamma",
        "photometric_shadow",
        "photometric_noise",
        "illumination_consistency_brightness",
        "illumination_consistency_contrast",
        "illumination_consistency_gamma",
        "illumination_consistency_shadow",
        "illumination_consistency_noise",
        "illumination_match_brightness",
        "illumination_match_contrast",
        "illumination_match_gamma",
        "illumination_match_shadow",
        "illumination_match_noise",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    render_manifests = _path_list(args.render_manifest)
    uint8_manifests = _path_list(args.uint8_manifest)
    records = _read_all_render_records(render_manifests, uint8_manifests)
    target_variants = tuple(args.target_variant) if args.target_variant else DEFAULT_TARGET_VARIANTS
    specs = build_pair_specs(
        records,
        split=args.split,
        reference_variant=args.reference_variant,
        target_variants=target_variants,
        image_source=args.image_source,
        limit_pairs=args.limit_pairs,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    if not specs:
        raise RuntimeError("no lazy pair specs found")

    metadata = {
        "render_manifest": str(_first_path(args.render_manifest) or ""),
        "uint8_manifest": str(_first_path(args.uint8_manifest) or ""),
        "render_manifests": [str(path) for path in render_manifests],
        "uint8_manifests": [str(path) for path in uint8_manifests],
        "records": len(records),
        "specs": len(specs),
        "target_variants": list(target_variants),
        "photometric_augmentation": vars(_photometric_config_from_args(args)),
        "illumination_consistency": {
            "enabled": float(args.illumination_consistency_weight) > 0.0,
            "weight": float(args.illumination_consistency_weight),
            "probability": float(args.illumination_consistency_probability),
            "points": int(args.illumination_consistency_points),
            "config": vars(_illumination_consistency_config_from_args(args)),
        },
        "illumination_match": {
            "enabled": float(args.illumination_match_weight) > 0.0,
            "weight": float(args.illumination_match_weight),
            "probability": float(args.illumination_match_probability),
            "changed_view": args.illumination_match_changed_view,
            "config": vars(_illumination_match_config_from_args(args)),
        },
    }
    (args.output_dir / "input_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)

    if args.mode == "preprocess":
        summary = run_preprocess(args, specs)
    else:
        summary = run_train(args, specs)
    print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
