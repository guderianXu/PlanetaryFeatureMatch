#!/usr/bin/env python3
"""Build cross-camera-position PFM pair caches from existing pose-sim archives.

The original pose-sim cache pairs images from the same orbit sample with
different virtual camera attitudes/focals. This script reuses those archives to
recover a base CameraA depth map, then projects CameraA at one sequence into
CameraA at a different sequence to create a harder cross-position pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from compact_pair_cache import (  # noqa: E402
    is_compact_pair_payload,
    load_shared_image,
    resolve_compact_image_path,
)

SIM_ROOT = PROJECT_ROOT / "辅助软件" / "数据模拟"
DEFAULT_SAT_SIM = SIM_ROOT / "build" / "sat_sim_cuda"
DEFAULT_DEM = SIM_ROOT / "dem" / "mar.tif"
DEFAULT_DOM = SIM_ROOT / "dom" / "HX1_GRAS_MoRIC_DOM_076m_Global_A.tif"

SOURCE_RE = re.compile(r"^source_(\d+)_(.+)$")
SOURCE_REPART_RE = re.compile(r"^source_repart_\d+_[^_]+_source_(\d+)_(.+)$")
SOURCE_CROSS_RE = re.compile(r"^source_cross_\d+_(.+)_off\d+_s(\d+)_to_s(\d+)$")
PAIR_VIEW_RE = re.compile(r"^pair_\d+_(.+)\.pt$")


class PairArchive(torch.nn.Module):
    def __init__(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        warp_a_to_b: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("view_a", view_a.contiguous())
        self.register_buffer("view_b", view_b.contiguous())
        self.register_buffer("warp_a_to_b", warp_a_to_b.contiguous())
        self.register_buffer("valid_mask", valid_mask.contiguous())

    def forward(self) -> torch.Tensor:
        return self.view_a


@dataclass(frozen=True)
class Camera:
    fu: float
    fv: float
    cu: float
    cv: float
    center: np.ndarray
    rotation_world_to_camera: np.ndarray


@dataclass(frozen=True)
class SourceRecord:
    split: str
    track: str
    seq: int
    source_dir: Path
    base_pair: Path
    camera_a_path: Path


@dataclass
class SourceData:
    view: torch.Tensor
    depth: np.ndarray
    camera_a: Camera


class CrossPair(NamedTuple):
    bucket: str
    offset: int
    source_a: SourceRecord
    source_b: SourceRecord


def parse_tsai(path: Path) -> Camera:
    values: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        try:
            values[key] = [float(part) for part in value.split()]
        except ValueError:
            continue
    missing = {"fu", "fv", "cu", "cv", "C", "R"} - values.keys()
    if missing:
        raise ValueError(f"{path} missing TSAI fields: {sorted(missing)}")
    r = values["R"]
    if len(r) != 9 or len(values["C"]) != 3:
        raise ValueError(f"{path} has invalid C/R field length")
    rotation = np.array(
        [[r[0], r[3], r[6]], [r[1], r[4], r[7]], [r[2], r[5], r[8]]],
        dtype=np.float64,
    )
    return Camera(
        fu=float(values["fu"][0]),
        fv=float(values["fv"][0]),
        cu=float(values["cu"][0]),
        cv=float(values["cv"][0]),
        center=np.asarray(values["C"], dtype=np.float64),
        rotation_world_to_camera=rotation,
    )


def load_archive(path: Path):
    try:
        return torch.jit.load(str(path), map_location="cpu")
    except RuntimeError as jit_error:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="'torch.load' received a zip file")
            payload = torch.load(path, map_location="cpu")
        if not is_compact_pair_payload(payload):
            raise jit_error
        image_a = resolve_compact_image_path(path, str(payload["image_a"]))
        image_b = resolve_compact_image_path(path, str(payload["image_b"]))
        return SimpleNamespace(
            view_a=load_shared_image(image_a),
            view_b=load_shared_image(image_b),
            warp_a_to_b=payload["warp_a_to_b"].detach().cpu().to(torch.float32).contiguous(),
            valid_mask=payload["valid_mask"].detach().cpu().to(torch.bool).contiguous(),
        )


def normalize_view(view: torch.Tensor) -> torch.Tensor:
    view = view.detach().cpu().to(torch.float32)
    if view.dim() == 2:
        view = view.unsqueeze(0)
    if view.dim() != 3 or view.size(0) != 1:
        raise ValueError(f"expected view tensor 1xHxW, got {tuple(view.shape)}")
    view = torch.nan_to_num(view, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    return view.contiguous()


def read_float_tif(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32, copy=False)


def asp_env() -> dict[str, str]:
    env = os.environ.copy()
    asp_lib = "/home/xjw/anaconda3/envs/asp36/lib"
    env["LD_LIBRARY_PATH"] = asp_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def ensure_dem_vrt(output_root: Path, dem_path: Path) -> Path:
    vrt_path = output_root / "depth_cache" / "dem_lon0_shifted_for_sat_sim.vrt"
    if vrt_path.exists():
        return vrt_path
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    info = json.loads(subprocess.check_output(["gdalinfo", "-json", str(dem_path)], text=True))
    width, height = info["size"]
    gt = info["geoTransform"]
    half = width // 2
    srs = "+proj=eqc +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +a=3396190 +b=3396190 +units=m +no_defs"
    vrt_path.write_text(
        f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <SRS>{srs}</SRS>
  <GeoTransform>{gt[0]}, {gt[1]}, {gt[2]}, {gt[3]}, {gt[4]}, {gt[5]}</GeoTransform>
  <VRTRasterBand dataType="Int16" band="1">
    <NoDataValue>-32768</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{dem_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="Int16" BlockXSize="128" BlockYSize="1"/>
      <SrcRect xOff="{half}" yOff="0" xSize="{width - half}" ySize="{height}"/>
      <DstRect xOff="0" yOff="0" xSize="{width - half}" ySize="{height}"/>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{dem_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="Int16" BlockXSize="128" BlockYSize="1"/>
      <SrcRect xOff="0" yOff="0" xSize="{half}" ySize="{height}"/>
      <DstRect xOff="{width - half}" yOff="0" xSize="{half}" ySize="{height}"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
""",
        encoding="utf-8",
    )
    return vrt_path


def render_camera_depth(
    *,
    output_root: Path,
    sat_sim: Path,
    dem: Path,
    dom: Path,
    camera_path: Path,
    track: str,
    seq: int,
    height: int,
    width: int,
    jobs: int,
) -> Path:
    depth_path = output_root / "depth_cache" / track / f"{seq:05d}.tif"
    if depth_path.exists():
        return depth_path
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_root / "depth_cache" / "work" / f"{track}_{seq:05d}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    links_dir = work_dir / "tsai_links"
    render_dir = work_dir / "render"
    links_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    link = links_dir / f"A_{seq:05d}.tsai"
    link.symlink_to(camera_path)
    camera_list = work_dir / "camera_list.txt"
    camera_list.write_text(str(link) + "\n", encoding="utf-8")
    dem_vrt = ensure_dem_vrt(output_root, dem)
    command = [
        str(sat_sim),
        "--dem",
        str(dem_vrt),
        "--ortho",
        str(dom),
        "--camera-list",
        str(camera_list),
        "--image-size",
        str(width),
        str(height),
        "--jobs",
        str(jobs),
        "--write-depth",
        "-o",
        str(render_dir / "batch"),
    ]
    log_path = output_root / "depth_cache" / "logs" / f"{track}_{seq:05d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=SIM_ROOT, env=asp_env(), stdout=handle, stderr=subprocess.STDOUT, check=True)
    rendered_depth = render_dir / "depth" / f"batch-A_{seq:05d}.tif"
    if not rendered_depth.exists():
        candidates = sorted((render_dir / "depth").glob("*.tif"))
        raise FileNotFoundError(f"expected {rendered_depth}, found {candidates[:5]}")
    shutil.move(str(rendered_depth), str(depth_path))
    shutil.rmtree(work_dir, ignore_errors=True)
    return depth_path


def _render_record_depth_worker(args: tuple[SourceRecord, Path, Path, Path, Path, int]) -> str:
    record, output_root, sat_sim, dem, dom, sat_sim_jobs = args
    module = load_archive(record.base_pair)
    view = normalize_view(module.view_a)
    _, height, width = view.shape
    path = render_camera_depth(
        output_root=output_root,
        sat_sim=sat_sim,
        dem=dem,
        dom=dom,
        camera_path=record.camera_a_path,
        track=record.track,
        seq=record.seq,
        height=height,
        width=width,
        jobs=sat_sim_jobs,
    )
    return str(path)


def pre_render_depth_cache(
    *,
    records: list[SourceRecord],
    output_root: Path,
    sat_sim: Path,
    dem: Path,
    dom: Path,
    sat_sim_jobs: int,
    workers: int,
) -> None:
    unique: dict[tuple[str, int], SourceRecord] = {}
    for record in records:
        unique.setdefault((record.track, record.seq), record)
    pending = [
        record
        for record in unique.values()
        if not (output_root / "depth_cache" / record.track / f"{record.seq:05d}.tif").exists()
    ]
    if not pending:
        return
    print(f"pre_render_depth pending={len(pending)} workers={workers} sat_sim_jobs={sat_sim_jobs}", flush=True)
    if workers <= 1:
        for idx, record in enumerate(pending, 1):
            _render_record_depth_worker((record, output_root, sat_sim, dem, dom, sat_sim_jobs))
            if idx == 1 or idx % 25 == 0:
                print(f"pre_render_depth done={idx}/{len(pending)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_render_record_depth_worker, (record, output_root, sat_sim, dem, dom, sat_sim_jobs))
            for record in pending
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            future.result()
            if idx == 1 or idx % 25 == 0:
                print(f"pre_render_depth done={idx}/{len(pending)}", flush=True)


def recover_depth_from_warp(module, camera_a: Camera, camera_b: Camera) -> np.ndarray:
    view = normalize_view(module.view_a)
    _, height, width = view.shape
    warp = module.warp_a_to_b.detach().cpu().numpy().astype(np.float64, copy=False)
    valid = module.valid_mask.detach().cpu().numpy().astype(bool, copy=False)
    if warp.shape != (height, width, 2) or valid.shape != (height, width):
        raise ValueError("archive warp/mask shapes do not match view_a")

    yy, xx = np.indices((height, width), dtype=np.float64)
    dx = (xx + 0.5 - camera_a.cu) / camera_a.fu
    dy = (yy + 0.5 - camera_a.cv) / camera_a.fv

    x_b = (warp[..., 0] + 0.5 - camera_b.cu) / camera_b.fu
    y_b = (warp[..., 1] + 0.5 - camera_b.cv) / camera_b.fv

    transform = camera_b.rotation_world_to_camera @ camera_a.rotation_world_to_camera.T
    qx = transform[0, 0] * dx + transform[0, 1] * dy + transform[0, 2]
    qy = transform[1, 0] * dx + transform[1, 1] * dy + transform[1, 2]
    qz = transform[2, 0] * dx + transform[2, 1] * dy + transform[2, 2]
    t = camera_b.rotation_world_to_camera @ (camera_a.center - camera_b.center)

    denom_x = qx - x_b * qz
    denom_y = qy - y_b * qz
    numer_x = x_b * t[2] - t[0]
    numer_y = y_b * t[2] - t[1]

    usable_x = np.abs(denom_x) > 1.0e-9
    usable_y = np.abs(denom_y) > 1.0e-9
    z_x = np.divide(numer_x, denom_x, out=np.zeros_like(numer_x), where=usable_x)
    z_y = np.divide(numer_y, denom_y, out=np.zeros_like(numer_y), where=usable_y)
    count = usable_x.astype(np.float64) + usable_y.astype(np.float64)
    depth = np.divide(
        z_x * usable_x.astype(np.float64) + z_y * usable_y.astype(np.float64),
        count,
        out=np.zeros_like(z_x),
        where=count > 0,
    )
    recovered = valid & np.isfinite(depth) & (depth > 0.0)
    depth = depth.astype(np.float32, copy=False)
    depth[~recovered] = 0.0
    return depth


def sphere_depth_for_camera(camera: Camera, height: int, width: int, *, mars_radius_m: float) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float64)
    dx = (xx + 0.5 - camera.cu) / camera.fu
    dy = (yy + 0.5 - camera.cv) / camera.fv
    # project_warp represents a point as camera_z * [dx, dy, 1].
    rotation_camera_to_world = camera.rotation_world_to_camera.T
    ray_x = rotation_camera_to_world[0, 0] * dx + rotation_camera_to_world[0, 1] * dy + rotation_camera_to_world[0, 2]
    ray_y = rotation_camera_to_world[1, 0] * dx + rotation_camera_to_world[1, 1] * dy + rotation_camera_to_world[1, 2]
    ray_z = rotation_camera_to_world[2, 0] * dx + rotation_camera_to_world[2, 1] * dy + rotation_camera_to_world[2, 2]
    a = ray_x * ray_x + ray_y * ray_y + ray_z * ray_z
    b = 2.0 * (camera.center[0] * ray_x + camera.center[1] * ray_y + camera.center[2] * ray_z)
    c = float(np.dot(camera.center, camera.center) - mars_radius_m * mars_radius_m)
    discriminant = b * b - 4.0 * a * c
    valid = discriminant >= 0.0
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
    z1 = (-b - sqrt_disc) / (2.0 * a)
    z2 = (-b + sqrt_disc) / (2.0 * a)
    depth = np.where((z1 > 0.0) & valid, z1, np.where((z2 > 0.0) & valid, z2, 0.0))
    depth = depth.astype(np.float32, copy=False)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def project_warp(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    camera_a: Camera,
    camera_b: Camera,
    *,
    absolute_depth_tolerance_m: float,
    relative_depth_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if depth_a.shape != depth_b.shape:
        raise ValueError(f"depth shape mismatch: {depth_a.shape} vs {depth_b.shape}")
    height, width = depth_a.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    z = depth_a.astype(np.float64, copy=False)
    valid_a = np.isfinite(z) & (z > 0.0)

    x_cam = (xx + 0.5 - camera_a.cu) / camera_a.fu * z
    y_cam = (yy + 0.5 - camera_a.cv) / camera_a.fv * z
    pts_cam = np.stack((x_cam, y_cam, z), axis=0).reshape(3, -1)
    world = camera_a.center[:, None] + camera_a.rotation_world_to_camera.T @ pts_cam
    projected_b = camera_b.rotation_world_to_camera @ (world - camera_b.center[:, None])
    pb_x = projected_b[0].reshape(height, width)
    pb_y = projected_b[1].reshape(height, width)
    pb_z = projected_b[2].reshape(height, width)

    u_b = camera_b.fu * (pb_x / pb_z) + camera_b.cu - 0.5
    v_b = camera_b.fv * (pb_y / pb_z) + camera_b.cv - 0.5
    inside_b = (pb_z > 0.0) & (u_b >= 0.0) & (u_b <= width - 1.0) & (v_b >= 0.0) & (v_b <= height - 1.0)

    sampled_depth_b = cv2.remap(
        depth_b.astype(np.float32, copy=False),
        u_b.astype(np.float32, copy=False),
        v_b.astype(np.float32, copy=False),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1.0,
    ).astype(np.float64, copy=False)
    tolerance = np.maximum(absolute_depth_tolerance_m, relative_depth_tolerance * np.abs(pb_z))
    depth_consistent = (
        np.isfinite(sampled_depth_b)
        & (sampled_depth_b > 0.0)
        & (np.abs(sampled_depth_b - pb_z) <= tolerance)
    )
    valid_mask_np = valid_a & inside_b & depth_consistent
    warp_np = np.stack((u_b, v_b), axis=-1).astype(np.float32, copy=False)
    warp_np[~np.isfinite(warp_np)] = 0.0
    metrics = {
        "height": float(height),
        "width": float(width),
        "valid_a_fraction": float(valid_a.mean()),
        "target_inside_fraction": float((valid_a & inside_b).sum() / max(1, int(valid_a.sum()))),
        "valid_pair_fraction": float(valid_mask_np.sum() / max(1, int(valid_a.sum()))),
        "valid_pixels": float(valid_mask_np.sum()),
    }
    return torch.from_numpy(warp_np.copy()).to(torch.float32), torch.from_numpy(valid_mask_np.copy()).to(torch.bool), metrics


def infer_base_sim_view(pair_path: Path) -> str:
    match = PAIR_VIEW_RE.match(pair_path.name)
    if match is None:
        raise ValueError(f"cannot infer sim view from {pair_path}")
    return "A_" + match.group(1)


def select_depth_pair(source_dir: Path) -> Path | None:
    preferred = sorted(source_dir.glob("pair_*_basic_xp5.pt"))
    if preferred:
        return preferred[0]
    candidates = sorted(source_dir.glob("pair_*.pt"))
    return candidates[0] if candidates else None


def discover_sources(dataset_root: Path, split: str) -> list[SourceRecord]:
    records_by_key: OrderedDict[tuple[str, int], SourceRecord] = OrderedDict()
    cache_split = dataset_root / "cache" / split
    for source_dir in sorted(cache_split.glob("source_*")):
        match = SOURCE_RE.match(source_dir.name)
        if match is None:
            match = SOURCE_REPART_RE.match(source_dir.name)
        if match is None:
            continue
        seq = int(match.group(1))
        track = match.group(2)
        key = (track, seq)
        if key in records_by_key:
            continue
        base_pair = select_depth_pair(source_dir)
        if base_pair is None:
            continue
        tsai_root = sorted((dataset_root / "tsai_tracks").glob(f"{track}_gap*_views10_*/tsai"))
        if not tsai_root:
            continue
        camera_a_path = tsai_root[0] / "CameraA" / f"{seq:05d}.tsai"
        if camera_a_path.exists():
            records_by_key[key] = (
                SourceRecord(
                    split=split,
                    track=track,
                    seq=seq,
                    source_dir=source_dir,
                    base_pair=base_pair,
                    camera_a_path=camera_a_path,
                )
            )
    return list(records_by_key.values())


class SourceDataCache:
    def __init__(
        self,
        max_items: int,
        *,
        depth_mode: str,
        mars_radius_m: float,
        output_root: Path,
        sat_sim: Path,
        dem: Path,
        dom: Path,
        sat_sim_jobs: int,
    ) -> None:
        self.max_items = max(1, max_items)
        self.depth_mode = depth_mode
        self.mars_radius_m = mars_radius_m
        self.output_root = output_root
        self.sat_sim = sat_sim
        self.dem = dem
        self.dom = dom
        self.sat_sim_jobs = sat_sim_jobs
        self.items: OrderedDict[Path, SourceData] = OrderedDict()

    def get(self, record: SourceRecord) -> SourceData:
        key = record.base_pair
        if key in self.items:
            value = self.items.pop(key)
            self.items[key] = value
            return value
        module = load_archive(record.base_pair)
        camera_a = parse_tsai(record.camera_a_path)
        view = normalize_view(module.view_a)
        if self.depth_mode == "rendered":
            _, height, width = view.shape
            depth_path = render_camera_depth(
                output_root=self.output_root,
                sat_sim=self.sat_sim,
                dem=self.dem,
                dom=self.dom,
                camera_path=record.camera_a_path,
                track=record.track,
                seq=record.seq,
                height=height,
                width=width,
                jobs=self.sat_sim_jobs,
            )
            depth = read_float_tif(depth_path)
            if depth.shape != (height, width):
                raise ValueError(f"rendered depth shape mismatch for {depth_path}: {depth.shape} vs {(height, width)}")
        elif self.depth_mode == "sphere":
            _, height, width = view.shape
            depth = sphere_depth_for_camera(camera_a, height, width, mars_radius_m=self.mars_radius_m)
        else:
            base_sim_view = infer_base_sim_view(record.base_pair)
            camera_b_path = record.camera_a_path.parent.parent / f"Camera{base_sim_view}" / record.camera_a_path.name
            camera_b = parse_tsai(camera_b_path)
            baseline = float(np.linalg.norm(camera_b.center - camera_a.center))
            if baseline < 1.0e-6:
                _, height, width = view.shape
                depth = sphere_depth_for_camera(camera_a, height, width, mars_radius_m=self.mars_radius_m)
            else:
                depth = recover_depth_from_warp(module, camera_a, camera_b)
        data = SourceData(
            view=view,
            depth=depth,
            camera_a=camera_a,
        )
        self.items[key] = data
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return data


def split_output_pair_path(
    output_root: Path,
    split: str,
    index: int,
    offset: int,
    source_a: SourceRecord,
    source_b: SourceRecord,
) -> Path:
    source_name = (
        f"source_cross_{index:06d}_{source_a.track}_off{offset:03d}"
        f"_s{source_a.seq:05d}_to_s{source_b.seq:05d}"
    )
    pair_name = f"pair_{index:06d}_cross_off{offset:03d}_s{source_a.seq:05d}_s{source_b.seq:05d}.pt"
    return output_root / "cache" / split / source_name / pair_name


def offset_bucket(offset: int) -> str:
    if offset <= 1:
        return "offset_001_neighbor"
    if offset <= 4:
        return "offset_002_004_short"
    if offset <= 8:
        return "offset_005_008_medium"
    return "offset_009_plus_extreme"


def candidate_pairs(records: list[SourceRecord], offsets: list[int]) -> list[CrossPair]:
    by_track: dict[str, list[SourceRecord]] = {}
    for record in records:
        by_track.setdefault(record.track, []).append(record)
    by_offset: dict[int, list[CrossPair]] = {offset: [] for offset in offsets if offset > 0}
    for track, items in sorted(by_track.items()):
        items = sorted(items, key=lambda item: item.seq)
        for offset in offsets:
            if offset <= 0:
                continue
            for index in range(0, max(0, len(items) - offset)):
                by_offset.setdefault(offset, []).append(
                    CrossPair(offset_bucket(offset), offset, items[index], items[index + offset])
                )
    pairs: list[CrossPair] = []
    max_len = max((len(items) for items in by_offset.values()), default=0)
    # Interleave offsets so a capped split receives a balanced mix instead of
    # being filled by the easiest near-neighbor offsets first.
    for pair_index in range(max_len):
        for offset in offsets:
            items = by_offset.get(offset, [])
            if pair_index < len(items):
                pairs.append(items[pair_index])
    return pairs


def reuse_key(source_a: SourceRecord, source_b: SourceRecord) -> tuple[str, int, str, int]:
    return (source_a.track, source_a.seq, source_b.track, source_b.seq)


def build_reuse_index(reuse_roots: list[Path]) -> dict[tuple[str, int, str, int], Path]:
    index: dict[tuple[str, int, str, int], Path] = {}
    for root in reuse_roots:
        root = root.resolve()
        manifest_paths = sorted((root / "manifests").glob("cross_position_*.csv"))
        if not manifest_paths:
            continue
        for manifest in manifest_paths:
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        key = (row["track_a"], int(row["seq_a"]), row["track_b"], int(row["seq_b"]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    pair_path = Path(row.get("pair_path", ""))
                    if not pair_path.is_absolute():
                        candidates = [pair_path, PROJECT_ROOT / pair_path, root / pair_path]
                        pair_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
                    if pair_path.exists():
                        index.setdefault(key, pair_path)
        for pair_path in sorted((root / "cache").glob("*/*/pair_*.pt")):
            match = SOURCE_CROSS_RE.match(pair_path.parent.name)
            if match is None:
                continue
            track = match.group(1)
            seq_a = int(match.group(2))
            seq_b = int(match.group(3))
            key = (track, seq_a, track, seq_b)
            index.setdefault(key, pair_path)
    return index


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def try_reuse_pair(reuse_path: Path, out_path: Path) -> dict[str, float] | None:
    if not reuse_path.exists():
        return None
    link_or_copy(reuse_path, out_path)
    metrics_path = reuse_path.with_suffix(".json")
    out_metrics_path = out_path.with_suffix(".json")
    metrics: dict[str, float] = {}
    if metrics_path.exists():
        link_or_copy(metrics_path, out_metrics_path)
        try:
            raw = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = {key: float(value) for key, value in raw.items() if isinstance(value, (int, float))}
        except (OSError, ValueError, TypeError):
            metrics = {}
    return metrics


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_index",
        "split",
        "bucket",
        "offset",
        "track_a",
        "seq_a",
        "track_b",
        "seq_b",
        "source_pair_a",
        "source_pair_b",
        "pair_path",
        "valid_pair_fraction",
        "valid_pixels",
        "target_inside_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_split(
    *,
    dataset_root: Path,
    output_root: Path,
    split: str,
    max_pairs: int,
    offsets: list[int],
    min_overlap: float,
    cache: SourceDataCache,
    absolute_depth_tolerance_m: float,
    relative_depth_tolerance: float,
    start_index: int,
    pre_render_workers: int,
    sat_sim: Path,
    dem: Path,
    dom: Path,
    sat_sim_jobs: int,
    reuse_index: dict[tuple[str, int, str, int], Path],
) -> tuple[int, list[dict[str, object]], dict[str, int]]:
    records = discover_sources(dataset_root, split)
    pairs = candidate_pairs(records, offsets)
    if cache.depth_mode == "rendered" and pre_render_workers > 0:
        selected_pairs = pairs if max_pairs <= 0 else pairs[:max_pairs]
        selected_records: list[SourceRecord] = []
        for pair in selected_pairs:
            if reuse_key(pair.source_a, pair.source_b) in reuse_index:
                continue
            source_a, source_b = pair.source_a, pair.source_b
            selected_records.extend((source_a, source_b))
        pre_render_depth_cache(
            records=selected_records,
            output_root=output_root,
            sat_sim=sat_sim,
            dem=dem,
            dom=dom,
            sat_sim_jobs=sat_sim_jobs,
            workers=pre_render_workers,
        )
    rows: list[dict[str, object]] = []
    stats = {"candidates": len(pairs), "kept": 0, "reused": 0, "low_overlap": 0, "errors": 0}
    pair_index = start_index
    for pair in pairs:
        if max_pairs > 0 and stats["kept"] >= max_pairs:
            break
        bucket, offset, source_a, source_b = pair.bucket, pair.offset, pair.source_a, pair.source_b
        out_path = split_output_pair_path(output_root, split, pair_index, offset, source_a, source_b)
        try:
            reused = False
            reuse_path = reuse_index.get(reuse_key(source_a, source_b))
            metrics = try_reuse_pair(reuse_path, out_path) if reuse_path is not None else None
            if metrics is not None:
                reused = True
                stats["reused"] += 1
            else:
                data_a = cache.get(source_a)
                data_b = cache.get(source_b)
                warp, valid_mask, metrics = project_warp(
                    data_a.depth,
                    data_b.depth,
                    data_a.camera_a,
                    data_b.camera_a,
                    absolute_depth_tolerance_m=absolute_depth_tolerance_m,
                    relative_depth_tolerance=relative_depth_tolerance,
                )
                if float(metrics["valid_pair_fraction"]) < min_overlap:
                    stats["low_overlap"] += 1
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                archive = PairArchive(data_a.view, data_b.view, warp, valid_mask)
                torch.jit.script(archive).save(str(out_path))
                out_path.with_suffix(".json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            rows.append(
                {
                    "pair_index": pair_index,
                    "split": split,
                    "bucket": bucket,
                    "offset": offset,
                    "track_a": source_a.track,
                    "seq_a": source_a.seq,
                    "track_b": source_b.track,
                    "seq_b": source_b.seq,
                    "source_pair_a": str(source_a.base_pair),
                    "source_pair_b": str(source_b.base_pair),
                    "pair_path": str(out_path),
                    "valid_pair_fraction": float(metrics["valid_pair_fraction"]),
                    "valid_pixels": int(metrics["valid_pixels"]),
                    "target_inside_fraction": float(metrics["target_inside_fraction"]),
                }
            )
            stats["kept"] += 1
            pair_index += 1
            if stats["kept"] == 1 or stats["kept"] % 25 == 0:
                print(
                    f"{split} kept={stats['kept']} idx={pair_index - 1} "
                    f"offset={offset} overlap={metrics.get('valid_pair_fraction', 0.0):.4f} "
                    f"{source_a.seq}->{source_b.seq} reused={int(reused)}",
                    flush=True,
                )
        except Exception as exc:
            stats["errors"] += 1
            print(f"skip_error split={split} {source_a.source_dir.name}->{source_b.source_dir.name}: {exc}", flush=True)
    return pair_index, rows, stats


def parse_offsets(value: str) -> list[int]:
    offsets = []
    for item in value.split(","):
        item = item.strip()
        if item:
            offsets.append(int(item))
    return offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cross-position pose-sim PFM pair caches.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-pairs", type=int, default=1200)
    parser.add_argument("--val-pairs", type=int, default=343)
    parser.add_argument("--test-pairs", type=int, default=171)
    parser.add_argument("--train-offsets", type=parse_offsets, default=parse_offsets("1,2,4,8,12,16"))
    parser.add_argument("--val-offsets", type=parse_offsets, default=parse_offsets("1,2,4,8,12,16"))
    parser.add_argument("--test-offsets", type=parse_offsets, default=parse_offsets("1,2,4,8,12,16"))
    parser.add_argument("--reuse-root", action="append", type=Path, default=[])
    parser.add_argument("--min-overlap", type=float, default=0.25)
    parser.add_argument("--cache-items", type=int, default=8)
    parser.add_argument("--depth-mode", choices=["rendered", "sphere", "warp-or-sphere"], default="rendered")
    parser.add_argument("--sat-sim", type=Path, default=DEFAULT_SAT_SIM)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--sat-sim-jobs", type=int, default=4)
    parser.add_argument("--depth-render-workers", type=int, default=1)
    parser.add_argument("--mars-radius-m", type=float, default=3396190.0)
    parser.add_argument("--absolute-depth-tolerance-m", type=float, default=50.0)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.002)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    args.sat_sim = args.sat_sim.resolve()
    args.dem = args.dem.resolve()
    args.dom = args.dom.resolve()
    args.reuse_root = [path.resolve() for path in args.reuse_root]
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.depth_mode == "rendered":
        for path_name, path in (("--sat-sim", args.sat_sim), ("--dem", args.dem), ("--dom", args.dom)):
            if not path.exists():
                raise FileNotFoundError(f"{path_name} path does not exist: {path}")
        if args.sat_sim_jobs <= 0:
            raise ValueError("--sat-sim-jobs must be positive")
        if args.depth_render_workers <= 0:
            raise ValueError("--depth-render-workers must be positive")
    cache = SourceDataCache(
        args.cache_items,
        depth_mode=args.depth_mode,
        mars_radius_m=args.mars_radius_m,
        output_root=args.output_root,
        sat_sim=args.sat_sim,
        dem=args.dem,
        dom=args.dom,
        sat_sim_jobs=args.sat_sim_jobs,
    )
    reuse_index = build_reuse_index(args.reuse_root)
    if reuse_index:
        print(f"reuse_pairs={len(reuse_index)}", flush=True)
    pair_index = 0
    all_rows: list[dict[str, object]] = []
    stats_by_split = {}
    for split, max_pairs, offsets in (
        ("train", args.train_pairs, args.train_offsets),
        ("val", args.val_pairs, args.val_offsets),
        ("test", args.test_pairs, args.test_offsets),
    ):
        pair_index, rows, stats = generate_split(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            split=split,
            max_pairs=max_pairs,
            offsets=offsets,
            min_overlap=args.min_overlap,
            cache=cache,
            absolute_depth_tolerance_m=args.absolute_depth_tolerance_m,
            relative_depth_tolerance=args.relative_depth_tolerance,
            start_index=pair_index,
            pre_render_workers=args.depth_render_workers,
            sat_sim=args.sat_sim,
            dem=args.dem,
            dom=args.dom,
            sat_sim_jobs=args.sat_sim_jobs,
            reuse_index=reuse_index,
        )
        write_manifest(args.output_root / "manifests" / f"cross_position_{split}.csv", rows)
        all_rows.extend(rows)
        stats_by_split[split] = stats
    write_manifest(args.output_root / "manifests" / "cross_position_all.csv", all_rows)
    metadata = {
        "source_dataset_root": str(args.dataset_root),
        "pair_type": "cross_camera_position",
        "min_overlap": args.min_overlap,
        "depth_mode": args.depth_mode,
        "sat_sim": str(args.sat_sim),
        "dem": str(args.dem),
        "dom": str(args.dom),
        "sat_sim_jobs": args.sat_sim_jobs,
        "depth_render_workers": args.depth_render_workers,
        "mars_radius_m": args.mars_radius_m,
        "train_offsets": args.train_offsets,
        "val_offsets": args.val_offsets,
        "test_offsets": args.test_offsets,
        "reuse_roots": [str(path) for path in args.reuse_root],
        "splits": stats_by_split,
    }
    (args.output_root / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
