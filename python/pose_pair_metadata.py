"""Training-only pose metadata loader for simulated PFM pair caches."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class CameraParameters:
    fu: float
    fv: float
    cu: float
    cv: float
    center: torch.Tensor
    rotation_world_to_camera: torch.Tensor
    path: Path


@dataclass(frozen=True)
class PosePairMetadata:
    pair_path: Path
    track: str
    seq: int
    sim_view: str
    camera_a: CameraParameters
    camera_b: CameraParameters
    baseline_m: float
    view_angle_deg: float
    focal_ratio: float
    overlap_fraction: float
    difficulty: str
    difficulty_score: float


PoseMetadataIndex = dict[str, PosePairMetadata]


def _parse_float_values(line: str) -> tuple[str, list[float]] | None:
    if "=" not in line:
        return None
    key, value = [part.strip() for part in line.split("=", 1)]
    try:
        return key, [float(part) for part in value.split()]
    except ValueError:
        return None


def parse_tsai(path: Path) -> CameraParameters:
    values: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_float_values(line)
        if parsed is None:
            continue
        key, floats = parsed
        values[key] = floats
    required = {"fu", "fv", "cu", "cv", "C", "R"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"{path} missing TSAI fields: {sorted(missing)}")
    center_values = values["C"]
    rotation_values = values["R"]
    if len(center_values) != 3 or len(rotation_values) != 9:
        raise ValueError(f"{path} has invalid C/R field length")
    # mars_orbit_to_tsai writes R in TSAI column-major order.
    rotation = torch.tensor(
        [
            [rotation_values[0], rotation_values[3], rotation_values[6]],
            [rotation_values[1], rotation_values[4], rotation_values[7]],
            [rotation_values[2], rotation_values[5], rotation_values[8]],
        ],
        dtype=torch.float64,
    )
    return CameraParameters(
        fu=float(values["fu"][0]),
        fv=float(values["fv"][0]),
        cu=float(values["cu"][0]),
        cv=float(values["cv"][0]),
        center=torch.tensor(center_values, dtype=torch.float64),
        rotation_world_to_camera=rotation,
        path=path,
    )


def camera_focal(camera: CameraParameters) -> float:
    return 0.5 * (float(camera.fu) + float(camera.fv))


def optical_axis_world(camera: CameraParameters) -> torch.Tensor:
    axis = camera.rotation_world_to_camera.T @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    norm = axis.norm().clamp_min(1.0e-12)
    return axis / norm


def view_angle_deg(camera_a: CameraParameters, camera_b: CameraParameters) -> float:
    axis_a = optical_axis_world(camera_a)
    axis_b = optical_axis_world(camera_b)
    cosine = float((axis_a * axis_b).sum().clamp(-1.0, 1.0))
    return math.degrees(math.acos(cosine))


def classify_difficulty(*, view_angle: float, focal_ratio: float, overlap_fraction: float) -> tuple[str, float]:
    scale_delta = abs(math.log(max(1.0e-6, float(focal_ratio))))
    if view_angle < 4.0 and scale_delta < 0.05 and overlap_fraction >= 0.85:
        return "easy", 0.0
    if view_angle < 14.0 and scale_delta < 0.25 and overlap_fraction >= 0.65:
        return "medium", 0.5
    return "hard", 1.0


def _track_tsai_root(dataset_root: Path, track: str, *, render_gap: int, image_size: int) -> Path | None:
    exact = dataset_root / "tsai_tracks" / f"{track}_gap{render_gap}_views10_{image_size}" / "tsai"
    if exact.exists():
        return exact
    matches = sorted((dataset_root / "tsai_tracks").glob(f"{track}_gap*_views10_*/tsai"))
    return matches[0] if matches else None


def _metadata_defaults(dataset_root: Path) -> tuple[int, int]:
    metadata_path = dataset_root / "dataset_metadata.json"
    if not metadata_path.exists():
        return 30, 2048
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata.get("render_gap", 30)), int(metadata.get("image_size", 2048))


def _index_keys(path: Path) -> list[str]:
    keys = [path.as_posix()]
    try:
        resolved = path.resolve(strict=False).as_posix()
        if resolved not in keys:
            keys.append(resolved)
    except OSError:
        pass
    parts = path.parts
    for index, part in enumerate(parts):
        if part == "cache" and index + 3 < len(parts):
            suffix = Path(*parts[index:]).as_posix()
            if suffix not in keys:
                keys.append(suffix)
    return keys


def load_pose_metadata_index(dataset_root: Path | str, *, strict: bool = False) -> PoseMetadataIndex:
    root = Path(dataset_root)
    render_gap, image_size = _metadata_defaults(root)
    index: PoseMetadataIndex = {}
    manifest_paths = sorted((root / "manifests").glob("*.csv"))
    for manifest_path in manifest_paths:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    track = str(row["track"])
                    seq = int(row["seq"])
                    sim_view = str(row["sim_view"])
                    pair_path = Path(row["pair_path"])
                    overlap = float(row.get("valid_pair_fraction") or 0.0)
                except (KeyError, TypeError, ValueError) as exc:
                    if strict:
                        raise ValueError(f"invalid manifest row in {manifest_path}: {row}") from exc
                    continue
                tsai_root = _track_tsai_root(root, track, render_gap=render_gap, image_size=image_size)
                if tsai_root is None:
                    if strict:
                        raise FileNotFoundError(f"missing tsai root for track {track}")
                    continue
                camera_a_path = tsai_root / "CameraA" / f"{seq:05d}.tsai"
                camera_b_path = tsai_root / f"Camera{sim_view}" / f"{seq:05d}.tsai"
                try:
                    camera_a = parse_tsai(camera_a_path)
                    camera_b = parse_tsai(camera_b_path)
                except (OSError, ValueError):
                    if strict:
                        raise
                    continue
                baseline = float((camera_b.center - camera_a.center).norm())
                focal_a = camera_focal(camera_a)
                focal_b = camera_focal(camera_b)
                focal_ratio = focal_b / max(1.0e-12, focal_a)
                angle = view_angle_deg(camera_a, camera_b)
                difficulty, difficulty_score = classify_difficulty(
                    view_angle=angle,
                    focal_ratio=focal_ratio,
                    overlap_fraction=overlap,
                )
                metadata = PosePairMetadata(
                    pair_path=pair_path,
                    track=track,
                    seq=seq,
                    sim_view=sim_view,
                    camera_a=camera_a,
                    camera_b=camera_b,
                    baseline_m=baseline,
                    view_angle_deg=angle,
                    focal_ratio=focal_ratio,
                    overlap_fraction=overlap,
                    difficulty=difficulty,
                    difficulty_score=difficulty_score,
                )
                for key in _index_keys(pair_path):
                    index[key] = metadata
    return index


def lookup_pose_metadata(index: PoseMetadataIndex | None, pair_path: Path | str) -> PosePairMetadata | None:
    if not index:
        return None
    path = Path(pair_path)
    for key in _index_keys(path):
        metadata = index.get(key)
        if metadata is not None:
            return metadata
    return None


def infer_pose_metadata_roots(cache_dirs: Iterable[Path | str], explicit_roots: Iterable[Path | str]) -> list[Path]:
    roots: list[Path] = []

    def add(root: Path) -> None:
        root = root.resolve(strict=False)
        if root not in roots:
            roots.append(root)

    for explicit in explicit_roots:
        add(Path(explicit))
    for cache_dir in cache_dirs:
        path = Path(cache_dir)
        if path.name in {"train", "val", "test"} and path.parent.name == "cache":
            add(path.parent.parent)
        elif path.name == "cache":
            add(path.parent)
        elif (path / "tsai_tracks").exists() and (path / "manifests").exists():
            add(path)
    return roots
