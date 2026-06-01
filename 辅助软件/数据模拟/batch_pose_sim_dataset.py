#!/usr/bin/env python3
"""Stream satsim renders into PFM pose-simulation pair caches."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ASP_PYTHON = Path("/home/xjw/anaconda3/envs/asp36/bin/python")
PFM_TRAIN_PYTHON = Path("/home/xjw/anaconda3/envs/pfm-train/bin/python")
SAT_SIM = SCRIPT_DIR / "build" / "sat_sim_cuda"
MARS_TO_TSAI = SCRIPT_DIR / "mars_orbit_to_tsai.py"
PAIR_BUILDER = SCRIPT_DIR / "pose_sim_to_pfm_cache.py"
DEM = SCRIPT_DIR / "dem" / "mar.tif"
DOM = SCRIPT_DIR / "dom" / "HX1_GRAS_MoRIC_DOM_076m_Global_A.tif"
KERNEL_DIR = SCRIPT_DIR / "kernel"

VIEW_SPEC = (
    "base=0,0,0;"
    "basic_xp5=5,0,0;"
    "basic_xn5=-5,0,0;"
    "basic_yp5r8=0,5,8;"
    "mid_xp12=12,0,0;"
    "mid_yn12r15=0,-12,15;"
    "mid_diag=-10,10,-20;"
    "ext_xp22=22,0,0;"
    "ext_yn22r25=0,-22,25;"
    "ext_diag=-18,18,-35"
)

TRACKS = [
    ("Orbiter_InfoCSV_20251215_110km", "train"),
    ("Orbiter_InfoCSV_20251215_340km", "train"),
    ("Orbiter_InfoCSV_20251216_110km", "val"),
    ("Orbiter_InfoCSV_20251216_340km", "test"),
]

SIM_VIEWS = [
    "A_basic_xp5",
    "A_basic_xn5",
    "A_basic_yp5r8",
    "A_mid_xp12",
    "A_mid_yn12r15",
    "A_mid_diag",
    "A_ext_xp22",
    "A_ext_yn22r25",
    "A_ext_diag",
]


@dataclass(frozen=True)
class TrackInfo:
    stem: str
    split: str
    csv_path: Path
    tsai_dir: Path
    frames: int


@dataclass(frozen=True)
class Candidate:
    index: int
    track: TrackInfo
    seq: int
    sim_view: str
    split: str


def run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None, log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def asp_env() -> dict[str, str]:
    env = os.environ.copy()
    lib = "/home/xjw/anaconda3/envs/asp36/lib"
    env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def ensure_dem_vrt(dataset_root: Path) -> Path:
    vrt_path = dataset_root / "dem_lon0_shifted_for_sat_sim.vrt"
    if vrt_path.exists():
        return vrt_path
    info = json.loads(subprocess.check_output(["gdalinfo", "-json", str(DEM)], text=True))
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
      <SourceFilename relativeToVRT="0">{DEM}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="Int16" BlockXSize="128" BlockYSize="1"/>
      <SrcRect xOff="{half}" yOff="0" xSize="{width - half}" ySize="{height}"/>
      <DstRect xOff="0" yOff="0" xSize="{width - half}" ySize="{height}"/>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{DEM}</SourceFilename>
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


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def prepare_tsai(dataset_root: Path, *, render_gap: int, image_size: int) -> list[TrackInfo]:
    tsai_root = dataset_root / "tsai_tracks"
    logs_dir = dataset_root / "logs" / "prepare"
    tracks: list[TrackInfo] = []
    for stem, split in TRACKS:
        csv_path = SCRIPT_DIR / "相机轨迹" / f"{stem}.csv"
        rows = count_csv_rows(csv_path)
        frames = (rows + render_gap - 1) // render_gap
        out_dir = tsai_root / f"{stem}_gap{render_gap}_views10_{image_size}"
        done_marker = out_dir / ".prepared.json"
        if not done_marker.exists():
            command = [
                str(ASP_PYTHON),
                str(MARS_TO_TSAI),
                "--orbit-csv",
                str(csv_path),
                "--output-dir",
                str(out_dir),
                "--render-gap",
                str(render_gap),
                "--cameras",
                "A",
                "--camera-perturbations",
                VIEW_SPEC,
                "--focal-scales",
                "f1p00=1.0",
                "--image-size",
                str(image_size),
                "--jobs",
                "1",
                "--dem",
                str(DEM),
                "--dom",
                str(DOM),
                "--sat-sim",
                str(SAT_SIM),
                "--spice-kernel-dir",
                str(KERNEL_DIR),
            ]
            run_command(command, cwd=SCRIPT_DIR, env=asp_env(), log_path=logs_dir / f"{stem}.log")
            done_marker.write_text(
                json.dumps({"stem": stem, "split": split, "frames": frames, "render_gap": render_gap}, indent=2) + "\n",
                encoding="utf-8",
            )
        tracks.append(TrackInfo(stem=stem, split=split, csv_path=csv_path, tsai_dir=out_dir / "tsai", frames=frames))
    return tracks


def iter_candidates(tracks: list[TrackInfo]) -> list[Candidate]:
    candidates: list[Candidate] = []
    index = 0
    max_frames = max(track.frames for track in tracks)
    for seq in range(1, max_frames + 1):
        for track in tracks:
            if seq > track.frames:
                continue
            for view in SIM_VIEWS:
                candidates.append(Candidate(index=index, track=track, seq=seq, sim_view=view, split=track.split))
                index += 1
    return candidates


def parse_split_ratio(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(":")]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("split ratio must look like 7:2:1")
    return parts[0], parts[1], parts[2]


def split_counts(total: int, ratio: tuple[int, int, int]) -> dict[str, int]:
    ratio_sum = sum(ratio)
    train = int(total * ratio[0] / ratio_sum)
    val = int(total * ratio[1] / ratio_sum)
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def apply_ratio_split(candidates: list[Candidate], *, ratio: tuple[int, int, int], seed: int) -> list[Candidate]:
    import random

    counts = split_counts(len(candidates), ratio)
    split_by_index: dict[int, str] = {}
    shuffled = list(range(len(candidates)))
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    for position, candidate_index in enumerate(shuffled):
        if position < train_end:
            split = "train"
        elif position < val_end:
            split = "val"
        else:
            split = "test"
        split_by_index[candidate_index] = split
    return [replace(candidate, split=split_by_index[index]) for index, candidate in enumerate(candidates)]


def output_pair_path(dataset_root: Path, candidate: Candidate) -> Path:
    source = f"source_{candidate.seq:05d}_{candidate.track.stem}"
    view_suffix = candidate.sim_view.removeprefix("A_")
    return (
        dataset_root
        / "cache"
        / candidate.split
        / source
        / f"pair_{candidate.index:06d}_{view_suffix}.pt"
    )


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024.0**3)


def frame_candidates(candidates: list[Candidate]) -> dict[tuple[str, int], list[Candidate]]:
    grouped: dict[tuple[str, int], list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.track.stem, candidate.seq), []).append(candidate)
    return grouped


def render_frame_views(dataset_root: Path, candidate_group: list[Candidate], *, image_size: int, jobs: int, run_name: str) -> Path:
    first = candidate_group[0]
    temp_dir = dataset_root / "work" / run_name / f"{first.track.stem}_{first.seq:05d}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    links_dir = temp_dir / "tsai_links"
    render_dir = temp_dir / "render"
    links_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    view_ids = ["A"] + SIM_VIEWS
    camera_list = temp_dir / "camera_list.txt"
    lines: list[str] = []
    for view_id in view_ids:
        source = first.track.tsai_dir / f"Camera{view_id}" / f"{first.seq:05d}.tsai"
        if not source.exists():
            raise FileNotFoundError(source)
        link = links_dir / f"{view_id}_{first.seq:05d}.tsai"
        link.symlink_to(source)
        lines.append(str(link))
    camera_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dem_vrt = ensure_dem_vrt(dataset_root)
    command = [
        str(SAT_SIM),
        "--dem",
        str(dem_vrt),
        "--ortho",
        str(DOM),
        "--camera-list",
        str(camera_list),
        "--image-size",
        str(image_size),
        str(image_size),
        "--jobs",
        str(jobs),
        "--write-depth",
        "-o",
        str(render_dir / "batch"),
    ]
    log_path = dataset_root / "logs" / run_name / "render" / f"{first.track.stem}_{first.seq:05d}.log"
    run_command(command, cwd=SCRIPT_DIR, env=asp_env(), log_path=log_path)
    return render_dir


def convert_pair(dataset_root: Path, candidate: Candidate, render_dir: Path, *, run_name: str) -> dict[str, str | int | float]:
    seq = candidate.seq
    real_stem = f"A_{seq:05d}"
    sim_stem = f"{candidate.sim_view}_{seq:05d}"
    pair_path = output_pair_path(dataset_root, candidate)
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PFM_TRAIN_PYTHON),
        str(PAIR_BUILDER),
        "--image-a",
        str(render_dir / f"batch-{real_stem}.tif"),
        "--depth-a",
        str(render_dir / "depth" / f"batch-{real_stem}.tif"),
        "--tsai-a",
        str(candidate.track.tsai_dir / "CameraA" / f"{seq:05d}.tsai"),
        "--image-b",
        str(render_dir / f"batch-{sim_stem}.tif"),
        "--depth-b",
        str(render_dir / "depth" / f"batch-{sim_stem}.tif"),
        "--tsai-b",
        str(candidate.track.tsai_dir / f"Camera{candidate.sim_view}" / f"{seq:05d}.tsai"),
        "--output-pair",
        str(pair_path),
    ]
    log_path = dataset_root / "logs" / run_name / "convert" / f"pair_{candidate.index:06d}.log"
    run_command(command, cwd=PROJECT_ROOT, log_path=log_path)
    metrics = json.loads(pair_path.with_suffix(".json").read_text(encoding="utf-8"))
    return {
        "candidate_index": candidate.index,
        "split": candidate.split,
        "track": candidate.track.stem,
        "seq": candidate.seq,
        "sim_view": candidate.sim_view,
        "pair_path": str(pair_path),
        "valid_pair_fraction": float(metrics["valid_pair_fraction"]),
        "valid_pixels": int(metrics["valid_pixels"]),
    }


def append_manifest(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_index",
        "split",
        "track",
        "seq",
        "sim_view",
        "pair_path",
        "valid_pair_fraction",
        "valid_pixels",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def process_frame_group(
    *,
    dataset_root: Path,
    key: tuple[str, int],
    group: list[Candidate],
    image_size: int,
    sat_sim_jobs: int,
    run_name: str,
    keep_rendered: bool,
) -> dict[str, object]:
    missing = [candidate for candidate in group if not output_pair_path(dataset_root, candidate).exists()]
    if not missing:
        return {"key": key, "rows": [], "kept": 0, "skipped": len(group), "rendered": 0}
    render_dir = render_frame_views(dataset_root, missing, image_size=image_size, jobs=sat_sim_jobs, run_name=run_name)
    rows: list[dict[str, str | int | float]] = []
    skipped = 0
    try:
        for candidate in missing:
            pair_path = output_pair_path(dataset_root, candidate)
            if pair_path.exists():
                skipped += 1
                continue
            rows.append(convert_pair(dataset_root, candidate, render_dir, run_name=run_name))
    finally:
        if not keep_rendered:
            shutil.rmtree(render_dir.parent, ignore_errors=True)
    return {"key": key, "rows": rows, "kept": len(rows), "skipped": skipped, "rendered": 1}


def write_dataset_metadata(dataset_root: Path, args: argparse.Namespace, tracks: list[TrackInfo], total_candidates: int) -> None:
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "image_size": args.image_size,
        "render_gap": args.render_gap,
        "sat_sim_jobs": args.sat_sim_jobs,
        "view_spec": VIEW_SPEC,
        "sim_views": SIM_VIEWS,
        "tracks": [{"stem": t.stem, "split": t.split, "frames": t.frames} for t in tracks],
        "split_mode": args.split_mode,
        "split_ratio": list(args.split_ratio),
        "split_seed": args.split_seed,
        "total_candidates": total_candidates,
    }
    (dataset_root / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_generation(args: argparse.Namespace) -> None:
    dataset_root = args.output_root / args.dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    tracks = prepare_tsai(dataset_root, render_gap=args.render_gap, image_size=args.image_size)
    candidates = iter_candidates(tracks)
    if args.split_mode == "ratio":
        candidates = apply_ratio_split(candidates, ratio=args.split_ratio, seed=args.split_seed)
    write_dataset_metadata(dataset_root, args, tracks, len(candidates))
    if args.prepare_only:
        print(f"prepared dataset_root={dataset_root}")
        print(f"total_candidates={len(candidates)}")
        return

    start = args.start_pair_index
    end = len(candidates) if args.max_pairs <= 0 else min(len(candidates), start + args.max_pairs)
    selected = [candidate for candidate in candidates if start <= candidate.index < end]
    grouped = frame_candidates(selected)
    manifest_path = dataset_root / "manifests" / f"{args.run_name}.csv"
    state_path = dataset_root / "logs" / args.run_name / "state.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    rendered = 0
    ordered_keys = sorted(grouped.keys(), key=lambda item: (min(c.index for c in grouped[item]), item[0], item[1]))
    if args.frame_workers > 1:
        ensure_dem_vrt(dataset_root)
        next_progress = args.progress_interval if args.progress_interval > 0 else 0
        with ThreadPoolExecutor(max_workers=args.frame_workers) as executor:
            futures = {}
            for key in ordered_keys:
                if free_gb(dataset_root) < args.min_free_gb:
                    print(f"stop: free space below guard {args.min_free_gb:.1f} GB", flush=True)
                    break
                futures[
                    executor.submit(
                        process_frame_group,
                        dataset_root=dataset_root,
                        key=key,
                        group=grouped[key],
                        image_size=args.image_size,
                        sat_sim_jobs=args.sat_sim_jobs,
                        run_name=args.run_name,
                        keep_rendered=args.keep_rendered,
                    )
                ] = key
            for future in as_completed(futures):
                result = future.result()
                key = result["key"]
                rows = result["rows"]
                kept += int(result["kept"])
                skipped += int(result["skipped"])
                rendered += int(result["rendered"])
                append_manifest(manifest_path, rows)
                with state_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"track": key[0], "seq": key[1], "kept": kept, "skipped": skipped}) + "\n")
                if rows and args.progress_interval > 0:
                    last_candidate = max(int(row["candidate_index"]) for row in rows)
                    while next_progress and kept >= next_progress:
                        print(f"kept={kept} last_candidate={last_candidate} free_gb={free_gb(dataset_root):.1f}", flush=True)
                        next_progress += args.progress_interval
        print(f"done run={args.run_name} kept={kept} skipped={skipped} rendered_frames={rendered} manifest={manifest_path}")
        return

    for key in ordered_keys:
        if free_gb(dataset_root) < args.min_free_gb:
            print(f"stop: free space below guard {args.min_free_gb:.1f} GB", flush=True)
            break
        group = grouped[key]
        missing = [candidate for candidate in group if not output_pair_path(dataset_root, candidate).exists()]
        if not missing:
            skipped += len(group)
            continue
        render_dir = render_frame_views(dataset_root, missing, image_size=args.image_size, jobs=args.sat_sim_jobs, run_name=args.run_name)
        rendered += 1
        rows: list[dict[str, str | int | float]] = []
        try:
            for candidate in missing:
                pair_path = output_pair_path(dataset_root, candidate)
                if pair_path.exists():
                    skipped += 1
                    continue
                row = convert_pair(dataset_root, candidate, render_dir, run_name=args.run_name)
                rows.append(row)
                kept += 1
                if kept % args.progress_interval == 0:
                    print(f"kept={kept} last_candidate={candidate.index} free_gb={free_gb(dataset_root):.1f}", flush=True)
            append_manifest(manifest_path, rows)
            with state_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"track": key[0], "seq": key[1], "kept": kept, "skipped": skipped}) + "\n")
        finally:
            if not args.keep_rendered:
                shutil.rmtree(render_dir.parent, ignore_errors=True)
    print(f"done run={args.run_name} kept={kept} skipped={skipped} rendered_frames={rendered} manifest={manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate streaming 2048 pose-sim PFM caches.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="pose_sim_2048_gap30_views10")
    parser.add_argument("--render-gap", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=2048)
    parser.add_argument("--start-pair-index", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means run to the end of the candidate range.")
    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--sat-sim-jobs", type=int, default=4, help="Internal sat_sim_cuda rendering workers per frame batch.")
    parser.add_argument("--frame-workers", type=int, default=1, help="Independent frame batches to render/convert concurrently.")
    parser.add_argument("--split-mode", choices=("track", "ratio"), default="track")
    parser.add_argument("--split-ratio", type=parse_split_ratio, default=parse_split_ratio("7:2:1"))
    parser.add_argument("--split-seed", type=int, default=20260601)
    parser.add_argument("--min-free-gb", type=float, default=500.0)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-rendered", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.render_gap <= 0:
        raise ValueError("--render-gap must be positive")
    if args.image_size <= 1:
        raise ValueError("--image-size must be > 1")
    if args.start_pair_index < 0:
        raise ValueError("--start-pair-index must be >= 0")
    if args.sat_sim_jobs <= 0:
        raise ValueError("--sat-sim-jobs must be positive")
    if args.frame_workers <= 0:
        raise ValueError("--frame-workers must be positive")
    run_generation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
