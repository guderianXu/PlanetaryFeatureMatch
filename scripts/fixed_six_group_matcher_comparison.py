#!/usr/bin/env python3
"""Run matcher comparison on the fixed six-group/two-pair presentation sample."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

AGENT4_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent4.py"
AGENT11_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent11.py"
SPLIT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "cross_view_1024_keypointonly_multistate_stylespecific_guard_calib_0step_seed1234"
    / "splits"
    / "test"
)
IMG_ROOT = Path("/media/xjw/8T/深度学习数据/img")
PFM_ROUTE = PROJECT_ROOT / "runs" / "cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234"

METRIC_FIELDS = [
    "style",
    "gate",
    "sample",
    "pair_pt",
    "algorithm",
    "family",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "inlier_matches",
    "correct",
    "wrong",
    "precision",
    "mean_error_px",
    "median_error_px",
    "visualization",
    "message",
]

SUMMARY_FIELDS = [
    "style",
    "gate",
    "algorithm",
    "family",
    "pairs",
    "ok_pairs",
    "raw_matches",
    "inlier_matches",
    "correct",
    "wrong",
    "precision",
    "mean_inliers_per_pair",
    "mean_correct_per_pair",
]


@dataclass(frozen=True)
class FixedPair:
    style: str
    gate: str
    sample: str
    source: str
    pair: str
    rotation_deg: str = ""


@dataclass(frozen=True)
class MetricRow:
    style: str
    gate: str
    sample: str
    pair_pt: str
    algorithm: str
    family: str
    status: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    inlier_matches: int
    correct: int
    wrong: int
    precision: float
    mean_error_px: float
    median_error_px: float
    visualization: str = ""
    message: str = ""


@dataclass(frozen=True)
class PFMRouteParams:
    style: str
    gate: str
    texture_blend_weight: float
    keypoint_score_mode: str
    min_margin: float
    min_target_gradient: float
    min_target_local_contrast: float
    pytorch_state_label: str
    pytorch_state: Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A4 = load_module(AGENT4_SCRIPT, "fixed_compare_agent4")
A11 = load_module(AGENT11_SCRIPT, "fixed_compare_agent11")


def fixed_pairs() -> list[FixedPair]:
    numeric_default = [
        ("01", "source_000201_72", "pair_002049.pt"),
        ("02", "source_000007_105", "pair_000238.pt"),
    ]
    timestamp_default = [
        ("01", "source_000123_20260514T144405909_NAS_PAN_L2b", "pair_003819.pt"),
        ("02", "source_000088_20260514T070226673_NAS_PAN_L2b", "pair_004708.pt"),
    ]
    rows: list[FixedPair] = []
    rows.extend(
        [
            FixedPair("numeric", "rotate", "rot90", "source_000201_72", "pair_001587.pt", "90"),
            FixedPair("numeric", "rotate", "rot180", "source_000007_105", "pair_002779.pt", "180"),
            FixedPair(
                "timestamp",
                "rotate",
                "rot90",
                "source_000123_20260514T144405909_NAS_PAN_L2b",
                "pair_001509.pt",
                "90",
            ),
            FixedPair(
                "timestamp",
                "rotate",
                "rot180",
                "source_000088_20260514T070226673_NAS_PAN_L2b",
                "pair_002860.pt",
                "180",
            ),
        ]
    )
    for gate in ("rotate", "viewpoint", "compound"):
        if gate == "rotate":
            continue
        rows.extend(FixedPair("numeric", gate, sample, source, pair) for sample, source, pair in numeric_default)
        rows.extend(FixedPair("timestamp", gate, sample, source, pair) for sample, source, pair in timestamp_default)
    return rows


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_algorithms(args: argparse.Namespace):
    algorithms, skipped = A11.make_matchers(args)
    return [item for item in algorithms if item.family != "pfm"], skipped


def resolve_project_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def pair_path(pair: FixedPair, *, split_root: Path, img_root: Path) -> Path:
    split_candidate = split_root / pair.style / pair.gate / pair.source / pair.pair
    if split_candidate.exists():
        return split_candidate
    img_subdir = {"rotate": "Rotate_1024", "viewpoint": "Viewpoint_1024", "compound": "CompoundViewpoint_1024"}[pair.gate]
    img_candidate = img_root / img_subdir / pair.source / pair.pair
    if img_candidate.exists():
        return img_candidate
    return split_candidate


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", "nan", None):
        return default
    return float(value)


def direct_pfm_route_params(args: argparse.Namespace) -> dict[tuple[str, str], PFMRouteParams]:
    return {
        (style, gate): PFMRouteParams(
            style=style,
            gate=gate,
            texture_blend_weight=args.pfm_texture_blend_weight,
            keypoint_score_mode=args.pfm_keypoint_score_mode,
            min_margin=args.pfm_min_margin,
            min_target_gradient=args.pfm_min_target_gradient,
            min_target_local_contrast=args.pfm_min_target_local_contrast,
            pytorch_state_label="direct",
            pytorch_state=resolve_project_path(args.pfm_state),
        )
        for style in ("numeric", "timestamp")
        for gate in ("rotate", "viewpoint", "compound")
    }


def load_pfm_route_params(route: Path, args: argparse.Namespace) -> dict[tuple[str, str], PFMRouteParams]:
    path = route / "calibration" / "selected_weights.csv"
    if not path.exists():
        return direct_pfm_route_params(args)
    params: dict[tuple[str, str], PFMRouteParams] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = PFMRouteParams(
                style=row["style"],
                gate=row["gate"],
                texture_blend_weight=as_float(row, "texture_blend_weight"),
                keypoint_score_mode=row.get("keypoint_score_mode") or "texture",
                min_margin=as_float(row, "min_margin"),
                min_target_gradient=as_float(row, "min_target_gradient"),
                min_target_local_contrast=as_float(row, "min_target_local_contrast"),
                pytorch_state_label=row.get("pytorch_state_label") or "trained",
                pytorch_state=resolve_project_path(row["pytorch_state"]),
            )
            params[(item.style, item.gate)] = item
    return params


def format_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fields})


def pfm_summary_row(pair: FixedPair, args: argparse.Namespace) -> MetricRow:
    summary_path = PFM_ROUTE / "eval" / pair.style / pair.gate / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    path = pair_path(pair, split_root=args.split_root, img_root=args.img_root)
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["pair_pt"]: row for row in csv.DictReader(handle)}
    row = rows.get(rel) or rows.get(path.as_posix())
    if row is None:
        raise KeyError(f"{rel} not found in {summary_path}")
    vis_name = f"{pair.sample}_{pair.source}_{Path(pair.pair).stem}.png"
    source_vis = PFM_ROUTE / "visualizations" / pair.style / pair.gate / vis_name
    return MetricRow(
        style=pair.style,
        gate=pair.gate,
        sample=pair.sample,
        pair_pt=path.as_posix(),
        algorithm="PlanetaryFeatureMatch-current",
        family="pfm",
        status="ok",
        keypoints_a=0,
        keypoints_b=0,
        raw_matches=int(row["matches"]),
        inlier_matches=int(row["matches"]),
        correct=int(row["correct"]),
        wrong=int(row["wrong"]),
        precision=float(row["precision"]),
        mean_error_px=math.nan,
        median_error_px=math.nan,
        visualization=source_vis.as_posix() if source_vis.exists() else "",
    )


def correctness_mask(
    points_a: np.ndarray,
    points_b: np.ndarray,
    warp_a_to_b,
    valid_mask,
    *,
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    if points_a.size == 0:
        return np.zeros((0,), dtype=bool), np.empty((0,), dtype=np.float32)
    target_b = A4.sample_warp(warp_a_to_b, points_a)
    valid = A4.valid_at_points(valid_mask, points_a)
    errors = np.linalg.norm(target_b - points_b, axis=1)
    errors = np.where(np.isfinite(errors) & valid, errors, np.inf).astype(np.float32, copy=False)
    return errors <= threshold_px, errors


def metric_from_points(
    points_a: np.ndarray,
    points_b: np.ndarray,
    warp_a_to_b,
    valid_mask,
    *,
    threshold_px: float,
) -> tuple[int, int, int, float, float, float, np.ndarray]:
    correct_by_match, errors = correctness_mask(points_a, points_b, warp_a_to_b, valid_mask, threshold_px=threshold_px)
    matches = int(points_a.shape[0])
    correct = int(np.count_nonzero(correct_by_match))
    wrong = matches - correct
    finite = errors[np.isfinite(errors)]
    mean_error = float(finite.mean()) if finite.size else math.nan
    median_error = float(np.median(finite)) if finite.size else math.nan
    precision = correct / matches if matches else 0.0
    return matches, correct, wrong, precision, mean_error, median_error, correct_by_match


def draw_truth_colored_visualization(
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    correct_by_match: np.ndarray,
    path: Path,
) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.concatenate([cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR), cv2.cvtColor(image_b, cv2.COLOR_GRAY2BGR)], axis=1)
    offset = image_a.shape[1]
    correct_color = (40, 210, 40)
    wrong_color = (35, 35, 230)
    order = list(np.flatnonzero(correct_by_match)) + list(np.flatnonzero(~correct_by_match))
    for index in order:
        ax, ay = points_a[index]
        bx, by = points_b[index]
        color = correct_color if bool(correct_by_match[index]) else wrong_color
        a_xy = (int(round(ax)), int(round(ay)))
        b_xy = (int(round(bx + offset)), int(round(by)))
        cv2.line(canvas, a_xy, b_xy, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, a_xy, 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, b_xy, 2, color, -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def load_pfm_model(params: PFMRouteParams, model_cache: dict[Path, object], device: str):
    cached = model_cache.get(params.pytorch_state)
    if cached is not None:
        return cached
    import pfm_model

    model, _ = pfm_model.load_pytorch_state(params.pytorch_state, device=device)
    model.eval()
    model_cache[params.pytorch_state] = model
    return model


def pfm_match_points(args: argparse.Namespace, pair: FixedPair, params: PFMRouteParams, model_cache: dict[Path, object]) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import cross_view_experiment as exp

    model = load_pfm_model(params, model_cache, args.device)
    _, _, points_a, points_b = exp.evaluated_match_tensors(
        model,
        pair_path(pair, split_root=args.split_root, img_root=args.img_root),
        device=torch.device(args.device),
        mode="blend",
        texture_blend_weight=params.texture_blend_weight,
        max_keypoints=args.pfm_max_keypoints,
        min_intensity=args.pfm_min_intensity,
        texture_fraction=1.0,
        threshold_px=args.threshold_px,
        topk=args.pfm_descriptor_topk,
        max_matches=args.pfm_max_matches,
        min_score=args.pfm_min_score,
        min_margin=params.min_margin,
        min_target_gradient=params.min_target_gradient,
        min_target_local_contrast=params.min_target_local_contrast,
        mutual=True,
        geometry_filter=args.pfm_geometry_filter,
        keypoint_spatial_bins=args.pfm_keypoint_spatial_bins,
        keypoint_score_mode=params.keypoint_score_mode,
    )
    return (
        points_a.detach().cpu().numpy().astype(np.float32, copy=False),
        points_b.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def evaluate_pfm_pair(
    args: argparse.Namespace,
    pair: FixedPair,
    params: PFMRouteParams,
    model_cache: dict[Path, object],
    vis_dir: Path,
) -> MetricRow:
    try:
        path = pair_path(pair, split_root=args.split_root, img_root=args.img_root)
        image_a, image_b, warp_a_to_b, valid_mask = A4.load_pair(path)
        points_a, points_b = pfm_match_points(args, pair, params, model_cache)
        matches, correct, wrong, precision, mean_error, median_error, correct_by_match = metric_from_points(
            points_a,
            points_b,
            warp_a_to_b,
            valid_mask,
            threshold_px=args.threshold_px,
        )
        vis_path = vis_dir / "pfm" / pair.style / pair.gate / pair.sample / f"{pair.sample}_{pair.source}_{Path(pair.pair).stem}.png"
        draw_truth_colored_visualization(image_a, image_b, points_a, points_b, correct_by_match, vis_path)
        return MetricRow(
            style=pair.style,
            gate=pair.gate,
            sample=pair.sample,
            pair_pt=path.as_posix(),
            algorithm="PlanetaryFeatureMatch-current",
            family="pfm",
            status="ok",
            keypoints_a=0,
            keypoints_b=0,
            raw_matches=matches,
            inlier_matches=matches,
            correct=correct,
            wrong=wrong,
            precision=precision,
            mean_error_px=mean_error,
            median_error_px=median_error,
            visualization=vis_path.as_posix(),
            message=(
                f"state={params.pytorch_state_label}; score={params.keypoint_score_mode}; "
                f"blend={params.texture_blend_weight:g}; margin={params.min_margin:g}; "
                f"target_contrast={params.min_target_local_contrast:g}"
            ),
        )
    except Exception as exc:
        return MetricRow(
            style=pair.style,
            gate=pair.gate,
            sample=pair.sample,
            pair_pt=pair_path(pair, split_root=args.split_root, img_root=args.img_root).as_posix(),
            algorithm="PlanetaryFeatureMatch-current",
            family="pfm",
            status="error",
            keypoints_a=0,
            keypoints_b=0,
            raw_matches=0,
            inlier_matches=0,
            correct=0,
            wrong=0,
            precision=0.0,
            mean_error_px=math.nan,
            median_error_px=math.nan,
            message=f"{type(exc).__name__}: {exc}",
        )


def evaluate_pair_raw(args: argparse.Namespace, pair: FixedPair, algorithm, vis_dir: Path) -> MetricRow:
    try:
        path = pair_path(pair, split_root=args.split_root, img_root=args.img_root)
        image_a, image_b, warp_a_to_b, valid_mask = A4.load_pair(path)
        raw = algorithm.matcher.match(image_a, image_b)
        matches, correct, wrong, precision, mean_error, median_error, correct_by_match = metric_from_points(
            raw.points_a,
            raw.points_b,
            warp_a_to_b,
            valid_mask,
            threshold_px=args.threshold_px,
        )
        vis_path = vis_dir / pair.style / pair.gate / pair.sample / f"{A4.safe_name(algorithm.name)}.png"
        draw_truth_colored_visualization(image_a, image_b, raw.points_a, raw.points_b, correct_by_match, vis_path)
        return MetricRow(
            style=pair.style,
            gate=pair.gate,
            sample=pair.sample,
            pair_pt=path.as_posix(),
            algorithm=algorithm.name,
            family=algorithm.family,
            status="ok",
            keypoints_a=raw.keypoints_a,
            keypoints_b=raw.keypoints_b,
            raw_matches=raw.raw_matches,
            inlier_matches=matches,
            correct=correct,
            wrong=wrong,
            precision=precision,
            mean_error_px=mean_error,
            median_error_px=median_error,
            visualization=vis_path.as_posix() if vis_path.exists() else "",
            message="raw matcher output; no geometric postprocessing",
        )
    except Exception as exc:
        return MetricRow(
            style=pair.style,
            gate=pair.gate,
            sample=pair.sample,
            pair_pt=pair_path(pair, split_root=args.split_root, img_root=args.img_root).as_posix(),
            algorithm=algorithm.name,
            family=algorithm.family,
            status="error",
            keypoints_a=0,
            keypoints_b=0,
            raw_matches=0,
            inlier_matches=0,
            correct=0,
            wrong=0,
            precision=0.0,
            mean_error_px=math.nan,
            median_error_px=math.nan,
            message=f"{type(exc).__name__}: {exc}",
        )


def copy_pfm_visualizations(rows: list[MetricRow], output_dir: Path) -> list[MetricRow]:
    copied: list[MetricRow] = []
    for row in rows:
        source = Path(row.visualization)
        target = output_dir / "figures" / "pfm" / row.style / row.gate / row.sample / source.name
        visualization = ""
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            visualization = target.as_posix()
        copied.append(MetricRow(**{**asdict(row), "visualization": visualization}))
    return copied


def aggregate(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.algorithm), []).append(row)
    out: list[dict[str, object]] = []
    for (style, gate, algorithm), items in sorted(grouped.items()):
        ok = [row for row in items if row.status == "ok"]
        raw_matches = sum(row.raw_matches for row in ok)
        inlier_matches = sum(row.inlier_matches for row in ok)
        correct = sum(row.correct for row in ok)
        wrong = sum(row.wrong for row in ok)
        out.append(
            {
                "style": style,
                "gate": gate,
                "algorithm": algorithm,
                "family": items[0].family,
                "pairs": len(items),
                "ok_pairs": len(ok),
                "raw_matches": raw_matches,
                "inlier_matches": inlier_matches,
                "correct": correct,
                "wrong": wrong,
                "precision": 0.0 if inlier_matches == 0 else correct / inlier_matches,
                "mean_inliers_per_pair": float(np.mean([row.inlier_matches for row in ok])) if ok else math.nan,
                "mean_correct_per_pair": float(np.mean([row.correct for row in ok])) if ok else math.nan,
            }
        )
    return out


def write_markdown(output_dir: Path, rows: list[MetricRow], summary: list[dict[str, object]], skipped: list[dict[str, str]]) -> None:
    def display_sample(row: MetricRow) -> str:
        return row.sample if row.gate == "rotate" and row.sample.startswith("rot") else f"sample{row.sample}"

    lines = [
        "# 六组固定匹配对的算法匹配效果对比",
        "",
        "## 数据口径",
        "",
        "- 使用之前生成的 6 组固定样本：numeric/timestamp 两种影像风格 x rotate/viewpoint/compound 三个 gate，每组 2 个匹配对，共 12 个 `.pt` pair。",
        "- rotate 组的两个样本固定为 RotationOnly cache 的 90 度和 180 度 pair；`fixed_pairs.csv` 的 `rotation_deg` 列记录角度。",
        "- PFM 使用 `runs/cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234` 的当前 route 参数现场重算匹配点。",
        "- 所有算法都在同一批 pair 上直接匹配 `view_a` 与 `view_b`，没有重新随机采样，也没有额外旋转。",
        "- 外部算法展示原始 matcher 输出；未执行 RANSAC/Homography 几何筛选或修复。",
        "- PFM 展示当前 postselected route 参数下的匹配点，但关闭额外 local/affine geometry filter。",
        "- 可视化统一使用人工合成数据的 GT warp 判定：绿色为正确匹配，红色为错误匹配。",
        "- 正确匹配阈值为 5 px。",
        "",
        "## 原始匹配汇总表",
        "",
        "| style | gate | algorithm | pairs | matches | correct | wrong | precision |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: (str(item["style"]), str(item["gate"]), str(item["algorithm"]))):
        lines.append(
            f"| {row['style']} | {row['gate']} | {row['algorithm']} | {row['ok_pairs']} | "
            f"{row['inlier_matches']} | {row['correct']} | {row['wrong']} | {float(row['precision']):.6f} |"
        )
    lines.extend(["", "## 匹配图", ""])
    for row in rows:
        if not row.visualization:
            continue
        rel = Path(row.visualization).relative_to(output_dir).as_posix()
        lines.append(f"- {row.style}/{row.gate}/{display_sample(row)} `{row.algorithm}`: [{rel}]({rel})")
    lines.extend(["", "## 不可用项", ""])
    if skipped:
        for item in skipped:
            lines.append(f"- {item['algorithm']}: {item['reason']}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## 原始文件",
            "",
            "- `fixed_pairs.csv`: 固定 12 个 pair 的路径。",
            "- `metrics.csv`: 每个算法原始匹配的匹配数、正确数、precision 和可视化路径。",
            "- `summary.csv`: 按 style/gate/algorithm 聚合的原始匹配结果。",
            "- `figures/`: 原始匹配可视化图。",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "对比文档")
    parser.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    parser.add_argument("--img-root", type=Path, default=IMG_ROOT)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--learned-max-keypoints", type=int, default=512)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--pfm-route", type=Path, default=PFM_ROUTE)
    parser.add_argument("--pfm-texture-blend-weight", type=float, default=1.0)
    parser.add_argument("--pfm-keypoint-score-mode", choices=["texture", "learned"], default="texture")
    parser.add_argument("--pfm-max-keypoints", type=int, default=4096)
    parser.add_argument("--pfm-max-matches", type=int, default=512)
    parser.add_argument("--pfm-descriptor-topk", type=int, default=32)
    parser.add_argument("--pfm-keypoint-spatial-bins", type=int, default=0)
    parser.add_argument("--pfm-geometry-filter", choices=["none", "local", "affine"], default="none")
    parser.add_argument("--pfm-state", type=Path, default=A11.DEFAULT_PFM_STATE)
    parser.add_argument("--pfm-min-intensity", type=float, default=0.01)
    parser.add_argument("--pfm-min-score", type=float, default=-1.0)
    parser.add_argument("--pfm-min-margin", type=float, default=0.0)
    parser.add_argument("--pfm-min-target-gradient", type=float, default=0.0)
    parser.add_argument("--pfm-min-target-local-contrast", type=float, default=0.0)
    parser.add_argument("--no-lightglue", action="store_true")
    parser.add_argument("--limit-algorithms", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    raw_figures_dir = args.output_dir / "figures_raw"
    if raw_figures_dir.exists():
        shutil.rmtree(raw_figures_dir)
    pairs = fixed_pairs()
    pfm_params = load_pfm_route_params(args.pfm_route, args)
    pfm_model_cache: dict[Path, object] = {}
    algorithms, skipped = make_algorithms(args)
    rows: list[MetricRow] = []
    for pair in pairs:
        path = pair_path(pair, split_root=args.split_root, img_root=args.img_root)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"group={pair.style}/{pair.gate} sample={pair.sample} pair={path}", flush=True)
        pfm_row = evaluate_pfm_pair(args, pair, pfm_params[(pair.style, pair.gate)], pfm_model_cache, args.output_dir / "figures")
        rows.append(pfm_row)
        print(
            f"  {'PlanetaryFeatureMatch-current':22s} matches={pfm_row.inlier_matches:4d} "
            f"correct={pfm_row.correct:4d} precision={pfm_row.precision:.4f} status={pfm_row.status}",
            flush=True,
        )
        for algorithm in algorithms:
            row = evaluate_pair_raw(args, pair, algorithm, args.output_dir / "figures" / "other_models")
            rows.append(row)
            print(
                f"  {algorithm.name:22s} raw={row.correct:4d}/{row.inlier_matches:4d} "
                f"precision={row.precision:.4f} status={row.status}",
                flush=True,
            )
    summary = aggregate(rows)
    write_csv(
        args.output_dir / "fixed_pairs.csv",
        [
            asdict(pair)
            | {"pair_pt": pair_path(pair, split_root=args.split_root, img_root=args.img_root).as_posix()}
            for pair in pairs
        ],
        ["style", "gate", "sample", "source", "pair", "rotation_deg", "pair_pt"],
    )
    write_csv(args.output_dir / "metrics.csv", [asdict(row) for row in rows], METRIC_FIELDS)
    write_csv(args.output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    for stale in (args.output_dir / "raw_metrics.csv", args.output_dir / "raw_summary.csv"):
        if stale.exists():
            stale.unlink()
    write_csv(args.output_dir / "skipped_algorithms.csv", skipped, ["algorithm", "reason"])
    write_markdown(args.output_dir, rows, summary, skipped)
    print(f"output_dir={args.output_dir}")
    print(f"summary={args.output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
