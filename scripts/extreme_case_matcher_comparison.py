#!/usr/bin/env python3
"""Run raw matcher comparison on the real extreme TIFF pair."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

AGENT11_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent11.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "对比文档" / "极端测试"
DEFAULT_IMAGE_A = DEFAULT_OUTPUT_DIR / "20260510T173954657_NAS_PAN_L2b.tif"
DEFAULT_IMAGE_B = DEFAULT_OUTPUT_DIR / "20260510T191252977_NAS_PAN_L2b.tif"
DEFAULT_LATEST_PFM_STATE = (
    PROJECT_ROOT
    / "runs"
    / "cross_view_1024_p1_viewpoint_keypointonly_w1n003_lr1e5_80_seed1234"
    / "training"
    / "pytorch_pfm_state.pt"
)

METRIC_FIELDS = [
    "case",
    "image_a",
    "image_b",
    "algorithm",
    "family",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "matches_drawn",
    "resize_max",
    "image_a_original_hw",
    "image_b_original_hw",
    "image_a_used_hw",
    "image_b_used_hw",
    "ratio",
    "visualization",
    "message",
]

SUMMARY_FIELDS = [
    "algorithm",
    "family",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "matches_drawn",
    "visualization",
    "message",
]


@dataclass(frozen=True)
class MetricRow:
    case: str
    image_a: str
    image_b: str
    algorithm: str
    family: str
    status: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    matches_drawn: int
    resize_max: int
    image_a_original_hw: str
    image_b_original_hw: str
    image_a_used_hw: str
    image_b_used_hw: str
    ratio: float
    visualization: str = ""
    message: str = ""


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A11 = load_module(AGENT11_SCRIPT, "extreme_compare_agent11")


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def hw_text(image: np.ndarray) -> str:
    return f"{int(image.shape[0])}x{int(image.shape[1])}"


def read_gray_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image
    array = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape[:2], dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        low = float(finite.min(initial=0.0))
        high = float(finite.max(initial=1.0))
    if high <= low:
        return np.zeros(array.shape[:2], dtype=np.uint8)
    normalized = (array - low) * (255.0 / (high - low))
    return np.clip(normalized, 0, 255).astype(np.uint8)


def resize_long_edge(image: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    if max_edge <= 0:
        return image, 1.0
    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image, 1.0
    scale = float(max_edge) / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA), scale


def draw_raw_visualization(
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    path: Path,
    *,
    max_lines: int,
) -> int:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    height = max(image_a.shape[0], image_b.shape[0])
    width = image_a.shape[1] + image_b.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: image_a.shape[0], : image_a.shape[1]] = cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR)
    canvas[: image_b.shape[0], image_a.shape[1] :] = cv2.cvtColor(image_b, cv2.COLOR_GRAY2BGR)
    offset = image_a.shape[1]
    count = min(int(points_a.shape[0]), int(max_lines))
    line_color = (235, 180, 40)
    point_color = (20, 220, 255)
    for index in range(count):
        ax, ay = points_a[index]
        bx, by = points_b[index]
        a_xy = (int(round(ax)), int(round(ay)))
        b_xy = (int(round(bx + offset)), int(round(by)))
        cv2.line(canvas, a_xy, b_xy, line_color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, a_xy, 2, point_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, b_xy, 2, point_color, -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return count


def parse_extra_pfm_state(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("extra PFM state must be label=path")
    label, path_text = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("extra PFM label is empty")
    path = Path(path_text.strip())
    return label, path if path.is_absolute() else PROJECT_ROOT / path


def make_algorithms(args: argparse.Namespace):
    algorithms = []
    skipped: list[dict[str, str]] = []
    cv2, unavailable = A11.cv2_status()
    skipped.extend(unavailable)
    if cv2 is not None and hasattr(cv2, "SIFT_create"):
        algorithms.extend(
            [
                A11.Algorithm(
                    "SIFT-r0.80-raw",
                    "classical",
                    A11.CvMatcher(
                        "SIFT",
                        ratio=0.80,
                        mode="ratio",
                        max_keypoints=args.max_keypoints,
                        max_matches=args.max_matches,
                        sift_contrast=args.sift_contrast,
                    ),
                    math.nan,
                    0.80,
                ),
                A11.Algorithm(
                    "RootSIFT-r0.80-raw",
                    "classical",
                    A11.CvMatcher(
                        "RootSIFT",
                        ratio=0.80,
                        mode="ratio",
                        max_keypoints=args.max_keypoints,
                        max_matches=args.max_matches,
                        sift_contrast=args.sift_contrast,
                    ),
                    math.nan,
                    0.80,
                ),
                A11.Algorithm(
                    "RootSIFT-r0.90-raw",
                    "classical",
                    A11.CvMatcher(
                        "RootSIFT",
                        ratio=0.90,
                        mode="ratio",
                        max_keypoints=args.max_keypoints,
                        max_matches=args.max_matches,
                        sift_contrast=args.sift_contrast,
                    ),
                    math.nan,
                    0.90,
                ),
            ]
        )
    if cv2 is not None and hasattr(cv2, "ORB_create"):
        algorithms.append(
            A11.Algorithm(
                "ORB-cross-raw",
                "classical",
                A11.CvMatcher(
                    "ORB",
                    ratio=math.nan,
                    mode="cross",
                    max_keypoints=args.max_keypoints,
                    max_matches=args.max_matches,
                    sift_contrast=args.sift_contrast,
                ),
                math.nan,
                math.nan,
            )
        )
    if cv2 is not None and hasattr(cv2, "AKAZE_create"):
        algorithms.append(
            A11.Algorithm(
                "AKAZE-cross-raw",
                "classical",
                A11.CvMatcher(
                    "AKAZE",
                    ratio=math.nan,
                    mode="cross",
                    max_keypoints=args.max_keypoints,
                    max_matches=args.max_matches,
                    sift_contrast=args.sift_contrast,
                ),
                math.nan,
                math.nan,
            )
        )
    if importlib.util.find_spec("lightglue") is None:
        skipped.append({"algorithm": "LightGlue-SIFT-raw", "reason": "module 'lightglue' unavailable"})
    elif args.no_lightglue:
        skipped.append({"algorithm": "LightGlue-SIFT-raw", "reason": "disabled by --no-lightglue"})
    else:
        try:
            algorithms.append(
                A11.Algorithm(
                    "LightGlue-SIFT-raw",
                    "learned",
                    A11.LightGlueMatcher(max_keypoints=args.learned_max_keypoints, device=args.device),
                    math.nan,
                    math.nan,
                )
            )
        except Exception as exc:
            skipped.append({"algorithm": "LightGlue-SIFT-raw", "reason": f"{type(exc).__name__}: {exc}"})
    if importlib.util.find_spec("match_pairs") is None and importlib.util.find_spec("superglue") is None:
        skipped.append({"algorithm": "SuperGlue", "reason": "modules 'match_pairs' and 'superglue' unavailable"})

    pfm_specs: list[tuple[str, Path]] = [("PFM-current-raw", args.pfm_state)]
    if not args.no_latest_pfm and DEFAULT_LATEST_PFM_STATE.exists():
        pfm_specs.append(("PFM-latest-p1-viewpoint-raw", DEFAULT_LATEST_PFM_STATE))
    pfm_specs.extend(args.extra_pfm_state or [])
    seen_pfm_paths: set[Path] = set()
    for label, state_path in pfm_specs:
        state_path = state_path if state_path.is_absolute() else PROJECT_ROOT / state_path
        if state_path in seen_pfm_paths:
            continue
        seen_pfm_paths.add(state_path)
        if not state_path.exists():
            skipped.append({"algorithm": label, "reason": f"state file not found: {state_path}"})
            continue
        try:
            pfm_args = copy.copy(args)
            pfm_args.pfm_state = state_path
            algorithms.append(A11.Algorithm(label, "pfm", A11.PFMMatcher(pfm_args), math.nan, math.nan))
        except Exception as exc:
            skipped.append({"algorithm": label, "reason": f"{type(exc).__name__}: {exc}"})

    if args.limit_algorithms:
        keep = set(args.limit_algorithms)
        algorithms = [item for item in algorithms if item.name in keep]
    return algorithms, skipped


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def evaluate(args: argparse.Namespace) -> tuple[list[MetricRow], list[dict[str, str]]]:
    original_a = read_gray_image(args.image_a)
    original_b = read_gray_image(args.image_b)
    image_a, scale_a = resize_long_edge(original_a, args.resize_max)
    image_b, scale_b = resize_long_edge(original_b, args.resize_max)
    algorithms, skipped = make_algorithms(args)
    rows: list[MetricRow] = []
    case_name = args.case_name
    for algorithm in algorithms:
        print(f"{algorithm.name:30s} matching...", flush=True)
        try:
            raw = algorithm.matcher.match(image_a, image_b)
            raw_matches = int(raw.points_a.shape[0])
            vis_path = ""
            drawn = 0
            if raw_matches > 0:
                target = args.output_dir / "figures" / f"{safe_name(algorithm.name)}.png"
                drawn = draw_raw_visualization(
                    image_a,
                    image_b,
                    raw.points_a,
                    raw.points_b,
                    target,
                    max_lines=args.max_drawn_matches,
                )
                vis_path = target.as_posix()
            row = MetricRow(
                case=case_name,
                image_a=args.image_a.as_posix(),
                image_b=args.image_b.as_posix(),
                algorithm=algorithm.name,
                family=algorithm.family,
                status="ok",
                keypoints_a=int(raw.keypoints_a),
                keypoints_b=int(raw.keypoints_b),
                raw_matches=raw_matches,
                matches_drawn=drawn,
                resize_max=args.resize_max,
                image_a_original_hw=hw_text(original_a),
                image_b_original_hw=hw_text(original_b),
                image_a_used_hw=f"{hw_text(image_a)} scale={scale_a:.6f}",
                image_b_used_hw=f"{hw_text(image_b)} scale={scale_b:.6f}",
                ratio=algorithm.ratio,
                visualization=vis_path,
                message="raw matcher output; no RANSAC/Homography filtering; no GT correctness for this real TIFF pair",
            )
            rows.append(row)
            print(f"  matches={raw_matches} keypoints=({raw.keypoints_a},{raw.keypoints_b}) status=ok", flush=True)
        except Exception as exc:
            rows.append(
                MetricRow(
                    case=case_name,
                    image_a=args.image_a.as_posix(),
                    image_b=args.image_b.as_posix(),
                    algorithm=algorithm.name,
                    family=algorithm.family,
                    status="error",
                    keypoints_a=0,
                    keypoints_b=0,
                    raw_matches=0,
                    matches_drawn=0,
                    resize_max=args.resize_max,
                    image_a_original_hw=hw_text(original_a),
                    image_b_original_hw=hw_text(original_b),
                    image_a_used_hw=f"{hw_text(image_a)} scale={scale_a:.6f}",
                    image_b_used_hw=f"{hw_text(image_b)} scale={scale_b:.6f}",
                    ratio=algorithm.ratio,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            print(f"  error: {type(exc).__name__}: {exc}", flush=True)
        clear_cuda_cache()
    return rows, skipped


def write_readme(args: argparse.Namespace, rows: list[MetricRow], skipped: list[dict[str, str]]) -> None:
    command = (
        "PYTHONPATH=python MKL_THREADING_LAYER=GNU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/{Path(__file__).name} "
        f"--device {args.device} --resize-max {args.resize_max}"
    )
    lines = [
        "# 极端测试匹配算法对比",
        "",
        "## 数据口径",
        "",
        f"- 图像 A：`{args.image_a.as_posix()}`。",
        f"- 图像 B：`{args.image_b.as_posix()}`。",
        "- 这是一对真实 TIFF 影像，不是 synthetic cache pair；当前目录没有对应的人工/合成 GT warp。",
        "- 因此本页不使用绿色/红色表示正确/错误，也不计算 precision/correct/wrong。",
        "- 所有算法均展示原始 matcher 输出，未执行 RANSAC、Homography、USAC 或其他几何筛选/修复。",
        f"- 为控制显存和运行时间，匹配前将长边缩放到 `{args.resize_max}`；CSV 中记录了原始尺寸和缩放后尺寸。",
        "- 可视化只画前若干条原始匹配线，颜色仅为中性显示，不代表对错。",
        "",
        "## 运行命令",
        "",
        f"```bash\n{command}\n```",
        "",
        "## 原始匹配数量",
        "",
        "| algorithm | family | status | keypoints A | keypoints B | raw matches | drawn | figure |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: (item.family, item.algorithm)):
        figure = ""
        if row.visualization:
            figure = Path(row.visualization).relative_to(args.output_dir).as_posix()
            figure = f"[{figure}]({figure})"
        lines.append(
            f"| {row.algorithm} | {row.family} | {row.status} | {row.keypoints_a} | {row.keypoints_b} | "
            f"{row.raw_matches} | {row.matches_drawn} | {figure} |"
        )
    lines.extend(["", "## 不可用项", ""])
    if skipped:
        for item in skipped:
            lines.append(f"- {item['algorithm']}: {item['reason']}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `metrics.csv`: 每个算法的原始匹配数量、关键点数量和可视化路径。",
            "- `summary.csv`: 与 `metrics.csv` 相同口径的简表。",
            "- `skipped_algorithms.csv`: 依赖缺失或初始化失败的算法。",
            "- `figures/`: 原始匹配线可视化。",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-a", type=Path, default=DEFAULT_IMAGE_A)
    parser.add_argument("--image-b", type=Path, default=DEFAULT_IMAGE_B)
    parser.add_argument("--case-name", default="extreme_tiff_pair_20260510")
    parser.add_argument("--resize-max", type=int, default=1600)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--learned-max-keypoints", type=int, default=1024)
    parser.add_argument("--pfm-max-keypoints", type=int, default=1024)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--max-drawn-matches", type=int, default=160)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--pfm-state", type=Path, default=A11.DEFAULT_PFM_STATE)
    parser.add_argument("--pfm-min-intensity", type=float, default=0.03)
    parser.add_argument("--pfm-min-score", type=float, default=-1.0)
    parser.add_argument("--pfm-min-margin", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-lightglue", action="store_true")
    parser.add_argument("--no-latest-pfm", action="store_true")
    parser.add_argument("--extra-pfm-state", action="append", type=parse_extra_pfm_state)
    parser.add_argument("--limit-algorithms", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    rows, skipped = evaluate(args)
    write_csv(args.output_dir / "metrics.csv", [asdict(row) for row in rows], METRIC_FIELDS)
    write_csv(args.output_dir / "summary.csv", [asdict(row) for row in rows], SUMMARY_FIELDS)
    write_csv(args.output_dir / "skipped_algorithms.csv", skipped, ["algorithm", "reason"])
    write_readme(args, rows, skipped)
    print(f"output_dir={args.output_dir}")
    print(f"summary={args.output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
