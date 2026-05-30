#!/usr/bin/env python3
"""Agent11 matcher iteration: rotated cross-view comparison across matcher families."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

AGENT4_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent4.py"
AGENT8_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent8.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent11"
DEFAULT_PFM_RUN = PROJECT_ROOT / "runs" / "cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234"
DEFAULT_PFM_STATE = DEFAULT_PFM_RUN / "training" / "pytorch_pfm_state.pt"

METRIC_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "pair_pt",
    "source_name",
    "algorithm",
    "family",
    "status",
    "pass_fail",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "inlier_matches",
    "correct",
    "wrong",
    "precision",
    "pass_gate",
    "min_correct",
    "mean_error_px",
    "median_error_px",
    "ransac_threshold_px",
    "ratio",
    "visualization",
    "message",
]

SUMMARY_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "algorithm",
    "family",
    "pairs",
    "ok_pairs",
    "pass_gate_pairs",
    "raw_matches",
    "inlier_matches",
    "correct",
    "wrong",
    "precision",
    "mean_pair_precision",
    "mean_inliers_per_pair",
    "median_inliers_per_pair",
    "mean_correct_per_pair",
]

GLOBAL_FIELDS = [
    "algorithm",
    "family",
    "pairs",
    "ok_pairs",
    "pass_gate_pairs",
    "raw_matches",
    "inlier_matches",
    "correct",
    "wrong",
    "precision",
    "mean_pair_precision",
    "mean_inliers_per_pair",
]


@dataclass(frozen=True)
class RawOutput:
    points_a: np.ndarray
    points_b: np.ndarray
    keypoints_a: int
    keypoints_b: int
    raw_matches: int


@dataclass(frozen=True)
class Algorithm:
    name: str
    family: str
    matcher: object
    ransac_threshold_px: float
    ratio: float


@dataclass(frozen=True)
class MetricRow:
    style: str
    gate: str
    rotation_deg: int
    pair_pt: str
    source_name: str
    algorithm: str
    family: str
    status: str
    pass_fail: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    inlier_matches: int
    correct: int
    wrong: int
    precision: float
    pass_gate: int
    min_correct: int
    mean_error_px: float
    median_error_px: float
    ransac_threshold_px: float
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


A4 = load_module(AGENT4_SCRIPT, "agent4_matcher_for_agent11")
A8 = load_module(AGENT8_SCRIPT, "agent8_matcher_for_agent11")


def empty_points() -> np.ndarray:
    return A4.empty_points()


def min_gate_labels(gate: str) -> int:
    return 8 if gate == "compound" else 20


def source_name(pair_path: Path) -> str:
    return pair_path.parent.name


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


def make_algorithm_specs() -> list[str]:
    return [
        "SIFT-r0.80-Ht2",
        "RootSIFT-r0.80-Ht2",
        "RootSIFT-r0.90-Ht2",
        "ORB-cross-Ht3",
        "AKAZE-cross-Ht3",
        "LightGlue-SIFT-Ht3",
        "PFM-raw",
        "PFM-Ht3",
    ]


class CvMatcher:
    def __init__(self, detector_name: str, *, ratio: float, mode: str, max_keypoints: int, max_matches: int, sift_contrast: float) -> None:
        import cv2

        self.detector_name = detector_name
        self.ratio = ratio
        self.mode = mode
        self.max_matches = max_matches
        if detector_name in {"SIFT", "RootSIFT"}:
            self.detector = cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=sift_contrast)
            self.descriptor_kind = "float"
        elif detector_name == "ORB":
            self.detector = cv2.ORB_create(nfeatures=max_keypoints)
            self.descriptor_kind = "binary"
        elif detector_name == "AKAZE":
            self.detector = cv2.AKAZE_create()
            self.descriptor_kind = "binary"
        else:
            raise ValueError(f"unknown cv detector: {detector_name}")

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> RawOutput:
        import cv2

        keypoints_a, descriptors_a = self.detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self.detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
            return RawOutput(empty_points(), empty_points(), len(keypoints_a or []), len(keypoints_b or []), 0)
        if self.detector_name == "RootSIFT":
            descriptors_a = A4.rootsift(descriptors_a.astype(np.float32, copy=False))
            descriptors_b = A4.rootsift(descriptors_b.astype(np.float32, copy=False))
        if self.descriptor_kind == "float":
            norm = cv2.NORM_L2
            descriptors_a = descriptors_a.astype(np.float32, copy=False)
            descriptors_b = descriptors_b.astype(np.float32, copy=False)
        else:
            norm = cv2.NORM_HAMMING
        if self.mode == "cross":
            matches = cv2.BFMatcher(norm, crossCheck=True).match(descriptors_a, descriptors_b)
        elif self.mode == "ratio":
            matches = A4.ratio_filter(cv2.BFMatcher(norm).knnMatch(descriptors_a, descriptors_b, k=2), self.ratio)
        else:
            raise ValueError(f"unknown match mode: {self.mode}")
        matches = sorted(matches, key=lambda item: item.distance)[: self.max_matches]
        output = A4.output_from_matches(keypoints_a, keypoints_b, matches)
        return RawOutput(output.points_a, output.points_b, output.keypoints_a, output.keypoints_b, len(matches))


class LightGlueMatcher:
    def __init__(self, *, max_keypoints: int, device: str) -> None:
        self.impl = A4.LightGlueSiftMatcher(max_keypoints=max_keypoints, device=device)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> RawOutput:
        output = self.impl.match(image_a, image_b)
        return RawOutput(output.points_a, output.points_b, output.keypoints_a, output.keypoints_b, output.points_a.shape[0])


class PFMMatcher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.impl = A8.PFMScoredMatcher(
            state_path=args.pfm_state,
            device=choose_device(args.device),
            max_keypoints=args.pfm_max_keypoints,
            max_matches=args.max_matches,
            min_intensity=args.pfm_min_intensity,
            min_score=args.pfm_min_score,
        )
        self.min_margin = args.pfm_min_margin

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> RawOutput:
        output = self.impl.match(image_a, image_b, min_margin=self.min_margin)
        return RawOutput(
            output.points_a,
            output.points_b_rotated,
            output.keypoints_a,
            output.keypoints_b,
            output.points_a.shape[0],
        )


def cv2_status() -> tuple[object | None, list[dict[str, str]]]:
    unavailable: list[dict[str, str]] = []
    try:
        import cv2

        checks = {"SIFT": "SIFT_create", "ORB": "ORB_create", "AKAZE": "AKAZE_create"}
        for label, attr in checks.items():
            if not hasattr(cv2, attr):
                unavailable.append({"algorithm": label, "reason": f"cv2.{attr} unavailable"})
        return cv2, unavailable
    except Exception as exc:
        unavailable.append({"algorithm": "OpenCV algorithms", "reason": f"{type(exc).__name__}: {exc}"})
        return None, unavailable


def make_matchers(args: argparse.Namespace) -> tuple[list[Algorithm], list[dict[str, str]]]:
    algorithms: list[Algorithm] = []
    skipped: list[dict[str, str]] = []
    cv2, unavailable = cv2_status()
    skipped.extend(unavailable)
    if cv2 is not None and hasattr(cv2, "SIFT_create"):
        algorithms.extend(
            [
                Algorithm("SIFT-r0.80-Ht2", "classical", CvMatcher("SIFT", ratio=0.80, mode="ratio", max_keypoints=args.max_keypoints, max_matches=args.max_matches, sift_contrast=args.sift_contrast), 2.0, 0.80),
                Algorithm("RootSIFT-r0.80-Ht2", "classical", CvMatcher("RootSIFT", ratio=0.80, mode="ratio", max_keypoints=args.max_keypoints, max_matches=args.max_matches, sift_contrast=args.sift_contrast), 2.0, 0.80),
                Algorithm("RootSIFT-r0.90-Ht2", "classical", CvMatcher("RootSIFT", ratio=0.90, mode="ratio", max_keypoints=args.max_keypoints, max_matches=args.max_matches, sift_contrast=args.sift_contrast), 2.0, 0.90),
            ]
        )
    if cv2 is not None and hasattr(cv2, "ORB_create"):
        algorithms.append(Algorithm("ORB-cross-Ht3", "classical", CvMatcher("ORB", ratio=math.nan, mode="cross", max_keypoints=args.max_keypoints, max_matches=args.max_matches, sift_contrast=args.sift_contrast), 3.0, math.nan))
    if cv2 is not None and hasattr(cv2, "AKAZE_create"):
        algorithms.append(Algorithm("AKAZE-cross-Ht3", "classical", CvMatcher("AKAZE", ratio=math.nan, mode="cross", max_keypoints=args.max_keypoints, max_matches=args.max_matches, sift_contrast=args.sift_contrast), 3.0, math.nan))

    if importlib.util.find_spec("lightglue") is None:
        skipped.append({"algorithm": "LightGlue-SIFT-Ht3", "reason": "module 'lightglue' unavailable"})
    elif args.no_lightglue:
        skipped.append({"algorithm": "LightGlue-SIFT-Ht3", "reason": "disabled by --no-lightglue"})
    else:
        try:
            algorithms.append(Algorithm("LightGlue-SIFT-Ht3", "learned", LightGlueMatcher(max_keypoints=args.learned_max_keypoints, device=choose_device(args.device)), 3.0, math.nan))
        except Exception as exc:
            skipped.append({"algorithm": "LightGlue-SIFT-Ht3", "reason": f"{type(exc).__name__}: {exc}"})

    if importlib.util.find_spec("match_pairs") is None and importlib.util.find_spec("superglue") is None:
        skipped.append({"algorithm": "SuperGlue", "reason": "modules 'match_pairs' and 'superglue' unavailable"})
    try:
        pfm = PFMMatcher(args)
        algorithms.append(Algorithm("PFM-raw", "pfm", pfm, math.nan, math.nan))
        algorithms.append(Algorithm("PFM-Ht3", "pfm", pfm, 3.0, math.nan))
    except Exception as exc:
        skipped.append({"algorithm": "PFM", "reason": f"{type(exc).__name__}: {exc}"})
    if args.limit_algorithms:
        keep = set(args.limit_algorithms)
        algorithms = [item for item in algorithms if item.name in keep]
    return algorithms, skipped


def apply_ransac(raw: RawOutput, threshold_px: float, min_inliers: int) -> RawOutput:
    if math.isnan(threshold_px):
        return raw
    inlier_a, inlier_b = A4.ransac_inliers(raw.points_a, raw.points_b, threshold_px=threshold_px)
    if inlier_a.shape[0] < min_inliers:
        inlier_a, inlier_b = empty_points(), empty_points()
    return RawOutput(inlier_a, inlier_b, raw.keypoints_a, raw.keypoints_b, raw.raw_matches)


def metric_row(
    *,
    args: argparse.Namespace,
    algorithm: Algorithm,
    style: str,
    gate: str,
    rotation_deg: int,
    pair_path: Path,
    output: RawOutput,
    image_b_shape: tuple[int, int],
    warp_a_to_b,
    valid_mask,
    status: str = "ok",
    message: str = "",
    visualization: str = "",
) -> tuple[MetricRow, np.ndarray]:
    points_b_original = A4.unrotate_points(output.points_b, image_b_shape[0], image_b_shape[1], rotation_deg)
    matches, correct, wrong, precision, mean_error, median_error = A4.compute_metrics(
        output.points_a,
        points_b_original,
        warp_a_to_b,
        valid_mask,
        threshold_px=args.threshold_px,
    )
    min_correct = min_gate_labels(gate)
    pass_gate = 1 if correct >= min_correct else 0
    row = MetricRow(
        style=style,
        gate=gate,
        rotation_deg=rotation_deg,
        pair_pt=pair_path.as_posix(),
        source_name=source_name(pair_path),
        algorithm=algorithm.name,
        family=algorithm.family,
        status=status,
        pass_fail="pass" if pass_gate else "fail",
        keypoints_a=output.keypoints_a,
        keypoints_b=output.keypoints_b,
        raw_matches=output.raw_matches,
        inlier_matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
        pass_gate=pass_gate,
        min_correct=min_correct,
        mean_error_px=mean_error,
        median_error_px=median_error,
        ransac_threshold_px=algorithm.ransac_threshold_px,
        ratio=algorithm.ratio,
        visualization=visualization,
        message=message,
    )
    return row, points_b_original


def evaluate_one(
    args: argparse.Namespace,
    algorithm: Algorithm,
    *,
    style: str,
    gate: str,
    rotation_deg: int,
    pair_path: Path,
    vis_budget: dict[str, int],
) -> MetricRow:
    try:
        image_a, image_b, warp_a_to_b, valid_mask = A4.load_pair(pair_path)
        image_b_rotated = A4.rotate_image(image_b, rotation_deg)
        raw = algorithm.matcher.match(image_a, image_b_rotated)
        output = apply_ransac(raw, algorithm.ransac_threshold_px, min_gate_labels(gate))
        row, points_b_original = metric_row(
            args=args,
            algorithm=algorithm,
            style=style,
            gate=gate,
            rotation_deg=rotation_deg,
            pair_path=pair_path,
            output=output,
            image_b_shape=image_b.shape[:2],
            warp_a_to_b=warp_a_to_b,
            valid_mask=valid_mask,
        )
        key = f"{style}/{gate}/rot{rotation_deg}/{algorithm.name}"
        if args.visualizations_per_algorithm > 0 and row.inlier_matches > 0 and vis_budget.get(key, 0) < args.visualizations_per_algorithm:
            vis_budget[key] = vis_budget.get(key, 0) + 1
            vis_path = args.output_dir / "visualizations" / style / gate / f"rot{rotation_deg}" / f"{pair_path.parent.name}_{pair_path.stem}_{A4.safe_name(algorithm.name)}.png"
            A4.draw_visualization(image_a, image_b, output.points_a, points_b_original, vis_path)
            row = MetricRow(**{**asdict(row), "visualization": vis_path.as_posix()})
        return row
    except Exception as exc:
        return MetricRow(
            style=style,
            gate=gate,
            rotation_deg=rotation_deg,
            pair_pt=pair_path.as_posix(),
            source_name=source_name(pair_path),
            algorithm=algorithm.name,
            family=algorithm.family,
            status="error",
            pass_fail="fail",
            keypoints_a=0,
            keypoints_b=0,
            raw_matches=0,
            inlier_matches=0,
            correct=0,
            wrong=0,
            precision=0.0,
            pass_gate=0,
            min_correct=min_gate_labels(gate),
            mean_error_px=math.nan,
            median_error_px=math.nan,
            ransac_threshold_px=algorithm.ransac_threshold_px,
            ratio=algorithm.ratio,
            message=f"{type(exc).__name__}: {exc}",
        )


def evaluate(args: argparse.Namespace) -> tuple[list[MetricRow], list[dict[str, object]], list[dict[str, str]]]:
    algorithms, skipped = make_matchers(args)
    rows: list[MetricRow] = []
    sampled: list[dict[str, object]] = []
    vis_budget: dict[str, int] = {}
    if not algorithms:
        skipped.append({"algorithm": "all", "reason": "no runnable algorithms after dependency checks"})
        return rows, sampled, skipped
    for style in args.styles:
        for gate in args.gates:
            pair_paths = A4.select_pairs(args, style, gate)
            sampled.extend({"style": style, "gate": gate, "source_name": source_name(path), "pair_pt": path.as_posix()} for path in pair_paths)
            print(f"group={style}/{gate} pairs={len(pair_paths)} rotations={len(args.rotations)} algorithms={len(algorithms)}", flush=True)
            for rotation_deg in args.rotations:
                for pair_index, pair_path in enumerate(pair_paths, start=1):
                    for algorithm in algorithms:
                        row = evaluate_one(
                            args,
                            algorithm,
                            style=style,
                            gate=gate,
                            rotation_deg=rotation_deg,
                            pair_path=pair_path,
                            vis_budget=vis_budget,
                        )
                        rows.append(row)
                    print(f"{style:9s} {gate:9s} rot={rotation_deg:3d} {pair_index:02d}/{len(pair_paths):02d} done", flush=True)
    return rows, sampled, skipped


def summarize_group(items: list[MetricRow]) -> dict[str, object]:
    ok = [row for row in items if row.status == "ok"]
    raw_matches = sum(row.raw_matches for row in ok)
    inlier_matches = sum(row.inlier_matches for row in ok)
    correct = sum(row.correct for row in ok)
    wrong = sum(row.wrong for row in ok)
    return {
        "family": items[0].family,
        "pairs": len(items),
        "ok_pairs": len(ok),
        "pass_gate_pairs": sum(row.pass_gate for row in ok),
        "raw_matches": raw_matches,
        "inlier_matches": inlier_matches,
        "correct": correct,
        "wrong": wrong,
        "precision": 0.0 if inlier_matches == 0 else correct / inlier_matches,
        "mean_pair_precision": float(np.mean([row.precision for row in ok])) if ok else math.nan,
        "mean_inliers_per_pair": float(np.mean([row.inlier_matches for row in ok])) if ok else math.nan,
        "median_inliers_per_pair": float(np.median([row.inlier_matches for row in ok])) if ok else math.nan,
        "mean_correct_per_pair": float(np.mean([row.correct for row in ok])) if ok else math.nan,
    }


def aggregate(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.rotation_deg, row.algorithm), []).append(row)
    out: list[dict[str, object]] = []
    for (style, gate, rotation_deg, algorithm), items in sorted(grouped.items()):
        data = summarize_group(items)
        out.append({"style": style, "gate": gate, "rotation_deg": rotation_deg, "algorithm": algorithm, **data})
    return out


def aggregate_global(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[str, list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.algorithm, []).append(row)
    return [{"algorithm": algorithm, **summarize_group(items)} for algorithm, items in sorted(grouped.items())]


def markdown_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| algorithm | family | ok pairs | pass gate | inliers | correct | precision | mean inliers |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (-int(item["pass_gate_pairs"]), -int(item["correct"]), str(item["algorithm"]))):
        lines.append(
            f"| {row['algorithm']} | {row['family']} | {row['ok_pairs']} | {row['pass_gate_pairs']} | "
            f"{row['inlier_matches']} | {row['correct']} | {float(row['precision']):.4f} | "
            f"{float(row['mean_inliers_per_pair']):.2f} |"
        )
    return lines


def write_readme(
    args: argparse.Namespace,
    rows: list[MetricRow],
    summary_rows: list[dict[str, object]],
    global_rows: list[dict[str, object]],
    sampled: list[dict[str, object]],
    skipped: list[dict[str, str]],
) -> None:
    command = (
        "PYTHONPATH=python MKL_THREADING_LAYER=GNU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/{Path(__file__).name} "
        f"--pairs-per-group {args.pairs_per_group} --visualizations-per-algorithm {args.visualizations_per_algorithm}"
    )
    rotated_evals = len({(row.style, row.gate, row.rotation_deg, row.pair_pt) for row in rows})
    lines = [
        "# Matcher Algorithm Iteration Agent11",
        "",
        "## Scope",
        "",
        "- Goal: compare classical, learned, and current PFM matchers on numeric and timestamp/NAS cross-view pairs.",
        "- B image is rotated by 90/180/270 degrees before matching; matched B points are unrotated before geometric evaluation.",
        f"- Styles: `{','.join(args.styles)}`; gates: `{','.join(args.gates)}`; rotations: `{','.join(str(item) for item in args.rotations)}`.",
        f"- Pairs per style/gate: `{args.pairs_per_group}`; sampled source rows: `{len(sampled)}`; rotated pair evaluations: `{rotated_evals}`.",
        f"- Correct threshold: `{args.threshold_px}` px; pass gate: 20 correct matches except compound uses 8.",
        "",
        "## Command",
        "",
        f"```bash\n{command}\n```",
        "",
        "## Global Summary",
        "",
        *markdown_table(global_rows),
        "",
        "## Skipped / Unavailable",
        "",
    ]
    if skipped:
        for item in skipped:
            lines.append(f"- {item['algorithm']}: {item['reason']}")
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `metrics.csv`: per algorithm/style/gate/rotation/pair pass/fail and inlier/correct metrics.",
            "- `summary_metrics.csv`: grouped by style/gate/rotation/algorithm.",
            "- `global_summary.csv`: grouped by algorithm across the run.",
            "- `sampled_pairs.csv`: selected numeric and timestamp/NAS source pairs.",
            "- `skipped_algorithms.csv`: algorithms skipped because dependencies or initialization were unavailable.",
            "- `visualizations/`: bounded match visualizations.",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    names = make_algorithm_specs()
    assert "RootSIFT-r0.80-Ht2" in names
    assert "PFM-raw" in names
    assert min_gate_labels("compound") == 8
    assert min_gate_labels("viewpoint") == 20
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert A4.rotate_image(image, 90).shape == (4, 3)
    point = np.array([[0.0, 1.0]], dtype=np.float32)
    assert np.allclose(A4.unrotate_points(point, 3, 4, 90), [[1.0, 2.0]])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pfm-run", type=Path, default=DEFAULT_PFM_RUN)
    parser.add_argument("--pfm-state", type=Path, default=DEFAULT_PFM_STATE)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--styles", nargs="+", default=["numeric", "timestamp"], choices=["numeric", "timestamp"])
    parser.add_argument("--gates", nargs="+", default=["viewpoint", "compound"], choices=["viewpoint", "compound"])
    parser.add_argument("--rotations", nargs="+", type=int, default=[90, 180, 270], choices=[90, 180, 270])
    parser.add_argument("--pairs-per-group", type=int, default=2)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--learned-max-keypoints", type=int, default=512)
    parser.add_argument("--pfm-max-keypoints", type=int, default=512)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--pfm-min-intensity", type=float, default=0.03)
    parser.add_argument("--pfm-min-score", type=float, default=-1.0)
    parser.add_argument("--pfm-min-margin", type=float, default=0.0)
    parser.add_argument("--visualizations-per-algorithm", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-lightglue", action="store_true")
    parser.add_argument("--limit-algorithms", nargs="*")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, sampled, skipped = evaluate(args)
    summary_rows = aggregate(rows)
    global_rows = aggregate_global(rows)
    write_csv(args.output_dir / "metrics.csv", [asdict(row) for row in rows], METRIC_FIELDS)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.output_dir / "global_summary.csv", global_rows, GLOBAL_FIELDS)
    write_csv(args.output_dir / "sampled_pairs.csv", sampled, ["style", "gate", "source_name", "pair_pt"])
    write_csv(args.output_dir / "skipped_algorithms.csv", skipped, ["algorithm", "reason"])
    write_readme(args, rows, summary_rows, global_rows, sampled, skipped)
    print(f"output_dir={args.output_dir}")
    print(f"metrics={args.output_dir / 'metrics.csv'}")
    print(f"summary={args.output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
