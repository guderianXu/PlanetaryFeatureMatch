#!/usr/bin/env python3
"""Agent8 matcher iteration: calibrate PFM fallback geometry filters."""

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
AGENT7_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent7"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent8"
DEFAULT_PFM_RUN = PROJECT_ROOT / "runs" / "cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234"
DEFAULT_PFM_STATE = DEFAULT_PFM_RUN / "training" / "pytorch_pfm_state.pt"

METRIC_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "pair_pt",
    "config",
    "match_filter",
    "min_margin",
    "ransac_threshold_px",
    "min_inliers",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "matches",
    "correct",
    "wrong",
    "precision",
    "coverage",
    "pass_gate",
    "mean_error_px",
    "median_error_px",
    "mean_score",
    "min_score",
    "max_score",
    "visualization",
    "message",
]

SUMMARY_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "config",
    "match_filter",
    "min_margin",
    "ransac_threshold_px",
    "min_inliers",
    "pairs",
    "ok_pairs",
    "covered_pairs",
    "pass_gate_pairs",
    "matches",
    "correct",
    "wrong",
    "precision",
    "mean_pair_precision",
    "mean_matches_per_pair",
    "median_matches_per_pair",
    "mean_score",
]

HARD_FIELDS = [
    "case_type",
    "style",
    "gate",
    "rotation_deg",
    "pair_pt",
    "config",
    "agent7_rootsift_matches",
    "agent7_rootsift_correct",
    "agent7_rootsift_precision",
    "agent7_pfm_matches",
    "agent7_pfm_correct",
    "agent7_pfm_precision",
    "filtered_matches",
    "filtered_correct",
    "filtered_wrong",
    "filtered_precision",
    "filtered_pass_gate",
    "mean_error_px",
    "median_error_px",
    "visualization",
]


@dataclass(frozen=True)
class ScoredOutput:
    points_a: np.ndarray
    points_b_rotated: np.ndarray
    scores: np.ndarray
    keypoints_a: int
    keypoints_b: int


@dataclass(frozen=True)
class MetricRow:
    style: str
    gate: str
    rotation_deg: int
    pair_pt: str
    config: str
    match_filter: str
    min_margin: float
    ransac_threshold_px: float
    min_inliers: int
    status: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    matches: int
    correct: int
    wrong: int
    precision: float
    coverage: int
    pass_gate: int
    mean_error_px: float
    median_error_px: float
    mean_score: float
    min_score: float
    max_score: float
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


A4 = load_module(AGENT4_SCRIPT, "agent4_matcher_for_agent8")


def min_gate_labels(gate: str) -> int:
    return 8 if gate == "compound" else 20


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


class PFMScoredMatcher:
    def __init__(
        self,
        *,
        state_path: Path,
        device: str,
        max_keypoints: int,
        max_matches: int,
        min_intensity: float,
        min_score: float,
    ) -> None:
        self.name = "PFM"
        self._state_path = state_path
        self._device = device
        self._max_keypoints = max_keypoints
        self._max_matches = max_matches
        self._min_intensity = min_intensity
        self._min_score = min_score
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        import pfm_model

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for PFM but torch.cuda.is_available() is false")
        self._model, _ = pfm_model.load_pytorch_state(self._state_path, device=self._device)
        self._model.eval()
        return self._model

    def match(self, image_a: np.ndarray, image_b: np.ndarray, *, min_margin: float) -> ScoredOutput:
        import torch
        import pytorch_cache_match_eval as eval_py
        from rotation_matcher_benchmark import image_to_tensor

        model = self._load()
        tensor_a = image_to_tensor(image_a, device=self._device)
        tensor_b = image_to_tensor(image_b, device=self._device)
        with torch.no_grad():
            descriptors_a = model.descriptor_map_single(tensor_a)
            descriptors_b = model.descriptor_map_single(tensor_b)
            keypoints_a, selected_a = eval_py.select_descriptor_keypoints(
                tensor_a.squeeze(0),
                descriptors_a,
                max_keypoints=self._max_keypoints,
                min_intensity=self._min_intensity,
            )
            keypoints_b, selected_b = eval_py.select_descriptor_keypoints(
                tensor_b.squeeze(0),
                descriptors_b,
                max_keypoints=self._max_keypoints,
                min_intensity=self._min_intensity,
            )
            rows_a = eval_py.gather_descriptor_rows(descriptors_a, selected_a)
            rows_b = eval_py.gather_descriptor_rows(descriptors_b, selected_b)
            matches, scores = eval_py.mutual_nearest_matches(
                rows_a,
                rows_b,
                max_matches=self._max_matches,
                min_score=self._min_score,
                min_margin=min_margin,
            )
            if matches.numel() == 0:
                return ScoredOutput(A4.empty_points(), A4.empty_points(), np.empty((0,), dtype=np.float32), int(keypoints_a.size(0)), int(keypoints_b.size(0)))
            points_a = eval_py._feature_to_image_points(
                keypoints_a.index_select(0, matches[:, 0].to(keypoints_a.device)),
                feature_height=descriptors_a.size(2),
                feature_width=descriptors_a.size(3),
                image_height=image_a.shape[0],
                image_width=image_a.shape[1],
            )
            points_b = eval_py._feature_to_image_points(
                keypoints_b.index_select(0, matches[:, 1].to(keypoints_b.device)),
                feature_height=descriptors_b.size(2),
                feature_width=descriptors_b.size(3),
                image_height=image_b.shape[0],
                image_width=image_b.shape[1],
            )
        return ScoredOutput(
            points_a.detach().cpu().numpy().astype(np.float32, copy=False),
            points_b.detach().cpu().numpy().astype(np.float32, copy=False),
            scores.detach().cpu().numpy().astype(np.float32, copy=False),
            int(keypoints_a.size(0)),
            int(keypoints_b.size(0)),
        )


def ransac_keep_mask(points_a: np.ndarray, points_b: np.ndarray, *, threshold_px: float) -> np.ndarray:
    if points_a.shape[0] < 4:
        return np.zeros((points_a.shape[0],), dtype=bool)
    import cv2

    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    _, mask = cv2.findHomography(
        points_a,
        points_b,
        method=method,
        ransacReprojThreshold=threshold_px,
        maxIters=3000,
        confidence=0.995,
    )
    if mask is None:
        return np.zeros((points_a.shape[0],), dtype=bool)
    return mask.reshape(-1).astype(bool)


def filtered_output(raw: ScoredOutput, *, threshold_px: float | None, min_inliers: int) -> ScoredOutput:
    if threshold_px is None:
        return raw
    keep = ransac_keep_mask(raw.points_a, raw.points_b_rotated, threshold_px=threshold_px)
    if int(np.count_nonzero(keep)) < min_inliers:
        keep = np.zeros_like(keep)
    return ScoredOutput(
        raw.points_a[keep],
        raw.points_b_rotated[keep],
        raw.scores[keep],
        raw.keypoints_a,
        raw.keypoints_b,
    )


def score_stats(scores: np.ndarray) -> tuple[float, float, float]:
    if scores.size == 0:
        return math.nan, math.nan, math.nan
    return float(scores.mean()), float(scores.min()), float(scores.max())


def metric_from_output(
    *,
    args: argparse.Namespace,
    style: str,
    gate: str,
    rotation_deg: int,
    pair_path: Path,
    config: str,
    match_filter: str,
    min_margin: float,
    ransac_threshold_px: float,
    min_inliers: int,
    raw_matches: int,
    output: ScoredOutput,
    warp_a_to_b,
    valid_mask,
    original_b_shape: tuple[int, int],
    visualization: str = "",
    status: str = "ok",
    message: str = "",
) -> MetricRow:
    points_b_original = A4.unrotate_points(output.points_b_rotated, original_b_shape[0], original_b_shape[1], rotation_deg)
    matches, correct, wrong, precision, mean_error, median_error = A4.compute_metrics(
        output.points_a,
        points_b_original,
        warp_a_to_b,
        valid_mask,
        threshold_px=args.threshold_px,
    )
    mean_score, min_score, max_score = score_stats(output.scores)
    return MetricRow(
        style=style,
        gate=gate,
        rotation_deg=rotation_deg,
        pair_pt=pair_path.as_posix(),
        config=config,
        match_filter=match_filter,
        min_margin=min_margin,
        ransac_threshold_px=ransac_threshold_px,
        min_inliers=min_inliers,
        status=status,
        keypoints_a=output.keypoints_a,
        keypoints_b=output.keypoints_b,
        raw_matches=raw_matches,
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
        coverage=1 if matches > 0 else 0,
        pass_gate=1 if correct >= min_gate_labels(gate) else 0,
        mean_error_px=mean_error,
        median_error_px=median_error,
        mean_score=mean_score,
        min_score=min_score,
        max_score=max_score,
        visualization=visualization,
        message=message,
    )


def config_label(*, margin: float, threshold: float | None, min_inliers: int) -> str:
    margin_text = f"m{margin:.3f}".replace(".", "p")
    if threshold is None:
        return f"PFM-mutual-{margin_text}-raw"
    threshold_text = f"t{threshold:g}".replace(".", "p")
    return f"PFM-mutual-{margin_text}-H{threshold_text}-min{min_inliers}"


def load_agent7_hard_cases(path: Path) -> dict[tuple[str, str, int, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        out = {}
        for row in reader:
            key = (row["style"], row["gate"], int(row["rotation_deg"]), row["pair_pt"])
            out[key] = row
        return out


def maybe_visualize(
    *,
    args: argparse.Namespace,
    row: MetricRow,
    image_a: np.ndarray,
    image_b: np.ndarray,
    output: ScoredOutput,
    used: dict[str, int],
    category: str,
) -> str:
    if output.points_a.size == 0:
        return ""
    key = f"{category}/{row.config}"
    if used.get(key, 0) >= args.visualizations_per_category:
        return ""
    used[key] = used.get(key, 0) + 1
    points_b_original = A4.unrotate_points(output.points_b_rotated, image_b.shape[0], image_b.shape[1], row.rotation_deg)
    pair_path = Path(row.pair_pt)
    path = (
        args.output_dir
        / "visualizations"
        / category
        / row.style
        / row.gate
        / f"rot{row.rotation_deg}_{pair_path.parent.name}_{pair_path.stem}_{A4.safe_name(row.config)}.png"
    )
    A4.draw_visualization(image_a, image_b, output.points_a, points_b_original, path)
    return path.as_posix()


def evaluate(args: argparse.Namespace) -> tuple[list[MetricRow], list[dict[str, object]], list[dict[str, object]]]:
    device = choose_device(args.device)
    pfm = PFMScoredMatcher(
        state_path=args.pfm_state,
        device=device,
        max_keypoints=args.pfm_max_keypoints,
        max_matches=args.max_matches,
        min_intensity=args.pfm_min_intensity,
        min_score=args.pfm_min_score,
    )
    hard_cases = load_agent7_hard_cases(args.agent7_hard_cases)
    rows: list[MetricRow] = []
    hard_report: list[dict[str, object]] = []
    sampled: list[dict[str, object]] = []
    vis_used: dict[str, int] = {}
    filter_specs: list[tuple[float | None, int]] = [(None, 0)]
    filter_specs.extend((threshold, min_inliers) for threshold in args.ransac_thresholds for min_inliers in args.min_inliers_grid)

    for style in args.styles:
        for gate in args.gates:
            pair_paths = A4.select_pairs(args, style, gate)
            sampled.extend({"style": style, "gate": gate, "pair_pt": path.as_posix()} for path in pair_paths)
            print(f"group={style}/{gate} pairs={len(pair_paths)}", flush=True)
            for pair_index, pair_path in enumerate(pair_paths, start=1):
                image_a, image_b, warp_a_to_b, valid_mask = A4.load_pair(pair_path)
                for rotation_deg in args.rotations:
                    image_b_rotated = A4.rotate_image(image_b, rotation_deg)
                    for margin in args.min_margins:
                        try:
                            raw = pfm.match(image_a, image_b_rotated, min_margin=margin)
                            raw_matches = int(raw.points_a.shape[0])
                            for threshold, min_inliers in filter_specs:
                                output = filtered_output(raw, threshold_px=threshold, min_inliers=min_inliers)
                                label = config_label(margin=margin, threshold=threshold, min_inliers=min_inliers)
                                match_filter = "raw" if threshold is None else "homography_ransac"
                                row = metric_from_output(
                                    args=args,
                                    style=style,
                                    gate=gate,
                                    rotation_deg=rotation_deg,
                                    pair_path=pair_path,
                                    config=label,
                                    match_filter=match_filter,
                                    min_margin=margin,
                                    ransac_threshold_px=math.nan if threshold is None else threshold,
                                    min_inliers=min_inliers,
                                    raw_matches=raw_matches,
                                    output=output,
                                    warp_a_to_b=warp_a_to_b,
                                    valid_mask=valid_mask,
                                    original_b_shape=image_b.shape[:2],
                                )
                                category = "pass_gate" if row.pass_gate else ("high_precision" if row.matches and row.precision >= 0.8 else "")
                                visualization = maybe_visualize(args=args, row=row, image_a=image_a, image_b=image_b, output=output, used=vis_used, category=category) if category else ""
                                if visualization:
                                    row = MetricRow(**{**asdict(row), "visualization": visualization})
                                rows.append(row)
                                hard_key = (style, gate, rotation_deg, pair_path.as_posix())
                                if hard_key in hard_cases:
                                    case = hard_cases[hard_key]
                                    hard_report.append(
                                        {
                                            "case_type": case["case_type"],
                                            "style": style,
                                            "gate": gate,
                                            "rotation_deg": rotation_deg,
                                            "pair_pt": pair_path.as_posix(),
                                            "config": label,
                                            "agent7_rootsift_matches": case["rootsift_matches"],
                                            "agent7_rootsift_correct": case["rootsift_correct"],
                                            "agent7_rootsift_precision": case["rootsift_precision"],
                                            "agent7_pfm_matches": case["pfm_matches"],
                                            "agent7_pfm_correct": case["pfm_correct"],
                                            "agent7_pfm_precision": case["pfm_precision"],
                                            "filtered_matches": row.matches,
                                            "filtered_correct": row.correct,
                                            "filtered_wrong": row.wrong,
                                            "filtered_precision": row.precision,
                                            "filtered_pass_gate": row.pass_gate,
                                            "mean_error_px": row.mean_error_px,
                                            "median_error_px": row.median_error_px,
                                            "visualization": row.visualization,
                                        }
                                    )
                        except Exception as exc:
                            label = config_label(margin=margin, threshold=None, min_inliers=0)
                            rows.append(
                                MetricRow(style, gate, rotation_deg, pair_path.as_posix(), label, "raw", margin, math.nan, 0, "error", 0, 0, 0, 0, 0, 0, 0.0, 0, 0, math.nan, math.nan, math.nan, math.nan, math.nan, message=f"{type(exc).__name__}: {exc}")
                            )
                    print(
                        f"{style:9s} {gate:9s} {pair_index:02d}/{len(pair_paths):02d} rot={rotation_deg:3d} done",
                        flush=True,
                    )
    return rows, hard_report, sampled


def aggregate(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.rotation_deg, row.config), []).append(row)
    out: list[dict[str, object]] = []
    for (style, gate, rotation_deg, config), items in sorted(grouped.items()):
        matches = sum(row.matches for row in items)
        correct = sum(row.correct for row in items)
        wrong = sum(row.wrong for row in items)
        ok = [row for row in items if row.status == "ok"]
        out.append(
            {
                "style": style,
                "gate": gate,
                "rotation_deg": rotation_deg,
                "config": config,
                "match_filter": items[0].match_filter,
                "min_margin": items[0].min_margin,
                "ransac_threshold_px": items[0].ransac_threshold_px,
                "min_inliers": items[0].min_inliers,
                "pairs": len(items),
                "ok_pairs": len(ok),
                "covered_pairs": sum(row.coverage for row in items),
                "pass_gate_pairs": sum(row.pass_gate for row in items),
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": 0.0 if matches == 0 else correct / matches,
                "mean_pair_precision": float(np.mean([row.precision for row in ok])) if ok else math.nan,
                "mean_matches_per_pair": float(np.mean([row.matches for row in ok])) if ok else math.nan,
                "median_matches_per_pair": float(np.median([row.matches for row in ok])) if ok else math.nan,
                "mean_score": float(np.mean([row.mean_score for row in ok if math.isfinite(row.mean_score)])) if any(math.isfinite(row.mean_score) for row in ok) else math.nan,
            }
        )
    return out


def best_rows(summary_rows: list[dict[str, object]], *, minimum_precision: float) -> list[dict[str, object]]:
    eligible = [row for row in summary_rows if int(row["matches"]) > 0 and float(row["precision"]) >= minimum_precision]
    return sorted(eligible, key=lambda row: (int(row["correct"]), int(row["matches"]), int(row["pass_gate_pairs"])), reverse=True)


def summarize_all(summary_rows: list[dict[str, object]]) -> dict[str, float]:
    matches = sum(int(row["matches"]) for row in summary_rows)
    correct = sum(int(row["correct"]) for row in summary_rows)
    return {
        "matches": matches,
        "correct": correct,
        "precision": 0.0 if matches == 0 else correct / matches,
        "pass_gate_pairs": sum(int(row["pass_gate_pairs"]) for row in summary_rows),
        "pairs": sum(int(row["pairs"]) for row in summary_rows),
    }


def global_config_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        item = grouped.setdefault(
            str(row["config"]),
            {
                "config": row["config"],
                "match_filter": row["match_filter"],
                "pairs": 0,
                "covered_pairs": 0,
                "pass_gate_pairs": 0,
                "matches": 0,
                "correct": 0,
                "wrong": 0,
            },
        )
        item["pairs"] = int(item["pairs"]) + int(row["pairs"])
        item["covered_pairs"] = int(item["covered_pairs"]) + int(row["covered_pairs"])
        item["pass_gate_pairs"] = int(item["pass_gate_pairs"]) + int(row["pass_gate_pairs"])
        item["matches"] = int(item["matches"]) + int(row["matches"])
        item["correct"] = int(item["correct"]) + int(row["correct"])
        item["wrong"] = int(item["wrong"]) + int(row["wrong"])
    out = []
    for item in grouped.values():
        matches = int(item["matches"])
        item["precision"] = 0.0 if matches == 0 else int(item["correct"]) / matches
        out.append(item)
    return sorted(out, key=lambda row: (float(row["precision"]), int(row["correct"]), int(row["matches"])), reverse=True)


def write_summary(args: argparse.Namespace, summary_rows: list[dict[str, object]], hard_report: list[dict[str, object]]) -> None:
    raw_rows = [row for row in summary_rows if row["match_filter"] == "raw" and float(row["min_margin"]) == 0.0]
    grid_rows = [row for row in summary_rows if row["match_filter"] == "homography_ransac"]
    best95 = best_rows(grid_rows, minimum_precision=0.95)[:10]
    best80 = best_rows(grid_rows, minimum_precision=0.80)[:10]
    all_raw = summarize_all(raw_rows)
    all_grid = summarize_all(grid_rows)
    best_slice = sorted(grid_rows, key=lambda row: (float(row["precision"]), int(row["correct"]), int(row["matches"])), reverse=True)[:1]
    global_rows = global_config_rows(summary_rows)
    global_min10 = [row for row in global_rows if int(row["matches"]) >= 10]

    hard_by_case: dict[str, dict[str, int]] = {}
    for row in hard_report:
        case_type = str(row["case_type"])
        stats = hard_by_case.setdefault(case_type, {"rows": 0, "matches": 0, "correct": 0, "pass_gate": 0})
        stats["rows"] += 1
        stats["matches"] += int(row["filtered_matches"])
        stats["correct"] += int(row["filtered_correct"])
        stats["pass_gate"] += int(row["filtered_pass_gate"])

    lines = [
        "# Matcher Algorithm Iteration Agent8",
        "",
        "## Scope",
        "",
        "- Experiment: PFM fallback confidence/geometric filtering on the same 1024-cache rotated-pair protocol used by agent7.",
        "- RootSIFT routing is not repeated; this run calibrates PFM as a possible fallback/pseudo-label source.",
        "- PFM matching uses existing mutual nearest descriptor matching. Scores are returned by the lower-level matcher, but the agent7 `MatchOutput` did not expose scores or margins; descriptor margin is therefore evaluated by calling the lower-level `min_margin` interface directly.",
        f"- Homography RANSAC thresholds: `{','.join(str(item) for item in args.ransac_thresholds)}` px; min inliers: `{','.join(str(item) for item in args.min_inliers_grid)}`.",
        f"- pairs per style/gate: `{args.pairs_per_group}`; rotations: `{','.join(str(item) for item in args.rotations)}`; extreme tests excluded.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=python MKL_THREADING_LAYER=GNU "
        f"/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/{Path(__file__).name} "
        f"--pairs-per-group {args.pairs_per_group} --device {args.device}",
        "```",
        "",
        "## Aggregate",
        "",
        f"- Raw PFM margin=0: matches={int(all_raw['matches'])}, correct={int(all_raw['correct'])}, precision={all_raw['precision']:.4f}, pass_gate_pairs={int(all_raw['pass_gate_pairs'])}/{int(all_raw['pairs'])}.",
        f"- All RANSAC grid rows combined: matches={int(all_grid['matches'])}, correct={int(all_grid['correct'])}, precision={all_grid['precision']:.4f}.",
    ]
    if best_slice:
        row = best_slice[0]
        lines.append(
            f"- Best slice-level config row: `{row['style']}/{row['gate']}/{row['rotation_deg']} {row['config']}` at precision={float(row['precision']):.4f}, matches={row['matches']}, correct={row['correct']}, pass_gate_pairs={row['pass_gate_pairs']}/{row['pairs']}."
        )
    if global_min10:
        row = global_min10[0]
        lines.append(
            f"- Best global config with at least 10 matches across all rotated pairs: `{row['config']}`, precision={float(row['precision']):.4f}, matches={row['matches']}, correct={row['correct']}, pass_gate_pairs={row['pass_gate_pairs']}/{row['pairs']}."
        )
    if not [row for row in global_rows if int(row["matches"]) > 0 and float(row["precision"]) >= 0.80]:
        lines.append("- No global config reached precision >=0.80.")
    lines.extend(["", "## Precision >= 0.95 Config Rows", "", "| style | gate | rot | config | pairs | pass gate | matches | correct | precision |", "|---|---|---:|---|---:|---:|---:|---:|---:|"])
    if best95:
        for row in best95:
            lines.append(f"| {row['style']} | {row['gate']} | {row['rotation_deg']} | {row['config']} | {row['pairs']} | {row['pass_gate_pairs']} | {row['matches']} | {row['correct']} | {float(row['precision']):.4f} |")
    else:
        lines.append("| | | | none | 0 | 0 | 0 | 0 | 0.0000 |")
    lines.extend(["", "## Precision >= 0.80 Config Rows", "", "| style | gate | rot | config | pairs | pass gate | matches | correct | precision |", "|---|---|---:|---|---:|---:|---:|---:|---:|"])
    if best80:
        for row in best80:
            lines.append(f"| {row['style']} | {row['gate']} | {row['rotation_deg']} | {row['config']} | {row['pairs']} | {row['pass_gate_pairs']} | {row['matches']} | {row['correct']} | {float(row['precision']):.4f} |")
    else:
        lines.append("| | | | none | 0 | 0 | 0 | 0 | 0.0000 |")
    lines.extend(["", "## Agent7 Hard Cases", ""])
    if hard_by_case:
        for case_type, stats in sorted(hard_by_case.items()):
            precision = 0.0 if stats["matches"] == 0 else stats["correct"] / stats["matches"]
            lines.append(f"- {case_type}: report rows={stats['rows']}, filtered matches={stats['matches']}, correct={stats['correct']}, precision={precision:.4f}, pass_gate rows={stats['pass_gate']}.")
    else:
        lines.append("- No agent7 hard case report rows were produced.")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
        ]
    )
    if best95:
        lines.append("- A high-precision PFM region exists for some style/gate/rotation slices, but use the retained matches and hard-case report before enabling it globally.")
    elif best80:
        lines.append("- PFM RANSAC can reach >=0.80 precision in limited slices, but there is no >=0.95 pseudo-label setting in this run.")
    else:
        lines.append("- Do not add PFM fallback pseudo-labels from this checkpoint/config: homography filtering did not expose a useful high-precision region on this protocol.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `metrics.csv`",
            "- `summary_metrics.csv`",
            "- `hard_case_filter_report.csv`",
            "- `sampled_pairs.csv`",
            "- `visualizations/` when any filtered rows pass gates or high precision.",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pfm-run", type=Path, default=DEFAULT_PFM_RUN)
    parser.add_argument("--pfm-state", type=Path, default=DEFAULT_PFM_STATE)
    parser.add_argument("--agent7-hard-cases", type=Path, default=AGENT7_DIR / "hard_cases.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--styles", nargs="+", default=["numeric", "timestamp"], choices=["numeric", "timestamp"])
    parser.add_argument("--gates", nargs="+", default=["viewpoint", "compound"], choices=["viewpoint", "compound"])
    parser.add_argument("--rotations", nargs="+", type=int, default=[90, 180, 270], choices=[0, 90, 180, 270])
    parser.add_argument("--pairs-per-group", type=int, default=4)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--pfm-max-keypoints", type=int, default=2048)
    parser.add_argument("--pfm-min-intensity", type=float, default=0.01)
    parser.add_argument("--pfm-min-score", type=float, default=-1.0)
    parser.add_argument("--min-margins", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.10])
    parser.add_argument("--ransac-thresholds", nargs="+", type=float, default=[2.0, 3.0, 4.0])
    parser.add_argument("--min-inliers-grid", nargs="+", type=int, default=[4, 8, 12, 20])
    parser.add_argument("--visualizations-per-category", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if not args.pfm_state.exists():
        raise FileNotFoundError(f"PFM state not found: {args.pfm_state}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, hard_report, sampled = evaluate(args)
    write_csv(args.output_dir / "metrics.csv", [asdict(row) for row in rows], METRIC_FIELDS)
    summary_rows = aggregate(rows)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.output_dir / "hard_case_filter_report.csv", hard_report, HARD_FIELDS)
    write_csv(args.output_dir / "sampled_pairs.csv", sampled, ["style", "gate", "pair_pt"])
    write_summary(args, summary_rows, hard_report)
    print(f"output_dir={args.output_dir}")
    print(f"metrics={args.output_dir / 'metrics.csv'}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
