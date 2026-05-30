#!/usr/bin/env python3
"""Agent13 stage6 route-gate quality diagnostic.

This stage deliberately avoids another hard-tail matcher search. It measures
target/B-view quality on the cached pair images and sweeps simple abstain gates
against the current best heatmap-side route.
"""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTE_DIR = PROJECT_ROOT / "runs" / "cross_view_1024_keypointonly_multistate_stylespecific_guard_calib_0step_seed1234"
DEFAULT_FULLVAL_DIR = PROJECT_ROOT / "runs" / "timestamp_compound_quality_gate_fullval_current_route_20260526"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage6"
COMPOUND_CACHE_PREFIX = PROJECT_ROOT / "img" / "CompoundViewpoint_1024"

SUMMARY_FIELDS = ["pair_pt", "matches", "correct", "wrong", "precision"]
QUALITY_FIELDS = [
    "split",
    "pair_pt",
    "pair_rel",
    "source_name",
    "pair_name",
    "matches",
    "correct",
    "wrong",
    "precision",
    "has_match",
    "has_correct",
    "image_b_path",
    "image_b_found",
    "image_b_mean",
    "image_b_std",
    "image_b_grad_mean",
    "image_b_grad_p50",
    "image_b_grad_p75",
    "image_b_grad_p90",
    "image_b_laplacian_var",
    "image_b_local_contrast_mean",
    "image_b_entropy",
    "image_b_sift_keypoints_clahe",
    "image_b_rootsift_candidates_clahe",
]
SWEEP_FIELDS = [
    "split",
    "gate",
    "direction",
    "threshold",
    "pairs_total",
    "pairs_kept",
    "pairs_dropped",
    "matches",
    "correct",
    "wrong",
    "precision",
    "baseline_matches",
    "baseline_correct",
    "baseline_wrong",
    "baseline_precision",
    "correct_retention",
    "match_retention",
    "precision_lift",
]
CANDIDATE_FIELDS = [
    "gate",
    "direction",
    "threshold",
    "validation_precision",
    "validation_precision_lift",
    "validation_correct_retention",
    "validation_pairs_kept",
    "validation_correct",
    "fixed_precision",
    "fixed_precision_lift",
    "fixed_correct_retention",
    "fixed_pairs_kept",
    "fixed_correct",
    "recommend",
    "reason",
]
ROUTE_GROUP_FIELDS = ["view_family", "perturbation", "pairs", "matches", "correct", "wrong", "precision", "correct_pair_count"]


@dataclass(frozen=True)
class GateSpec:
    metric: str
    direction: str

    @property
    def name(self) -> str:
        suffix = "ge" if self.direction == "ge" else "le"
        return f"{self.metric}_{suffix}"


GATES = [
    GateSpec("image_b_grad_mean", "ge"),
    GateSpec("image_b_grad_p75", "ge"),
    GateSpec("image_b_laplacian_var", "ge"),
    GateSpec("image_b_local_contrast_mean", "ge"),
    GateSpec("image_b_entropy", "ge"),
    GateSpec("image_b_sift_keypoints_clahe", "ge"),
    GateSpec("image_b_rootsift_candidates_clahe", "ge"),
    GateSpec("image_b_std", "ge"),
]


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def format_value(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return value


def as_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, object], key: str, default: int = 0) -> int:
    return int(round(as_float(row, key, float(default))))


def infer_pair_rel(pair_pt: str) -> str:
    path = Path(pair_pt)
    parts = path.parts
    if "timestamp" in parts and "compound" in parts:
        source = path.parent.name
        return (Path("img") / "CompoundViewpoint_1024" / source / path.name).as_posix()
    if "img" in parts:
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    raise ValueError(f"cannot infer cache pair path from {pair_pt}")


def b_image_path_from_pair_rel(pair_rel: str) -> Path:
    pair_path = repo_path(pair_rel)
    return pair_path.with_name(f"{pair_path.stem}_view_b.png")


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.uint8)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gray32 = gray.astype(np.float32)
    gx = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def clahe_image(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def count_clahe_sift(gray: np.ndarray, max_features: int) -> int:
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints = sift.detect(clahe_image(gray), None)
    return len(keypoints)


def image_quality_metrics(gray: np.ndarray, max_features: int = 4096) -> dict[str, float | int]:
    grad = gradient_magnitude(gray)
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    local_mean = cv2.blur(gray.astype(np.float32), (9, 9))
    local_sq_mean = cv2.blur(np.square(gray.astype(np.float32)), (9, 9))
    local_var = np.maximum(local_sq_mean - np.square(local_mean), 0.0)
    sift_count = count_clahe_sift(gray, max_features=max_features)
    return {
        "image_b_mean": float(np.mean(gray)),
        "image_b_std": float(np.std(gray)),
        "image_b_grad_mean": float(np.mean(grad)),
        "image_b_grad_p50": float(np.percentile(grad, 50)),
        "image_b_grad_p75": float(np.percentile(grad, 75)),
        "image_b_grad_p90": float(np.percentile(grad, 90)),
        "image_b_laplacian_var": float(np.var(lap)),
        "image_b_local_contrast_mean": float(np.mean(np.sqrt(local_var))),
        "image_b_entropy": entropy(gray),
        "image_b_sift_keypoints_clahe": int(sift_count),
        "image_b_rootsift_candidates_clahe": int(sift_count),
    }


def load_summary_rows(path: Path, split: str, reuse_metrics: dict[str, dict[str, str]] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(path):
        pair_pt = row["pair_pt"]
        pair_rel = row.get("pair_rel") or infer_pair_rel(pair_pt)
        pair_path = Path(pair_rel)
        source_name = row.get("source_name") or pair_path.parent.name
        pair_name = row.get("pair_name") or pair_path.name
        out: dict[str, object] = {
            "split": split,
            "pair_pt": pair_pt,
            "pair_rel": pair_rel,
            "source_name": source_name,
            "pair_name": pair_name,
            "matches": as_int(row, "matches"),
            "correct": as_int(row, "correct"),
            "wrong": as_int(row, "wrong"),
            "precision": as_float(row, "precision"),
        }
        out["has_match"] = int(as_int(out, "matches") > 0)
        out["has_correct"] = int(as_int(out, "correct") > 0)
        b_path = b_image_path_from_pair_rel(pair_rel)
        out["image_b_path"] = rel(b_path)
        out["image_b_found"] = int(b_path.exists())
        reused = reuse_metrics.get(pair_pt) if reuse_metrics else None
        if reused is not None:
            for key in ("image_b_std", "image_b_grad_mean", "image_b_grad_p75"):
                if key in reused:
                    out[key] = as_float(reused, key)
        if b_path.exists():
            out.update(image_quality_metrics(load_gray(b_path)))
        else:
            for key in QUALITY_FIELDS:
                if key.startswith("image_b_") and key not in ("image_b_path", "image_b_found"):
                    out[key] = math.nan
        rows.append(out)
    return rows


def load_reuse_metrics(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["pair_pt"]: row for row in read_csv(path)}


def baseline(rows: list[dict[str, object]]) -> dict[str, float | int]:
    matches = sum(as_int(row, "matches") for row in rows)
    correct = sum(as_int(row, "correct") for row in rows)
    wrong = sum(as_int(row, "wrong") for row in rows)
    precision = correct / matches if matches else 0.0
    return {"matches": matches, "correct": correct, "wrong": wrong, "precision": precision}


def gate_mask(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    valid = np.isfinite(values)
    if direction == "ge":
        return valid & (values >= threshold)
    if direction == "le":
        return valid & (values <= threshold)
    raise ValueError(direction)


def threshold_grid(values: np.ndarray) -> list[float]:
    finite = np.unique(values[np.isfinite(values)])
    if finite.size == 0:
        return []
    thresholds = set(float(v) for v in finite)
    for q in (0, 1, 2, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 98, 99, 100):
        thresholds.add(float(np.percentile(finite, q)))
    return sorted(thresholds)


def sweep_gate(rows: list[dict[str, object]], split: str, gate: GateSpec) -> list[dict[str, object]]:
    base = baseline(rows)
    values = np.array([as_float(row, gate.metric, math.nan) for row in rows], dtype=np.float64)
    out: list[dict[str, object]] = []
    for threshold in threshold_grid(values):
        mask = gate_mask(values, threshold, gate.direction)
        kept = [row for row, keep in zip(rows, mask) if bool(keep)]
        matches = sum(as_int(row, "matches") for row in kept)
        correct = sum(as_int(row, "correct") for row in kept)
        wrong = sum(as_int(row, "wrong") for row in kept)
        precision = correct / matches if matches else 0.0
        base_matches = int(base["matches"])
        base_correct = int(base["correct"])
        out.append(
            {
                "split": split,
                "gate": gate.name,
                "direction": gate.direction,
                "threshold": threshold,
                "pairs_total": len(rows),
                "pairs_kept": len(kept),
                "pairs_dropped": len(rows) - len(kept),
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": precision,
                "baseline_matches": base_matches,
                "baseline_correct": base_correct,
                "baseline_wrong": int(base["wrong"]),
                "baseline_precision": float(base["precision"]),
                "correct_retention": correct / base_correct if base_correct else 0.0,
                "match_retention": matches / base_matches if base_matches else 0.0,
                "precision_lift": precision - float(base["precision"]),
            }
        )
    return out


def sweep_all(rows_by_split: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, split_rows in rows_by_split.items():
        for gate in GATES:
            rows.extend(sweep_gate(split_rows, split, gate))
    return rows


def best_rows(sweep_rows: list[dict[str, object]], split: str, min_correct_retention: float, min_correct: int = 1) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in sweep_rows:
        if row["split"] != split:
            continue
        if as_float(row, "correct_retention") < min_correct_retention or as_int(row, "correct") < min_correct:
            continue
        current = out.get(str(row["gate"]))
        key = (as_float(row, "precision"), as_float(row, "correct_retention"), as_int(row, "correct"), as_int(row, "pairs_kept"))
        if current is None:
            out[str(row["gate"])] = row
            continue
        current_key = (
            as_float(current, "precision"),
            as_float(current, "correct_retention"),
            as_int(current, "correct"),
            as_int(current, "pairs_kept"),
        )
        if key > current_key:
            out[str(row["gate"])] = row
    return out


def nearest_fixed_row(sweep_rows: list[dict[str, object]], validation_row: dict[str, object]) -> dict[str, object] | None:
    gate = validation_row["gate"]
    threshold = as_float(validation_row, "threshold")
    candidates = [row for row in sweep_rows if row["split"] == "fixed_test" and row["gate"] == gate]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(as_float(row, "threshold") - threshold))


def build_candidates(sweep_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    validation_best = best_rows(sweep_rows, "full_val", min_correct_retention=0.80, min_correct=1)
    candidates: list[dict[str, object]] = []
    for gate, val in validation_best.items():
        fixed = nearest_fixed_row(sweep_rows, val)
        fixed_precision_lift = as_float(fixed, "precision_lift") if fixed else math.nan
        fixed_correct_retention = as_float(fixed, "correct_retention") if fixed else math.nan
        val_lift = as_float(val, "precision_lift")
        val_retention = as_float(val, "correct_retention")
        recommend = int(val_lift >= 0.02 and val_retention >= 0.80 and fixed is not None and fixed_precision_lift >= 0.0)
        if recommend:
            reason = "validation lift with >=80% correct retention and non-negative fixed-test lift"
        elif fixed is None:
            reason = "no fixed-test comparison"
        elif fixed_precision_lift < 0.0:
            reason = "validation lift does not transfer to fixed-test"
        else:
            reason = "validation lift below recommendation threshold"
        candidates.append(
            {
                "gate": gate,
                "direction": val["direction"],
                "threshold": val["threshold"],
                "validation_precision": val["precision"],
                "validation_precision_lift": val_lift,
                "validation_correct_retention": val_retention,
                "validation_pairs_kept": val["pairs_kept"],
                "validation_correct": val["correct"],
                "fixed_precision": fixed["precision"] if fixed else math.nan,
                "fixed_precision_lift": fixed_precision_lift,
                "fixed_correct_retention": fixed_correct_retention,
                "fixed_pairs_kept": fixed["pairs_kept"] if fixed else "",
                "fixed_correct": fixed["correct"] if fixed else "",
                "recommend": recommend,
                "reason": reason,
            }
        )
    candidates.sort(
        key=lambda row: (
            int(row["recommend"]),
            as_float(row, "validation_precision_lift"),
            as_float(row, "validation_precision"),
            as_float(row, "fixed_precision_lift", -999.0),
        ),
        reverse=True,
    )
    return candidates


def load_route_group_summary(route_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary in sorted((route_dir / "eval").glob("*/*/summary.csv")):
        perturbation = summary.parent.name
        view_family = summary.parent.parent.name
        summary_rows = read_csv(summary)
        matches = sum(as_int(row, "matches") for row in summary_rows)
        correct = sum(as_int(row, "correct") for row in summary_rows)
        wrong = sum(as_int(row, "wrong") for row in summary_rows)
        rows.append(
            {
                "view_family": view_family,
                "perturbation": perturbation,
                "pairs": len(summary_rows),
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": correct / matches if matches else 0.0,
                "correct_pair_count": sum(1 for row in summary_rows if as_int(row, "correct") > 0),
            }
        )
    return rows


def outcome_group_rows(rows: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    grouped = {
        "correct_pairs": [row for row in rows if as_int(row, "correct") > 0],
        "wrong_only_pairs": [row for row in rows if as_int(row, "matches") > 0 and as_int(row, "correct") == 0],
        "no_match_pairs": [row for row in rows if as_int(row, "matches") == 0],
    }
    out: list[dict[str, object]] = []
    for name, group in grouped.items():
        metrics = baseline(group)
        out.append(
            {
                "split": split,
                "group": name,
                "pairs": len(group),
                "matches": metrics["matches"],
                "correct": metrics["correct"],
                "wrong": metrics["wrong"],
                "precision": metrics["precision"],
                "image_b_grad_mean_median": median_metric(group, "image_b_grad_mean"),
                "image_b_laplacian_var_median": median_metric(group, "image_b_laplacian_var"),
                "image_b_sift_keypoints_clahe_median": median_metric(group, "image_b_sift_keypoints_clahe"),
            }
        )
    return out


def median_metric(rows: list[dict[str, object]], key: str) -> float:
    values = np.array([as_float(row, key, math.nan) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.median(values))


def summarize_markdown(
    output_dir: Path,
    route_rows: list[dict[str, object]],
    rows_by_split: dict[str, list[dict[str, object]]],
    candidates: list[dict[str, object]],
    outcome_rows: list[dict[str, object]],
) -> str:
    lines: list[str] = []
    lines.append("# Matcher Algorithm Iteration Agent13 Stage6")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Direction: quality/routing availability diagnostic, not another hard-tail sparse teacher search.")
    lines.append(f"- Current route: `{rel(DEFAULT_ROUTE_DIR)}`")
    lines.append(f"- Output dir: `{rel(output_dir)}`")
    lines.append("")
    lines.append("## Current Route Six-Group Summary")
    lines.append("")
    lines.append("| view | perturbation | pairs | matches | correct | precision | correct pairs |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in route_rows:
        lines.append(
            f"| {row['view_family']} | {row['perturbation']} | {row['pairs']} | {row['matches']} | "
            f"{row['correct']} | {as_float(row, 'precision'):.6f} | {row['correct_pair_count']} |"
        )
    lines.append("")
    lines.append("## Timestamp/Compound Baselines")
    lines.append("")
    lines.append("| split | pairs | matches | correct | wrong | precision |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split, rows in rows_by_split.items():
        base = baseline(rows)
        lines.append(
            f"| {split} | {len(rows)} | {base['matches']} | {base['correct']} | "
            f"{base['wrong']} | {float(base['precision']):.6f} |"
        )
    lines.append("")
    lines.append("## Outcome Quality Medians")
    lines.append("")
    lines.append("| split | group | pairs | matches | correct | precision | B grad mean | B lap var | CLAHE SIFT kp |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in outcome_rows:
        lines.append(
            f"| {row['split']} | {row['group']} | {row['pairs']} | {row['matches']} | {row['correct']} | "
            f"{as_float(row, 'precision'):.6f} | {as_float(row, 'image_b_grad_mean_median'):.3f} | "
            f"{as_float(row, 'image_b_laplacian_var_median'):.1f} | "
            f"{as_float(row, 'image_b_sift_keypoints_clahe_median'):.0f} |"
        )
    lines.append("")
    lines.append("## Gate Candidates")
    lines.append("")
    if candidates:
        lines.append("| gate | threshold | val precision | val lift | val retention | fixed precision | fixed lift | fixed retention | recommend |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in candidates[:10]:
            lines.append(
                f"| {row['gate']} | {as_float(row, 'threshold'):.6f} | {as_float(row, 'validation_precision'):.6f} | "
                f"{as_float(row, 'validation_precision_lift'):.6f} | {as_float(row, 'validation_correct_retention'):.3f} | "
                f"{as_float(row, 'fixed_precision'):.6f} | {as_float(row, 'fixed_precision_lift'):.6f} | "
                f"{as_float(row, 'fixed_correct_retention'):.3f} | {row['recommend']} |"
            )
    else:
        lines.append("No candidate retained at least 80% of full-val correct matches.")
    lines.append("")
    recommended = [row for row in candidates if as_int(row, "recommend") == 1]
    lines.append("## Recommendation")
    lines.append("")
    if recommended:
        best = recommended[0]
        lines.append(
            f"Recommend considering `{best['gate']}` at threshold `{as_float(best, 'threshold'):.6f}` as a calibration/routing dimension, "
            "with validation backing and non-negative fixed-test transfer. Treat it as an abstain/routing feature, not as a trained model."
        )
    else:
        lines.append(
            "Do not adopt a hard-coded gate into the main route yet. The validation-backed sweeps either do not transfer cleanly to fixed-test "
            "or the lift is too small for a stable calibration dimension."
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for name in (
        "route_group_summary.csv",
        "pair_quality_metrics.csv",
        "outcome_group_summary.csv",
        "threshold_sweep.csv",
        "route_gate_candidates.csv",
        "summary.md",
    ):
        lines.append(f"- `{name}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    route_dir = args.route_dir
    fullval_summary = args.fullval_summary or (args.fullval_dir / "summary.csv")
    fixed_summary = args.fixed_summary or (route_dir / "eval" / "timestamp" / "compound" / "summary.csv")

    route_rows = load_route_group_summary(route_dir)
    reuse_metrics = load_reuse_metrics(args.fullval_dir / "pair_quality_metrics.csv")
    rows_by_split = {
        "fixed_test": load_summary_rows(fixed_summary, "fixed_test", reuse_metrics=None),
        "full_val": load_summary_rows(fullval_summary, "full_val", reuse_metrics=reuse_metrics),
    }
    all_quality_rows = rows_by_split["fixed_test"] + rows_by_split["full_val"]
    sweep_rows = sweep_all(rows_by_split)
    candidates = build_candidates(sweep_rows)
    outcome_rows = outcome_group_rows(rows_by_split["fixed_test"], "fixed_test") + outcome_group_rows(rows_by_split["full_val"], "full_val")

    write_csv(output_dir / "route_group_summary.csv", route_rows, ROUTE_GROUP_FIELDS)
    write_csv(output_dir / "pair_quality_metrics.csv", all_quality_rows, QUALITY_FIELDS)
    write_csv(
        output_dir / "outcome_group_summary.csv",
        outcome_rows,
        [
            "split",
            "group",
            "pairs",
            "matches",
            "correct",
            "wrong",
            "precision",
            "image_b_grad_mean_median",
            "image_b_laplacian_var_median",
            "image_b_sift_keypoints_clahe_median",
        ],
    )
    write_csv(output_dir / "threshold_sweep.csv", sweep_rows, SWEEP_FIELDS)
    write_csv(output_dir / "route_gate_candidates.csv", candidates, CANDIDATE_FIELDS)
    (output_dir / "summary.md").write_text(summarize_markdown(output_dir, route_rows, rows_by_split, candidates, outcome_rows), encoding="utf-8")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image_dir = root / "img" / "CompoundViewpoint_1024" / "source_000001"
        image_dir.mkdir(parents=True)
        low = np.full((64, 64), 96, dtype=np.uint8)
        high = np.indices((64, 64)).sum(axis=0).astype(np.uint8) * 4
        cv2.imwrite(str(image_dir / "pair_000001_view_b.png"), low)
        cv2.imwrite(str(image_dir / "pair_000002_view_b.png"), high)
        low_metrics = image_quality_metrics(load_gray(image_dir / "pair_000001_view_b.png"), max_features=128)
        high_metrics = image_quality_metrics(load_gray(image_dir / "pair_000002_view_b.png"), max_features=128)
        assert high_metrics["image_b_grad_mean"] > low_metrics["image_b_grad_mean"]
        rows = [
            {"matches": 10, "correct": 1, "wrong": 9, "image_b_grad_mean": 1.0, "image_b_sift_keypoints_clahe": 0},
            {"matches": 10, "correct": 9, "wrong": 1, "image_b_grad_mean": 10.0, "image_b_sift_keypoints_clahe": 20},
        ]
        sweep = sweep_gate(rows, "synthetic", GateSpec("image_b_grad_mean", "ge"))
        best = max(sweep, key=lambda row: as_float(row, "precision"))
        assert 1.0 < as_float(best, "threshold") <= 10.0
        assert as_int(best, "correct") == 9
        assert abs(as_float(best, "precision") - 0.9) < 1e-9
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--fullval-dir", type=Path, default=DEFAULT_FULLVAL_DIR)
    parser.add_argument("--fixed-summary", type=Path, default=None)
    parser.add_argument("--fullval-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
