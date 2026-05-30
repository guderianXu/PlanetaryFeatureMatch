#!/usr/bin/env python3
"""Agent7 matcher iteration: RootSIFT/PFM routing and failure-case mining."""

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
ROTATION_BENCH_SCRIPT = PROJECT_ROOT / "python" / "rotation_matcher_benchmark.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent7"
DEFAULT_PFM_RUN = PROJECT_ROOT / "runs" / "cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234"
DEFAULT_PFM_STATE = DEFAULT_PFM_RUN / "training" / "pytorch_pfm_state.pt"

METRIC_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "pair_pt",
    "row_type",
    "name",
    "selected_algorithm",
    "status",
    "keypoints_a",
    "keypoints_b",
    "matches",
    "correct",
    "wrong",
    "precision",
    "coverage",
    "pass_gate",
    "mean_error_px",
    "median_error_px",
    "visualization",
    "message",
]

SUMMARY_FIELDS = [
    "style",
    "gate",
    "rotation_deg",
    "row_type",
    "name",
    "pairs",
    "ok_pairs",
    "selected_rootsift",
    "selected_pfm",
    "covered_pairs",
    "pass_gate_pairs",
    "matches",
    "correct",
    "wrong",
    "precision",
    "mean_pair_precision",
    "mean_matches_per_pair",
    "median_matches_per_pair",
]

CASE_FIELDS = [
    "case_type",
    "style",
    "gate",
    "rotation_deg",
    "pair_pt",
    "rootsift_matches",
    "rootsift_correct",
    "rootsift_precision",
    "pfm_matches",
    "pfm_correct",
    "pfm_precision",
    "visualization",
]


@dataclass(frozen=True)
class MetricRow:
    style: str
    gate: str
    rotation_deg: int
    pair_pt: str
    row_type: str
    name: str
    selected_algorithm: str
    status: str
    keypoints_a: int
    keypoints_b: int
    matches: int
    correct: int
    wrong: int
    precision: float
    coverage: int
    pass_gate: int
    mean_error_px: float
    median_error_px: float
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


A4 = load_module(AGENT4_SCRIPT, "agent4_matcher_for_agent7")
RMB = load_module(ROTATION_BENCH_SCRIPT, "rotation_matcher_for_agent7")


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


def metric_from_match(
    *,
    args: argparse.Namespace,
    style: str,
    gate: str,
    rotation_deg: int,
    pair_path: Path,
    algorithm: str,
    output,
    warp_a_to_b,
    valid_mask,
    original_b_shape: tuple[int, int],
    status: str = "ok",
    message: str = "",
) -> tuple[MetricRow, np.ndarray]:
    points_b_original = A4.unrotate_points(output.points_b, original_b_shape[0], original_b_shape[1], rotation_deg)
    matches, correct, wrong, precision, mean_error, median_error = A4.compute_metrics(
        output.points_a,
        points_b_original,
        warp_a_to_b,
        valid_mask,
        threshold_px=args.threshold_px,
    )
    gate_min = min_gate_labels(gate)
    row = MetricRow(
        style=style,
        gate=gate,
        rotation_deg=rotation_deg,
        pair_pt=pair_path.as_posix(),
        row_type="algorithm",
        name=algorithm,
        selected_algorithm=algorithm,
        status=status,
        keypoints_a=int(output.keypoints_a),
        keypoints_b=int(output.keypoints_b),
        matches=matches,
        correct=correct,
        wrong=wrong,
        precision=precision,
        coverage=1 if matches > 0 else 0,
        pass_gate=1 if correct >= gate_min else 0,
        mean_error_px=mean_error,
        median_error_px=median_error,
        message=message,
    )
    return row, points_b_original


def route_row(
    *,
    base: MetricRow,
    name: str,
    selected: MetricRow,
) -> MetricRow:
    return MetricRow(
        style=base.style,
        gate=base.gate,
        rotation_deg=base.rotation_deg,
        pair_pt=base.pair_pt,
        row_type="routing",
        name=name,
        selected_algorithm=selected.name,
        status=selected.status,
        keypoints_a=selected.keypoints_a,
        keypoints_b=selected.keypoints_b,
        matches=selected.matches,
        correct=selected.correct,
        wrong=selected.wrong,
        precision=selected.precision,
        coverage=selected.coverage,
        pass_gate=selected.pass_gate,
        mean_error_px=selected.mean_error_px,
        median_error_px=selected.median_error_px,
        visualization=selected.visualization,
        message=selected.message,
    )


def deployed_route(rootsift: MetricRow, pfm: MetricRow) -> MetricRow:
    return rootsift if rootsift.matches >= min_gate_labels(rootsift.gate) else pfm


def oracle_route(rootsift: MetricRow, pfm: MetricRow) -> MetricRow:
    root_key = (rootsift.pass_gate, rootsift.correct, rootsift.precision, rootsift.matches)
    pfm_key = (pfm.pass_gate, pfm.correct, pfm.precision, pfm.matches)
    return rootsift if root_key >= pfm_key else pfm


def visualize_case(
    *,
    args: argparse.Namespace,
    case_type: str,
    style: str,
    gate: str,
    rotation_deg: int,
    pair_path: Path,
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    points_b_original: np.ndarray,
    used: dict[str, int],
) -> str:
    key = f"{case_type}/{style}/{gate}/rot{rotation_deg}"
    if used.get(key, 0) >= args.visualizations_per_case:
        return ""
    used[key] = used.get(key, 0) + 1
    path = (
        args.output_dir
        / "visualizations"
        / case_type
        / style
        / gate
        / f"rot{rotation_deg}_{pair_path.parent.name}_{pair_path.stem}.png"
    )
    A4.draw_visualization(image_a, image_b, points_a, points_b_original, path)
    return path.as_posix()


def evaluate(args: argparse.Namespace) -> tuple[list[MetricRow], list[dict[str, object]], list[dict[str, object]]]:
    device = choose_device(args.device)
    rootsift_base = A4.RootSiftFlannMatcher(
        name="RootSIFT-HRANSAC-r0.80-t2",
        max_keypoints=args.rootsift_max_keypoints,
        max_matches=args.max_matches,
        ratio=0.80,
        mutual=False,
        sift_contrast=args.sift_contrast,
    )
    pfm = RMB.PFMPyTorchMatcher(
        state_path=args.pfm_state,
        device=device,
        max_keypoints=args.pfm_max_keypoints,
        max_matches=args.max_matches,
        min_intensity=args.pfm_min_intensity,
        min_score=args.pfm_min_score,
    )

    rows: list[MetricRow] = []
    cases: list[dict[str, object]] = []
    sampled: list[dict[str, object]] = []
    vis_used: dict[str, int] = {}
    for style in args.styles:
        for gate in args.gates:
            pair_paths = A4.select_pairs(args, style, gate)
            sampled.extend({"style": style, "gate": gate, "pair_pt": path.as_posix()} for path in pair_paths)
            print(f"group={style}/{gate} pairs={len(pair_paths)}", flush=True)
            rootsift = A4.HomographyRansacWrapper(rootsift_base, threshold_px=2.0, min_inliers=min_gate_labels(gate))
            rootsift.name = "RootSIFT-HRANSAC-r0.80-t2"
            for pair_index, pair_path in enumerate(pair_paths, start=1):
                image_a, image_b, warp_a_to_b, valid_mask = A4.load_pair(pair_path)
                for rotation_deg in args.rotations:
                    image_b_rotated = A4.rotate_image(image_b, rotation_deg)
                    try:
                        root_output = rootsift.match(image_a, image_b_rotated)
                        root_row, root_b_original = metric_from_match(
                            args=args,
                            style=style,
                            gate=gate,
                            rotation_deg=rotation_deg,
                            pair_path=pair_path,
                            algorithm=rootsift.name,
                            output=root_output,
                            warp_a_to_b=warp_a_to_b,
                            valid_mask=valid_mask,
                            original_b_shape=image_b.shape[:2],
                        )
                    except Exception as exc:
                        root_output = A4.MatchOutput(A4.empty_points(), A4.empty_points(), 0, 0)
                        root_b_original = A4.empty_points()
                        root_row = MetricRow(style, gate, rotation_deg, pair_path.as_posix(), "algorithm", rootsift.name, rootsift.name, "error", 0, 0, 0, 0, 0, 0.0, 0, 0, math.nan, math.nan, message=f"{type(exc).__name__}: {exc}")
                    try:
                        pfm_output = pfm.match(image_a, image_b_rotated)
                        pfm_row, pfm_b_original = metric_from_match(
                            args=args,
                            style=style,
                            gate=gate,
                            rotation_deg=rotation_deg,
                            pair_path=pair_path,
                            algorithm=pfm.name,
                            output=pfm_output,
                            warp_a_to_b=warp_a_to_b,
                            valid_mask=valid_mask,
                            original_b_shape=image_b.shape[:2],
                        )
                    except Exception as exc:
                        pfm_output = A4.MatchOutput(A4.empty_points(), A4.empty_points(), 0, 0)
                        pfm_b_original = A4.empty_points()
                        pfm_row = MetricRow(style, gate, rotation_deg, pair_path.as_posix(), "algorithm", pfm.name, pfm.name, "error", 0, 0, 0, 0, 0, 0.0, 0, 0, math.nan, math.nan, message=f"{type(exc).__name__}: {exc}")

                    rows.extend([root_row, pfm_row])
                    rows.append(route_row(base=root_row, name="RouteRootSIFTMinElsePFM", selected=deployed_route(root_row, pfm_row)))
                    rows.append(route_row(base=root_row, name="OracleBestOfRootSIFTPFM", selected=oracle_route(root_row, pfm_row)))

                    case_type = ""
                    case_points_a = A4.empty_points()
                    case_points_b = A4.empty_points()
                    if root_row.pass_gate and not pfm_row.pass_gate:
                        case_type = "rootsift_only_pass"
                        case_points_a, case_points_b = root_output.points_a, root_b_original
                    elif pfm_row.pass_gate and not root_row.pass_gate:
                        case_type = "pfm_only_pass"
                        case_points_a, case_points_b = pfm_output.points_a, pfm_b_original
                    elif root_row.matches == 0 and pfm_row.matches > 0:
                        case_type = "rootsift_empty_pfm_nonempty"
                        case_points_a, case_points_b = pfm_output.points_a, pfm_b_original
                    elif pfm_row.matches == 0 and root_row.matches > 0:
                        case_type = "pfm_empty_rootsift_nonempty"
                        case_points_a, case_points_b = root_output.points_a, root_b_original
                    if case_type:
                        visualization = visualize_case(
                            args=args,
                            case_type=case_type,
                            style=style,
                            gate=gate,
                            rotation_deg=rotation_deg,
                            pair_path=pair_path,
                            image_a=image_a,
                            image_b=image_b,
                            points_a=case_points_a,
                            points_b_original=case_points_b,
                            used=vis_used,
                        )
                        cases.append(
                            {
                                "case_type": case_type,
                                "style": style,
                                "gate": gate,
                                "rotation_deg": rotation_deg,
                                "pair_pt": pair_path.as_posix(),
                                "rootsift_matches": root_row.matches,
                                "rootsift_correct": root_row.correct,
                                "rootsift_precision": root_row.precision,
                                "pfm_matches": pfm_row.matches,
                                "pfm_correct": pfm_row.correct,
                                "pfm_precision": pfm_row.precision,
                                "visualization": visualization,
                            }
                        )
                    print(
                        f"{style:9s} {gate:9s} {pair_index:02d}/{len(pair_paths):02d} rot={rotation_deg:3d} "
                        f"RootSIFT {root_row.correct:3d}/{root_row.matches:3d} p={root_row.precision:.3f} "
                        f"PFM {pfm_row.correct:3d}/{pfm_row.matches:3d} p={pfm_row.precision:.3f}",
                        flush=True,
                    )
    return rows, cases, sampled


def aggregate(rows: list[MetricRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.style, row.gate, row.rotation_deg, row.row_type, row.name), []).append(row)
    out: list[dict[str, object]] = []
    for (style, gate, rotation_deg, row_type, name), items in sorted(grouped.items()):
        matches = sum(row.matches for row in items)
        correct = sum(row.correct for row in items)
        wrong = sum(row.wrong for row in items)
        ok = [row for row in items if row.status == "ok"]
        out.append(
            {
                "style": style,
                "gate": gate,
                "rotation_deg": rotation_deg,
                "row_type": row_type,
                "name": name,
                "pairs": len(items),
                "ok_pairs": len(ok),
                "selected_rootsift": sum(1 for row in items if row.selected_algorithm.startswith("RootSIFT")),
                "selected_pfm": sum(1 for row in items if row.selected_algorithm == "PFM"),
                "covered_pairs": sum(row.coverage for row in items),
                "pass_gate_pairs": sum(row.pass_gate for row in items),
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": 0.0 if matches == 0 else correct / matches,
                "mean_pair_precision": float(np.mean([row.precision for row in ok])) if ok else math.nan,
                "mean_matches_per_pair": float(np.mean([row.matches for row in ok])) if ok else math.nan,
                "median_matches_per_pair": float(np.median([row.matches for row in ok])) if ok else math.nan,
            }
        )
    return out


def summary_table(rows: list[dict[str, object]], *, row_type: str) -> list[str]:
    selected = [row for row in rows if row["row_type"] == row_type]
    lines = [
        "| style | gate | rot | name | pairs | pass gate | matches | precision | selected RootSIFT/PFM |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['style']} | {row['gate']} | {row['rotation_deg']} | {row['name']} | "
            f"{row['pairs']} | {row['pass_gate_pairs']} | {row['matches']} | {float(row['precision']):.4f} | "
            f"{row['selected_rootsift']}/{row['selected_pfm']} |"
        )
    return lines


def write_summary(args: argparse.Namespace, summary_rows: list[dict[str, object]], cases: list[dict[str, object]]) -> None:
    routing = [row for row in summary_rows if row["row_type"] == "routing"]
    algo = [row for row in summary_rows if row["row_type"] == "algorithm"]
    route_rows = [row for row in routing if row["name"] == "RouteRootSIFTMinElsePFM"]
    oracle_rows = [row for row in routing if row["name"] == "OracleBestOfRootSIFTPFM"]
    lines = [
        "# Matcher Algorithm Iteration Agent7",
        "",
        "## Scope",
        "",
        "- Experiment: RootSIFT/PFM routing on 1024 cached cross-view pairs with B-image rotations.",
        "- RootSIFT: ratio=0.80, Homography RANSAC threshold=2px, min labels 20 for viewpoint and 8 for compound.",
        f"- PFM checkpoint: `{args.pfm_state}`.",
        f"- pairs per style/gate: `{args.pairs_per_group}`; rotations: `{','.join(str(item) for item in args.rotations)}`.",
        "- Extreme tests are intentionally excluded.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=python MKL_THREADING_LAYER=GNU "
        f"/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/{Path(__file__).name} "
        f"--pairs-per-group {args.pairs_per_group} --device {args.device}",
        "```",
        "",
        "## Routing Summary",
        "",
        *summary_table(route_rows + oracle_rows, row_type="routing"),
        "",
        "## Algorithm Baselines",
        "",
        *summary_table(algo, row_type="algorithm"),
        "",
        "## Hard Cases",
        "",
        f"- hard case rows: `{len(cases)}` in `hard_cases.csv`.",
    ]
    case_counts: dict[str, int] = {}
    for case in cases:
        case_counts[str(case["case_type"])] = case_counts.get(str(case["case_type"]), 0) + 1
    for case_type, count in sorted(case_counts.items()):
        lines.append(f"- {case_type}: {count}")
    if route_rows:
        route_pass = sum(int(row["pass_gate_pairs"]) for row in route_rows)
        route_pairs = sum(int(row["pairs"]) for row in route_rows)
        route_matches = sum(int(row["matches"]) for row in route_rows)
        route_correct = sum(int(row["correct"]) for row in route_rows)
        route_precision = 0.0 if route_matches == 0 else route_correct / route_matches
        lines.extend(
            [
                "",
                "## Conclusion",
                "",
                f"- Deployable routing passed the label gate on {route_pass}/{route_pairs} rotated pair evaluations with aggregate precision {route_precision:.4f}.",
            ]
        )
    if oracle_rows:
        oracle_pass = sum(int(row["pass_gate_pairs"]) for row in oracle_rows)
        oracle_pairs = sum(int(row["pairs"]) for row in oracle_rows)
        lines.append(f"- Oracle best-of-two upper bound passed {oracle_pass}/{oracle_pairs}; the gap estimates how much a better confidence router could recover.")
    lines.extend(
        [
            "- Use `hard_cases.csv` and `visualizations/` for the next training/mining pass, especially `pfm_only_pass` and `rootsift_only_pass` rows.",
            "",
            "## Outputs",
            "",
            "- `metrics.csv`",
            "- `summary_metrics.csv`",
            "- `hard_cases.csv`",
            "- `sampled_pairs.csv`",
            "- `visualizations/`",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pfm-run", type=Path, default=DEFAULT_PFM_RUN)
    parser.add_argument("--pfm-state", type=Path, default=DEFAULT_PFM_STATE)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--styles", nargs="+", default=["numeric", "timestamp"], choices=["numeric", "timestamp"])
    parser.add_argument("--gates", nargs="+", default=["viewpoint", "compound"], choices=["viewpoint", "compound"])
    parser.add_argument("--rotations", nargs="+", type=int, default=[90, 180, 270], choices=[0, 90, 180, 270])
    parser.add_argument("--pairs-per-group", type=int, default=4)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--rootsift-max-keypoints", type=int, default=1024)
    parser.add_argument("--pfm-max-keypoints", type=int, default=2048)
    parser.add_argument("--pfm-min-intensity", type=float, default=0.01)
    parser.add_argument("--pfm-min-score", type=float, default=-1.0)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--visualizations-per-case", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if not args.pfm_state.exists():
        raise FileNotFoundError(f"PFM state not found: {args.pfm_state}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, cases, sampled = evaluate(args)
    write_csv(args.output_dir / "metrics.csv", [asdict(row) for row in rows], METRIC_FIELDS)
    summary_rows = aggregate(rows)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.output_dir / "hard_cases.csv", cases, CASE_FIELDS)
    write_csv(args.output_dir / "sampled_pairs.csv", sampled, ["style", "gate", "pair_pt"])
    write_summary(args, summary_rows, cases)
    print(f"output_dir={args.output_dir}")
    print(f"metrics={args.output_dir / 'metrics.csv'}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
