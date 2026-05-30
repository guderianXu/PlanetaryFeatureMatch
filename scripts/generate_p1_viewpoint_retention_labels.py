#!/usr/bin/env python3
"""Generate train-only P1 viewpoint retention labels with strict caps."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import pfm_model  # noqa: E402
import pseudo_label_generation as plg  # noqa: E402
import pytorch_cache_match_eval as eval_py  # noqa: E402
from patch_descriptor_training import discover_pair_archives  # noqa: E402


DEFAULT_SPLIT_ROOT = Path("runs/cross_view_1024_keypointonly_multistate_stylespecific_guard_calib_0step_seed1234/splits")
DEFAULT_SELECTED_WEIGHTS = Path(
    "runs/cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234/"
    "calibration/selected_weights.csv"
)
DEFAULT_GROUPS = "numeric/viewpoint,timestamp/viewpoint"
STYLES = {"numeric", "timestamp"}
GATES = {"rotate", "viewpoint", "compound"}


@dataclass(frozen=True)
class RetentionCandidate:
    style: str
    gate: str
    source_name: str
    pair_pt: str
    ransac_matches: int
    truth_filtered_matches: int
    wrong: int
    retained_labels: int = 0

    @property
    def precision(self) -> float:
        if self.ransac_matches <= 0:
            return 0.0
        return float(self.truth_filtered_matches) / float(self.ransac_matches)


@dataclass(frozen=True)
class GeneratedCandidate:
    candidate: RetentionCandidate
    cache_rel: Path
    pair_rel: Path
    points_a: np.ndarray
    points_b: np.ndarray
    errors: np.ndarray
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    mean_error_px: float
    message: str = ""


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_to_project(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def parse_group_keys(text: str) -> list[tuple[str, str]]:
    groups: list[tuple[str, str]] = []
    for item in text.split(","):
        value = item.strip()
        if not value:
            continue
        style, sep, gate = value.partition("/")
        if sep != "/" or style not in STYLES or gate not in GATES:
            raise ValueError(f"invalid group key: {value!r}")
        groups.append((style, gate))
    if not groups:
        raise ValueError("at least one group is required")
    return groups


def read_route_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with project_path(path).open(newline="", encoding="utf-8") as handle:
        return {(row["style"], row["gate"]): row for row in csv.DictReader(handle)}


def load_model_cache(device: str):
    cache: dict[Path, pfm_model.PlanetaryFeatureMatcher] = {}

    def load(path: Path) -> pfm_model.PlanetaryFeatureMatcher:
        resolved = project_path(path)
        model = cache.get(resolved)
        if model is None:
            model, _ = pfm_model.load_pytorch_state(resolved, device=device)
            model.eval()
            cache[resolved] = model
        return model

    return load


def select_kept_candidates(
    candidates: list[RetentionCandidate],
    *,
    eligible_groups: set[tuple[str, str]],
    label_cap_per_pair: int,
    label_cap_per_group: int,
    pair_min_inliers: int,
    pair_min_precision: float,
    pair_max_wrong: int,
    max_source_pairs: int,
) -> list[RetentionCandidate]:
    selected: list[RetentionCandidate] = []
    labels_by_group: dict[tuple[str, str], int] = {}
    pairs_by_source: dict[tuple[str, str, str], int] = {}
    for candidate in candidates:
        group = (candidate.style, candidate.gate)
        if group not in eligible_groups:
            continue
        if candidate.ransac_matches < pair_min_inliers:
            continue
        if candidate.wrong > pair_max_wrong:
            continue
        if candidate.precision < pair_min_precision:
            continue
        source_key = (candidate.style, candidate.gate, candidate.source_name)
        if max_source_pairs > 0 and pairs_by_source.get(source_key, 0) >= max_source_pairs:
            continue
        group_labels = labels_by_group.get(group, 0)
        remaining = max(0, label_cap_per_group - group_labels)
        if remaining <= 0:
            continue
        retained = min(label_cap_per_pair, candidate.truth_filtered_matches, remaining)
        if retained <= 0:
            continue
        selected_candidate = replace(candidate, retained_labels=retained)
        selected.append(selected_candidate)
        labels_by_group[group] = group_labels + retained
        pairs_by_source[source_key] = pairs_by_source.get(source_key, 0) + 1
    return selected


def evaluate_pfm_pair(
    model: pfm_model.PlanetaryFeatureMatcher,
    route: dict[str, str],
    pair_path: Path,
    *,
    device: torch.device,
    max_keypoints: int,
    descriptor_topk: int,
    geometry_filter: str,
) -> eval_py.PairEvaluation:
    return eval_py.evaluate_pair_path(
        model,
        pair_path,
        device=device,
        mode="blend",
        texture_blend_weight=float(route["texture_blend_weight"]),
        max_keypoints=max_keypoints,
        min_intensity=0.01,
        texture_fraction=1.0,
        threshold_px=5.0,
        topk=descriptor_topk,
        max_matches=512,
        min_score=-1.0,
        min_margin=float(route["min_margin"]),
        min_target_gradient=float(route.get("min_target_gradient") or 0.0),
        min_target_local_contrast=float(route.get("min_target_local_contrast") or 0.0),
        mutual=True,
        geometry_filter=geometry_filter,
        keypoint_spatial_bins=0,
        keypoint_score_mode=route["keypoint_score_mode"],
    )


def generate_uncapped_candidate(
    *,
    style: str,
    gate: str,
    cache_rel: Path,
    pair_path: Path,
    max_keypoints: int,
    max_raw_matches: int,
    ratio: float,
    sift_contrast: float,
    ransac_threshold_px: float,
    truth_threshold_px: float,
) -> GeneratedCandidate:
    image_a, image_b, warp_a_to_b, valid_mask = plg.load_pair(pair_path)
    raw = plg.rootsift_flann_ratio_match(
        image_a,
        image_b,
        max_keypoints=max_keypoints,
        max_matches=max_raw_matches,
        ratio=ratio,
        sift_contrast=sift_contrast,
    )
    ransac_a, ransac_b = plg.homography_inliers(raw.points_a, raw.points_b, threshold_px=ransac_threshold_px)
    truth_a, truth_b, errors = plg.filter_matches_by_warp_truth(
        ransac_a,
        ransac_b,
        warp_a_to_b,
        valid_mask,
        threshold_px=truth_threshold_px,
    )
    ransac_count = int(ransac_a.shape[0])
    truth_count = int(truth_a.shape[0])
    wrong = max(0, ransac_count - truth_count)
    pair_rel = relative_to_project(pair_path)
    candidate = RetentionCandidate(
        style=style,
        gate=gate,
        source_name=pair_path.parent.name,
        pair_pt=pair_rel.as_posix(),
        ransac_matches=ransac_count,
        truth_filtered_matches=truth_count,
        wrong=wrong,
    )
    return GeneratedCandidate(
        candidate=candidate,
        cache_rel=cache_rel,
        pair_rel=pair_rel,
        points_a=truth_a,
        points_b=truth_b,
        errors=errors,
        keypoints_a=raw.keypoints_a,
        keypoints_b=raw.keypoints_b,
        raw_matches=int(raw.points_a.shape[0]),
        mean_error_px=float(errors.mean()) if errors.size else math.nan,
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--selected-weights", type=Path, default=DEFAULT_SELECTED_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", default=DEFAULT_GROUPS)
    parser.add_argument("--sample-pairs-per-group", type=int, default=128)
    parser.add_argument("--max-zero-pairs-per-group", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-threshold-px", type=float, default=2.0)
    parser.add_argument("--truth-threshold-px", type=float, default=3.0)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--sift-max-keypoints", type=int, default=2048)
    parser.add_argument("--sift-max-raw-matches", type=int, default=512)
    parser.add_argument("--label-cap-per-pair", type=int, default=12)
    parser.add_argument("--label-cap-per-group", type=int, default=384)
    parser.add_argument("--pair-min-inliers", type=int, default=50)
    parser.add_argument("--pair-min-precision", type=float, default=0.995)
    parser.add_argument("--pair-max-wrong", type=int, default=0)
    parser.add_argument("--max-source-pairs", type=int, default=2)
    parser.add_argument("--pfm-max-keypoints", type=int, default=4096)
    parser.add_argument("--pfm-descriptor-topk", type=int, default=32)
    parser.add_argument("--pfm-geometry-filter", default="local", choices=["none", "local", "affine"])
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = [
        ("--sample-pairs-per-group", args.sample_pairs_per_group),
        ("--max-zero-pairs-per-group", args.max_zero_pairs_per_group),
        ("--label-cap-per-pair", args.label_cap_per_pair),
        ("--label-cap-per-group", args.label_cap_per_group),
        ("--pair-min-inliers", args.pair_min_inliers),
        ("--sift-max-keypoints", args.sift_max_keypoints),
        ("--sift-max-raw-matches", args.sift_max_raw_matches),
    ]
    for name, value in positive:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= args.pair_min_precision <= 1.0:
        raise ValueError("--pair-min-precision must be in [0, 1]")
    if args.pair_max_wrong < 0:
        raise ValueError("--pair-max-wrong must be nonnegative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    groups = parse_group_keys(args.groups)
    eligible_groups = set(groups)
    routes = read_route_rows(args.selected_weights)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_model = load_model_cache(args.device)
    device = torch.device(args.device)

    sampled_rows: list[dict[str, object]] = []
    generated: list[GeneratedCandidate] = []
    pair_summary: list[plg.PairSummaryRow] = []

    for group_index, (style, gate) in enumerate(groups, start=1):
        route = routes[(style, gate)]
        cache_dir = project_path(args.split_root) / args.split / style / gate
        cache_rel = relative_to_project(cache_dir)
        pair_paths = eval_py.limit_pair_paths(
            discover_pair_archives([cache_dir], limit_pairs=0, exclude_self_pairs=True),
            limit_pairs=args.sample_pairs_per_group,
            sample_seed=args.seed + group_index * 1009,
        )
        model = load_model(Path(route["pytorch_state"]))
        zero_seen = 0
        for pair_index, pair_path in enumerate(pair_paths, start=1):
            pair_rel = relative_to_project(pair_path)
            pfm_result = evaluate_pfm_pair(
                model,
                route,
                pair_path,
                device=device,
                max_keypoints=args.pfm_max_keypoints,
                descriptor_topk=args.pfm_descriptor_topk,
                geometry_filter=args.pfm_geometry_filter,
            )
            status = "not_gate_zero"
            gen: GeneratedCandidate | None = None
            if pfm_result.matches == 0:
                if zero_seen >= args.max_zero_pairs_per_group:
                    status = "not_selected_max_zero"
                else:
                    zero_seen += 1
                    try:
                        gen = generate_uncapped_candidate(
                            style=style,
                            gate=gate,
                            cache_rel=cache_rel,
                            pair_path=pair_path,
                            max_keypoints=args.sift_max_keypoints,
                            max_raw_matches=args.sift_max_raw_matches,
                            ratio=args.ratio,
                            sift_contrast=args.sift_contrast,
                            ransac_threshold_px=args.ransac_threshold_px,
                            truth_threshold_px=args.truth_threshold_px,
                        )
                        generated.append(gen)
                        status = "generated"
                        pair_summary.append(
                            plg.PairSummaryRow(
                                cache_dir=cache_rel.as_posix(),
                                pair_pt=pair_rel.as_posix(),
                                status=status,
                                keypoints_a=gen.keypoints_a,
                                keypoints_b=gen.keypoints_b,
                                raw_matches=gen.raw_matches,
                                ransac_matches=gen.candidate.ransac_matches,
                                truth_filtered_matches=gen.candidate.truth_filtered_matches,
                                mean_error_px=gen.mean_error_px,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - keep batch generation moving.
                        status = "error"
                        pair_summary.append(
                            plg.PairSummaryRow(
                                cache_dir=cache_rel.as_posix(),
                                pair_pt=pair_rel.as_posix(),
                                status=status,
                                keypoints_a=0,
                                keypoints_b=0,
                                raw_matches=0,
                                ransac_matches=0,
                                truth_filtered_matches=0,
                                mean_error_px=math.nan,
                                message=str(exc),
                            )
                        )
            sampled_rows.append(
                {
                    "style": style,
                    "gate": gate,
                    "source_name": pair_path.parent.name,
                    "pair_pt": pair_rel.as_posix(),
                    "pfm_matches": pfm_result.matches,
                    "pfm_correct": pfm_result.correct,
                    "pfm_precision": f"{pfm_result.precision:.6f}",
                    "is_gate_zero": int(pfm_result.matches == 0),
                    "generation_status": status,
                    "ransac_matches": gen.candidate.ransac_matches if gen else 0,
                    "truth_filtered_matches": gen.candidate.truth_filtered_matches if gen else 0,
                    "truth_precision": f"{gen.candidate.precision:.6f}" if gen else "0.000000",
                    "wrong": gen.candidate.wrong if gen else 0,
                }
            )
            print(
                f"{style}/{gate} {pair_index:03d}/{len(pair_paths):03d} "
                f"pfm={pfm_result.correct}/{pfm_result.matches} status={status}",
                flush=True,
            )

    selected = select_kept_candidates(
        [item.candidate for item in generated],
        eligible_groups=eligible_groups,
        label_cap_per_pair=args.label_cap_per_pair,
        label_cap_per_group=args.label_cap_per_group,
        pair_min_inliers=args.pair_min_inliers,
        pair_min_precision=args.pair_min_precision,
        pair_max_wrong=args.pair_max_wrong,
        max_source_pairs=args.max_source_pairs,
    )
    selected_by_pair = {item.pair_pt: item for item in selected}
    matcher_name = f"RootSIFT-FLANN-r{args.ratio:.2f}+HomographyUSAC-t{args.ransac_threshold_px:g}-P1"
    label_rows: list[plg.PseudoLabelRow] = []
    selected_rows: list[dict[str, object]] = []
    for item in generated:
        selected_candidate = selected_by_pair.get(item.candidate.pair_pt)
        retained = selected_candidate.retained_labels if selected_candidate else 0
        reason = "selected" if selected_candidate else "filtered"
        if selected_candidate is None:
            if (item.candidate.style, item.candidate.gate) not in eligible_groups:
                reason = "excluded_group"
            elif item.candidate.ransac_matches < args.pair_min_inliers:
                reason = "below_min_inliers"
            elif item.candidate.wrong > args.pair_max_wrong:
                reason = "above_wrong_limit"
            elif item.candidate.precision < args.pair_min_precision:
                reason = "below_min_precision"
            else:
                reason = "cap_filtered"
        if retained > 0:
            capped_a, capped_b, capped_errors = plg.cap_matches(
                item.points_a,
                item.points_b,
                item.errors,
                max_matches=retained,
                seed=args.seed + len(label_rows) + 17,
            )
            label_rows.extend(
                plg.rows_from_matches(
                    pair_path=item.pair_rel,
                    points_a=capped_a,
                    points_b=capped_b,
                    errors=capped_errors,
                    matcher=matcher_name,
                    stage="homography_truth_p1_retention",
                    cache_dir=item.cache_rel,
                )
            )
        selected_rows.append(
            {
                "style": item.candidate.style,
                "gate": item.candidate.gate,
                "source_name": item.candidate.source_name,
                "pair_pt": item.candidate.pair_pt,
                "ransac_matches": item.candidate.ransac_matches,
                "truth_filtered_matches": item.candidate.truth_filtered_matches,
                "wrong": item.candidate.wrong,
                "truth_precision": f"{item.candidate.precision:.6f}",
                "retained_labels": retained,
                "selected": int(retained > 0),
                "selection_reason": reason,
            }
        )

    group_rows: list[dict[str, object]] = []
    for style, gate in groups:
        sampled_group = [row for row in sampled_rows if row["style"] == style and row["gate"] == gate]
        generated_group = [row for row in selected_rows if row["style"] == style and row["gate"] == gate]
        selected_group = [row for row in generated_group if int(row["selected"]) == 1]
        group_rows.append(
            {
                "style": style,
                "gate": gate,
                "sampled_pairs": len(sampled_group),
                "gate_zero_pairs": sum(int(row["is_gate_zero"]) for row in sampled_group),
                "generated_pairs": len(generated_group),
                "selected_pairs": len(selected_group),
                "retained_labels": sum(int(row["retained_labels"]) for row in selected_group),
                "min_selected_precision": (
                    f"{min(float(row['truth_precision']) for row in selected_group):.6f}" if selected_group else "0.000000"
                ),
                "max_selected_wrong": max((int(row["wrong"]) for row in selected_group), default=0),
            }
        )

    plg.write_pseudo_label_csv(output_dir / "pseudo_labels.csv", label_rows)
    plg.write_summary_csv(output_dir / "pair_summary.csv", pair_summary)
    write_csv(
        output_dir / "candidate_pairs.csv",
        sampled_rows,
        [
            "style",
            "gate",
            "source_name",
            "pair_pt",
            "pfm_matches",
            "pfm_correct",
            "pfm_precision",
            "is_gate_zero",
            "generation_status",
            "ransac_matches",
            "truth_filtered_matches",
            "truth_precision",
            "wrong",
        ],
    )
    write_csv(
        output_dir / "retained_pair_candidates.csv",
        selected_rows,
        [
            "style",
            "gate",
            "source_name",
            "pair_pt",
            "ransac_matches",
            "truth_filtered_matches",
            "wrong",
            "truth_precision",
            "retained_labels",
            "selected",
            "selection_reason",
        ],
    )
    write_csv(
        output_dir / "group_summary.csv",
        group_rows,
        [
            "style",
            "gate",
            "sampled_pairs",
            "gate_zero_pairs",
            "generated_pairs",
            "selected_pairs",
            "retained_labels",
            "min_selected_precision",
            "max_selected_wrong",
        ],
    )
    summary = [
        "# P1 Viewpoint Retention Labels",
        "",
        "- Split: train only.",
        f"- Groups: `{args.groups}`.",
        f"- Matcher: RootSIFT ratio {args.ratio:.2f} + HomographyUSAC {args.ransac_threshold_px:g}px + warp truth {args.truth_threshold_px:g}px.",
        f"- Pair filter: min inliers {args.pair_min_inliers}, min precision {args.pair_min_precision:.3f}, max wrong {args.pair_max_wrong}.",
        f"- Caps: per pair {args.label_cap_per_pair}, per group {args.label_cap_per_group}, max source pairs {args.max_source_pairs}.",
        f"- Selected pairs: {sum(int(row['selected_pairs']) for row in group_rows)}.",
        f"- Retained labels: {len(label_rows)}.",
        "",
        "| style | gate | sampled | gate-zero | generated | selected | labels | min selected precision | max wrong |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        summary.append(
            f"| {row['style']} | {row['gate']} | {row['sampled_pairs']} | {row['gate_zero_pairs']} | "
            f"{row['generated_pairs']} | {row['selected_pairs']} | {row['retained_labels']} | "
            f"{row['min_selected_precision']} | {row['max_selected_wrong']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"pseudo_labels={output_dir / 'pseudo_labels.csv'}")
    print(f"retained_pair_candidates={output_dir / 'retained_pair_candidates.csv'}")
    print(f"labels={len(label_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
