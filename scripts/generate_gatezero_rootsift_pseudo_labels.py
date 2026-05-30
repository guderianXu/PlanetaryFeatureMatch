#!/usr/bin/env python3
"""Generate train-split RootSIFT pseudo labels for current-route zero-match pairs."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import pfm_model  # noqa: E402
import pseudo_label_generation as plg  # noqa: E402
import pytorch_cache_match_eval as eval_py  # noqa: E402
from patch_descriptor_training import discover_pair_archives  # noqa: E402


STYLES = ("numeric", "timestamp")
GATES = ("rotate", "viewpoint", "compound")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("runs/cross_view_1024_keypointonly_multistate_stylespecific_guard_calib_0step_seed1234/splits"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--selected-weights",
        type=Path,
        default=Path(
            "runs/cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234/"
            "calibration/selected_weights.csv"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--groups",
        default="numeric/rotate,numeric/viewpoint,numeric/compound,timestamp/rotate,timestamp/viewpoint,timestamp/compound",
    )
    parser.add_argument("--sample-pairs-per-group", type=int, default=64)
    parser.add_argument("--max-zero-pairs-per-group", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-threshold-px", type=float, default=2.0)
    parser.add_argument("--truth-threshold-px", type=float, default=3.0)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--sift-max-keypoints", type=int, default=2048)
    parser.add_argument("--sift-max-raw-matches", type=int, default=512)
    parser.add_argument("--max-labels-per-pair", type=int, default=64)
    parser.add_argument("--min-labels-per-pair", type=int, default=8)
    parser.add_argument("--pfm-max-keypoints", type=int, default=4096)
    parser.add_argument("--pfm-descriptor-topk", type=int, default=32)
    parser.add_argument("--pfm-geometry-filter", default="local", choices=["none", "local", "affine"])
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_to_project(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def read_route_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with project_path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["style"], row["gate"]): row for row in rows}


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


def rewrite_rows(
    rows: list[plg.PseudoLabelRow],
    *,
    pair_rel: Path,
    cache_rel: Path,
    matcher: str,
) -> list[plg.PseudoLabelRow]:
    return [
        plg.PseudoLabelRow(
            pair_pt=pair_rel.as_posix(),
            ax=row.ax,
            ay=row.ay,
            bx=row.bx,
            by=row.by,
            matcher=matcher,
            stage=row.stage,
            error_px=row.error_px,
            cache_dir=cache_rel.as_posix(),
        )
        for row in rows
    ]


def write_candidate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "style",
        "gate",
        "pair_pt",
        "pfm_matches",
        "pfm_correct",
        "pfm_precision",
        "is_gate_zero",
        "labels",
        "label_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.sample_pairs_per_group <= 0:
        raise ValueError("--sample-pairs-per-group must be positive")
    if args.max_zero_pairs_per_group <= 0:
        raise ValueError("--max-zero-pairs-per-group must be positive")
    if args.min_labels_per_pair < 0:
        raise ValueError("--min-labels-per-pair must be non-negative")

    groups = parse_group_keys(args.groups)
    route_rows = read_route_rows(args.selected_weights)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_model = load_model_cache(args.device)
    device = torch.device(args.device)

    candidate_rows: list[dict[str, str]] = []
    label_rows: list[plg.PseudoLabelRow] = []
    summary_rows: list[plg.PairSummaryRow] = []
    group_summary_rows: list[dict[str, str]] = []

    for group_index, (style, gate) in enumerate(groups, start=1):
        route = route_rows[(style, gate)]
        cache_dir = project_path(args.split_root) / args.split / style / gate
        cache_rel = relative_to_project(cache_dir)
        pair_paths = eval_py.limit_pair_paths(
            discover_pair_archives([cache_dir], limit_pairs=0, exclude_self_pairs=True),
            limit_pairs=args.sample_pairs_per_group,
            sample_seed=args.seed + group_index * 1009,
        )
        model = load_model(Path(route["pytorch_state"]))
        zero_seen = 0
        kept_pairs = 0
        group_labels = 0
        group_candidates = 0
        for pair_index, pair_path in enumerate(pair_paths, start=1):
            pair_rel = relative_to_project(pair_path)
            result = eval_py.evaluate_pair_path(
                model,
                pair_path,
                device=device,
                mode="blend",
                texture_blend_weight=float(route["texture_blend_weight"]),
                max_keypoints=args.pfm_max_keypoints,
                min_intensity=0.01,
                texture_fraction=1.0,
                threshold_px=5.0,
                topk=args.pfm_descriptor_topk,
                max_matches=512,
                min_score=-1.0,
                min_margin=float(route["min_margin"]),
                min_target_gradient=float(route.get("min_target_gradient") or 0.0),
                min_target_local_contrast=float(route.get("min_target_local_contrast") or 0.0),
                mutual=True,
                geometry_filter=args.pfm_geometry_filter,
                keypoint_spatial_bins=0,
                keypoint_score_mode=route["keypoint_score_mode"],
            )
            is_zero = int(result.matches == 0)
            label_status = "not_gate_zero"
            labels = 0
            if is_zero:
                if zero_seen >= args.max_zero_pairs_per_group:
                    label_status = "not_selected_max_zero"
                else:
                    zero_seen += 1
                    group_candidates += 1
            if is_zero and label_status != "not_selected_max_zero":
                try:
                    rows, summary = plg.generate_for_pair(
                        cache_rel,
                        pair_path,
                        max_keypoints=args.sift_max_keypoints,
                        max_raw_matches=args.sift_max_raw_matches,
                        ratio=args.ratio,
                        sift_contrast=args.sift_contrast,
                        ransac_threshold_px=args.ransac_threshold_px,
                        truth_threshold_px=args.truth_threshold_px,
                        max_labels_per_pair=args.max_labels_per_pair,
                        seed=args.seed + group_index * 100000 + pair_index,
                    )
                    labels = len(rows)
                    if labels >= args.min_labels_per_pair:
                        matcher = f"RootSIFT-FLANN-r{args.ratio:.2f}+HomographyUSAC-t{args.ransac_threshold_px:g}"
                        rewritten = rewrite_rows(rows, pair_rel=pair_rel, cache_rel=cache_rel, matcher=matcher)
                        label_rows.extend(rewritten)
                        kept_pairs += 1
                        group_labels += len(rewritten)
                        label_status = "ok"
                    else:
                        label_status = "too_few_labels"
                    summary_rows.append(
                        plg.PairSummaryRow(
                            cache_dir=cache_rel.as_posix(),
                            pair_pt=pair_rel.as_posix(),
                            status=label_status,
                            keypoints_a=summary.keypoints_a,
                            keypoints_b=summary.keypoints_b,
                            raw_matches=summary.raw_matches,
                            ransac_matches=summary.ransac_matches,
                            truth_filtered_matches=summary.truth_filtered_matches,
                            mean_error_px=summary.mean_error_px,
                            message=summary.message,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - keep batch generation moving.
                    label_status = "error"
                    summary_rows.append(
                        plg.PairSummaryRow(
                            cache_dir=cache_rel.as_posix(),
                            pair_pt=pair_rel.as_posix(),
                            status="error",
                            keypoints_a=0,
                            keypoints_b=0,
                            raw_matches=0,
                            ransac_matches=0,
                            truth_filtered_matches=0,
                            mean_error_px=math.nan,
                            message=str(exc),
                        )
                    )
            candidate_rows.append(
                {
                    "style": style,
                    "gate": gate,
                    "pair_pt": pair_rel.as_posix(),
                    "pfm_matches": str(result.matches),
                    "pfm_correct": str(result.correct),
                    "pfm_precision": f"{result.precision:.6f}",
                    "is_gate_zero": str(is_zero),
                    "labels": str(labels),
                    "label_status": label_status,
                }
            )
            print(
                f"{style}/{gate} {pair_index:03d}/{len(pair_paths):03d} "
                f"pfm={result.correct}/{result.matches} zero={is_zero} "
                f"labels={labels} status={label_status}",
                flush=True,
            )
        zero_total = sum(1 for row in candidate_rows if row["style"] == style and row["gate"] == gate and row["is_gate_zero"] == "1")
        group_summary_rows.append(
            {
                "style": style,
                "gate": gate,
                "sampled_pairs": str(len(pair_paths)),
                "gate_zero_pairs": str(zero_total),
                "generated_candidate_pairs": str(group_candidates),
                "kept_pairs": str(kept_pairs),
                "labels": str(group_labels),
            }
        )

    plg.write_pseudo_label_csv(output_dir / "pseudo_labels.csv", label_rows)
    plg.write_summary_csv(output_dir / "pair_summary.csv", summary_rows)
    write_candidate_csv(output_dir / "candidate_pairs.csv", candidate_rows)
    with (output_dir / "group_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["style", "gate", "sampled_pairs", "gate_zero_pairs", "generated_candidate_pairs", "kept_pairs", "labels"],
        )
        writer.writeheader()
        writer.writerows(group_summary_rows)

    total_candidates = sum(int(row["generated_candidate_pairs"]) for row in group_summary_rows)
    total_kept = sum(int(row["kept_pairs"]) for row in group_summary_rows)
    summary = [
        "# Gate-Zero RootSIFT Pseudo Labels",
        "",
        f"- Split: `{args.split}`",
        f"- Groups: `{args.groups}`",
        f"- Sample pairs per group: {args.sample_pairs_per_group}",
        f"- Max zero pairs per group: {args.max_zero_pairs_per_group}",
        f"- Matcher: RootSIFT-FLANN ratio {args.ratio:.2f} + HomographyUSAC threshold {args.ransac_threshold_px:g}px + warp truth {args.truth_threshold_px:g}px",
        f"- Kept pairs: {total_kept}/{total_candidates}",
        f"- Labels: {len(label_rows)}",
        "",
        "| style | gate | sampled | gate_zero | generated | kept | labels |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_summary_rows:
        summary.append(
            f"| {row['style']} | {row['gate']} | {row['sampled_pairs']} | {row['gate_zero_pairs']} | "
            f"{row['generated_candidate_pairs']} | {row['kept_pairs']} | {row['labels']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"pseudo_labels={output_dir / 'pseudo_labels.csv'}")
    print(f"candidate_pairs={output_dir / 'candidate_pairs.csv'}")
    print(f"kept_pairs={total_kept} candidate_pairs={total_candidates} labels={len(label_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
