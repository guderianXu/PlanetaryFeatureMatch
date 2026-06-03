#!/usr/bin/env python3
"""Agent14 Stage12 sidecar: safe matcher-label retention diagnostic.

This script does not train and does not run the matcher. It reads existing
pure-PFM, Stage10, Stage11, train pseudo-label, and negative heatmap artifacts
to decide whether the external fallback evidence can be converted into a safer
PFM distillation/training policy.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent14_stage12"

PURE_ROUTE = PROJECT_ROOT / "runs" / "cross_view_1024_keypointonly_multistate_lowcontrast_targetcontrast_postselect_0step_seed1234"
STAGE10_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent14_stage10"
STAGE11_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent14_stage11"
NEGATIVE_HEATMAP_DIR = (
    PROJECT_ROOT / "runs" / "cross_view_1024_gatezero_r075t2_keypointonly_selectedparams_eval_0step_seed1234"
)
TRAIN_PSEUDO_DIR = PROJECT_ROOT / "runs" / "rootsift_pseudo_labels_gatezero_r075t2_train_sample64_seed1234"

GROUPS = [(style, gate) for style in ("numeric", "timestamp") for gate in ("rotate", "viewpoint", "compound")]
STRICT_ALGORITHM = "RootSIFT-FLANN-r0.75+HomographyUSAC-t2"
SUPPORT_ALGORITHM = "RootSIFT-FLANN-r0.80+HomographyUSAC-t2"

GROUP_POLICY_FIELDS = [
    "style",
    "gate",
    "fixed_test_pairs",
    "fixed_test_zero_pairs",
    "fixed_test_low_support_pairs",
    "fixed_test_pure_matches",
    "fixed_test_pure_correct",
    "fixed_test_pure_precision",
    "full_val_pairs",
    "full_val_zero_pairs",
    "full_val_low_support_pairs",
    "full_val_zero_or_low_pairs",
    "full_val_pure_matches",
    "full_val_pure_correct",
    "full_val_pure_precision",
    "stage10_candidate_pairs",
    "stage10_covered_pairs",
    "stage10_coverage",
    "stage10_fallback_matches",
    "stage10_fallback_correct",
    "stage10_fallback_wrong",
    "stage10_fallback_precision",
    "stage10_mean_pair_matches",
    "stage10_pairs_ge_50_inliers",
    "stage11_candidate_pairs",
    "stage11_covered_pairs",
    "stage11_fallback_matches",
    "stage11_fallback_correct",
    "stage11_fallback_wrong",
    "stage11_fallback_precision",
    "train_sampled_pairs",
    "train_gate_zero_pairs",
    "train_generated_candidate_pairs",
    "train_kept_pairs",
    "train_labels",
    "train_labels_per_kept_pair",
    "negative_fixed_test_matches",
    "negative_fixed_test_correct",
    "negative_fixed_test_precision",
    "negative_full_val_matches",
    "negative_full_val_correct",
    "negative_full_val_precision",
    "current_full_val_compound_precision",
    "recommended_training_action",
    "group_risk",
]

POLICY_FIELDS = [
    "policy_id",
    "policy_type",
    "algorithm",
    "recommend",
    "train_eligible",
    "source_evidence",
    "eligible_groups",
    "excluded_groups",
    "selected_pairs",
    "estimated_retained_labels",
    "estimated_raw_precision",
    "min_group_raw_precision",
    "estimated_wrong_before_truth_filter",
    "label_cap_per_pair",
    "label_cap_per_group",
    "pair_min_inliers",
    "pair_min_precision",
    "pair_max_wrong",
    "max_source_pairs",
    "hard_negative_policy",
    "retention_constraints",
    "validation_gate",
    "reason",
]

EXCLUDED_FIELDS = [
    "scope",
    "style",
    "gate",
    "policy_id",
    "reason",
    "evidence",
    "next_allowed_action",
]

RETAINED_PAIR_FIELDS = [
    "policy_id",
    "style",
    "gate",
    "pair_pt",
    "source_name",
    "pair_name",
    "matches",
    "correct",
    "wrong",
    "precision",
    "mean_error_px",
    "median_error_px",
    "retained_label_cap",
    "retained_labels",
    "selected",
    "selection_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pure-route", type=Path, default=PURE_ROUTE)
    parser.add_argument("--stage10-dir", type=Path, default=STAGE10_DIR)
    parser.add_argument("--stage11-dir", type=Path, default=STAGE11_DIR)
    parser.add_argument("--negative-heatmap-dir", type=Path, default=NEGATIVE_HEATMAP_DIR)
    parser.add_argument("--train-pseudo-dir", type=Path, default=TRAIN_PSEUDO_DIR)
    parser.add_argument("--low-support-threshold", type=int, default=8)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {rel(path)}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{field: format_cell(row.get(field, "")) for field in fields} for row in rows])


def format_cell(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return value


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", "nan", None):
        return default
    return int(float(value))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def precision(correct: int, matches: int) -> float:
    return correct / matches if matches else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


@dataclass(frozen=True)
class PairScore:
    style: str
    gate: str
    pair_pt: str
    source_name: str
    pair_name: str
    matches: int
    correct: int
    wrong: int
    precision: float
    mean_error_px: float
    median_error_px: float


@dataclass(frozen=True)
class RetentionSpec:
    policy_id: str
    policy_type: str
    algorithm: str
    eligible_groups: tuple[tuple[str, str], ...]
    label_cap_per_pair: int
    label_cap_per_group: int
    pair_min_inliers: int
    pair_min_precision: float
    pair_max_wrong: int
    max_source_pairs: int
    recommend: int
    train_eligible: int
    hard_negative_policy: str
    retention_constraints: str
    validation_gate: str
    reason: str


def summarize_pair_rows(rows: list[dict[str, str]], low_support_threshold: int) -> dict[str, object]:
    matches = [as_int(row, "matches") for row in rows]
    correct = [as_int(row, "correct") for row in rows]
    total_matches = sum(matches)
    total_correct = sum(correct)
    low_support = sum(1 for value in matches if 0 < value < low_support_threshold)
    return {
        "pairs": len(rows),
        "zero_pairs": sum(1 for value in matches if value == 0),
        "low_support_pairs": low_support,
        "zero_or_low_pairs": sum(1 for value in matches if value < low_support_threshold),
        "matches": total_matches,
        "correct": total_correct,
        "precision": precision(total_correct, total_matches),
    }


def load_pure_route_summaries(pure_route: Path, stage10_dir: Path, low_support_threshold: int) -> dict[tuple[str, str], dict[str, dict[str, object]]]:
    summaries: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for style, gate in GROUPS:
        fixed_path = pure_route / "eval" / style / gate / "summary.csv"
        full_val_path = stage10_dir / "pure_pfm_fullval" / "eval" / style / gate / "summary.csv"
        summaries[(style, gate)] = {
            "fixed_test": summarize_pair_rows(read_csv(fixed_path), low_support_threshold),
            "full_val": summarize_pair_rows(read_csv(full_val_path), low_support_threshold),
        }
    return summaries


def load_group_rows(path: Path, algorithm: str) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("algorithm") == algorithm and row.get("style") != "overall":
            result[(row["style"], row["gate"])] = row
    missing = [f"{style}/{gate}" for style, gate in GROUPS if (style, gate) not in result]
    if missing:
        raise RuntimeError(f"{rel(path)} is missing {algorithm} rows for: {', '.join(missing)}")
    return result


def load_train_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    result = {(row["style"], row["gate"]): row for row in rows}
    missing = [f"{style}/{gate}" for style, gate in GROUPS if (style, gate) not in result]
    if missing:
        raise RuntimeError(f"{rel(path)} is missing train pseudo rows for: {', '.join(missing)}")
    return result


def load_negative_fixed_test(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    result = {(row["style"], row["gate"]): row for row in rows}
    missing = [f"{style}/{gate}" for style, gate in GROUPS if (style, gate) not in result]
    if missing:
        raise RuntimeError(f"{rel(path)} is missing fixed-test negative rows for: {', '.join(missing)}")
    return result


def load_compound_negative(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["route"].startswith("gatezero_checkpoint"):
            result[(row["style"], row["gate"])] = row
    return result


def load_current_compound(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["route"].startswith("current_route"):
            result[(row["style"], row["gate"])] = row
    return result


def load_pair_scores(path: Path, algorithm: str) -> list[PairScore]:
    rows = read_csv(path)
    pairs: list[PairScore] = []
    for row in rows:
        if row.get("algorithm") != algorithm:
            continue
        if row.get("style") == "overall":
            continue
        pairs.append(
            PairScore(
                style=row["style"],
                gate=row["gate"],
                pair_pt=row["pair_pt"],
                source_name=row.get("source_name", ""),
                pair_name=row.get("pair_name", ""),
                matches=as_int(row, "matches"),
                correct=as_int(row, "correct"),
                wrong=as_int(row, "wrong"),
                precision=as_float(row, "precision"),
                mean_error_px=as_float(row, "mean_error_px", math.nan),
                median_error_px=as_float(row, "median_error_px", math.nan),
            )
        )
    if not pairs:
        raise RuntimeError(f"{rel(path)} has no pair metrics for {algorithm}")
    return pairs


def group_key_text(groups: Iterable[tuple[str, str]]) -> str:
    return ";".join(f"{style}/{gate}" for style, gate in groups)


def build_group_policy_summary(
    *,
    pure_summaries: dict[tuple[str, str], dict[str, dict[str, object]]],
    stage10_groups: dict[tuple[str, str], dict[str, str]],
    stage11_groups: dict[tuple[str, str], dict[str, str]],
    train_groups: dict[tuple[str, str], dict[str, str]],
    negative_fixed: dict[tuple[str, str], dict[str, str]],
    negative_compound: dict[tuple[str, str], dict[str, str]],
    current_compound: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for style, gate in GROUPS:
        key = (style, gate)
        fixed = pure_summaries[key]["fixed_test"]
        full = pure_summaries[key]["full_val"]
        s10 = stage10_groups[key]
        s11 = stage11_groups[key]
        train = train_groups[key]
        neg = negative_fixed[key]
        neg_compound = negative_compound.get(key, {})
        current_comp = current_compound.get(key, {})
        train_kept = as_int(train, "kept_pairs")
        train_labels = as_int(train, "labels")
        s10_precision = as_float(s10, "fallback_precision")
        neg_full_precision = as_float(neg_compound, "precision", math.nan)
        current_full_comp_precision = as_float(current_comp, "precision", math.nan)

        if gate == "compound":
            action = "exclude_from_heatmap; descriptor_or_correspondence_probe_only_with_tiny_cap"
            risk = "high_overactivation"
        elif s10_precision >= 0.995 and as_float(s11, "fallback_precision") >= 0.997:
            action = "eligible_for_tiny_pair_filtered_retention"
            risk = "low"
        else:
            action = "eligible_only_with_pair_filter_and_group_cap"
            risk = "medium"

        rows.append(
            {
                "style": style,
                "gate": gate,
                "fixed_test_pairs": fixed["pairs"],
                "fixed_test_zero_pairs": fixed["zero_pairs"],
                "fixed_test_low_support_pairs": fixed["low_support_pairs"],
                "fixed_test_pure_matches": fixed["matches"],
                "fixed_test_pure_correct": fixed["correct"],
                "fixed_test_pure_precision": fixed["precision"],
                "full_val_pairs": full["pairs"],
                "full_val_zero_pairs": full["zero_pairs"],
                "full_val_low_support_pairs": full["low_support_pairs"],
                "full_val_zero_or_low_pairs": full["zero_or_low_pairs"],
                "full_val_pure_matches": full["matches"],
                "full_val_pure_correct": full["correct"],
                "full_val_pure_precision": full["precision"],
                "stage10_candidate_pairs": as_int(s10, "candidate_pairs"),
                "stage10_covered_pairs": as_int(s10, "covered_pairs"),
                "stage10_coverage": as_float(s10, "coverage"),
                "stage10_fallback_matches": as_int(s10, "fallback_matches"),
                "stage10_fallback_correct": as_int(s10, "fallback_correct"),
                "stage10_fallback_wrong": as_int(s10, "fallback_wrong"),
                "stage10_fallback_precision": s10_precision,
                "stage10_mean_pair_matches": as_float(s10, "mean_matches_per_candidate"),
                "stage10_pairs_ge_50_inliers": as_int(s10, "pairs_ge_50_inliers"),
                "stage11_candidate_pairs": as_int(s11, "candidate_pairs"),
                "stage11_covered_pairs": as_int(s11, "covered_pairs"),
                "stage11_fallback_matches": as_int(s11, "fallback_matches"),
                "stage11_fallback_correct": as_int(s11, "fallback_correct"),
                "stage11_fallback_wrong": as_int(s11, "fallback_wrong"),
                "stage11_fallback_precision": as_float(s11, "fallback_precision"),
                "train_sampled_pairs": as_int(train, "sampled_pairs"),
                "train_gate_zero_pairs": as_int(train, "gate_zero_pairs"),
                "train_generated_candidate_pairs": as_int(train, "generated_candidate_pairs"),
                "train_kept_pairs": train_kept,
                "train_labels": train_labels,
                "train_labels_per_kept_pair": train_labels / train_kept if train_kept else 0.0,
                "negative_fixed_test_matches": as_int(neg, "matches"),
                "negative_fixed_test_correct": as_int(neg, "correct"),
                "negative_fixed_test_precision": as_float(neg, "precision"),
                "negative_full_val_matches": as_int(neg_compound, "matches") if neg_compound else "",
                "negative_full_val_correct": as_int(neg_compound, "correct") if neg_compound else "",
                "negative_full_val_precision": neg_full_precision if neg_compound else "",
                "current_full_val_compound_precision": current_full_comp_precision if current_comp else "",
                "recommended_training_action": action,
                "group_risk": risk,
            }
        )
    return rows


def select_retained_pairs(spec: RetentionSpec, pairs: list[PairScore]) -> tuple[dict[str, object], list[dict[str, object]]]:
    eligible = set(spec.eligible_groups)
    selected_by_group: dict[tuple[str, str], int] = defaultdict(int)
    selected_by_source: dict[tuple[str, str, str], int] = defaultdict(int)
    pair_rows: list[dict[str, object]] = []
    selected_pairs: list[PairScore] = []
    rejected = 0

    ordered = sorted(
        pairs,
        key=lambda item: (
            item.style,
            item.gate,
            -item.precision,
            item.wrong,
            item.mean_error_px if not math.isnan(item.mean_error_px) else 999.0,
            -item.correct,
            item.pair_pt,
        ),
    )
    for pair in ordered:
        group = (pair.style, pair.gate)
        retained = min(pair.correct, spec.label_cap_per_pair)
        selected = False
        reason = "excluded_group"
        if group in eligible:
            if pair.matches < spec.pair_min_inliers:
                reason = "below_min_inliers"
            elif pair.precision < spec.pair_min_precision:
                reason = "below_min_precision"
            elif pair.wrong > spec.pair_max_wrong:
                reason = "too_many_wrong_matches"
            elif selected_by_group[group] + retained > spec.label_cap_per_group:
                reason = "group_label_cap_reached"
            elif selected_by_source[(pair.style, pair.gate, pair.source_name)] >= spec.max_source_pairs:
                reason = "source_pair_cap_reached"
            elif retained <= 0:
                reason = "no_truth_filtered_labels"
            else:
                selected = True
                reason = "selected"
                selected_by_group[group] += retained
                selected_by_source[(pair.style, pair.gate, pair.source_name)] += 1
                selected_pairs.append(pair)
        if not selected:
            rejected += 1
        pair_rows.append(
            {
                "policy_id": spec.policy_id,
                "style": pair.style,
                "gate": pair.gate,
                "pair_pt": pair.pair_pt,
                "source_name": pair.source_name,
                "pair_name": pair.pair_name,
                "matches": pair.matches,
                "correct": pair.correct,
                "wrong": pair.wrong,
                "precision": pair.precision,
                "mean_error_px": pair.mean_error_px,
                "median_error_px": pair.median_error_px,
                "retained_label_cap": spec.label_cap_per_pair,
                "retained_labels": retained if selected else 0,
                "selected": int(selected),
                "selection_reason": reason,
            }
        )

    selected_matches = sum(pair.matches for pair in selected_pairs)
    selected_correct = sum(pair.correct for pair in selected_pairs)
    selected_wrong = sum(pair.wrong for pair in selected_pairs)
    retained_labels = sum(as_int(row, "retained_labels") for row in pair_rows if row["selected"] == 1)
    selected_groups = sorted({(pair.style, pair.gate) for pair in selected_pairs})
    group_precisions = []
    for group in selected_groups:
        group_pairs = [pair for pair in selected_pairs if (pair.style, pair.gate) == group]
        group_matches = sum(pair.matches for pair in group_pairs)
        group_correct = sum(pair.correct for pair in group_pairs)
        group_precisions.append(precision(group_correct, group_matches))

    row = {
        "policy_id": spec.policy_id,
        "policy_type": spec.policy_type,
        "algorithm": spec.algorithm,
        "recommend": spec.recommend,
        "train_eligible": spec.train_eligible,
        "source_evidence": "Stage10 full-val per-pair metrics; Stage11 fixed-test confirmation; negative heatmap full-val guard",
        "eligible_groups": group_key_text(spec.eligible_groups),
        "excluded_groups": group_key_text(group for group in GROUPS if group not in set(spec.eligible_groups)),
        "selected_pairs": len(selected_pairs),
        "estimated_retained_labels": retained_labels,
        "estimated_raw_precision": precision(selected_correct, selected_matches),
        "min_group_raw_precision": min(group_precisions) if group_precisions else 0.0,
        "estimated_wrong_before_truth_filter": selected_wrong,
        "label_cap_per_pair": spec.label_cap_per_pair,
        "label_cap_per_group": spec.label_cap_per_group,
        "pair_min_inliers": spec.pair_min_inliers,
        "pair_min_precision": spec.pair_min_precision,
        "pair_max_wrong": spec.pair_max_wrong,
        "max_source_pairs": spec.max_source_pairs,
        "hard_negative_policy": spec.hard_negative_policy,
        "retention_constraints": spec.retention_constraints,
        "validation_gate": spec.validation_gate,
        "reason": f"{spec.reason}; rejected_pairs={rejected}",
    }
    return row, pair_rows


def build_policy_candidates(pair_scores: list[PairScore]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    specs = [
        RetentionSpec(
            policy_id="P1_r075_pairfiltered_noncompound_viewpoint_tiny",
            policy_type="recommended_training_probe",
            algorithm=STRICT_ALGORITHM,
            eligible_groups=(("numeric", "viewpoint"), ("timestamp", "viewpoint")),
            label_cap_per_pair=12,
            label_cap_per_group=384,
            pair_min_inliers=50,
            pair_min_precision=0.995,
            pair_max_wrong=1,
            max_source_pairs=2,
            recommend=1,
            train_eligible=1,
            hard_negative_policy="for every retained positive point, sample >=3 non-match negatives outside 8px and keep original pure-PFM negatives",
            retention_constraints="truth-filter labels; do not create dense heatmap targets; no zero-match pair receives more than 12 positives; no group exceeds 384 labels",
            validation_gate="must improve or hold full-val precision on both viewpoint groups before any compound probe",
            reason="targets weak viewpoint groups while avoiding compound heatmap failure mode and dense positive floods",
        ),
        RetentionSpec(
            policy_id="P2_r075_pairfiltered_noncompound_all_tiny",
            policy_type="guarded_ablation",
            algorithm=STRICT_ALGORITHM,
            eligible_groups=(("numeric", "rotate"), ("numeric", "viewpoint"), ("timestamp", "rotate"), ("timestamp", "viewpoint")),
            label_cap_per_pair=8,
            label_cap_per_group=256,
            pair_min_inliers=80,
            pair_min_precision=0.998,
            pair_max_wrong=0,
            max_source_pairs=1,
            recommend=0,
            train_eligible=1,
            hard_negative_policy="positive labels must be paired with >=4 local hard negatives and source-balanced negatives",
            retention_constraints="truth-filter labels; no compound groups; cap by source, pair, and group; descriptor/correspondence loss preferred over heatmap-only",
            validation_gate="fixed-test cannot be used alone; require full-val non-regression on all noncompound groups",
            reason="precision-first ablation to test whether any retained positives can be learned without broad activation",
        ),
        RetentionSpec(
            policy_id="P3_r075_compound_microprobe_descriptor_only",
            policy_type="high_risk_microprobe",
            algorithm=STRICT_ALGORITHM,
            eligible_groups=(("numeric", "compound"), ("timestamp", "compound")),
            label_cap_per_pair=4,
            label_cap_per_group=96,
            pair_min_inliers=100,
            pair_min_precision=0.999,
            pair_max_wrong=0,
            max_source_pairs=1,
            recommend=0,
            train_eligible=0,
            hard_negative_policy="mandatory >=5 hard negatives per positive plus retain target-contrast gate; abort on activation growth",
            retention_constraints="descriptor/correspondence-only; no heatmap positives; micro-cap; evaluate compound full-val first",
            validation_gate="only after P1 succeeds; compound full-val precision must not fall below current route",
            reason="compound fallback is externally precise, but prior heatmap distillation collapsed full-val compound precision",
        ),
    ]
    policy_rows: list[dict[str, object]] = []
    retained_rows: list[dict[str, object]] = []
    for spec in specs:
        policy, pair_rows = select_retained_pairs(spec, pair_scores)
        policy_rows.append(policy)
        retained_rows.extend(pair_rows)
    policy_rows.extend(build_forbidden_policy_rows())
    return policy_rows, retained_rows


def build_forbidden_policy_rows() -> list[dict[str, object]]:
    return [
        {
            "policy_id": "F1_broad_all_gatezero_heatmap",
            "policy_type": "forbidden",
            "algorithm": STRICT_ALGORITHM,
            "recommend": 0,
            "train_eligible": 0,
            "source_evidence": "negative heatmap run and compound full-val guard",
            "eligible_groups": group_key_text(GROUPS),
            "excluded_groups": "",
            "selected_pairs": "",
            "estimated_retained_labels": "",
            "estimated_raw_precision": "",
            "min_group_raw_precision": "",
            "estimated_wrong_before_truth_filter": "",
            "label_cap_per_pair": "64",
            "label_cap_per_group": "none",
            "pair_min_inliers": "8",
            "pair_min_precision": "none",
            "pair_max_wrong": "none",
            "max_source_pairs": "none",
            "hard_negative_policy": "none",
            "retention_constraints": "none",
            "validation_gate": "failed: compound full-val precision collapsed",
            "reason": "repeats the rejected broad gate-zero heatmap distillation and over-activates weak compound groups",
        },
        {
            "policy_id": "F2_r080_support_max_training_teacher",
            "policy_type": "forbidden",
            "algorithm": SUPPORT_ALGORITHM,
            "recommend": 0,
            "train_eligible": 0,
            "source_evidence": "Stage10/11 r0.80 is a high-support external baseline, not the safest teacher",
            "eligible_groups": group_key_text(GROUPS),
            "excluded_groups": "",
            "selected_pairs": "",
            "estimated_retained_labels": "",
            "estimated_raw_precision": "",
            "min_group_raw_precision": "",
            "estimated_wrong_before_truth_filter": "",
            "label_cap_per_pair": "none",
            "label_cap_per_group": "none",
            "pair_min_inliers": "none",
            "pair_min_precision": "none",
            "pair_max_wrong": "none",
            "max_source_pairs": "none",
            "hard_negative_policy": "unspecified",
            "retention_constraints": "unspecified",
            "validation_gate": "blocked by lower Stage10/11 precision than r0.75/H2",
            "reason": "higher support increases activation pressure while precision is lower than r0.75/H2",
        },
    ]


def build_exclusions(group_rows: list[dict[str, object]], policy_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in group_rows:
        style = str(row["style"])
        gate = str(row["gate"])
        if gate == "compound":
            rows.append(
                {
                    "scope": "group",
                    "style": style,
                    "gate": gate,
                    "policy_id": "P1_r075_pairfiltered_noncompound_viewpoint_tiny",
                    "reason": "compound excluded from recommended policy",
                    "evidence": (
                        f"negative full-val precision={row['negative_full_val_precision']} vs current="
                        f"{row['current_full_val_compound_precision']}; heatmap over-activation risk"
                    ),
                    "next_allowed_action": "descriptor-only microprobe after noncompound policy succeeds",
                }
            )
        if gate == "rotate":
            rows.append(
                {
                    "scope": "group",
                    "style": style,
                    "gate": gate,
                    "policy_id": "P1_r075_pairfiltered_noncompound_viewpoint_tiny",
                    "reason": "rotate already has strong pure-PFM precision and is not the bottleneck",
                    "evidence": f"fixed_test_pure_precision={row['fixed_test_pure_precision']}; full_val_pure_precision={row['full_val_pure_precision']}",
                    "next_allowed_action": "use only as ablation control with stricter caps",
                }
            )
    for row in policy_rows:
        if str(row.get("policy_type")) == "forbidden":
            rows.append(
                {
                    "scope": "policy",
                    "style": "all",
                    "gate": "all",
                    "policy_id": row["policy_id"],
                    "reason": row["reason"],
                    "evidence": row["validation_gate"],
                    "next_allowed_action": "do not train this policy",
                }
            )
    return rows


def write_summary(
    path: Path,
    *,
    group_rows: list[dict[str, object]],
    policy_rows: list[dict[str, object]],
    retained_rows: list[dict[str, object]],
) -> None:
    recommended = next(row for row in policy_rows if row["policy_id"] == "P1_r075_pairfiltered_noncompound_viewpoint_tiny")
    forbidden = [row for row in policy_rows if row["policy_type"] == "forbidden"]
    total_full_val_pairs = sum(as_int(row, "full_val_pairs") for row in group_rows)
    total_zero = sum(as_int(row, "full_val_zero_pairs") for row in group_rows)
    total_low = sum(as_int(row, "full_val_low_support_pairs") for row in group_rows)
    total_s10_matches = sum(as_int(row, "stage10_fallback_matches") for row in group_rows)
    total_s10_correct = sum(as_int(row, "stage10_fallback_correct") for row in group_rows)
    total_s10_wrong = sum(as_int(row, "stage10_fallback_wrong") for row in group_rows)
    selected_pairs = sum(1 for row in retained_rows if row["policy_id"] == recommended["policy_id"] and row["selected"] == 1)

    lines = [
        "# Matcher Algorithm Iteration Agent14 Stage12",
        "",
        "## Scope",
        "",
        "- This is a sidecar diagnostic only. It does not train, evaluate PFM, run matchers, or modify main training/evaluation code.",
        "- Inputs are existing pure-PFM route summaries, Stage10/Stage11 matcher fallback artifacts, train pseudo-label sample artifacts, and the rejected gate-zero heatmap run.",
        "- Output policy is meant to decide whether a small next training probe is justified without blocking pure-PFM work.",
        "",
        "## Key Validation Facts",
        "",
        f"- Current full-val pure-PFM route has {total_zero}/{total_full_val_pairs} zero-match pairs and {total_low} low-support nonzero pairs with threshold `<8` matches.",
        f"- Stage10 `{STRICT_ALGORITHM}` full-val fallback on zero-match rows: {total_s10_correct}/{total_s10_matches} correct, {total_s10_wrong} wrong, precision {precision(total_s10_correct, total_s10_matches):.6f}.",
        "- Stage11 fixed-test confirms r0.75/H2 is the higher-precision fallback versus the r0.80/H2 support baseline.",
        "- The rejected heatmap checkpoint over-activated compound rows: numeric/compound full-val precision 0.212075 -> 0.084589 and timestamp/compound 0.095238 -> 0.007152.",
        "",
        "## Recommendation",
        "",
        f"- Recommended trainable probe: `{recommended['policy_id']}`.",
        f"- Eligible groups: `{recommended['eligible_groups']}`.",
        f"- Estimated selected validation-backed pairs: {selected_pairs}; retained-label budget: {recommended['estimated_retained_labels']} labels.",
        f"- Raw fallback precision before truth filtering under this policy: {float(recommended['estimated_raw_precision']):.6f}; estimated wrong matches before truth filter: {recommended['estimated_wrong_before_truth_filter']}.",
        "- Use truth-filtered correspondence labels with hard negatives. Do not use dense positive-only heatmap labels.",
        "- Keep caps active: per-pair cap, per-group cap, source cap, pair min-inliers, pair min-precision, and max wrong-match guard.",
        "",
        "## Why Broad Heatmap Distillation Failed",
        "",
        "- Stage10/11 show external fallback is accurate as an inference-side fallback, but that does not mean every fallback point is a safe heatmap positive.",
        "- Gate-zero pairs are exactly where PFM abstained; using many positives on those pairs teaches the heatmap to fire in regions the current route was gating away.",
        "- The failed run used broad gate-zero labels and produced many more full-val compound matches with much lower precision, which is the signature of over-activation rather than better matching.",
        "- The next probe must preserve abstention behavior: small caps, source balance, hard negatives, and validation gates before any compound expansion.",
        "",
        "## Forbidden Strategies",
        "",
    ]
    for row in forbidden:
        lines.append(f"- `{row['policy_id']}`: {row['reason']}.")
    lines.extend(
        [
            "",
            "## Group Summary",
            "",
            "| style | gate | full-val zero | full-val low | Stage10 fallback precision | train labels | action | risk |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in group_rows:
        lines.append(
            f"| {row['style']} | {row['gate']} | {row['full_val_zero_pairs']} | "
            f"{row['full_val_low_support_pairs']} | {float(row['stage10_fallback_precision']):.6f} | "
            f"{row['train_labels']} | {row['recommended_training_action']} | {row['group_risk']} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `policy_candidates.csv`",
            "- `group_policy_summary.csv`",
            "- `excluded_policy_reasons.csv`",
            "- `retained_pair_candidates.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    if output_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError(f"Stage12 output directory must be exactly {rel(OUTPUT_DIR)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pure_route = project_path(args.pure_route)
    stage10_dir = project_path(args.stage10_dir)
    stage11_dir = project_path(args.stage11_dir)
    negative_dir = project_path(args.negative_heatmap_dir)
    train_dir = project_path(args.train_pseudo_dir)

    pure_summaries = load_pure_route_summaries(pure_route, stage10_dir, args.low_support_threshold)
    stage10_groups = load_group_rows(stage10_dir / "per_group_policy_summary.csv", STRICT_ALGORITHM)
    stage11_groups = load_group_rows(stage11_dir / "per_group_policy_summary.csv", STRICT_ALGORITHM)
    train_groups = load_train_summary(train_dir / "group_summary.csv")
    negative_fixed = load_negative_fixed_test(negative_dir / "metrics.csv")
    negative_compound = load_compound_negative(negative_dir / "compound_fullval_comparison.csv")
    current_compound = load_current_compound(negative_dir / "compound_fullval_comparison.csv")
    pair_scores = load_pair_scores(stage10_dir / "per_pair_metrics.csv", STRICT_ALGORITHM)

    group_rows = build_group_policy_summary(
        pure_summaries=pure_summaries,
        stage10_groups=stage10_groups,
        stage11_groups=stage11_groups,
        train_groups=train_groups,
        negative_fixed=negative_fixed,
        negative_compound=negative_compound,
        current_compound=current_compound,
    )
    policy_rows, retained_rows = build_policy_candidates(pair_scores)
    excluded_rows = build_exclusions(group_rows, policy_rows)

    write_csv(output_dir / "group_policy_summary.csv", group_rows, GROUP_POLICY_FIELDS)
    write_csv(output_dir / "policy_candidates.csv", policy_rows, POLICY_FIELDS)
    write_csv(output_dir / "excluded_policy_reasons.csv", excluded_rows, EXCLUDED_FIELDS)
    write_csv(output_dir / "retained_pair_candidates.csv", retained_rows, RETAINED_PAIR_FIELDS)
    write_summary(output_dir / "summary.md", group_rows=group_rows, policy_rows=policy_rows, retained_rows=retained_rows)

    recommended = next(row for row in policy_rows if row["policy_id"] == "P1_r075_pairfiltered_noncompound_viewpoint_tiny")
    print(f"wrote {rel(output_dir)}")
    print(
        "recommended="
        f"{recommended['policy_id']} selected_pairs={recommended['selected_pairs']} "
        f"retained_labels={recommended['estimated_retained_labels']} "
        f"raw_precision={float(recommended['estimated_raw_precision']):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
