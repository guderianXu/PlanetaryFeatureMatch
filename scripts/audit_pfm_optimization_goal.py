#!/usr/bin/env python3
"""Audit current PlanetaryFeatureMatcher optimization-goal evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditItem:
    requirement_id: str
    status: str
    evidence: str
    risk: str


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def has_all(text: str, patterns: list[str]) -> bool:
    return all(pattern in text for pattern in patterns)


def promotion_passed(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(data.get("promote")) and not data.get("failed_reasons")


def _read_json_object(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _audit_active_mainline_validation(path: Path | None) -> AuditItem:
    data = _read_json_object(path)
    if data is None:
        return AuditItem(
            requirement_id="active_mainline.selector_validation",
            status="MISSING",
            evidence="",
            risk="no active mainline validation JSON was provided or it could not be parsed",
        )

    active_score = data.get("active_score")
    active_score = active_score if isinstance(active_score, dict) else {}
    selector_config = active_score.get("selector_config")
    selector_config = selector_config if isinstance(selector_config, dict) else {}
    target_variants = selector_config.get("target_variants")
    target_variants = target_variants if isinstance(target_variants, list) else []
    target_variant_names = {str(variant) for variant in target_variants}
    errors = data.get("errors")
    errors = errors if isinstance(errors, list) else []
    failed_reasons = active_score.get("failed_reasons")
    failed_reasons = failed_reasons if isinstance(failed_reasons, list) else []
    correct_delta = _float_value(active_score, "correct_delta")
    wrong_delta = _float_value(active_score, "wrong_delta")
    precision_delta = _float_value(active_score, "precision_delta")
    required_extremes = {"extreme_02", "extreme_03"}

    failures: list[str] = []
    if not bool(data.get("valid")):
        failures.append("validation_not_valid")
    if errors:
        failures.append("validation_errors_present")
    if not bool(active_score.get("promote")):
        failures.append("active_selector_not_promoted")
    if failed_reasons:
        failures.append("promotion_failed_reasons_present")
    if not bool(active_score.get("regression_guard_clean")):
        failures.append("regression_guard_not_clean")
    if correct_delta <= 0:
        failures.append("no_positive_correct_delta")
    if wrong_delta > 0:
        failures.append("wrong_delta_positive")
    if not required_extremes.issubset(target_variant_names):
        failures.append("missing_extreme_02_or_extreme_03_target")

    active_selector = str(data.get("active_selector", ""))
    active_label = str(data.get("active_label", ""))
    evidence = (
        f"active_selector={active_selector}; "
        f"active_label={active_label}; "
        f"correct_delta={correct_delta:g}; "
        f"wrong_delta={wrong_delta:g}; "
        f"precision_delta={precision_delta:g}; "
        f"regression_guard_clean={bool(active_score.get('regression_guard_clean'))}; "
        f"target_variants={','.join(sorted(target_variant_names))}"
    )
    return AuditItem(
        requirement_id="active_mainline.selector_validation",
        status="PASS" if not failures else "PARTIAL",
        evidence=evidence,
        risk="; ".join(failures),
    )


def _audit_hybrid_lightglue_validation(path: Path | None) -> AuditItem:
    data = _read_json_object(path)
    if data is None:
        return AuditItem(
            requirement_id="hybrid.lightglue_gate_validation",
            status="MISSING",
            evidence="",
            risk="no hybrid LightGlue validation JSON was provided or it could not be parsed",
        )

    errors = data.get("errors")
    errors = errors if isinstance(errors, list) else []
    correct_delta = _float_value(data, "correct_delta_vs_lightglue")
    wrong_delta = _float_value(data, "wrong_delta_vs_lightglue")
    precision_delta = _float_value(data, "precision_delta_vs_lightglue")

    failures: list[str] = []
    if not bool(data.get("valid")):
        failures.append("validation_not_valid")
    if errors:
        failures.append("validation_errors_present")
    if correct_delta <= 0:
        failures.append("no_positive_correct_delta")
    if wrong_delta > 0:
        failures.append("wrong_delta_positive")
    if precision_delta < 0:
        failures.append("precision_delta_negative")

    evidence = (
        f"valid={bool(data.get('valid'))}; "
        f"correct_delta={correct_delta:g}; "
        f"wrong_delta={wrong_delta:g}; "
        f"precision_delta={precision_delta:g}"
    )
    return AuditItem(
        requirement_id="hybrid.lightglue_gate_validation",
        status="PASS" if not failures else "PARTIAL",
        evidence=evidence,
        risk="; ".join(failures),
    )


def _selector_summary_payload(data: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    aggregate = data.get("aggregate")
    if isinstance(aggregate, dict):
        by_split = data.get("by_split")
        return aggregate, by_split if isinstance(by_split, dict) else {}

    comparison = data.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    selector = comparison.get("selector")
    if isinstance(selector, dict):
        by_split = comparison.get("selector_by_split")
        return selector, by_split if isinstance(by_split, dict) else {}

    selector = data.get("selector")
    if isinstance(selector, dict):
        by_split = data.get("selector_by_split")
        return selector, by_split if isinstance(by_split, dict) else {}

    return {}, {}


def _audit_true_geometry_selector_validation(
    selector_summary_json: Path | None,
    manifest_validation_json: Path | None,
) -> AuditItem:
    data = _read_json_object(selector_summary_json)
    if data is None:
        return AuditItem(
            requirement_id="true_geometry.selector_fresh_validation",
            status="MISSING",
            evidence="",
            risk="no true-geometry selector summary JSON was provided or it could not be parsed",
        )

    aggregate, by_split = _selector_summary_payload(data)
    if not aggregate:
        return AuditItem(
            requirement_id="true_geometry.selector_fresh_validation",
            status="MISSING",
            evidence="",
            risk="true-geometry selector summary does not contain aggregate selector metrics",
        )

    manifest = _read_json_object(manifest_validation_json)
    if manifest is None:
        embedded_manifest = data.get("manifest_validation")
        manifest = embedded_manifest if isinstance(embedded_manifest, dict) else None

    rows = _float_value(aggregate, "rows")
    selected_matches = _float_value(aggregate, "selected_matches", _float_value(aggregate, "pfm_matches"))
    selected_correct = _float_value(aggregate, "selected_correct", _float_value(aggregate, "pfm_correct"))
    selected_wrong = _float_value(aggregate, "selected_wrong", _float_value(aggregate, "pfm_wrong"))
    selected_precision = _float_value(
        aggregate,
        "selected_precision",
        selected_correct / selected_matches if selected_matches else 0.0,
    )
    lightglue_correct = _float_value(aggregate, "lightglue_correct")
    lightglue_wrong = _float_value(aggregate, "lightglue_wrong")
    lightglue_matches = _float_value(aggregate, "lightglue_matches")
    lightglue_precision = _float_value(
        aggregate,
        "lightglue_precision",
        lightglue_correct / lightglue_matches if lightglue_matches else 0.0,
    )
    correct_delta = _float_value(aggregate, "correct_delta_vs_lightglue", selected_correct - lightglue_correct)
    wrong_delta = _float_value(aggregate, "wrong_delta_vs_lightglue", selected_wrong - lightglue_wrong)

    failures: list[str] = []
    if rows <= 0:
        failures.append("no_selector_rows")
    if selected_correct <= 0:
        failures.append("no_selected_correct_matches")
    if correct_delta <= 0:
        failures.append("no_positive_correct_delta")
    if wrong_delta > 0:
        failures.append("wrong_delta_positive")
    if selected_precision < lightglue_precision:
        failures.append("precision_below_lightglue")
    if not by_split:
        failures.append("missing_by_split_metrics")
    for split, split_value in by_split.items():
        split_metrics = split_value if isinstance(split_value, dict) else {}
        if _float_value(split_metrics, "rows") <= 0:
            failures.append(f"{split}_has_no_rows")
        if _float_value(split_metrics, "correct_delta_vs_lightglue") <= 0:
            failures.append(f"{split}_no_positive_correct_delta")
        if _float_value(split_metrics, "wrong_delta_vs_lightglue") > 0:
            failures.append(f"{split}_wrong_delta_positive")

    base_disjoint = None
    counts: dict[str, object] = {}
    excluded_base_ids = 0.0
    if manifest is None:
        failures.append("missing_fresh_manifest_validation")
    else:
        base_disjoint = bool(manifest.get("base_disjoint"))
        if not base_disjoint:
            failures.append("fresh_manifest_not_base_disjoint")
        raw_counts = manifest.get("counts")
        counts = raw_counts if isinstance(raw_counts, dict) else {}
        for split in ("dev", "val", "lockbox"):
            if _float_value(counts, split) <= 0:
                failures.append(f"fresh_manifest_missing_{split}")
        excluded_base_ids = _float_value(manifest, "excluded_base_ids")

    count_text = ",".join(f"{key}={_float_value(counts, key):g}" for key in sorted(counts))
    split_text = ",".join(str(key) for key in sorted(by_split))
    evidence = (
        f"rows={rows:g}; "
        f"selected_correct={selected_correct:g}; "
        f"selected_wrong={selected_wrong:g}; "
        f"lightglue_correct={lightglue_correct:g}; "
        f"lightglue_wrong={lightglue_wrong:g}; "
        f"correct_delta={correct_delta:g}; "
        f"wrong_delta={wrong_delta:g}; "
        f"selected_precision={selected_precision:.6g}; "
        f"lightglue_precision={lightglue_precision:.6g}; "
        f"base_disjoint={base_disjoint}; "
        f"excluded_base_ids={excluded_base_ids:g}; "
        f"counts={count_text}; "
        f"splits={split_text}"
    )
    return AuditItem(
        requirement_id="true_geometry.selector_fresh_validation",
        status="PASS" if not failures else "PARTIAL",
        evidence=evidence,
        risk="; ".join(failures),
    )


def _float_value(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value if value not in ("", None) else default)
    except (TypeError, ValueError):
        return default


def _last_csv_row(path: Path | None) -> dict[str, str] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def audit_goal(
    *,
    project_root: Path,
    selector_promotion_json: Path | None = None,
    active_mainline_validation_json: Path | None = None,
    hybrid_lightglue_validation_json: Path | None = None,
    true_geometry_selector_summary_json: Path | None = None,
    true_geometry_manifest_validation_json: Path | None = None,
    train_metrics_csv: Path | None = None,
) -> list[AuditItem]:
    bench = read_text(project_root / "scripts" / "benchmark_lazy_pose_pairs.py")
    train = read_text(project_root / "python" / "pfm_pytorch_training.py")
    model = read_text(project_root / "python" / "pfm_model.py")
    selector = read_text(project_root / "scripts" / "run_dual_checkpoint_rescue_eval.py")
    promotion_pipeline = read_text(project_root / "scripts" / "run_fov76_checkpoint_promotion_pipeline.py")
    hard_mining = read_text(project_root / "scripts" / "mine_hard_failure_pairs.py")
    rescue_gain = read_text(project_root / "scripts" / "build_rescue_gain_hard_set.py")
    balanced_manifest = read_text(project_root / "scripts" / "build_phase2j_balanced_manifest.py")

    items: list[AuditItem] = []

    items.append(_audit_active_mainline_validation(active_mainline_validation_json))
    items.append(_audit_hybrid_lightglue_validation(hybrid_lightglue_validation_json))
    items.append(
        _audit_true_geometry_selector_validation(
            true_geometry_selector_summary_json,
            true_geometry_manifest_validation_json,
        )
    )

    checkpoint_patterns = [
        "best_by_match_score_pytorch_pfm_state.pt",
        "best_by_recall_pytorch_pfm_state.pt",
        "best_by_ransac_inlier_pytorch_pfm_state.pt",
        "best_by_extreme_score_pytorch_pfm_state.pt",
        "last_good_pytorch_pfm_state.pt",
    ]
    items.append(
        AuditItem(
            requirement_id="phase0.checkpoints",
            status="PASS" if has_all(bench, checkpoint_patterns) else "MISSING",
            evidence=", ".join(pattern for pattern in checkpoint_patterns if pattern in bench),
            risk="" if has_all(bench, checkpoint_patterns) else "checkpoint save targets are not all visible in training entry",
        )
    )

    stability_patterns = [
        "stability-auto-recovery",
        "true_match_rejected_by_dustbin_ratio",
        "positive_vs_dustbin_margin_mean",
        "visual_RANSAC_inlier_count",
        "visual_extreme_RANSAC_inlier_count",
    ]
    items.append(
        AuditItem(
            requirement_id="phase0.stability_visual_logging",
            status="PASS" if has_all(bench, stability_patterns) else "MISSING",
            evidence=", ".join(pattern for pattern in stability_patterns if pattern in bench),
            risk="" if has_all(bench, stability_patterns) else "stability or visual logging evidence is incomplete",
        )
    )

    matcher_patterns = [
        "true_match_in_topk@64",
        "true_match_in_topk@256",
        "positive_vs_dustbin_margin_mean",
        "matcher_candidate_topk",
        "no_match_prior_weight",
    ]
    calibrated_default = "graph_matcher_metadata_mode" in train and "calibrated" in train
    reliability_off = (
        "matcher_reliability_pair_bias" in bench
        and "matcher_reliability_dustbin_bias" in bench
        and '"off"' in bench
    )
    matcher_pass = has_all(bench, matcher_patterns) and calibrated_default and reliability_off
    items.append(
        AuditItem(
            requirement_id="phase1.matcher_calibration",
            status="PASS" if matcher_pass else "PARTIAL",
            evidence=(
                ", ".join(pattern for pattern in matcher_patterns if pattern in bench)
                + f"; calibrated_default={calibrated_default}; reliability_off={reliability_off}"
            ),
            risk="" if matcher_pass else "mainline calibration evidence is incomplete",
        )
    )

    hard_mining_patterns = [
        "only_extreme_variants",
        "extreme_02",
        "extreme_03",
        "low_match_count",
        "low_precision",
        "high_false",
        "high_loss",
    ]
    rescue_gain_patterns = [
        "rescue_false_negative",
        "extreme_02",
        "extreme_03",
        "delta_correct",
    ]
    balanced_patterns = [
        "phase2j_bucket",
        "protected",
        "extreme_02",
        "extreme_03",
    ]
    hard_mining_pass = has_all(hard_mining, hard_mining_patterns)
    rescue_gain_pass = has_all(rescue_gain, rescue_gain_patterns)
    balanced_pass = has_all(balanced_manifest, balanced_patterns)
    phase2_hard_status = "PASS" if hard_mining_pass and rescue_gain_pass and balanced_pass else "PARTIAL"
    if not (hard_mining or rescue_gain or balanced_manifest):
        phase2_hard_status = "MISSING"
    items.append(
        AuditItem(
            requirement_id="phase2.extreme_hard_mining",
            status=phase2_hard_status,
            evidence=(
                "extreme_02/extreme_03; "
                f"failure_buckets={hard_mining_pass}; "
                f"rescue_false_negative={rescue_gain_pass}; "
                f"protected_balanced_manifest={balanced_pass}"
            ),
            risk="" if phase2_hard_status == "PASS" else "extreme-only hard mining or protected replay evidence is incomplete",
        )
    )

    loss_text = bench + "\n" + train
    matcher_only = (
        "--train-graph-calibration-only" in bench
        and "train_graph_calibration_only" in train
        and "accept_head" in train
        and "geometry_bias" in train
    )
    true_vs_dustbin_loss = (
        "graph_matcher_true_match_margin_weight" in loss_text
        and "graph_matcher_positive_dustbin_margin_weight" in loss_text
    )
    false_negative_penalty = (
        "graph_matcher_final_false_match_weight" in loss_text
        or "graph_matcher_mined_false_match_weight" in loss_text
        or "graph_matcher_raw_false_match_weight" in loss_text
    )
    rejection_not_strengthened = (
        "no_match_prior_weight" in loss_text
        and ("0.0" in loss_text or "0.000000" in loss_text)
        and "matcher_reliability_pair_bias" in bench
        and "matcher_reliability_dustbin_bias" in bench
        and '"off"' in bench
    )
    ransac_consistency_loss = (
        "ransac_consistency_loss" in loss_text
        or "RANSAC consistency loss" in loss_text
        or "graph_matcher_ransac" in loss_text
    )
    phase2_loss_core = matcher_only and true_vs_dustbin_loss and false_negative_penalty and rejection_not_strengthened
    if phase2_loss_core and ransac_consistency_loss:
        phase2_loss_status = "PASS"
    elif any([matcher_only, true_vs_dustbin_loss, false_negative_penalty, rejection_not_strengthened]):
        phase2_loss_status = "PARTIAL"
    else:
        phase2_loss_status = "MISSING"
    phase2_loss_risk = ""
    if not ransac_consistency_loss:
        phase2_loss_risk = "RANSAC consistency loss is not proven in the training loss path"
    elif phase2_loss_status != "PASS":
        phase2_loss_risk = "matcher-only or non-rejection loss constraints are incomplete"
    items.append(
        AuditItem(
            requirement_id="phase2.matcher_only_loss_constraints",
            status=phase2_loss_status,
            evidence=(
                f"matcher_only={matcher_only}; "
                f"true_vs_dustbin_loss={true_vs_dustbin_loss}; "
                f"false_negative_penalty={false_negative_penalty}; "
                f"rejection_not_strengthened={rejection_not_strengthened}; "
                f"ransac_consistency_loss={ransac_consistency_loss}"
            ),
            risk=phase2_loss_risk,
        )
    )

    legacy_shortcut = "matchability - 0.5" in model and "no_match_prior - 0.5" in model
    calibrated_removes_reliability = (
        "if mode == \"calibrated\"" in train
        and ("[:, 12:13]" in train or "[:, 12]" in train)
        and ("[:, 14 : min" in train or "[:, 14:16]" in train or "[:, 14]" in train)
    )
    reliability_status = "PASS" if calibrated_removes_reliability and not legacy_shortcut else "PARTIAL"
    items.append(
        AuditItem(
            requirement_id="phase3.reliability_decoupling",
            status=reliability_status,
            evidence=(
                f"calibrated_removes_reliability={calibrated_removes_reliability}; "
                f"legacy_full_shortcut_present={legacy_shortcut}"
            ),
            risk=(
                "legacy full reliability shortcut still exists for non-mainline compatibility"
                if legacy_shortcut
                else ""
            ),
        )
    )

    geometry_patterns = [
        "descriptor_geometry_safety_schedule",
        "descriptor_geometry_blend_weight",
        "descriptor_geometry_safety_for_progress",
    ]
    geometry_text = bench + "\n" + model
    items.append(
        AuditItem(
            requirement_id="phase4.geometry_pooling_safety",
            status="PASS" if has_all(geometry_text, geometry_patterns) else "MISSING",
            evidence=", ".join(pattern for pattern in geometry_patterns if pattern in geometry_text),
            risk="" if has_all(geometry_text, geometry_patterns) else "descriptor geometry safety schedule evidence missing",
        )
    )

    selector_pass = (
        "SelectorConfig" in selector
        and "--dual-checkpoint-rescue-selector" in promotion_pipeline
        and promotion_passed(selector_promotion_json)
    )
    items.append(
        AuditItem(
            requirement_id="selector.dual_checkpoint",
            status="PASS" if selector_pass else "PARTIAL",
            evidence=(
                f"selector_script={bool(selector)}; pipeline_flag={'--dual-checkpoint-rescue-selector' in promotion_pipeline}; "
                f"promotion_passed={promotion_passed(selector_promotion_json)}"
            ),
            risk="" if selector_pass else "selector is not fully proven by a passing promotion decision",
        )
    )

    protected_gate = (
        "regression_guard" in promotion_pipeline
        and "extreme_gain" in promotion_pipeline
        and "--formal-protected-variants" in promotion_pipeline
        and "--max-protected-variant-precision-drop" in promotion_pipeline
        and "--max-formal-target-precision-drop" in promotion_pipeline
    )
    phase5_pass = protected_gate and promotion_passed(selector_promotion_json)
    items.append(
        AuditItem(
            requirement_id="phase5.extreme_accuracy_guard",
            status="PASS" if phase5_pass else "PARTIAL",
            evidence=(
                f"regression_guard={protected_gate and 'regression_guard' in promotion_pipeline}; "
                f"extreme_gain={protected_gate and 'extreme_gain' in promotion_pipeline}; "
                f"protected_variant_gate={protected_gate}; "
                f"promotion_passed={promotion_passed(selector_promotion_json)}"
            ),
            risk="" if phase5_pass else "extreme boost is not fully proven by protected promotion gates",
        )
    )

    metrics_row = _last_csv_row(train_metrics_csv)
    if metrics_row is None:
        items.append(
            AuditItem(
                requirement_id="success.training_metrics",
                status="MISSING",
                evidence="",
                risk="no train_metrics.csv was provided for runtime success criteria",
            )
        )
    else:
        rejected = _float_value(metrics_row, "true_match_rejected_by_dustbin_ratio", default=1.0)
        margin = _float_value(metrics_row, "positive_vs_dustbin_margin_mean", default=-1.0)
        filtered = _float_value(metrics_row, "visual_num_filtered_matches", default=0.0)
        extreme_filtered = _float_value(metrics_row, "visual_extreme_num_filtered_matches", default=0.0)
        ransac = _float_value(metrics_row, "visual_RANSAC_inlier_count", default=0.0)
        extreme_ransac = _float_value(metrics_row, "visual_extreme_RANSAC_inlier_count", default=0.0)
        passed = (
            rejected < 0.25
            and margin > 0.0
            and filtered > 0.0
            and extreme_filtered > 0.0
            and ransac > 0.0
            and extreme_ransac > 0.0
        )
        items.append(
            AuditItem(
                requirement_id="success.training_metrics",
                status="PASS" if passed else "PARTIAL",
                evidence=(
                    f"true_match_rejected_by_dustbin_ratio={rejected:.6g}; "
                    f"positive_vs_dustbin_margin_mean={margin:.6g}; "
                    f"visual_num_filtered_matches={filtered:.6g}; "
                    f"visual_extreme_num_filtered_matches={extreme_filtered:.6g}; "
                    f"visual_RANSAC_inlier_count={ransac:.6g}; "
                    f"visual_extreme_RANSAC_inlier_count={extreme_ransac:.6g}"
                ),
                risk="" if passed else "runtime metrics do not fully satisfy success thresholds",
            )
        )

    return items


def write_json(path: Path, items: list[AuditItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_html(path: Path, items: list[AuditItem]) -> None:
    rows = []
    for item in items:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.requirement_id)}</td>"
            f"<td>{html.escape(item.status)}</td>"
            f"<td>{html.escape(item.evidence)}</td>"
            f"<td>{html.escape(item.risk)}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>PFM optimization goal audit</title>
  <style>
    body {{ font-family: sans-serif; line-height: 1.45; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
  </style>
</head>
<body>
  <h1>PFM optimization goal audit</h1>
  <p>该报告只审计当前代码和已生成 promotion 证据，不代表目标已完全完成。</p>
  <table>
    <thead><tr><th>requirement</th><th>status</th><th>evidence</th><th>risk</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--selector-promotion-json", type=Path, default=None)
    parser.add_argument("--active-mainline-validation-json", type=Path, default=None)
    parser.add_argument("--hybrid-lightglue-validation-json", type=Path, default=None)
    parser.add_argument("--true-geometry-selector-summary-json", type=Path, default=None)
    parser.add_argument("--true-geometry-manifest-validation-json", type=Path, default=None)
    parser.add_argument("--train-metrics-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = audit_goal(
        project_root=args.project_root,
        selector_promotion_json=args.selector_promotion_json,
        active_mainline_validation_json=args.active_mainline_validation_json,
        hybrid_lightglue_validation_json=args.hybrid_lightglue_validation_json,
        true_geometry_selector_summary_json=args.true_geometry_selector_summary_json,
        true_geometry_manifest_validation_json=args.true_geometry_manifest_validation_json,
        train_metrics_csv=args.train_metrics_csv,
    )
    write_json(args.output_json, items)
    write_html(args.output_html, items)
    for item in items:
        print(f"{item.requirement_id}: {item.status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
