import ast
import csv
import importlib
import json
import unittest

import numpy as np
import torch

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from patch_descriptor_training import SyntheticPair
from continuous_rotation_stress_eval import rotate_pair_from_view
from illumination_stress_eval import make_illumination_variants
from benchmark_lazy_pose_pairs import (
    CropWindow,
    LazyPairSpec,
    LazyPairResult,
    PAIR_TYPE_CROSS_CAMERA,
    RenderRecord,
    write_pair_spec_manifest,
)
import visualize_lazy_pose_matches as visual_mod
import training_visual_report as training_report_mod
import run_graph_depth_ablation as depth_ablation_mod
import run_graph_filter_sweep as filter_sweep_mod
import mine_hard_failure_pairs as hard_mine_mod
import mine_pair_delta_regression_pairs as delta_mine_mod
import build_rescue_gain_hard_set as rescue_gain_mod
import evaluate_checkpoint_promotion as promotion_mod
import run_fov76_checkpoint_promotion_pipeline as fov76_gate_mod
import analyze_rescue_candidates as rescue_mod
from visualize_lazy_pose_matches import (
    LazyMatchVisual,
    filter_visual_matches,
    make_illumination_stress_lazy_results,
    selected_draw_indices,
)


class StressEvalScriptsTest(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def dual_rescue_module(self):
        try:
            return importlib.import_module("run_dual_checkpoint_rescue_eval")
        except ModuleNotFoundError as exc:
            self.fail(f"missing dual-checkpoint rescue selector module: {exc}")

    def goal_audit_module(self):
        try:
            return importlib.import_module("audit_pfm_optimization_goal")
        except ModuleNotFoundError as exc:
            self.fail(f"missing optimization goal audit module: {exc}")

    def active_selector_validator_module(self):
        try:
            return importlib.import_module("validate_fov76_active_selector")
        except ModuleNotFoundError as exc:
            self.fail(f"missing fov76 active selector validator module: {exc}")

    def selector_disagreement_module(self):
        try:
            return importlib.import_module("mine_selector_disagreement_pairs")
        except ModuleNotFoundError as exc:
            self.fail(f"missing selector disagreement mining module: {exc}")

    def dual_rescue_rows(self) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        field_defaults = {
            "label": "filtered",
            "split": "test",
            "valid_fraction": 1.0,
            "precision": 1.0,
            "homography_residual_valid": 1,
            "median_error_px": 0.0,
        }

        def row(
            base_id: str,
            variant: str,
            *,
            matches: int,
            correct: int,
            wrong: int,
            score_mean: float,
            h_median: float,
            h_p90: float,
        ) -> dict[str, object]:
            precision = float(correct) / float(matches) if matches else 0.0
            return {
                **field_defaults,
                "base_id": base_id,
                "target_variant": variant,
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": precision,
                "score_mean": score_mean,
                "homography_residual_median_px": h_median,
                "homography_residual_p90_px": h_p90,
            }

        baseline = [
            row("pair_mid", "mid_01", matches=10, correct=10, wrong=0, score_mean=20.0, h_median=0.8, h_p90=1.2),
            row("pair_ext2", "extreme_02", matches=10, correct=9, wrong=1, score_mean=20.0, h_median=1.0, h_p90=2.0),
            row("pair_ext3", "extreme_03", matches=12, correct=11, wrong=1, score_mean=22.0, h_median=1.0, h_p90=2.0),
            row("pair_ext1", "extreme_01", matches=4, correct=4, wrong=0, score_mean=18.0, h_median=0.8, h_p90=1.1),
        ]
        rescue = [
            row("pair_mid", "mid_01", matches=20, correct=18, wrong=2, score_mean=25.0, h_median=0.8, h_p90=1.2),
            row("pair_ext2", "extreme_02", matches=12, correct=12, wrong=0, score_mean=21.0, h_median=1.0, h_p90=2.0),
            row("pair_ext3", "extreme_03", matches=30, correct=20, wrong=10, score_mean=10.0, h_median=1.0, h_p90=2.0),
            row("pair_ext1", "extreme_01", matches=10, correct=10, wrong=0, score_mean=24.0, h_median=0.8, h_p90=1.1),
        ]
        return baseline, rescue

    def test_dual_checkpoint_rescue_selector_only_switches_safe_extreme_rows(self) -> None:
        dual_rescue_mod = self.dual_rescue_module()
        baseline, rescue = self.dual_rescue_rows()
        config = dual_rescue_mod.SelectorConfig(
            target_variants=("extreme_02", "extreme_03"),
            min_match_gain=1,
            min_rescue_matches=8,
            max_rescue_homography_p90_px=3.2,
            max_rescue_homography_median_px=1.8,
            min_rescue_score_mean=16.0,
            require_rescue_score_mean_not_lower=True,
        )

        combined = dual_rescue_mod.combine_summary_rows(
            baseline,
            rescue,
            config=config,
            source="formal",
            split="test",
            baseline_label="phase3zn",
            rescue_label="phase5d",
        )

        self.assertEqual([row["selected_model"] for row in combined], ["phase3zn", "phase5d", "phase3zn", "phase3zn"])
        self.assertEqual([int(row["matches"]) for row in combined], [10, 12, 12, 4])
        self.assertEqual([int(row["correct"]) for row in combined], [10, 12, 11, 4])
        self.assertEqual([int(row["wrong"]) for row in combined], [0, 0, 1, 0])
        self.assertEqual(combined[1]["selector_reason"], "rescue_selected")
        self.assertIn("target_variant", combined[0]["selector_reason"])
        self.assertIn("score_mean", combined[2]["selector_reason"])
        self.assertIn("target_variant", combined[3]["selector_reason"])

        summary = dual_rescue_mod.summarize_rows(combined, label="selected", source="formal", split="test")
        self.assertEqual(summary["filtered_matches"], 38)
        self.assertEqual(summary["filtered_correct"], 37)
        self.assertEqual(summary["filtered_wrong"], 1)
        self.assertAlmostEqual(float(summary["filtered_precision"]), 37.0 / 38.0, places=6)

    def test_dual_checkpoint_rescue_cli_combines_multiple_sources(self) -> None:
        dual_rescue_mod = self.dual_rescue_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline, rescue = self.dual_rescue_rows()
            fieldnames = list(baseline[0].keys())
            base_formal = root / "base_formal.csv"
            rescue_formal = root / "rescue_formal.csv"
            base_guard = root / "base_guard.csv"
            rescue_guard = root / "rescue_guard.csv"
            self.write_csv(base_formal, fieldnames, baseline)
            self.write_csv(rescue_formal, fieldnames, rescue)
            self.write_csv(base_guard, fieldnames, baseline[:1])
            self.write_csv(rescue_guard, fieldnames, rescue[:1])

            output_dir = root / "dual_eval"
            dual_rescue_mod.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--baseline-label",
                    "phase3zn",
                    "--rescue-label",
                    "phase5d",
                    "--source",
                    f"formal,test,{base_formal},{rescue_formal}",
                    "--source",
                    f"regression_guard,test,{base_guard},{rescue_guard}",
                ]
            )

            combined_path = output_dir / "combined_filtered_summary.csv"
            summary_path = output_dir / "summary.csv"
            variant_path = output_dir / "variant_summary.csv"
            html_path = output_dir / "index.html"
            self.assertTrue(combined_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(variant_path.exists())
            self.assertTrue(html_path.exists())

            with combined_path.open("r", encoding="utf-8", newline="") as handle:
                combined_rows = list(csv.DictReader(handle))
            selected_by_base = {row["base_id"]: row["selected_model"] for row in combined_rows if row["source"] == "formal"}
            self.assertEqual(selected_by_base["pair_ext2"], "phase5d")
            self.assertEqual(selected_by_base["pair_mid"], "phase3zn")

            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = {(row["source"], row["model"]): row for row in csv.DictReader(handle)}
            formal_selected = summary_rows[("formal", "selected")]
            guard_selected = summary_rows[("regression_guard", "selected")]
            self.assertEqual(formal_selected["filtered_correct"], "37")
            self.assertEqual(formal_selected["filtered_wrong"], "1")
            self.assertEqual(guard_selected["filtered_correct"], "10")
            self.assertEqual(guard_selected["filtered_wrong"], "0")

    def test_goal_audit_reports_mainline_evidence_and_legacy_reliability_risk(self) -> None:
        audit_mod = self.goal_audit_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "python").mkdir()
            (root / "runs").mkdir()
            (root / "scripts" / "benchmark_lazy_pose_pairs.py").write_text(
                "\n".join(
                    [
                        "best_by_match_score_pytorch_pfm_state.pt",
                        "best_by_recall_pytorch_pfm_state.pt",
                        "best_by_ransac_inlier_pytorch_pfm_state.pt",
                        "best_by_extreme_score_pytorch_pfm_state.pt",
                        "last_good_pytorch_pfm_state.pt",
                        "stability-auto-recovery",
                        "true_match_rejected_by_dustbin_ratio",
                        "positive_vs_dustbin_margin_mean",
                        "true_match_in_topk@64",
                        "true_match_in_topk@256",
                        "visual_RANSAC_inlier_count",
                        "visual_extreme_RANSAC_inlier_count",
                        "descriptor_geometry_safety_schedule",
                        "matcher_candidate_topk = 256",
                        "matcher_reliability_pair_bias\": \"off\"",
                        "matcher_reliability_dustbin_bias\": \"off\"",
                        "no_match_prior_weight\": 0.0",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "python" / "pfm_pytorch_training.py").write_text(
                "\n".join(
                    [
                        "graph_matcher_metadata_mode: str = \"calibrated\"",
                        "def apply_graph_metadata_mode(metadata, mode):",
                        "    if mode == \"calibrated\":",
                        "        adjusted[:, 12:13] = 0.0",
                        "        adjusted[:, 14 : min(adjusted.size(1), 16)] = 0.0",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "python" / "pfm_model.py").write_text(
                "return (matchability - 0.5) - (uncertainty - 0.5) - (no_match_prior - 0.5)\n",
                encoding="utf-8",
            )
            (root / "scripts" / "run_dual_checkpoint_rescue_eval.py").write_text("SelectorConfig\n", encoding="utf-8")
            (root / "scripts" / "run_fov76_checkpoint_promotion_pipeline.py").write_text(
                "--dual-checkpoint-rescue-selector\n",
                encoding="utf-8",
            )
            decision = root / "runs" / "promotion_decision.json"
            decision.write_text('{"promote": true, "failed_reasons": []}\n', encoding="utf-8")
            metrics = root / "runs" / "train_metrics.csv"
            self.write_csv(
                metrics,
                [
                    "step",
                    "true_match_rejected_by_dustbin_ratio",
                    "positive_vs_dustbin_margin_mean",
                    "visual_num_filtered_matches",
                    "visual_extreme_num_filtered_matches",
                    "visual_RANSAC_inlier_count",
                    "visual_extreme_RANSAC_inlier_count",
                ],
                [
                    {
                        "step": 10,
                        "true_match_rejected_by_dustbin_ratio": 0.1,
                        "positive_vs_dustbin_margin_mean": 5.0,
                        "visual_num_filtered_matches": 100,
                        "visual_extreme_num_filtered_matches": 40,
                        "visual_RANSAC_inlier_count": 100,
                        "visual_extreme_RANSAC_inlier_count": 40,
                    }
                ],
            )

            items = audit_mod.audit_goal(
                project_root=root,
                selector_promotion_json=decision,
                train_metrics_csv=metrics,
            )
            by_id = {item.requirement_id: item for item in items}

            self.assertEqual(by_id["phase0.checkpoints"].status, "PASS")
            self.assertEqual(by_id["phase1.matcher_calibration"].status, "PASS")
            self.assertEqual(by_id["phase3.reliability_decoupling"].status, "PARTIAL")
            self.assertIn("calibrated_removes_reliability=True", by_id["phase3.reliability_decoupling"].evidence)
            self.assertIn("legacy", by_id["phase3.reliability_decoupling"].risk.lower())
            self.assertEqual(by_id["selector.dual_checkpoint"].status, "PASS")
            self.assertEqual(by_id["success.training_metrics"].status, "PASS")

    def test_goal_audit_reports_validated_active_mainline_selector(self) -> None:
        audit_mod = self.goal_audit_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "python").mkdir()
            validation = root / "active_mainline_validation.json"
            validation.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "active_selector": "phase5d_selector",
                        "active_label": "phase3zn_phase5d_selector",
                        "active_score": {
                            "correct_delta": 61,
                            "wrong_delta": 0,
                            "precision_delta": 0.001225,
                            "promote": True,
                            "failed_reasons": [],
                            "regression_guard_clean": True,
                            "selector_config": {
                                "target_variants": ["extreme_02", "extreme_03"],
                                "min_rescue_matches": 8,
                            },
                        },
                        "backup_scores": {
                            "phase5e_ransac_minmatch16_selector": {
                                "correct_delta": 47,
                                "wrong_delta": 1,
                                "promote": True,
                                "regression_guard_clean": True,
                            }
                        },
                        "errors": [],
                    }
                ),
                encoding="utf-8",
            )

            items = audit_mod.audit_goal(
                project_root=root,
                active_mainline_validation_json=validation,
            )
            by_id = {item.requirement_id: item for item in items}

            self.assertEqual(by_id["active_mainline.selector_validation"].status, "PASS")
            self.assertIn("phase3zn_phase5d_selector", by_id["active_mainline.selector_validation"].evidence)
            self.assertIn("correct_delta=61", by_id["active_mainline.selector_validation"].evidence)
            self.assertIn("wrong_delta=0", by_id["active_mainline.selector_validation"].evidence)
            self.assertEqual(by_id["active_mainline.selector_validation"].risk, "")

    def test_goal_audit_reports_phase2_and_phase5_hard_mining_evidence(self) -> None:
        audit_mod = self.goal_audit_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "python").mkdir()
            (root / "runs").mkdir()
            (root / "scripts" / "benchmark_lazy_pose_pairs.py").write_text(
                "\n".join(
                    [
                        "best_by_match_score_pytorch_pfm_state.pt",
                        "best_by_recall_pytorch_pfm_state.pt",
                        "best_by_ransac_inlier_pytorch_pfm_state.pt",
                        "best_by_extreme_score_pytorch_pfm_state.pt",
                        "last_good_pytorch_pfm_state.pt",
                        "stability-auto-recovery",
                        "true_match_rejected_by_dustbin_ratio",
                        "positive_vs_dustbin_margin_mean",
                        "true_match_in_topk@64",
                        "true_match_in_topk@256",
                        "visual_RANSAC_inlier_count",
                        "visual_extreme_RANSAC_inlier_count",
                        "descriptor_geometry_safety_schedule",
                        "matcher_candidate_topk = 256",
                        "matcher_reliability_pair_bias\": \"off\"",
                        "matcher_reliability_dustbin_bias\": \"off\"",
                        "no_match_prior_weight\": 0.0",
                        "--train-graph-calibration-only",
                        "graph_matcher_true_match_margin_weight",
                        "graph_matcher_positive_dustbin_margin_weight",
                        "graph_matcher_final_false_match_weight",
                        "graph_matcher_mined_false_match_weight",
                        "graph_matcher_raw_false_match_weight",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "python" / "pfm_pytorch_training.py").write_text(
                "\n".join(
                    [
                        "graph_matcher_metadata_mode: str = \"calibrated\"",
                        "def apply_graph_metadata_mode(metadata, mode):",
                        "    if mode == \"calibrated\":",
                        "        adjusted[:, 12:13] = 0.0",
                        "        adjusted[:, 14 : min(adjusted.size(1), 16)] = 0.0",
                        "train_graph_calibration_only",
                        "accept_head",
                        "geometry_bias",
                        "dustbin_bias",
                        "graph_matcher_true_match_margin_weight",
                        "graph_matcher_final_false_match_weight",
                        "graph_matcher_mined_false_match_weight",
                        "no_match_prior_weight: float = 0.0",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "python" / "pfm_model.py").write_text("legacy reliability removed\n", encoding="utf-8")
            (root / "scripts" / "mine_hard_failure_pairs.py").write_text(
                "\n".join(
                    [
                        "only_extreme_variants",
                        "extreme_02",
                        "extreme_03",
                        "low_match_count",
                        "low_precision",
                        "high_false",
                        "high_loss",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "scripts" / "build_rescue_gain_hard_set.py").write_text(
                "\n".join(["rescue_false_negative", "extreme_02", "extreme_03", "delta_correct"]),
                encoding="utf-8",
            )
            (root / "scripts" / "build_phase2j_balanced_manifest.py").write_text(
                "\n".join(["phase2j_bucket", "protected", "extreme_02", "extreme_03"]),
                encoding="utf-8",
            )
            (root / "scripts" / "run_dual_checkpoint_rescue_eval.py").write_text("SelectorConfig\n", encoding="utf-8")
            (root / "scripts" / "run_fov76_checkpoint_promotion_pipeline.py").write_text(
                "\n".join(
                    [
                        "--dual-checkpoint-rescue-selector",
                        "regression_guard",
                        "extreme_gain",
                        "--formal-protected-variants",
                        "--max-protected-variant-precision-drop",
                        "--max-formal-target-precision-drop",
                    ]
                ),
                encoding="utf-8",
            )
            decision = root / "runs" / "promotion_decision.json"
            decision.write_text('{"promote": true, "failed_reasons": []}\n', encoding="utf-8")

            items = audit_mod.audit_goal(project_root=root, selector_promotion_json=decision)
            by_id = {item.requirement_id: item for item in items}

            self.assertEqual(by_id["phase2.extreme_hard_mining"].status, "PASS")
            self.assertIn("extreme_02/extreme_03", by_id["phase2.extreme_hard_mining"].evidence)
            self.assertEqual(by_id["phase2.matcher_only_loss_constraints"].status, "PARTIAL")
            self.assertIn("RANSAC consistency loss", by_id["phase2.matcher_only_loss_constraints"].risk)
            self.assertEqual(by_id["phase5.extreme_accuracy_guard"].status, "PASS")
            self.assertIn("regression_guard", by_id["phase5.extreme_accuracy_guard"].evidence)

    def test_fov76_active_selector_validator_keeps_phase5d_as_mainline(self) -> None:
        validator_mod = self.active_selector_validator_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            phase5d_decision = root / "phase5d_promotion_decision.json"
            phase5e_decision = root / "phase5e_promotion_decision.json"
            phase5d_metadata = root / "phase5d_selector_metadata.json"
            phase5e_metadata = root / "phase5e_selector_metadata.json"
            config = root / "active_config.json"

            def decision(correct_delta: int, wrong_delta: int, precision_delta: float) -> dict[str, object]:
                return {
                    "promote": True,
                    "failed_reasons": [],
                    "comparisons": [
                        {
                            "context": "formal_target_total",
                            "split": "all",
                            "correct_delta": correct_delta,
                            "wrong_delta": wrong_delta,
                            "precision_delta": precision_delta,
                        },
                        {
                            "context": "regression_guard",
                            "split": "val",
                            "correct_delta": 0,
                            "wrong_delta": 0,
                            "precision_delta": 0.0,
                        },
                        {
                            "context": "regression_guard",
                            "split": "test",
                            "correct_delta": 0,
                            "wrong_delta": 0,
                            "precision_delta": 0.0,
                        },
                    ],
                }

            phase5d_decision.write_text(json.dumps(decision(61, 0, 0.001225)), encoding="utf-8")
            phase5e_decision.write_text(json.dumps(decision(47, 1, 0.000389)), encoding="utf-8")
            phase5d_metadata.write_text(
                json.dumps({"config": {"target_variants": ["extreme_02", "extreme_03"], "min_rescue_matches": 8}}),
                encoding="utf-8",
            )
            phase5e_metadata.write_text(
                json.dumps({"config": {"target_variants": ["extreme_02", "extreme_03"], "min_rescue_matches": 16}}),
                encoding="utf-8",
            )
            config.write_text(
                json.dumps(
                    {
                        "active_selector": "phase5d_selector",
                        "active_label": "phase3zn_phase5d_selector",
                        "candidates": [
                            {
                                "name": "phase5d_selector",
                                "label": "phase3zn_phase5d_selector",
                                "decision_path": str(phase5d_decision),
                                "metadata_path": str(phase5d_metadata),
                            },
                            {
                                "name": "phase5e_ransac_minmatch16_selector",
                                "label": "phase3zn_phase5e_ransac_minmatch16_selector",
                                "decision_path": str(phase5e_decision),
                                "metadata_path": str(phase5e_metadata),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = validator_mod.validate_active_selector_config(config)

            self.assertTrue(result.valid, result.errors)
            self.assertEqual(result.active_selector, "phase5d_selector")
            self.assertEqual(result.active_label, "phase3zn_phase5d_selector")
            self.assertEqual(result.active_score["correct_delta"], 61)
            self.assertEqual(result.active_score["wrong_delta"], 0)
            self.assertEqual(result.backup_scores["phase5e_ransac_minmatch16_selector"]["wrong_delta"], 1)

    def test_selector_disagreement_mining_outputs_extreme_manifest_rows(self) -> None:
        miner_mod = self.selector_disagreement_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_manifest = root / "overlap_edges_test.csv"
            active_summary = root / "phase5d_selector.csv"
            backup_summary = root / "phase5e_selector.csv"
            output_manifest = root / "selector_disagreement.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"

            pair_fields = [
                "pair_index",
                "split",
                "pair_type",
                "reference_dataset_id",
                "reference_pose_id",
                "reference_base_id",
                "reference_variant",
                "target_dataset_id",
                "target_pose_id",
                "target_base_id",
                "target_variant",
                "valid_fraction",
                "valid_pixels",
                "attempts",
                "crop_a_x0",
                "crop_a_y0",
                "crop_a_x1",
                "crop_a_y1",
                "crop_b_x0",
                "crop_b_y0",
                "crop_b_x1",
                "crop_b_y1",
            ]
            pair_rows = [
                {
                    "pair_index": "5",
                    "split": "test",
                    "pair_type": "same-position",
                    "reference_dataset_id": "dom76",
                    "reference_pose_id": "pose0",
                    "reference_base_id": "base0",
                    "reference_variant": "nadir",
                    "target_dataset_id": "dom76",
                    "target_pose_id": "pose0_ext2",
                    "target_base_id": "base0",
                    "target_variant": "extreme_02",
                    "valid_fraction": "0.8",
                    "valid_pixels": "1000",
                    "attempts": "1",
                    "crop_a_x0": "0",
                    "crop_a_y0": "0",
                    "crop_a_x1": "2048",
                    "crop_a_y1": "2048",
                    "crop_b_x0": "0",
                    "crop_b_y0": "0",
                    "crop_b_x1": "2048",
                    "crop_b_y1": "2048",
                },
                {
                    "pair_index": "0",
                    "split": "test",
                    "pair_type": "same-position",
                    "reference_dataset_id": "dom76",
                    "reference_pose_id": "pose1",
                    "reference_base_id": "base1",
                    "reference_variant": "nadir",
                    "target_dataset_id": "dom76",
                    "target_pose_id": "pose1_mid",
                    "target_base_id": "base1",
                    "target_variant": "mid_01",
                    "valid_fraction": "0.9",
                    "valid_pixels": "1000",
                    "attempts": "1",
                    "crop_a_x0": "0",
                    "crop_a_y0": "0",
                    "crop_a_x1": "2048",
                    "crop_a_y1": "2048",
                    "crop_b_x0": "0",
                    "crop_b_y0": "0",
                    "crop_b_x1": "2048",
                    "crop_b_y1": "2048",
                },
            ]
            self.write_csv(pair_manifest, pair_fields, pair_rows)

            summary_fields = [
                "label",
                "base_id",
                "target_variant",
                "split",
                "matches",
                "correct",
                "wrong",
                "precision",
                "score_mean",
                "homography_residual_median_px",
                "homography_residual_p90_px",
                "source",
                "row_index",
                "selected_model",
                "selector_reason",
            ]
            self.write_csv(
                active_summary,
                summary_fields,
                [
                    {
                        "label": "active",
                        "base_id": "base0",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "matches": 14,
                        "correct": 14,
                        "wrong": 0,
                        "precision": 1.0,
                        "score_mean": 21.0,
                        "homography_residual_median_px": 1.0,
                        "homography_residual_p90_px": 2.0,
                        "source": "formal",
                        "row_index": 0,
                        "selected_model": "phase5d",
                        "selector_reason": "rescue_selected",
                    },
                    {
                        "label": "active",
                        "base_id": "base1",
                        "target_variant": "mid_01",
                        "split": "test",
                        "matches": 10,
                        "correct": 10,
                        "wrong": 0,
                        "precision": 1.0,
                        "score_mean": 20.0,
                        "homography_residual_median_px": 1.0,
                        "homography_residual_p90_px": 2.0,
                        "source": "formal",
                        "row_index": 1,
                        "selected_model": "phase3zn",
                        "selector_reason": "blocked_target_variant:mid_01",
                    },
                ],
            )
            self.write_csv(
                backup_summary,
                summary_fields,
                [
                    {
                        "label": "backup",
                        "base_id": "base0",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "matches": 10,
                        "correct": 9,
                        "wrong": 1,
                        "precision": 0.9,
                        "score_mean": 19.0,
                        "homography_residual_median_px": 1.2,
                        "homography_residual_p90_px": 2.4,
                        "source": "formal",
                        "row_index": 0,
                        "selected_model": "phase3zn",
                        "selector_reason": "blocked_match_gain:9<12",
                    },
                    {
                        "label": "backup",
                        "base_id": "base1",
                        "target_variant": "mid_01",
                        "split": "test",
                        "matches": 7,
                        "correct": 6,
                        "wrong": 1,
                        "precision": 0.857143,
                        "score_mean": 19.0,
                        "homography_residual_median_px": 1.0,
                        "homography_residual_p90_px": 2.0,
                        "source": "formal",
                        "row_index": 1,
                        "selected_model": "phase5e",
                        "selector_reason": "rescue_selected",
                    },
                ],
            )

            miner_mod.main(
                [
                    "--active-summary",
                    str(active_summary),
                    "--candidate-summary",
                    str(backup_summary),
                    "--pair-manifest-source",
                    f"formal,test,{pair_manifest}",
                    "--active-label",
                    "phase5d_selector",
                    "--candidate-label",
                    "phase5e_selector",
                    "--output-manifest",
                    str(output_manifest),
                    "--summary-json",
                    str(summary_json),
                    "--output-html",
                    str(report_html),
                ]
            )

            with output_manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["target_variant"], "extreme_02")
            self.assertEqual(rows[0]["reference_pose_id"], "pose0")
            self.assertIn("candidate_missed_active_correct", rows[0]["hard_reasons"])
            self.assertIn("candidate_wrong_increase", rows[0]["hard_reasons"])
            self.assertIn("selector_choice_disagreement", rows[0]["hard_reasons"])
            self.assertEqual(rows[0]["source_active_correct"], "14")
            self.assertEqual(rows[0]["source_candidate_correct"], "9")
            self.assertEqual(rows[0]["correct_delta_active_minus_candidate"], "5")
            self.assertEqual(rows[0]["match_delta"], "4")
            self.assertEqual(rows[0]["correct_delta"], "5")
            self.assertEqual(rows[0]["wrong_delta"], "-1")
            self.assertEqual(rows[0]["precision_delta"], "0.100000")
            self.assertTrue(summary_json.exists())
            self.assertTrue(report_html.exists())

    def test_selector_disagreement_candidate_gain_mode_outputs_clean_extreme_gains(self) -> None:
        miner_mod = self.selector_disagreement_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_manifest = root / "overlap_edges_test.csv"
            active_summary = root / "phase5d_selector.csv"
            candidate_summary = root / "phase5f_selector.csv"
            output_manifest = root / "selector_candidate_gains.csv"
            summary_json = root / "summary.json"
            report_html = root / "index.html"

            pair_fields = [
                "pair_index",
                "split",
                "pair_type",
                "reference_dataset_id",
                "reference_pose_id",
                "reference_base_id",
                "reference_variant",
                "target_dataset_id",
                "target_pose_id",
                "target_base_id",
                "target_variant",
                "valid_fraction",
                "valid_pixels",
                "attempts",
                "crop_a_x0",
                "crop_a_y0",
                "crop_a_x1",
                "crop_a_y1",
                "crop_b_x0",
                "crop_b_y0",
                "crop_b_x1",
                "crop_b_y1",
            ]
            self.write_csv(
                pair_manifest,
                pair_fields,
                [
                    {
                        "pair_index": "9",
                        "split": "test",
                        "pair_type": "same-position",
                        "reference_dataset_id": "dom76",
                        "reference_pose_id": "pose_gain",
                        "reference_base_id": "base_gain",
                        "reference_variant": "nadir",
                        "target_dataset_id": "dom76",
                        "target_pose_id": "pose_gain_ext2",
                        "target_base_id": "base_gain",
                        "target_variant": "extreme_02",
                        "valid_fraction": "0.8",
                        "valid_pixels": "1000",
                        "attempts": "1",
                        "crop_a_x0": "0",
                        "crop_a_y0": "0",
                        "crop_a_x1": "2048",
                        "crop_a_y1": "2048",
                        "crop_b_x0": "0",
                        "crop_b_y0": "0",
                        "crop_b_x1": "2048",
                        "crop_b_y1": "2048",
                    },
                    {
                        "pair_index": "2",
                        "split": "test",
                        "pair_type": "same-position",
                        "reference_dataset_id": "dom76",
                        "reference_pose_id": "pose_wrong",
                        "reference_base_id": "base_wrong",
                        "reference_variant": "nadir",
                        "target_dataset_id": "dom76",
                        "target_pose_id": "pose_wrong_ext3",
                        "target_base_id": "base_wrong",
                        "target_variant": "extreme_03",
                        "valid_fraction": "0.8",
                        "valid_pixels": "1000",
                        "attempts": "1",
                        "crop_a_x0": "0",
                        "crop_a_y0": "0",
                        "crop_a_x1": "2048",
                        "crop_a_y1": "2048",
                        "crop_b_x0": "0",
                        "crop_b_y0": "0",
                        "crop_b_x1": "2048",
                        "crop_b_y1": "2048",
                    },
                    {
                        "pair_index": "3",
                        "split": "test",
                        "pair_type": "same-position",
                        "reference_dataset_id": "dom76",
                        "reference_pose_id": "pose_mid",
                        "reference_base_id": "base_mid",
                        "reference_variant": "nadir",
                        "target_dataset_id": "dom76",
                        "target_pose_id": "pose_mid",
                        "target_base_id": "base_mid",
                        "target_variant": "mid_01",
                        "valid_fraction": "0.8",
                        "valid_pixels": "1000",
                        "attempts": "1",
                        "crop_a_x0": "0",
                        "crop_a_y0": "0",
                        "crop_a_x1": "2048",
                        "crop_a_y1": "2048",
                        "crop_b_x0": "0",
                        "crop_b_y0": "0",
                        "crop_b_x1": "2048",
                        "crop_b_y1": "2048",
                    },
                ],
            )

            summary_fields = [
                "label",
                "base_id",
                "target_variant",
                "split",
                "matches",
                "correct",
                "wrong",
                "precision",
                "score_mean",
                "homography_residual_median_px",
                "homography_residual_p90_px",
                "source",
                "row_index",
                "selected_model",
                "selector_reason",
            ]
            self.write_csv(
                active_summary,
                summary_fields,
                [
                    {
                        "label": "active",
                        "base_id": "base_gain",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "matches": 20,
                        "correct": 18,
                        "wrong": 2,
                        "precision": 0.9,
                        "score_mean": 20.0,
                        "homography_residual_median_px": 1.0,
                        "homography_residual_p90_px": 2.0,
                        "source": "formal",
                        "row_index": 0,
                        "selected_model": "phase3zn",
                        "selector_reason": "baseline_selected",
                    },
                    {
                        "label": "active",
                        "base_id": "base_wrong",
                        "target_variant": "extreme_03",
                        "split": "test",
                        "matches": 15,
                        "correct": 12,
                        "wrong": 3,
                        "precision": 0.8,
                        "score_mean": 20.0,
                        "homography_residual_median_px": 1.0,
                        "homography_residual_p90_px": 2.0,
                        "source": "formal",
                        "row_index": 1,
                        "selected_model": "phase3zn",
                        "selector_reason": "baseline_selected",
                    },
                    {
                        "label": "active",
                        "base_id": "base_mid",
                        "target_variant": "mid_01",
                        "split": "test",
                        "matches": 30,
                        "correct": 30,
                        "wrong": 0,
                        "precision": 1.0,
                        "score_mean": 21.0,
                        "homography_residual_median_px": 1.0,
                        "homography_residual_p90_px": 2.0,
                        "source": "formal",
                        "row_index": 2,
                        "selected_model": "phase3zn",
                        "selector_reason": "protected_variant",
                    },
                ],
            )
            self.write_csv(
                candidate_summary,
                summary_fields,
                [
                    {
                        "label": "candidate",
                        "base_id": "base_gain",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "matches": 26,
                        "correct": 25,
                        "wrong": 1,
                        "precision": 0.961538,
                        "score_mean": 22.0,
                        "homography_residual_median_px": 0.8,
                        "homography_residual_p90_px": 1.6,
                        "source": "formal",
                        "row_index": 0,
                        "selected_model": "phase5f",
                        "selector_reason": "rescue_selected",
                    },
                    {
                        "label": "candidate",
                        "base_id": "base_wrong",
                        "target_variant": "extreme_03",
                        "split": "test",
                        "matches": 20,
                        "correct": 19,
                        "wrong": 5,
                        "precision": 0.791667,
                        "score_mean": 22.0,
                        "homography_residual_median_px": 0.8,
                        "homography_residual_p90_px": 1.6,
                        "source": "formal",
                        "row_index": 1,
                        "selected_model": "phase5f",
                        "selector_reason": "rescue_selected",
                    },
                    {
                        "label": "candidate",
                        "base_id": "base_mid",
                        "target_variant": "mid_01",
                        "split": "test",
                        "matches": 33,
                        "correct": 33,
                        "wrong": 0,
                        "precision": 1.0,
                        "score_mean": 22.0,
                        "homography_residual_median_px": 0.8,
                        "homography_residual_p90_px": 1.6,
                        "source": "formal",
                        "row_index": 2,
                        "selected_model": "phase5f",
                        "selector_reason": "rescue_selected",
                    },
                ],
            )

            miner_mod.main(
                [
                    "--active-summary",
                    str(active_summary),
                    "--candidate-summary",
                    str(candidate_summary),
                    "--pair-manifest-source",
                    f"formal,test,{pair_manifest}",
                    "--active-label",
                    "phase5d_selector",
                    "--candidate-label",
                    "phase5f_selector",
                    "--mine-mode",
                    "candidate_gains",
                    "--output-manifest",
                    str(output_manifest),
                    "--summary-json",
                    str(summary_json),
                    "--output-html",
                    str(report_html),
                ]
            )

            with output_manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["reference_pose_id"], "pose_gain")
            self.assertEqual(row["target_variant"], "extreme_02")
            self.assertIn("candidate_correct_gain", row["hard_reasons"])
            self.assertIn("candidate_match_gain", row["hard_reasons"])
            self.assertEqual(row["correct_delta"], "7")
            self.assertEqual(row["wrong_delta"], "-1")
            self.assertEqual(row["match_delta"], "6")
            self.assertEqual(row["correct_delta_active_minus_candidate"], "-7")
            self.assertEqual(row["correct_delta_candidate_minus_active"], "7")
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["mine_mode"], "candidate_gains")
            self.assertEqual(summary["totals"]["correct_delta"], 7)
            self.assertEqual(summary["totals"]["wrong_delta"], -1)

    def test_checkpoint_promotion_rejects_formal_or_regression_guard_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            formal = root / "formal.csv"
            guard = root / "guard.csv"
            self.write_csv(
                formal,
                [
                    "label",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "label": "baseline",
                        "split": "test",
                        "filtered_matches": 3367,
                        "filtered_correct": 2997,
                        "filtered_wrong": 370,
                        "filtered_precision": 0.890110,
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "filtered_matches": 3375,
                        "filtered_correct": 2979,
                        "filtered_wrong": 396,
                        "filtered_precision": 0.882667,
                    },
                ],
            )
            self.write_csv(
                guard,
                [
                    "model",
                    "set",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "model": "baseline",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 972,
                        "filtered_correct": 889,
                        "filtered_wrong": 83,
                        "filtered_precision": 0.914609,
                    },
                    {
                        "model": "candidate",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 984,
                        "filtered_correct": 871,
                        "filtered_wrong": 113,
                        "filtered_precision": 0.885163,
                    },
                    {
                        "model": "baseline",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 270,
                        "filtered_correct": 228,
                        "filtered_wrong": 42,
                        "filtered_precision": 0.844444,
                    },
                    {
                        "model": "candidate",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 293,
                        "filtered_correct": 251,
                        "filtered_wrong": 42,
                        "filtered_precision": 0.856655,
                    },
                ],
            )

            decision = promotion_mod.evaluate_promotion(
                formal_summary=formal,
                guard_summary=guard,
                baseline_label="baseline",
                candidate_label="candidate",
                splits=["test"],
                max_formal_precision_drop=0.0,
                max_formal_correct_drop=0,
                max_formal_wrong_increase=0,
                max_guard_precision_drop=0.0,
                max_guard_correct_drop=0,
                max_guard_wrong_increase=0,
                min_extreme_correct_gain=1,
            )

            self.assertFalse(decision.promote)
            self.assertTrue(any("formal/test" in reason for reason in decision.failed_reasons))
            self.assertTrue(any("regression_guard/test" in reason for reason in decision.failed_reasons))
            self.assertTrue(any("extreme_gain/test" in reason for reason in decision.passed_reasons))

    def test_checkpoint_promotion_accepts_non_regressive_extreme_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            formal = root / "formal.csv"
            guard = root / "guard.csv"
            self.write_csv(
                formal,
                [
                    "label",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "label": "baseline",
                        "split": "test",
                        "filtered_matches": 100,
                        "filtered_correct": 90,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.900000,
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "filtered_matches": 110,
                        "filtered_correct": 100,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.909091,
                    },
                ],
            )
            self.write_csv(
                guard,
                [
                    "model",
                    "set",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "model": "baseline",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 80,
                        "filtered_correct": 76,
                        "filtered_wrong": 4,
                        "filtered_precision": 0.950000,
                    },
                    {
                        "model": "candidate",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 84,
                        "filtered_correct": 80,
                        "filtered_wrong": 4,
                        "filtered_precision": 0.952381,
                    },
                    {
                        "model": "baseline",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 40,
                        "filtered_correct": 30,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.750000,
                    },
                    {
                        "model": "candidate",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 48,
                        "filtered_correct": 38,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.791667,
                    },
                ],
            )

            decision = promotion_mod.evaluate_promotion(
                formal_summary=formal,
                guard_summary=guard,
                baseline_label="baseline",
                candidate_label="candidate",
                splits=["test"],
                max_formal_precision_drop=0.0,
                max_formal_correct_drop=0,
                max_formal_wrong_increase=0,
                max_guard_precision_drop=0.0,
                max_guard_correct_drop=0,
                max_guard_wrong_increase=0,
                min_extreme_correct_gain=1,
            )

            self.assertTrue(decision.promote)
            self.assertEqual(decision.failed_reasons, [])
            self.assertTrue(any("formal/test" in reason for reason in decision.passed_reasons))
            self.assertTrue(any("regression_guard/test" in reason for reason in decision.passed_reasons))
            self.assertTrue(any("extreme_gain/test" in reason for reason in decision.passed_reasons))

    def test_checkpoint_promotion_rejects_crashed_sweep_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            formal = root / "formal.csv"
            guard = root / "guard.csv"
            self.write_csv(
                formal,
                [
                    "label",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                    "sweep_failed",
                    "sweep_error",
                ],
                [
                    {
                        "label": "baseline",
                        "split": "test",
                        "filtered_matches": 100,
                        "filtered_correct": 90,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.900000,
                        "sweep_failed": 0,
                        "sweep_error": "",
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "filtered_matches": 110,
                        "filtered_correct": 100,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.909091,
                        "sweep_failed": 0,
                        "sweep_error": "",
                    },
                ],
            )
            self.write_csv(
                guard,
                [
                    "model",
                    "set",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                    "sweep_failed",
                    "sweep_error",
                ],
                [
                    {
                        "model": "baseline",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 80,
                        "filtered_correct": 76,
                        "filtered_wrong": 4,
                        "filtered_precision": 0.950000,
                        "sweep_failed": 0,
                        "sweep_error": "",
                    },
                    {
                        "model": "candidate",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 84,
                        "filtered_correct": 80,
                        "filtered_wrong": 4,
                        "filtered_precision": 0.952381,
                        "sweep_failed": 0,
                        "sweep_error": "",
                    },
                    {
                        "model": "baseline",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 40,
                        "filtered_correct": 30,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.750000,
                        "sweep_failed": 0,
                        "sweep_error": "",
                    },
                    {
                        "model": "candidate",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 0,
                        "filtered_correct": 0,
                        "filtered_wrong": 0,
                        "filtered_precision": 0.0,
                        "sweep_failed": 1,
                        "sweep_error": "SIGSEGV",
                    },
                ],
            )

            decision = promotion_mod.evaluate_promotion(
                formal_summary=formal,
                guard_summary=guard,
                baseline_label="baseline",
                candidate_label="candidate",
                splits=["test"],
                max_formal_precision_drop=0.0,
                max_formal_correct_drop=0,
                max_formal_wrong_increase=0,
                max_guard_precision_drop=0.0,
                max_guard_correct_drop=0,
                max_guard_wrong_increase=0,
                min_extreme_correct_gain=1,
            )

            self.assertFalse(decision.promote)
            self.assertTrue(any("sweep_failed" in reason for reason in decision.failed_reasons))

    def test_checkpoint_promotion_rejects_extra_regression_guard_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            formal = root / "formal.csv"
            guard = root / "guard.csv"
            self.write_csv(
                formal,
                [
                    "label",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "label": "baseline",
                        "split": "test",
                        "filtered_matches": 100,
                        "filtered_correct": 90,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.900000,
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "filtered_matches": 110,
                        "filtered_correct": 100,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.909091,
                    },
                ],
            )
            self.write_csv(
                guard,
                [
                    "model",
                    "set",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "model": "baseline",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 80,
                        "filtered_correct": 76,
                        "filtered_wrong": 4,
                        "filtered_precision": 0.950000,
                    },
                    {
                        "model": "candidate",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 84,
                        "filtered_correct": 80,
                        "filtered_wrong": 4,
                        "filtered_precision": 0.952381,
                    },
                    {
                        "model": "baseline",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 40,
                        "filtered_correct": 30,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.750000,
                    },
                    {
                        "model": "candidate",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 48,
                        "filtered_correct": 38,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.791667,
                    },
                    {
                        "model": "baseline",
                        "set": "phase5h_false_cluster_guard",
                        "split": "test",
                        "filtered_matches": 18,
                        "filtered_correct": 18,
                        "filtered_wrong": 0,
                        "filtered_precision": 1.000000,
                    },
                    {
                        "model": "candidate",
                        "set": "phase5h_false_cluster_guard",
                        "split": "test",
                        "filtered_matches": 22,
                        "filtered_correct": 20,
                        "filtered_wrong": 2,
                        "filtered_precision": 0.909091,
                    },
                ],
            )

            decision = promotion_mod.evaluate_promotion(
                formal_summary=formal,
                guard_summary=guard,
                baseline_label="baseline",
                candidate_label="candidate",
                splits=["test"],
                max_formal_precision_drop=0.0,
                max_formal_correct_drop=0,
                max_formal_wrong_increase=0,
                max_guard_precision_drop=0.0,
                max_guard_correct_drop=0,
                max_guard_wrong_increase=0,
                min_extreme_correct_gain=1,
                extra_regression_guard_sets=["phase5h_false_cluster_guard"],
                max_extra_guard_precision_drop=0.03,
                max_extra_guard_correct_drop=0,
                max_extra_guard_wrong_increase=1,
            )

            self.assertFalse(decision.promote)
            self.assertTrue(
                any("phase5h_false_cluster_guard/test" in reason for reason in decision.failed_reasons)
            )

    def test_checkpoint_promotion_uses_extra_guard_specific_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            formal = root / "formal.csv"
            guard = root / "guard.csv"
            self.write_csv(
                formal,
                [
                    "label",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "label": "baseline",
                        "split": "test",
                        "filtered_matches": 100,
                        "filtered_correct": 90,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.900000,
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "filtered_matches": 110,
                        "filtered_correct": 100,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.909091,
                    },
                ],
            )
            self.write_csv(
                guard,
                [
                    "model",
                    "set",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "model": "baseline",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 80,
                        "filtered_correct": 76,
                        "filtered_wrong": 4,
                        "filtered_precision": 0.950000,
                    },
                    {
                        "model": "candidate",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 84,
                        "filtered_correct": 80,
                        "filtered_wrong": 4,
                        "filtered_precision": 0.952381,
                    },
                    {
                        "model": "baseline",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 40,
                        "filtered_correct": 30,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.750000,
                    },
                    {
                        "model": "candidate",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 48,
                        "filtered_correct": 38,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.791667,
                    },
                    {
                        "model": "baseline",
                        "set": "phase5h_false_cluster_guard",
                        "split": "test",
                        "filtered_matches": 14,
                        "filtered_correct": 14,
                        "filtered_wrong": 0,
                        "filtered_precision": 1.000000,
                    },
                    {
                        "model": "candidate",
                        "set": "phase5h_false_cluster_guard",
                        "split": "test",
                        "filtered_matches": 36,
                        "filtered_correct": 35,
                        "filtered_wrong": 1,
                        "filtered_precision": 0.972222,
                    },
                ],
            )

            decision = promotion_mod.evaluate_promotion(
                formal_summary=formal,
                guard_summary=guard,
                baseline_label="baseline",
                candidate_label="candidate",
                splits=["test"],
                max_formal_precision_drop=0.0,
                max_formal_correct_drop=0,
                max_formal_wrong_increase=0,
                max_guard_precision_drop=0.0,
                max_guard_correct_drop=0,
                max_guard_wrong_increase=0,
                min_extreme_correct_gain=1,
                extra_regression_guard_sets=["phase5h_false_cluster_guard"],
                max_extra_guard_precision_drop=0.03,
                max_extra_guard_correct_drop=0,
                max_extra_guard_wrong_increase=1,
            )

            self.assertTrue(decision.promote)
            self.assertTrue(
                any("phase5h_false_cluster_guard/test" in reason for reason in decision.passed_reasons)
            )

    def test_checkpoint_promotion_allows_distinct_formal_and_guard_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            formal = root / "formal.csv"
            guard = root / "guard.csv"
            self.write_csv(
                formal,
                [
                    "label",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "label": "baseline_ransac",
                        "split": "test",
                        "filtered_matches": 100,
                        "filtered_correct": 90,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.900000,
                    },
                    {
                        "label": "candidate_ransac",
                        "split": "test",
                        "filtered_matches": 110,
                        "filtered_correct": 100,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.909091,
                    },
                ],
            )
            self.write_csv(
                guard,
                [
                    "model",
                    "set",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "model": "baseline",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 80,
                        "filtered_correct": 76,
                        "filtered_wrong": 4,
                        "filtered_precision": 0.950000,
                    },
                    {
                        "model": "candidate",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 84,
                        "filtered_correct": 80,
                        "filtered_wrong": 4,
                        "filtered_precision": 0.952381,
                    },
                    {
                        "model": "baseline",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 40,
                        "filtered_correct": 30,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.750000,
                    },
                    {
                        "model": "candidate",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 48,
                        "filtered_correct": 38,
                        "filtered_wrong": 10,
                        "filtered_precision": 0.791667,
                    },
                ],
            )

            decision = promotion_mod.evaluate_promotion(
                formal_summary=formal,
                guard_summary=guard,
                baseline_label="baseline_ransac",
                candidate_label="candidate_ransac",
                guard_baseline_label="baseline",
                guard_candidate_label="candidate",
                splits=["test"],
                max_formal_precision_drop=0.0,
                max_formal_correct_drop=0,
                max_formal_wrong_increase=0,
                max_guard_precision_drop=0.0,
                max_guard_correct_drop=0,
                max_guard_wrong_increase=0,
                min_extreme_correct_gain=1,
            )

            self.assertTrue(decision.promote)

    def test_checkpoint_promotion_can_gate_formal_results_by_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            formal = root / "formal.csv"
            formal_variant = root / "formal_variant.csv"
            guard = root / "guard.csv"
            self.write_csv(
                formal,
                [
                    "label",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "label": "baseline",
                        "split": "test",
                        "filtered_matches": 2070,
                        "filtered_correct": 1995,
                        "filtered_wrong": 75,
                        "filtered_precision": 0.963768,
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "filtered_matches": 2090,
                        "filtered_correct": 2014,
                        "filtered_wrong": 76,
                        "filtered_precision": 0.963636,
                    },
                ],
            )
            self.write_csv(
                formal_variant,
                ["label", "split", "variant", "matches", "correct", "wrong", "precision"],
                [
                    {
                        "label": "baseline",
                        "split": "test",
                        "variant": "extreme_01",
                        "matches": 369,
                        "correct": 352,
                        "wrong": 17,
                        "precision": 0.953930,
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "variant": "extreme_01",
                        "matches": 369,
                        "correct": 352,
                        "wrong": 17,
                        "precision": 0.953930,
                    },
                    {
                        "label": "baseline",
                        "split": "test",
                        "variant": "extreme_02",
                        "matches": 438,
                        "correct": 415,
                        "wrong": 23,
                        "precision": 0.947489,
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "variant": "extreme_02",
                        "matches": 438,
                        "correct": 415,
                        "wrong": 23,
                        "precision": 0.947489,
                    },
                    {
                        "label": "baseline",
                        "split": "test",
                        "variant": "extreme_03",
                        "matches": 278,
                        "correct": 256,
                        "wrong": 22,
                        "precision": 0.920863,
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "variant": "extreme_03",
                        "matches": 298,
                        "correct": 275,
                        "wrong": 23,
                        "precision": 0.922819,
                    },
                    {
                        "label": "baseline",
                        "split": "test",
                        "variant": "mid_01",
                        "matches": 370,
                        "correct": 367,
                        "wrong": 3,
                        "precision": 0.991892,
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "variant": "mid_01",
                        "matches": 370,
                        "correct": 367,
                        "wrong": 3,
                        "precision": 0.991892,
                    },
                    {
                        "label": "baseline",
                        "split": "test",
                        "variant": "mid_02",
                        "matches": 615,
                        "correct": 605,
                        "wrong": 10,
                        "precision": 0.983740,
                    },
                    {
                        "label": "candidate",
                        "split": "test",
                        "variant": "mid_02",
                        "matches": 615,
                        "correct": 605,
                        "wrong": 10,
                        "precision": 0.983740,
                    },
                ],
            )
            self.write_csv(
                guard,
                [
                    "model",
                    "set",
                    "split",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                ],
                [
                    {
                        "model": "baseline",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 631,
                        "filtered_correct": 616,
                        "filtered_wrong": 15,
                        "filtered_precision": 0.976228,
                    },
                    {
                        "model": "candidate",
                        "set": "regression_guard",
                        "split": "test",
                        "filtered_matches": 631,
                        "filtered_correct": 616,
                        "filtered_wrong": 15,
                        "filtered_precision": 0.976228,
                    },
                    {
                        "model": "baseline",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 162,
                        "filtered_correct": 143,
                        "filtered_wrong": 19,
                        "filtered_precision": 0.882716,
                    },
                    {
                        "model": "candidate",
                        "set": "extreme_gain",
                        "split": "test",
                        "filtered_matches": 162,
                        "filtered_correct": 143,
                        "filtered_wrong": 19,
                        "filtered_precision": 0.882716,
                    },
                ],
            )

            decision = promotion_mod.evaluate_promotion(
                formal_summary=formal,
                formal_variant_summary=formal_variant,
                guard_summary=guard,
                baseline_label="baseline",
                candidate_label="candidate",
                splits=["test"],
                formal_target_variants=["extreme_02", "extreme_03"],
                formal_protected_variants=["extreme_01", "mid_01", "mid_02"],
                min_formal_target_correct_gain=1,
                min_formal_target_total_correct_gain=1,
                max_formal_target_wrong_increase=1,
                max_formal_target_precision_drop=0.0,
                max_protected_variant_precision_drop=0.0,
                max_protected_variant_correct_drop=0,
                max_protected_variant_wrong_increase=0,
                max_formal_precision_drop=0.0,
                max_formal_correct_drop=0,
                max_formal_wrong_increase=0,
                max_guard_precision_drop=0.0,
                max_guard_correct_drop=0,
                max_guard_wrong_increase=0,
                min_extreme_correct_gain=0,
            )

            self.assertTrue(decision.promote)
            self.assertEqual(decision.failed_reasons, [])
            self.assertTrue(
                any("formal_target_variants:extreme_02,extreme_03/test" in reason for reason in decision.passed_reasons)
            )
            self.assertTrue(any("formal_target_total/all" in reason for reason in decision.passed_reasons))
            self.assertTrue(any("formal_protected_variant:mid_01/test" in reason for reason in decision.passed_reasons))

    def test_fov76_promotion_pipeline_builds_formal_guard_and_gate_commands(self) -> None:
        args = SimpleNamespace(
            pair_root=Path("/data/pairs"),
            guard_root=Path("/data/pairs/hard_mining/guard"),
            output_dir=Path("/out/eval"),
            baseline_state=Path("/runs/base/state.pt"),
            baseline_run_dir=Path("/runs/base/train_output"),
            candidate_state=Path("/runs/cand/state.pt"),
            candidate_run_dir=Path("/runs/cand/train_output"),
            baseline_label="phase2h_ransac",
            candidate_label="phase3f_ransac",
            guard_baseline_label="phase2h",
            guard_candidate_label="phase3f",
            splits=["val", "test"],
            device="cuda",
            python_executable="/env/bin/python",
            seed=20260614,
            crop_size=2048,
            max_image_size=768,
            max_keypoints=512,
            matcher_candidate_topk=256,
            graph_layers=4,
            geometry_threshold_px=10.0,
            filtered_min_matches=16,
            filtered_min_matches_by_variant=["extreme_02=8,extreme_03=8"],
            baseline_filtered_min_matches_by_variant=[],
            candidate_filtered_min_matches_by_variant=[],
            write_match_details=True,
            max_formal_precision_drop=0.0,
            max_formal_correct_drop=0,
            max_formal_wrong_increase=0,
            formal_target_variants="extreme_02,extreme_03",
            formal_protected_variants="extreme_01,mid_01,mid_02,nadir",
            min_formal_target_correct_gain=1,
            min_formal_target_total_correct_gain=1,
            max_formal_target_precision_drop=0.001,
            max_formal_target_wrong_increase=1,
            max_protected_variant_precision_drop=0.0,
            max_protected_variant_correct_drop=0,
            max_protected_variant_wrong_increase=0,
            max_guard_precision_drop=0.0,
            max_guard_correct_drop=0,
            max_guard_wrong_increase=0,
            extra_regression_guard_set=["phase5h_false_cluster_guard"],
            max_extra_guard_precision_drop=0.03,
            max_extra_guard_correct_drop=0,
            max_extra_guard_wrong_increase=1,
            min_extreme_correct_gain=1,
            max_extreme_precision_drop=0.02,
            max_extreme_wrong_increase=999,
        )
        model = fov76_gate_mod.EvalModel(
            label=args.candidate_label,
            guard_label=args.guard_candidate_label,
            state=args.candidate_state,
            run_dir=args.candidate_run_dir,
        )

        formal = fov76_gate_mod.build_formal_sweep_command(args, model=model, split="test")
        guard = fov76_gate_mod.build_guard_sweep_command(
            args,
            model=model,
            set_name="regression_guard",
            split="val",
        )
        extra_guard = fov76_gate_mod.build_guard_sweep_command(
            args,
            model=model,
            set_name="phase5h_false_cluster_guard",
            split="test",
        )
        promotion = fov76_gate_mod.build_promotion_command(
            args,
            formal_summary=Path("/out/eval/formal_summary.csv"),
            formal_variant_summary=Path("/out/eval/formal_variant_summary.csv"),
            guard_summary=Path("/out/eval/guard_summary.csv"),
        )

        self.assertTrue(any(item.endswith("scripts/run_graph_filter_sweep.py") for item in formal))
        self.assertEqual(formal[formal.index("--candidate-pairs") + 1], "60")
        self.assertEqual(formal[formal.index("--pair-spec-manifest") + 1], "/data/pairs/overlap_edges_test.csv")
        self.assertEqual(formal[formal.index("--pytorch-state") + 1], "/runs/cand/state.pt")
        self.assertEqual(formal[formal.index("--matcher-candidate-topk") + 1], "256")
        self.assertEqual(formal[formal.index("--graph-max-attention-layers") + 1], "4")
        self.assertIn("--filtered-min-matches-by-variant", formal)
        self.assertEqual(
            formal[formal.index("--filtered-min-matches-by-variant") + 1],
            "extreme_02=8,extreme_03=8",
        )
        self.assertIn("--write-all-summary", formal)
        self.assertIn("--write-match-details", formal)

        self.assertEqual(guard[guard.index("--candidate-pairs") + 1], "100")
        self.assertEqual(
            guard[guard.index("--pair-spec-manifest") + 1],
            "/data/pairs/hard_mining/guard/regression_guard_val.csv",
        )
        self.assertEqual(guard[guard.index("--split") + 1], "val")
        self.assertEqual(
            extra_guard[extra_guard.index("--pair-spec-manifest") + 1],
            "/data/pairs/hard_mining/guard/phase5h_false_cluster_guard_test.csv",
        )

        self.assertTrue(any(item.endswith("scripts/evaluate_checkpoint_promotion.py") for item in promotion))
        self.assertEqual(promotion[promotion.index("--baseline-label") + 1], "phase2h_ransac")
        self.assertEqual(promotion[promotion.index("--candidate-label") + 1], "phase3f_ransac")
        self.assertEqual(promotion[promotion.index("--guard-baseline-label") + 1], "phase2h")
        self.assertEqual(promotion[promotion.index("--guard-candidate-label") + 1], "phase3f")
        self.assertEqual(promotion[promotion.index("--splits") + 1], "val,test")
        self.assertEqual(promotion[promotion.index("--formal-variant-summary") + 1], "/out/eval/formal_variant_summary.csv")
        self.assertEqual(promotion[promotion.index("--formal-target-variants") + 1], "extreme_02,extreme_03")
        self.assertEqual(promotion[promotion.index("--min-formal-target-total-correct-gain") + 1], "1")
        self.assertEqual(promotion[promotion.index("--max-formal-target-wrong-increase") + 1], "1")
        self.assertEqual(
            promotion[promotion.index("--extra-regression-guard-set") + 1],
            "phase5h_false_cluster_guard",
        )
        self.assertEqual(promotion[promotion.index("--max-extra-guard-precision-drop") + 1], "0.03")
        self.assertEqual(promotion[promotion.index("--max-extra-guard-correct-drop") + 1], "0")
        self.assertEqual(promotion[promotion.index("--max-extra-guard-wrong-increase") + 1], "1")

    def test_fov76_promotion_pipeline_allows_larger_validation_pair_counts(self) -> None:
        args = SimpleNamespace(
            pair_root=Path("/data/pairs"),
            guard_root=Path("/data/pairs/hard_mining/guard"),
            output_dir=Path("/out/eval"),
            baseline_state=Path("/runs/base/state.pt"),
            baseline_run_dir=Path("/runs/base/train_output"),
            candidate_state=Path("/runs/cand/state.pt"),
            candidate_run_dir=Path("/runs/cand/train_output"),
            baseline_label="phase2h_geo5_strict_rescue",
            candidate_label="phase2h_geo5_gain5_geo10_rescue",
            guard_baseline_label="phase2h_geo5_strict_rescue",
            guard_candidate_label="phase2h_geo5_gain5_geo10_rescue",
            splits=["val", "test"],
            device="cuda",
            python_executable="/env/bin/python",
            seed=20260614,
            crop_size=2048,
            max_image_size=768,
            max_keypoints=512,
            matcher_candidate_topk=256,
            graph_layers=4,
            geometry_threshold_px=5.0,
            filtered_min_matches=16,
            filtered_min_matches_by_variant=[],
            baseline_filtered_min_matches_by_variant=[],
            candidate_filtered_min_matches_by_variant=[],
            formal_candidate_pairs=180,
            guard_candidate_pairs=240,
            adaptive_geometry_rescue_variants="",
            baseline_adaptive_geometry_rescue_variants="",
            candidate_adaptive_geometry_rescue_variants="extreme_02,extreme_03",
            adaptive_geometry_rescue_threshold_px=10.0,
            adaptive_geometry_rescue_min_match_gain=5,
            adaptive_geometry_rescue_max_base_matches=16,
            adaptive_geometry_rescue_max_homography_p90_px=4.5,
            adaptive_geometry_rescue_max_homography_median_px=3.0,
            adaptive_geometry_rescue_require_score_mean_not_lower=False,
            write_match_details=False,
        )
        model = fov76_gate_mod.EvalModel(
            label=args.candidate_label,
            guard_label=args.guard_candidate_label,
            state=args.candidate_state,
            run_dir=args.candidate_run_dir,
        )

        formal = fov76_gate_mod.build_formal_sweep_command(args, model=model, split="test")
        guard = fov76_gate_mod.build_guard_sweep_command(
            args,
            model=model,
            set_name="regression_guard",
            split="val",
        )

        self.assertEqual(formal[formal.index("--candidate-pairs") + 1], "180")
        self.assertEqual(guard[guard.index("--candidate-pairs") + 1], "240")

    def test_fov76_promotion_pipeline_allows_candidate_only_variant_min_match_gate(self) -> None:
        args = SimpleNamespace(
            pair_root=Path("/data/pairs"),
            guard_root=Path("/data/pairs/hard_mining/guard"),
            output_dir=Path("/out/eval"),
            baseline_state=Path("/runs/base/state.pt"),
            baseline_run_dir=Path("/runs/base/train_output"),
            candidate_state=Path("/runs/cand/state.pt"),
            candidate_run_dir=Path("/runs/cand/train_output"),
            baseline_label="phase2h_default",
            candidate_label="phase2h_extreme_min8",
            guard_baseline_label="phase2h_default",
            guard_candidate_label="phase2h_extreme_min8",
            splits=["val", "test"],
            device="cuda",
            python_executable="/env/bin/python",
            seed=20260614,
            crop_size=2048,
            max_image_size=768,
            max_keypoints=512,
            matcher_candidate_topk=256,
            graph_layers=4,
            geometry_threshold_px=10.0,
            filtered_min_matches=16,
            filtered_min_matches_by_variant=[],
            baseline_filtered_min_matches_by_variant=[],
            candidate_filtered_min_matches_by_variant=["extreme_02=8,extreme_03=8"],
            max_formal_precision_drop=0.0,
            max_formal_correct_drop=0,
            max_formal_wrong_increase=0,
            max_guard_precision_drop=0.0,
            max_guard_correct_drop=0,
            max_guard_wrong_increase=0,
            min_extreme_correct_gain=1,
            max_extreme_precision_drop=0.02,
            max_extreme_wrong_increase=999,
        )
        baseline = fov76_gate_mod.EvalModel(
            label=args.baseline_label,
            guard_label=args.guard_baseline_label,
            state=args.baseline_state,
            run_dir=args.baseline_run_dir,
        )
        candidate = fov76_gate_mod.EvalModel(
            label=args.candidate_label,
            guard_label=args.guard_candidate_label,
            state=args.candidate_state,
            run_dir=args.candidate_run_dir,
        )

        baseline_command = fov76_gate_mod.build_formal_sweep_command(args, model=baseline, split="test")
        candidate_command = fov76_gate_mod.build_formal_sweep_command(args, model=candidate, split="test")

        self.assertNotIn("--filtered-min-matches-by-variant", baseline_command)
        self.assertIn("--filtered-min-matches-by-variant", candidate_command)
        self.assertEqual(
            candidate_command[candidate_command.index("--filtered-min-matches-by-variant") + 1],
            "extreme_02=8,extreme_03=8",
        )

    def test_fov76_promotion_pipeline_allows_candidate_only_adaptive_rescue(self) -> None:
        args = SimpleNamespace(
            pair_root=Path("/data/pairs"),
            guard_root=Path("/data/pairs/hard_mining/guard"),
            output_dir=Path("/out/eval"),
            baseline_state=Path("/runs/base/state.pt"),
            baseline_run_dir=Path("/runs/base/train_output"),
            candidate_state=Path("/runs/cand/state.pt"),
            candidate_run_dir=Path("/runs/cand/train_output"),
            baseline_label="phase2h_default",
            candidate_label="phase2h_adaptive",
            guard_baseline_label="phase2h_default",
            guard_candidate_label="phase2h_adaptive",
            splits=["val", "test"],
            device="cuda",
            python_executable="/env/bin/python",
            seed=20260614,
            crop_size=2048,
            max_image_size=768,
            max_keypoints=512,
            matcher_candidate_topk=256,
            graph_layers=4,
            geometry_threshold_px=5.0,
            filtered_min_matches=16,
            filtered_min_matches_by_variant=[],
            baseline_filtered_min_matches_by_variant=[],
            candidate_filtered_min_matches_by_variant=[],
            adaptive_geometry_rescue_variants="",
            baseline_adaptive_geometry_rescue_variants="",
            candidate_adaptive_geometry_rescue_variants="extreme_02,extreme_03",
            adaptive_geometry_rescue_threshold_px=10.0,
            adaptive_geometry_rescue_min_match_gain=20,
            adaptive_geometry_rescue_max_base_matches=0,
            adaptive_geometry_rescue_max_homography_p90_px=4.5,
            adaptive_geometry_rescue_max_homography_median_px=1.8,
            adaptive_geometry_rescue_require_score_mean_not_lower=True,
            max_formal_precision_drop=0.0,
            max_formal_correct_drop=0,
            max_formal_wrong_increase=0,
            max_guard_precision_drop=0.0,
            max_guard_correct_drop=0,
            max_guard_wrong_increase=0,
            min_extreme_correct_gain=1,
            max_extreme_precision_drop=0.02,
            max_extreme_wrong_increase=999,
        )
        baseline = fov76_gate_mod.EvalModel(
            label=args.baseline_label,
            guard_label=args.guard_baseline_label,
            state=args.baseline_state,
            run_dir=args.baseline_run_dir,
        )
        candidate = fov76_gate_mod.EvalModel(
            label=args.candidate_label,
            guard_label=args.guard_candidate_label,
            state=args.candidate_state,
            run_dir=args.candidate_run_dir,
        )

        baseline_command = fov76_gate_mod.build_formal_sweep_command(args, model=baseline, split="test")
        candidate_command = fov76_gate_mod.build_formal_sweep_command(args, model=candidate, split="test")

        self.assertNotIn("--adaptive-geometry-rescue-variants", baseline_command)
        self.assertIn("--adaptive-geometry-rescue-variants", candidate_command)
        self.assertEqual(
            candidate_command[candidate_command.index("--adaptive-geometry-rescue-variants") + 1],
            "extreme_02,extreme_03",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--adaptive-geometry-rescue-threshold-px") + 1],
            "10.0",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--adaptive-geometry-rescue-min-match-gain") + 1],
            "20",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--adaptive-geometry-rescue-max-base-matches") + 1],
            "0",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--adaptive-geometry-rescue-max-homography-p90-px") + 1],
            "4.5",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--adaptive-geometry-rescue-max-homography-median-px") + 1],
            "1.8",
        )
        self.assertIn("--adaptive-geometry-rescue-require-score-mean-not-lower", candidate_command)

    def test_fov76_promotion_pipeline_allows_candidate_only_low_match_geometry_guard(self) -> None:
        args = SimpleNamespace(
            pair_root=Path("/data/pairs"),
            guard_root=Path("/data/pairs/hard_mining/guard"),
            output_dir=Path("/out/eval"),
            baseline_state=Path("/runs/base/state.pt"),
            baseline_run_dir=Path("/runs/base/train_output"),
            candidate_state=Path("/runs/cand/state.pt"),
            candidate_run_dir=Path("/runs/cand/train_output"),
            baseline_label="phase2h_default",
            candidate_label="phase2h_low_match_guard",
            guard_baseline_label="phase2h_default",
            guard_candidate_label="phase2h_low_match_guard",
            splits=["val", "test"],
            device="cuda",
            python_executable="/env/bin/python",
            seed=20260614,
            crop_size=2048,
            max_image_size=768,
            max_keypoints=512,
            matcher_candidate_topk=256,
            graph_layers=4,
            geometry_threshold_px=5.0,
            filtered_min_matches=16,
            filtered_min_matches_by_variant=[],
            baseline_filtered_min_matches_by_variant=[],
            candidate_filtered_min_matches_by_variant=[],
            adaptive_geometry_rescue_variants="",
            baseline_adaptive_geometry_rescue_variants="",
            candidate_adaptive_geometry_rescue_variants="",
            adaptive_geometry_rescue_threshold_px=0.0,
            adaptive_geometry_rescue_min_match_gain=0,
            adaptive_geometry_rescue_max_base_matches=-1,
            adaptive_geometry_rescue_max_homography_p90_px=-1.0,
            adaptive_geometry_rescue_max_homography_median_px=-1.0,
            adaptive_geometry_rescue_require_score_mean_not_lower=False,
            low_match_geometry_guard_variants="",
            baseline_low_match_geometry_guard_variants="",
            candidate_low_match_geometry_guard_variants="extreme_02,extreme_03",
            low_match_geometry_guard_min_matches=0,
            low_match_geometry_guard_max_matches=-1,
            low_match_geometry_guard_max_homography_p90_px=-1.0,
            low_match_geometry_guard_max_homography_median_px=-1.0,
            low_match_geometry_guard_min_score_mean=float("-inf"),
            baseline_low_match_geometry_guard_min_matches=None,
            baseline_low_match_geometry_guard_max_matches=None,
            baseline_low_match_geometry_guard_max_homography_p90_px=None,
            baseline_low_match_geometry_guard_max_homography_median_px=None,
            baseline_low_match_geometry_guard_min_score_mean=None,
            candidate_low_match_geometry_guard_min_matches=12,
            candidate_low_match_geometry_guard_max_matches=15,
            candidate_low_match_geometry_guard_max_homography_p90_px=2.8,
            candidate_low_match_geometry_guard_max_homography_median_px=1.5,
            candidate_low_match_geometry_guard_min_score_mean=19.0,
        )
        baseline = fov76_gate_mod.EvalModel(
            label=args.baseline_label,
            guard_label=args.guard_baseline_label,
            state=args.baseline_state,
            run_dir=args.baseline_run_dir,
        )
        candidate = fov76_gate_mod.EvalModel(
            label=args.candidate_label,
            guard_label=args.guard_candidate_label,
            state=args.candidate_state,
            run_dir=args.candidate_run_dir,
        )

        baseline_command = fov76_gate_mod.build_formal_sweep_command(args, model=baseline, split="test")
        candidate_command = fov76_gate_mod.build_formal_sweep_command(args, model=candidate, split="test")

        self.assertNotIn("--low-match-geometry-guard-variants", baseline_command)
        self.assertIn("--low-match-geometry-guard-variants", candidate_command)
        self.assertEqual(
            candidate_command[candidate_command.index("--low-match-geometry-guard-variants") + 1],
            "extreme_02,extreme_03",
        )
        self.assertEqual(candidate_command[candidate_command.index("--low-match-geometry-guard-min-matches") + 1], "12")
        self.assertEqual(candidate_command[candidate_command.index("--low-match-geometry-guard-max-matches") + 1], "15")
        self.assertEqual(
            candidate_command[candidate_command.index("--low-match-geometry-guard-max-homography-p90-px") + 1],
            "2.8",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--low-match-geometry-guard-max-homography-median-px") + 1],
            "1.5",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--low-match-geometry-guard-min-score-mean") + 1],
            "19.0",
        )

    def test_fov76_promotion_pipeline_allows_per_side_adaptive_rescue_thresholds(self) -> None:
        args = SimpleNamespace(
            pair_root=Path("/data/pairs"),
            guard_root=Path("/data/pairs/hard_mining/guard"),
            output_dir=Path("/out/eval"),
            baseline_state=Path("/runs/base/state.pt"),
            baseline_run_dir=Path("/runs/base/train_output"),
            candidate_state=Path("/runs/cand/state.pt"),
            candidate_run_dir=Path("/runs/cand/train_output"),
            baseline_label="phase2h_default",
            candidate_label="phase2h_rescue",
            guard_baseline_label="phase2h_default",
            guard_candidate_label="phase2h_rescue",
            splits=["val", "test"],
            device="cuda",
            python_executable="/env/bin/python",
            seed=20260614,
            crop_size=2048,
            max_image_size=768,
            max_keypoints=512,
            matcher_candidate_topk=256,
            graph_layers=4,
            geometry_threshold_px=5.0,
            filtered_min_matches=16,
            filtered_min_matches_by_variant=[],
            baseline_filtered_min_matches_by_variant=[],
            candidate_filtered_min_matches_by_variant=[],
            adaptive_geometry_rescue_variants="",
            baseline_adaptive_geometry_rescue_variants="extreme_02,extreme_03",
            candidate_adaptive_geometry_rescue_variants="extreme_02,extreme_03",
            adaptive_geometry_rescue_threshold_px=10.0,
            adaptive_geometry_rescue_min_match_gain=20,
            adaptive_geometry_rescue_max_base_matches=0,
            adaptive_geometry_rescue_max_homography_p90_px=4.5,
            adaptive_geometry_rescue_max_homography_median_px=1.8,
            adaptive_geometry_rescue_require_score_mean_not_lower=True,
            baseline_adaptive_geometry_rescue_threshold_px=10.0,
            baseline_adaptive_geometry_rescue_min_match_gain=20,
            baseline_adaptive_geometry_rescue_max_base_matches=0,
            baseline_adaptive_geometry_rescue_max_homography_p90_px=4.5,
            baseline_adaptive_geometry_rescue_max_homography_median_px=1.8,
            baseline_adaptive_geometry_rescue_require_score_mean_not_lower=True,
            candidate_adaptive_geometry_rescue_threshold_px=10.0,
            candidate_adaptive_geometry_rescue_min_match_gain=5,
            candidate_adaptive_geometry_rescue_max_base_matches=16,
            candidate_adaptive_geometry_rescue_max_homography_p90_px=4.5,
            candidate_adaptive_geometry_rescue_max_homography_median_px=3.0,
            candidate_adaptive_geometry_rescue_require_score_mean_not_lower=False,
            max_formal_precision_drop=0.0,
            max_formal_correct_drop=0,
            max_formal_wrong_increase=0,
            max_guard_precision_drop=0.0,
            max_guard_correct_drop=0,
            max_guard_wrong_increase=0,
            min_extreme_correct_gain=1,
            max_extreme_precision_drop=0.02,
            max_extreme_wrong_increase=999,
        )
        baseline = fov76_gate_mod.EvalModel(
            label=args.baseline_label,
            guard_label=args.guard_baseline_label,
            state=args.baseline_state,
            run_dir=args.baseline_run_dir,
        )
        candidate = fov76_gate_mod.EvalModel(
            label=args.candidate_label,
            guard_label=args.guard_candidate_label,
            state=args.candidate_state,
            run_dir=args.candidate_run_dir,
        )

        baseline_command = fov76_gate_mod.build_formal_sweep_command(args, model=baseline, split="test")
        candidate_command = fov76_gate_mod.build_formal_sweep_command(args, model=candidate, split="test")

        self.assertEqual(
            baseline_command[baseline_command.index("--adaptive-geometry-rescue-min-match-gain") + 1],
            "20",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--adaptive-geometry-rescue-min-match-gain") + 1],
            "5",
        )
        self.assertEqual(
            baseline_command[baseline_command.index("--adaptive-geometry-rescue-max-base-matches") + 1],
            "0",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--adaptive-geometry-rescue-max-base-matches") + 1],
            "16",
        )
        self.assertEqual(
            baseline_command[baseline_command.index("--adaptive-geometry-rescue-max-homography-median-px") + 1],
            "1.8",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--adaptive-geometry-rescue-max-homography-median-px") + 1],
            "3.0",
        )
        self.assertIn("--adaptive-geometry-rescue-require-score-mean-not-lower", baseline_command)
        self.assertNotIn("--adaptive-geometry-rescue-require-score-mean-not-lower", candidate_command)

    def test_fov76_promotion_pipeline_profile_sets_geo5_geo10_extreme_rescue(self) -> None:
        args = SimpleNamespace(
            pair_root=Path("/data/pairs"),
            guard_root=Path("/data/pairs/hard_mining/guard"),
            output_dir=Path("/out/eval"),
            baseline_state=Path("/runs/base/state.pt"),
            baseline_run_dir=Path("/runs/base/train_output"),
            candidate_state=Path("/runs/cand/state.pt"),
            candidate_run_dir=Path("/runs/cand/train_output"),
            baseline_label="phase2h_geo5_strict_rescue",
            candidate_label="phase2h_geo5_gain5_geo10_rescue",
            guard_baseline_label="phase2h_geo5_strict_rescue",
            guard_candidate_label="phase2h_geo5_gain5_geo10_rescue",
            splits=["val", "test"],
            device="cuda",
            python_executable="/env/bin/python",
            seed=20260614,
            crop_size=2048,
            max_image_size=768,
            max_keypoints=512,
            matcher_candidate_topk=256,
            graph_layers=4,
            geometry_threshold_px=10.0,
            filtered_min_matches=16,
            filtered_min_matches_by_variant=[],
            baseline_filtered_min_matches_by_variant=[],
            candidate_filtered_min_matches_by_variant=[],
            post_filter_profile="fov76_geo5_geo10_extreme_rescue",
            adaptive_geometry_rescue_variants="",
            baseline_adaptive_geometry_rescue_variants="",
            candidate_adaptive_geometry_rescue_variants="",
            adaptive_geometry_rescue_threshold_px=0.0,
            adaptive_geometry_rescue_min_match_gain=0,
            adaptive_geometry_rescue_max_base_matches=-1,
            adaptive_geometry_rescue_max_homography_p90_px=-1.0,
            adaptive_geometry_rescue_max_homography_median_px=-1.0,
            adaptive_geometry_rescue_require_score_mean_not_lower=False,
            baseline_adaptive_geometry_rescue_threshold_px=None,
            baseline_adaptive_geometry_rescue_min_match_gain=None,
            baseline_adaptive_geometry_rescue_max_base_matches=None,
            baseline_adaptive_geometry_rescue_max_homography_p90_px=None,
            baseline_adaptive_geometry_rescue_max_homography_median_px=None,
            baseline_adaptive_geometry_rescue_require_score_mean_not_lower=None,
            candidate_adaptive_geometry_rescue_threshold_px=None,
            candidate_adaptive_geometry_rescue_min_match_gain=None,
            candidate_adaptive_geometry_rescue_max_base_matches=None,
            candidate_adaptive_geometry_rescue_max_homography_p90_px=None,
            candidate_adaptive_geometry_rescue_max_homography_median_px=None,
            candidate_adaptive_geometry_rescue_require_score_mean_not_lower=None,
            max_formal_precision_drop=0.0,
            max_formal_correct_drop=0,
            max_formal_wrong_increase=0,
            formal_target_variants="",
            formal_protected_variants="",
            min_formal_target_correct_gain=0,
            min_formal_target_total_correct_gain=0,
            max_formal_target_precision_drop=0.0,
            max_formal_target_wrong_increase=0,
            max_protected_variant_precision_drop=0.0,
            max_protected_variant_correct_drop=0,
            max_protected_variant_wrong_increase=0,
            max_guard_precision_drop=0.0,
            max_guard_correct_drop=0,
            max_guard_wrong_increase=0,
            min_extreme_correct_gain=1,
            max_extreme_precision_drop=0.02,
            max_extreme_wrong_increase=10**12,
        )

        fov76_gate_mod.apply_post_filter_profile(args)
        baseline = fov76_gate_mod.EvalModel(
            label=args.baseline_label,
            guard_label=args.guard_baseline_label,
            state=args.baseline_state,
            run_dir=args.baseline_run_dir,
        )
        candidate = fov76_gate_mod.EvalModel(
            label=args.candidate_label,
            guard_label=args.guard_candidate_label,
            state=args.candidate_state,
            run_dir=args.candidate_run_dir,
        )

        baseline_command = fov76_gate_mod.build_formal_sweep_command(args, model=baseline, split="test")
        candidate_command = fov76_gate_mod.build_formal_sweep_command(args, model=candidate, split="test")
        promotion_command = fov76_gate_mod.build_promotion_command(
            args,
            formal_summary=args.output_dir / "formal_summary.csv",
            guard_summary=args.output_dir / "guard_summary.csv",
            formal_variant_summary=args.output_dir / "formal_variant_summary.csv",
        )

        self.assertEqual(baseline_command[baseline_command.index("--geometry-threshold-px") + 1], "5.0")
        self.assertEqual(baseline_command[baseline_command.index("--adaptive-geometry-rescue-threshold-px") + 1], "10.0")
        self.assertEqual(baseline_command[baseline_command.index("--adaptive-geometry-rescue-min-match-gain") + 1], "20")
        self.assertEqual(candidate_command[candidate_command.index("--adaptive-geometry-rescue-min-match-gain") + 1], "5")
        self.assertEqual(candidate_command[candidate_command.index("--adaptive-geometry-rescue-max-base-matches") + 1], "16")
        self.assertEqual(candidate_command[candidate_command.index("--adaptive-geometry-rescue-max-homography-p90-px") + 1], "4.2")
        self.assertEqual(candidate_command[candidate_command.index("--adaptive-geometry-rescue-max-homography-median-px") + 1], "2.3")
        self.assertIn("--adaptive-geometry-rescue-require-score-mean-not-lower", baseline_command)
        self.assertNotIn("--adaptive-geometry-rescue-require-score-mean-not-lower", candidate_command)
        self.assertEqual(promotion_command[promotion_command.index("--min-extreme-correct-gain") + 1], "0")
        self.assertEqual(promotion_command[promotion_command.index("--formal-target-variants") + 1], "extreme_02,extreme_03")
        self.assertEqual(
            promotion_command[promotion_command.index("--formal-protected-variants") + 1],
            "mid_01,mid_02,extreme_01,nadir",
        )
        self.assertEqual(promotion_command[promotion_command.index("--max-formal-target-precision-drop") + 1], "0.0")
        self.assertEqual(promotion_command[promotion_command.index("--max-formal-target-wrong-increase") + 1], "1")
        self.assertEqual(promotion_command[promotion_command.index("--max-guard-precision-drop") + 1], "0.0")
        self.assertEqual(promotion_command[promotion_command.index("--max-guard-wrong-increase") + 1], "0")
        self.assertEqual(promotion_command[promotion_command.index("--min-formal-target-total-correct-gain") + 1], "1")

    def test_fov76_promotion_pipeline_profile_sets_active_low_match_guard(self) -> None:
        argv = [
            "run_fov76_checkpoint_promotion_pipeline.py",
            "--pair-root",
            "/data/pairs",
            "--guard-root",
            "/data/pairs/hard_mining/guard",
            "--output-dir",
            "/out/eval",
            "--baseline-state",
            "/runs/base/state.pt",
            "--baseline-run-dir",
            "/runs/base/train_output",
            "--candidate-state",
            "/runs/cand/state.pt",
            "--candidate-run-dir",
            "/runs/cand/train_output",
            "--candidate-label",
            "candidate_active",
            "--guard-candidate-label",
            "candidate_active",
            "--post-filter-profile",
            "fov76_geo5_geo10_extreme_rescue_lowmatch_guard",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = fov76_gate_mod.parse_args()
        fov76_gate_mod.apply_post_filter_profile(args)
        selector_config = fov76_gate_mod._dual_selector_config_from_args(args)
        baseline = fov76_gate_mod.EvalModel(
            label=args.baseline_label,
            guard_label=args.guard_baseline_label,
            state=args.baseline_state,
            run_dir=args.baseline_run_dir,
        )
        candidate = fov76_gate_mod.EvalModel(
            label=args.candidate_label,
            guard_label=args.guard_candidate_label,
            state=args.candidate_state,
            run_dir=args.candidate_run_dir,
        )

        baseline_command = fov76_gate_mod.build_formal_sweep_command(args, model=baseline, split="test")
        candidate_command = fov76_gate_mod.build_formal_sweep_command(args, model=candidate, split="test")
        promotion_command = fov76_gate_mod.build_promotion_command(
            args,
            formal_summary=args.output_dir / "formal_summary.csv",
            guard_summary=args.output_dir / "guard_summary.csv",
            formal_variant_summary=args.output_dir / "formal_variant_summary.csv",
        )

        for command in (baseline_command, candidate_command):
            self.assertEqual(command[command.index("--geometry-threshold-px") + 1], "5.0")
            self.assertEqual(command[command.index("--adaptive-geometry-rescue-variants") + 1], "extreme_02,extreme_03")
            self.assertEqual(command[command.index("--adaptive-geometry-rescue-threshold-px") + 1], "10.0")
            self.assertEqual(command[command.index("--adaptive-geometry-rescue-min-match-gain") + 1], "5")
            self.assertEqual(command[command.index("--adaptive-geometry-rescue-max-base-matches") + 1], "16")
            self.assertEqual(command[command.index("--adaptive-geometry-rescue-max-homography-p90-px") + 1], "4.2")
            self.assertEqual(command[command.index("--adaptive-geometry-rescue-max-homography-median-px") + 1], "2.3")
            self.assertNotIn("--adaptive-geometry-rescue-require-score-mean-not-lower", command)
            self.assertEqual(command[command.index("--low-match-geometry-guard-variants") + 1], "extreme_02,extreme_03")
            self.assertEqual(command[command.index("--low-match-geometry-guard-min-matches") + 1], "12")
            self.assertEqual(command[command.index("--low-match-geometry-guard-max-matches") + 1], "15")
            self.assertEqual(command[command.index("--low-match-geometry-guard-max-homography-p90-px") + 1], "2.8")
            self.assertEqual(command[command.index("--low-match-geometry-guard-max-homography-median-px") + 1], "1.5")
            self.assertEqual(command[command.index("--low-match-geometry-guard-min-score-mean") + 1], "19.0")

        self.assertEqual(promotion_command[promotion_command.index("--min-extreme-correct-gain") + 1], "0")
        self.assertEqual(promotion_command[promotion_command.index("--formal-target-variants") + 1], "extreme_02,extreme_03")
        self.assertEqual(
            promotion_command[promotion_command.index("--formal-protected-variants") + 1],
            "mid_01,mid_02,extreme_01,nadir",
        )
        self.assertEqual(promotion_command[promotion_command.index("--max-formal-target-precision-drop") + 1], "0.0")
        self.assertEqual(promotion_command[promotion_command.index("--max-formal-target-wrong-increase") + 1], "1")
        self.assertEqual(promotion_command[promotion_command.index("--max-guard-precision-drop") + 1], "0.0")
        self.assertEqual(promotion_command[promotion_command.index("--max-guard-wrong-increase") + 1], "0")
        self.assertEqual(args.dual_checkpoint_rescue_min_rescue_matches, 16)
        self.assertEqual(args.dual_checkpoint_rescue_min_match_gain, 3)
        self.assertEqual(selector_config.min_rescue_matches, 16)
        self.assertEqual(selector_config.min_match_gain, 3)

    def test_fov76_low_match_guard_profile_preserves_explicit_dual_selector_minmatch(self) -> None:
        argv = [
            "run_fov76_checkpoint_promotion_pipeline.py",
            "--pair-root",
            "/data/pairs",
            "--guard-root",
            "/data/pairs/hard_mining/guard",
            "--output-dir",
            "/out/eval",
            "--baseline-state",
            "/runs/base/state.pt",
            "--baseline-run-dir",
            "/runs/base/train_output",
            "--candidate-state",
            "/runs/cand/state.pt",
            "--candidate-run-dir",
            "/runs/cand/train_output",
            "--candidate-label",
            "candidate_active",
            "--guard-candidate-label",
            "candidate_active",
            "--post-filter-profile",
            "fov76_geo5_geo10_extreme_rescue_lowmatch_guard",
            "--dual-checkpoint-rescue-min-match-gain",
            "2",
            "--dual-checkpoint-rescue-min-rescue-matches",
            "8",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = fov76_gate_mod.parse_args()
        fov76_gate_mod.apply_post_filter_profile(args)
        selector_config = fov76_gate_mod._dual_selector_config_from_args(args)

        self.assertEqual(args.dual_checkpoint_rescue_min_rescue_matches, 8)
        self.assertEqual(args.dual_checkpoint_rescue_min_match_gain, 2)
        self.assertEqual(selector_config.min_rescue_matches, 8)
        self.assertEqual(selector_config.min_match_gain, 2)

    def test_fov76_promotion_pipeline_dual_rescue_profile_sets_ransac_minmatch16(self) -> None:
        argv = [
            "run_fov76_checkpoint_promotion_pipeline.py",
            "--pair-root",
            "/data/pairs",
            "--guard-root",
            "/data/pairs/hard_mining/guard",
            "--output-dir",
            "/out/eval",
            "--baseline-state",
            "/runs/base/state.pt",
            "--baseline-run-dir",
            "/runs/base/train_output",
            "--candidate-state",
            "/runs/cand/state.pt",
            "--candidate-run-dir",
            "/runs/cand/train_output",
            "--candidate-label",
            "phase5e_ransac_consistency",
            "--guard-candidate-label",
            "phase5e_ransac_consistency",
            "--dual-checkpoint-rescue-profile",
            "fov76_ransac_minmatch16",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = fov76_gate_mod.parse_args()

        self.assertEqual(args.dual_checkpoint_rescue_min_rescue_matches, 8)
        self.assertEqual(args.dual_checkpoint_rescue_min_match_gain, 1)
        fov76_gate_mod.apply_dual_checkpoint_rescue_profile(args)
        config = fov76_gate_mod._dual_selector_config_from_args(args)

        self.assertEqual(args.dual_checkpoint_rescue_min_rescue_matches, 16)
        self.assertEqual(args.dual_checkpoint_rescue_min_match_gain, 3)
        self.assertEqual(config.min_rescue_matches, 16)
        self.assertEqual(config.min_match_gain, 3)

    def test_fov76_promotion_pipeline_dual_rescue_profile_preserves_explicit_minmatch(self) -> None:
        argv = [
            "run_fov76_checkpoint_promotion_pipeline.py",
            "--pair-root",
            "/data/pairs",
            "--guard-root",
            "/data/pairs/hard_mining/guard",
            "--output-dir",
            "/out/eval",
            "--baseline-state",
            "/runs/base/state.pt",
            "--baseline-run-dir",
            "/runs/base/train_output",
            "--candidate-state",
            "/runs/cand/state.pt",
            "--candidate-run-dir",
            "/runs/cand/train_output",
            "--candidate-label",
            "phase5e_ransac_consistency",
            "--guard-candidate-label",
            "phase5e_ransac_consistency",
            "--dual-checkpoint-rescue-profile",
            "fov76_ransac_minmatch16",
            "--dual-checkpoint-rescue-min-rescue-matches",
            "12",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = fov76_gate_mod.parse_args()

        fov76_gate_mod.apply_dual_checkpoint_rescue_profile(args)

        self.assertEqual(args.dual_checkpoint_rescue_min_rescue_matches, 12)

    def test_fov76_promotion_pipeline_dual_rescue_profile_preserves_explicit_default_minmatch(self) -> None:
        argv = [
            "run_fov76_checkpoint_promotion_pipeline.py",
            "--pair-root",
            "/data/pairs",
            "--guard-root",
            "/data/pairs/hard_mining/guard",
            "--output-dir",
            "/out/eval",
            "--baseline-state",
            "/runs/base/state.pt",
            "--baseline-run-dir",
            "/runs/base/train_output",
            "--candidate-state",
            "/runs/cand/state.pt",
            "--candidate-run-dir",
            "/runs/cand/train_output",
            "--candidate-label",
            "phase5e_ransac_consistency",
            "--guard-candidate-label",
            "phase5e_ransac_consistency",
            "--dual-checkpoint-rescue-profile",
            "fov76_ransac_minmatch16",
            "--dual-checkpoint-rescue-min-rescue-matches",
            "8",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = fov76_gate_mod.parse_args()

        fov76_gate_mod.apply_dual_checkpoint_rescue_profile(args)

        self.assertEqual(args.dual_checkpoint_rescue_min_rescue_matches, 8)

    def test_fov76_promotion_pipeline_dry_run_writes_dual_selector_profile_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_root = root / "pairs"
            guard_root = pair_root / "hard_mining" / "guard"
            output_dir = root / "out"
            base_run = root / "base" / "train_output"
            cand_run = root / "cand" / "train_output"
            for path in [
                pair_root / "manifests",
                guard_root,
                base_run,
                cand_run,
            ]:
                path.mkdir(parents=True, exist_ok=True)
            for path in [
                root / "base" / "state.pt",
                root / "cand" / "state.pt",
                base_run / "train_metrics.csv",
                cand_run / "train_metrics.csv",
                pair_root / "manifests" / "h100km_fov076_render_manifest.csv",
                pair_root / "manifests" / "h100km_fov076_uint8_manifest.csv",
                pair_root / "overlap_edges_val.csv",
                pair_root / "overlap_edges_test.csv",
                guard_root / "regression_guard_val.csv",
                guard_root / "regression_guard_test.csv",
                guard_root / "extreme_gain_val.csv",
                guard_root / "extreme_gain_test.csv",
            ]:
                path.write_text("x\n", encoding="utf-8")
            argv = [
                "run_fov76_checkpoint_promotion_pipeline.py",
                "--pair-root",
                str(pair_root),
                "--guard-root",
                str(guard_root),
                "--output-dir",
                str(output_dir),
                "--baseline-state",
                str(root / "base" / "state.pt"),
                "--baseline-run-dir",
                str(base_run),
                "--candidate-state",
                str(root / "cand" / "state.pt"),
                "--candidate-run-dir",
                str(cand_run),
                "--baseline-label",
                "phase3zn",
                "--candidate-label",
                "phase5e_ransac_consistency",
                "--guard-baseline-label",
                "phase3zn",
                "--guard-candidate-label",
                "phase5e_ransac_consistency",
                "--dual-checkpoint-rescue-selector",
                "--dual-checkpoint-rescue-profile",
                "fov76_ransac_minmatch16",
                "--dry-run",
            ]

            with mock.patch.object(sys, "argv", argv):
                args = fov76_gate_mod.parse_args()
            exit_code = fov76_gate_mod.run_pipeline(args)

            metadata = json.loads((output_dir / "promotion_pipeline_metadata.json").read_text(encoding="utf-8"))
            selector = metadata["dual_checkpoint_rescue"]

            self.assertEqual(exit_code, 0)
            self.assertTrue(selector["enabled"])
            self.assertEqual(selector["profile"], "fov76_ransac_minmatch16")
            self.assertEqual(selector["selected_label"], "dual_checkpoint_rescue_selected")
            self.assertEqual(selector["config"]["min_rescue_matches"], 16)
            self.assertEqual(selector["config"]["target_variants"], ["extreme_02", "extreme_03"])

    def test_fov76_promotion_pipeline_builds_dual_checkpoint_selector_inputs(self) -> None:
        def make_visual_row(
            base_id: str,
            variant: str,
            *,
            matches: int,
            correct: int,
            wrong: int,
            score_mean: float,
            h_median: float = 1.0,
            h_p90: float = 2.0,
        ) -> dict[str, object]:
            return {
                "label": "cfg / all-filtered",
                "base_id": base_id,
                "target_variant": variant,
                "split": "val",
                "valid_fraction": 1.0,
                "matches": matches,
                "correct": correct,
                "wrong": wrong,
                "precision": float(correct) / float(matches) if matches else 0.0,
                "score_mean": score_mean,
                "median_error_px": 0.0,
                "homography_residual_valid": 1,
                "homography_residual_median_px": h_median,
                "homography_residual_p90_px": h_p90,
            }

        def write_sweep(
            output_dir: Path,
            group: str,
            sweep_name: str,
            rows: list[dict[str, object]],
        ) -> None:
            sweep_dir = output_dir / group / sweep_name
            report_dir = sweep_dir / "01_cfg"
            report_dir.mkdir(parents=True)
            fields = list(rows[0].keys())
            self.write_csv(report_dir / "all_filtered_summary.csv", fields, rows)
            matches = sum(int(row["matches"]) for row in rows)
            correct = sum(int(row["correct"]) for row in rows)
            wrong = sum(int(row["wrong"]) for row in rows)
            self.write_csv(
                sweep_dir / "graph_filter_sweep_summary.csv",
                [
                    "geometry_threshold_px",
                    "filtered_min_matches",
                    "filtered_rows",
                    "filtered_matches",
                    "filtered_correct",
                    "filtered_wrong",
                    "filtered_precision",
                    "filtered_median_error_px",
                    "report_dir",
                ],
                [
                    {
                        "geometry_threshold_px": 5.0,
                        "filtered_min_matches": 16,
                        "filtered_rows": len(rows),
                        "filtered_matches": matches,
                        "filtered_correct": correct,
                        "filtered_wrong": wrong,
                        "filtered_precision": f"{(float(correct) / float(matches)):.6f}" if matches else "0.000000",
                        "filtered_median_error_px": 0.0,
                        "report_dir": "01_cfg",
                    }
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "promotion"
            baseline_rows = [
                make_visual_row("pair_mid", "mid_01", matches=485, correct=476, wrong=9, score_mean=20.0),
                make_visual_row("pair_ext2", "extreme_02", matches=10, correct=9, wrong=1, score_mean=20.0),
            ]
            rescue_rows = [
                make_visual_row("pair_mid", "mid_01", matches=500, correct=480, wrong=20, score_mean=25.0),
                make_visual_row("pair_ext2", "extreme_02", matches=12, correct=12, wrong=0, score_mean=21.0),
            ]
            guard_baseline_rows = [
                make_visual_row("pair_guard", "mid_02", matches=599, correct=587, wrong=12, score_mean=20.0),
            ]
            guard_rescue_rows = [
                make_visual_row("pair_guard", "mid_02", matches=594, correct=577, wrong=17, score_mean=25.0),
            ]
            for set_name in ("regression_guard", "extreme_gain"):
                write_sweep(
                    output_dir,
                    "guard",
                    f"phase3zn_{set_name}_val_geo10_minmatch16",
                    guard_baseline_rows,
                )
                write_sweep(
                    output_dir,
                    "guard",
                    f"phase5d_{set_name}_val_geo10_minmatch16",
                    guard_rescue_rows,
                )
            write_sweep(output_dir, "formal", "phase3zn_val_geo10_minmatch16", baseline_rows)
            write_sweep(output_dir, "formal", "phase5d_val_geo10_minmatch16", rescue_rows)

            args = SimpleNamespace(
                output_dir=output_dir,
                baseline_label="phase3zn",
                candidate_label="phase5d",
                guard_baseline_label="phase3zn",
                guard_candidate_label="phase5d",
                dual_checkpoint_rescue_label="phase3zn_phase5d_selector",
                dual_checkpoint_rescue_target_variants="extreme_02,extreme_03",
                dual_checkpoint_rescue_min_match_gain=1,
                dual_checkpoint_rescue_min_rescue_matches=8,
                dual_checkpoint_rescue_max_homography_p90_px=3.2,
                dual_checkpoint_rescue_max_homography_median_px=1.8,
                dual_checkpoint_rescue_min_score_mean=16.0,
                dual_checkpoint_rescue_allow_score_mean_drop=False,
            )
            formal_summary, formal_variant_summary = fov76_gate_mod.combine_formal_summaries(output_dir)
            guard_summary, guard_variant_summary = fov76_gate_mod.combine_guard_summaries(
                output_dir,
                guard_labels=["phase3zn", "phase5d"],
            )

            selector_inputs = fov76_gate_mod.build_dual_checkpoint_selector_promotion_inputs(
                args,
                formal_summary=formal_summary,
                formal_variant_summary=formal_variant_summary,
                guard_summary=guard_summary,
                guard_variant_summary=guard_variant_summary,
            )

            with selector_inputs.formal_summary.open("r", encoding="utf-8", newline="") as handle:
                formal_rows = {(row["label"], row["split"]): row for row in csv.DictReader(handle)}
            selected_formal = formal_rows[("phase3zn_phase5d_selector", "val")]
            self.assertEqual(selected_formal["filtered_correct"], "488")
            self.assertEqual(selected_formal["filtered_wrong"], "9")
            self.assertNotIn(("phase5d", "val"), formal_rows)

            with selector_inputs.formal_variant_summary.open("r", encoding="utf-8", newline="") as handle:
                variant_rows = {(row["label"], row["variant"]): row for row in csv.DictReader(handle)}
            self.assertEqual(variant_rows[("phase3zn_phase5d_selector", "mid_01")]["correct"], "476")
            self.assertEqual(variant_rows[("phase3zn_phase5d_selector", "extreme_02")]["correct"], "12")

            with selector_inputs.guard_summary.open("r", encoding="utf-8", newline="") as handle:
                guard_rows = {(row["model"], row["set"]): row for row in csv.DictReader(handle)}
            selected_guard = guard_rows[("phase3zn_phase5d_selector", "regression_guard")]
            self.assertEqual(selected_guard["filtered_correct"], "587")
            self.assertEqual(selected_guard["filtered_wrong"], "12")

            decision = promotion_mod.evaluate_promotion(
                formal_summary=selector_inputs.formal_summary,
                formal_variant_summary=selector_inputs.formal_variant_summary,
                guard_summary=selector_inputs.guard_summary,
                baseline_label="phase3zn",
                candidate_label="phase3zn_phase5d_selector",
                guard_baseline_label="phase3zn",
                guard_candidate_label="phase3zn_phase5d_selector",
                splits=["val"],
                formal_target_variants=["extreme_02"],
                formal_protected_variants=["mid_01"],
                min_formal_target_correct_gain=1,
                min_formal_target_total_correct_gain=1,
                max_formal_target_precision_drop=0.0,
                max_formal_target_wrong_increase=0,
                max_protected_variant_precision_drop=0.0,
                max_protected_variant_correct_drop=0,
                max_protected_variant_wrong_increase=0,
                max_formal_precision_drop=0.0,
                max_formal_correct_drop=0,
                max_formal_wrong_increase=0,
                max_guard_precision_drop=0.0,
                max_guard_correct_drop=0,
                max_guard_wrong_increase=0,
                min_extreme_correct_gain=0,
            )
            self.assertTrue(decision.promote, decision.failed_reasons)

    def test_fov76_promotion_pipeline_plans_selector_as_promotion_candidate(self) -> None:
        args = SimpleNamespace(
            pair_root=Path("/data/pairs"),
            guard_root=Path("/data/pairs/hard_mining/guard"),
            output_dir=Path("/out/eval"),
            baseline_state=Path("/runs/base/state.pt"),
            baseline_run_dir=Path("/runs/base/train_output"),
            candidate_state=Path("/runs/cand/state.pt"),
            candidate_run_dir=Path("/runs/cand/train_output"),
            baseline_label="phase3zn",
            candidate_label="phase5d",
            guard_baseline_label="phase3zn",
            guard_candidate_label="phase5d",
            splits=["val"],
            device="cuda",
            python_executable="/env/bin/python",
            seed=20260614,
            crop_size=2048,
            max_image_size=768,
            max_keypoints=512,
            matcher_candidate_topk=256,
            graph_layers=4,
            geometry_threshold_px=5.0,
            filtered_min_matches=16,
            filtered_min_matches_by_variant=[],
            baseline_filtered_min_matches_by_variant=[],
            candidate_filtered_min_matches_by_variant=[],
            adaptive_geometry_rescue_variants="",
            baseline_adaptive_geometry_rescue_variants="",
            candidate_adaptive_geometry_rescue_variants="",
            adaptive_geometry_rescue_threshold_px=0.0,
            adaptive_geometry_rescue_min_match_gain=0,
            adaptive_geometry_rescue_max_base_matches=-1,
            adaptive_geometry_rescue_max_homography_p90_px=-1.0,
            adaptive_geometry_rescue_max_homography_median_px=-1.0,
            adaptive_geometry_rescue_require_score_mean_not_lower=False,
            baseline_adaptive_geometry_rescue_threshold_px=None,
            baseline_adaptive_geometry_rescue_min_match_gain=None,
            baseline_adaptive_geometry_rescue_max_base_matches=None,
            baseline_adaptive_geometry_rescue_max_homography_p90_px=None,
            baseline_adaptive_geometry_rescue_max_homography_median_px=None,
            baseline_adaptive_geometry_rescue_require_score_mean_not_lower=None,
            candidate_adaptive_geometry_rescue_threshold_px=None,
            candidate_adaptive_geometry_rescue_min_match_gain=None,
            candidate_adaptive_geometry_rescue_max_base_matches=None,
            candidate_adaptive_geometry_rescue_max_homography_p90_px=None,
            candidate_adaptive_geometry_rescue_max_homography_median_px=None,
            candidate_adaptive_geometry_rescue_require_score_mean_not_lower=None,
            low_match_geometry_guard_variants="",
            baseline_low_match_geometry_guard_variants="",
            candidate_low_match_geometry_guard_variants="",
            low_match_geometry_guard_min_matches=0,
            low_match_geometry_guard_max_matches=-1,
            low_match_geometry_guard_max_homography_p90_px=-1.0,
            low_match_geometry_guard_max_homography_median_px=-1.0,
            low_match_geometry_guard_min_score_mean=float("-inf"),
            baseline_low_match_geometry_guard_min_matches=None,
            baseline_low_match_geometry_guard_max_matches=None,
            baseline_low_match_geometry_guard_max_homography_p90_px=None,
            baseline_low_match_geometry_guard_max_homography_median_px=None,
            baseline_low_match_geometry_guard_min_score_mean=None,
            candidate_low_match_geometry_guard_min_matches=None,
            candidate_low_match_geometry_guard_max_matches=None,
            candidate_low_match_geometry_guard_max_homography_p90_px=None,
            candidate_low_match_geometry_guard_max_homography_median_px=None,
            candidate_low_match_geometry_guard_min_score_mean=None,
            write_match_details=False,
            max_formal_precision_drop=0.0,
            max_formal_correct_drop=0,
            max_formal_wrong_increase=0,
            formal_target_variants="extreme_02,extreme_03",
            formal_protected_variants="mid_01,mid_02,extreme_01,nadir",
            min_formal_target_correct_gain=1,
            min_formal_target_total_correct_gain=1,
            max_formal_target_precision_drop=0.0,
            max_formal_target_wrong_increase=0,
            max_protected_variant_precision_drop=0.0,
            max_protected_variant_correct_drop=0,
            max_protected_variant_wrong_increase=0,
            max_guard_precision_drop=0.0,
            max_guard_correct_drop=0,
            max_guard_wrong_increase=0,
            min_extreme_correct_gain=0,
            max_extreme_precision_drop=0.02,
            max_extreme_wrong_increase=999,
            dual_checkpoint_rescue_selector=True,
            dual_checkpoint_rescue_label="phase3zn_phase5d_selector",
        )

        promotion = fov76_gate_mod.planned_commands(args)[-1]

        self.assertEqual(promotion[promotion.index("--candidate-label") + 1], "phase3zn_phase5d_selector")
        self.assertEqual(promotion[promotion.index("--guard-candidate-label") + 1], "phase3zn_phase5d_selector")
        self.assertEqual(
            promotion[promotion.index("--formal-summary") + 1],
            "/out/eval/dual_checkpoint_rescue_selector/promotion_formal_summary.csv",
        )
        self.assertEqual(
            promotion[promotion.index("--guard-summary") + 1],
            "/out/eval/dual_checkpoint_rescue_selector/promotion_guard_summary.csv",
        )

    def test_fov76_promotion_pipeline_validates_required_inputs_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_root = root / "pairs"
            guard_root = pair_root / "hard_mining" / "guard"
            base_run = root / "base" / "train_output"
            cand_run = root / "cand" / "train_output"
            for path in [
                pair_root / "manifests",
                guard_root,
                base_run,
                cand_run,
            ]:
                path.mkdir(parents=True, exist_ok=True)
            for path in [
                pair_root / "manifests" / "h100km_fov076_render_manifest.csv",
                pair_root / "manifests" / "h100km_fov076_uint8_manifest.csv",
                pair_root / "overlap_edges_val.csv",
                pair_root / "overlap_edges_test.csv",
                guard_root / "regression_guard_val.csv",
                guard_root / "regression_guard_test.csv",
                guard_root / "extreme_gain_val.csv",
                guard_root / "extreme_gain_test.csv",
                base_run / "train_metrics.csv",
                cand_run / "train_metrics.csv",
                root / "base_state.pt",
                root / "candidate_state.pt",
            ]:
                path.write_text("x\n", encoding="utf-8")
            args = SimpleNamespace(
                pair_root=pair_root,
                guard_root=guard_root,
                output_dir=root / "out",
                baseline_state=root / "base_state.pt",
                baseline_run_dir=base_run,
                candidate_state=root / "candidate_state.pt",
                candidate_run_dir=cand_run,
                baseline_label="phase2h_ransac",
                candidate_label="phase3f_ransac",
                guard_baseline_label="phase2h",
                guard_candidate_label="phase3f",
                splits=["val", "test"],
                device="cuda",
                python_executable="/env/bin/python",
                seed=20260614,
                crop_size=2048,
                max_image_size=768,
                max_keypoints=512,
                matcher_candidate_topk=256,
                graph_layers=4,
                geometry_threshold_px=10.0,
                filtered_min_matches=16,
                filtered_min_matches_by_variant=[],
                baseline_filtered_min_matches_by_variant=[],
                candidate_filtered_min_matches_by_variant=[],
                max_formal_precision_drop=0.0,
                max_formal_correct_drop=0,
                max_formal_wrong_increase=0,
                max_guard_precision_drop=0.0,
                max_guard_correct_drop=0,
                max_guard_wrong_increase=0,
                min_extreme_correct_gain=1,
                max_extreme_precision_drop=0.02,
                max_extreme_wrong_increase=999,
            )

            fov76_gate_mod.validate_inputs(args)
            (guard_root / "extreme_gain_test.csv").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "extreme_gain_test.csv"):
                fov76_gate_mod.validate_inputs(args)

    def test_fov76_promotion_pipeline_records_failed_sweep_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sweep_dir = root / "promotion" / "guard" / "candidate_extreme_gain_test_geo10_minmatch16"
            command = [
                "/env/bin/python",
                "run_graph_filter_sweep.py",
                "--output-dir",
                str(sweep_dir),
            ]

            fov76_gate_mod.record_failed_sweep_command(
                command,
                returncode=-11,
                error="SIGSEGV",
            )

            summary = sweep_dir / "graph_filter_sweep_summary.csv"
            report = sweep_dir / "failed_sweep" / "all_filtered_summary.csv"
            failure = sweep_dir / "sweep_failure.json"

            self.assertTrue(summary.exists())
            self.assertTrue(report.exists())
            self.assertTrue(failure.exists())
            with summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["sweep_failed"], "1")
            self.assertIn("SIGSEGV", rows[0]["sweep_error"])
            guard_summary, _guard_variant_summary = fov76_gate_mod.combine_guard_summaries(
                root / "promotion",
                guard_labels=["baseline", "candidate"],
            )
            with guard_summary.open(newline="", encoding="utf-8") as handle:
                guard_rows = list(csv.DictReader(handle))
            self.assertEqual(guard_rows[0]["sweep_failed"], "1")

    def test_graph_depth_ablation_parses_unique_positive_depths(self) -> None:
        self.assertEqual(depth_ablation_mod.parse_depths("1,2,2,4"), [1, 2, 4])
        with self.assertRaises(ValueError):
            depth_ablation_mod.parse_depths("0,2")

    def test_graph_depth_ablation_builds_visual_command_with_depth_control(self) -> None:
        args = SimpleNamespace(
            render_manifest=Path("render.csv"),
            uint8_manifest=Path("uint8.csv"),
            pytorch_state=Path("model.pt"),
            run_dir=None,
            metrics_csv=None,
            split="train",
            reference_variant="nadir",
            pair_mode="spatial-index",
            image_source="uint8",
            candidate_pairs=12,
            select_count=4,
            seed=7,
            crop_size=2048,
            max_image_size=768,
            device="cuda",
            descriptor_mode="learned",
            keypoint_score_mode="learned",
            max_keypoints=512,
            matcher_candidate_topk=256,
            max_matches=0,
            draw_matches=0,
            threshold_px=5.0,
            graph_max_attention_work_fraction=0.5,
            graph_width_prune_keep_ratio=0.75,
            graph_width_prune_min_score=-1.0,
            graph_early_stop_min_confidence=-1.0,
            graph_dustbin_delta=0.1,
            graph_acceptance_margin=0.2,
            graph_min_raw_score=0.3,
            graph_min_raw_margin=0.04,
            graph_min_accept_probability=0.7,
            filtered_geometry_filter="local",
            filtered_min_margin=0.02,
            filtered_min_score=-1.0,
            filtered_max_matches=0,
            filtered_draw_matches=0,
            pair_spec_manifest=Path("pairs.csv"),
            target_variant=["mid_01"],
            cross_pair_variant=["mid_01"],
            cross_camera_offsets="1,3",
            cross_fov_offsets="0,2",
            pair_type_weights="same_position_view=0,cross_camera=1,cross_fov=0",
            spatial_index_height_km="100",
            spatial_index_planet_radius_m=3396190.0,
            spatial_index_footprint_samples=5,
            spatial_index_margin_m=2000.0,
            shuffle=True,
            filtered_report=True,
            filtered_mutual=True,
            illumination_stress=False,
            input_local_contrast=False,
            input_local_contrast_strength=0.0,
            input_local_contrast_kernel=31,
            adaptive_geometry_rescue_variants="extreme_02,extreme_03",
            adaptive_geometry_rescue_threshold_px=10.0,
            adaptive_geometry_rescue_min_match_gain=20,
            adaptive_geometry_rescue_max_base_matches=0,
            adaptive_geometry_rescue_max_homography_p90_px=4.5,
            adaptive_geometry_rescue_max_homography_median_px=1.8,
            adaptive_geometry_rescue_require_score_mean_not_lower=True,
        )

        command = depth_ablation_mod.build_visual_command(args, depth=2, report_dir=Path("out/layers_2"))

        self.assertIn("--graph-max-attention-layers", command)
        self.assertEqual(command[command.index("--graph-max-attention-layers") + 1], "2")
        self.assertIn("--graph-max-attention-work-fraction", command)
        self.assertEqual(command[command.index("--graph-max-attention-work-fraction") + 1], "0.5")
        self.assertIn("--graph-width-prune-keep-ratio", command)
        self.assertEqual(command[command.index("--graph-width-prune-keep-ratio") + 1], "0.75")
        self.assertIn("--graph-dustbin-delta", command)
        self.assertEqual(command[command.index("--graph-dustbin-delta") + 1], "0.1")
        self.assertIn("--graph-acceptance-margin", command)
        self.assertEqual(command[command.index("--graph-acceptance-margin") + 1], "0.2")
        self.assertIn("--graph-min-raw-score", command)
        self.assertEqual(command[command.index("--graph-min-raw-score") + 1], "0.3")
        self.assertIn("--graph-min-raw-margin", command)
        self.assertEqual(command[command.index("--graph-min-raw-margin") + 1], "0.04")
        self.assertIn("--graph-min-accept-probability", command)
        self.assertEqual(command[command.index("--graph-min-accept-probability") + 1], "0.7")

    def test_training_visual_report_parse_args_accepts_lightglue_graph_options(self) -> None:
        argv = [
            "training_visual_report.py",
            "--run-dir",
            "run",
            "--validation-cache-dir",
            "val",
            "--graph-width-prune-min-score",
            "0.25",
            "--graph-early-stop-min-confidence",
            "0.85",
            "--graph-max-attention-layers",
            "2",
            "--graph-max-attention-work-fraction",
            "0.5",
            "--graph-width-prune-keep-ratio",
            "0.75",
            "--graph-inference-preset",
            "fast",
            "--graph-min-accept-probability",
            "0.7",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = training_report_mod.parse_args()

        self.assertEqual(args.graph_width_prune_min_score, 0.25)
        self.assertEqual(args.graph_early_stop_min_confidence, 0.85)
        self.assertEqual(args.graph_max_attention_layers, 2)
        self.assertEqual(args.graph_max_attention_work_fraction, 0.5)
        self.assertEqual(args.graph_width_prune_keep_ratio, 0.75)
        self.assertEqual(args.graph_inference_preset, "fast")
        self.assertEqual(args.graph_min_accept_probability, 0.7)

    def test_lazy_visual_parse_args_accepts_graph_depth_controls(self) -> None:
        argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "report",
            "--matcher-mode",
            "graph_matcher",
            "--graph-max-attention-layers",
            "2",
            "--graph-max-attention-work-fraction",
            "0.5",
            "--graph-width-prune-keep-ratio",
            "0.75",
            "--graph-dustbin-delta",
            "0.1",
            "--graph-acceptance-margin",
            "0.2",
            "--graph-min-raw-score",
            "0.3",
            "--graph-min-raw-margin",
            "0.04",
            "--graph-min-accept-probability",
            "0.7",
            "--matcher-candidate-topk",
            "256",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = visual_mod.parse_args()

        self.assertEqual(args.graph_max_attention_layers, 2)
        self.assertEqual(args.graph_max_attention_work_fraction, 0.5)
        self.assertEqual(args.graph_width_prune_keep_ratio, 0.75)
        self.assertEqual(args.graph_dustbin_delta, 0.1)
        self.assertEqual(args.graph_acceptance_margin, 0.2)
        self.assertEqual(args.graph_min_raw_score, 0.3)
        self.assertEqual(args.graph_min_raw_margin, 0.04)
        self.assertEqual(args.graph_min_accept_probability, 0.7)
        self.assertEqual(args.matcher_candidate_topk, 256)

    def test_lazy_visual_parse_args_accepts_robust_geometry_filters(self) -> None:
        argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "report",
            "--geometry-filter",
            "magsac",
            "--filtered-geometry-filter",
            "ransac",
            "--geometry-threshold-px",
            "4.0",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = visual_mod.parse_args()

        self.assertEqual(args.geometry_filter, "magsac")
        self.assertEqual(args.filtered_geometry_filter, "ransac")
        self.assertEqual(args.geometry_threshold_px, 4.0)

    def test_lazy_visual_parse_args_applies_fov76_geo5_geo10_tight_rescue_profile(self) -> None:
        argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "report",
            "--post-filter-profile",
            "fov76_geo5_geo10_extreme_rescue",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = visual_mod.parse_args()

        self.assertEqual(args.geometry_filter, "local")
        self.assertEqual(args.geometry_threshold_px, 5.0)
        self.assertEqual(args.filtered_geometry_filter, "magsac")
        self.assertEqual(args.filtered_min_margin, 0.0)
        self.assertEqual(args.filtered_min_matches, 16)
        self.assertEqual(args.adaptive_geometry_rescue_variants, "extreme_02,extreme_03")
        self.assertEqual(args.adaptive_geometry_rescue_threshold_px, 10.0)
        self.assertEqual(args.adaptive_geometry_rescue_min_match_gain, 5)
        self.assertEqual(args.adaptive_geometry_rescue_max_base_matches, 16)
        self.assertEqual(args.adaptive_geometry_rescue_max_homography_p90_px, 4.2)
        self.assertEqual(args.adaptive_geometry_rescue_max_homography_median_px, 2.3)
        self.assertFalse(args.adaptive_geometry_rescue_require_score_mean_not_lower)

    def test_lazy_visual_parse_args_applies_fov76_low_match_guard_profile(self) -> None:
        argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "report",
            "--post-filter-profile",
            "fov76_geo5_geo10_extreme_rescue_lowmatch_guard",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = visual_mod.parse_args()

        self.assertEqual(args.geometry_filter, "local")
        self.assertEqual(args.geometry_threshold_px, 5.0)
        self.assertEqual(args.filtered_geometry_filter, "magsac")
        self.assertEqual(args.filtered_min_margin, 0.0)
        self.assertEqual(args.filtered_min_matches, 16)
        self.assertEqual(args.adaptive_geometry_rescue_variants, "extreme_02,extreme_03")
        self.assertEqual(args.adaptive_geometry_rescue_threshold_px, 10.0)
        self.assertEqual(args.adaptive_geometry_rescue_min_match_gain, 5)
        self.assertEqual(args.adaptive_geometry_rescue_max_base_matches, 16)
        self.assertEqual(args.adaptive_geometry_rescue_max_homography_p90_px, 4.2)
        self.assertEqual(args.adaptive_geometry_rescue_max_homography_median_px, 2.3)
        self.assertEqual(args.low_match_geometry_guard_variants, "extreme_02,extreme_03")
        self.assertEqual(args.low_match_geometry_guard_min_matches, 12)
        self.assertEqual(args.low_match_geometry_guard_max_matches, 15)
        self.assertEqual(args.low_match_geometry_guard_max_homography_p90_px, 2.8)
        self.assertEqual(args.low_match_geometry_guard_max_homography_median_px, 1.5)
        self.assertEqual(args.low_match_geometry_guard_min_score_mean, 19.0)

    def test_lazy_visual_parse_args_accepts_variant_min_match_gate(self) -> None:
        argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "report",
            "--filtered-min-matches",
            "16",
            "--filtered-min-matches-by-variant",
            "extreme_02=8,extreme_03=6",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = visual_mod.parse_args()

        self.assertEqual(args.filtered_min_matches, 16)
        self.assertEqual(args.filtered_min_matches_by_variant, {"extreme_02": 8, "extreme_03": 6})

    def test_lazy_visual_parse_args_accepts_all_summary_output(self) -> None:
        argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "report",
            "--write-all-summary",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = visual_mod.parse_args()

        self.assertTrue(args.write_all_summary)

    def test_lazy_visual_summary_csv_includes_score_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            record = RenderRecord(
                pose_id="pose_nadir",
                base_id="base_a",
                variant="nadir",
                split="val",
                tsai_path=root / "a.tsai",
                image_path=root / "a.tif",
                uint8_path=root / "a.png",
                depth_path=root / "a_depth.tif",
            )
            target = RenderRecord(
                pose_id="pose_extreme",
                base_id="base_a",
                variant="extreme_03",
                split="val",
                tsai_path=root / "b.tsai",
                image_path=root / "b.tif",
                uint8_path=root / "b.png",
                depth_path=root / "b_depth.tif",
            )
            pair = SyntheticPair(
                view_a=torch.zeros(1, 4, 4),
                view_b=torch.zeros(1, 4, 4),
                warp_a_to_b=torch.zeros(4, 4, 2),
                valid_mask=torch.ones(4, 4, dtype=torch.bool),
            )
            visual = LazyMatchVisual(
                label="candidate",
                spec=LazyPairSpec(pair_index=0, split="val", reference=record, target=target),
                pair=pair,
                valid_fraction=0.5,
                points_a=np.zeros((3, 2), dtype=np.float32),
                points_b=np.zeros((3, 2), dtype=np.float32),
                scores=np.asarray([0.2, 0.6, 0.8], dtype=np.float32),
                errors=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                correct=np.asarray([True, True, False], dtype=bool),
            )
            path = root / "summary.csv"

            visual_mod.write_summary_csv([visual], path)

            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["score_min"], "0.200000")
            self.assertEqual(row["score_mean"], "0.533333")
            self.assertEqual(row["score_median"], "0.600000")
            self.assertEqual(row["score_max"], "0.800000")

    def test_lazy_visual_summary_csv_includes_inference_geometry_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            record = RenderRecord(
                pose_id="pose_nadir",
                base_id="base_geom",
                variant="nadir",
                split="test",
                tsai_path=root / "a.tsai",
                image_path=root / "a.tif",
                uint8_path=root / "a.png",
                depth_path=root / "a_depth.tif",
            )
            target = RenderRecord(
                pose_id="pose_extreme",
                base_id="base_geom",
                variant="extreme_02",
                split="test",
                tsai_path=root / "b.tsai",
                image_path=root / "b.tif",
                uint8_path=root / "b.png",
                depth_path=root / "b_depth.tif",
            )
            pair = SyntheticPair(
                view_a=torch.zeros(1, 16, 16),
                view_b=torch.zeros(1, 16, 16),
                warp_a_to_b=torch.zeros(16, 16, 2),
                valid_mask=torch.ones(16, 16, dtype=torch.bool),
            )
            points_a = np.asarray(
                [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 5.0]],
                dtype=np.float32,
            )
            points_b = points_a + np.asarray([3.0, 4.0], dtype=np.float32)
            visual = LazyMatchVisual(
                label="candidate",
                spec=LazyPairSpec(pair_index=0, split="test", reference=record, target=target),
                pair=pair,
                valid_fraction=1.0,
                points_a=points_a,
                points_b=points_b,
                scores=np.ones(5, dtype=np.float32),
                errors=np.zeros(5, dtype=np.float32),
                correct=np.ones(5, dtype=bool),
            )
            path = root / "summary.csv"

            visual_mod.write_summary_csv([visual], path)

            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["span_a_x_px"], "10.000")
            self.assertEqual(row["span_a_y_px"], "10.000")
            self.assertEqual(row["span_b_x_px"], "10.000")
            self.assertEqual(row["span_b_y_px"], "10.000")
            self.assertEqual(row["bbox_area_a_px2"], "100.000")
            self.assertEqual(row["bbox_area_b_px2"], "100.000")
            self.assertEqual(row["displacement_median_px"], "5.000")
            self.assertEqual(row["displacement_mad_px"], "0.000")
            self.assertEqual(row["homography_residual_valid"], "1")
            self.assertAlmostEqual(float(row["homography_residual_median_px"]), 0.0, places=3)
            self.assertAlmostEqual(float(row["homography_residual_p90_px"]), 0.0, places=3)

    def test_lazy_visual_match_detail_csv_exports_each_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            record = RenderRecord(
                pose_id="pose_nadir",
                base_id="base_match_detail",
                variant="nadir",
                split="test",
                tsai_path=root / "a.tsai",
                image_path=root / "a.tif",
                uint8_path=root / "a.png",
                depth_path=root / "a_depth.tif",
            )
            target = RenderRecord(
                pose_id="pose_extreme",
                base_id="base_match_detail",
                variant="extreme_03",
                split="test",
                tsai_path=root / "b.tsai",
                image_path=root / "b.tif",
                uint8_path=root / "b.png",
                depth_path=root / "b_depth.tif",
            )
            pair = SyntheticPair(
                view_a=torch.zeros(1, 16, 16),
                view_b=torch.zeros(1, 16, 16),
                warp_a_to_b=torch.zeros(16, 16, 2),
                valid_mask=torch.ones(16, 16, dtype=torch.bool),
            )
            visual = LazyMatchVisual(
                label="candidate",
                spec=LazyPairSpec(pair_index=7, split="test", reference=record, target=target),
                pair=pair,
                valid_fraction=0.75,
                points_a=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                points_b=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
                scores=np.asarray([9.5, 2.25], dtype=np.float32),
                errors=np.asarray([0.75, 12.5], dtype=np.float32),
                correct=np.asarray([True, False], dtype=bool),
                pair_logits=np.asarray([10.0, 3.0], dtype=np.float32),
                row_dustbin_logits=np.asarray([1.0, 0.5], dtype=np.float32),
                col_dustbin_logits=np.asarray([2.0, 0.25], dtype=np.float32),
                positive_vs_dustbin_margins=np.asarray([7.0, 2.25], dtype=np.float32),
                raw_similarities=np.asarray([0.8, 0.6], dtype=np.float32),
                raw_margins=np.asarray([0.2, 0.05], dtype=np.float32),
                accept_logits=np.asarray([1.5, -0.5], dtype=np.float32),
                accept_probabilities=np.asarray([0.817574, 0.377541], dtype=np.float32),
            )
            path = root / "matches.csv"

            visual_mod.write_match_detail_csv([visual], path)

            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["pair_index"], "7")
            self.assertEqual(rows[0]["base_id"], "base_match_detail")
            self.assertEqual(rows[0]["target_variant"], "extreme_03")
            self.assertEqual(rows[0]["match_index"], "0")
            self.assertEqual(rows[0]["score"], "9.500000")
            self.assertEqual(rows[0]["error_px"], "0.750000")
            self.assertEqual(rows[0]["correct"], "1")
            self.assertEqual(rows[0]["point_a_x_px"], "1.000")
            self.assertEqual(rows[0]["pair_logit"], "10.000000")
            self.assertEqual(rows[0]["row_dustbin_logit"], "1.000000")
            self.assertEqual(rows[0]["col_dustbin_logit"], "2.000000")
            self.assertEqual(rows[0]["positive_vs_dustbin_margin"], "7.000000")
            self.assertEqual(rows[0]["raw_similarity"], "0.800000")
            self.assertEqual(rows[0]["raw_margin"], "0.200000")
            self.assertEqual(rows[0]["accept_logit"], "1.500000")
            self.assertEqual(rows[0]["accept_probability"], "0.817574")
            self.assertEqual(rows[1]["match_index"], "1")
            self.assertEqual(rows[1]["correct"], "0")
            self.assertEqual(rows[1]["point_b_y_px"], "8.000")
            self.assertEqual(rows[1]["positive_vs_dustbin_margin"], "2.250000")

    def test_compare_match_detail_reports_summarizes_pair_deltas_and_thresholds(self) -> None:
        import compare_match_detail_reports as compare_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.csv"
            candidate = root / "candidate.csv"
            fields = [
                "pair_index",
                "base_id",
                "target_variant",
                "split",
                "match_index",
                "score",
                "error_px",
                "correct",
            ]
            self.write_csv(
                baseline,
                fields,
                [
                    {
                        "pair_index": 0,
                        "base_id": "base_a",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "match_index": 0,
                        "score": 10.0,
                        "error_px": 2.0,
                        "correct": 1,
                    },
                    {
                        "pair_index": 0,
                        "base_id": "base_a",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "match_index": 1,
                        "score": 18.0,
                        "error_px": 8.0,
                        "correct": 0,
                    },
                    {
                        "pair_index": 1,
                        "base_id": "base_b",
                        "target_variant": "mid_02",
                        "split": "test",
                        "match_index": 0,
                        "score": 12.0,
                        "error_px": 1.5,
                        "correct": 1,
                    },
                ],
            )
            self.write_csv(
                candidate,
                fields,
                [
                    {
                        "pair_index": 0,
                        "base_id": "base_a",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "match_index": 0,
                        "score": 11.0,
                        "error_px": 2.0,
                        "correct": 1,
                    },
                    {
                        "pair_index": 0,
                        "base_id": "base_a",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "match_index": 1,
                        "score": 19.0,
                        "error_px": 3.0,
                        "correct": 1,
                    },
                    {
                        "pair_index": 0,
                        "base_id": "base_a",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "match_index": 2,
                        "score": 20.0,
                        "error_px": 12.0,
                        "correct": 0,
                    },
                ],
            )

            comparison = compare_mod.compare_match_detail_reports(
                baseline,
                candidate,
                baseline_label="baseline",
                candidate_label="candidate",
                score_thresholds=[0.0, 15.0],
            )

            self.assertEqual(comparison.overall[0].correct, 2)
            self.assertEqual(comparison.overall[1].correct, 2)
            self.assertEqual(comparison.per_pair[0].correct_delta, 1)
            self.assertEqual(comparison.per_pair[0].wrong_delta, 0)
            self.assertEqual(comparison.per_pair[1].correct_delta, -1)
            threshold_15 = [row for row in comparison.threshold_sweep if row.threshold == 15.0][0]
            self.assertEqual(threshold_15.matches, 2)
            self.assertEqual(threshold_15.correct, 1)
            self.assertEqual(threshold_15.wrong, 1)

    def test_combine_match_reports_by_variant_uses_rescue_only_for_selected_variants(self) -> None:
        import combine_match_reports_by_variant as combine_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "geo5.csv"
            rescue = root / "geo8.csv"
            fields = [
                "label",
                "pair_index",
                "base_id",
                "reference_variant",
                "target_variant",
                "split",
                "match_index",
                "score",
                "error_px",
                "correct",
            ]
            self.write_csv(
                baseline,
                fields,
                [
                    {
                        "label": "geo5",
                        "pair_index": 0,
                        "base_id": "base_a",
                        "reference_variant": "nadir",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "match_index": 0,
                        "score": 12.0,
                        "error_px": 3.0,
                        "correct": 1,
                    },
                    {
                        "label": "geo5",
                        "pair_index": 1,
                        "base_id": "base_b",
                        "reference_variant": "nadir",
                        "target_variant": "mid_02",
                        "split": "test",
                        "match_index": 0,
                        "score": 11.0,
                        "error_px": 2.0,
                        "correct": 1,
                    },
                ],
            )
            self.write_csv(
                rescue,
                fields,
                [
                    {
                        "label": "geo8",
                        "pair_index": 0,
                        "base_id": "base_a",
                        "reference_variant": "nadir",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "match_index": 0,
                        "score": 21.0,
                        "error_px": 2.0,
                        "correct": 1,
                    },
                    {
                        "label": "geo8",
                        "pair_index": 0,
                        "base_id": "base_a",
                        "reference_variant": "nadir",
                        "target_variant": "extreme_02",
                        "split": "test",
                        "match_index": 1,
                        "score": 14.0,
                        "error_px": 12.0,
                        "correct": 0,
                    },
                    {
                        "label": "geo8",
                        "pair_index": 1,
                        "base_id": "base_b",
                        "reference_variant": "nadir",
                        "target_variant": "mid_02",
                        "split": "test",
                        "match_index": 0,
                        "score": 25.0,
                        "error_px": 20.0,
                        "correct": 0,
                    },
                ],
            )

            merged = combine_mod.combine_match_detail_rows(
                combine_mod.read_rows(baseline),
                combine_mod.read_rows(rescue),
                rescue_variants={"extreme_02"},
                rescue_min_score=20.0,
                fallback_if_empty=True,
            )
            summary = combine_mod.summarize_rows(merged, label="variant_rescue")

        self.assertEqual(summary.matches, 2)
        self.assertEqual(summary.correct, 2)
        self.assertEqual(summary.wrong, 0)
        self.assertEqual([row["label"] for row in merged], ["variant_rescue:geo8", "variant_rescue:geo5"])
        self.assertEqual([row["target_variant"] for row in merged], ["extreme_02", "mid_02"])

    def test_combine_match_reports_by_variant_can_gate_rescue_by_pair_strength(self) -> None:
        import combine_match_reports_by_variant as combine_mod

        baseline_rows = [
            {
                "label": "geo5",
                "pair_index": 0,
                "base_id": "already_good",
                "reference_variant": "mid_02",
                "target_variant": "extreme_02",
                "split": "test",
                "match_index": str(index),
                "score": "22.0",
                "correct": "1",
            }
            for index in range(24)
        ]
        rescue_rows = [
            {
                "label": "geo8",
                "pair_index": 0,
                "base_id": "already_good",
                "reference_variant": "mid_02",
                "target_variant": "extreme_02",
                "split": "test",
                "match_index": str(index),
                "score": "23.0",
                "correct": "1",
            }
            for index in range(30)
        ]
        rescue_rows.extend(
            [
                {
                    "label": "geo8",
                    "pair_index": 1,
                    "base_id": "needs_rescue",
                    "reference_variant": "nadir",
                    "target_variant": "extreme_03",
                    "split": "test",
                    "match_index": str(index),
                    "score": "21.0",
                    "correct": "1",
                }
                for index in range(16)
            ]
        )
        rescue_rows.extend(
            [
                {
                    "label": "geo8",
                    "pair_index": 2,
                    "base_id": "low_score_rescue",
                    "reference_variant": "nadir",
                    "target_variant": "extreme_03",
                    "split": "test",
                    "match_index": str(index),
                    "score": "12.0",
                    "correct": "0",
                }
                for index in range(16)
            ]
        )

        merged = combine_mod.combine_match_detail_rows(
            baseline_rows,
            rescue_rows,
            rescue_variants={"extreme_02", "extreme_03"},
            rescue_min_score=-1.0,
            fallback_if_empty=True,
            rescue_max_baseline_matches=8,
            rescue_min_pair_matches=12,
            rescue_min_score_mean=18.0,
        )

        by_base = {row["base_id"] for row in merged}
        self.assertIn("already_good", by_base)
        self.assertIn("needs_rescue", by_base)
        self.assertNotIn("low_score_rescue", by_base)
        self.assertEqual(len([row for row in merged if row["base_id"] == "already_good"]), 24)
        self.assertEqual(len([row for row in merged if row["base_id"] == "needs_rescue"]), 16)

    def test_lazy_visual_compute_visual_calls_forward_geometry_threshold(self) -> None:
        source = (ROOT / "scripts" / "visualize_lazy_pose_matches.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compute_visual"
        ]

        self.assertGreaterEqual(len(calls), 3)
        for call in calls:
            self.assertIn("geometry_threshold_px", {keyword.arg for keyword in call.keywords})

    def test_graph_filter_sweep_builds_visual_command(self) -> None:
        config = filter_sweep_mod.GraphFilterConfig(
            min_score=0.05,
            dustbin_delta=0.1,
            acceptance_margin=0.2,
            min_raw_score=0.3,
            min_raw_margin=0.04,
            min_accept_probability=0.7,
            geometry_threshold_px=6.0,
            filtered_min_matches=12,
        )
        args = SimpleNamespace(
            render_manifest=Path("render.csv"),
            uint8_manifest=Path("uint8.csv"),
            pytorch_state=Path("model.pt"),
            run_dir=None,
            metrics_csv=None,
            split="train",
            reference_variant="nadir",
            pair_mode="spatial-index",
            image_source="uint8",
            candidate_pairs=12,
            select_count=4,
            seed=7,
            crop_size=2048,
            max_image_size=768,
            device="cuda",
            descriptor_mode="learned",
            keypoint_score_mode="learned",
            max_keypoints=512,
            matcher_candidate_topk=256,
            max_matches=0,
            draw_matches=0,
            threshold_px=5.0,
            geometry_filter="local",
            geometry_threshold_px=10.0,
            graph_max_attention_layers=2,
            graph_max_attention_work_fraction=1.0,
            graph_width_prune_keep_ratio=1.0,
            graph_width_prune_min_score=-1.0,
            graph_early_stop_min_confidence=-1.0,
            filtered_geometry_filter="magsac",
            filtered_min_margin=0.02,
            filtered_min_score=-1.0,
            filtered_min_matches=16,
            filtered_min_matches_by_variant=["extreme_02=8,extreme_03=6"],
            low_match_geometry_guard_variants="extreme_02,extreme_03",
            low_match_geometry_guard_min_matches=12,
            low_match_geometry_guard_max_matches=15,
            low_match_geometry_guard_max_homography_p90_px=2.8,
            low_match_geometry_guard_max_homography_median_px=1.5,
            low_match_geometry_guard_min_score_mean=19.0,
            filtered_max_matches=0,
            filtered_draw_matches=0,
            write_all_summary=True,
            write_match_details=True,
            pair_spec_manifest=Path("pairs.csv"),
            target_variant=["mid_01"],
            cross_pair_variant=["mid_01"],
            cross_camera_offsets="1,3",
            cross_fov_offsets="0,2",
            pair_type_weights="same_position_view=0,cross_camera=1,cross_fov=0",
            spatial_index_height_km="100",
            spatial_index_planet_radius_m=3396190.0,
            spatial_index_footprint_samples=5,
            spatial_index_margin_m=2000.0,
            shuffle=True,
            filtered_report=True,
            filtered_mutual=True,
            illumination_stress=False,
            input_local_contrast=False,
            input_local_contrast_strength=0.0,
            input_local_contrast_kernel=31,
            adaptive_geometry_rescue_variants="extreme_02,extreme_03",
            adaptive_geometry_rescue_threshold_px=10.0,
            adaptive_geometry_rescue_min_match_gain=20,
            adaptive_geometry_rescue_max_base_matches=0,
            adaptive_geometry_rescue_max_homography_p90_px=4.5,
            adaptive_geometry_rescue_max_homography_median_px=1.8,
            adaptive_geometry_rescue_require_score_mean_not_lower=True,
        )

        command = filter_sweep_mod.build_visual_command(args, config=config, report_dir=Path("out/cfg"))

        self.assertIn("--min-score", command)
        self.assertEqual(command[command.index("--min-score") + 1], "0.05")
        self.assertIn("--graph-dustbin-delta", command)
        self.assertEqual(command[command.index("--graph-dustbin-delta") + 1], "0.1")
        self.assertIn("--graph-acceptance-margin", command)
        self.assertEqual(command[command.index("--graph-acceptance-margin") + 1], "0.2")
        self.assertIn("--graph-min-raw-score", command)
        self.assertEqual(command[command.index("--graph-min-raw-score") + 1], "0.3")
        self.assertIn("--graph-min-raw-margin", command)
        self.assertEqual(command[command.index("--graph-min-raw-margin") + 1], "0.04")
        self.assertIn("--graph-min-accept-probability", command)
        self.assertEqual(command[command.index("--graph-min-accept-probability") + 1], "0.7")
        self.assertIn("--geometry-threshold-px", command)
        self.assertEqual(command[command.index("--geometry-threshold-px") + 1], "6.0")
        self.assertIn("--matcher-candidate-topk", command)
        self.assertIn("--low-match-geometry-guard-variants", command)
        self.assertEqual(command[command.index("--low-match-geometry-guard-variants") + 1], "extreme_02,extreme_03")
        self.assertEqual(command[command.index("--low-match-geometry-guard-min-matches") + 1], "12")
        self.assertEqual(command[command.index("--low-match-geometry-guard-max-matches") + 1], "15")
        self.assertEqual(command[command.index("--low-match-geometry-guard-max-homography-p90-px") + 1], "2.8")
        self.assertEqual(command[command.index("--low-match-geometry-guard-max-homography-median-px") + 1], "1.5")
        self.assertEqual(command[command.index("--low-match-geometry-guard-min-score-mean") + 1], "19.0")
        self.assertEqual(command[command.index("--matcher-candidate-topk") + 1], "256")
        self.assertIn("--geometry-filter", command)
        self.assertEqual(command[command.index("--geometry-filter") + 1], "local")
        self.assertIn("--filtered-geometry-filter", command)
        self.assertEqual(command[command.index("--filtered-geometry-filter") + 1], "magsac")
        self.assertIn("--filtered-min-matches", command)
        self.assertEqual(command[command.index("--filtered-min-matches") + 1], "12")
        self.assertIn("--filtered-min-matches-by-variant", command)
        self.assertEqual(command[command.index("--filtered-min-matches-by-variant") + 1], "extreme_02=8,extreme_03=6")
        self.assertIn("--adaptive-geometry-rescue-variants", command)
        self.assertEqual(command[command.index("--adaptive-geometry-rescue-variants") + 1], "extreme_02,extreme_03")
        self.assertIn("--adaptive-geometry-rescue-threshold-px", command)
        self.assertEqual(command[command.index("--adaptive-geometry-rescue-threshold-px") + 1], "10.0")
        self.assertIn("--adaptive-geometry-rescue-min-match-gain", command)
        self.assertEqual(command[command.index("--adaptive-geometry-rescue-min-match-gain") + 1], "20")
        self.assertIn("--adaptive-geometry-rescue-max-base-matches", command)
        self.assertEqual(command[command.index("--adaptive-geometry-rescue-max-base-matches") + 1], "0")
        self.assertIn("--adaptive-geometry-rescue-require-score-mean-not-lower", command)
        self.assertIn("--write-all-summary", command)
        self.assertIn("--write-match-details", command)

    def test_graph_filter_sweep_parse_args_accepts_magsac_and_all_summary(self) -> None:
        argv = [
            "run_graph_filter_sweep.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "out",
            "--filtered-geometry-filter",
            "magsac",
            "--geometry-filter",
            "local",
            "--geometry-threshold-px",
            "10.0",
            "--geometry-threshold-px-values",
            "6,8",
            "--filtered-min-matches",
            "16",
            "--filtered-min-matches-values",
            "8,16",
            "--filtered-min-matches-by-variant",
            "extreme_02=8,extreme_03=6",
            "--adaptive-geometry-rescue-variants",
            "extreme_02,extreme_03",
            "--adaptive-geometry-rescue-threshold-px",
            "10",
            "--adaptive-geometry-rescue-min-match-gain",
            "20",
            "--adaptive-geometry-rescue-max-base-matches",
            "0",
            "--adaptive-geometry-rescue-max-homography-p90-px",
            "4.5",
            "--adaptive-geometry-rescue-max-homography-median-px",
            "1.8",
            "--adaptive-geometry-rescue-require-score-mean-not-lower",
            "--matcher-candidate-topk",
            "256",
            "--write-all-summary",
            "--write-match-details",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = filter_sweep_mod.parse_args()

        self.assertEqual(args.filtered_geometry_filter, "magsac")
        self.assertEqual(args.geometry_filter, "local")
        self.assertEqual(args.geometry_threshold_px, 10.0)
        self.assertEqual(args.geometry_threshold_px_values, [6.0, 8.0])
        self.assertEqual(args.filtered_min_matches, 16)
        self.assertEqual(args.filtered_min_matches_values, [8, 16])
        self.assertEqual(args.filtered_min_matches_by_variant, ["extreme_02=8,extreme_03=6"])
        self.assertEqual(args.adaptive_geometry_rescue_variants, "extreme_02,extreme_03")
        self.assertEqual(args.adaptive_geometry_rescue_threshold_px, 10.0)
        self.assertEqual(args.adaptive_geometry_rescue_min_match_gain, 20)
        self.assertEqual(args.adaptive_geometry_rescue_max_base_matches, 0)
        self.assertEqual(args.adaptive_geometry_rescue_max_homography_p90_px, 4.5)
        self.assertEqual(args.adaptive_geometry_rescue_max_homography_median_px, 1.8)
        self.assertTrue(args.adaptive_geometry_rescue_require_score_mean_not_lower)
        self.assertEqual(args.matcher_candidate_topk, 256)
        self.assertTrue(args.write_all_summary)
        self.assertTrue(args.write_match_details)

    def test_graph_filter_sweep_parse_args_applies_fov76_geo5_geo10_tight_rescue_profile(self) -> None:
        argv = [
            "run_graph_filter_sweep.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "out",
            "--post-filter-profile",
            "fov76_geo5_geo10_extreme_rescue",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = filter_sweep_mod.parse_args()

        self.assertEqual(args.geometry_filter, "local")
        self.assertEqual(args.geometry_threshold_px, 5.0)
        self.assertEqual(args.geometry_threshold_px_values, [5.0])
        self.assertEqual(args.filtered_geometry_filter, "magsac")
        self.assertEqual(args.filtered_min_margin, 0.0)
        self.assertEqual(args.filtered_min_matches, 16)
        self.assertEqual(args.filtered_min_matches_values, [16])
        self.assertEqual(args.adaptive_geometry_rescue_variants, "extreme_02,extreme_03")
        self.assertEqual(args.adaptive_geometry_rescue_threshold_px, 10.0)
        self.assertEqual(args.adaptive_geometry_rescue_min_match_gain, 5)
        self.assertEqual(args.adaptive_geometry_rescue_max_base_matches, 16)
        self.assertEqual(args.adaptive_geometry_rescue_max_homography_p90_px, 4.2)
        self.assertEqual(args.adaptive_geometry_rescue_max_homography_median_px, 2.3)
        self.assertFalse(args.adaptive_geometry_rescue_require_score_mean_not_lower)

    def test_graph_filter_sweep_parse_args_applies_fov76_low_match_guard_profile(self) -> None:
        argv = [
            "run_graph_filter_sweep.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "out",
            "--post-filter-profile",
            "fov76_geo5_geo10_extreme_rescue_lowmatch_guard",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = filter_sweep_mod.parse_args()

        self.assertEqual(args.geometry_filter, "local")
        self.assertEqual(args.geometry_threshold_px, 5.0)
        self.assertEqual(args.geometry_threshold_px_values, [5.0])
        self.assertEqual(args.filtered_geometry_filter, "magsac")
        self.assertEqual(args.filtered_min_margin, 0.0)
        self.assertEqual(args.filtered_min_matches, 16)
        self.assertEqual(args.filtered_min_matches_values, [16])
        self.assertEqual(args.adaptive_geometry_rescue_variants, "extreme_02,extreme_03")
        self.assertEqual(args.adaptive_geometry_rescue_threshold_px, 10.0)
        self.assertEqual(args.adaptive_geometry_rescue_min_match_gain, 5)
        self.assertEqual(args.adaptive_geometry_rescue_max_base_matches, 16)
        self.assertEqual(args.adaptive_geometry_rescue_max_homography_p90_px, 4.2)
        self.assertEqual(args.adaptive_geometry_rescue_max_homography_median_px, 2.3)
        self.assertEqual(args.low_match_geometry_guard_variants, "extreme_02,extreme_03")
        self.assertEqual(args.low_match_geometry_guard_min_matches, 12)
        self.assertEqual(args.low_match_geometry_guard_max_matches, 15)
        self.assertEqual(args.low_match_geometry_guard_max_homography_p90_px, 2.8)
        self.assertEqual(args.low_match_geometry_guard_max_homography_median_px, 1.5)
        self.assertEqual(args.low_match_geometry_guard_min_score_mean, 19.0)

    def test_graph_filter_sweep_configs_include_geometry_and_min_match_lists(self) -> None:
        args = SimpleNamespace(
            min_score_values=[-1.0],
            graph_dustbin_delta_values=[0.0],
            graph_acceptance_margin_values=[0.0],
            graph_min_raw_score_values=[-1.0],
            graph_min_raw_margin_values=[0.0],
            graph_min_accept_probability_values=[-1.0],
            geometry_threshold_px=10.0,
            geometry_threshold_px_values=[6.0, 8.0],
            filtered_min_matches=16,
            filtered_min_matches_values=[8, 12],
        )

        configs = filter_sweep_mod.iter_sweep_configs(args)

        self.assertEqual(len(configs), 4)
        self.assertEqual(
            {(cfg.geometry_threshold_px, cfg.filtered_min_matches) for cfg in configs},
            {(6.0, 8), (6.0, 12), (8.0, 8), (8.0, 12)},
        )

    def test_graph_filter_sweep_parses_float_lists_and_slugs_config(self) -> None:
        self.assertEqual(filter_sweep_mod.parse_float_list("-1,0.05,0.5"), [-1.0, 0.05, 0.5])
        config = filter_sweep_mod.GraphFilterConfig(
            min_score=0.05,
            dustbin_delta=-0.1,
            acceptance_margin=0.2,
            min_raw_score=-1.0,
            min_raw_margin=0.04,
            min_accept_probability=0.7,
            geometry_threshold_px=6.0,
            filtered_min_matches=12,
        )

        self.assertEqual(
            filter_sweep_mod.slug_for_config(config),
            "score0p05_dustneg0p1_accept0p2_rawneg1_margin0p04_prob0p7_geo6_minmatch12",
        )

    def test_graph_filter_sweep_summarizes_raw_and_filtered_reports(self) -> None:
        config = filter_sweep_mod.GraphFilterConfig(
            min_score=0.05,
            dustbin_delta=0.1,
            acceptance_margin=0.2,
            min_raw_score=0.3,
            min_raw_margin=0.04,
            min_accept_probability=0.7,
            geometry_threshold_px=6.0,
            filtered_min_matches=12,
        )
        with tempfile.TemporaryDirectory() as temp:
            report_dir = Path(temp)
            (report_dir / "all_summary.csv").write_text(
                "label,matches,correct,wrong,precision,median_error_px\n"
                "a,10,7,3,0.700000,2.5\n"
                "b,5,2,3,0.400000,9.0\n",
                encoding="utf-8",
            )
            (report_dir / "summary.csv").write_text(
                "label,matches,correct,wrong,precision,median_error_px\n"
                "selected,1,0,1,0.000000,10.0\n",
                encoding="utf-8",
            )
            (report_dir / "all_filtered_summary.csv").write_text(
                "label,matches,correct,wrong,precision,median_error_px\n"
                "a,4,4,0,1.000000,1.5\n",
                encoding="utf-8",
            )
            (report_dir / "filtered_summary.csv").write_text(
                "label,matches,correct,wrong,precision,median_error_px\n"
                "selected-filtered,1,0,1,0.000000,10.0\n",
                encoding="utf-8",
            )

            summary = filter_sweep_mod.summarize_report(report_dir, config=config)

        self.assertEqual(summary.raw_rows, 2)
        self.assertEqual(summary.raw_matches, 15)
        self.assertEqual(summary.raw_correct, 9)
        self.assertAlmostEqual(summary.raw_precision, 0.6)
        self.assertEqual(summary.filtered_rows, 1)
        self.assertEqual(summary.filtered_matches, 4)
        self.assertEqual(summary.filtered_correct, 4)
        self.assertAlmostEqual(summary.filtered_precision, 1.0)

    def test_rescue_candidate_analysis_aligns_reports_and_sweeps_score_rules(self) -> None:
        baseline_rows = [
            {
                "base_id": "base_good",
                "target_variant": "extreme_03",
                "split": "val",
                "matches": "0",
                "correct": "0",
                "wrong": "0",
                "precision": "0.0",
                "score_min": "0",
                "score_mean": "0",
                "score_median": "0",
                "score_max": "0",
                "median_error_px": "0",
                "valid_fraction": "0.5",
            },
            {
                "base_id": "base_bad",
                "target_variant": "extreme_02",
                "split": "val",
                "matches": "0",
                "correct": "0",
                "wrong": "0",
                "precision": "0.0",
                "score_min": "0",
                "score_mean": "0",
                "score_median": "0",
                "score_max": "0",
                "median_error_px": "0",
                "valid_fraction": "0.4",
            },
            {
                "base_id": "base_mid",
                "target_variant": "mid_02",
                "split": "val",
                "matches": "0",
                "correct": "0",
                "wrong": "0",
                "precision": "0.0",
                "score_min": "0",
                "score_mean": "0",
                "score_median": "0",
                "score_max": "0",
                "median_error_px": "0",
                "valid_fraction": "0.8",
            },
        ]
        candidate_rows = [
            baseline_rows[0]
            | {
                "matches": "8",
                "correct": "8",
                "wrong": "0",
                "precision": "1.0",
                "score_min": "11",
                "score_mean": "16",
                "score_median": "17",
                "score_max": "18",
                "median_error_px": "2.5",
                "bbox_area_a_px2": "100",
                "bbox_area_b_px2": "100",
                "displacement_mad_px": "1",
                "homography_residual_valid": "1",
                "homography_residual_p90_px": "2",
            },
            baseline_rows[1]
            | {
                "matches": "14",
                "correct": "6",
                "wrong": "8",
                "precision": "0.428571",
                "score_min": "17",
                "score_mean": "20",
                "score_median": "20",
                "score_max": "22",
                "median_error_px": "5.4",
                "bbox_area_a_px2": "100",
                "bbox_area_b_px2": "100",
                "displacement_mad_px": "5",
                "homography_residual_valid": "1",
                "homography_residual_p90_px": "8",
            },
            baseline_rows[2]
            | {
                "matches": "10",
                "correct": "10",
                "wrong": "0",
                "precision": "1.0",
                "score_min": "12",
                "score_mean": "18",
            },
        ]

        candidates = rescue_mod.find_rescue_candidates(
            baseline_rows,
            candidate_rows,
            target_variants=("extreme_02", "extreme_03"),
            min_candidate_matches=8,
            max_candidate_matches=15,
        )
        sweep = rescue_mod.sweep_score_rules(
            candidates,
            score_min_thresholds=[0.0, 12.0, 16.0],
            score_mean_thresholds=[0.0, 18.0],
        )

        self.assertEqual([row.base_id for row in candidates], ["base_good", "base_bad"])
        self.assertEqual(candidates[0].candidate_matches, 8)
        self.assertEqual(candidates[1].candidate_wrong, 8)
        self.assertEqual(sweep[0].rows, 2)
        self.assertEqual(sweep[0].correct, 14)
        self.assertEqual(sweep[0].wrong, 8)
        strict = [row for row in sweep if row.score_min_threshold == 16.0 and row.score_mean_threshold == 18.0][0]
        self.assertEqual(strict.rows, 1)
        self.assertAlmostEqual(strict.precision, 6 / 14)
        geometry_sweep = rescue_mod.sweep_score_rules(
            candidates,
            score_min_thresholds=[0.0],
            score_mean_thresholds=[0.0],
            min_bbox_area_a_px2_thresholds=[90.0],
            max_homography_residual_p90_px_thresholds=[3.0],
            max_displacement_mad_px_thresholds=[2.0],
        )
        self.assertEqual(geometry_sweep[0].rows, 1)
        self.assertEqual(geometry_sweep[0].correct, 8)
        self.assertEqual(geometry_sweep[0].wrong, 0)
        self.assertAlmostEqual(geometry_sweep[0].precision, 1.0)

    def test_hard_failure_mining_selects_metric_failures_and_keeps_pair_manifest_columns(self) -> None:
        pair_rows = [
            {
                "pair_index": "0",
                "split": "train",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_a_nadir",
                "reference_base_id": "base_a",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_a_extreme03",
                "target_base_id": "base_a",
                "target_variant": "extreme_03",
                "valid_fraction": "0.6",
                "valid_pixels": "100",
                "attempts": "1",
                "crop_a_x0": "",
                "crop_a_y0": "",
                "crop_a_x1": "",
                "crop_a_y1": "",
                "crop_b_x0": "",
                "crop_b_y0": "",
                "crop_b_x1": "",
                "crop_b_y1": "",
            },
            {
                "pair_index": "1",
                "split": "train",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_b_nadir",
                "reference_base_id": "base_b",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_b_mid01",
                "target_base_id": "base_b",
                "target_variant": "mid_01",
                "valid_fraction": "0.7",
                "valid_pixels": "100",
                "attempts": "1",
                "crop_a_x0": "",
                "crop_a_y0": "",
                "crop_a_x1": "",
                "crop_a_y1": "",
                "crop_b_x0": "",
                "crop_b_y0": "",
                "crop_b_x1": "",
                "crop_b_y1": "",
            },
        ]
        summary_rows = [
            {"base_id": "base_a", "target_variant": "extreme_03", "split": "train", "matches": "220", "correct": "80", "wrong": "140", "precision": "0.363636"},
            {"base_id": "base_b", "target_variant": "mid_01", "split": "train", "matches": "2", "correct": "2", "wrong": "0", "precision": "1.0"},
        ]
        config = hard_mine_mod.HardFailureConfig(
            low_precision_threshold=0.85,
            high_wrong_threshold=32,
            low_match_threshold=4,
            extreme_variants=("extreme_02", "extreme_03"),
            include_extreme_without_failure=False,
        )

        mined = hard_mine_mod.mine_hard_failure_rows(pair_rows, summary_rows, config=config)

        self.assertEqual(len(mined), 2)
        by_target = {row["target_variant"]: row for row in mined}
        self.assertIn("low_precision", by_target["extreme_03"]["hard_reasons"])
        self.assertIn("high_false", by_target["extreme_03"]["hard_reasons"])
        self.assertIn("extreme_view", by_target["extreme_03"]["hard_reasons"])
        self.assertIn("low_match_count", by_target["mid_01"]["hard_reasons"])
        self.assertEqual(by_target["extreme_03"]["reference_pose_id"], "pose_a_nadir")

    def test_rescue_gain_hard_set_mining_selects_extreme_correct_gain_pairs(self) -> None:
        pair_rows = [
            {
                "pair_index": "0",
                "split": "val",
                "pair_type": "same_position_view",
                "reference_dataset_id": "h100km_fov076",
                "reference_pose_id": "base_nadir",
                "reference_base_id": "base",
                "reference_variant": "nadir",
                "target_dataset_id": "h100km_fov076",
                "target_pose_id": "base_extreme_02",
                "target_base_id": "base",
                "target_variant": "extreme_02",
                "valid_fraction": "1.0",
                "valid_pixels": "4194304",
                "attempts": "1",
                "crop_a_x0": "0",
                "crop_a_y0": "0",
                "crop_a_x1": "2048",
                "crop_a_y1": "2048",
                "crop_b_x0": "0",
                "crop_b_y0": "0",
                "crop_b_x1": "2048",
                "crop_b_y1": "2048",
            },
            {
                "pair_index": "1",
                "split": "val",
                "pair_type": "same_position_view",
                "reference_dataset_id": "h100km_fov076",
                "reference_pose_id": "base_nadir",
                "reference_base_id": "base",
                "reference_variant": "nadir",
                "target_dataset_id": "h100km_fov076",
                "target_pose_id": "base_mid_02",
                "target_base_id": "base",
                "target_variant": "mid_02",
                "valid_fraction": "1.0",
                "valid_pixels": "4194304",
                "attempts": "1",
                "crop_a_x0": "0",
                "crop_a_y0": "0",
                "crop_a_x1": "2048",
                "crop_a_y1": "2048",
                "crop_b_x0": "0",
                "crop_b_y0": "0",
                "crop_b_x1": "2048",
                "crop_b_y1": "2048",
            },
            {
                "pair_index": "2",
                "split": "val",
                "pair_type": "same_position_view",
                "reference_dataset_id": "h100km_fov076",
                "reference_pose_id": "other_nadir",
                "reference_base_id": "other",
                "reference_variant": "nadir",
                "target_dataset_id": "h100km_fov076",
                "target_pose_id": "other_extreme_03",
                "target_base_id": "other",
                "target_variant": "extreme_03",
                "valid_fraction": "1.0",
                "valid_pixels": "4194304",
                "attempts": "1",
                "crop_a_x0": "0",
                "crop_a_y0": "0",
                "crop_a_x1": "2048",
                "crop_a_y1": "2048",
                "crop_b_x0": "0",
                "crop_b_y0": "0",
                "crop_b_x1": "2048",
                "crop_b_y1": "2048",
            },
        ]
        baseline_rows = [
            {"target_variant": "extreme_02", "matches": "0", "correct": "0", "wrong": "0", "precision": "0.0"},
            {"target_variant": "mid_02", "matches": "10", "correct": "10", "wrong": "0", "precision": "1.0"},
            {"target_variant": "extreme_03", "matches": "20", "correct": "18", "wrong": "2", "precision": "0.9"},
        ]
        candidate_rows = [
            {
                "target_variant": "extreme_02",
                "matches": "16",
                "correct": "13",
                "wrong": "3",
                "precision": "0.8125",
                "homography_residual_p90_px": "4.2",
                "homography_residual_median_px": "1.9",
                "score_mean": "18.1",
            },
            {"target_variant": "mid_02", "matches": "20", "correct": "20", "wrong": "0", "precision": "1.0"},
            {"target_variant": "extreme_03", "matches": "21", "correct": "18", "wrong": "3", "precision": "0.857"},
        ]

        mined = rescue_gain_mod.mine_rescue_gain_rows(
            pair_rows,
            baseline_rows,
            candidate_rows,
            split="val",
            target_variants=("extreme_02", "extreme_03"),
        )

        self.assertEqual(len(mined), 1)
        self.assertEqual(mined[0]["target_variant"], "extreme_02")
        self.assertEqual(mined[0]["reference_pose_id"], "base_nadir")
        self.assertEqual(mined[0]["delta_correct"], "13")
        self.assertEqual(mined[0]["delta_wrong"], "3")
        self.assertEqual(mined[0]["correct_delta"], "13")
        self.assertEqual(mined[0]["wrong_delta"], "3")
        self.assertEqual(mined[0]["match_delta"], "16")
        self.assertIn("rescue_correct_gain", mined[0]["hard_reasons"])
        self.assertIn("rescue_false_negative", mined[0]["hard_reasons"])
        self.assertEqual(mined[0]["source_pair_index"], "0")
        self.assertEqual(mined[0]["pair_index"], "0")

    def test_hard_failure_mining_preserves_repeated_base_variant_pairs(self) -> None:
        pair_rows = [
            {
                "pair_index": "0",
                "split": "train",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_a_nadir",
                "reference_base_id": "base_repeat",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_a_extreme03_camera0",
                "target_base_id": "base_repeat",
                "target_variant": "extreme_03",
            },
            {
                "pair_index": "1",
                "split": "train",
                "pair_type": "cross_camera",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_b_nadir",
                "reference_base_id": "base_repeat",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_b_extreme03_camera1",
                "target_base_id": "base_repeat",
                "target_variant": "extreme_03",
            },
        ]
        summary_rows = [
            {"base_id": "base_repeat", "target_variant": "extreme_03", "split": "train", "matches": "2", "correct": "0", "wrong": "2", "precision": "0.0"},
            {"base_id": "base_repeat", "target_variant": "extreme_03", "split": "train", "matches": "3", "correct": "1", "wrong": "2", "precision": "0.333333"},
        ]
        config = hard_mine_mod.HardFailureConfig(
            low_precision_threshold=0.95,
            high_wrong_threshold=1,
            low_match_threshold=8,
            extreme_variants=("extreme_02", "extreme_03"),
        )

        mined = hard_mine_mod.mine_hard_failure_rows(pair_rows, summary_rows, config=config)

        self.assertEqual(len(mined), 2)
        self.assertEqual(
            [row["target_pose_id"] for row in mined],
            ["pose_a_extreme03_camera0", "pose_b_extreme03_camera1"],
        )

    def test_hard_failure_mining_builds_mixed_manifest_with_target_fraction(self) -> None:
        base_rows = [
            {"pair_index": str(index), "reference_base_id": f"base_{index}", "reference_variant": "nadir", "target_variant": "mid_01"}
            for index in range(10)
        ]
        hard_rows = [
            {"pair_index": "0", "reference_base_id": "hard_a", "reference_variant": "nadir", "target_variant": "extreme_03", "hard_reasons": "low_precision"},
            {"pair_index": "1", "reference_base_id": "hard_b", "reference_variant": "nadir", "target_variant": "extreme_02", "hard_reasons": "high_false"},
        ]

        mixed = hard_mine_mod.build_mixed_manifest_rows(base_rows, hard_rows, target_hard_fraction=0.35)

        hard_count = sum(1 for row in mixed if row.get("hard_reasons"))
        self.assertGreaterEqual(hard_count / len(mixed), 0.35)
        self.assertEqual([row["pair_index"] for row in mixed], [str(index) for index in range(len(mixed))])

    def test_hard_failure_mining_can_mix_hard_rows_into_separate_base_manifest(self) -> None:
        def pair_row(index: int, base_id: str, target_variant: str) -> dict[str, str]:
            return {
                field: ""
                for field in hard_mine_mod.PAIR_MANIFEST_FIELDS
            } | {
                "pair_index": str(index),
                "split": "train",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": f"{base_id}_nadir",
                "reference_base_id": base_id,
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": f"{base_id}_{target_variant}",
                "target_base_id": base_id,
                "target_variant": target_variant,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_manifest = root / "sample.csv"
            full_base_manifest = root / "full_base.csv"
            summary_csv = root / "all_filtered_summary.csv"
            hard_manifest = root / "hard.csv"
            mixed_manifest = root / "mixed.csv"

            self.write_csv(sample_manifest, hard_mine_mod.PAIR_MANIFEST_FIELDS, [pair_row(0, "failed", "extreme_03")])
            self.write_csv(
                full_base_manifest,
                hard_mine_mod.PAIR_MANIFEST_FIELDS,
                [pair_row(index, f"base_{index}", "extreme_02") for index in range(10)],
            )
            self.write_csv(
                summary_csv,
                ["base_id", "target_variant", "split", "matches", "correct", "wrong", "precision"],
                [
                    {
                        "base_id": "failed",
                        "target_variant": "extreme_03",
                        "split": "train",
                        "matches": "4",
                        "correct": "1",
                        "wrong": "3",
                        "precision": "0.25",
                    }
                ],
            )

            argv = [
                "mine_hard_failure_pairs.py",
                "--pair-manifest",
                str(sample_manifest),
                "--summary-csv",
                str(summary_csv),
                "--output-manifest",
                str(hard_manifest),
                "--mixed-base-manifest",
                str(full_base_manifest),
                "--mixed-output-manifest",
                str(mixed_manifest),
                "--mixed-hard-fraction",
                "0.2",
                "--residual-filtered",
            ]
            with mock.patch.object(sys, "argv", argv):
                hard_mine_mod.main()

            with mixed_manifest.open("r", encoding="utf-8", newline="") as handle:
                mixed_rows = list(csv.DictReader(handle))

        self.assertGreater(len(mixed_rows), 10)
        self.assertTrue(any(row["reference_base_id"] == "base_9" for row in mixed_rows))
        self.assertTrue(any(row["reference_base_id"] == "failed" and row["hard_reasons"] for row in mixed_rows))

    def test_hard_failure_residual_filtered_preset_targets_post_magsac_failures(self) -> None:
        argv = [
            "mine_hard_failure_pairs.py",
            "--pair-manifest",
            "pairs.csv",
            "--summary-csv",
            "all_filtered_summary.csv",
            "--output-manifest",
            "residual.csv",
            "--residual-filtered",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = hard_mine_mod.parse_args()
        config = hard_mine_mod.config_from_args(args)
        pair_rows = [
            {
                "pair_index": "0",
                "split": "val",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_failed_nadir",
                "reference_base_id": "failed",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_failed_extreme03",
                "target_base_id": "failed",
                "target_variant": "extreme_03",
            },
            {
                "pair_index": "1",
                "split": "val",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_clean_nadir",
                "reference_base_id": "clean",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_clean_extreme03",
                "target_base_id": "clean",
                "target_variant": "extreme_03",
            },
        ]
        summary_rows = [
            {"base_id": "failed", "target_variant": "extreme_03", "split": "val", "matches": "6", "correct": "0", "wrong": "6", "precision": "0.0"},
            {"base_id": "clean", "target_variant": "extreme_03", "split": "val", "matches": "72", "correct": "72", "wrong": "0", "precision": "1.0"},
        ]

        mined = hard_mine_mod.mine_hard_failure_rows(pair_rows, summary_rows, config=config)

        self.assertEqual(args.failure_preset, "residual_filtered")
        self.assertEqual(config.low_precision_threshold, 0.95)
        self.assertEqual(config.high_wrong_threshold, 1)
        self.assertEqual(config.low_match_threshold, 8)
        self.assertEqual(len(mined), 1)
        self.assertEqual(mined[0]["reference_base_id"], "failed")
        self.assertIn("low_precision", mined[0]["hard_reasons"])
        self.assertIn("high_false", mined[0]["hard_reasons"])
        self.assertIn("low_match_count", mined[0]["hard_reasons"])
        self.assertIn("extreme_view", mined[0]["hard_reasons"])

    def test_hard_failure_mining_can_restrict_to_extreme_variants(self) -> None:
        pair_rows = [
            {
                "pair_index": "0",
                "split": "train",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_mid_nadir",
                "reference_base_id": "base_mid",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_mid_mid01",
                "target_base_id": "base_mid",
                "target_variant": "mid_01",
            },
            {
                "pair_index": "1",
                "split": "train",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_extreme_nadir",
                "reference_base_id": "base_extreme",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_extreme_extreme03",
                "target_base_id": "base_extreme",
                "target_variant": "extreme_03",
            },
        ]
        summary_rows = [
            {"base_id": "base_mid", "target_variant": "mid_01", "split": "train", "matches": "1", "correct": "1", "wrong": "0", "precision": "1.0"},
            {"base_id": "base_extreme", "target_variant": "extreme_03", "split": "train", "matches": "2", "correct": "0", "wrong": "2", "precision": "0.0"},
        ]
        config = hard_mine_mod.HardFailureConfig(
            low_precision_threshold=0.95,
            high_wrong_threshold=1,
            low_match_threshold=8,
            extreme_variants=("extreme_02", "extreme_03"),
            only_extreme_variants=True,
        )

        mined = hard_mine_mod.mine_hard_failure_rows(pair_rows, summary_rows, config=config)

        self.assertEqual(len(mined), 1)
        self.assertEqual(mined[0]["reference_base_id"], "base_extreme")
        self.assertEqual(mined[0]["target_variant"], "extreme_03")
        self.assertIn("extreme_view", mined[0]["hard_reasons"])

    def test_hard_failure_mining_can_filter_rows_by_required_reasons(self) -> None:
        rows = [
            {"reference_base_id": "low_precision_pair", "hard_reasons": "low_precision|extreme_view"},
            {"reference_base_id": "low_match_pair", "hard_reasons": "low_match_count|extreme_view"},
            {"reference_base_id": "high_false_pair", "hard_reasons": "high_false|low_precision|extreme_view"},
        ]

        low_precision = hard_mine_mod.filter_hard_failure_rows_by_required_reasons(rows, ["low_precision"])
        extreme_high_false = hard_mine_mod.filter_hard_failure_rows_by_required_reasons(
            rows,
            ["extreme_view", "high_false"],
        )
        unfiltered = hard_mine_mod.filter_hard_failure_rows_by_required_reasons(rows, [])

        self.assertEqual([row["reference_base_id"] for row in low_precision], ["low_precision_pair", "high_false_pair"])
        self.assertEqual([row["reference_base_id"] for row in extreme_high_false], ["high_false_pair"])
        self.assertEqual(unfiltered, rows)

    def test_hard_failure_mining_parse_args_accepts_required_reason_filters(self) -> None:
        argv = [
            "mine_hard_failure_pairs.py",
            "--pair-manifest",
            "pairs.csv",
            "--summary-csv",
            "all_filtered_summary.csv",
            "--output-manifest",
            "low_precision.csv",
            "--required-reason",
            "low_precision",
            "--required-reason",
            "extreme_view",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = hard_mine_mod.parse_args()

        self.assertEqual(args.required_reason, ["low_precision", "extreme_view"])

    def test_pair_delta_regression_mining_selects_candidate_regressions(self) -> None:
        pair_rows = [
            {
                "pair_index": "10",
                "split": "test",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_a_nadir",
                "reference_base_id": "base_a",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_a_mid01",
                "target_base_id": "base_a",
                "target_variant": "mid_01",
            },
            {
                "pair_index": "11",
                "split": "test",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_b_mid02",
                "reference_base_id": "base_b",
                "reference_variant": "mid_02",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_b_extreme02",
                "target_base_id": "base_b",
                "target_variant": "extreme_02",
            },
            {
                "pair_index": "12",
                "split": "test",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_c_mid01",
                "reference_base_id": "base_c",
                "reference_variant": "mid_01",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_c_extreme01",
                "target_base_id": "base_c",
                "target_variant": "extreme_01",
            },
            {
                "pair_index": "13",
                "split": "test",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_d_mid02",
                "reference_base_id": "base_d",
                "reference_variant": "mid_02",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_d_extreme01",
                "target_base_id": "base_d",
                "target_variant": "extreme_01",
            },
        ]
        delta_rows = [
            {
                "split": "test",
                "pair_index": "0",
                "base_id": "base_a",
                "reference_variant": "nadir",
                "target_variant": "mid_01",
                "baseline_matches": "100",
                "baseline_correct": "98",
                "baseline_wrong": "2",
                "baseline_precision": "0.98",
                "candidate_matches": "102",
                "candidate_correct": "101",
                "candidate_wrong": "1",
                "candidate_precision": "0.990196",
                "match_delta": "2",
                "correct_delta": "3",
                "wrong_delta": "-1",
                "precision_delta": "0.010196",
            },
            {
                "split": "test",
                "pair_index": "1",
                "base_id": "base_b",
                "reference_variant": "mid_02",
                "target_variant": "extreme_02",
                "baseline_matches": "80",
                "baseline_correct": "75",
                "baseline_wrong": "5",
                "baseline_precision": "0.9375",
                "candidate_matches": "84",
                "candidate_correct": "74",
                "candidate_wrong": "10",
                "candidate_precision": "0.880952",
                "match_delta": "4",
                "correct_delta": "-1",
                "wrong_delta": "5",
                "precision_delta": "-0.056548",
            },
            {
                "split": "test",
                "pair_index": "2",
                "base_id": "base_c",
                "reference_variant": "mid_01",
                "target_variant": "extreme_01",
                "baseline_matches": "50",
                "baseline_correct": "48",
                "baseline_wrong": "2",
                "baseline_precision": "0.96",
                "candidate_matches": "48",
                "candidate_correct": "42",
                "candidate_wrong": "6",
                "candidate_precision": "0.875",
                "match_delta": "-2",
                "correct_delta": "-6",
                "wrong_delta": "4",
                "precision_delta": "-0.085",
            },
            {
                "split": "test",
                "pair_index": "3",
                "base_id": "base_d",
                "reference_variant": "mid_02",
                "target_variant": "extreme_01",
                "baseline_matches": "0",
                "baseline_correct": "0",
                "baseline_wrong": "0",
                "baseline_precision": "0.0",
                "candidate_matches": "19",
                "candidate_correct": "13",
                "candidate_wrong": "6",
                "candidate_precision": "0.684211",
                "match_delta": "19",
                "correct_delta": "13",
                "wrong_delta": "6",
                "precision_delta": "0.684211",
            },
        ]
        config = delta_mine_mod.PairDeltaMiningConfig(min_precision_drop=0.01)

        mined = delta_mine_mod.mine_pair_delta_regression_rows(pair_rows, delta_rows, config=config)

        self.assertEqual(len(mined), 3)
        by_base = {row["reference_base_id"]: row for row in mined}
        self.assertNotIn("base_a", by_base)
        self.assertIn("wrong_increase", by_base["base_b"]["hard_reasons"])
        self.assertIn("precision_regression", by_base["base_b"]["hard_reasons"])
        self.assertIn("correct_regression", by_base["base_c"]["hard_reasons"])
        self.assertIn("candidate_wrong_from_zero", by_base["base_d"]["hard_reasons"])
        self.assertEqual(by_base["base_b"]["reference_pose_id"], "pose_b_mid02")
        self.assertEqual(by_base["base_b"]["source_wrong_delta"], "5")

    def test_pair_delta_regression_mining_parse_args_accepts_multiple_sources(self) -> None:
        argv = [
            "mine_pair_delta_regression_pairs.py",
            "--pair-manifest",
            "regression_val.csv",
            "--pair-manifest",
            "regression_test.csv",
            "--pair-delta-csv",
            "delta_val.csv",
            "--pair-delta-csv",
            "delta_test.csv",
            "--output-manifest",
            "hard.csv",
            "--mixed-base-manifest",
            "train.csv",
            "--mixed-output-manifest",
            "mixed.csv",
            "--mixed-hard-fraction",
            "0.2",
            "--required-reason",
            "wrong_increase",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = delta_mine_mod.parse_args()
        sources = delta_mine_mod.expand_sources(args.pair_manifest, args.pair_delta_csv)

        self.assertEqual([source.pair_manifest.name for source in sources], ["regression_val.csv", "regression_test.csv"])
        self.assertEqual([source.pair_delta_csv.name for source in sources], ["delta_val.csv", "delta_test.csv"])
        self.assertEqual(args.mixed_hard_fraction, 0.2)
        self.assertEqual(args.required_reason, ["wrong_increase"])

    def test_pair_delta_summary_builder_aligns_pair_manifest_with_filtered_summaries(self) -> None:
        import build_pair_delta_summary_from_filtered_summaries as delta_summary_mod

        pair_rows = [
            {
                "pair_index": "0",
                "split": "val",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_a_mid01",
                "reference_base_id": "base_a",
                "reference_variant": "mid_01",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_a_extreme02",
                "target_base_id": "base_a",
                "target_variant": "extreme_02",
            },
            {
                "pair_index": "1",
                "split": "val",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "pose_b_nadir",
                "reference_base_id": "base_b",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "pose_b_mid01",
                "target_base_id": "base_b",
                "target_variant": "mid_01",
            },
        ]
        baseline_rows = [
            {"base_id": "base_a", "target_variant": "extreme_02", "matches": "10", "correct": "8", "wrong": "2", "precision": "0.8"},
            {"base_id": "base_b", "target_variant": "mid_01", "matches": "20", "correct": "20", "wrong": "0", "precision": "1.0"},
        ]
        candidate_rows = [
            {"base_id": "base_a", "target_variant": "extreme_02", "matches": "14", "correct": "13", "wrong": "1", "precision": "0.928571"},
            {"base_id": "base_b", "target_variant": "mid_01", "matches": "18", "correct": "17", "wrong": "1", "precision": "0.944444"},
        ]

        rows = delta_summary_mod.build_pair_delta_rows(
            pair_rows,
            baseline_rows,
            candidate_rows,
            split="val",
            source_name="candidate_vs_baseline",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["reference_variant"], "mid_01")
        self.assertEqual(rows[0]["target_variant"], "extreme_02")
        self.assertEqual(rows[0]["match_delta"], "4")
        self.assertEqual(rows[0]["correct_delta"], "5")
        self.assertEqual(rows[0]["wrong_delta"], "-1")
        self.assertEqual(rows[1]["reference_variant"], "nadir")
        self.assertEqual(rows[1]["target_variant"], "mid_01")
        self.assertEqual(rows[1]["correct_delta"], "-3")
        self.assertEqual(rows[1]["wrong_delta"], "1")
        self.assertAlmostEqual(float(rows[1]["precision_delta"]), -0.055556, places=6)

    def test_train_pattern_replay_samples_only_train_rows_from_delta_patterns(self) -> None:
        import build_train_replay_from_pair_deltas as replay_mod

        train_rows = [
            {
                "pair_index": "0",
                "split": "train",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "train_a_nadir",
                "reference_base_id": "train_a",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "train_a_mid02",
                "target_base_id": "train_a",
                "target_variant": "mid_02",
            },
            {
                "pair_index": "1",
                "split": "train",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "train_b_mid02",
                "reference_base_id": "train_b",
                "reference_variant": "mid_02",
                "target_dataset_id": "fov076",
                "target_pose_id": "train_b_extreme02",
                "target_base_id": "train_b",
                "target_variant": "extreme_02",
            },
            {
                "pair_index": "2",
                "split": "val",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "val_leak_nadir",
                "reference_base_id": "val_leak",
                "reference_variant": "nadir",
                "target_dataset_id": "fov076",
                "target_pose_id": "val_leak_mid02",
                "target_base_id": "val_leak",
                "target_variant": "mid_02",
            },
            {
                "pair_index": "3",
                "split": "train",
                "pair_type": "same_position_view",
                "reference_dataset_id": "fov076",
                "reference_pose_id": "train_c_mid02",
                "reference_base_id": "train_c",
                "reference_variant": "mid_02",
                "target_dataset_id": "fov076",
                "target_pose_id": "train_c_extreme03",
                "target_base_id": "train_c",
                "target_variant": "extreme_03",
            },
        ]
        regression_delta_rows = [
            {
                "split": "test",
                "base_id": "test_a",
                "reference_variant": "nadir",
                "target_variant": "mid_02",
                "match_delta": "-8",
                "correct_delta": "-23",
                "wrong_delta": "3",
                "precision_delta": "-0.03",
            },
            {
                "split": "test",
                "base_id": "test_b",
                "reference_variant": "mid_02",
                "target_variant": "extreme_02",
                "match_delta": "4",
                "correct_delta": "-14",
                "wrong_delta": "5",
                "precision_delta": "-0.06",
            },
        ]
        gain_delta_rows = [
            {
                "split": "test",
                "base_id": "test_c",
                "reference_variant": "mid_02",
                "target_variant": "extreme_03",
                "match_delta": "3",
                "correct_delta": "9",
                "wrong_delta": "-6",
                "precision_delta": "0.12",
            }
        ]

        patterns = replay_mod.collect_delta_patterns(
            regression_delta_rows=regression_delta_rows,
            gain_delta_rows=gain_delta_rows,
            config=replay_mod.PatternReplayConfig(min_precision_drop=0.01, min_gain_correct=1),
        )
        sampled = replay_mod.sample_train_rows_by_patterns(
            train_rows,
            patterns,
            max_per_pattern=2,
            seed=11,
        )

        self.assertEqual({row["reference_base_id"] for row in sampled}, {"train_a", "train_b", "train_c"})
        self.assertTrue(all(row["split"] == "train" for row in sampled))
        by_base = {row["reference_base_id"]: row for row in sampled}
        self.assertIn("protect_regression", by_base["train_a"]["pattern_reasons"])
        self.assertIn("correct_regression", by_base["train_b"]["pattern_reasons"])
        self.assertIn("extreme_gain", by_base["train_c"]["pattern_reasons"])
        self.assertEqual([row["pair_index"] for row in sampled], ["0", "1", "2"])

    def test_lazy_visual_parse_args_defaults_to_filtered_all_match_report(self) -> None:
        argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "report",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = visual_mod.parse_args()

        self.assertTrue(args.filtered_report)
        self.assertTrue(args.filtered_mutual)
        self.assertEqual(args.filtered_geometry_filter, "local")
        self.assertEqual(args.max_matches, 0)
        self.assertEqual(args.draw_matches, 0)
        self.assertEqual(args.filtered_max_matches, 0)
        self.assertEqual(args.filtered_draw_matches, 0)
        self.assertEqual(args.filtered_min_matches, 0)
        self.assertGreater(args.filtered_min_margin, 0.0)

    def test_lazy_visual_selects_pair_specs_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = RenderRecord(
                pose_id="fov090_a_mid",
                base_id="fov090:a",
                variant="mid_01",
                split="train",
                tsai_path=root / "a.tsai",
                image_path=root / "a.tif",
                uint8_path=root / "a.png",
                depth_path=root / "a_depth.tif",
                dataset_id="fov090",
                raw_base_id="a",
            )
            target = RenderRecord(
                pose_id="fov090_b_extreme",
                base_id="fov090:b",
                variant="extreme_03",
                split="train",
                tsai_path=root / "b.tsai",
                image_path=root / "b.tif",
                uint8_path=root / "b.png",
                depth_path=root / "b_depth.tif",
                dataset_id="fov090",
                raw_base_id="b",
            )
            manifest = root / "overlap_edges.csv"
            pair = SyntheticPair(
                view_a=torch.zeros(1, 4, 4),
                view_b=torch.zeros(1, 4, 4),
                warp_a_to_b=torch.zeros(4, 4, 2),
                valid_mask=torch.ones(4, 4, dtype=torch.bool),
            )
            write_pair_spec_manifest(
                manifest,
                [
                    LazyPairResult(
                        spec=LazyPairSpec(
                            pair_index=0,
                            split="train",
                            reference=reference,
                            target=target,
                            pair_type=PAIR_TYPE_CROSS_CAMERA,
                        ),
                        pair=pair,
                        valid_fraction=0.5,
                        valid_pixels=16,
                        attempt_count=1,
                        elapsed_ms=1.0,
                        crop_a=CropWindow(1, 2, 5, 6),
                        crop_b=CropWindow(7, 8, 11, 12),
                    )
                ],
            )
            args = SimpleNamespace(
                pair_spec_manifest=manifest,
                split="train",
                reference_variant="nadir",
                target_variant=[],
                pair_mode="same-position",
                cross_pair_variant=[],
                cross_camera_offsets=(1,),
                cross_fov_offsets=(0,),
                pair_type_weights={},
                spatial_index_planet_radius_m=3396190.0,
                spatial_index_footprint_samples=5,
                spatial_index_margin_m=2000.0,
                spatial_index_height_km=[],
                image_source="uint8",
                limit_pairs=0,
                seed=123,
                shuffle=False,
            )

            with mock.patch.object(visual_mod, "build_lazy_pair_specs", side_effect=AssertionError("should not build")):
                specs, pair_source, pair_type_counts = visual_mod.select_visual_pair_specs(args, [reference, target])

        self.assertEqual(pair_source, "pair_spec_manifest")
        self.assertEqual(pair_type_counts[PAIR_TYPE_CROSS_CAMERA], 1)
        self.assertEqual(specs[0].target.pose_id, target.pose_id)
        self.assertEqual(specs[0].fixed_crop_a, CropWindow(1, 2, 5, 6))
        self.assertEqual(specs[0].fixed_crop_b, CropWindow(7, 8, 11, 12))

    def test_lazy_visual_reads_manifest_dataset_id_compatible_with_pair_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            render_manifest = manifest_dir / "h100km_fov090_render_manifest.csv"
            uint8_manifest = manifest_dir / "h100km_fov090_uint8_manifest.csv"
            render_manifest.write_text(
                "pose_id,base_id,variant,split,lon_deg,lat_deg,tsai_path,image_path,depth_path,chunk_index\n"
                f"pose_a,base_a,mid_01,train,0,0,{root / 'a.tsai'},{root / 'a.tif'},{root / 'a_depth.tif'},0\n"
                f"pose_b,base_b,extreme_03,train,0,0,{root / 'b.tsai'},{root / 'b.tif'},{root / 'b_depth.tif'},0\n",
                encoding="utf-8",
            )
            uint8_manifest.write_text("source_path,uint8_path\n", encoding="utf-8")
            records = [
                RenderRecord(
                    pose_id="pose_a",
                    base_id="base_a",
                    variant="mid_01",
                    split="train",
                    tsai_path=root / "a.tsai",
                    image_path=root / "a.tif",
                    uint8_path=root / "a.tif",
                    depth_path=root / "a_depth.tif",
                    dataset_id="h100km_fov090",
                    raw_base_id="base_a",
                ),
                RenderRecord(
                    pose_id="pose_b",
                    base_id="base_b",
                    variant="extreme_03",
                    split="train",
                    tsai_path=root / "b.tsai",
                    image_path=root / "b.tif",
                    uint8_path=root / "b.tif",
                    depth_path=root / "b_depth.tif",
                    dataset_id="h100km_fov090",
                    raw_base_id="base_b",
                ),
            ]
            manifest = root / "overlap_edges.csv"
            pair = SyntheticPair(
                view_a=torch.zeros(1, 4, 4),
                view_b=torch.zeros(1, 4, 4),
                warp_a_to_b=torch.zeros(4, 4, 2),
                valid_mask=torch.ones(4, 4, dtype=torch.bool),
            )
            write_pair_spec_manifest(
                manifest,
                [
                    LazyPairResult(
                        spec=LazyPairSpec(
                            pair_index=0,
                            split="train",
                            reference=records[0],
                            target=records[1],
                            pair_type=PAIR_TYPE_CROSS_CAMERA,
                        ),
                        pair=pair,
                        valid_fraction=0.5,
                        valid_pixels=16,
                        attempt_count=1,
                        elapsed_ms=1.0,
                        crop_a=CropWindow(0, 0, 4, 4),
                        crop_b=CropWindow(0, 0, 4, 4),
                    )
                ],
            )
            args = SimpleNamespace(
                pair_spec_manifest=manifest,
                split="train",
                reference_variant="nadir",
                target_variant=[],
                pair_mode="same-position",
                cross_pair_variant=[],
                cross_camera_offsets=(1,),
                cross_fov_offsets=(0,),
                pair_type_weights={},
                spatial_index_planet_radius_m=3396190.0,
                spatial_index_footprint_samples=5,
                spatial_index_margin_m=2000.0,
                spatial_index_height_km=[],
                image_source="uint8",
                limit_pairs=0,
                seed=123,
                shuffle=False,
            )

            visual_records = visual_mod.read_visual_records(render_manifest, uint8_manifest)
            specs, pair_source, pair_type_counts = visual_mod.select_visual_pair_specs(args, visual_records)

        self.assertEqual(pair_source, "pair_spec_manifest")
        self.assertEqual(pair_type_counts[PAIR_TYPE_CROSS_CAMERA], 1)
        self.assertEqual(specs[0].reference.dataset_id, "h100km_fov090")
        self.assertEqual(specs[0].target.pose_id, "pose_b")

    def test_smooth_series_keeps_short_series_length(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0]).numpy()

        smoothed = visual_mod.smooth_series(values, window=5)

        self.assertEqual(smoothed.shape, values.shape)

    def test_illumination_variants_preserve_shape(self) -> None:
        image = torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(1, 4, 4)
        variants = make_illumination_variants(image)

        names = [name for name, _ in variants]
        self.assertIn("original", names)
        self.assertIn("gamma_dark", names)
        self.assertIn("shadow_band", names)
        self.assertTrue(all(variant.shape == image.shape for _, variant in variants))
        self.assertFalse(torch.allclose(dict(variants)["gamma_dark"], image))

    def test_zero_degree_rotation_keeps_identity_warp(self) -> None:
        image = torch.arange(16, dtype=torch.float32).view(1, 4, 4) / 15.0
        pair = rotate_pair_from_view(image, angle_deg=0.0)

        self.assertIsInstance(pair, SyntheticPair)
        self.assertTrue(torch.allclose(pair.view_a, image))
        self.assertTrue(torch.allclose(pair.view_b, image, atol=1e-5))
        self.assertTrue(pair.valid_mask.all())
        self.assertTrue(torch.allclose(pair.warp_a_to_b[0, 0], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.allclose(pair.warp_a_to_b[-1, -1], torch.tensor([3.0, 3.0])))

    def test_lazy_visual_illumination_stress_keeps_geometry(self) -> None:
        record = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        spec = LazyPairSpec(
            pair_index=7,
            split="train",
            reference=record,
            target=RenderRecord(
                pose_id="pose_b",
                base_id="base_001",
                variant="extreme_03",
                split="train",
                tsai_path=Path("b.tsai"),
                image_path=Path("b.tif"),
                uint8_path=Path("b.png"),
                depth_path=Path("b_depth.tif"),
            ),
        )
        pair = SyntheticPair(
            view_a=torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(1, 4, 4),
            view_b=torch.linspace(1.0, 0.0, 16, dtype=torch.float32).view(1, 4, 4),
            warp_a_to_b=torch.zeros(4, 4, 2, dtype=torch.float32),
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        visual = LazyMatchVisual(
            label="困难/失败",
            spec=spec,
            pair=pair,
            valid_fraction=0.5,
            points_a=torch.empty(0, 2).numpy(),
            points_b=torch.empty(0, 2).numpy(),
            scores=torch.empty(0).numpy(),
            errors=torch.empty(0).numpy(),
            correct=torch.empty(0, dtype=torch.bool).numpy(),
        )

        variants = make_illumination_stress_lazy_results([visual])

        self.assertGreater(len(variants), 1)
        self.assertTrue(any(item.label.endswith("gamma_dark") for item in variants))
        for item in variants:
            self.assertIsInstance(item.result, LazyPairResult)
            self.assertTrue(torch.allclose(item.result.pair.view_a, pair.view_a))
            self.assertTrue(torch.allclose(item.result.pair.warp_a_to_b, pair.warp_a_to_b))
            self.assertTrue(torch.equal(item.result.pair.valid_mask, pair.valid_mask))

    def test_lazy_visual_draw_zero_selects_all_matches(self) -> None:
        record = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        spec = LazyPairSpec(
            pair_index=1,
            split="train",
            reference=record,
            target=record,
        )
        pair = SyntheticPair(
            view_a=torch.zeros(1, 4, 4),
            view_b=torch.zeros(1, 4, 4),
            warp_a_to_b=torch.zeros(4, 4, 2),
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        visual = LazyMatchVisual(
            label="测试",
            spec=spec,
            pair=pair,
            valid_fraction=1.0,
            points_a=torch.zeros(5, 2).numpy(),
            points_b=torch.zeros(5, 2).numpy(),
            scores=torch.linspace(0.1, 0.5, 5).numpy(),
            errors=torch.zeros(5).numpy(),
            correct=torch.tensor([True, False, True, False, True]).numpy(),
        )

        self.assertEqual(selected_draw_indices(visual, 0).tolist(), [0, 1, 2, 3, 4])

    def test_lazy_visual_html_report_shows_source_image_paths(self) -> None:
        reference = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="val",
            tsai_path=Path("a.tsai"),
            image_path=Path("/raw/a.tif"),
            uint8_path=Path("/uint8/a.png"),
            depth_path=Path("a_depth.tif"),
        )
        target = RenderRecord(
            pose_id="pose_b",
            base_id="base_001",
            variant="extreme_03",
            split="val",
            tsai_path=Path("b.tsai"),
            image_path=Path("/raw/b.tif"),
            uint8_path=Path("/uint8/b.png"),
            depth_path=Path("b_depth.tif"),
        )
        spec = LazyPairSpec(pair_index=1, split="val", reference=reference, target=target)
        pair = SyntheticPair(
            view_a=torch.zeros(1, 4, 4),
            view_b=torch.zeros(1, 4, 4),
            warp_a_to_b=torch.zeros(4, 4, 2),
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        visual = LazyMatchVisual(
            label="测试",
            spec=spec,
            pair=pair,
            valid_fraction=1.0,
            points_a=torch.zeros(1, 2).numpy(),
            points_b=torch.zeros(1, 2).numpy(),
            scores=torch.ones(1).numpy(),
            errors=torch.zeros(1).numpy(),
            correct=torch.ones(1, dtype=torch.bool).numpy(),
            image_name="pair.png",
            crop_a=CropWindow(x0=10, y0=20, x1=778, y1=788),
            crop_b=CropWindow(x0=30, y0=40, x1=798, y1=808),
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "pair.png"
            image_path.write_bytes(b"fake-png")
            html_path = tmp_path / "index.html"

            visual_mod.write_html_report(
                html_path,
                args=SimpleNamespace(pytorch_state=Path("state.pt")),
                all_results=[visual],
                selected=[visual],
                image_paths={"pair.png": image_path},
                artifact_paths={},
                elapsed_s=1.0,
            )

            html_text = html_path.read_text(encoding="utf-8")

        self.assertIn("A图文件", html_text)
        self.assertIn("B图文件", html_text)
        self.assertIn("A图 crop", html_text)
        self.assertIn("B图 crop", html_text)
        self.assertIn("/uint8/a.png", html_text)
        self.assertIn("/uint8/b.png", html_text)
        self.assertIn("x=10, y=20, w=768, h=768", html_text)
        self.assertIn("x=30, y=40, w=768, h=768", html_text)

    def test_lazy_visual_geometry_filter_removes_outlier_matches(self) -> None:
        record = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        spec = LazyPairSpec(
            pair_index=1,
            split="train",
            reference=record,
            target=record,
        )
        yy, xx = torch.meshgrid(torch.arange(16, dtype=torch.float32), torch.arange(16, dtype=torch.float32), indexing="ij")
        warp = torch.stack([xx, yy], dim=-1)
        pair = SyntheticPair(
            view_a=torch.ones(1, 16, 16),
            view_b=torch.ones(1, 16, 16),
            warp_a_to_b=warp,
            valid_mask=torch.ones(16, 16, dtype=torch.bool),
        )
        visual = LazyMatchVisual(
            label="raw",
            spec=spec,
            pair=pair,
            valid_fraction=1.0,
            points_a=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0], [14.0, 2.0]]).numpy(),
            points_b=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0], [0.0, 15.0]]).numpy(),
            scores=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.95]).numpy(),
            errors=torch.tensor([0.0, 0.0, 0.0, 0.0, 20.0]).numpy(),
            correct=torch.tensor([True, True, True, True, False]).numpy(),
        )

        filtered = filter_visual_matches(visual, geometry_filter="local", threshold_px=1.0)

        self.assertEqual(filtered.label, "raw / filtered")
        self.assertEqual(filtered.matches, 4)
        self.assertEqual(filtered.correct_count, 4)

    def test_lazy_visual_robust_geometry_filter_removes_homography_outlier(self) -> None:
        record = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        spec = LazyPairSpec(pair_index=1, split="train", reference=record, target=record)
        pair = SyntheticPair(
            view_a=torch.ones(1, 32, 32),
            view_b=torch.ones(1, 32, 32),
            warp_a_to_b=torch.zeros(32, 32, 2),
            valid_mask=torch.ones(32, 32, dtype=torch.bool),
        )
        visual = LazyMatchVisual(
            label="raw",
            spec=spec,
            pair=pair,
            valid_fraction=1.0,
            points_a=torch.tensor([[2.0, 2.0], [20.0, 2.0], [2.0, 20.0], [20.0, 20.0], [12.0, 12.0]]).numpy(),
            points_b=torch.tensor([[5.0, 6.0], [23.0, 6.0], [5.0, 24.0], [23.0, 24.0], [30.0, 1.0]]).numpy(),
            scores=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.95]).numpy(),
            errors=torch.tensor([0.0, 0.0, 0.0, 0.0, 20.0]).numpy(),
            correct=torch.tensor([True, True, True, True, False]).numpy(),
        )

        filtered = filter_visual_matches(visual, geometry_filter="ransac", threshold_px=1.5)

        self.assertEqual(filtered.matches, 4)
        self.assertEqual(filtered.correct_count, 4)

    def test_lazy_visual_summary_artifacts_can_include_all_candidates_and_filtered_all(self) -> None:
        record = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        spec = LazyPairSpec(pair_index=1, split="train", reference=record, target=record)
        pair = SyntheticPair(
            view_a=torch.ones(1, 16, 16),
            view_b=torch.ones(1, 16, 16),
            warp_a_to_b=torch.zeros(16, 16, 2),
            valid_mask=torch.ones(16, 16, dtype=torch.bool),
        )
        good = LazyMatchVisual(
            label="good",
            spec=spec,
            pair=pair,
            valid_fraction=1.0,
            points_a=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0]]).numpy(),
            points_b=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0]]).numpy(),
            scores=torch.tensor([0.9, 0.8, 0.7, 0.6]).numpy(),
            errors=torch.zeros(4).numpy(),
            correct=torch.ones(4, dtype=torch.bool).numpy(),
        )
        bad = LazyMatchVisual(
            label="bad",
            spec=spec,
            pair=pair,
            valid_fraction=1.0,
            points_a=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0], [14.0, 2.0]]).numpy(),
            points_b=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0], [0.0, 15.0]]).numpy(),
            scores=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.95]).numpy(),
            errors=torch.tensor([0.0, 0.0, 0.0, 0.0, 20.0]).numpy(),
            correct=torch.tensor([True, True, True, True, False]).numpy(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            visual_mod.write_visual_summary_artifacts(
                output_dir,
                selected=[good],
                filtered_selected=[],
                all_results=[good, bad],
                write_all_summary=True,
                filtered_geometry_filter="local",
                filtered_threshold_px=1.0,
            )
            all_text = (output_dir / "all_summary.csv").read_text(encoding="utf-8")
            all_filtered_text = (output_dir / "all_filtered_summary.csv").read_text(encoding="utf-8")

        self.assertEqual(len(all_text.strip().splitlines()), 3)
        self.assertIn("bad", all_text)
        self.assertIn("all-filtered", all_filtered_text)
        self.assertIn(",4,4,0,1.000000,", all_filtered_text)

    def test_lazy_visual_all_filtered_summary_applies_min_match_gate(self) -> None:
        record = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        spec = LazyPairSpec(pair_index=1, split="train", reference=record, target=record)
        pair = SyntheticPair(
            view_a=torch.ones(1, 16, 16),
            view_b=torch.ones(1, 16, 16),
            warp_a_to_b=torch.zeros(16, 16, 2),
            valid_mask=torch.ones(16, 16, dtype=torch.bool),
        )
        low_count = LazyMatchVisual(
            label="low-count",
            spec=spec,
            pair=pair,
            valid_fraction=1.0,
            points_a=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0]]).numpy(),
            points_b=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0]]).numpy(),
            scores=torch.tensor([0.9, 0.8, 0.7, 0.6]).numpy(),
            errors=torch.zeros(4).numpy(),
            correct=torch.ones(4, dtype=torch.bool).numpy(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            visual_mod.write_visual_summary_artifacts(
                output_dir,
                selected=[low_count],
                filtered_selected=[low_count],
                all_results=[low_count],
                write_all_summary=True,
                filtered_geometry_filter="local",
                filtered_threshold_px=1.0,
                filtered_min_matches=8,
            )
            filtered_text = (output_dir / "filtered_summary.csv").read_text(encoding="utf-8")
            all_filtered_text = (output_dir / "all_filtered_summary.csv").read_text(encoding="utf-8")

        self.assertIn(",0,0,0,0.000000,", filtered_text)
        self.assertIn(",0,0,0,0.000000,", all_filtered_text)

    def test_lazy_visual_all_filtered_summary_uses_variant_min_match_gate(self) -> None:
        reference = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        extreme = RenderRecord(
            pose_id="pose_b",
            base_id="base_001",
            variant="extreme_03",
            split="train",
            tsai_path=Path("b.tsai"),
            image_path=Path("b.tif"),
            uint8_path=Path("b.png"),
            depth_path=Path("b_depth.tif"),
        )
        pair = SyntheticPair(
            view_a=torch.ones(1, 16, 16),
            view_b=torch.ones(1, 16, 16),
            warp_a_to_b=torch.zeros(16, 16, 2),
            valid_mask=torch.ones(16, 16, dtype=torch.bool),
        )

        def make_visual(label: str, target: RenderRecord) -> LazyMatchVisual:
            spec = LazyPairSpec(pair_index=1, split="train", reference=reference, target=target)
            return LazyMatchVisual(
                label=label,
                spec=spec,
                pair=pair,
                valid_fraction=1.0,
                points_a=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0]]).numpy(),
                points_b=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0]]).numpy(),
                scores=torch.tensor([0.9, 0.8, 0.7, 0.6]).numpy(),
                errors=torch.zeros(4).numpy(),
                correct=torch.ones(4, dtype=torch.bool).numpy(),
            )

        low_extreme = make_visual("low-extreme", extreme)
        low_default = make_visual("low-default", reference)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            visual_mod.write_visual_summary_artifacts(
                output_dir,
                selected=[],
                filtered_selected=[low_extreme, low_default],
                all_results=[low_extreme, low_default],
                write_all_summary=True,
                filtered_geometry_filter="none",
                filtered_threshold_px=1.0,
                filtered_min_matches=8,
                filtered_min_matches_by_variant={"extreme_03": 4},
            )
            with (output_dir / "filtered_summary.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = {row["label"]: row for row in csv.DictReader(handle)}

        self.assertEqual(rows["low-extreme"]["matches"], "4")
        self.assertEqual(rows["low-extreme"]["correct"], "4")
        self.assertEqual(rows["low-default"]["matches"], "0")
        self.assertEqual(rows["low-default"]["correct"], "0")

    def test_adaptive_geometry_rescue_uses_observable_match_gain_residual_and_score(self) -> None:
        reference = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        extreme = RenderRecord(
            pose_id="pose_b",
            base_id="base_001",
            variant="extreme_02",
            split="train",
            tsai_path=Path("b.tsai"),
            image_path=Path("b.tif"),
            uint8_path=Path("b.png"),
            depth_path=Path("b_depth.tif"),
        )
        mid = RenderRecord(
            pose_id="pose_c",
            base_id="base_001",
            variant="mid_01",
            split="train",
            tsai_path=Path("c.tsai"),
            image_path=Path("c.tif"),
            uint8_path=Path("c.png"),
            depth_path=Path("c_depth.tif"),
        )
        pair = SyntheticPair(
            view_a=torch.ones(1, 16, 16),
            view_b=torch.ones(1, 16, 16),
            warp_a_to_b=torch.zeros(16, 16, 2),
            valid_mask=torch.ones(16, 16, dtype=torch.bool),
        )

        def make_visual(target: RenderRecord, count: int, score: float) -> LazyMatchVisual:
            points = np.stack([np.arange(count, dtype=np.float32), np.arange(count, dtype=np.float32)], axis=1)
            spec = LazyPairSpec(pair_index=1, split="val", reference=reference, target=target)
            return LazyMatchVisual(
                label="candidate / all-filtered",
                spec=spec,
                pair=pair,
                valid_fraction=1.0,
                points_a=points,
                points_b=points.copy(),
                scores=np.full(count, score, dtype=np.float32),
                errors=np.zeros(count, dtype=np.float32),
                correct=np.zeros(count, dtype=np.bool_),
            )

        config = visual_mod.AdaptiveGeometryRescueConfig(
            enabled=True,
            target_variants=("extreme_02", "extreme_03"),
            rescue_threshold_px=10.0,
            min_match_gain=2,
            max_base_matches=4,
            max_homography_p90_px=1.0,
            max_homography_median_px=1.0,
            require_score_mean_not_lower=True,
        )

        rescued = visual_mod.select_adaptive_geometry_rescue(
            make_visual(extreme, 4, 0.6),
            make_visual(extreme, 7, 0.7),
            config=config,
        )
        low_score = visual_mod.select_adaptive_geometry_rescue(
            make_visual(extreme, 4, 0.6),
            make_visual(extreme, 7, 0.5),
            config=config,
        )
        wrong_variant = visual_mod.select_adaptive_geometry_rescue(
            make_visual(mid, 4, 0.6),
            make_visual(mid, 7, 0.7),
            config=config,
        )
        base_already_strong = visual_mod.select_adaptive_geometry_rescue(
            make_visual(extreme, 5, 0.6),
            make_visual(extreme, 8, 0.7),
            config=config,
        )

        self.assertEqual(rescued.matches, 7)
        self.assertEqual(low_score.matches, 4)
        self.assertEqual(wrong_variant.matches, 4)
        self.assertEqual(base_already_strong.matches, 5)

    def test_low_match_geometry_guard_only_keeps_clean_target_small_sets(self) -> None:
        reference = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        extreme = RenderRecord(
            pose_id="pose_b",
            base_id="base_001",
            variant="extreme_02",
            split="train",
            tsai_path=Path("b.tsai"),
            image_path=Path("b.tif"),
            uint8_path=Path("b.png"),
            depth_path=Path("b_depth.tif"),
        )
        mid = RenderRecord(
            pose_id="pose_c",
            base_id="base_001",
            variant="mid_01",
            split="train",
            tsai_path=Path("c.tsai"),
            image_path=Path("c.tif"),
            uint8_path=Path("c.png"),
            depth_path=Path("c_depth.tif"),
        )
        pair = SyntheticPair(
            view_a=torch.ones(1, 16, 16),
            view_b=torch.ones(1, 16, 16),
            warp_a_to_b=torch.zeros(16, 16, 2),
            valid_mask=torch.ones(16, 16, dtype=torch.bool),
        )

        def make_visual(target: RenderRecord, count: int, *, score: float, noisy: bool = False) -> LazyMatchVisual:
            points = np.stack(
                [
                    np.arange(count, dtype=np.float32) % 5,
                    np.arange(count, dtype=np.float32) // 5,
                ],
                axis=1,
            )
            points_b = points.copy()
            if noisy:
                points_b[::3] += np.array([5.0, -3.0], dtype=np.float32)
            spec = LazyPairSpec(pair_index=1, split="val", reference=reference, target=target)
            return LazyMatchVisual(
                label="candidate / all-filtered",
                spec=spec,
                pair=pair,
                valid_fraction=1.0,
                points_a=points,
                points_b=points_b,
                scores=np.full(count, score, dtype=np.float32),
                errors=np.zeros(count, dtype=np.float32),
                correct=np.zeros(count, dtype=np.bool_),
            )

        config = visual_mod.LowMatchGeometryGuardConfig(
            enabled=True,
            target_variants=("extreme_02", "extreme_03"),
            min_matches=12,
            max_matches=15,
            max_homography_p90_px=2.8,
            max_homography_median_px=1.5,
            min_score_mean=19.0,
        )

        clean = visual_mod.apply_min_match_gate(
            make_visual(extreme, 15, score=20.0),
            min_matches=16,
            low_match_geometry_guard_config=config,
        )
        low_score = visual_mod.apply_min_match_gate(
            make_visual(extreme, 15, score=18.0),
            min_matches=16,
            low_match_geometry_guard_config=config,
        )
        noisy = visual_mod.apply_min_match_gate(
            make_visual(extreme, 14, score=20.0, noisy=True),
            min_matches=16,
            low_match_geometry_guard_config=config,
        )
        wrong_variant = visual_mod.apply_min_match_gate(
            make_visual(mid, 15, score=20.0),
            min_matches=16,
            low_match_geometry_guard_config=config,
        )
        already_enough = visual_mod.apply_min_match_gate(
            make_visual(extreme, 16, score=18.0, noisy=True),
            min_matches=16,
            low_match_geometry_guard_config=config,
        )

        self.assertEqual(clean.matches, 15)
        self.assertIn("low-match-guard", clean.label)
        self.assertEqual(low_score.matches, 0)
        self.assertEqual(noisy.matches, 0)
        self.assertEqual(wrong_variant.matches, 0)
        self.assertEqual(already_enough.matches, 16)


if __name__ == "__main__":
    unittest.main()
