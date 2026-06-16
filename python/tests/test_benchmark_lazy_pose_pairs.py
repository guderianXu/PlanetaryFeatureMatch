import argparse
import csv
import math
import numpy as np
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

import benchmark_lazy_pose_pairs as lazy_bench
from benchmark_lazy_pose_pairs import (
    PhotometricAugmentConfig,
    StreamingCsvRows,
    apply_local_contrast_normalization,
    apply_photometric_augmentation,
    make_illumination_match_pair,
)
from patch_descriptor_training import SyntheticPair


class BenchmarkLazyPosePairsTest(unittest.TestCase):
    def make_pair(self) -> SyntheticPair:
        view_a = torch.linspace(0.05, 0.95, 16, dtype=torch.float32).view(1, 4, 4)
        view_b = torch.linspace(0.95, 0.05, 16, dtype=torch.float32).view(1, 4, 4)
        yy, xx = torch.meshgrid(torch.arange(4, dtype=torch.float32), torch.arange(4, dtype=torch.float32), indexing="ij")
        warp = torch.stack((xx, yy), dim=-1)
        valid = torch.ones(4, 4, dtype=torch.bool)
        return SyntheticPair(view_a=view_a, view_b=view_b, warp_a_to_b=warp, valid_mask=valid)

    def record(
        self,
        root: Path,
        raw_base_id: str,
        variant: str,
        *,
        dataset_id: str = "fov090",
        split: str = "train",
        lon_deg: float | None = None,
        lat_deg: float | None = None,
    ) -> lazy_bench.RenderRecord:
        item_dir = root / dataset_id / raw_base_id / variant
        item_dir.mkdir(parents=True, exist_ok=True)
        image_path = item_dir / "image.tif"
        depth_path = item_dir / "depth.tif"
        tsai_path = item_dir / "camera.tsai"
        image_path.write_bytes(b"x")
        depth_path.write_bytes(b"x")
        tsai_path.write_text("x", encoding="utf-8")
        return lazy_bench.RenderRecord(
            pose_id=f"{dataset_id}_{raw_base_id}_{variant}",
            base_id=f"{dataset_id}:{raw_base_id}",
            variant=variant,
            split=split,
            tsai_path=tsai_path,
            image_path=image_path,
            uint8_path=image_path,
            depth_path=depth_path,
            dataset_id=dataset_id,
            raw_base_id=raw_base_id,
            lon_deg=lon_deg,
            lat_deg=lat_deg,
        )

    def make_render_records(
        self,
        root: Path,
        *,
        dataset_id: str,
        base_ids: tuple[str, ...],
        variants: tuple[str, ...],
    ) -> list[lazy_bench.RenderRecord]:
        return [
            self.record(root, raw_base_id, variant, dataset_id=dataset_id)
            for raw_base_id in base_ids
            for variant in variants
        ]

    def test_photometric_augmentation_preserves_geometry_and_is_deterministic(self) -> None:
        pair = self.make_pair()
        config = PhotometricAugmentConfig(
            enabled=True,
            probability=1.0,
            brightness=0.20,
            contrast=0.35,
            gamma=0.45,
            shadow=0.40,
            noise=0.02,
        )

        first = apply_photometric_augmentation(pair, config, seed=123)
        second = apply_photometric_augmentation(pair, config, seed=123)

        self.assertTrue(torch.allclose(first.view_a, second.view_a))
        self.assertTrue(torch.allclose(first.view_b, second.view_b))
        self.assertFalse(torch.allclose(first.view_a, pair.view_a))
        self.assertFalse(torch.allclose(first.view_b, pair.view_b))
        self.assertTrue(torch.equal(first.valid_mask, pair.valid_mask))
        self.assertTrue(torch.allclose(first.warp_a_to_b, pair.warp_a_to_b))
        self.assertGreaterEqual(float(first.view_a.min()), 0.0)
        self.assertLessEqual(float(first.view_a.max()), 1.0)

    def test_disabled_photometric_augmentation_keeps_pair_unchanged(self) -> None:
        pair = self.make_pair()
        config = PhotometricAugmentConfig(enabled=False)

        augmented = apply_photometric_augmentation(pair, config, seed=456)

        self.assertTrue(torch.allclose(augmented.view_a, pair.view_a))
        self.assertTrue(torch.allclose(augmented.view_b, pair.view_b))
        self.assertTrue(torch.allclose(augmented.warp_a_to_b, pair.warp_a_to_b))
        self.assertTrue(torch.equal(augmented.valid_mask, pair.valid_mask))

    def test_local_contrast_normalization_preserves_geometry(self) -> None:
        pair = self.make_pair()

        normalized = apply_local_contrast_normalization(pair, strength=0.75, kernel_size=3)

        self.assertFalse(torch.allclose(normalized.view_a, pair.view_a))
        self.assertFalse(torch.allclose(normalized.view_b, pair.view_b))
        self.assertTrue(torch.equal(normalized.valid_mask, pair.valid_mask))
        self.assertTrue(torch.allclose(normalized.warp_a_to_b, pair.warp_a_to_b))
        self.assertGreaterEqual(float(normalized.view_a.min()), 0.0)
        self.assertLessEqual(float(normalized.view_a.max()), 1.0)

    def test_training_transforms_are_deterministic_and_preserve_geometry(self) -> None:
        pair = self.make_pair()
        config = PhotometricAugmentConfig(
            enabled=True,
            probability=1.0,
            brightness=0.12,
            contrast=0.25,
            gamma=0.30,
            shadow=0.20,
            noise=0.01,
        )

        first = lazy_bench.apply_training_transforms(
            pair,
            photometric_config=config,
            seed=789,
            input_local_contrast=True,
            local_contrast_strength=0.5,
            local_contrast_kernel=3,
        )
        second = lazy_bench.apply_training_transforms(
            pair,
            photometric_config=config,
            seed=789,
            input_local_contrast=True,
            local_contrast_strength=0.5,
            local_contrast_kernel=3,
        )

        self.assertTrue(torch.allclose(first.view_a, second.view_a))
        self.assertTrue(torch.allclose(first.view_b, second.view_b))
        self.assertFalse(torch.allclose(first.view_a, pair.view_a))
        self.assertFalse(torch.allclose(first.view_b, pair.view_b))
        self.assertTrue(torch.equal(first.valid_mask, pair.valid_mask))
        self.assertTrue(torch.allclose(first.warp_a_to_b, pair.warp_a_to_b))

    def test_parse_args_accepts_rejection_and_hard_negative_options(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--train-graph-matcher",
            "--graph-matcher-loss-weight",
            "0.4",
            "--graph-matcher-no-match-points",
            "64",
            "--graph-matcher-no-match-weight",
            "0.2",
            "--graph-matcher-assignment-weight",
            "0.35",
            "--graph-matcher-train-max-attention-layers",
            "2",
            "--graph-matcher-train-random-attention-layers",
            "--graph-matcher-train-max-attention-work-fraction",
            "0.5",
            "--graph-matcher-train-width-keep-ratio",
            "0.75",
            "--graph-matcher-deep-supervision-depths",
            "1,2,4",
            "--graph-matcher-deep-supervision-weight",
            "0.4",
            "--matcher-reliability-pair-bias",
            "off",
            "--matcher-reliability-dustbin-bias",
            "matchability",
            "--matcher-final-accept-score-mode",
            "add",
            "--matcher-accept-assignment-mode",
            "off",
            "--matcher-final-accept-score-alpha",
            "0.07",
            "--matcher-geometry-bias-scale",
            "0.25",
            "--matcher-geometry-bias-clamp",
            "1.5",
            "--matcher-attention-residual-gate-init",
            "0.05",
            "--matcher-attention-residual-gate-start-layer",
            "5",
            "--matcher-candidate-topk",
            "96",
            "--graph-matcher-online-false-no-match",
            "--graph-matcher-accept-weight",
            "0.2",
            "--graph-matcher-train-candidate-topk",
            "64",
            "--graph-matcher-prune-ranking-weight",
            "0.15",
            "--graph-matcher-stop-confidence-weight",
            "0.07",
            "--graph-matcher-dustbin-warmup-steps",
            "100",
            "--graph-matcher-dustbin-ramp-steps",
            "300",
            "--graph-matcher-positive-dustbin-margin-weight",
            "0.45",
            "--graph-matcher-positive-dustbin-margin",
            "0.2",
            "--graph-matcher-mined-false-match-loss-cap",
            "3.5",
            "--graph-matcher-mined-false-match-reference-margin",
            "0.5",
            "--graph-matcher-raw-false-match-weight",
            "0.04",
            "--graph-matcher-raw-false-match-topk",
            "3",
            "--graph-matcher-raw-false-match-min-similarity",
            "0.82",
            "--graph-matcher-raw-false-match-margin",
            "0.45",
            "--graph-matcher-raw-false-match-spatial-min-distance",
            "6.5",
            "--graph-matcher-ransac-consistency-weight",
            "0.07",
            "--graph-matcher-ransac-consistency-topk",
            "5",
            "--graph-matcher-ransac-consistency-residual-threshold-px",
            "2.5",
            "--graph-matcher-ransac-consistency-min-score",
            "0.03",
            "--graph-matcher-ransac-consistency-margin",
            "0.4",
            "--graph-matcher-depth-distillation-weight",
            "0.6",
            "--graph-matcher-depth-distillation-teacher-layers",
            "4",
            "--graph-matcher-depth-distillation-temperature",
            "1.5",
            "--graph-matcher-teacher-guard-state",
            "/tmp/best_teacher.pt",
            "--graph-matcher-teacher-guard-weight",
            "0.55",
            "--graph-matcher-teacher-guard-positive-margin-tolerance",
            "0.15",
            "--graph-matcher-teacher-guard-false-margin-tolerance",
            "0.25",
            "--graph-matcher-teacher-score-floor-weight",
            "0.45",
            "--graph-matcher-teacher-score-floor-tolerance",
            "0.35",
            "--graph-matcher-teacher-score-floor-min-score",
            "0.2",
            "--graph-matcher-teacher-match-count-floor-weight",
            "0.03",
            "--graph-matcher-teacher-match-count-floor-threshold",
            "18.0",
            "--graph-matcher-teacher-match-count-floor-margin",
            "0.5",
            "--graph-matcher-teacher-distillation-weight",
            "0.35",
            "--graph-matcher-teacher-distillation-temperature",
            "1.75",
            "--graph-matcher-positive-dustbin-guard-reject-threshold",
            "0.2",
            "--graph-matcher-positive-dustbin-guard-margin-threshold",
            "1.0",
            "--freeze-extractor-warmup-steps",
            "600",
            "--abstention-weight",
            "0.3",
            "--warp-hard-negative-weight",
            "0.2",
            "--false-match-csv",
            "false.csv",
            "--false-match-weight",
            "0.5",
            "--false-match-curriculum-max-probability",
            "0.75",
            "--false-match-mine-every",
            "4",
            "--gpu-snapshot-every",
            "25",
            "--no-gpu-monitor",
            "--gpu-sample-interval-s",
            "0.5",
            "--illumination-consistency-weight",
            "0.08",
            "--illumination-consistency-probability",
            "0.6",
            "--illumination-consistency-points",
            "96",
            "--illumination-consistency-gamma",
            "0.8",
            "--illumination-consistency-shadow",
            "0.7",
            "--illumination-match-weight",
            "0.35",
            "--illumination-match-probability",
            "0.7",
            "--hard-variant",
            "extreme",
            "--hard-valid-fraction-max",
            "0.55",
            "--hard-curriculum-max-probability",
            "0.8",
            "--input-local-contrast",
            "--input-local-contrast-strength",
            "0.6",
            "--train-reliability-head",
            "--matchability-weight",
            "0.11",
            "--descriptor-uncertainty-weight",
            "0.12",
            "--no-match-prior-weight",
            "0.13",
            "--reliability-negative-points",
            "48",
            "--reliability-negative-min-distance",
            "6.5",
            "--rotation-descriptor-consistency-weight",
            "0.21",
            "--orientation-consistency-weight",
            "0.22",
            "--scale-consistency-weight",
            "0.23",
            "--affine-consistency-weight",
            "0.24",
            "--rotation-consistency-degrees",
            "90,270",
            "--amp",
            "--amp-dtype",
            "bfloat16",
            "--activation-checkpointing",
            "--visual-filtered-min-matches",
            "16",
            "--visual-post-filter-profile",
            "fov76_geo5_geo10_extreme_rescue_lowmatch_guard",
            "--visual-eval-every-steps",
            "100",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertTrue(args.train_graph_matcher)
        self.assertEqual(args.graph_matcher_no_match_points, 64)
        self.assertAlmostEqual(args.graph_matcher_assignment_weight, 0.35)
        self.assertEqual(args.graph_matcher_train_max_attention_layers, 2)
        self.assertTrue(args.graph_matcher_train_random_attention_layers)
        self.assertAlmostEqual(args.graph_matcher_train_max_attention_work_fraction, 0.5)
        self.assertAlmostEqual(args.graph_matcher_train_width_keep_ratio, 0.75)
        self.assertEqual(args.graph_matcher_deep_supervision_depths, [1, 2, 4])
        self.assertAlmostEqual(args.graph_matcher_deep_supervision_weight, 0.4)
        self.assertEqual(args.matcher_reliability_pair_bias, "off")
        self.assertEqual(args.matcher_reliability_dustbin_bias, "matchability")
        self.assertEqual(args.matcher_final_accept_score_mode, "add")
        self.assertEqual(args.matcher_accept_assignment_mode, "off")
        self.assertAlmostEqual(args.matcher_final_accept_score_alpha, 0.07)
        self.assertAlmostEqual(args.matcher_geometry_bias_scale, 0.25)
        self.assertAlmostEqual(args.matcher_geometry_bias_clamp, 1.5)
        self.assertAlmostEqual(args.matcher_attention_residual_gate_init, 0.05)
        self.assertEqual(args.matcher_attention_residual_gate_start_layer, 5)
        self.assertEqual(args.matcher_candidate_topk, 96)
        self.assertTrue(args.graph_matcher_online_false_no_match)
        self.assertAlmostEqual(args.graph_matcher_accept_weight, 0.2)
        self.assertEqual(args.graph_matcher_train_candidate_topk, 64)
        self.assertAlmostEqual(args.graph_matcher_prune_ranking_weight, 0.15)
        self.assertAlmostEqual(args.graph_matcher_stop_confidence_weight, 0.07)
        self.assertEqual(args.graph_matcher_dustbin_warmup_steps, 100)
        self.assertEqual(args.graph_matcher_dustbin_ramp_steps, 300)
        self.assertAlmostEqual(args.graph_matcher_positive_dustbin_margin_weight, 0.45)
        self.assertAlmostEqual(args.graph_matcher_positive_dustbin_margin, 0.2)
        self.assertAlmostEqual(args.graph_matcher_mined_false_match_loss_cap, 3.5)
        self.assertAlmostEqual(args.graph_matcher_mined_false_match_reference_margin, 0.5)
        self.assertAlmostEqual(args.graph_matcher_raw_false_match_weight, 0.04)
        self.assertEqual(args.graph_matcher_raw_false_match_topk, 3)
        self.assertAlmostEqual(args.graph_matcher_raw_false_match_min_similarity, 0.82)
        self.assertAlmostEqual(args.graph_matcher_raw_false_match_margin, 0.45)
        self.assertAlmostEqual(args.graph_matcher_raw_false_match_spatial_min_distance, 6.5)
        self.assertAlmostEqual(args.graph_matcher_ransac_consistency_weight, 0.07)
        self.assertEqual(args.graph_matcher_ransac_consistency_topk, 5)
        self.assertAlmostEqual(args.graph_matcher_ransac_consistency_residual_threshold_px, 2.5)
        self.assertAlmostEqual(args.graph_matcher_ransac_consistency_min_score, 0.03)
        self.assertAlmostEqual(args.graph_matcher_ransac_consistency_margin, 0.4)
        self.assertAlmostEqual(args.graph_matcher_depth_distillation_weight, 0.6)
        self.assertEqual(args.graph_matcher_depth_distillation_teacher_layers, 4)
        self.assertAlmostEqual(args.graph_matcher_depth_distillation_temperature, 1.5)
        self.assertEqual(args.graph_matcher_teacher_guard_state, Path("/tmp/best_teacher.pt"))
        self.assertAlmostEqual(args.graph_matcher_teacher_guard_weight, 0.55)
        self.assertAlmostEqual(args.graph_matcher_teacher_guard_positive_margin_tolerance, 0.15)
        self.assertAlmostEqual(args.graph_matcher_teacher_guard_false_margin_tolerance, 0.25)
        self.assertAlmostEqual(args.graph_matcher_teacher_score_floor_weight, 0.45)
        self.assertAlmostEqual(args.graph_matcher_teacher_score_floor_tolerance, 0.35)
        self.assertAlmostEqual(args.graph_matcher_teacher_score_floor_min_score, 0.2)
        self.assertAlmostEqual(args.graph_matcher_teacher_match_count_floor_weight, 0.03)
        self.assertAlmostEqual(args.graph_matcher_teacher_match_count_floor_threshold, 18.0)
        self.assertAlmostEqual(args.graph_matcher_teacher_match_count_floor_margin, 0.5)
        self.assertAlmostEqual(args.graph_matcher_teacher_distillation_weight, 0.35)
        self.assertAlmostEqual(args.graph_matcher_teacher_distillation_temperature, 1.75)
        self.assertAlmostEqual(args.graph_matcher_positive_dustbin_guard_reject_threshold, 0.2)
        self.assertAlmostEqual(args.graph_matcher_positive_dustbin_guard_margin_threshold, 1.0)
        self.assertEqual(args.freeze_extractor_warmup_steps, 600)
        self.assertEqual(args.false_match_csv, [Path("false.csv")])
        self.assertEqual(args.false_match_mine_every, 4)
        self.assertEqual(args.gpu_snapshot_every, 25)
        self.assertFalse(args.gpu_monitor)
        self.assertAlmostEqual(args.gpu_sample_interval_s, 0.5)
        self.assertAlmostEqual(args.illumination_consistency_weight, 0.08)
        self.assertAlmostEqual(args.illumination_consistency_probability, 0.6)
        self.assertEqual(args.illumination_consistency_points, 96)
        self.assertAlmostEqual(args.illumination_consistency_gamma, 0.8)
        self.assertAlmostEqual(args.illumination_consistency_shadow, 0.7)
        self.assertAlmostEqual(args.illumination_match_weight, 0.35)
        self.assertAlmostEqual(args.illumination_match_probability, 0.7)
        self.assertEqual(args.hard_variant, ["extreme"])
        self.assertTrue(args.input_local_contrast)
        self.assertAlmostEqual(args.input_local_contrast_strength, 0.6)
        self.assertTrue(args.train_reliability_head)
        self.assertAlmostEqual(args.matchability_weight, 0.11)
        self.assertAlmostEqual(args.descriptor_uncertainty_weight, 0.12)
        self.assertAlmostEqual(args.no_match_prior_weight, 0.13)
        self.assertEqual(args.reliability_negative_points, 48)
        self.assertAlmostEqual(args.reliability_negative_min_distance, 6.5)
        self.assertAlmostEqual(args.rotation_descriptor_consistency_weight, 0.21)
        self.assertAlmostEqual(args.orientation_consistency_weight, 0.22)
        self.assertAlmostEqual(args.scale_consistency_weight, 0.23)
        self.assertAlmostEqual(args.affine_consistency_weight, 0.24)
        self.assertEqual(args.rotation_consistency_degrees, [90, 270])
        self.assertTrue(args.amp)
        self.assertEqual(args.amp_dtype, "bfloat16")
        self.assertTrue(args.activation_checkpointing)
        self.assertEqual(args.visual_filtered_min_matches, 16)
        self.assertEqual(args.visual_post_filter_profile, "fov76_geo5_geo10_extreme_rescue_lowmatch_guard")
        self.assertEqual(args.visual_eval_every_steps, 100)

    def test_parse_args_enable_rejection_training_expands_safe_defaults(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--enable-rejection-training",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertTrue(args.enable_rejection_training)
        self.assertTrue(args.train_graph_matcher)
        self.assertTrue(args.inline_false_match_mining)
        self.assertTrue(args.graph_matcher_online_false_no_match)
        self.assertAlmostEqual(args.graph_matcher_loss_weight, 0.75)
        self.assertEqual(args.graph_matcher_no_match_points, 128)
        self.assertAlmostEqual(args.graph_matcher_no_match_weight, 0.45)
        self.assertAlmostEqual(args.graph_matcher_assignment_weight, 0.35)
        self.assertAlmostEqual(args.graph_matcher_accept_weight, 0.30)
        self.assertAlmostEqual(args.graph_matcher_prune_ranking_weight, 0.10)
        self.assertAlmostEqual(args.graph_matcher_stop_confidence_weight, 0.05)
        self.assertAlmostEqual(args.graph_matcher_hard_negative_dustbin_weight, 0.075)
        self.assertEqual(args.graph_matcher_hard_negative_dustbin_topk, 16)
        self.assertAlmostEqual(args.graph_matcher_hard_negative_dustbin_margin, 0.35)
        self.assertEqual(args.graph_matcher_semi_dense_no_match_points, 128)
        self.assertAlmostEqual(args.false_match_weight, 0.15)
        self.assertEqual(args.false_match_max_points, 192)
        self.assertAlmostEqual(args.keypoint_weight, 0.05)
        self.assertAlmostEqual(args.keypoint_negative_weight, 0.02)
        self.assertEqual(args.reliability_negative_points, 128)
        self.assertAlmostEqual(args.matchability_weight, 0.08)
        self.assertAlmostEqual(args.descriptor_uncertainty_weight, 0.05)
        self.assertAlmostEqual(args.no_match_prior_weight, 0.08)
        self.assertAlmostEqual(args.rotation_descriptor_consistency_weight, 0.03)
        self.assertEqual(args.visual_matcher_mode, "graph_matcher")
        self.assertEqual(args.visual_keypoint_score_mode, "learned")

    def test_parse_args_stable_graph_preset_uses_local10_visual_safety(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--stable-graph-matcher-training",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertEqual(args.visual_matcher_mode, "graph_matcher")
        self.assertEqual(args.visual_geometry_filter, "local")
        self.assertAlmostEqual(args.visual_geometry_threshold_px, 10.0)
        self.assertEqual(args.false_match_mine_matcher_mode, "graph_matcher")
        self.assertEqual(args.false_match_mine_geometry_filter, "local")
        self.assertAlmostEqual(args.false_match_mine_geometry_threshold_px, 10.0)
        self.assertEqual(args.false_match_mine_source, "geometry_rejected_truth_wrong")

    def test_parse_args_accepts_graph_geometry_false_mining_options(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--false-match-mine-matcher-mode",
            "graph_matcher",
            "--false-match-mine-geometry-filter",
            "magsac",
            "--false-match-mine-geometry-threshold-px",
            "4.0",
            "--false-match-mine-source",
            "truth_and_geometry_kept",
            "--false-match-mine-target-variant",
            "extreme_02",
            "--false-match-mine-target-variant",
            "extreme_03",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertEqual(args.false_match_mine_matcher_mode, "graph_matcher")
        self.assertEqual(args.false_match_mine_geometry_filter, "magsac")
        self.assertAlmostEqual(args.false_match_mine_geometry_threshold_px, 4.0)
        self.assertEqual(args.false_match_mine_source, "truth_and_geometry_kept")
        self.assertEqual(args.false_match_mine_target_variant, ["extreme_02", "extreme_03"])

    def test_parse_args_accepts_graph_calibration_only_training(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--train-graph-matcher",
            "--train-graph-calibration-only",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertTrue(args.train_graph_matcher)
        self.assertTrue(args.train_graph_calibration_only)

    def test_mine_false_matches_can_use_graph_local_geometry_rejected_wrong_edges(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.training = True
                self.config = SimpleNamespace(graph_keypoint_meta_dim=2)

            def eval(self) -> None:
                self.training = False

            def train(self, mode: bool = True) -> None:
                self.training = mode

        yy, xx = torch.meshgrid(
            torch.arange(16, dtype=torch.float32),
            torch.arange(16, dtype=torch.float32),
            indexing="ij",
        )
        pair = SyntheticPair(
            view_a=torch.ones(1, 16, 16),
            view_b=torch.ones(1, 16, 16),
            warp_a_to_b=torch.stack([xx, yy], dim=-1),
            valid_mask=torch.ones(16, 16, dtype=torch.bool),
        )
        keypoints_a = torch.tensor(
            [[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0], [14.0, 2.0]],
            dtype=torch.float32,
        )
        keypoints_b = torch.tensor(
            [[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0], [0.0, 15.0]],
            dtype=torch.float32,
        )
        descriptors = torch.ones(1, 1, 16, 16)
        matches = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.long)
        scores = torch.tensor([0.4, 0.5, 0.6, 0.7, 0.95], dtype=torch.float32)

        with (
            mock.patch.object(
                lazy_bench.match_eval,
                "feature_maps_and_keypoint_scores_for_pair",
                return_value=(descriptors, descriptors, None, None, object(), object()),
            ),
            mock.patch.object(
                lazy_bench.match_eval,
                "select_descriptor_keypoints",
                side_effect=[(keypoints_a, torch.arange(5)), (keypoints_b, torch.arange(5))],
            ),
            mock.patch.object(lazy_bench.match_eval, "gather_descriptor_rows", return_value=torch.ones(5, 1)),
            mock.patch.object(lazy_bench.match_eval, "graph_metadata_from_raw_features", return_value=torch.zeros(5, 2)),
            mock.patch.object(lazy_bench.match_eval, "graph_matcher_matches", return_value=(matches, scores)) as graph_matches,
            mock.patch.object(
                lazy_bench.match_eval,
                "mutual_nearest_matches",
                side_effect=AssertionError("graph mining should not use raw mutual matches"),
            ),
        ):
            labels, rows = lazy_bench.mine_false_matches_for_lazy_pair(
                FakeModel(),
                pair,
                Path("lazy_pair_refs/step_000001_pair_00.pt"),
                device=torch.device("cpu"),
                descriptor_mode="learned",
                texture_blend_weight=0.0,
                keypoint_score_mode="learned",
                max_keypoints=5,
                max_matches=0,
                min_intensity=0.0,
                min_score=-1.0,
                min_margin=0.0,
                threshold_px=5.0,
                matcher_mode="graph_matcher",
                geometry_filter="local",
                geometry_threshold_px=10.0,
                false_source="geometry_rejected_truth_wrong",
            )

        graph_matches.assert_called_once()
        key = "lazy_pair_refs/step_000001_pair_00.pt"
        self.assertEqual(set(labels), {key})
        self.assertTrue(torch.allclose(labels[key].points_a_xy, torch.tensor([[14.0, 2.0]])))
        self.assertTrue(torch.allclose(labels[key].points_b_xy, torch.tensor([[0.0, 15.0]])))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mine_source"], "geometry_rejected_truth_wrong")
        self.assertEqual(rows[0]["geometry_rejected"], "1")

    def test_mine_false_matches_can_target_truth_wrong_edges_kept_by_geometry(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.training = True
                self.config = SimpleNamespace(graph_keypoint_meta_dim=2)

            def eval(self) -> None:
                self.training = False

            def train(self, mode: bool = True) -> None:
                self.training = mode

        yy, xx = torch.meshgrid(
            torch.arange(16, dtype=torch.float32),
            torch.arange(16, dtype=torch.float32),
            indexing="ij",
        )
        pair = SyntheticPair(
            view_a=torch.ones(1, 16, 16),
            view_b=torch.ones(1, 16, 16),
            warp_a_to_b=torch.stack([xx, yy], dim=-1),
            valid_mask=torch.ones(16, 16, dtype=torch.bool),
        )
        keypoints_a = torch.tensor([[1.0, 1.0], [6.0, 6.0], [12.0, 2.0]], dtype=torch.float32)
        keypoints_b = torch.tensor([[1.0, 1.0], [8.0, 8.0], [0.0, 15.0]], dtype=torch.float32)
        descriptors = torch.ones(1, 1, 16, 16)
        matches = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
        scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        kept_after_geometry = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)

        with (
            mock.patch.object(
                lazy_bench.match_eval,
                "feature_maps_and_keypoint_scores_for_pair",
                return_value=(descriptors, descriptors, None, None, object(), object()),
            ),
            mock.patch.object(
                lazy_bench.match_eval,
                "select_descriptor_keypoints",
                side_effect=[(keypoints_a, torch.arange(3)), (keypoints_b, torch.arange(3))],
            ),
            mock.patch.object(lazy_bench.match_eval, "gather_descriptor_rows", return_value=torch.ones(3, 1)),
            mock.patch.object(lazy_bench.match_eval, "graph_metadata_from_raw_features", return_value=torch.zeros(3, 2)),
            mock.patch.object(lazy_bench.match_eval, "graph_matcher_matches", return_value=(matches, scores)),
            mock.patch.object(
                lazy_bench.match_eval,
                "apply_geometry_filter_to_matches",
                return_value=(kept_after_geometry, scores[:2]),
            ),
        ):
            labels, rows = lazy_bench.mine_false_matches_for_lazy_pair(
                FakeModel(),
                pair,
                Path("lazy_pair_refs/step_000002_pair_00.pt"),
                device=torch.device("cpu"),
                descriptor_mode="learned",
                texture_blend_weight=0.0,
                keypoint_score_mode="learned",
                max_keypoints=3,
                max_matches=0,
                min_intensity=0.0,
                min_score=-1.0,
                min_margin=0.0,
                threshold_px=1.0,
                matcher_mode="graph_matcher",
                geometry_filter="magsac",
                geometry_threshold_px=10.0,
                false_source="truth_and_geometry_kept",
            )

        key = "lazy_pair_refs/step_000002_pair_00.pt"
        self.assertEqual(set(labels), {key})
        self.assertTrue(torch.allclose(labels[key].points_a_xy, torch.tensor([[6.0, 6.0]])))
        self.assertTrue(torch.allclose(labels[key].points_b_xy, torch.tensor([[8.0, 8.0]])))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mine_source"], "truth_and_geometry_kept")
        self.assertEqual(rows[0]["geometry_rejected"], "0")

    def test_mine_false_matches_valid_truth_ignores_invalid_low_error_edges(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.training = True
                self.config = SimpleNamespace(graph_keypoint_meta_dim=2)

            def eval(self) -> None:
                self.training = False

            def train(self, mode: bool = True) -> None:
                self.training = mode

        yy, xx = torch.meshgrid(
            torch.arange(16, dtype=torch.float32),
            torch.arange(16, dtype=torch.float32),
            indexing="ij",
        )
        valid_mask = torch.ones(16, 16, dtype=torch.bool)
        valid_mask[6, 6] = False
        pair = SyntheticPair(
            view_a=torch.ones(1, 16, 16),
            view_b=torch.ones(1, 16, 16),
            warp_a_to_b=torch.stack([xx, yy], dim=-1),
            valid_mask=valid_mask,
        )
        keypoints_a = torch.tensor([[1.0, 1.0], [6.0, 6.0], [12.0, 12.0]], dtype=torch.float32)
        keypoints_b = torch.tensor([[1.0, 1.0], [6.0, 6.0], [0.0, 15.0]], dtype=torch.float32)
        descriptors = torch.ones(1, 1, 16, 16)
        matches = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
        scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)

        with (
            mock.patch.object(
                lazy_bench.match_eval,
                "feature_maps_and_keypoint_scores_for_pair",
                return_value=(descriptors, descriptors, None, None, object(), object()),
            ),
            mock.patch.object(
                lazy_bench.match_eval,
                "select_descriptor_keypoints",
                side_effect=[(keypoints_a, torch.arange(3)), (keypoints_b, torch.arange(3))],
            ),
            mock.patch.object(lazy_bench.match_eval, "gather_descriptor_rows", return_value=torch.ones(3, 1)),
            mock.patch.object(lazy_bench.match_eval, "graph_metadata_from_raw_features", return_value=torch.zeros(3, 2)),
            mock.patch.object(lazy_bench.match_eval, "graph_matcher_matches", return_value=(matches, scores)),
        ):
            labels, rows = lazy_bench.mine_false_matches_for_lazy_pair(
                FakeModel(),
                pair,
                Path("lazy_pair_refs/step_000003_pair_00.pt"),
                device=torch.device("cpu"),
                descriptor_mode="learned",
                texture_blend_weight=0.0,
                keypoint_score_mode="learned",
                max_keypoints=3,
                max_matches=0,
                min_intensity=0.0,
                min_score=-1.0,
                min_margin=0.0,
                threshold_px=5.0,
                matcher_mode="graph_matcher",
                geometry_filter="none",
                false_source="valid_truth",
            )

        key = "lazy_pair_refs/step_000003_pair_00.pt"
        self.assertEqual(set(labels), {key})
        self.assertTrue(torch.allclose(labels[key].points_a_xy, torch.tensor([[12.0, 12.0]])))
        self.assertTrue(torch.allclose(labels[key].points_b_xy, torch.tensor([[0.0, 15.0]])))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mine_source"], "valid_truth")

    def test_false_match_mining_graph_kwargs_use_defaults_without_eval_only_args(self) -> None:
        args = SimpleNamespace(
            visual_graph_width_prune_min_score=-1.0,
            visual_graph_early_stop_min_confidence=-1.0,
            visual_graph_max_attention_layers=8,
            visual_graph_max_attention_work_fraction=1.0,
            visual_graph_width_prune_keep_ratio=1.0,
        )

        kwargs = lazy_bench.false_match_mining_graph_kwargs(args)

        self.assertEqual(kwargs["graph_dustbin_delta"], 0.0)
        self.assertEqual(kwargs["graph_acceptance_margin"], 0.0)
        self.assertEqual(kwargs["graph_min_raw_score"], -1.0)
        self.assertEqual(kwargs["graph_min_raw_margin"], 0.0)
        self.assertEqual(kwargs["graph_min_accept_probability"], -1.0)
        self.assertEqual(kwargs["graph_width_prune_min_score"], -1.0)
        self.assertEqual(kwargs["graph_max_attention_layers"], 8)

    def test_false_match_mining_can_feed_graph_final_false_loss_without_descriptor_false_weight(self) -> None:
        args = SimpleNamespace(
            false_match_weight=0.0,
            train_graph_matcher=True,
            graph_matcher_loss_weight=0.3,
            graph_matcher_final_false_match_weight=0.002,
        )

        self.assertTrue(lazy_bench.false_match_mining_has_training_consumer(args))

    def test_parse_args_graph_only_false_mining_gets_active_default_probability(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--mine-false-matches",
            "--train-graph-matcher",
            "--graph-matcher-loss-weight",
            "0.3",
            "--graph-matcher-mined-false-match-weight",
            "0.001",
            "--false-match-weight",
            "0.0",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertAlmostEqual(args.false_match_curriculum_max_probability, 1.0)

    def test_false_match_mining_is_disabled_without_any_training_consumer(self) -> None:
        args = SimpleNamespace(
            false_match_weight=0.0,
            train_graph_matcher=True,
            graph_matcher_loss_weight=0.3,
            graph_matcher_final_false_match_weight=0.0,
        )

        self.assertFalse(lazy_bench.false_match_mining_has_training_consumer(args))

    def test_false_match_mining_target_variant_filter_allows_only_requested_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.record(root, "pose001", "nadir", dataset_id="fov076")
            target_extreme_02 = self.record(root, "pose001", "extreme_02", dataset_id="fov076")
            target_extreme_03 = self.record(root, "pose001", "extreme_03", dataset_id="fov076")
            spec_extreme_02 = lazy_bench.LazyPairSpec(
                pair_index=0,
                split="train",
                reference=reference,
                target=target_extreme_02,
                pair_type=lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW,
            )
            spec_extreme_03 = lazy_bench.LazyPairSpec(
                pair_index=1,
                split="train",
                reference=reference,
                target=target_extreme_03,
                pair_type=lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW,
            )

            self.assertTrue(
                lazy_bench.false_match_mining_target_variant_allowed(spec_extreme_02, ["extreme_02"])
            )
            self.assertFalse(
                lazy_bench.false_match_mining_target_variant_allowed(spec_extreme_03, ["extreme_02"])
            )
            self.assertTrue(lazy_bench.false_match_mining_target_variant_allowed(spec_extreme_03, []))

    def test_annotate_false_match_rows_includes_pair_variant_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.record(root, "pose001", "nadir", dataset_id="fov076")
            target = self.record(root, "pose001", "extreme_02", dataset_id="fov076")
            spec = lazy_bench.LazyPairSpec(
                pair_index=0,
                split="train",
                reference=reference,
                target=target,
                pair_type=lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW,
            )

            rows = [
                {
                    "pair_pt": "lazy_pair_refs/step_000001_pair_00.pt",
                    "ax": "1.0",
                    "ay": "2.0",
                    "bx": "3.0",
                    "by": "4.0",
                }
            ]

            annotated = lazy_bench.annotate_false_match_rows_with_pair_metadata(rows, spec)

        self.assertEqual(annotated[0]["reference_base_id"], reference.base_id)
        self.assertEqual(annotated[0]["reference_variant"], "nadir")
        self.assertEqual(annotated[0]["target_base_id"], target.base_id)
        self.assertEqual(annotated[0]["target_variant"], "extreme_02")
        self.assertEqual(annotated[0]["pair_type"], lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW)
        self.assertEqual(annotated[0]["ax"], "1.0")

    def test_parse_args_accepts_training_stability_controls(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--stability-window",
            "300",
            "--stability-min-steps",
            "1500",
            "--stability-max-nan-in-window",
            "10",
            "--stability-min-top1-mean",
            "0.35",
            "--stability-max-loss-multiplier",
            "2.5",
            "--stability-min-match-score",
            "-0.25",
            "--stability-max-dustbin-rejection-ratio",
            "0.85",
            "--stability-min-num-filtered-matches",
            "16",
            "--stability-auto-recovery",
            "--stability-max-recoveries",
            "2",
            "--stability-lr-reduction-factor",
            "0.25",
            "--save-best-checkpoints",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertEqual(args.stability_window, 300)
        self.assertEqual(args.stability_min_steps, 1500)
        self.assertEqual(args.stability_max_nan_in_window, 10)
        self.assertAlmostEqual(args.stability_min_top1_mean, 0.35)
        self.assertAlmostEqual(args.stability_max_loss_multiplier, 2.5)
        self.assertAlmostEqual(args.stability_min_match_score, -0.25)
        self.assertAlmostEqual(args.stability_max_dustbin_rejection_ratio, 0.85)
        self.assertEqual(args.stability_min_num_filtered_matches, 16)
        self.assertTrue(args.stability_auto_recovery)
        self.assertEqual(args.stability_max_recoveries, 2)
        self.assertAlmostEqual(args.stability_lr_reduction_factor, 0.25)
        self.assertTrue(args.save_best_checkpoints)

    def test_stability_metric_fields_include_rolling_diagnostics(self) -> None:
        self.assertIn("nan_count", lazy_bench.STABILITY_METRIC_FIELDS)
        self.assertIn("recent_loss_mean", lazy_bench.STABILITY_METRIC_FIELDS)
        self.assertIn("recent_top1_mean", lazy_bench.STABILITY_METRIC_FIELDS)

    def test_parse_args_skips_nonfinite_steps_by_default(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertTrue(args.skip_nonfinite_steps)

        with mock.patch.object(sys, "argv", [*argv, "--no-skip-nonfinite-steps"]):
            args = lazy_bench.parse_args()

        self.assertFalse(args.skip_nonfinite_steps)

    def test_graph_matcher_dustbin_diagnostic_fields_are_declared_for_training_logs(self) -> None:
        self.assertIn("true_match_rejected_by_dustbin_ratio", lazy_bench.GRAPH_MATCHER_DIAGNOSTIC_METRIC_FIELDS)
        self.assertIn("positive_vs_dustbin_margin_mean", lazy_bench.GRAPH_MATCHER_DIAGNOSTIC_METRIC_FIELDS)
        self.assertIn("dustbin_logit_mean", lazy_bench.GRAPH_MATCHER_DIAGNOSTIC_METRIC_FIELDS)
        self.assertIn("false_match_accepted_ratio", lazy_bench.GRAPH_MATCHER_DIAGNOSTIC_METRIC_FIELDS)
        self.assertIn("accept_logit_mean", lazy_bench.GRAPH_MATCHER_DIAGNOSTIC_METRIC_FIELDS)
        self.assertIn("dustbin_prob_for_true_match_mean", lazy_bench.GRAPH_MATCHER_DIAGNOSTIC_METRIC_FIELDS)

    def test_parse_args_accepts_graph_architecture_overrides(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--graph-hidden-dim",
            "256",
            "--graph-attention-layers",
            "2",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertEqual(args.graph_hidden_dim, 256)
        self.assertEqual(args.graph_attention_layers, 2)

    def test_parse_args_accepts_extractor_geometry_controls(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--descriptor-geometry-mode",
            "orientation_scale",
            "--descriptor-geometry-blend-weight",
            "0.35",
            "--descriptor-scale-log-clamp-min",
            "-0.7",
            "--descriptor-scale-log-clamp-max",
            "0.7",
            "--descriptor-geometry-safety-schedule",
            "phase4",
            "--quality-score-mode",
            "soft",
            "--affine-regularization-weight",
            "0.03",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertEqual(args.descriptor_geometry_mode, "orientation_scale")
        self.assertAlmostEqual(args.descriptor_geometry_blend_weight, 0.35)
        self.assertAlmostEqual(args.descriptor_scale_log_clamp_min, -0.7)
        self.assertAlmostEqual(args.descriptor_scale_log_clamp_max, 0.7)
        self.assertEqual(args.descriptor_geometry_safety_schedule, "phase4")
        self.assertEqual(args.quality_score_mode, "soft")
        self.assertAlmostEqual(args.affine_regularization_weight, 0.03)

    def test_stable_graph_matcher_training_preset_uses_two_layer_low_rejection(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--enable-rejection-training",
            "--stable-graph-matcher-training",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertTrue(args.stable_graph_matcher_training)
        self.assertTrue(args.train_graph_matcher)
        self.assertEqual(args.graph_hidden_dim, 256)
        self.assertEqual(args.graph_attention_layers, 2)
        self.assertEqual(args.graph_matcher_train_max_attention_layers, 2)
        self.assertAlmostEqual(args.graph_matcher_train_width_keep_ratio, 1.0)
        self.assertAlmostEqual(args.graph_matcher_no_match_weight, 0.05)
        self.assertAlmostEqual(args.graph_matcher_hard_negative_dustbin_weight, 0.02)
        self.assertAlmostEqual(args.graph_matcher_stop_confidence_weight, 0.0)
        self.assertAlmostEqual(args.no_match_prior_weight, 0.0)
        self.assertEqual(args.reliability_negative_points, 32)

    def test_stable_graph_matcher_training_preset_keeps_explicit_overrides(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "train",
            "--stable-graph-matcher-training",
            "--graph-hidden-dim",
            "384",
            "--graph-matcher-no-match-weight",
            "0.12",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertEqual(args.graph_hidden_dim, 384)
        self.assertAlmostEqual(args.graph_matcher_no_match_weight, 0.12)

    def test_parse_args_accepts_multiple_render_and_uint8_manifests(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "fov090/render_manifest.csv",
            "fov110/render_manifest.csv",
            "--uint8-manifest",
            "fov090/images_u8/uint8_manifest.csv",
            "fov110/images_u8/uint8_manifest.csv",
            "--output-dir",
            "run",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertEqual(
            args.render_manifest,
            [Path("fov090/render_manifest.csv"), Path("fov110/render_manifest.csv")],
        )
        self.assertEqual(
            args.uint8_manifest,
            [Path("fov090/images_u8/uint8_manifest.csv"), Path("fov110/images_u8/uint8_manifest.csv")],
        )

    def test_parse_args_accepts_cross_domain_pair_options(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "fov090/render_manifest.csv",
            "fov110/render_manifest.csv",
            "--output-dir",
            "run",
            "--pair-mode",
            "mixed",
            "--cross-camera-offsets",
            "1,3,7",
            "--cross-fov-offsets",
            "0,2",
            "--cross-pair-variant",
            "nadir",
            "--cross-pair-variant",
            "extreme_01",
            "--pair-type-weights",
            "same_position_view=0.2,cross_camera=0.5,cross_fov=0.3",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertEqual(args.pair_mode, "mixed")
        self.assertEqual(args.cross_camera_offsets, [1, 3, 7])
        self.assertEqual(args.cross_fov_offsets, [0, 2])
        self.assertEqual(args.cross_pair_variant, ["nadir", "extreme_01"])
        self.assertEqual(
            args.pair_type_weights,
            {"same_position_view": 0.2, "cross_camera": 0.5, "cross_fov": 0.3},
        )

    def test_parse_args_accepts_spatial_index_pair_mode(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--pair-mode",
            "spatial-index",
            "--spatial-index-footprint-samples",
            "5",
            "--spatial-index-margin-m",
            "2000",
            "--spatial-index-height-km",
            "100,250",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertEqual(args.pair_mode, "spatial-index")
        self.assertEqual(args.spatial_index_footprint_samples, 5)
        self.assertEqual(args.spatial_index_margin_m, 2000.0)
        self.assertEqual(args.spatial_index_height_km, [100, 250])

    def test_parse_args_accepts_overlap_list_mode_and_pair_spec_manifest(self) -> None:
        argv = [
            "benchmark_lazy_pose_pairs.py",
            "--render-manifest",
            "render.csv",
            "--output-dir",
            "run",
            "--mode",
            "overlap-list",
            "--pair-spec-manifest",
            "overlap_pairs.csv",
            "--overlap-scan-all",
            "--overlap-resume",
            "--overlap-start-index",
            "12",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = lazy_bench.parse_args()

        self.assertEqual(args.mode, "overlap-list")
        self.assertEqual(args.pair_spec_manifest, Path("overlap_pairs.csv"))
        self.assertTrue(args.overlap_scan_all)
        self.assertTrue(args.overlap_resume)
        self.assertEqual(args.overlap_start_index, 12)

    def test_gpu_snapshot_interval_collects_first_and_interval_steps(self) -> None:
        self.assertTrue(lazy_bench._should_collect_gpu_snapshot(1, 25))
        self.assertFalse(lazy_bench._should_collect_gpu_snapshot(2, 25))
        self.assertFalse(lazy_bench._should_collect_gpu_snapshot(24, 25))
        self.assertTrue(lazy_bench._should_collect_gpu_snapshot(25, 25))
        self.assertTrue(lazy_bench._should_collect_gpu_snapshot(50, 25))

    def test_gpu_usage_monitor_writes_latest_snapshot_without_step_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            samples = iter(
                [
                    {
                        "gpu_util_percent": "34",
                        "gpu_mem_used_mib": "1024",
                        "gpu_mem_total_mib": "16303",
                    },
                    {
                        "gpu_util_percent": "91",
                        "gpu_mem_used_mib": "12000",
                        "gpu_mem_total_mib": "16303",
                    },
                ]
            )

            monitor = lazy_bench.GpuUsageMonitor(
                Path(temp) / "gpu_metrics.csv",
                sample_interval_s=1.0,
                snapshot_fn=lambda: next(samples),
                clock=lambda: 10.0,
            )
            monitor.sample_once()
            monitor.sample_once()
            monitor.close()

            self.assertEqual(monitor.latest()["gpu_util_percent"], "91")
            with (Path(temp) / "gpu_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["elapsed_s"], "0.000")
            self.assertEqual(rows[1]["gpu_mem_used_mib"], "12000")

    def test_illumination_consistency_pair_changes_light_without_geometry(self) -> None:
        pair = self.make_pair()
        config = PhotometricAugmentConfig(enabled=True, probability=1.0, brightness=0.4, gamma=0.5, shadow=0.4)

        changed = lazy_bench.make_illumination_consistency_pair(pair, config, seed=123)

        self.assertFalse(torch.allclose(changed.view_a, pair.view_a))
        self.assertFalse(torch.allclose(changed.view_b, pair.view_b))
        self.assertTrue(torch.equal(changed.warp_a_to_b, pair.warp_a_to_b))
        self.assertTrue(torch.equal(changed.valid_mask, pair.valid_mask))

    def test_illumination_match_pair_can_change_only_target_view(self) -> None:
        pair = self.make_pair()
        config = PhotometricAugmentConfig(enabled=True, probability=1.0, brightness=0.4, gamma=0.5, shadow=0.4)

        changed = make_illumination_match_pair(pair, config, seed=123, changed_view="b")

        self.assertTrue(torch.allclose(changed.view_a, pair.view_a))
        self.assertFalse(torch.allclose(changed.view_b, pair.view_b))
        self.assertTrue(torch.equal(changed.warp_a_to_b, pair.warp_a_to_b))
        self.assertTrue(torch.equal(changed.valid_mask, pair.valid_mask))

    def test_render_manifest_uses_image_path_as_uint8_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path = root / "images_u8" / "sample.tif"
            depth_path = root / "depth.tif"
            tsai_path = root / "sample.tsai"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"x")
            depth_path.write_bytes(b"x")
            tsai_path.write_text("x", encoding="utf-8")
            manifest = root / "render_manifest.csv"
            manifest.write_text(
                "pose_id,base_id,variant,split,tsai_path,image_path,depth_path\n"
                f"p,b,nadir,train,{tsai_path},{image_path},{depth_path}\n",
                encoding="utf-8",
            )

            records = lazy_bench._read_render_manifest(manifest, {})

        self.assertEqual(records[0].uint8_path, image_path)

    def test_build_pair_specs_rejects_uint8_directory_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            depth_path = root / "depth.tif"
            tsai_path = root / "sample.tsai"
            depth_path.write_bytes(b"x")
            tsai_path.write_text("x", encoding="utf-8")
            manifest = root / "render_manifest.csv"
            manifest.write_text(
                "pose_id,base_id,variant,split,tsai_path,image_path,depth_path\n"
                f"p0,b0,nadir,train,{tsai_path},{root},{depth_path}\n"
                f"p1,b0,small_01,train,{tsai_path},{root},{depth_path}\n",
                encoding="utf-8",
            )
            records = lazy_bench._read_render_manifest(manifest, {})

            specs = lazy_bench.build_pair_specs(
                records,
                split="train",
                reference_variant="nadir",
                target_variants=("small_01",),
                image_source="uint8",
                limit_pairs=0,
                seed=123,
                shuffle=False,
            )

        self.assertEqual(specs, [])

    def test_read_all_render_records_uses_manifest_stem_when_parent_names_collide(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = root / "manifests"
            manifests.mkdir()
            image_path = root / "image.tif"
            depth_path = root / "depth.tif"
            tsai_path = root / "camera.tsai"
            image_path.write_bytes(b"x")
            depth_path.write_bytes(b"x")
            tsai_path.write_text("x", encoding="utf-8")
            manifest_paths = []
            for name in ("fov090_render_manifest", "fov110_render_manifest"):
                manifest = manifests / f"{name}.csv"
                manifest.write_text(
                    "pose_id,base_id,variant,split,tsai_path,image_path,depth_path\n"
                    f"{name}_pose,{name}_base,nadir,train,{tsai_path},{image_path},{depth_path}\n",
                    encoding="utf-8",
                )
                manifest_paths.append(manifest)

            records = lazy_bench._read_all_render_records(manifest_paths, [])

        self.assertEqual([record.dataset_id for record in records], ["fov090", "fov110"])
        self.assertEqual([record.base_id for record in records], ["fov090:fov090_render_manifest_base", "fov110:fov110_render_manifest_base"])

    def test_read_all_render_records_uses_manifest_stem_for_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = root / "manifests"
            manifests.mkdir()
            image_path = root / "image.tif"
            depth_path = root / "depth.tif"
            tsai_path = root / "camera.tsai"
            image_path.write_bytes(b"x")
            depth_path.write_bytes(b"x")
            tsai_path.write_text("x", encoding="utf-8")
            manifest = manifests / "h100km_fov090_render_manifest.csv"
            manifest.write_text(
                "pose_id,base_id,variant,split,tsai_path,image_path,depth_path\n"
                f"p0,b0,mid_01,train,{tsai_path},{image_path},{depth_path}\n",
                encoding="utf-8",
            )

            records = lazy_bench._read_all_render_records([manifest], [])

        self.assertEqual([record.dataset_id for record in records], ["h100km_fov090"])
        self.assertEqual([record.base_id for record in records], ["b0"])

    def test_build_cross_camera_pair_specs_uses_different_base_same_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            records = self.make_render_records(
                Path(temp),
                dataset_id="fov090",
                base_ids=("b001", "b002"),
                variants=("nadir", "mid_01"),
            )

            specs = lazy_bench.build_cross_camera_pair_specs(
                records,
                split="train",
                cross_variants=("nadir", "mid_01"),
                offsets=(1,),
                image_source="uint8",
                start_index=0,
            )

        self.assertTrue(specs)
        self.assertTrue(all(spec.pair_type == lazy_bench.PAIR_TYPE_CROSS_CAMERA for spec in specs))
        self.assertTrue(all(spec.reference.raw_base_id != spec.target.raw_base_id for spec in specs))
        self.assertTrue(all(spec.reference.dataset_id == spec.target.dataset_id for spec in specs))

    def test_build_cross_fov_pair_specs_uses_different_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = self.make_render_records(
                root,
                dataset_id="fov090",
                base_ids=("b001",),
                variants=("nadir",),
            ) + self.make_render_records(
                root,
                dataset_id="fov110",
                base_ids=("b001",),
                variants=("nadir",),
            )

            specs = lazy_bench.build_cross_fov_pair_specs(
                records,
                split="train",
                cross_variants=("nadir",),
                offsets=(0,),
                image_source="uint8",
                start_index=0,
            )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].pair_type, lazy_bench.PAIR_TYPE_CROSS_FOV)
        self.assertNotEqual(specs[0].reference.dataset_id, specs[0].target.dataset_id)

    def test_build_cross_fov_pair_specs_uses_nearest_spatial_position(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [
                self.record(root, "b001", "nadir", dataset_id="fov090", lon_deg=0.0, lat_deg=0.0),
                self.record(root, "b002", "nadir", dataset_id="fov090", lon_deg=50.0, lat_deg=0.0),
                self.record(root, "b101", "nadir", dataset_id="fov110", lon_deg=49.0, lat_deg=0.0),
                self.record(root, "b102", "nadir", dataset_id="fov110", lon_deg=1.0, lat_deg=0.0),
            ]

            specs = lazy_bench.build_cross_fov_pair_specs(
                records,
                split="train",
                cross_variants=("nadir",),
                offsets=(0,),
                image_source="uint8",
                start_index=0,
            )

        matched = {(spec.reference.raw_base_id, spec.target.raw_base_id) for spec in specs}
        self.assertIn(("b001", "b102"), matched)
        self.assertIn(("b002", "b101"), matched)
        self.assertNotIn(("b001", "b101"), matched)

    def test_build_spatial_index_pair_specs_uses_intersecting_footprints(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [
                self.record(root, "b001", "nadir", dataset_id="fov090"),
                self.record(root, "b999", "nadir", dataset_id="fov110"),
                self.record(root, "b002", "nadir", dataset_id="fov110"),
            ]
            footprints = {
                records[0].pose_id: lazy_bench.SpatialFootprint(
                    record=records[0],
                    bounds=(0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
                ),
                records[1].pose_id: lazy_bench.SpatialFootprint(
                    record=records[1],
                    bounds=(5.0, 5.0, 5.0, 15.0, 15.0, 15.0),
                ),
                records[2].pose_id: lazy_bench.SpatialFootprint(
                    record=records[2],
                    bounds=(100.0, 100.0, 100.0, 110.0, 110.0, 110.0),
                ),
            }

            specs = lazy_bench.build_spatial_index_pair_specs(
                records,
                split="train",
                image_source="uint8",
                footprints=footprints,
                start_index=0,
            )

        matched = {(spec.reference.raw_base_id, spec.target.raw_base_id) for spec in specs}
        self.assertEqual(matched, {("b001", "b999")})
        self.assertEqual(specs[0].pair_type, lazy_bench.PAIR_TYPE_CROSS_FOV)

    def test_build_spatial_index_pair_specs_classifies_same_dataset_cross_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [
                self.record(root, "b001", "nadir", dataset_id="fov090"),
                self.record(root, "b002", "mid_01", dataset_id="fov090"),
            ]
            footprints = {
                records[0].pose_id: lazy_bench.SpatialFootprint(
                    record=records[0],
                    bounds=(0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
                ),
                records[1].pose_id: lazy_bench.SpatialFootprint(
                    record=records[1],
                    bounds=(5.0, 5.0, 5.0, 15.0, 15.0, 15.0),
                ),
            }

            specs = lazy_bench.build_spatial_index_pair_specs(
                records,
                split="train",
                image_source="uint8",
                footprints=footprints,
                start_index=10,
            )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].pair_index, 10)
        self.assertEqual(specs[0].pair_type, lazy_bench.PAIR_TYPE_CROSS_CAMERA)

    def test_build_lazy_pair_specs_spatial_index_can_sample_only_cross_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [
                self.record(root, "b001", "mid_01", dataset_id="fov090"),
                self.record(root, "b001", "nadir", dataset_id="fov090"),
                self.record(root, "b002", "nadir", dataset_id="fov090"),
            ]
            footprints = {
                record.pose_id: lazy_bench.SpatialFootprint(
                    record=record,
                    bounds=(0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
                )
                for record in records
            }

            with mock.patch.object(lazy_bench, "build_camera_spatial_footprints", return_value=footprints):
                specs, counts = lazy_bench.build_lazy_pair_specs(
                    records,
                    split="train",
                    pair_mode="spatial-index",
                    reference_variant="nadir",
                    target_variants=("mid_01",),
                    cross_variants=("nadir", "mid_01"),
                    cross_camera_offsets=(1,),
                    cross_fov_offsets=(0,),
                    image_source="uint8",
                    limit_pairs=2,
                    seed=123,
                    shuffle=False,
                    pair_type_weights={lazy_bench.PAIR_TYPE_CROSS_CAMERA: 1.0},
                )

        self.assertEqual(len(specs), 2)
        self.assertEqual(counts[lazy_bench.PAIR_TYPE_CROSS_CAMERA], 2)
        self.assertTrue(all(spec.pair_type == lazy_bench.PAIR_TYPE_CROSS_CAMERA for spec in specs))
        self.assertTrue(all(spec.reference.raw_base_id != spec.target.raw_base_id for spec in specs))

    def test_camera_sphere_footprint_bounds_uses_camera_rays(self) -> None:
        camera = SimpleNamespace(
            fu=10.0,
            fv=10.0,
            cu=5.0,
            cv=5.0,
            center=np.asarray([0.0, 0.0, 10.0], dtype=np.float64),
            rotation_world_to_camera=np.diag([1.0, 1.0, -1.0]),
        )

        bounds = lazy_bench.camera_sphere_footprint_bounds(
            camera,
            width=10,
            height=10,
            planet_radius_m=5.0,
            sample_grid=3,
            bbox_margin_m=0.0,
        )

        self.assertLess(bounds[0], 0.0)
        self.assertGreater(bounds[3], 0.0)
        self.assertLess(bounds[1], 0.0)
        self.assertGreater(bounds[4], 0.0)
        self.assertTrue(math.isclose(bounds[5], 5.0, rel_tol=0.0, abs_tol=1.0e-6))

    def test_build_camera_spatial_footprints_filters_allowed_heights(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [
                self.record(root, "b001", "nadir", dataset_id="fov090"),
                self.record(root, "b002", "nadir", dataset_id="fov090"),
                self.record(root, "b003", "nadir", dataset_id="fov090"),
            ]
            height_records = []
            for record, height in zip(records, (100, 150, 250)):
                tsai_path = root / "tsai" / f"h{height}km" / "fov090" / f"{record.raw_base_id}.tsai"
                tsai_path.parent.mkdir(parents=True, exist_ok=True)
                tsai_path.write_text("x", encoding="utf-8")
                height_records.append(replace(record, tsai_path=tsai_path))

            with (
                mock.patch.object(lazy_bench, "_cached_camera", return_value=object()),
                mock.patch.object(lazy_bench, "_camera_image_size_from_intrinsics", return_value=(10, 10)),
                mock.patch.object(
                    lazy_bench,
                    "camera_sphere_footprint_bounds",
                    return_value=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                ),
            ):
                footprints = lazy_bench.build_camera_spatial_footprints(
                    height_records,
                    split="train",
                    image_source="uint8",
                    planet_radius_m=lazy_bench.DEFAULT_PLANET_RADIUS_M,
                    sample_grid=5,
                    bbox_margin_m=0.0,
                    height_km_filter={100, 250},
                )

        self.assertEqual(set(footprints), {height_records[0].pose_id, height_records[2].pose_id})

    def test_build_lazy_pair_specs_mixed_includes_all_requested_pair_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = self.make_render_records(
                root,
                dataset_id="fov090",
                base_ids=("b001", "b002"),
                variants=("nadir", "mid_01"),
            ) + self.make_render_records(
                root,
                dataset_id="fov110",
                base_ids=("b001", "b002"),
                variants=("nadir", "mid_01"),
            )

            specs, counts = lazy_bench.build_lazy_pair_specs(
                records,
                split="train",
                pair_mode="mixed",
                reference_variant="nadir",
                target_variants=("mid_01",),
                cross_variants=("nadir", "mid_01"),
                cross_camera_offsets=(1,),
                cross_fov_offsets=(0,),
                image_source="uint8",
                limit_pairs=0,
                seed=123,
                shuffle=False,
            )

        self.assertGreater(counts[lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW], 0)
        self.assertGreater(counts[lazy_bench.PAIR_TYPE_CROSS_CAMERA], 0)
        self.assertGreater(counts[lazy_bench.PAIR_TYPE_CROSS_FOV], 0)
        self.assertEqual(sum(counts.values()), len(specs))

    def test_build_lazy_pair_specs_cross_fov_requires_two_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            records = self.make_render_records(
                Path(temp),
                dataset_id="fov090",
                base_ids=("b001", "b002"),
                variants=("nadir",),
            )

            with self.assertRaisesRegex(ValueError, "cross-fov pair mode requires"):
                lazy_bench.build_lazy_pair_specs(
                    records,
                    split="train",
                    pair_mode="cross-fov",
                    reference_variant="nadir",
                    target_variants=("nadir",),
                    cross_variants=("nadir",),
                    cross_camera_offsets=(1,),
                    cross_fov_offsets=(0,),
                    image_source="uint8",
                    limit_pairs=0,
                    seed=123,
                    shuffle=False,
                )

    def test_build_lazy_pair_specs_mixed_partial_weights_disable_missing_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            records = self.make_render_records(
                Path(temp),
                dataset_id="fov090",
                base_ids=("b001", "b002"),
                variants=("nadir",),
            )

            specs, counts = lazy_bench.build_lazy_pair_specs(
                records,
                split="train",
                pair_mode="mixed",
                reference_variant="nadir",
                target_variants=("nadir",),
                cross_variants=("nadir",),
                cross_camera_offsets=(1,),
                cross_fov_offsets=(0,),
                image_source="uint8",
                limit_pairs=0,
                seed=123,
                shuffle=False,
                pair_type_weights={lazy_bench.PAIR_TYPE_CROSS_CAMERA: 1.0},
            )

        self.assertTrue(specs)
        self.assertEqual(counts[lazy_bench.PAIR_TYPE_CROSS_CAMERA], len(specs))
        self.assertEqual(counts[lazy_bench.PAIR_TYPE_CROSS_FOV], 0)

    def test_pair_type_metric_columns_uses_stable_column_names(self) -> None:
        counts = {
            lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW: 2,
            lazy_bench.PAIR_TYPE_CROSS_CAMERA: 3,
            lazy_bench.PAIR_TYPE_CROSS_FOV: 0,
        }

        row = lazy_bench.pair_type_metric_columns(counts)

        self.assertEqual(row["pair_type_same_position_view"], 2)
        self.assertEqual(row["pair_type_cross_camera"], 3)
        self.assertEqual(row["pair_type_cross_fov"], 0)

    def test_pair_spec_manifest_round_trips_fixed_crop_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.record(root, "b001", "nadir", dataset_id="fov090")
            target = self.record(root, "b002", "mid_01", dataset_id="fov090")
            spec = lazy_bench.LazyPairSpec(
                pair_index=7,
                split="train",
                reference=reference,
                target=target,
                pair_type=lazy_bench.PAIR_TYPE_CROSS_CAMERA,
            )
            result = lazy_bench.LazyPairResult(
                spec=spec,
                pair=self.make_pair(),
                valid_fraction=0.25,
                valid_pixels=16,
                attempt_count=3,
                elapsed_ms=12.5,
                crop_a=lazy_bench.CropWindow(1, 2, 5, 6),
                crop_b=lazy_bench.CropWindow(7, 8, 11, 12),
            )
            manifest = root / "overlap_pairs.csv"

            lazy_bench.write_pair_spec_manifest(manifest, [result])
            specs = lazy_bench.read_pair_spec_manifest(manifest, [reference, target])

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].pair_index, 0)
        self.assertEqual(specs[0].pair_type, lazy_bench.PAIR_TYPE_CROSS_CAMERA)
        self.assertEqual(specs[0].reference.pose_id, reference.pose_id)
        self.assertEqual(specs[0].target.pose_id, target.pose_id)
        self.assertEqual(specs[0].fixed_crop_a, lazy_bench.CropWindow(1, 2, 5, 6))
        self.assertEqual(specs[0].fixed_crop_b, lazy_bench.CropWindow(7, 8, 11, 12))

    def test_prepare_lazy_pair_specs_uses_pair_spec_manifest_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.record(root, "b001", "nadir", dataset_id="fov090")
            target = self.record(root, "b002", "mid_01", dataset_id="fov090")
            result = lazy_bench.LazyPairResult(
                spec=lazy_bench.LazyPairSpec(
                    pair_index=0,
                    split="train",
                    reference=reference,
                    target=target,
                    pair_type=lazy_bench.PAIR_TYPE_CROSS_CAMERA,
                ),
                pair=self.make_pair(),
                valid_fraction=0.25,
                valid_pixels=16,
                attempt_count=1,
                elapsed_ms=1.0,
                crop_a=lazy_bench.CropWindow(1, 2, 5, 6),
                crop_b=lazy_bench.CropWindow(7, 8, 11, 12),
            )
            manifest = root / "overlap_pairs.csv"
            lazy_bench.write_pair_spec_manifest(manifest, [result])
            args = argparse.Namespace(
                pair_spec_manifest=manifest,
                mode="train",
                split="train",
                pair_mode="mixed",
                reference_variant="nadir",
                target_variants=("mid_01",),
                cross_variants=("nadir", "mid_01"),
                cross_camera_offsets=(1,),
                cross_fov_offsets=(0,),
                image_source="uint8",
                limit_pairs=0,
                seed=123,
                shuffle=True,
            )

            with mock.patch.object(lazy_bench, "build_lazy_pair_specs", side_effect=AssertionError("should not build")):
                specs, counts, source = lazy_bench.prepare_lazy_pair_specs(
                    args,
                    [reference, target],
                    target_variants=("mid_01",),
                    cross_variants=("nadir", "mid_01"),
                    effective_pair_type_weights={lazy_bench.PAIR_TYPE_CROSS_CAMERA: 1.0},
                )

        self.assertEqual(source, "pair_spec_manifest")
        self.assertEqual(counts[lazy_bench.PAIR_TYPE_CROSS_CAMERA], 1)
        self.assertEqual(specs[0].fixed_crop_a, lazy_bench.CropWindow(1, 2, 5, 6))
        self.assertEqual(specs[0].fixed_crop_b, lazy_bench.CropWindow(7, 8, 11, 12))

    def test_run_visual_report_passes_training_pair_selection_to_visualizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(
                auto_visual_report=True,
                render_manifest=root / "render.csv",
                uint8_manifest=root / "uint8.csv",
                output_dir=root / "out",
                visual_split="",
                split="train",
                reference_variant="nadir",
                target_variant=["mid_01", "extreme_03"],
                pair_mode="spatial-index",
                pair_type_weights={
                    lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW: 0.0,
                    lazy_bench.PAIR_TYPE_CROSS_CAMERA: 1.0,
                    lazy_bench.PAIR_TYPE_CROSS_FOV: 0.0,
                },
                cross_pair_variant=["mid_01", "extreme_03"],
                cross_camera_offsets=[1, 3],
                cross_fov_offsets=[0, 2],
                spatial_index_planet_radius_m=3396190.0,
                spatial_index_footprint_samples=5,
                spatial_index_margin_m=2000.0,
                spatial_index_height_km=[100],
                pair_spec_manifest=root / "overlap_edges.csv",
                image_source="uint8",
                limit_pairs=5000,
                shuffle=True,
                visual_candidate_pairs=24,
                visual_select_count=6,
                seed=123,
                crop_size=2048,
                visual_max_image_size=768,
                max_attempts=8,
                min_valid_fraction=0.05,
                absolute_depth_tolerance_m=100.0,
                relative_depth_tolerance=0.005,
                visual_device="",
                device="cuda",
                visual_descriptor_mode="learned",
                visual_keypoint_score_mode="texture",
                visual_matcher_mode="graph_matcher",
                visual_max_keypoints=512,
                visual_max_matches=0,
                visual_draw_matches=0,
                visual_threshold_px=5.0,
                visual_post_filter_profile="fov76_geo5_geo10_extreme_rescue_lowmatch_guard",
                visual_geometry_filter="local",
                visual_geometry_threshold_px=8.0,
                visual_graph_width_prune_min_score=0.25,
                visual_graph_early_stop_min_confidence=0.85,
                visual_graph_max_attention_layers=2,
                visual_graph_max_attention_work_fraction=0.5,
                visual_graph_width_prune_keep_ratio=0.75,
                visual_filtered_report=True,
                visual_filtered_geometry_filter="local",
                visual_filtered_min_margin=0.02,
                visual_filtered_min_score=-1.0,
                visual_filtered_min_matches=16,
                visual_filtered_max_matches=0,
                visual_filtered_draw_matches=0,
                input_local_contrast=True,
                input_local_contrast_strength=0.35,
                input_local_contrast_kernel=31,
                visual_filtered_mutual=True,
            )

            with mock.patch.object(lazy_bench.subprocess, "run") as run:
                report_dir = lazy_bench._run_visual_report(args, root / "state.pt")

        command = run.call_args.args[0]
        self.assertEqual(report_dir, root / "out" / "visual_report")
        self.assertIn("--pair-spec-manifest", command)
        self.assertEqual(command[command.index("--pair-spec-manifest") + 1], str(root / "overlap_edges.csv"))
        self.assertIn("--pair-mode", command)
        self.assertEqual(command[command.index("--pair-mode") + 1], "spatial-index")
        self.assertIn("--pair-type-weights", command)
        self.assertIn("cross_camera=1", command[command.index("--pair-type-weights") + 1])
        self.assertIn("--spatial-index-height-km", command)
        self.assertEqual(command[command.index("--spatial-index-height-km") + 1], "100")
        self.assertIn("--graph-max-attention-layers", command)
        self.assertEqual(command[command.index("--graph-max-attention-layers") + 1], "2")
        self.assertIn("--graph-max-attention-work-fraction", command)
        self.assertEqual(command[command.index("--graph-max-attention-work-fraction") + 1], "0.5")
        self.assertIn("--graph-width-prune-keep-ratio", command)
        self.assertEqual(command[command.index("--graph-width-prune-keep-ratio") + 1], "0.75")
        self.assertIn("--geometry-filter", command)
        self.assertEqual(command[command.index("--geometry-filter") + 1], "local")
        self.assertIn("--geometry-threshold-px", command)
        self.assertEqual(command[command.index("--geometry-threshold-px") + 1], "8.0")
        self.assertIn("--post-filter-profile", command)
        self.assertEqual(
            command[command.index("--post-filter-profile") + 1],
            "fov76_geo5_geo10_extreme_rescue_lowmatch_guard",
        )
        self.assertIn("--filtered-min-matches", command)
        self.assertEqual(command[command.index("--filtered-min-matches") + 1], "16")

    def test_run_visual_report_can_use_step_specific_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(
                auto_visual_report=True,
                render_manifest=root / "render.csv",
                uint8_manifest=root / "uint8.csv",
                output_dir=root / "out",
                visual_split="",
                split="train",
                reference_variant="nadir",
                target_variant=[],
                pair_mode="same-position",
                pair_type_weights=lazy_bench.DEFAULT_PAIR_TYPE_WEIGHTS,
                cross_pair_variant=[],
                cross_camera_offsets=[1],
                cross_fov_offsets=[0],
                spatial_index_planet_radius_m=3396190.0,
                spatial_index_footprint_samples=5,
                spatial_index_margin_m=2000.0,
                spatial_index_height_km=[],
                pair_spec_manifest=None,
                image_source="uint8",
                limit_pairs=0,
                shuffle=False,
                visual_candidate_pairs=4,
                visual_select_count=2,
                seed=123,
                crop_size=1024,
                visual_max_image_size=512,
                max_attempts=4,
                min_valid_fraction=0.05,
                absolute_depth_tolerance_m=100.0,
                relative_depth_tolerance=0.005,
                visual_device="cpu",
                device="cuda",
                visual_descriptor_mode="learned",
                visual_keypoint_score_mode="learned",
                visual_matcher_mode="graph_matcher",
                visual_max_keypoints=128,
                visual_max_matches=0,
                visual_draw_matches=0,
                visual_threshold_px=5.0,
                visual_post_filter_profile="",
                visual_geometry_filter="local",
                visual_geometry_threshold_px=8.0,
                visual_graph_width_prune_min_score=-1.0,
                visual_graph_early_stop_min_confidence=-1.0,
                visual_graph_max_attention_layers=0,
                visual_graph_max_attention_work_fraction=1.0,
                visual_graph_width_prune_keep_ratio=1.0,
                visual_filtered_report=True,
                visual_filtered_geometry_filter="magsac",
                visual_filtered_min_margin=0.02,
                visual_filtered_min_score=-1.0,
                visual_filtered_min_matches=16,
                visual_filtered_max_matches=0,
                visual_filtered_draw_matches=0,
                input_local_contrast=False,
                input_local_contrast_strength=0.0,
                input_local_contrast_kernel=31,
                visual_filtered_mutual=True,
            )
            step_report_dir = root / "out" / "visual_report_step_000010"

            with mock.patch.object(lazy_bench.subprocess, "run") as run:
                report_dir = lazy_bench._run_visual_report(args, root / "state.pt", report_dir=step_report_dir)

        command = run.call_args.args[0]
        self.assertEqual(report_dir, step_report_dir)
        self.assertEqual(command[command.index("--output-dir") + 1], str(step_report_dir))
        self.assertIn("--write-match-details", command)

    def test_summarize_visual_report_metrics_includes_extreme_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_dir = root / "visual_report"
            report_dir.mkdir()
            with (report_dir / "all_filtered_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "label",
                        "base_id",
                        "target_variant",
                        "split",
                        "valid_fraction",
                        "matches",
                        "correct",
                        "wrong",
                        "precision",
                        "mean_error_px",
                        "median_error_px",
                        "image",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "label": "all-filtered",
                        "base_id": "b001",
                        "target_variant": "mid_01",
                        "split": "val",
                        "valid_fraction": "0.8",
                        "matches": "100",
                        "correct": "100",
                        "wrong": "0",
                        "precision": "1.0",
                        "mean_error_px": "1.0",
                        "median_error_px": "1.0",
                        "image": "",
                    }
                )
                writer.writerow(
                    {
                        "label": "all-filtered",
                        "base_id": "b002",
                        "target_variant": "extreme_02",
                        "split": "val",
                        "valid_fraction": "0.4",
                        "matches": "20",
                        "correct": "18",
                        "wrong": "2",
                        "precision": "0.9",
                        "mean_error_px": "2.0",
                        "median_error_px": "2.0",
                        "image": "",
                    }
                )
                writer.writerow(
                    {
                        "label": "all-filtered",
                        "base_id": "b003",
                        "target_variant": "extreme_03",
                        "split": "val",
                        "valid_fraction": "0.5",
                        "matches": "0",
                        "correct": "0",
                        "wrong": "0",
                        "precision": "0.0",
                        "mean_error_px": "0.0",
                        "median_error_px": "0.0",
                        "image": "",
                    }
                )

            metrics = lazy_bench.summarize_visual_report_metrics(report_dir)

        self.assertEqual(metrics["visual_filtered_rows"], 3)
        self.assertEqual(metrics["visual_num_filtered_matches"], 120)
        self.assertEqual(metrics["visual_RANSAC_inlier_count"], 120)
        self.assertEqual(metrics["visual_filtered_correct"], 118)
        self.assertEqual(metrics["visual_filtered_wrong"], 2)
        self.assertAlmostEqual(metrics["visual_filtered_precision"], 118 / 120)
        self.assertAlmostEqual(metrics["visual_filtered_recall"], 2 / 3)
        self.assertEqual(metrics["visual_filtered_matches_min"], 0)
        self.assertAlmostEqual(metrics["visual_filtered_matches_mean"], 40.0)
        self.assertEqual(metrics["visual_filtered_matches_max"], 100)
        self.assertEqual(metrics["visual_extreme_rows"], 2)
        self.assertEqual(metrics["visual_extreme_num_filtered_matches"], 20)
        self.assertEqual(metrics["visual_extreme_RANSAC_inlier_count"], 20)
        self.assertEqual(metrics["visual_extreme_correct"], 18)
        self.assertEqual(metrics["visual_extreme_wrong"], 2)
        self.assertAlmostEqual(metrics["visual_extreme_precision"], 18 / 20)
        self.assertAlmostEqual(metrics["visual_extreme_recall"], 1 / 2)
        self.assertEqual(metrics["visual_extreme_matches_min"], 0)
        self.assertAlmostEqual(metrics["visual_extreme_matches_mean"], 10.0)
        self.assertEqual(metrics["visual_extreme_matches_max"], 20)

    def test_summarize_visual_report_metrics_includes_matcher_logit_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_dir = root / "visual_report"
            report_dir.mkdir()

            def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)

            write_csv(
                report_dir / "all_filtered_summary.csv",
                [
                    "label",
                    "base_id",
                    "target_variant",
                    "split",
                    "valid_fraction",
                    "matches",
                    "correct",
                    "wrong",
                    "precision",
                    "mean_error_px",
                    "median_error_px",
                    "image",
                ],
                [
                    {
                        "label": "all-filtered",
                        "base_id": "b001",
                        "target_variant": "extreme_02",
                        "split": "val",
                        "valid_fraction": "0.8",
                        "matches": "3",
                        "correct": "2",
                        "wrong": "1",
                        "precision": "0.6667",
                        "mean_error_px": "1.0",
                        "median_error_px": "1.0",
                        "image": "",
                    }
                ],
            )
            write_csv(
                report_dir / "all_filtered_match_details.csv",
                [
                    "label",
                    "pair_index",
                    "base_id",
                    "reference_variant",
                    "target_variant",
                    "split",
                    "match_index",
                    "score",
                    "pair_logit",
                    "row_dustbin_logit",
                    "col_dustbin_logit",
                    "positive_vs_dustbin_margin",
                    "raw_similarity",
                    "raw_margin",
                    "accept_logit",
                    "accept_probability",
                    "error_px",
                    "correct",
                    "valid_fraction",
                ],
                [
                    {
                        "label": "all-filtered",
                        "pair_index": "0",
                        "base_id": "b001",
                        "reference_variant": "nadir",
                        "target_variant": "extreme_02",
                        "split": "val",
                        "match_index": "0",
                        "score": "8.0",
                        "pair_logit": "10.0",
                        "row_dustbin_logit": "1.0",
                        "col_dustbin_logit": "2.0",
                        "positive_vs_dustbin_margin": "7.0",
                        "raw_similarity": "0.9",
                        "raw_margin": "0.3",
                        "accept_logit": "2.0",
                        "accept_probability": "0.88",
                        "error_px": "0.5",
                        "correct": "1",
                        "valid_fraction": "0.8",
                    },
                    {
                        "label": "all-filtered",
                        "pair_index": "0",
                        "base_id": "b001",
                        "reference_variant": "nadir",
                        "target_variant": "extreme_02",
                        "split": "val",
                        "match_index": "1",
                        "score": "1.0",
                        "pair_logit": "3.0",
                        "row_dustbin_logit": "2.0",
                        "col_dustbin_logit": "2.5",
                        "positive_vs_dustbin_margin": "-1.5",
                        "raw_similarity": "0.4",
                        "raw_margin": "0.05",
                        "accept_logit": "-1.0",
                        "accept_probability": "0.27",
                        "error_px": "9.0",
                        "correct": "0",
                        "valid_fraction": "0.8",
                    },
                ],
            )

            metrics = lazy_bench.summarize_visual_report_metrics(report_dir)

        self.assertEqual(metrics["visual_match_detail_rows"], 2)
        self.assertAlmostEqual(metrics["visual_pair_logit_mean"], 6.5)
        self.assertAlmostEqual(metrics["visual_dustbin_logit_for_match_mean"], 3.75)
        self.assertAlmostEqual(metrics["visual_positive_vs_dustbin_margin_mean"], 2.75)
        self.assertAlmostEqual(metrics["visual_positive_vs_dustbin_margin_p10"], -1.5)
        self.assertAlmostEqual(metrics["visual_positive_vs_dustbin_margin_below0_ratio"], 0.5)
        self.assertAlmostEqual(metrics["visual_accept_logit_mean"], 0.5)

    def test_checkpoint_selection_scores_use_training_and_visual_metrics(self) -> None:
        row = {
            "top1_accuracy": "0.875",
            "loss": "4.2",
        }
        visual_metrics = {
            "visual_RANSAC_inlier_count": 120,
            "visual_extreme_score": 12,
        }

        self.assertAlmostEqual(lazy_bench.recall_checkpoint_score(row), 0.875)
        self.assertEqual(lazy_bench.ransac_inlier_checkpoint_score(visual_metrics), 120.0)
        self.assertEqual(lazy_bench.extreme_checkpoint_score(visual_metrics), 12.0)
        self.assertIsNone(lazy_bench.recall_checkpoint_score({"top1_accuracy": "nan"}))
        self.assertIsNone(lazy_bench.ransac_inlier_checkpoint_score({}))
        self.assertIsNone(lazy_bench.extreme_checkpoint_score({}))

    def test_should_run_periodic_visual_eval_only_on_positive_intervals(self) -> None:
        self.assertFalse(lazy_bench.should_run_periodic_visual_eval(step=5, every_steps=0))
        self.assertFalse(lazy_bench.should_run_periodic_visual_eval(step=5, every_steps=-1))
        self.assertFalse(lazy_bench.should_run_periodic_visual_eval(step=5, every_steps=3))
        self.assertTrue(lazy_bench.should_run_periodic_visual_eval(step=6, every_steps=3))

    def test_restore_last_good_checkpoint_restores_model_and_reduces_optimizer_lr(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "last_good.pt"
            model = torch.nn.Linear(2, 1)
            with torch.no_grad():
                model.weight.fill_(1.5)
                model.bias.fill_(0.25)
            saved_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            torch.save({"model": saved_state, "training": {"step": 7}}, checkpoint)

            with torch.no_grad():
                model.weight.fill_(9.0)
                model.bias.fill_(9.0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
            loss = model(torch.ones(1, 2)).sum()
            loss.backward()
            optimizer.step()
            self.assertTrue(optimizer.state)

            recovery = lazy_bench.restore_last_good_checkpoint(
                checkpoint,
                model,
                optimizer,
                device=torch.device("cpu"),
                lr_factor=0.25,
            )

        self.assertTrue(torch.allclose(model.weight, saved_state["weight"]))
        self.assertTrue(torch.allclose(model.bias, saved_state["bias"]))
        self.assertFalse(optimizer.state)
        self.assertEqual(recovery["checkpoint_step"], 7)
        self.assertEqual(recovery["old_lrs"], [0.1])
        self.assertEqual(recovery["new_lrs"], [0.025])

    def test_run_overlap_list_writes_pair_spec_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.record(root, "b001", "nadir", dataset_id="fov090")
            target = self.record(root, "b002", "mid_01", dataset_id="fov090")
            spec = lazy_bench.LazyPairSpec(
                pair_index=0,
                split="train",
                reference=reference,
                target=target,
                pair_type=lazy_bench.PAIR_TYPE_CROSS_CAMERA,
            )
            result = lazy_bench.LazyPairResult(
                spec=spec,
                pair=self.make_pair(),
                valid_fraction=0.25,
                valid_pixels=16,
                attempt_count=2,
                elapsed_ms=12.5,
                crop_a=lazy_bench.CropWindow(1, 2, 5, 6),
                crop_b=lazy_bench.CropWindow(7, 8, 11, 12),
            )
            args = argparse.Namespace(
                output_dir=root / "out",
                pair_spec_manifest=root / "out" / "overlap_pairs.csv",
                pairs=1,
                workers=1,
                prefetch_batches=1,
                worker_cache_items=0,
                crop_size=4,
                image_source="uint8",
                max_attempts=2,
                min_valid_fraction=0.05,
                absolute_depth_tolerance_m=100.0,
                relative_depth_tolerance=0.005,
                seed=123,
                skip_bad_pairs=True,
                max_bad_pairs=1,
                progress_every=1,
                overlap_scan_all=False,
            )

            with mock.patch.object(lazy_bench, "iter_lazy_pairs", return_value=iter([result])):
                summary = lazy_bench.run_overlap_list(args, [spec])

            specs = lazy_bench.read_pair_spec_manifest(args.pair_spec_manifest, [reference, target])
            metrics_exists = (args.output_dir / "overlap_metrics.csv").is_file()

        self.assertEqual(summary["mode"], "overlap-list")
        self.assertEqual(summary["pairs"], 1)
        self.assertEqual(specs[0].fixed_crop_a, lazy_bench.CropWindow(1, 2, 5, 6))
        self.assertEqual(specs[0].fixed_crop_b, lazy_bench.CropWindow(7, 8, 11, 12))
        self.assertTrue(metrics_exists)

    def test_run_train_passes_teacher_score_floor_to_train_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.record(root, "b001", "nadir", dataset_id="fov090")
            target = self.record(root, "b001", "extreme_02", dataset_id="fov090")
            spec = lazy_bench.LazyPairSpec(
                pair_index=0,
                split="train",
                reference=reference,
                target=target,
                pair_type=lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW,
            )
            result = lazy_bench.LazyPairResult(
                spec=spec,
                pair=self.make_pair(),
                valid_fraction=0.25,
                valid_pixels=16,
                attempt_count=1,
                elapsed_ms=1.0,
                crop_a=lazy_bench.CropWindow(0, 0, 4, 4),
                crop_b=lazy_bench.CropWindow(0, 0, 4, 4),
            )
            argv = [
                "benchmark_lazy_pose_pairs.py",
                "--render-manifest",
                str(root / "render.csv"),
                "--output-dir",
                str(root / "out"),
                "--mode",
                "train",
                "--device",
                "cpu",
                "--steps",
                "1",
                "--batch-pairs",
                "1",
                "--samples-per-pair",
                "2",
                "--workers",
                "1",
                "--prefetch-batches",
                "1",
                "--worker-cache-items",
                "0",
                "--train-graph-matcher",
                "--graph-matcher-teacher-guard-state",
                str(root / "teacher.pt"),
                "--graph-matcher-teacher-score-floor-weight",
                "0.25",
                "--graph-matcher-teacher-score-floor-tolerance",
                "0.3",
                "--graph-matcher-teacher-score-floor-min-score",
                "0.4",
                "--graph-matcher-teacher-match-count-floor-weight",
                "0.02",
                "--graph-matcher-teacher-match-count-floor-threshold",
                "18.0",
                "--graph-matcher-teacher-match-count-floor-margin",
                "0.5",
                "--no-save-best-checkpoints",
                "--no-auto-visual-report",
                "--no-gpu-monitor",
            ]
            teacher = object()
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            train_calls: list[dict] = []

            def fake_train_step(*_args, **kwargs):
                train_calls.append(kwargs)
                return {
                    "loss": 1.0,
                    "top1_accuracy": 0.5,
                    "mean_positive_rank": 1.0,
                    "points": 2.0,
                    "graph_matcher_teacher_score_floor_loss": 0.125,
                    "graph_matcher_teacher_score_floor_violations": 2.0,
                    "graph_matcher_teacher_score_floor_delta_mean": -0.4,
                    "graph_matcher_teacher_score_floor_teacher_score_mean": 1.2,
                    "graph_matcher_teacher_match_count_floor_loss": 0.0625,
                    "graph_matcher_teacher_match_count_floor_teacher_count": 8.0,
                    "graph_matcher_teacher_match_count_floor_student_count": 6.0,
                    "graph_matcher_teacher_match_count_floor_count_deficit": 2.0,
                }

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(lazy_bench, "_load_model", return_value=(model, optimizer)),
                mock.patch.object(lazy_bench, "load_graph_matcher_teacher_guard_model", return_value=teacher),
                mock.patch.object(lazy_bench, "iter_lazy_pairs", return_value=iter([result])),
                mock.patch.object(lazy_bench, "train_step", side_effect=fake_train_step),
                mock.patch.object(lazy_bench, "_save_training_state"),
            ):
                args = lazy_bench.parse_args()
                summary = lazy_bench.run_train(args, [spec])

            metrics_text = (args.output_dir / "train_metrics.csv").read_text(encoding="utf-8")

        self.assertEqual(summary["steps"], 1)
        self.assertEqual(len(train_calls), 1)
        self.assertIs(train_calls[0]["graph_matcher_teacher_guard_model"], teacher)
        self.assertAlmostEqual(train_calls[0]["graph_matcher_teacher_score_floor_weight"], 0.25)
        self.assertAlmostEqual(train_calls[0]["graph_matcher_teacher_score_floor_tolerance"], 0.3)
        self.assertAlmostEqual(train_calls[0]["graph_matcher_teacher_score_floor_min_score"], 0.4)
        self.assertAlmostEqual(train_calls[0]["graph_matcher_teacher_match_count_floor_weight"], 0.02)
        self.assertAlmostEqual(train_calls[0]["graph_matcher_teacher_match_count_floor_threshold"], 18.0)
        self.assertAlmostEqual(train_calls[0]["graph_matcher_teacher_match_count_floor_margin"], 0.5)
        self.assertIn("graph_matcher_teacher_score_floor_loss", metrics_text)
        self.assertIn("graph_matcher_teacher_match_count_floor_loss", metrics_text)
        self.assertIn("0.125000", metrics_text)

    def test_run_train_passes_training_image_size_controls_to_train_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.record(root, "b001", "nadir", dataset_id="fov090")
            target = self.record(root, "b001", "extreme_02", dataset_id="fov090")
            spec = lazy_bench.LazyPairSpec(
                pair_index=0,
                split="train",
                reference=reference,
                target=target,
                pair_type=lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW,
            )
            result = lazy_bench.LazyPairResult(
                spec=spec,
                pair=self.make_pair(),
                valid_fraction=0.25,
                valid_pixels=16,
                attempt_count=1,
                elapsed_ms=1.0,
                crop_a=lazy_bench.CropWindow(0, 0, 4, 4),
                crop_b=lazy_bench.CropWindow(0, 0, 4, 4),
            )
            argv = [
                "benchmark_lazy_pose_pairs.py",
                "--render-manifest",
                str(root / "render.csv"),
                "--output-dir",
                str(root / "out"),
                "--mode",
                "train",
                "--device",
                "cpu",
                "--steps",
                "1",
                "--batch-pairs",
                "1",
                "--samples-per-pair",
                "2",
                "--workers",
                "1",
                "--prefetch-batches",
                "1",
                "--worker-cache-items",
                "0",
                "--training-crop-size",
                "96",
                "--training-max-image-size",
                "128",
                "--no-save-best-checkpoints",
                "--no-auto-visual-report",
                "--no-gpu-monitor",
            ]
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            train_calls: list[dict] = []

            def fake_train_step(*_args, **kwargs):
                train_calls.append(kwargs)
                return {
                    "loss": 1.0,
                    "top1_accuracy": 0.5,
                    "mean_positive_rank": 1.0,
                    "points": 2.0,
                }

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(lazy_bench, "_load_model", return_value=(model, optimizer)),
                mock.patch.object(lazy_bench, "iter_lazy_pairs", return_value=iter([result])),
                mock.patch.object(lazy_bench, "train_step", side_effect=fake_train_step),
                mock.patch.object(lazy_bench, "_save_training_state"),
            ):
                args = lazy_bench.parse_args()
                summary = lazy_bench.run_train(args, [spec])

        self.assertEqual(summary["steps"], 1)
        self.assertEqual(len(train_calls), 1)
        self.assertEqual(train_calls[0]["training_crop_size"], 96)
        self.assertEqual(train_calls[0]["training_max_image_size"], 128)

    def test_run_overlap_list_scan_all_processes_specs_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.record(root, "b001", "nadir", dataset_id="fov090")
            target_a = self.record(root, "b002", "mid_01", dataset_id="fov090")
            target_b = self.record(root, "b003", "mid_01", dataset_id="fov090")
            specs = [
                lazy_bench.LazyPairSpec(
                    pair_index=0,
                    split="train",
                    reference=reference,
                    target=target_a,
                    pair_type=lazy_bench.PAIR_TYPE_CROSS_CAMERA,
                ),
                lazy_bench.LazyPairSpec(
                    pair_index=1,
                    split="train",
                    reference=reference,
                    target=target_b,
                    pair_type=lazy_bench.PAIR_TYPE_CROSS_CAMERA,
                ),
            ]
            results = [
                lazy_bench.LazyPairResult(
                    spec=specs[0],
                    pair=self.make_pair(),
                    valid_fraction=0.25,
                    valid_pixels=16,
                    attempt_count=1,
                    elapsed_ms=1.0,
                    crop_a=lazy_bench.CropWindow(1, 2, 5, 6),
                    crop_b=lazy_bench.CropWindow(7, 8, 11, 12),
                ),
                lazy_bench.LazyPairResult(
                    spec=specs[1],
                    pair=self.make_pair(),
                    valid_fraction=0.35,
                    valid_pixels=20,
                    attempt_count=1,
                    elapsed_ms=2.0,
                    crop_a=lazy_bench.CropWindow(2, 3, 6, 7),
                    crop_b=lazy_bench.CropWindow(8, 9, 12, 13),
                ),
            ]
            args = argparse.Namespace(
                output_dir=root / "out",
                pair_spec_manifest=root / "out" / "overlap_pairs.csv",
                pairs=999,
                workers=1,
                prefetch_batches=1,
                worker_cache_items=0,
                crop_size=4,
                image_source="uint8",
                max_attempts=2,
                min_valid_fraction=0.05,
                absolute_depth_tolerance_m=100.0,
                relative_depth_tolerance=0.005,
                seed=123,
                skip_bad_pairs=True,
                max_bad_pairs=1,
                progress_every=1,
                overlap_scan_all=True,
            )

            with (
                mock.patch.object(lazy_bench, "iter_lazy_pairs", side_effect=AssertionError("should not cycle")),
                mock.patch.object(lazy_bench, "iter_lazy_pair_specs_once", return_value=iter(results)) as scan_once,
            ):
                summary = lazy_bench.run_overlap_list(args, specs)

            round_tripped = lazy_bench.read_pair_spec_manifest(args.pair_spec_manifest, [reference, target_a, target_b])

        self.assertEqual(summary["mode"], "overlap-list")
        self.assertTrue(summary["overlap_scan_all"])
        self.assertEqual(summary["candidate_specs"], 2)
        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(len(round_tripped), 2)
        scan_once.assert_called_once()

    def test_generate_lazy_pair_rejects_best_crop_below_min_valid_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.record(root, "b001", "nadir")
            target = self.record(root, "b002", "nadir")
            spec = lazy_bench.LazyPairSpec(
                pair_index=0,
                split="train",
                reference=reference,
                target=target,
                pair_type=lazy_bench.PAIR_TYPE_CROSS_CAMERA,
            )
            pair = self.make_pair()
            crop = lazy_bench.CropWindow(0, 0, 4, 4)

            with (
                mock.patch.object(lazy_bench, "_cached_view", return_value=pair.view_a),
                mock.patch.object(lazy_bench, "_cached_depth", return_value=torch.ones(4, 4).numpy()),
                mock.patch.object(lazy_bench, "_cached_camera", return_value=object()),
                mock.patch.object(lazy_bench, "_project_crop_pair", return_value=(pair, 0.0, 0, crop, crop)),
            ):
                with self.assertRaisesRegex(RuntimeError, "below min_valid_fraction"):
                    lazy_bench.generate_lazy_pair(
                        spec,
                        crop_size=4,
                        image_source="uint8",
                        max_attempts=2,
                        min_valid_fraction=0.05,
                        absolute_depth_tolerance_m=100.0,
                        relative_depth_tolerance=0.005,
                        seed=123,
                    )

    def test_generate_lazy_pair_uses_fixed_crop_windows_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self.record(root, "b001", "nadir")
            target = self.record(root, "b002", "nadir")
            crop_a = lazy_bench.CropWindow(1, 2, 5, 6)
            crop_b = lazy_bench.CropWindow(7, 8, 11, 12)
            spec = lazy_bench.LazyPairSpec(
                pair_index=0,
                split="train",
                reference=reference,
                target=target,
                pair_type=lazy_bench.PAIR_TYPE_CROSS_CAMERA,
                fixed_crop_a=crop_a,
                fixed_crop_b=crop_b,
            )
            pair = self.make_pair()

            with (
                mock.patch.object(lazy_bench, "_cached_view", return_value=pair.view_a),
                mock.patch.object(lazy_bench, "_cached_depth", return_value=torch.ones(4, 4).numpy()),
                mock.patch.object(lazy_bench, "_cached_camera", return_value=object()),
                mock.patch.object(lazy_bench, "_project_crop_pair", return_value=(pair, 0.25, 4, crop_a, crop_b)) as project,
            ):
                result = lazy_bench.generate_lazy_pair(
                    spec,
                    crop_size=4,
                    image_source="uint8",
                    max_attempts=8,
                    min_valid_fraction=0.05,
                    absolute_depth_tolerance_m=100.0,
                    relative_depth_tolerance=0.005,
                    seed=123,
                )

        self.assertEqual(result.crop_a, crop_a)
        self.assertEqual(result.crop_b, crop_b)
        self.assertEqual(project.call_count, 1)
        self.assertEqual(project.call_args.kwargs["fixed_crop_a"], crop_a)
        self.assertEqual(project.call_args.kwargs["fixed_crop_b"], crop_b)

    def test_streaming_csv_rows_flushes_after_each_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.csv"
            with StreamingCsvRows(path, ["step", "loss"]) as writer:
                writer.write({"step": 1, "loss": "2.5", "ignored": "x"})
                self.assertIn("1,2.5", path.read_text(encoding="utf-8"))

            text = path.read_text(encoding="utf-8")
            self.assertIn("step,loss", text)
            self.assertNotIn("ignored", text)

    def test_streaming_csv_rows_append_preserves_existing_rows_without_extra_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.csv"
            with StreamingCsvRows(path, ["step", "loss"]) as writer:
                writer.write({"step": 1, "loss": "2.5"})
            with StreamingCsvRows(path, ["step", "loss"], append=True) as writer:
                writer.write({"step": 2, "loss": "1.5"})

            lines = path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(lines, ["step,loss", "1,2.5", "2,1.5"])

    def test_read_overlap_resume_state_uses_last_source_pair_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "overlap_edges.csv"
            metrics = root / "overlap_metrics.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=lazy_bench.PAIR_SPEC_MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow({"pair_index": 0, "split": "train"})
                writer.writerow({"pair_index": 1, "split": "train"})
            with metrics.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["index", "source_pair_index"])
                writer.writeheader()
                writer.writerow({"index": 1, "source_pair_index": 7})
                writer.writerow({"index": 2, "source_pair_index": 11})

            state = lazy_bench._read_overlap_resume_state(manifest, metrics)

        self.assertEqual(state.pair_count, 2)
        self.assertEqual(state.next_source_pair_index, 12)


if __name__ == "__main__":
    unittest.main()
