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
            "--graph-matcher-online-false-no-match",
            "--graph-matcher-accept-weight",
            "0.2",
            "--graph-matcher-prune-ranking-weight",
            "0.15",
            "--graph-matcher-stop-confidence-weight",
            "0.07",
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
        self.assertTrue(args.graph_matcher_online_false_no_match)
        self.assertAlmostEqual(args.graph_matcher_accept_weight, 0.2)
        self.assertAlmostEqual(args.graph_matcher_prune_ranking_weight, 0.15)
        self.assertAlmostEqual(args.graph_matcher_stop_confidence_weight, 0.07)
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
        self.assertAlmostEqual(args.graph_matcher_hard_negative_dustbin_weight, 0.25)
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
        self.assertTrue(args.save_best_checkpoints)

    def test_graph_matcher_dustbin_diagnostic_fields_are_declared_for_training_logs(self) -> None:
        self.assertIn("true_match_rejected_by_dustbin_ratio", lazy_bench.GRAPH_MATCHER_DIAGNOSTIC_METRIC_FIELDS)
        self.assertIn("positive_vs_dustbin_margin_mean", lazy_bench.GRAPH_MATCHER_DIAGNOSTIC_METRIC_FIELDS)
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
                visual_graph_width_prune_min_score=0.25,
                visual_graph_early_stop_min_confidence=0.85,
                visual_filtered_report=True,
                visual_filtered_geometry_filter="local",
                visual_filtered_min_margin=0.02,
                visual_filtered_min_score=-1.0,
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
