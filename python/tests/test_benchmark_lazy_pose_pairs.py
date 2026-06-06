import csv
import unittest
import tempfile
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

    def test_streaming_csv_rows_flushes_after_each_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.csv"
            with StreamingCsvRows(path, ["step", "loss"]) as writer:
                writer.write({"step": 1, "loss": "2.5", "ignored": "x"})
                self.assertIn("1,2.5", path.read_text(encoding="utf-8"))

            text = path.read_text(encoding="utf-8")
            self.assertIn("step,loss", text)
            self.assertNotIn("ignored", text)


if __name__ == "__main__":
    unittest.main()
