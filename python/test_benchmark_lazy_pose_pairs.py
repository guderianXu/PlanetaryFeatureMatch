import unittest
import tempfile
from unittest import mock

import torch

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

import benchmark_lazy_pose_pairs as lazy_bench
from benchmark_lazy_pose_pairs import (
    PhotometricAugmentConfig,
    StreamingCsvRows,
    apply_local_contrast_normalization,
    apply_photometric_augmentation,
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
        self.assertEqual(args.false_match_csv, [Path("false.csv")])
        self.assertEqual(args.hard_variant, ["extreme"])
        self.assertTrue(args.input_local_contrast)
        self.assertAlmostEqual(args.input_local_contrast_strength, 0.6)

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
