import unittest

import torch

from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from patch_descriptor_training import SyntheticPair
from continuous_rotation_stress_eval import rotate_pair_from_view
from illumination_stress_eval import make_illumination_variants
from benchmark_lazy_pose_pairs import LazyPairSpec, LazyPairResult, RenderRecord
import visualize_lazy_pose_matches as visual_mod
from visualize_lazy_pose_matches import (
    LazyMatchVisual,
    filter_visual_matches,
    make_illumination_stress_lazy_results,
    selected_draw_indices,
)


class StressEvalScriptsTest(unittest.TestCase):
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
        self.assertGreater(args.filtered_min_margin, 0.0)

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


if __name__ == "__main__":
    unittest.main()
