import unittest

import torch

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from patch_descriptor_training import SyntheticPair
from continuous_rotation_stress_eval import rotate_pair_from_view
from illumination_stress_eval import make_illumination_variants
from benchmark_lazy_pose_pairs import LazyPairSpec, LazyPairResult, RenderRecord
from visualize_lazy_pose_matches import LazyMatchVisual, make_illumination_stress_lazy_results


class StressEvalScriptsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
