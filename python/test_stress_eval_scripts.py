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


if __name__ == "__main__":
    unittest.main()
