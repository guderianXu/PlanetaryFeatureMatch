import unittest
import tempfile

import torch

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

from benchmark_lazy_pose_pairs import PhotometricAugmentConfig, StreamingCsvRows, apply_photometric_augmentation
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
