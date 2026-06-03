import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import pseudo_label_generation as plg


class PseudoLabelGenerationTest(unittest.TestCase):
    def test_filter_matches_by_warp_truth_keeps_only_valid_close_matches(self):
        warp = torch.zeros(4, 4, 2)
        for y in range(4):
            for x in range(4):
                warp[y, x] = torch.tensor([x + 1.0, y + 2.0])
        valid = torch.ones(4, 4, dtype=torch.bool)
        valid[2, 2] = False
        points_a = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
        points_b = np.array([[2.0, 3.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

        filtered_a, filtered_b, errors = plg.filter_matches_by_warp_truth(
            points_a,
            points_b,
            warp,
            valid,
            threshold_px=0.25,
        )

        self.assertTrue(np.allclose(filtered_a, np.array([[1.0, 1.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(filtered_b, np.array([[2.0, 3.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(errors, np.array([0.0], dtype=np.float32)))

    def test_pseudo_label_rows_use_csv_fields_and_fixed_precision(self):
        rows = plg.rows_from_matches(
            pair_path=Path("cache/source_a/pair_000001.pt"),
            points_a=np.array([[1.25, 2.5]], dtype=np.float32),
            points_b=np.array([[3.75, 4.0]], dtype=np.float32),
            errors=np.array([0.125], dtype=np.float32),
            matcher="RootSIFT-FLANN-ratio",
            stage="homography",
            cache_dir=Path("cache"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pseudo.csv"
            plg.write_pseudo_label_csv(output, rows)
            text = output.read_text(encoding="utf-8")

        self.assertIn("pair_pt,ax,ay,bx,by,matcher,stage,error_px,cache_dir", text.splitlines()[0])
        self.assertIn("cache/source_a/pair_000001.pt,1.250,2.500,3.750,4.000,RootSIFT-FLANN-ratio,homography,0.125,cache", text)

    def test_cap_matches_is_reproducible(self):
        points_a = np.stack([np.arange(10), np.arange(10)], axis=1).astype(np.float32)
        points_b = points_a + 1.0
        errors = np.arange(10, dtype=np.float32)

        first = plg.cap_matches(points_a, points_b, errors, max_matches=4, seed=7)
        second = plg.cap_matches(points_a, points_b, errors, max_matches=4, seed=7)

        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertEqual(first[0].shape[0], 4)


if __name__ == "__main__":
    unittest.main()
