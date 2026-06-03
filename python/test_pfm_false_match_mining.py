import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mine_pfm_false_matches as miner


class PFMFalseMatchMiningTest(unittest.TestCase):
    def test_false_match_rows_keep_only_wrong_matches(self):
        warp = torch.zeros(4, 4, 2)
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        valid = torch.ones(4, 4, dtype=torch.bool)
        points_a = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        points_b = torch.tensor([[1.1, 1.0], [0.0, 0.0]])
        scores = torch.tensor([0.9, 0.8])
        margins = torch.tensor([0.2, 0.1])

        rows = miner.false_match_rows_from_tensors(
            pair_pt="cache/source_a/pair_000001.pt",
            style="numeric",
            gate="viewpoint",
            points_a=points_a,
            points_b=points_b,
            scores=scores,
            margins=margins,
            warp_a_to_b=warp,
            valid_mask=valid,
            threshold_px=1.0,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].pair_pt, "cache/source_a/pair_000001.pt")
        self.assertEqual(rows[0].style, "numeric")
        self.assertEqual(rows[0].gate, "viewpoint")
        self.assertGreater(rows[0].error_px, 1.0)
        self.assertAlmostEqual(rows[0].score, 0.8)
        self.assertAlmostEqual(rows[0].margin, 0.1)


if __name__ == "__main__":
    unittest.main()
