import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from graph_matcher_modes import graph_matcher_mode_config


class GraphMatcherModesTest(unittest.TestCase):
    def test_high_precision_mode_defaults(self) -> None:
        cfg = graph_matcher_mode_config("high_precision")

        self.assertEqual(cfg.name, "high_precision")
        self.assertEqual(cfg.matcher_mode, "graph_matcher")
        self.assertEqual(cfg.graph_metadata_mode, "no_xy")
        self.assertLess(cfg.graph_min_raw_score, 0.0)
        self.assertEqual(cfg.graph_min_raw_margin, 0.0)
        self.assertIn("graphmatcher_no_xy_dustbin", cfg.pytorch_state)

    def test_balanced_mode_keeps_raw_filter(self) -> None:
        cfg = graph_matcher_mode_config("balanced")

        self.assertEqual(cfg.name, "balanced")
        self.assertEqual(cfg.graph_min_raw_score, 0.4)
        self.assertEqual(cfg.graph_min_raw_margin, 0.01)
        self.assertEqual(cfg.max_matches, 512)
        self.assertEqual(cfg.spatial_bins, 8)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            graph_matcher_mode_config("not-a-mode")


if __name__ == "__main__":
    unittest.main()
