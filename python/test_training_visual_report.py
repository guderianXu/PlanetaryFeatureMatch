import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import training_visual_report as report


class TrainingVisualReportTest(unittest.TestCase):
    def test_parse_args_accepts_graph_attention_work_budget(self):
        argv = [
            "training_visual_report.py",
            "--run-dir",
            "run",
            "--validation-cache-dir",
            "val",
            "--graph-max-attention-work-fraction",
            "0.55",
            "--graph-width-prune-keep-ratio",
            "0.4",
            "--no-pdf",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = report.parse_args()

        self.assertEqual(args.graph_max_attention_work_fraction, 0.55)
        self.assertEqual(args.graph_width_prune_keep_ratio, 0.4)


if __name__ == "__main__":
    unittest.main()
