import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


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

    def test_match_summary_writes_lightglue_graph_telemetry(self):
        pair = report.SyntheticPair(
            view_a=torch.zeros(1, 8, 8),
            view_b=torch.zeros(1, 8, 8),
            warp_a_to_b=torch.zeros(8, 8, 2),
            valid_mask=torch.ones(8, 8, dtype=torch.bool),
        )
        result = report.VisualMatchResult(
            pair_path=Path("pair.pt"),
            pair=pair,
            points_a=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
            points_b=np.asarray([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32),
            scores=np.asarray([0.9, 0.2], dtype=np.float32),
            errors=np.asarray([0.5, 12.0], dtype=np.float32),
            correct=np.asarray([True, False]),
            graph_executed_layers=2,
            graph_input_keypoints_a=8,
            graph_input_keypoints_b=10,
            graph_kept_keypoints_a=4,
            graph_kept_keypoints_b=5,
            graph_pruned_keypoints_a=4,
            graph_pruned_keypoints_b=5,
            graph_attention_work_units=40,
            graph_full_attention_work_units=80,
            graph_attention_work_fraction=0.5,
        )
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "match_visual_summary.csv"
            report.write_match_summary([result], output, requested_matches=4)
            text = output.read_text(encoding="utf-8")

        self.assertIn("graph_executed_layers", text)
        self.assertIn("graph_attention_work_fraction", text)
        self.assertIn("graph_pruned_keypoint_fraction", text)
        self.assertIn(",2,8,10,4,5,4,5,40,80,0.500000,0.500000,0.500000", text)


if __name__ == "__main__":
    unittest.main()
