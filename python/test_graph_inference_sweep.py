import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sweep_graph_inference_configs as sweep


class GraphInferenceSweepTest(unittest.TestCase):
    def test_parse_float_list_validates_accept_probability_range(self):
        self.assertEqual(sweep.parse_float_list("-1, 0.5,0.7"), [-1.0, 0.5, 0.7])

        with self.assertRaisesRegex(ValueError, r"\[-1, 1\]"):
            sweep.parse_float_list("1.2")

    def test_build_eval_command_includes_strict_graph_options(self):
        args = Namespace(
            cache_dir=[Path("/cache/train")],
            pytorch_state=Path("/models/state.pt"),
            checkpoint=None,
            output_dir=Path("/out"),
            device="cuda",
            mode="blend",
            texture_blend_weight=1.0,
            max_keypoints=2048,
            max_matches=0,
            min_intensity=0.0,
            texture_keypoint_fraction=1.0,
            weak_texture_keypoint_fraction=0.0,
            keypoint_spatial_bins=8,
            keypoint_cell_cap=0,
            keypoint_score_mode="texture",
            threshold_px=5.0,
            descriptor_topk=1,
            mutual=False,
            geometry_filter="none",
            min_score=-1.0,
            min_margin=0.0,
            graph_dustbin_delta=0.0,
            graph_acceptance_margin=0.0,
            graph_min_raw_score=-1.0,
            graph_min_raw_margin=0.0,
            min_target_gradient=0.0,
            min_target_local_contrast=0.0,
            limit_pairs=16,
            sample_seed=123,
            exclude_self_pairs=True,
            hard_summary=[],
            hard_limit=64,
            hard_min_matches=4,
            hard_max_precision=0.9,
        )
        config = sweep.GraphSweepConfig("high_precision", 0.7, "none")

        command = sweep.build_eval_command(args, config, Path("/out/high_precision.csv"))

        self.assertIn("--matcher-mode", command)
        self.assertIn("graph_matcher", command)
        self.assertIn("--graph-inference-preset", command)
        self.assertIn("high_precision", command)
        self.assertIn("--graph-min-accept-probability", command)
        self.assertIn("0.7", command)
        self.assertIn("--graph-fallback-mode", command)
        self.assertIn("none", command)
        self.assertIn("--sample-seed", command)
        self.assertIn("123", command)
        self.assertIn("--exclude-self-pairs", command)

    def test_summarize_eval_csv_and_write_reports(self):
        config = sweep.GraphSweepConfig("high_precision", 0.7, "none")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_csv = tmp_path / "eval.csv"
            eval_csv.write_text(
                "pair_pt,matches,correct,wrong,precision,"
                "graph_executed_layers,graph_input_keypoints_a,graph_input_keypoints_b,"
                "graph_kept_keypoints_a,graph_kept_keypoints_b,"
                "graph_pruned_keypoints_a,graph_pruned_keypoints_b,"
                "graph_attention_work_units,graph_full_attention_work_units,graph_attention_work_fraction\n"
                "a.pt,10,8,2,0.8,2,10,8,7,6,3,2,10,20,0.5\n"
                "b.pt,5,5,0,1.0,4,20,18,16,15,4,3,8,32,0.25\n",
                encoding="utf-8",
            )

            summary = sweep.summarize_eval_csv(eval_csv, config)

            self.assertEqual(summary.pairs, 2)
            self.assertEqual(summary.matches, 15)
            self.assertEqual(summary.correct, 13)
            self.assertEqual(summary.wrong, 2)
            self.assertAlmostEqual(summary.precision, 13 / 15)
            self.assertEqual(summary.low_precision_pairs, 1)
            self.assertAlmostEqual(summary.avg_executed_layers, 3.0)
            self.assertAlmostEqual(summary.avg_input_keypoints_a, 15.0)
            self.assertAlmostEqual(summary.avg_input_keypoints_b, 13.0)
            self.assertAlmostEqual(summary.avg_kept_keypoints_a, 11.5)
            self.assertAlmostEqual(summary.avg_kept_keypoints_b, 10.5)
            self.assertAlmostEqual(summary.pruned_keypoint_fraction, 12 / 56)
            self.assertAlmostEqual(summary.attention_work_fraction, 18 / 52)

            summary_csv = tmp_path / "summary.csv"
            report_html = tmp_path / "report.html"
            sweep.write_summary_csv([summary], summary_csv)
            sweep.write_report_html([summary], report_html)

            self.assertIn("high_precision", summary_csv.read_text(encoding="utf-8"))
            self.assertIn("0.866667", summary_csv.read_text(encoding="utf-8"))
            self.assertIn("avg_executed_layers", summary_csv.read_text(encoding="utf-8"))
            self.assertIn("3.000", summary_csv.read_text(encoding="utf-8"))
            self.assertIn("0.214286", summary_csv.read_text(encoding="utf-8"))
            self.assertIn("attention_work_fraction", summary_csv.read_text(encoding="utf-8"))
            self.assertIn("0.346154", summary_csv.read_text(encoding="utf-8"))
            html = report_html.read_text(encoding="utf-8")
            self.assertIn("<html", html)
            self.assertIn("high_precision", html)
            self.assertIn("严格图匹配", html)
            self.assertIn("平均执行层数", html)
            self.assertIn("剪枝比例", html)
            self.assertIn("计算量比例", html)


if __name__ == "__main__":
    unittest.main()
