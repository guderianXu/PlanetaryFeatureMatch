import tempfile
import unittest
from pathlib import Path

import analyze_fov76_checkpoint_delta as mod


class AnalyzeFov76CheckpointDeltaTest(unittest.TestCase):
    def test_summarizes_direct_and_selector_deltas(self):
        rows = [
            {
                "source": "formal",
                "split": "val",
                "base_id": "b1",
                "target_variant": "extreme_02",
                "match_delta": "5",
                "correct_delta": "5",
                "wrong_delta": "0",
                "selected_model": "phase5g_active",
                "selector_reason": "blocked_homography_p90:3.8>3.2",
            },
            {
                "source": "formal",
                "split": "test",
                "base_id": "b2",
                "target_variant": "mid_02",
                "match_delta": "-8",
                "correct_delta": "-8",
                "wrong_delta": "0",
                "selected_model": "phase5g_active",
                "selector_reason": "blocked_target_variant:mid_02",
            },
        ]

        summary = mod.summarize_combined_rows(rows)

        self.assertEqual(summary["formal"]["val"]["correct_delta_sum"], 5)
        self.assertEqual(summary["formal"]["test"]["correct_delta_sum"], -8)
        self.assertEqual(summary["formal"]["val"]["gain_rows"], 1)
        self.assertEqual(summary["formal"]["test"]["loss_rows"], 1)
        self.assertEqual(summary["selector_reason_counts"]["blocked_target_variant:mid_02"], 1)

    def test_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            combined = output / "combined_filtered_summary.csv"
            combined.write_text(
                "source,split,base_id,target_variant,match_delta,correct_delta,wrong_delta,selected_model,selector_reason\n"
                "formal,val,b1,extreme_02,5,5,0,phase5g_active,blocked_homography_p90:3.8>3.2\n",
                encoding="utf-8",
            )

            result = mod.run_analysis(combined_csv=combined, output_dir=output)

            self.assertTrue((output / "delta_summary.json").exists())
            self.assertTrue((output / "delta_by_variant.csv").exists())
            self.assertTrue((output / "delta_top_gains.csv").exists())
            self.assertTrue((output / "delta_top_losses.csv").exists())
            self.assertTrue((output / "index.html").exists())
            self.assertEqual(result["formal"]["val"]["correct_delta_sum"], 5)


if __name__ == "__main__":
    unittest.main()
