import argparse
import csv
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import watch_lazy_visual_report as watch_report


class WatchLazyVisualReportTest(unittest.TestCase):
    def test_read_last_step_handles_missing_and_valid_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing = root / "missing.csv"
            self.assertEqual(watch_report.read_last_step(missing), 0)

            metrics = root / "train_metrics.csv"
            with metrics.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["step", "loss"])
                writer.writeheader()
                writer.writerow({"step": "2", "loss": "1.0"})
                writer.writerow({"step": "12.0", "loss": "0.5"})

            self.assertEqual(watch_report.read_last_step(metrics), 12)

    def test_read_last_step_returns_zero_for_bad_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = Path(temp) / "train_metrics.csv"
            with metrics.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["step"])
                writer.writeheader()
                writer.writerow({"step": "bad"})

            self.assertEqual(watch_report.read_last_step(metrics), 0)

    def test_build_visual_command_includes_core_and_local_contrast_args(self) -> None:
        args = argparse.Namespace(
            render_manifest=Path("render_manifest.csv"),
            uint8_manifest=Path("uint8_manifest.csv"),
            checkpoint=Path("state.pt"),
            output_dir=Path("visual"),
            run_dir=Path("run"),
            metrics_csv=Path("train_metrics.csv"),
            split="val",
            reference_variant="nadir",
            candidate_pairs=24,
            select_count=6,
            seed=123,
            crop_size=768,
            max_image_size=768,
            max_attempts=40,
            min_valid_fraction=0.1,
            absolute_depth_tolerance_m=100.0,
            relative_depth_tolerance=0.005,
            device="cpu",
            descriptor_mode="learned",
            keypoint_score_mode="texture",
            max_keypoints=384,
            max_matches=0,
            draw_matches=0,
            threshold_px=5.0,
            filtered_geometry_filter="local",
            filtered_min_score=-1.0,
            filtered_min_margin=0.02,
            filtered_max_matches=0,
            filtered_draw_matches=0,
            filtered_report=True,
            illumination_stress=True,
            input_local_contrast=True,
            input_local_contrast_strength=0.55,
            input_local_contrast_kernel=31,
        )

        command = watch_report.build_visual_command(args)

        self.assertIn("scripts/visualize_lazy_pose_matches.py", command[1])
        self.assertIn("--filtered-report", command)
        self.assertIn("--illumination-stress", command)
        self.assertIn("--input-local-contrast", command)
        self.assertIn("--input-local-contrast-strength", command)
        self.assertIn("0.55", command)
        self.assertIn("--device", command)
        self.assertIn("cpu", command)


if __name__ == "__main__":
    unittest.main()
