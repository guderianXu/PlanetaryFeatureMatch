import tempfile
import unittest
from pathlib import Path

from pfm_dashboard.commands import TrainingRequest, create_training_runs


class DashboardCommandsTest(unittest.TestCase):
    def test_create_python_training_run_writes_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = TrainingRequest(
                experiment_name="exp",
                backend="python",
                cache_dirs=["/cache/train"],
                validation_cache_dirs=["/cache/val"],
                output_root=Path(temp),
                device="cuda",
                epochs=2,
                resize=512,
                training_crop_size=512,
                learning_rate=3.0e-5,
                graph_matcher_no_match_points=32,
                graph_matcher_no_match_weight=0.1,
                graph_matcher_no_match_min_distance=5.0,
                graph_matcher_stop_confidence_weight=0.07,
                graph_matcher_stop_confidence_margin=0.6,
                graph_min_accept_probability=0.7,
                graph_max_attention_work_fraction=0.55,
                graph_width_prune_keep_ratio=0.4,
            )

            runs = create_training_runs(request)
            run_html = runs[0].html_path.read_text(encoding="utf-8")

        self.assertEqual(len(runs), 1)
        script = runs[0].script_text
        self.assertIn("python/pfm_pytorch_training.py", script)
        self.assertIn("--cache-dir /cache/train", script)
        self.assertIn("--validation-cache-dir /cache/val", script)
        self.assertIn("--training-crop-size 512", script)
        self.assertIn("--training-max-image-size 512", script)
        self.assertIn("--learning-rate 3e-05", script)
        self.assertIn("--graph-matcher-no-match-points 32", script)
        self.assertIn("--graph-matcher-no-match-weight 0.1", script)
        self.assertIn("--graph-matcher-no-match-min-distance 5.0", script)
        self.assertIn("--graph-matcher-stop-confidence-weight 0.07", script)
        self.assertIn("--graph-matcher-stop-confidence-margin 0.6", script)
        self.assertIn("--generate-training-report", script)
        self.assertIn("--report-matcher-mode graph_matcher", script)
        self.assertIn("--report-graph-inference-preset fast", script)
        self.assertIn("--report-graph-width-prune-min-score 0.25", script)
        self.assertIn("--report-graph-early-stop-min-confidence 0.85", script)
        self.assertIn("--report-graph-min-accept-probability 0.7", script)
        self.assertIn("--report-graph-max-attention-work-fraction 0.55", script)
        self.assertIn("--report-graph-width-prune-keep-ratio 0.4", script)
        self.assertIn("graph_max_attention_work_fraction=0.55", run_html)
        self.assertIn("graph_width_prune_keep_ratio=0.4", run_html)
        self.assertIn("graph_matcher_no_match_points=32", run_html)
        self.assertIn("graph_matcher_no_match_weight=0.1", run_html)
        self.assertIn("graph_matcher_stop_confidence_weight=0.07", run_html)

    def test_create_python_training_run_accepts_high_precision_graph_report_preset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = TrainingRequest(
                experiment_name="exp",
                backend="python",
                cache_dirs=["/cache/train"],
                validation_cache_dirs=["/cache/val"],
                output_root=Path(temp),
                graph_inference_preset="high_precision",
            )

            runs = create_training_runs(request)

        script = runs[0].script_text
        self.assertIn("--report-graph-inference-preset high_precision", script)
        self.assertIn("--report-graph-width-prune-min-score 0.5", script)
        self.assertIn("--report-graph-early-stop-min-confidence 0.85", script)

    def test_create_cpp_training_run_writes_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = TrainingRequest(
                experiment_name="exp",
                backend="cpp",
                cache_dirs=["/cache/train"],
                output_root=Path(temp),
                device="cuda",
                full_v21=True,
                memory_cache_items=16,
                graph_matcher_no_match_points=24,
                graph_matcher_no_match_min_distance=6.0,
                graph_matcher_stop_confidence_weight=0.07,
                graph_matcher_stop_confidence_margin=0.6,
            )

            runs = create_training_runs(request)

        self.assertEqual(len(runs), 1)
        script = runs[0].script_text
        self.assertIn("./build/pfm_cli train", script)
        self.assertIn("--training-profile full", script)
        self.assertIn("--pair-cache-dir /cache/train", script)
        self.assertIn("--full-v21", script)
        self.assertIn("--memory-cache-items 16", script)
        self.assertIn("--training-crop-size 512", script)
        self.assertIn("--weight-decay 0.0001", script)
        self.assertIn("--graph-matcher-no-match-points 24", script)
        self.assertIn("--graph-matcher-no-match-min-distance 6.0", script)
        self.assertIn("--graph-matcher-stop-confidence-weight 0.07", script)
        self.assertIn("--graph-matcher-stop-confidence-margin 0.6", script)
        self.assertIn("--train-backbone", script)
        self.assertIn("--train-blended-descriptors", script)
        self.assertIn("--train-graph-matcher", script)

    def test_create_cpp_python_compare_run_uses_constant_learning_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = TrainingRequest(
                experiment_name="exp",
                backend="cpp",
                cache_dirs=["/cache/train"],
                output_root=Path(temp),
                profile="python-compare",
            )

            runs = create_training_runs(request)

        script = runs[0].script_text
        self.assertIn("--training-profile python-compare", script)
        self.assertIn("--min-learning-rate-ratio 1.0", script)

    def test_compare_backend_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = TrainingRequest(
                experiment_name="exp",
                backend="compare",
                cache_dirs=["/cache/train"],
                output_root=Path(temp),
            )

            with self.assertRaisesRegex(ValueError, "backend must be python or cpp"):
                create_training_runs(request)

    def test_graph_min_accept_probability_range_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = TrainingRequest(
                experiment_name="exp",
                backend="python",
                cache_dirs=["/cache/train"],
                output_root=Path(temp),
                graph_min_accept_probability=1.5,
            )

            with self.assertRaisesRegex(ValueError, "graph_min_accept_probability must be in"):
                create_training_runs(request)

    def test_graph_max_attention_work_fraction_range_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = TrainingRequest(
                experiment_name="exp",
                backend="python",
                cache_dirs=["/cache/train"],
                output_root=Path(temp),
                graph_max_attention_work_fraction=1.5,
            )

            with self.assertRaisesRegex(ValueError, "graph_max_attention_work_fraction must be in"):
                create_training_runs(request)


if __name__ == "__main__":
    unittest.main()
