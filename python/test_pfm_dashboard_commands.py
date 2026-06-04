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
            )

            runs = create_training_runs(request)

        self.assertEqual(len(runs), 1)
        script = runs[0].script_text
        self.assertIn("python/pfm_pytorch_training.py", script)
        self.assertIn("--cache-dir /cache/train", script)
        self.assertIn("--validation-cache-dir /cache/val", script)
        self.assertIn("--training-crop-size 512", script)
        self.assertIn("--training-max-image-size 512", script)
        self.assertIn("--learning-rate 3e-05", script)

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


if __name__ == "__main__":
    unittest.main()
