import csv
import tempfile
import unittest
from pathlib import Path

from pfm_dashboard.services import (
    dataset_split_counts,
    discover_runs,
    pid_status,
    read_metrics_csv,
    tail_text,
)


class DashboardServicesTest(unittest.TestCase):
    def test_read_metrics_csv_keeps_numeric_series(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = Path(temp) / "metrics.csv"
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["step", "loss", "descriptor_accuracy", "note"])
                writer.writeheader()
                writer.writerow({"step": "1", "loss": "7.5", "descriptor_accuracy": "0.25", "note": "first"})
                writer.writerow({"step": "2", "loss": "3.25", "descriptor_accuracy": "0.5", "note": "second"})

            parsed = read_metrics_csv(metrics)

        self.assertEqual(parsed.columns, ["step", "loss", "descriptor_accuracy", "note"])
        self.assertEqual(parsed.rows[-1]["step"], 2.0)
        self.assertEqual(parsed.rows[-1]["loss"], 3.25)
        self.assertEqual(parsed.latest["descriptor_accuracy"], 0.5)

    def test_discover_runs_reports_metrics_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "cpp_run"
            run.mkdir()
            (run / "train.log").write_text("hello\nloss=1\n", encoding="utf-8")
            (run / "run.html").write_text("<html></html>", encoding="utf-8")
            (run / "model.pt").write_text("checkpoint", encoding="utf-8")
            (run / "metrics.csv").write_text("step,loss\n1,9.0\n", encoding="utf-8")

            runs = discover_runs(root)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].name, "cpp_run")
        self.assertEqual(runs[0].backend, "cpp")
        self.assertEqual(runs[0].latest_metrics["loss"], 9.0)
        self.assertEqual(runs[0].checkpoint_count, 1)
        self.assertTrue(runs[0].has_report)

    def test_tail_text_returns_last_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "train.log"
            log.write_text("a\nb\nc\n", encoding="utf-8")
            self.assertEqual(tail_text(log, lines=2), "b\nc\n")

    def test_pid_status_detects_missing_and_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.pid"
            self.assertEqual(pid_status(missing), "missing")
            current = Path(temp) / "current.pid"
            current.write_text(str(__import__("os").getpid()), encoding="utf-8")
            self.assertEqual(pid_status(current), "running")

    def test_dataset_split_counts_counts_pair_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split, count in {"train": 2, "val": 1, "test": 3}.items():
                split_dir = root / split / "source_000001"
                split_dir.mkdir(parents=True)
                for index in range(count):
                    (split_dir / f"pair_{index:06d}.pt").write_text("x", encoding="utf-8")

            counts = dataset_split_counts(root)

        self.assertEqual(counts["train"], 2)
        self.assertEqual(counts["val"], 1)
        self.assertEqual(counts["test"], 3)
        self.assertEqual(counts["total"], 6)


if __name__ == "__main__":
    unittest.main()
