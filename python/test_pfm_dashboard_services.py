import csv
import os
import tempfile
import time
import unittest
from pathlib import Path

from pfm_dashboard.services import (
    dataset_split_counts,
    delete_run,
    discover_runs,
    pid_status,
    read_metrics_csv,
    start_run_script,
    stop_run,
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
        self.assertEqual(runs[0].progress_percent, 100.0)
        self.assertIsNotNone(runs[0].created_at)
        self.assertIsNotNone(runs[0].completed_at)
        self.assertTrue(runs[0].can_delete)

    def test_discover_runs_infers_step_progress_from_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "python_run"
            run.mkdir()
            (run / "metrics.csv").write_text("step,loss\n4,1.0\n", encoding="utf-8")
            (run / "train.sh").write_text("#!/usr/bin/env bash\npython train.py --steps 10\n", encoding="utf-8")

            runs = discover_runs(root)

        self.assertEqual(runs[0].progress_percent, 40.0)
        self.assertEqual(runs[0].progress_label, "4/10 步")
        self.assertTrue(runs[0].can_start)

    def test_discover_runs_infers_cpp_iteration_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "cpp_run"
            run.mkdir()
            (run / "metrics.csv").write_text(
                "epoch,total_epochs,iteration,total_iterations,loss_total\n"
                "1,1,25,100,3.5\n",
                encoding="utf-8",
            )

            runs = discover_runs(root)

        self.assertEqual(runs[0].progress_percent, 25.0)
        self.assertEqual(runs[0].progress_label, "25/100 步")

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

    def test_pid_status_treats_zombie_as_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pid_file = Path(temp) / "zombie.pid"
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            try:
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    stat_path = Path(f"/proc/{pid}/stat")
                    if stat_path.exists() and ") Z" in stat_path.read_text(encoding="utf-8", errors="replace"):
                        break
                    time.sleep(0.01)
                pid_file.write_text(str(pid), encoding="utf-8")
                self.assertEqual(pid_status(pid_file), "stopped")
            finally:
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass

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

    def test_start_and_stop_run_script_manage_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            run.mkdir()
            script = run / "train.sh"
            script.write_text("#!/usr/bin/env bash\nsleep 60\n", encoding="utf-8")
            script.chmod(0o755)

            pid = start_run_script(run)
            try:
                self.assertEqual(pid_status(run / "train.pid"), "running")
                self.assertEqual(stop_run(run), pid)
                deadline = time.time() + 5.0
                while time.time() < deadline and pid_status(run / "train.pid") == "running":
                    waited, _ = os.waitpid(pid, os.WNOHANG)
                    if waited == pid:
                        break
                    time.sleep(0.05)
                self.assertNotEqual(pid_status(run / "train.pid"), "running")
            finally:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass

    def test_delete_run_moves_directory_to_trash_and_discovery_skips_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "old_run"
            run.mkdir()
            (run / "train.sh").write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")

            target = delete_run(run)
            runs = discover_runs(root)

            self.assertFalse(run.exists())
            self.assertIn(".trash", target.parts)
            self.assertTrue(target.exists())
            self.assertEqual(runs, [])


if __name__ == "__main__":
    unittest.main()
