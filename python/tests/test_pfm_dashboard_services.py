import csv
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from pfm_dashboard import services as dashboard_services
from pfm_dashboard.services import (
    dataset_split_counts,
    delete_run,
    discover_hybrid_gate_runs,
    discover_runs,
    discover_true_geometry_selector_runs,
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

    def test_read_metrics_csv_ignores_partial_tail_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = Path(temp) / "metrics.csv"
            metrics.write_text(
                "iteration,total_iterations,loss_total,descriptor_accuracy,descriptor_positive_rank\n"
                "9,10,0.7,0.002,240\n"
                "10,10,0.6\n",
                encoding="utf-8",
            )

            parsed = read_metrics_csv(metrics)

        self.assertEqual(len(parsed.rows), 1)
        self.assertEqual(parsed.latest["iteration"], 9.0)
        self.assertEqual(parsed.latest["descriptor_accuracy"], 0.002)
        self.assertEqual(parsed.latest["descriptor_positive_rank"], 240.0)

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

    def test_discover_runs_skips_hybrid_gate_summary_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "python_run"
            run.mkdir()
            (run / "metrics.csv").write_text("step,loss\n1,2.0\n", encoding="utf-8")
            hybrid = root / "hybrid_gate"
            hybrid.mkdir()
            (hybrid / "summary.json").write_text(
                '{"hybrid_correct": 28444, "hybrid_wrong": 37}',
                encoding="utf-8",
            )

            runs = discover_runs(root)

        self.assertEqual([run.name for run in runs], ["python_run"])

    def test_discover_runs_marks_external_output_dir_process_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "python_external"
            run.mkdir()
            (run / "train_metrics.csv").write_text("step,loss\n30,2.0\n", encoding="utf-8")
            (root / "python_external.sh").write_text(
                "#!/usr/bin/env bash\npython scripts/benchmark_lazy_pose_pairs.py --steps 100\n",
                encoding="utf-8",
            )

            with mock.patch.object(dashboard_services, "_active_training_output_dirs", return_value={run.resolve()}):
                runs = dashboard_services.discover_runs(root)

        self.assertEqual(runs[0].status, "running")
        self.assertEqual(runs[0].progress_percent, 30.0)
        self.assertEqual(runs[0].progress_label, "30/100 步")
        self.assertTrue(runs[0].can_stop)
        self.assertFalse(runs[0].can_delete)
        self.assertIsNone(runs[0].completed_at)

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

    def test_discover_hybrid_gate_runs_reads_pipeline_and_apply_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline = root / "hybrid_pipeline"
            pipeline.mkdir()
            (pipeline / "pipeline_summary.json").write_text(
                """
{
  "valid": true,
  "hybrid_summary_json": "hybrid_pipeline/summary.json",
  "validation_json": "hybrid_pipeline/validation.json",
  "optimization_audit_json": "hybrid_pipeline/optimization_audit.json",
  "correct_delta_vs_lightglue": 661,
  "wrong_delta_vs_lightglue": 0,
  "precision_delta_vs_lightglue": 0.0000308
}
""",
                encoding="utf-8",
            )
            (pipeline / "summary.json").write_text(
                """
{
  "rows": 200,
  "kept_pfm_rows": 2,
  "fallback_lightglue_rows": 198,
  "hybrid_correct": 28444,
  "hybrid_wrong": 37,
  "hybrid_precision": 0.998700888,
  "lightglue_correct": 27783,
  "lightglue_wrong": 37,
  "lightglue_precision": 0.998670022,
  "hybrid_correct_delta_vs_lightglue": 661,
  "hybrid_wrong_delta_vs_lightglue": 0,
  "threshold": 0.1008127,
  "reject_action": "lightglue"
}
""",
                encoding="utf-8",
            )
            standalone = root / "pfm_only_gate"
            standalone.mkdir()
            (standalone / "summary.json").write_text(
                """
{
  "rows": 200,
  "kept_pfm_rows": 23,
  "rejected_rows": 177,
  "hybrid_correct": 12930,
  "hybrid_wrong": 37,
  "hybrid_precision": 0.997146603,
  "lightglue_correct": 27783,
  "lightglue_wrong": 37,
  "hybrid_correct_delta_vs_lightglue": -14853,
  "hybrid_wrong_delta_vs_lightglue": 0,
  "threshold": 0.1789114,
  "reject_action": "zero"
}
""",
                encoding="utf-8",
            )

            gates = discover_hybrid_gate_runs(root)

        names = {gate["name"] for gate in gates}
        self.assertIn("hybrid_pipeline", names)
        self.assertIn("pfm_only_gate", names)
        by_name = {gate["name"]: gate for gate in gates}
        self.assertTrue(by_name["hybrid_pipeline"]["valid"])
        self.assertEqual(by_name["hybrid_pipeline"]["correct_delta_vs_lightglue"], 661.0)
        self.assertEqual(by_name["hybrid_pipeline"]["fallback_lightglue_rows"], 198)
        self.assertEqual(by_name["pfm_only_gate"]["reject_action"], "zero")
        self.assertEqual(by_name["pfm_only_gate"]["rejected_rows"], 177)

    def test_discover_true_geometry_selector_runs_reads_phase56_summary_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            phase56 = root / "phase56_large_fresh_true_geometry_selector_eval"
            phase56.mkdir()
            (phase56 / "summary.json").write_text(
                """
{
  "manifest_validation": {
    "counts": {"train": 26, "dev": 26, "val": 26, "lockbox": 26},
    "excluded_base_ids": 676,
    "base_disjoint": true
  },
  "comparison": {
    "selector": {
      "rows": 78,
      "selected_matches": 19061,
      "selected_correct": 19061,
      "selected_wrong": 0,
      "selected_precision": 1.0,
      "lightglue_correct": 3615,
      "lightglue_wrong": 38,
      "lightglue_precision": 0.989597591,
      "correct_delta_vs_lightglue": 15446,
      "wrong_delta_vs_lightglue": -38
    },
    "selector_by_split": {
      "dev": {"rows": 26, "correct_delta_vs_lightglue": 5190, "wrong_delta_vs_lightglue": -12},
      "val": {"rows": 26, "correct_delta_vs_lightglue": 4764, "wrong_delta_vs_lightglue": -11},
      "lockbox": {"rows": 26, "correct_delta_vs_lightglue": 5492, "wrong_delta_vs_lightglue": -15}
    }
  }
}
""",
                encoding="utf-8",
            )
            (phase56 / "optimization_audit.json").write_text(
                """
[
  {
    "requirement_id": "true_geometry.selector_fresh_validation",
    "status": "PASS",
    "evidence": "rows=78; correct_delta=15446; wrong_delta=-38",
    "risk": ""
  }
]
""",
                encoding="utf-8",
            )
            (phase56 / "summary.html").write_text("<html><body>phase56 summary</body></html>", encoding="utf-8")
            standalone_eval = root / "phase45_eval"
            standalone_eval.mkdir()
            (standalone_eval / "summary.json").write_text(
                """
{
  "aggregate": {
    "rows": 78,
    "pfm_matches": 18745,
    "pfm_correct": 18745,
    "pfm_wrong": 0,
    "lightglue_correct": 3615,
    "lightglue_wrong": 38,
    "correct_delta_vs_lightglue": 15130,
    "wrong_delta_vs_lightglue": -38
  }
}
""",
                encoding="utf-8",
            )

            selectors = discover_true_geometry_selector_runs(root)

        self.assertEqual(len(selectors), 1)
        selector = selectors[0]
        self.assertEqual(selector["name"], "phase56_large_fresh_true_geometry_selector_eval")
        self.assertEqual(selector["audit_status"], "PASS")
        self.assertTrue(selector["base_disjoint"])
        self.assertEqual(selector["excluded_base_ids"], 676)
        self.assertEqual(selector["rows"], 78)
        self.assertEqual(selector["selected_correct"], 19061)
        self.assertEqual(selector["selected_wrong"], 0)
        self.assertEqual(selector["lightglue_correct"], 3615)
        self.assertEqual(selector["lightglue_wrong"], 38)
        self.assertEqual(selector["correct_delta_vs_lightglue"], 15446)
        self.assertEqual(selector["wrong_delta_vs_lightglue"], -38)
        self.assertEqual(selector["split_counts"]["dev"], 26)
        self.assertEqual(selector["split_counts"]["lockbox"], 26)
        self.assertTrue(selector["report_html"].name.endswith("summary.html"))

    def test_discover_true_geometry_selector_runs_reads_validation_gate_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selector_dir = root / "phase56_selector_with_validation"
            selector_dir.mkdir()
            (selector_dir / "summary.json").write_text(
                """
{
  "manifest_validation": {
    "counts": {"train": 26, "dev": 26, "val": 26, "lockbox": 26},
    "excluded_base_ids": 676,
    "base_disjoint": true
  },
  "comparison": {
    "selector": {
      "rows": 78,
      "selected_correct": 19061,
      "selected_wrong": 0,
      "lightglue_correct": 3615,
      "lightglue_wrong": 38,
      "correct_delta_vs_lightglue": 15446,
      "wrong_delta_vs_lightglue": -38
    },
    "selector_by_split": {
      "dev": {"rows": 26, "correct_delta_vs_lightglue": 5190, "wrong_delta_vs_lightglue": -12},
      "val": {"rows": 26, "correct_delta_vs_lightglue": 4764, "wrong_delta_vs_lightglue": -11},
      "lockbox": {"rows": 26, "correct_delta_vs_lightglue": 5492, "wrong_delta_vs_lightglue": -15}
    }
  }
}
""",
                encoding="utf-8",
            )
            (selector_dir / "true_geometry_selector_validation.json").write_text(
                '{"valid": true, "errors": [], "correct_delta_vs_lightglue": 15446, "wrong_delta_vs_lightglue": -38}\n',
                encoding="utf-8",
            )

            selectors = discover_true_geometry_selector_runs(root)

        self.assertEqual(len(selectors), 1)
        self.assertEqual(selectors[0]["audit_status"], "PASS")
        self.assertEqual(selectors[0]["audit_risk"], "")
        self.assertTrue(str(selectors[0]["validation_json"]).endswith("true_geometry_selector_validation.json"))

    def test_discover_true_geometry_selector_runs_reads_multiseed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selector_dir = root / "phase59_true_geometry_selector_multiseed_eval"
            selector_dir.mkdir()
            (selector_dir / "summary.json").write_text(
                """
{
  "valid": true,
  "errors": [],
  "totals": {
    "rows": 240,
    "selector_correct": 52841,
    "selector_wrong": 0,
    "lightglue_correct": 10167,
    "lightglue_wrong": 149,
    "correct_delta_vs_lightglue": 42674,
    "wrong_delta_vs_lightglue": -149
  },
  "seed_results": [
    {
      "seed": 20260623,
      "valid": true,
      "rows": 78,
      "selector_correct": 17453,
      "selector_wrong": 0,
      "lightglue_correct": 3642,
      "lightglue_wrong": 55,
      "correct_delta_vs_lightglue": 13811,
      "wrong_delta_vs_lightglue": -55,
      "manifest_counts": {"train": 26, "dev": 26, "val": 26, "lockbox": 26},
      "base_disjoint": true,
      "split_results": {
        "dev": {"rows": 26, "correct_delta_vs_lightglue": 4559, "wrong_delta_vs_lightglue": -12},
        "val": {"rows": 26, "correct_delta_vs_lightglue": 4403, "wrong_delta_vs_lightglue": -34},
        "lockbox": {"rows": 26, "correct_delta_vs_lightglue": 4849, "wrong_delta_vs_lightglue": -9}
      }
    },
    {
      "seed": 20260624,
      "valid": true,
      "rows": 78,
      "selector_correct": 18521,
      "selector_wrong": 0,
      "lightglue_correct": 3531,
      "lightglue_wrong": 42,
      "correct_delta_vs_lightglue": 14990,
      "wrong_delta_vs_lightglue": -42,
      "manifest_counts": {"train": 26, "dev": 26, "val": 26, "lockbox": 26},
      "base_disjoint": true,
      "split_results": {
        "dev": {"rows": 26, "correct_delta_vs_lightglue": 5221, "wrong_delta_vs_lightglue": -13},
        "val": {"rows": 26, "correct_delta_vs_lightglue": 4897, "wrong_delta_vs_lightglue": -10},
        "lockbox": {"rows": 26, "correct_delta_vs_lightglue": 4872, "wrong_delta_vs_lightglue": -19}
      }
    },
    {
      "seed": 20260625,
      "valid": true,
      "rows": 84,
      "selector_correct": 16867,
      "selector_wrong": 0,
      "lightglue_correct": 2994,
      "lightglue_wrong": 52,
      "correct_delta_vs_lightglue": 13873,
      "wrong_delta_vs_lightglue": -52,
      "manifest_counts": {"train": 28, "dev": 28, "val": 28, "lockbox": 28},
      "base_disjoint": true,
      "split_results": {
        "dev": {"rows": 28, "correct_delta_vs_lightglue": 3854, "wrong_delta_vs_lightglue": -24},
        "val": {"rows": 28, "correct_delta_vs_lightglue": 4616, "wrong_delta_vs_lightglue": -11},
        "lockbox": {"rows": 28, "correct_delta_vs_lightglue": 5403, "wrong_delta_vs_lightglue": -17}
      }
    }
  ]
}
""",
                encoding="utf-8",
            )
            (selector_dir / "summary.html").write_text("<html><body>phase59 summary</body></html>", encoding="utf-8")

            selectors = discover_true_geometry_selector_runs(root)

        self.assertEqual(len(selectors), 1)
        selector = selectors[0]
        self.assertEqual(selector["name"], "phase59_true_geometry_selector_multiseed_eval")
        self.assertEqual(selector["audit_status"], "PASS")
        self.assertEqual(selector["rows"], 240)
        self.assertEqual(selector["selected_correct"], 52841)
        self.assertEqual(selector["selected_wrong"], 0)
        self.assertEqual(selector["lightglue_correct"], 10167)
        self.assertEqual(selector["lightglue_wrong"], 149)
        self.assertEqual(selector["correct_delta_vs_lightglue"], 42674)
        self.assertEqual(selector["wrong_delta_vs_lightglue"], -149)
        self.assertTrue(selector["base_disjoint"])
        self.assertEqual(selector["split_counts"]["dev"], 80)
        self.assertEqual(selector["split_counts"]["lockbox"], 80)
        self.assertEqual(selector["by_split"]["dev"]["correct_delta_vs_lightglue"], 13634)
        self.assertEqual(selector["by_split"]["dev"]["wrong_delta_vs_lightglue"], -49)


if __name__ == "__main__":
    unittest.main()
