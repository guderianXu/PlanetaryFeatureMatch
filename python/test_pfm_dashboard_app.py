import json
import tempfile
import threading
import unittest
import urllib.parse
import urllib.request
from pathlib import Path

from pfm_dashboard.app import make_server


class DashboardAppTest(unittest.TestCase):
    def test_dashboard_routes_respond(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs" / "python_sample"
            run.mkdir(parents=True)
            (run / "metrics.csv").write_text("step,loss\n1,2.5\n", encoding="utf-8")
            (run / "train.sh").write_text("#!/usr/bin/env bash\npython train.py --steps 2\n", encoding="utf-8")
            server = make_server("127.0.0.1", 0, project_root=root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                for path in ["/", "/train", "/runs", "/compare", "/datasets"]:
                    with urllib.request.urlopen(base + path, timeout=5) as response:
                        self.assertEqual(response.status, 200)
                        self.assertIn(b"PFM Lab", response.read())
                with urllib.request.urlopen(base + "/train", timeout=5) as response:
                    train_html = response.read().decode("utf-8")
                self.assertIn("实时训练进度", train_html)
                self.assertIn('data-live-training', train_html)
                self.assertIn('data-live-epoch', train_html)
                self.assertIn('data-live-batch', train_html)
                self.assertIn('class="live-chart-section"', train_html)
                self.assertIn("训练指标曲线", train_html)
                self.assertIn("浅线：每 batch 原始值", train_html)
                self.assertIn("亮线：平滑趋势", train_html)
                self.assertIn("圆点：当前 batch", train_html)
                self.assertIn('data-live-chart="loss"', train_html)
                self.assertIn("C++ 的完整训练定义已经默认与 Python 对齐", train_html)
                self.assertIn('value="dashboard_cpp"', train_html)
                self.assertIn('<option value="cpp">C++ 训练</option>', train_html)
                self.assertIn('name="graph_inference_preset"', train_html)
                self.assertIn("LightGlue 快速剪枝", train_html)
                self.assertNotIn("Python + C++ 对比", train_html)
                self.assertNotIn('name="align_python_compare"', train_html)
                self.assertNotIn('name="profile"', train_html)
                with urllib.request.urlopen(base + "/api/runs", timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["runs"][0]["name"], "python_sample")
                self.assertEqual(payload["runs"][0]["progress_label"], "1/2 步")
                with urllib.request.urlopen(base + "/api/metrics?runs=python_sample", timeout=5) as response:
                    metrics = json.loads(response.read().decode("utf-8"))
                self.assertEqual(metrics["metrics"]["python_sample"]["rows"][0]["loss"], 2.5)
                with urllib.request.urlopen(base + "/static/dashboard.css", timeout=5) as response:
                    self.assertEqual(response.headers.get_content_type(), "text/css")
                    dashboard_css = response.read()
                    self.assertIn(b"--bg: #0b1015", dashboard_css)
                    self.assertIn(b".live-chart-section", dashboard_css)
                    self.assertIn(b"max-height: 440px", dashboard_css)
                    self.assertIn(b"grid-template-columns: repeat(2, minmax(0, 1fr))", dashboard_css)
                with urllib.request.urlopen(base + "/static/dashboard.js", timeout=5) as response:
                    self.assertEqual(response.headers.get_content_type(), "application/javascript")
                    dashboard_js = response.read()
                    self.assertIn(b"installLiveTraining", dashboard_js)
                    self.assertIn(b"loss_total", dashboard_js)
                    self.assertIn(b"LIVE_CHART_WINDOW_BATCHES = 300", dashboard_js)
                    self.assertIn(b"liveChartRun", dashboard_js)
                    self.assertIn(b"updateChartMeta", dashboard_js)
                    self.assertIn(b"movingAveragePoints", dashboard_js)
                    self.assertIn("当前 ".encode("utf-8"), dashboard_js)
                    self.assertIn("平滑 ".encode("utf-8"), dashboard_js)
                    self.assertIn(b"live-chart-line-smooth", dashboard_js)
                    self.assertIn(b"runEpochLabel", dashboard_js)
                    self.assertIn(b"runBatchLabel", dashboard_js)
                with urllib.request.urlopen(base + "/runs", timeout=5) as response:
                    runs_html = response.read()
                self.assertIn("进度".encode("utf-8"), runs_html)
                self.assertIn("控制".encode("utf-8"), runs_html)
                self.assertIn("开始".encode("utf-8"), runs_html)
                self.assertIn("创建时间".encode("utf-8"), runs_html)
                self.assertIn("完成时间".encode("utf-8"), runs_html)
                self.assertIn("删除".encode("utf-8"), runs_html)
            finally:
                server.shutdown()
                server.server_close()

    def test_train_post_uses_full_cpp_profile_without_alignment_checkbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache" / "train"
            cache.mkdir(parents=True)
            server = make_server("127.0.0.1", 0, project_root=root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                form = {
                    "experiment_name": "align_default",
                    "backend": "cpp",
                    "cache_dirs": str(cache),
                }
                data = urllib.parse.urlencode(form).encode("utf-8")
                urllib.request.urlopen(base + "/train", data=data, timeout=5).read()
                script = next((root / "runs").glob("align_default*/train.sh")).read_text(encoding="utf-8")
                self.assertIn("--training-profile full", script)
                self.assertIn("--train-backbone", script)
                self.assertIn("--train-graph-matcher", script)
            finally:
                server.shutdown()
                server.server_close()

    def test_history_discovers_midstep_visual_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs" / "lazy_midstep_run"
            report = run / "midstep_3000_visual_report"
            report.mkdir(parents=True)
            (run / "train_metrics.csv").write_text("step,loss\n3000,1.5\n", encoding="utf-8")
            (report / "index.html").write_text("<html><body>中期匹配报告</body></html>", encoding="utf-8")
            (report / "summary.csv").write_text(
                "label,target_variant,matches,correct,wrong,precision\n"
                "filtered,extreme_01,20,18,2,0.9\n",
                encoding="utf-8",
            )
            server = make_server("127.0.0.1", 0, project_root=root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                with urllib.request.urlopen(base + "/history?run=lazy_midstep_run", timeout=5) as response:
                    history_html = response.read().decode("utf-8")
                self.assertIn("有图", history_html)
                self.assertIn("filtered", history_html)
                self.assertIn("extreme_01", history_html)
                with urllib.request.urlopen(base + "/runs/lazy_midstep_run/visual-report", timeout=5) as response:
                    report_html = response.read().decode("utf-8")
                self.assertIn("中期匹配报告", report_html)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
