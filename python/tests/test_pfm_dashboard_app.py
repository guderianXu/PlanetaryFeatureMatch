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
                self.assertIn('data-live-chart="no_match"', train_html)
                self.assertIn('data-live-chart="online_false"', train_html)
                self.assertIn("拒配损失", train_html)
                self.assertIn("在线错配", train_html)
                self.assertIn("默认启动 Python 拒配训练", train_html)
                self.assertIn('value="dashboard_python"', train_html)
                self.assertIn('<option value="python" selected>Python 训练</option>', train_html)
                self.assertIn('<option value="cpp">C++ 训练</option>', train_html)
                self.assertIn('name="graph_inference_preset"', train_html)
                self.assertIn('name="graph_min_accept_probability"', train_html)
                self.assertIn('name="graph_max_attention_work_fraction"', train_html)
                self.assertIn('name="graph_width_prune_keep_ratio"', train_html)
                self.assertIn('name="graph_matcher_metadata_mode"', train_html)
                self.assertIn('name="graph_matcher_no_match_points"', train_html)
                self.assertIn('name="graph_matcher_no_match_weight"', train_html)
                self.assertIn('name="graph_matcher_no_match_min_distance"', train_html)
                self.assertIn('name="graph_matcher_assignment_weight"', train_html)
                self.assertIn('name="graph_matcher_accept_weight"', train_html)
                self.assertIn('name="graph_matcher_accept_negative_topk"', train_html)
                self.assertIn('name="graph_matcher_prune_ranking_weight"', train_html)
                self.assertIn('name="graph_matcher_prune_ranking_margin"', train_html)
                self.assertIn('name="graph_matcher_train_max_attention_layers"', train_html)
                self.assertIn('name="graph_matcher_train_random_attention_layers"', train_html)
                self.assertIn('name="graph_matcher_online_false_no_match"', train_html)
                self.assertIn('name="graph_matcher_train_max_attention_work_fraction"', train_html)
                self.assertIn('name="graph_matcher_train_width_keep_ratio"', train_html)
                self.assertIn('name="graph_matcher_stop_confidence_weight"', train_html)
                self.assertIn('name="graph_matcher_stop_confidence_weight" value="0"', train_html)
                self.assertIn('name="graph_matcher_stop_confidence_margin"', train_html)
                self.assertIn('name="graph_matcher_raw_preservation_weight"', train_html)
                self.assertIn('name="graph_matcher_raw_preservation_margin"', train_html)
                self.assertIn('name="graph_matcher_raw_preservation_raw_margin"', train_html)
                self.assertIn('name="graph_matcher_hard_negative_dustbin_weight"', train_html)
                self.assertIn('name="graph_matcher_hard_negative_dustbin_topk"', train_html)
                self.assertIn('name="graph_matcher_hard_negative_dustbin_margin"', train_html)
                self.assertIn('name="graph_matcher_hard_negative_dustbin_spatial_min_distance"', train_html)
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
                    self.assertIn(b"graph_matcher_no_match_loss", dashboard_js)
                    self.assertIn(b"online_false_match_points", dashboard_js)
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
                    "graph_max_attention_work_fraction": "0.55",
                    "graph_width_prune_keep_ratio": "0.4",
                    "graph_matcher_metadata_mode": "no_xy",
                    "graph_matcher_no_match_points": "24",
                    "graph_matcher_no_match_weight": "0.1",
                    "graph_matcher_no_match_min_distance": "6.0",
                    "graph_matcher_assignment_weight": "0.2",
                    "graph_matcher_accept_weight": "0.3",
                    "graph_matcher_accept_negative_topk": "6",
                    "graph_matcher_prune_ranking_weight": "0.04",
                    "graph_matcher_prune_ranking_margin": "0.2",
                    "graph_matcher_train_max_attention_layers": "2",
                    "graph_matcher_train_random_attention_layers": "on",
                    "graph_matcher_online_false_no_match": "on",
                    "graph_matcher_train_max_attention_work_fraction": "0.5",
                    "graph_matcher_train_width_keep_ratio": "0.5",
                    "graph_matcher_raw_preservation_weight": "0.11",
                    "graph_matcher_raw_preservation_margin": "0.9",
                    "graph_matcher_raw_preservation_raw_margin": "0.04",
                    "graph_matcher_hard_negative_dustbin_weight": "0.13",
                    "graph_matcher_hard_negative_dustbin_topk": "5",
                    "graph_matcher_hard_negative_dustbin_margin": "0.3",
                    "graph_matcher_hard_negative_dustbin_spatial_min_distance": "2.5",
                }
                data = urllib.parse.urlencode(form).encode("utf-8")
                urllib.request.urlopen(base + "/train", data=data, timeout=5).read()
                run_dir = next((root / "runs").glob("align_default*"))
                script = (run_dir / "train.sh").read_text(encoding="utf-8")
                report = (run_dir / "run.html").read_text(encoding="utf-8")
                self.assertIn("--training-profile full", script)
                self.assertIn("--train-backbone", script)
                self.assertIn("--train-graph-matcher", script)
                self.assertIn("--graph-matcher-metadata-mode no_xy", script)
                self.assertIn("--graph-matcher-no-match-points 24", script)
                self.assertIn("--graph-matcher-no-match-min-distance 6.0", script)
                self.assertIn("--graph-matcher-accept-weight 0.3", script)
                self.assertIn("--graph-matcher-accept-negative-topk 6", script)
                self.assertIn("--graph-matcher-prune-ranking-weight 0.04", script)
                self.assertIn("--graph-matcher-prune-ranking-margin 0.2", script)
                self.assertIn("--graph-matcher-train-max-attention-layers 2", script)
                self.assertIn("--graph-matcher-train-random-attention-layers", script)
                self.assertIn("--graph-matcher-train-max-attention-work-fraction 0.5", script)
                self.assertIn("--graph-matcher-train-width-keep-ratio 0.5", script)
                self.assertIn("--graph-matcher-raw-preservation-weight 0.11", script)
                self.assertIn("--graph-matcher-raw-preservation-margin 0.9", script)
                self.assertIn("--graph-matcher-raw-preservation-raw-margin 0.04", script)
                self.assertIn("--graph-matcher-hard-negative-dustbin-weight 0.13", script)
                self.assertIn("--graph-matcher-hard-negative-dustbin-topk 5", script)
                self.assertIn("--graph-matcher-hard-negative-dustbin-margin 0.3", script)
                self.assertIn("--graph-matcher-hard-negative-dustbin-spatial-min-distance 2.5", script)
                self.assertIn("graph_max_attention_work_fraction=0.55", report)
                self.assertIn("graph_matcher_metadata_mode=no_xy", report)
                self.assertIn("graph_matcher_no_match_points=24", report)
                self.assertIn("graph_matcher_accept_weight=0.3", report)
                self.assertIn("graph_matcher_prune_ranking_weight=0.04", report)
                self.assertIn("graph_matcher_train_max_attention_layers=2", report)
                self.assertIn("graph_matcher_train_random_attention_layers=True", report)
                self.assertIn("graph_matcher_train_max_attention_work_fraction=0.5", report)
                self.assertIn("graph_matcher_train_width_keep_ratio=0.5", report)
                self.assertIn("graph_matcher_raw_preservation_weight=0.11", report)
                self.assertIn("graph_matcher_raw_preservation_margin=0.9", report)
                self.assertIn("graph_matcher_raw_preservation_raw_margin=0.04", report)
                self.assertIn("graph_matcher_hard_negative_dustbin_weight=0.13", report)
                self.assertIn("graph_matcher_hard_negative_dustbin_topk=5", report)
                self.assertIn("graph_matcher_hard_negative_dustbin_margin=0.3", report)
                self.assertIn("graph_matcher_hard_negative_dustbin_spatial_min_distance=2.5", report)
                self.assertIn("graph_width_prune_keep_ratio=0.4", report)
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

    def test_history_shows_lightglue_graph_efficiency_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs" / "graph_efficiency_run"
            visual = run / "visual_report"
            visual.mkdir(parents=True)
            (run / "metrics.csv").write_text(
                "step,loss,graph_attention_work_fraction,graph_executed_layers,graph_pruned_keypoint_fraction\n"
                "1,2.0,0.60,3,0.25\n"
                "2,1.5,0.40,2,0.50\n",
                encoding="utf-8",
            )
            (visual / "index.html").write_text("<html><body>匹配报告</body></html>", encoding="utf-8")
            (visual / "match_visual_summary.csv").write_text(
                "pair_pt,matches,correct,wrong,precision,graph_executed_layers,"
                "graph_attention_work_fraction,graph_pruned_keypoint_fraction\n"
                "a.pt,20,18,2,0.9,2,0.5,0.4\n",
                encoding="utf-8",
            )
            server = make_server("127.0.0.1", 0, project_root=root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                with urllib.request.urlopen(base + "/history?run=graph_efficiency_run", timeout=5) as response:
                    history_html = response.read().decode("utf-8")

                self.assertIn("LightGlue 自适应推理", history_html)
                self.assertIn("平均计算量占比", history_html)
                self.assertIn("50.0%", history_html)
                self.assertIn("平均执行层数", history_html)
                self.assertIn("2.50", history_html)
                self.assertIn("平均剪枝比例", history_html)
                self.assertIn("37.5%", history_html)
                self.assertIn("计算量占比", history_html)
                self.assertIn("执行层数", history_html)
                self.assertIn("剪枝比例", history_html)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
