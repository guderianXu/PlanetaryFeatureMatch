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
                self.assertIn("C++ 的完整训练定义已经默认与 Python 对齐", train_html)
                self.assertNotIn('name="align_python_compare"', train_html)
                self.assertNotIn('name="profile"', train_html)
                with urllib.request.urlopen(base + "/api/runs", timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["runs"][0]["name"], "python_sample")
                self.assertEqual(payload["runs"][0]["progress_label"], "1/2 步")
                with urllib.request.urlopen(base + "/static/dashboard.css", timeout=5) as response:
                    self.assertEqual(response.headers.get_content_type(), "text/css")
                    self.assertIn(b"--bg: #0b1015", response.read())
                with urllib.request.urlopen(base + "/static/dashboard.js", timeout=5) as response:
                    self.assertEqual(response.headers.get_content_type(), "application/javascript")
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
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
