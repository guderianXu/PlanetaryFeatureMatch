import json
import tempfile
import threading
import unittest
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
                with urllib.request.urlopen(base + "/api/runs", timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["runs"][0]["name"], "python_sample")
                self.assertEqual(payload["runs"][0]["progress_label"], "1/2 steps")
                with urllib.request.urlopen(base + "/runs", timeout=5) as response:
                    runs_html = response.read()
                self.assertIn(b"Progress", runs_html)
                self.assertIn(b"Control", runs_html)
                self.assertIn(b"Start", runs_html)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
