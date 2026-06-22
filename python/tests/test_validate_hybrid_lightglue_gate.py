import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class ValidateHybridLightGlueGateTest(unittest.TestCase):
    def write_summary(self, path: Path, *, hybrid_correct: int, hybrid_wrong: int) -> None:
        path.write_text(
            json.dumps(
                {
                    "rows": 2,
                    "lightglue_correct": 100,
                    "lightglue_wrong": 2,
                    "lightglue_precision": 100 / 102,
                    "hybrid_correct": hybrid_correct,
                    "hybrid_wrong": hybrid_wrong,
                    "hybrid_precision": hybrid_correct / (hybrid_correct + hybrid_wrong),
                    "hybrid_correct_delta_vs_lightglue": hybrid_correct - 100,
                    "hybrid_wrong_delta_vs_lightglue": hybrid_wrong - 2,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_cli_accepts_hybrid_that_beats_lightglue_without_wrong_increase(self) -> None:
        import validate_hybrid_lightglue_gate as gate

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_json = root / "summary.json"
            output_json = root / "validation.json"
            output_html = root / "validation.html"
            self.write_summary(summary_json, hybrid_correct=105, hybrid_wrong=2)

            exit_code = gate.main(
                [
                    "--summary-json",
                    str(summary_json),
                    "--output-json",
                    str(output_json),
                    "--output-html",
                    str(output_html),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["correct_delta_vs_lightglue"], 5)
            self.assertEqual(payload["wrong_delta_vs_lightglue"], 0)
            self.assertEqual(payload["errors"], [])
            self.assertIn("Hybrid LightGlue gate validation", output_html.read_text(encoding="utf-8"))

    def test_cli_rejects_hybrid_with_added_wrong_matches(self) -> None:
        import validate_hybrid_lightglue_gate as gate

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_json = root / "summary.json"
            output_json = root / "validation.json"
            output_html = root / "validation.html"
            self.write_summary(summary_json, hybrid_correct=108, hybrid_wrong=4)

            exit_code = gate.main(
                [
                    "--summary-json",
                    str(summary_json),
                    "--output-json",
                    str(output_json),
                    "--output-html",
                    str(output_html),
                ]
            )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertFalse(payload["valid"])
            self.assertIn("wrong_delta_exceeds_limit", payload["errors"])


if __name__ == "__main__":
    unittest.main()
