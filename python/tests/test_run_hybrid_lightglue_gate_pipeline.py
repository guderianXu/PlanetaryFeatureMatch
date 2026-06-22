import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class RunHybridLightGlueGatePipelineTest(unittest.TestCase):
    def write_dataset(self, path: Path) -> None:
        rows = [
            {
                "source_name": "unit",
                "split": "all",
                "pair_index": "0",
                "pair_type": "same_position_view",
                "base_id": "keep_pair",
                "reference_variant": "nadir",
                "target_variant": "mid_01",
                "reject_label": "0",
                "pfm_matches": "100",
                "pfm_correct": "100",
                "pfm_wrong": "0",
                "pfm_precision": "1.0",
                "lightglue_matches": "40",
                "lightglue_correct": "40",
                "lightglue_wrong": "0",
                "lightglue_precision": "1.0",
                "feature_badness": "-5.0",
            },
            {
                "source_name": "unit",
                "split": "all",
                "pair_index": "1",
                "pair_type": "same_position_view",
                "base_id": "fallback_pair",
                "reference_variant": "nadir",
                "target_variant": "extreme_03",
                "reject_label": "1",
                "pfm_matches": "90",
                "pfm_correct": "80",
                "pfm_wrong": "10",
                "pfm_precision": "0.888889",
                "lightglue_matches": "50",
                "lightglue_correct": "50",
                "lightglue_wrong": "0",
                "lightglue_precision": "1.0",
                "feature_badness": "5.0",
            },
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_model(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "standardized_logistic_regression",
                    "feature_columns": ["feature_badness"],
                    "means": [0.0],
                    "scales": [1.0],
                    "weights": [1.0],
                    "bias": 0.0,
                    "label_column": "reject_label",
                    "threshold": 0.5,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_cli_runs_apply_validate_and_audit_outputs(self) -> None:
        import run_hybrid_lightglue_gate_pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_csv = root / "rejection_dataset.csv"
            model_json = root / "model.json"
            output_dir = root / "hybrid_pipeline"
            self.write_dataset(dataset_csv)
            self.write_model(model_json)

            exit_code = pipeline.main(
                [
                    "--dataset-csv",
                    str(dataset_csv),
                    "--model-json",
                    str(model_json),
                    "--project-root",
                    str(ROOT),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected_files = [
                "hybrid_summary.csv",
                "summary.json",
                "application.html",
                "validation.json",
                "validation.html",
                "optimization_audit.json",
                "optimization_audit.html",
                "pipeline_summary.json",
                "index.html",
            ]
            for relative_path in expected_files:
                self.assertTrue((output_dir / relative_path).exists(), relative_path)

            validation = json.loads((output_dir / "validation.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["correct_delta_vs_lightglue"], 60)
            self.assertEqual(validation["wrong_delta_vs_lightglue"], 0)

            audit_items = json.loads((output_dir / "optimization_audit.json").read_text(encoding="utf-8"))
            by_id = {item["requirement_id"]: item for item in audit_items}
            self.assertEqual(by_id["hybrid.lightglue_gate_validation"]["status"], "PASS")

            pipeline_summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(pipeline_summary["valid"])
            self.assertEqual(pipeline_summary["validation_exit_code"], 0)
            self.assertIn("Hybrid LightGlue gate pipeline", (output_dir / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
