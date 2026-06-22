import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class RunObservableLightGlueGatePipelineTest(unittest.TestCase):
    def write_dataset(self, path: Path, *, split: str, keep_correct: int, lightglue_keep_correct: int) -> None:
        rows = [
            {
                "source_name": "unit",
                "split": split,
                "pair_index": "0",
                "pair_type": "same_position_view",
                "base_id": f"{split}_keep",
                "reference_variant": "nadir",
                "target_variant": "mid_01",
                "reject_label": "0",
                "pfm_matches": str(keep_correct),
                "pfm_correct": str(keep_correct),
                "pfm_wrong": "0",
                "pfm_precision": "1.0",
                "lightglue_matches": str(lightglue_keep_correct),
                "lightglue_correct": str(lightglue_keep_correct),
                "lightglue_wrong": "0",
                "lightglue_precision": "1.0",
                "feature_score_min": "3.0",
            },
            {
                "source_name": "unit",
                "split": split,
                "pair_index": "1",
                "pair_type": "same_position_view",
                "base_id": f"{split}_fallback",
                "reference_variant": "nadir",
                "target_variant": "extreme_03",
                "reject_label": "1",
                "pfm_matches": "20",
                "pfm_correct": "10",
                "pfm_wrong": "10",
                "pfm_precision": "0.5",
                "lightglue_matches": "8",
                "lightglue_correct": "8",
                "lightglue_wrong": "0",
                "lightglue_precision": "1.0",
                "feature_score_min": "12.0",
            },
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_match_details(self, path: Path, *, keep_base_id: str, target_variant: str, correct: int) -> None:
        fieldnames = ["pair_index", "base_id", "target_variant", "correct"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for _ in range(correct):
                writer.writerow(
                    {
                        "pair_index": "0",
                        "base_id": keep_base_id,
                        "target_variant": target_variant,
                        "correct": "1",
                    }
                )

    def test_cli_applies_observable_gate_across_sources_and_writes_aggregate_audit(self) -> None:
        import run_observable_lightglue_gate_pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sources = []
            for split, keep_correct, lightglue_keep_correct in [
                ("dev_val", 100, 40),
                ("fresh6", 70, 30),
            ]:
                dataset_csv = root / f"{split}_dataset.csv"
                details_csv = root / f"{split}_details.csv"
                self.write_dataset(
                    dataset_csv,
                    split=split,
                    keep_correct=keep_correct,
                    lightglue_keep_correct=lightglue_keep_correct,
                )
                self.write_match_details(
                    details_csv,
                    keep_base_id=f"{split}_keep",
                    target_variant="mid_01",
                    correct=keep_correct,
                )
                sources.append(f"{split},{dataset_csv},{details_csv}")
            output_dir = root / "observable_pipeline"

            exit_code = pipeline.main(
                [
                    "--source",
                    sources[0],
                    "--source",
                    sources[1],
                    "--gate",
                    "feature_score_min <= 5.0",
                    "--project-root",
                    str(ROOT),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected_files = [
                "aggregate_summary.json",
                "aggregate_validation.json",
                "per_split_validation.csv",
                "optimization_audit.json",
                "optimization_audit.html",
                "pipeline_summary.json",
                "index.html",
                "dev_val/summary.json",
                "fresh6/summary.json",
            ]
            for relative_path in expected_files:
                self.assertTrue((output_dir / relative_path).exists(), relative_path)

            aggregate = json.loads((output_dir / "aggregate_validation.json").read_text(encoding="utf-8"))
            self.assertTrue(aggregate["valid"])
            self.assertEqual(aggregate["correct_delta_vs_lightglue"], 100)
            self.assertEqual(aggregate["wrong_delta_vs_lightglue"], 0)
            self.assertEqual(len(aggregate["splits"]), 2)

            pipeline_summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(pipeline_summary["valid"])
            self.assertEqual(pipeline_summary["correct_delta_vs_lightglue"], 100)
            self.assertEqual(pipeline_summary["wrong_delta_vs_lightglue"], 0)
            self.assertIn("aggregate_summary.json", pipeline_summary["hybrid_summary_json"])

            audit_items = json.loads((output_dir / "optimization_audit.json").read_text(encoding="utf-8"))
            by_id = {item["requirement_id"]: item for item in audit_items}
            self.assertEqual(by_id["hybrid.lightglue_gate_validation"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
