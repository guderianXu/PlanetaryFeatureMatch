import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class RunTrueGeometrySelectorPipelineTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_pair_manifest(self, path: Path) -> None:
        self.write_csv(
            path,
            [
                {
                    "pair_index": "0",
                    "split": "dev",
                    "pair_type": "cross_camera",
                    "reference_base_id": "base_a",
                    "reference_variant": "extreme_01",
                    "target_base_id": "base_b",
                    "target_variant": "extreme_03",
                    "valid_fraction": "0.40",
                }
            ],
        )

    def write_lightglue_metrics(self, path: Path, *, correct: int, wrong: int = 0) -> None:
        self.write_csv(
            path,
            [
                {
                    "label": "LightGlue-SIFT-MAGSAC-min16",
                    "pair_index": "0",
                    "manifest_pair_index": "0",
                    "matches": str(correct + wrong),
                    "correct": str(correct),
                    "wrong": str(wrong),
                }
            ],
        )

    def write_candidate_report(self, root: Path, *, name: str, matches: int, wrong: int = 0) -> None:
        report_dir = root / name / "dev" / "pfm_eval"
        self.write_csv(
            report_dir / "all_filtered_summary.csv",
            [
                {
                    "label": "PFM / all-filtered",
                    "base_id": "base_a",
                    "target_variant": "extreme_03",
                    "split": "dev",
                    "valid_fraction": "0.40",
                    "matches": str(matches),
                    "correct": str(matches - wrong),
                    "wrong": str(wrong),
                    "precision": "1.000000" if wrong == 0 else "0.800000",
                    "score_mean": str(20 + matches),
                }
            ],
        )
        detail_rows = [
            {
                "label": "PFM / all-filtered",
                "pair_index": "0",
                "base_id": "base_a",
                "reference_variant": "extreme_01",
                "target_variant": "extreme_03",
                "split": "dev",
                "match_index": str(index),
                "point_a_x_px": "1.0",
                "point_a_y_px": "2.0",
                "point_b_x_px": "3.0",
                "point_b_y_px": "4.0",
                "score": str(100 + index),
                "correct": "1",
                "error_px": "1.5",
                "valid_fraction": "0.40",
            }
            for index in range(matches)
        ]
        self.write_csv(report_dir / "all_filtered_match_details.csv", detail_rows)

    def prepare_fixture(self, root: Path, *, lightglue_correct: int, candidate_b_matches: int) -> tuple[Path, Path]:
        pair_manifest = root / "manifests" / "dev_pairs.csv"
        lightglue_metrics = root / "lightglue" / "lightglue_sift_metrics.csv"
        self.write_pair_manifest(pair_manifest)
        self.write_lightglue_metrics(lightglue_metrics, correct=lightglue_correct)
        self.write_candidate_report(root, name="phase45", matches=4)
        self.write_candidate_report(root, name="phase49b", matches=candidate_b_matches)
        return pair_manifest, lightglue_metrics

    def pipeline_args(self, root: Path, pair_manifest: Path, lightglue_metrics: Path, output_root: Path) -> list[str]:
        return [
            "--source",
            f"dev,{pair_manifest},{lightglue_metrics}",
            "--candidate",
            f"phase45,{root / 'phase45'},pfm_eval",
            "--candidate",
            f"phase49b,{root / 'phase49b'},pfm_eval",
            "--output-root",
            str(output_root),
            "--validation-output-json",
            str(output_root / "true_geometry_selector_validation.json"),
            "--validation-output-html",
            str(output_root / "true_geometry_selector_validation.html"),
            "--no-require-base-disjoint",
        ]

    def test_dry_run_expands_visual_selector_validation_and_audit_steps(self) -> None:
        import run_true_geometry_selector_pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pair_manifest, lightglue_metrics = self.prepare_fixture(
                root,
                lightglue_correct=2,
                candidate_b_matches=9,
            )
            output_root = root / "pipeline"

            exit_code = pipeline.main(
                [
                    *self.pipeline_args(root, pair_manifest, lightglue_metrics, output_root),
                    "--required-variant",
                    "extreme_03",
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "dry_run")
            step_names = [step["name"] for step in summary["steps"]]
            self.assertIn("visual_eval_input_check", step_names)
            command_text = "\n".join(" ".join(step.get("command", [])) for step in summary["steps"])
            self.assertIn("visualize_lazy_pose_matches.py", command_text)
            self.assertIn("select_true_geometry_pair_reports.py", command_text)
            self.assertIn("validate_true_geometry_selector.py", command_text)
            self.assertIn("--required-variant extreme_03", command_text)
            self.assertIn("audit_pfm_optimization_goal.py", command_text)

    def test_validation_failure_exits_nonzero_and_records_failure_reason(self) -> None:
        import run_true_geometry_selector_pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pair_manifest, lightglue_metrics = self.prepare_fixture(
                root,
                lightglue_correct=12,
                candidate_b_matches=6,
            )
            output_root = root / "pipeline"

            exit_code = pipeline.main(self.pipeline_args(root, pair_manifest, lightglue_metrics, output_root))

            self.assertEqual(exit_code, 1)
            validation = json.loads((output_root / "true_geometry_selector_validation.json").read_text(encoding="utf-8"))
            self.assertFalse(validation["valid"])
            self.assertIn("correct_delta_below_minimum", validation["errors"])
            html = (output_root / "summary.html").read_text(encoding="utf-8")
            self.assertIn("validation_failed", html)
            self.assertIn("correct_delta_below_minimum", html)

    def test_success_writes_summary_selector_validation_and_audit_outputs(self) -> None:
        import run_true_geometry_selector_pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pair_manifest, lightglue_metrics = self.prepare_fixture(
                root,
                lightglue_correct=2,
                candidate_b_matches=9,
            )
            output_root = root / "pipeline"

            exit_code = pipeline.main(
                [
                    *self.pipeline_args(root, pair_manifest, lightglue_metrics, output_root),
                    "--required-variant",
                    "extreme_03",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "valid")
            self.assertTrue(summary["validation"]["valid"])
            self.assertEqual(summary["validation"]["correct_delta_vs_lightglue"], 7)
            self.assertEqual(summary["validation"]["variant_results"]["extreme_03"]["correct_delta_vs_lightglue"], 7)
            self.assertTrue((output_root / "summary.html").is_file())
            self.assertTrue((output_root / "selector" / "pair_selection.csv").is_file())
            self.assertTrue((output_root / "true_geometry_selector_validation.json").is_file())
            self.assertTrue((output_root / "true_geometry_selector_validation.html").is_file())
            self.assertTrue((output_root / "optimization_audit.json").is_file())
            self.assertTrue((output_root / "optimization_audit.html").is_file())


if __name__ == "__main__":
    unittest.main()
