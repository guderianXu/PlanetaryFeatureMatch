import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class AuditPfmOptimizationGoalTest(unittest.TestCase):
    def test_audit_accepts_true_geometry_selector_fresh_validation(self) -> None:
        import audit_pfm_optimization_goal as audit

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selector_summary = root / "selector_summary.json"
            selector_summary.write_text(
                json.dumps(
                    {
                        "aggregate": {
                            "rows": 78,
                            "selected_matches": 19061,
                            "selected_correct": 19061,
                            "selected_wrong": 0,
                            "selected_precision": 1.0,
                            "lightglue_matches": 3653,
                            "lightglue_correct": 3615,
                            "lightglue_wrong": 38,
                            "lightglue_precision": 0.989597591,
                            "correct_delta_vs_lightglue": 15446,
                            "wrong_delta_vs_lightglue": -38,
                        },
                        "by_split": {
                            "dev": {
                                "rows": 26,
                                "correct_delta_vs_lightglue": 5190,
                                "wrong_delta_vs_lightglue": -12,
                            },
                            "val": {
                                "rows": 26,
                                "correct_delta_vs_lightglue": 4764,
                                "wrong_delta_vs_lightglue": -11,
                            },
                            "lockbox": {
                                "rows": 26,
                                "correct_delta_vs_lightglue": 5492,
                                "wrong_delta_vs_lightglue": -15,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest_validation = root / "fresh_manifest_validation.json"
            manifest_validation.write_text(
                json.dumps(
                    {
                        "counts": {"train": 26, "dev": 26, "val": 26, "lockbox": 26},
                        "excluded_base_ids": 676,
                        "base_disjoint": True,
                    }
                ),
                encoding="utf-8",
            )

            items = audit.audit_goal(
                project_root=root,
                true_geometry_selector_summary_json=selector_summary,
                true_geometry_manifest_validation_json=manifest_validation,
            )
            by_id = {item.requirement_id: item for item in items}

            self.assertEqual(by_id["true_geometry.selector_fresh_validation"].status, "PASS")
            self.assertIn("rows=78", by_id["true_geometry.selector_fresh_validation"].evidence)
            self.assertIn("correct_delta=15446", by_id["true_geometry.selector_fresh_validation"].evidence)
            self.assertIn("wrong_delta=-38", by_id["true_geometry.selector_fresh_validation"].evidence)
            self.assertIn("base_disjoint=True", by_id["true_geometry.selector_fresh_validation"].evidence)
            self.assertEqual(by_id["true_geometry.selector_fresh_validation"].risk, "")


if __name__ == "__main__":
    unittest.main()
