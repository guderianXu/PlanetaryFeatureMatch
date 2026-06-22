import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class ValidateTrueGeometrySelectorTest(unittest.TestCase):
    def write_summary(
        self,
        path: Path,
        *,
        selected_correct: int = 19061,
        selected_wrong: int = 0,
        lightglue_correct: int = 3615,
        lightglue_wrong: int = 38,
        lockbox_wrong_delta: int = -15,
    ) -> None:
        selected_matches = selected_correct + selected_wrong
        lightglue_matches = lightglue_correct + lightglue_wrong
        path.write_text(
            json.dumps(
                {
                    "comparison": {
                        "selector": {
                            "rows": 78,
                            "selected_matches": selected_matches,
                            "selected_correct": selected_correct,
                            "selected_wrong": selected_wrong,
                            "selected_precision": selected_correct / selected_matches,
                            "lightglue_matches": lightglue_matches,
                            "lightglue_correct": lightglue_correct,
                            "lightglue_wrong": lightglue_wrong,
                            "lightglue_precision": lightglue_correct / lightglue_matches,
                            "correct_delta_vs_lightglue": selected_correct - lightglue_correct,
                            "wrong_delta_vs_lightglue": selected_wrong - lightglue_wrong,
                        },
                        "selector_by_split": {
                            "dev": {
                                "rows": 26,
                                "selected_correct": 6366,
                                "selected_wrong": 0,
                                "lightglue_correct": 1176,
                                "lightglue_wrong": 12,
                                "correct_delta_vs_lightglue": 5190,
                                "wrong_delta_vs_lightglue": -12,
                            },
                            "val": {
                                "rows": 26,
                                "selected_correct": 5924,
                                "selected_wrong": 0,
                                "lightglue_correct": 1160,
                                "lightglue_wrong": 11,
                                "correct_delta_vs_lightglue": 4764,
                                "wrong_delta_vs_lightglue": -11,
                            },
                            "lockbox": {
                                "rows": 26,
                                "selected_correct": 6771,
                                "selected_wrong": max(0, 15 + lockbox_wrong_delta),
                                "lightglue_correct": 1279,
                                "lightglue_wrong": 15,
                                "correct_delta_vs_lightglue": 5492,
                                "wrong_delta_vs_lightglue": lockbox_wrong_delta,
                            },
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

    def write_manifest(self, path: Path, *, base_disjoint: bool = True) -> None:
        path.write_text(
            json.dumps(
                {
                    "counts": {"train": 26, "dev": 26, "val": 26, "lockbox": 26},
                    "excluded_base_ids": 676,
                    "base_disjoint": base_disjoint,
                }
            ),
            encoding="utf-8",
        )

    def test_cli_accepts_selector_that_beats_lightglue_on_fresh_splits(self) -> None:
        import validate_true_geometry_selector as validator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_json = root / "summary.json"
            manifest_json = root / "fresh_manifest_validation.json"
            output_json = root / "validation.json"
            output_html = root / "validation.html"
            self.write_summary(summary_json)
            self.write_manifest(manifest_json)

            exit_code = validator.main(
                [
                    "--summary-json",
                    str(summary_json),
                    "--manifest-validation-json",
                    str(manifest_json),
                    "--output-json",
                    str(output_json),
                    "--output-html",
                    str(output_html),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["correct_delta_vs_lightglue"], 15446)
            self.assertEqual(payload["wrong_delta_vs_lightglue"], -38)
            self.assertEqual(payload["errors"], [])
            self.assertTrue(payload["base_disjoint"])
            self.assertIn("True Geometry Selector validation", output_html.read_text(encoding="utf-8"))

    def test_cli_rejects_selector_with_wrong_increase_or_nonfresh_manifest(self) -> None:
        import validate_true_geometry_selector as validator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_json = root / "summary.json"
            manifest_json = root / "fresh_manifest_validation.json"
            output_json = root / "validation.json"
            output_html = root / "validation.html"
            self.write_summary(summary_json, lockbox_wrong_delta=1)
            self.write_manifest(manifest_json, base_disjoint=False)

            exit_code = validator.main(
                [
                    "--summary-json",
                    str(summary_json),
                    "--manifest-validation-json",
                    str(manifest_json),
                    "--output-json",
                    str(output_json),
                    "--output-html",
                    str(output_html),
                ]
            )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertFalse(payload["valid"])
            self.assertIn("fresh_manifest_not_base_disjoint", payload["errors"])
            self.assertIn("lockbox_wrong_delta_exceeds_limit", payload["errors"])

    def test_cli_allows_selector_only_summary_when_base_disjoint_check_is_disabled(self) -> None:
        import validate_true_geometry_selector as validator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_json = root / "selector_summary.json"
            output_json = root / "validation.json"
            output_html = root / "validation.html"
            summary_json.write_text(
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
                            "dev": {"rows": 26, "correct_delta_vs_lightglue": 5190, "wrong_delta_vs_lightglue": -12},
                            "val": {"rows": 26, "correct_delta_vs_lightglue": 4764, "wrong_delta_vs_lightglue": -11},
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

            exit_code = validator.main(
                [
                    "--summary-json",
                    str(summary_json),
                    "--no-require-base-disjoint",
                    "--output-json",
                    str(output_json),
                    "--output-html",
                    str(output_html),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["errors"], [])

    def test_cli_required_split_override_does_not_append_default_splits(self) -> None:
        import validate_true_geometry_selector as validator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_json = root / "selector_summary.json"
            output_json = root / "validation.json"
            output_html = root / "validation.html"
            summary_json.write_text(
                json.dumps(
                    {
                        "aggregate": {
                            "rows": 1,
                            "selected_matches": 9,
                            "selected_correct": 9,
                            "selected_wrong": 0,
                            "selected_precision": 1.0,
                            "lightglue_matches": 2,
                            "lightglue_correct": 2,
                            "lightglue_wrong": 0,
                            "lightglue_precision": 1.0,
                            "correct_delta_vs_lightglue": 7,
                            "wrong_delta_vs_lightglue": 0,
                        },
                        "by_split": {
                            "dev": {
                                "rows": 1,
                                "correct_delta_vs_lightglue": 7,
                                "wrong_delta_vs_lightglue": 0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            exit_code = validator.main(
                [
                    "--summary-json",
                    str(summary_json),
                    "--required-split",
                    "dev",
                    "--no-require-base-disjoint",
                    "--output-json",
                    str(output_json),
                    "--output-html",
                    str(output_html),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["required_splits"], ["dev"])
            self.assertTrue(payload["valid"])

    def test_cli_can_require_variant_level_lightglue_improvement(self) -> None:
        import validate_true_geometry_selector as validator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_json = root / "selector_summary.json"
            output_json = root / "validation.json"
            output_html = root / "validation.html"
            summary_json.write_text(
                json.dumps(
                    {
                        "aggregate": {
                            "rows": 2,
                            "pfm_matches": 31,
                            "pfm_correct": 30,
                            "pfm_wrong": 1,
                            "pfm_precision": 30 / 31,
                            "lightglue_matches": 18,
                            "lightglue_correct": 15,
                            "lightglue_wrong": 3,
                            "lightglue_precision": 15 / 18,
                            "correct_delta_vs_lightglue": 15,
                            "wrong_delta_vs_lightglue": -2,
                        },
                        "by_split": {
                            "dev": {
                                "rows": 2,
                                "correct_delta_vs_lightglue": 15,
                                "wrong_delta_vs_lightglue": -2,
                            },
                        },
                        "by_variant": {
                            "extreme_01": {
                                "rows": 1,
                                "pfm_correct": 20,
                                "pfm_wrong": 0,
                                "lightglue_correct": 10,
                                "lightglue_wrong": 2,
                            },
                            "extreme_02": {
                                "rows": 1,
                                "pfm_correct": 10,
                                "pfm_wrong": 1,
                                "lightglue_correct": 5,
                                "lightglue_wrong": 1,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            exit_code = validator.main(
                [
                    "--summary-json",
                    str(summary_json),
                    "--required-split",
                    "dev",
                    "--required-variant",
                    "extreme_01",
                    "--required-variant",
                    "extreme_02",
                    "--max-wrong-delta-vs-lightglue",
                    "-1",
                    "--no-require-base-disjoint",
                    "--output-json",
                    str(output_json),
                    "--output-html",
                    str(output_html),
                ]
            )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertFalse(payload["valid"])
            self.assertEqual(payload["required_variants"], ["extreme_01", "extreme_02"])
            self.assertEqual(payload["variant_results"]["extreme_01"]["correct_delta_vs_lightglue"], 10)
            self.assertEqual(payload["variant_results"]["extreme_01"]["wrong_delta_vs_lightglue"], -2)
            self.assertEqual(payload["variant_results"]["extreme_02"]["correct_delta_vs_lightglue"], 5)
            self.assertEqual(payload["variant_results"]["extreme_02"]["wrong_delta_vs_lightglue"], 0)
            self.assertIn("extreme_02_wrong_delta_exceeds_limit", payload["errors"])

    def test_cli_accepts_multiseed_selector_summary(self) -> None:
        import validate_true_geometry_selector as validator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_json = root / "phase59_summary.json"
            output_json = root / "validation.json"
            output_html = root / "validation.html"
            summary_json.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "errors": [],
                        "totals": {
                            "rows": 240,
                            "selector_correct": 52841,
                            "selector_wrong": 0,
                            "lightglue_correct": 10167,
                            "lightglue_wrong": 149,
                            "correct_delta_vs_lightglue": 42674,
                            "wrong_delta_vs_lightglue": -149,
                        },
                        "seed_results": [
                            {
                                "seed": 20260623,
                                "valid": True,
                                "rows": 78,
                                "selector_correct": 17453,
                                "selector_wrong": 0,
                                "lightglue_correct": 3642,
                                "lightglue_wrong": 55,
                                "correct_delta_vs_lightglue": 13811,
                                "wrong_delta_vs_lightglue": -55,
                                "base_disjoint": True,
                                "manifest_counts": {"train": 26, "dev": 26, "val": 26, "lockbox": 26},
                                "split_results": {
                                    "dev": {"rows": 26, "correct_delta_vs_lightglue": 4559, "wrong_delta_vs_lightglue": -12},
                                    "val": {"rows": 26, "correct_delta_vs_lightglue": 4403, "wrong_delta_vs_lightglue": -34},
                                    "lockbox": {
                                        "rows": 26,
                                        "correct_delta_vs_lightglue": 4849,
                                        "wrong_delta_vs_lightglue": -9,
                                    },
                                },
                            },
                            {
                                "seed": 20260624,
                                "valid": True,
                                "rows": 78,
                                "selector_correct": 18521,
                                "selector_wrong": 0,
                                "lightglue_correct": 3531,
                                "lightglue_wrong": 42,
                                "correct_delta_vs_lightglue": 14990,
                                "wrong_delta_vs_lightglue": -42,
                                "base_disjoint": True,
                                "manifest_counts": {"train": 26, "dev": 26, "val": 26, "lockbox": 26},
                                "split_results": {
                                    "dev": {"rows": 26, "correct_delta_vs_lightglue": 5221, "wrong_delta_vs_lightglue": -13},
                                    "val": {"rows": 26, "correct_delta_vs_lightglue": 4897, "wrong_delta_vs_lightglue": -10},
                                    "lockbox": {
                                        "rows": 26,
                                        "correct_delta_vs_lightglue": 4872,
                                        "wrong_delta_vs_lightglue": -19,
                                    },
                                },
                            },
                            {
                                "seed": 20260625,
                                "valid": True,
                                "rows": 84,
                                "selector_correct": 16867,
                                "selector_wrong": 0,
                                "lightglue_correct": 2994,
                                "lightglue_wrong": 52,
                                "correct_delta_vs_lightglue": 13873,
                                "wrong_delta_vs_lightglue": -52,
                                "base_disjoint": True,
                                "manifest_counts": {"train": 28, "dev": 28, "val": 28, "lockbox": 28},
                                "split_results": {
                                    "dev": {"rows": 28, "correct_delta_vs_lightglue": 3854, "wrong_delta_vs_lightglue": -24},
                                    "val": {"rows": 28, "correct_delta_vs_lightglue": 4616, "wrong_delta_vs_lightglue": -11},
                                    "lockbox": {
                                        "rows": 28,
                                        "correct_delta_vs_lightglue": 5403,
                                        "wrong_delta_vs_lightglue": -17,
                                    },
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            exit_code = validator.main(
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
            self.assertEqual(payload["rows"], 240)
            self.assertEqual(payload["selected_correct"], 52841)
            self.assertEqual(payload["selected_wrong"], 0)
            self.assertEqual(payload["lightglue_correct"], 10167)
            self.assertEqual(payload["lightglue_wrong"], 149)
            self.assertEqual(payload["correct_delta_vs_lightglue"], 42674)
            self.assertEqual(payload["wrong_delta_vs_lightglue"], -149)
            self.assertEqual(payload["split_results"]["dev"]["rows"], 80)
            self.assertEqual(payload["split_results"]["dev"]["correct_delta_vs_lightglue"], 13634)
            self.assertEqual(payload["split_results"]["dev"]["wrong_delta_vs_lightglue"], -49)
            self.assertTrue(payload["base_disjoint"])
            self.assertEqual(payload["manifest_counts"]["lockbox"], 80)


if __name__ == "__main__":
    unittest.main()
