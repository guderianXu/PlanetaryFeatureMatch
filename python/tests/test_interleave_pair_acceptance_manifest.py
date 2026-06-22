import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class InterleavePairAcceptanceManifestTest(unittest.TestCase):
    def write_manifest(self, path: Path, labels: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["pair_index", "reference_base_id", "target_variant", "pair_accept_label"],
            )
            writer.writeheader()
            for index, label in enumerate(labels):
                writer.writerow(
                    {
                        "pair_index": str(index),
                        "reference_base_id": f"base_{index}",
                        "target_variant": "extreme_02",
                        "pair_accept_label": label,
                    }
                )

    def read_labels(self, path: Path) -> list[str]:
        with path.open(newline="", encoding="utf-8") as handle:
            return [row["pair_accept_label"] for row in csv.DictReader(handle)]

    def test_interleaves_accept_and_reject_rows_with_reject_repeats(self) -> None:
        import interleave_pair_acceptance_manifest as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            output = root / "out" / "interleaved.csv"
            summary_json = root / "out" / "summary.json"
            report_html = root / "out" / "index.html"
            self.write_manifest(source, ["1", "1", "1", "1", "0", "0"])

            exit_code = mod.main(
                [
                    "--input-manifest",
                    str(source),
                    "--output-manifest",
                    str(output),
                    "--summary-json",
                    str(summary_json),
                    "--report-html",
                    str(report_html),
                    "--reject-repeat",
                    "2",
                    "--seed",
                    "7",
                ]
            )

            labels = self.read_labels(output)
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            report = report_html.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(labels.count("1"), 4)
        self.assertEqual(labels.count("0"), 4)
        self.assertLessEqual(summary["max_same_label_run"], 2)
        self.assertEqual(summary["label_counts"], {"0": 4, "1": 4})
        self.assertIn("Interleaved pair acceptance manifest", report)

    def test_rejects_manifest_without_both_labels(self) -> None:
        import interleave_pair_acceptance_manifest as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            self.write_manifest(source, ["1", "1"])

            with self.assertRaises(ValueError):
                mod.interleave_manifest_rows(
                    mod.read_manifest(source)[1],
                    seed=0,
                    reject_repeat=1,
                )

    def test_spreads_extra_rejects_instead_of_appending_tail_block(self) -> None:
        import interleave_pair_acceptance_manifest as mod

        rows = [
            {"pair_index": str(index), "pair_accept_label": label}
            for index, label in enumerate(["1"] * 10 + ["0"] * 5)
        ]

        output = mod.interleave_manifest_rows(rows, seed=3, reject_repeat=3)

        self.assertEqual([row["pair_accept_label"] for row in output].count("1"), 10)
        self.assertEqual([row["pair_accept_label"] for row in output].count("0"), 15)
        self.assertLessEqual(mod.max_same_label_run(output), 2)


if __name__ == "__main__":
    unittest.main()
