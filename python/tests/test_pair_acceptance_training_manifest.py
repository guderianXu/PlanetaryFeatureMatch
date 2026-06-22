import csv
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "build_pair_acceptance_training_manifest.py"


class PairAcceptanceTrainingManifestTest(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_cli_preserves_pair_order_and_appends_acceptance_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_manifest = root / "pairs.csv"
            rejection_dataset = root / "rejection_dataset.csv"
            output_manifest = root / "nested" / "pair_acceptance_train.csv"

            pair_fields = [
                "split",
                "pair_index",
                "reference_base_id",
                "target_base_id",
                "target_variant",
                "custom_field",
            ]
            pair_rows = [
                {
                    "split": "train",
                    "pair_index": "1",
                    "reference_base_id": "",
                    "target_base_id": "base_reject",
                    "target_variant": "extreme_03",
                    "custom_field": "second_in_index_first_in_manifest",
                },
                {
                    "split": "train",
                    "pair_index": "0",
                    "reference_base_id": "base_keep",
                    "target_base_id": "base_keep_target",
                    "target_variant": "mid_01",
                    "custom_field": "first_in_index_second_in_manifest",
                },
            ]
            self.write_csv(pair_manifest, pair_fields, pair_rows)

            rejection_fields = [
                "split",
                "pair_index",
                "base_id",
                "target_base_id",
                "target_variant",
                "keep_label",
                "reject_label",
                "pfm_wrong",
                "pfm_precision",
            ]
            rejection_rows = [
                {
                    "split": "train",
                    "pair_index": "0",
                    "base_id": "base_keep",
                    "target_base_id": "",
                    "target_variant": "mid_01",
                    "keep_label": "1",
                    "reject_label": "0",
                    "pfm_wrong": "0",
                    "pfm_precision": "1",
                },
                {
                    "split": "train",
                    "pair_index": "1",
                    "base_id": "",
                    "target_base_id": "base_reject",
                    "target_variant": "extreme_03",
                    "keep_label": "0",
                    "reject_label": "1",
                    "pfm_wrong": "5",
                    "pfm_precision": "0.9333334",
                },
            ]
            self.write_csv(rejection_dataset, rejection_fields, rejection_rows)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--pair-manifest",
                    str(pair_manifest),
                    "--rejection-dataset",
                    str(rejection_dataset),
                    "--output",
                    str(output_manifest),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            with output_manifest.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                output_rows = list(reader)
                output_fields = reader.fieldnames

        self.assertEqual(
            output_fields,
            pair_fields
            + [
                "pair_accept_label",
                "pair_accept_weight",
                "pair_accept_source_wrong",
                "pair_accept_source_precision",
            ],
        )
        self.assertEqual([row["pair_index"] for row in output_rows], ["1", "0"])
        self.assertEqual(output_rows[0]["custom_field"], "second_in_index_first_in_manifest")
        self.assertEqual(output_rows[0]["pair_accept_label"], "0")
        self.assertEqual(output_rows[0]["pair_accept_weight"], "3.000000")
        self.assertEqual(output_rows[0]["pair_accept_source_wrong"], "5")
        self.assertEqual(output_rows[0]["pair_accept_source_precision"], "0.933333")
        self.assertEqual(output_rows[1]["pair_accept_label"], "1")
        self.assertEqual(output_rows[1]["pair_accept_weight"], "1.000000")
        self.assertEqual(output_rows[1]["pair_accept_source_wrong"], "0")
        self.assertEqual(output_rows[1]["pair_accept_source_precision"], "1.000000")

    def test_cli_matches_unique_rejection_row_when_split_names_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_manifest = root / "pairs.csv"
            rejection_dataset = root / "rejection_dataset.csv"
            output_manifest = root / "pair_acceptance_train.csv"

            self.write_csv(
                pair_manifest,
                ["pair_index", "split", "reference_base_id", "target_base_id", "target_variant"],
                [
                    {
                        "pair_index": "65",
                        "split": "val",
                        "reference_base_id": "h100_f076_b000020",
                        "target_base_id": "h100_f076_b000020",
                        "target_variant": "mid_01",
                    }
                ],
            )
            self.write_csv(
                rejection_dataset,
                ["pair_index", "split", "base_id", "target_variant", "keep_label", "reject_label", "pfm_wrong", "pfm_precision"],
                [
                    {
                        "pair_index": "65",
                        "split": "dev_train",
                        "base_id": "h100_f076_b000020",
                        "target_variant": "mid_01",
                        "keep_label": "0",
                        "reject_label": "1",
                        "pfm_wrong": "2",
                        "pfm_precision": "0.995402",
                    }
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--pair-manifest",
                    str(pair_manifest),
                    "--rejection-dataset",
                    str(rejection_dataset),
                    "--output",
                    str(output_manifest),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            with output_manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["pair_accept_label"], "0")
        self.assertEqual(rows[0]["pair_accept_source_wrong"], "2")


if __name__ == "__main__":
    unittest.main()
