import csv
import tempfile
import unittest
from pathlib import Path


class SamplePairSpecManifestTest(unittest.TestCase):
    def write_manifest(self, path: Path, rows: list[dict[str, object]]) -> None:
        fieldnames = [
            "pair_index",
            "split",
            "pair_type",
            "reference_dataset_id",
            "reference_pose_id",
            "reference_base_id",
            "reference_variant",
            "target_dataset_id",
            "target_pose_id",
            "target_base_id",
            "target_variant",
            "valid_fraction",
            "valid_pixels",
            "attempts",
            "crop_a_x0",
            "crop_a_y0",
            "crop_a_x1",
            "crop_a_y1",
            "crop_b_x0",
            "crop_b_y0",
            "crop_b_x1",
            "crop_b_y1",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def make_row(self, index: int, split: str, variant: str, base_id: str) -> dict[str, object]:
        return {
            "pair_index": index,
            "split": split,
            "pair_type": "same_position_view",
            "reference_dataset_id": "h100km_fov076",
            "reference_pose_id": f"{base_id}_nadir",
            "reference_base_id": base_id,
            "reference_variant": "nadir",
            "target_dataset_id": "h100km_fov076",
            "target_pose_id": f"{base_id}_{variant}",
            "target_base_id": base_id,
            "target_variant": variant,
            "valid_fraction": "1.000000",
            "valid_pixels": 2048 * 2048,
            "attempts": 1,
            "crop_a_x0": 0,
            "crop_a_y0": 0,
            "crop_a_x1": 2048,
            "crop_a_y1": 2048,
            "crop_b_x0": 0,
            "crop_b_y0": 0,
            "crop_b_x1": 2048,
            "crop_b_y1": 2048,
        }

    def read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_samples_stratified_base_disjoint_outputs_and_excludes_history(self) -> None:
        import sample_pair_spec_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            excluded = root / "excluded.csv"
            rows = []
            index = 0
            for split in ("val", "test"):
                for variant in ("mid_01", "extreme_01"):
                    for item in range(5):
                        base_id = f"{split}_{variant}_b{item:03d}"
                        rows.append(self.make_row(index, split, variant, base_id))
                        index += 1
            self.write_manifest(source, rows)
            self.write_manifest(excluded, [self.make_row(999, "val", "mid_01", "val_mid_01_b000")])

            train_out = root / "phase33_dev_train.csv"
            val_out = root / "phase33_dev_val.csv"
            lockbox_out = root / "phase33_lockbox.csv"
            summary_json = root / "summary.json"
            summary_html = root / "summary.html"

            exit_code = sample_pair_spec_manifest.main(
                [
                    "--source-manifest",
                    str(source),
                    "--exclude-manifest",
                    str(excluded),
                    "--split",
                    "val",
                    "--split",
                    "test",
                    "--target-variant",
                    "mid_01",
                    "--target-variant",
                    "extreme_01",
                    "--output-spec",
                    f"dev_train,{train_out},1",
                    "--output-spec",
                    f"dev_val,{val_out},1",
                    "--output-spec",
                    f"lockbox,{lockbox_out},1",
                    "--summary-json",
                    str(summary_json),
                    "--summary-html",
                    str(summary_html),
                    "--seed",
                    "11",
                ]
            )

            self.assertEqual(exit_code, 0)
            outputs = {
                "dev_train": self.read_rows(train_out),
                "dev_val": self.read_rows(val_out),
                "lockbox": self.read_rows(lockbox_out),
            }
            for output_rows in outputs.values():
                self.assertEqual(len(output_rows), 4)
                self.assertEqual(
                    sorted((row["split"], row["target_variant"]) for row in output_rows),
                    [
                        ("test", "extreme_01"),
                        ("test", "mid_01"),
                        ("val", "extreme_01"),
                        ("val", "mid_01"),
                    ],
                )
                self.assertNotIn(
                    "val_mid_01_b000",
                    {row["reference_base_id"] for row in output_rows} | {row["target_base_id"] for row in output_rows},
                )

            bases_by_output = {
                name: {row["target_base_id"] for row in output_rows}
                for name, output_rows in outputs.items()
            }
            self.assertFalse(bases_by_output["dev_train"] & bases_by_output["dev_val"])
            self.assertFalse(bases_by_output["dev_train"] & bases_by_output["lockbox"])
            self.assertFalse(bases_by_output["dev_val"] & bases_by_output["lockbox"])
            self.assertIn("Phase33 pair-spec sample", summary_html.read_text(encoding="utf-8"))

    def test_raises_when_bucket_has_too_few_base_disjoint_candidates(self) -> None:
        import sample_pair_spec_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            self.write_manifest(
                source,
                [
                    self.make_row(0, "val", "mid_01", "same_base"),
                    self.make_row(1, "val", "mid_01", "same_base"),
                ],
            )

            with self.assertRaises(RuntimeError):
                sample_pair_spec_manifest.main(
                    [
                        "--source-manifest",
                        str(source),
                        "--split",
                        "val",
                        "--target-variant",
                        "mid_01",
                        "--output-spec",
                        f"dev_train,{root / 'train.csv'},1",
                        "--output-spec",
                        f"dev_val,{root / 'val.csv'},1",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
