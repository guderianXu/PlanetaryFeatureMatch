import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


PAIR_FIELDS = [
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


def make_row(
    index: int,
    *,
    reference_base_id: str,
    target_base_id: str,
    reference_variant: str,
    target_variant: str,
    valid_fraction: float,
    pair_type: str = "cross_camera",
) -> dict[str, str]:
    return {
        "pair_index": str(index),
        "split": "all",
        "pair_type": pair_type,
        "reference_dataset_id": "h100km_fov076",
        "reference_pose_id": f"{reference_base_id}_{reference_variant}",
        "reference_base_id": reference_base_id,
        "reference_variant": reference_variant,
        "target_dataset_id": "h100km_fov076",
        "target_pose_id": f"{target_base_id}_{target_variant}",
        "target_base_id": target_base_id,
        "target_variant": target_variant,
        "valid_fraction": f"{valid_fraction:.6f}",
        "valid_pixels": "4194304",
        "attempts": "1",
        "crop_a_x0": "0",
        "crop_a_y0": "0",
        "crop_a_x1": "2048",
        "crop_a_y1": "2048",
        "crop_b_x0": "0",
        "crop_b_y0": "0",
        "crop_b_x1": "2048",
        "crop_b_y1": "2048",
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


class CrossCameraExtremeManifestTest(unittest.TestCase):
    def test_valid_fraction_bucket_boundaries(self) -> None:
        import build_cross_camera_extreme_manifests as mod

        self.assertEqual(mod.valid_fraction_bucket(0.019), "reject")
        self.assertEqual(mod.valid_fraction_bucket(0.020), "low")
        self.assertEqual(mod.valid_fraction_bucket(0.149), "low")
        self.assertEqual(mod.valid_fraction_bucket(0.150), "mid")
        self.assertEqual(mod.valid_fraction_bucket(0.499), "mid")
        self.assertEqual(mod.valid_fraction_bucket(0.500), "high")

    def test_selects_cross_camera_extreme_rows_and_keeps_outputs_base_disjoint(self) -> None:
        import build_cross_camera_extreme_manifests as mod

        rows: list[dict[str, str]] = []
        index = 0
        for target_variant, valid_fraction in (
            ("extreme_01", 0.05),
            ("extreme_02", 0.25),
            ("extreme_03", 0.75),
        ):
            for item in range(4):
                rows.append(
                    make_row(
                        index,
                        reference_base_id=f"ref_{target_variant}_{item}",
                        target_base_id=f"tgt_{target_variant}_{item}",
                        reference_variant="extreme_01",
                        target_variant=target_variant,
                        valid_fraction=valid_fraction,
                    )
                )
                index += 1
        rows.append(
            make_row(
                index,
                reference_base_id="same_ref",
                target_base_id="same_tgt",
                reference_variant="nadir",
                target_variant="mid_01",
                valid_fraction=0.75,
                pair_type="same_position_view",
            )
        )

        selected, summary = mod.select_split_rows(
            rows,
            output_specs=[
                mod.OutputSpec("train", 1),
                mod.OutputSpec("dev", 1),
                mod.OutputSpec("lockbox", 1),
            ],
            seed=7,
        )

        self.assertEqual(summary["eligible_rows"], 12)
        self.assertEqual(summary["rejected_rows"], 1)
        self.assertEqual({name: len(items) for name, items in selected.items()}, {"train": 3, "dev": 3, "lockbox": 3})
        for output_rows in selected.values():
            self.assertEqual({row["pair_type"] for row in output_rows}, {"cross_camera"})
            self.assertEqual(
                sorted(row["target_variant"] for row in output_rows),
                ["extreme_01", "extreme_02", "extreme_03"],
            )

        bases_by_output = {
            name: {
                base_id
                for row in output_rows
                for base_id in (row["reference_base_id"], row["target_base_id"])
            }
            for name, output_rows in selected.items()
        }
        self.assertFalse(bases_by_output["train"] & bases_by_output["dev"])
        self.assertFalse(bases_by_output["train"] & bases_by_output["lockbox"])
        self.assertFalse(bases_by_output["dev"] & bases_by_output["lockbox"])

    def test_cli_writes_split_manifests_and_summary_reports(self) -> None:
        import build_cross_camera_extreme_manifests as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "candidate.csv"
            rows = [
                make_row(
                    index,
                    reference_base_id=f"ref_{index}",
                    target_base_id=f"tgt_{index}",
                    reference_variant="extreme_01",
                    target_variant="extreme_02",
                    valid_fraction=0.25,
                )
                for index in range(4)
            ]
            write_manifest(source, rows)
            out = root / "out"
            summary_json = out / "summary.json"
            summary_html = out / "index.html"

            exit_code = mod.main(
                [
                    "--candidate-manifest",
                    str(source),
                    "--output-dir",
                    str(out),
                    "--train-per-bucket",
                    "1",
                    "--dev-per-bucket",
                    "1",
                    "--val-per-bucket",
                    "1",
                    "--lockbox-per-bucket",
                    "1",
                    "--summary-json",
                    str(summary_json),
                    "--summary-html",
                    str(summary_html),
                    "--seed",
                    "3",
                ]
            )

            self.assertEqual(exit_code, 0)
            for split in ("train", "dev", "val", "lockbox"):
                output_rows = read_manifest(out / f"{split}_pairs.csv")
                self.assertEqual(len(output_rows), 1)
                self.assertEqual(output_rows[0]["pair_type"], "cross_camera")
                self.assertEqual(output_rows[0]["target_variant"], "extreme_02")
            self.assertIn("Cross-Camera Extreme Manifest", summary_html.read_text(encoding="utf-8"))
            self.assertIn('"eligible_rows": 4', summary_json.read_text(encoding="utf-8"))

    def test_can_drop_underfilled_buckets_and_report_them(self) -> None:
        import build_cross_camera_extreme_manifests as mod

        full_bucket = [
            make_row(
                index,
                reference_base_id=f"ref_full_{index}",
                target_base_id=f"tgt_full_{index}",
                reference_variant="extreme_01",
                target_variant="extreme_02",
                valid_fraction=0.25,
            )
            for index in range(4)
        ]
        sparse_bucket = [
            make_row(
                100,
                reference_base_id="ref_sparse",
                target_base_id="tgt_sparse",
                reference_variant="extreme_02",
                target_variant="extreme_02",
                valid_fraction=0.25,
            )
        ]

        selected, summary = mod.select_split_rows(
            [*full_bucket, *sparse_bucket],
            output_specs=[
                mod.OutputSpec("train", 1),
                mod.OutputSpec("dev", 1),
                mod.OutputSpec("val", 1),
                mod.OutputSpec("lockbox", 1),
            ],
            seed=5,
            drop_underfilled=True,
        )

        self.assertEqual({name: len(rows) for name, rows in selected.items()}, {"train": 1, "dev": 1, "val": 1, "lockbox": 1})
        self.assertEqual(summary["dropped_bucket_count"], 1)
        self.assertEqual(
            summary["dropped_buckets"],
            {"extreme_02:extreme_02->extreme_02:mid": 1},
        )

    def test_selected_rows_are_relabelled_with_output_split(self) -> None:
        import build_cross_camera_extreme_manifests as mod

        rows = [
            make_row(
                index,
                reference_base_id=f"ref_{index}",
                target_base_id=f"tgt_{index}",
                reference_variant="extreme_01",
                target_variant="extreme_02",
                valid_fraction=0.25,
            )
            for index in range(4)
        ]
        for row, split in zip(rows, ["train", "val", "test", "test"]):
            row["split"] = split

        selected, _summary = mod.select_split_rows(
            rows,
            output_specs=[
                mod.OutputSpec("train", 1),
                mod.OutputSpec("dev", 1),
                mod.OutputSpec("lockbox", 1),
            ],
            seed=7,
        )

        self.assertEqual(selected["train"][0]["split"], "train")
        self.assertEqual(selected["dev"][0]["split"], "dev")
        self.assertEqual(selected["lockbox"][0]["split"], "lockbox")

    def test_select_split_rows_excludes_base_ids_from_existing_manifests(self) -> None:
        import build_cross_camera_extreme_manifests as mod

        rows = [
            make_row(
                0,
                reference_base_id="excluded_ref",
                target_base_id="excluded_tgt",
                reference_variant="extreme_01",
                target_variant="extreme_02",
                valid_fraction=0.25,
            ),
            make_row(
                1,
                reference_base_id="kept_ref_a",
                target_base_id="kept_tgt_a",
                reference_variant="extreme_01",
                target_variant="extreme_02",
                valid_fraction=0.25,
            ),
            make_row(
                2,
                reference_base_id="kept_ref_b",
                target_base_id="kept_tgt_b",
                reference_variant="extreme_01",
                target_variant="extreme_02",
                valid_fraction=0.25,
            ),
        ]

        selected, summary = mod.select_split_rows(
            rows,
            output_specs=[mod.OutputSpec("train", 1)],
            seed=3,
            excluded_base_keys={
                ("h100km_fov076", "excluded_ref"),
                ("h100km_fov076", "excluded_tgt"),
            },
        )

        selected_bases = {
            base_id
            for row in selected["train"]
            for base_id in (row["reference_base_id"], row["target_base_id"])
        }
        self.assertFalse({"excluded_ref", "excluded_tgt"} & selected_bases)
        self.assertEqual(summary["excluded_rows"], 1)


if __name__ == "__main__":
    unittest.main()
