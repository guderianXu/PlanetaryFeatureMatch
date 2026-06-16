import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_phase2j_balanced_manifest as phase2j_mod


def pair_row(index: int, reference_variant: str, target_variant: str, *, split: str = "train") -> dict[str, str]:
    base_id = f"b{index:04d}"
    return {
        "pair_index": str(index),
        "split": split,
        "pair_type": "same_position_view",
        "reference_dataset_id": "h100km_fov076",
        "reference_pose_id": f"{base_id}_{reference_variant}",
        "reference_base_id": base_id,
        "reference_variant": reference_variant,
        "target_dataset_id": "h100km_fov076",
        "target_pose_id": f"{base_id}_{target_variant}",
        "target_base_id": base_id,
        "target_variant": target_variant,
        "valid_fraction": "1.0",
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


class Phase2JBalancedManifestTest(unittest.TestCase):
    def test_build_balanced_manifest_uses_train_only_and_balances_buckets(self) -> None:
        extreme_rows = [
            pair_row(1, "nadir", "extreme_02"),
            pair_row(2, "mid_01", "extreme_02"),
            pair_row(3, "nadir", "extreme_03"),
            pair_row(4, "mid_02", "extreme_03"),
            pair_row(5, "nadir", "extreme_02", split="val"),
        ]
        train_rows = [
            *extreme_rows,
            pair_row(10, "nadir", "mid_02"),
            pair_row(11, "mid_01", "mid_02"),
            pair_row(12, "mid_02", "extreme_01"),
            pair_row(13, "nadir", "extreme_01"),
            pair_row(14, "mid_02", "extreme_01", split="test"),
        ]
        protected_pattern_rows = [
            pair_row(100, "nadir", "mid_02", split="val"),
            pair_row(101, "mid_01", "mid_02", split="val"),
            pair_row(101, "mid_02", "extreme_01", split="test"),
        ]
        residual_rows = [
            pair_row(200, "nadir", "extreme_02"),
            pair_row(201, "nadir", "extreme_03", split="val"),
        ]

        rows = phase2j_mod.build_phase2j_manifest_rows(
            extreme_rows=extreme_rows,
            train_rows=train_rows,
            protected_pattern_rows=protected_pattern_rows,
            residual_rows=residual_rows,
            config=phase2j_mod.Phase2JConfig(
                extreme_count=4,
                protected_count=3,
                residual_count=2,
                seed=17,
            ),
        )

        self.assertTrue(rows)
        self.assertEqual([row["pair_index"] for row in rows], [str(index) for index in range(len(rows))])
        self.assertTrue(all(row["split"] == "train" for row in rows))
        identities = [phase2j_mod.pair_identity(row) for row in rows]
        self.assertEqual(len(identities), len(set(identities)))

        bucket_counts: dict[str, int] = {}
        for row in rows:
            bucket_counts[row["phase2j_bucket"]] = bucket_counts.get(row["phase2j_bucket"], 0) + 1
        self.assertEqual(bucket_counts["extreme_main"], 4)
        self.assertEqual(bucket_counts["protected_replay"], 3)
        self.assertEqual(bucket_counts["residual_hard"], 1)

        protected_patterns = {
            (row["reference_variant"], row["target_variant"])
            for row in rows
            if row["phase2j_bucket"] == "protected_replay"
        }
        self.assertEqual(protected_patterns, {("nadir", "mid_02"), ("mid_01", "mid_02"), ("mid_02", "extreme_01")})

        extreme_patterns = {
            (row["reference_variant"], row["target_variant"])
            for row in rows
            if row["phase2j_bucket"] == "extreme_main"
        }
        self.assertEqual(
            extreme_patterns,
            {
                ("nadir", "extreme_02"),
                ("mid_01", "extreme_02"),
                ("nadir", "extreme_03"),
                ("mid_02", "extreme_03"),
            },
        )

    def test_build_balanced_manifest_can_repeat_and_interleave_buckets_for_short_runs(self) -> None:
        extreme_rows = [
            pair_row(1, "nadir", "extreme_02"),
            pair_row(2, "mid_01", "extreme_02"),
            pair_row(3, "nadir", "extreme_03"),
            pair_row(4, "mid_02", "extreme_03"),
        ]
        train_rows = [
            *extreme_rows,
            pair_row(10, "nadir", "mid_02"),
            pair_row(11, "mid_01", "mid_02"),
            pair_row(12, "mid_02", "extreme_01"),
            pair_row(13, "nadir", "extreme_01"),
        ]
        protected_pattern_rows = [
            pair_row(100, "nadir", "mid_02", split="val"),
            pair_row(101, "mid_01", "mid_02", split="val"),
            pair_row(102, "mid_02", "extreme_01", split="test"),
            pair_row(103, "nadir", "extreme_01", split="test"),
        ]
        residual_rows = [
            pair_row(200, "nadir", "extreme_02"),
            pair_row(201, "nadir", "extreme_03"),
        ]

        rows = phase2j_mod.build_phase2j_manifest_rows(
            extreme_rows=extreme_rows,
            train_rows=train_rows,
            protected_pattern_rows=protected_pattern_rows,
            residual_rows=residual_rows,
            config=phase2j_mod.Phase2JConfig(
                extreme_count=4,
                protected_count=4,
                residual_count=2,
                residual_repeat=2,
                seed=17,
                interleave_cycle=(
                    "residual_hard",
                    "protected_replay",
                    "protected_replay",
                    "extreme_main",
                    "extreme_main",
                ),
            ),
        )

        self.assertEqual([row["pair_index"] for row in rows], [str(index) for index in range(len(rows))])
        self.assertEqual(
            [row["phase2j_bucket"] for row in rows[:5]],
            ["residual_hard", "protected_replay", "protected_replay", "extreme_main", "extreme_main"],
        )
        residual_count = sum(1 for row in rows if row["phase2j_bucket"] == "residual_hard")
        protected_count = sum(1 for row in rows if row["phase2j_bucket"] == "protected_replay")
        extreme_count = sum(1 for row in rows if row["phase2j_bucket"] == "extreme_main")
        self.assertEqual(residual_count, 4)
        self.assertEqual(protected_count, 4)
        self.assertEqual(extreme_count, 4)


if __name__ == "__main__":
    unittest.main()
