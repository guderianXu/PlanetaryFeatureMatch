import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from stratify_graph_match_errors import summarize_rows


class StratifyGraphMatchErrorsTest(unittest.TestCase):
    def test_summarize_rows_groups_precision_and_weak_texture(self) -> None:
        rows = [
            {
                "dataset_group": "same_position",
                "difficulty": "normal",
                "matches": "100",
                "correct": "85",
                "weak_matches": "40",
                "weak_correct": "35",
            },
            {
                "dataset_group": "same_position",
                "difficulty": "normal",
                "matches": "50",
                "correct": "25",
                "weak_matches": "10",
                "weak_correct": "5",
            },
            {
                "dataset_group": "extreme_cross_position",
                "difficulty": "hard",
                "matches": "20",
                "correct": "19",
                "weak_matches": "0",
                "weak_correct": "0",
            },
        ]

        summary = summarize_rows(rows)
        keys = {row["group_key"] for row in summary}

        self.assertIn("same_position|normal|precision_080_090|weak_mid", keys)
        self.assertIn("same_position|normal|precision_lt_080|weak_low", keys)
        self.assertIn("extreme_cross_position|hard|precision_090_098|weak_none", keys)

        same_high = next(
            row
            for row in summary
            if row["group_key"] == "same_position|normal|precision_080_090|weak_mid"
        )
        self.assertEqual(same_high["pairs"], 1)
        self.assertEqual(same_high["matches"], 100)
        self.assertEqual(same_high["correct"], 85)
        self.assertAlmostEqual(same_high["precision"], 0.85)


if __name__ == "__main__":
    unittest.main()
