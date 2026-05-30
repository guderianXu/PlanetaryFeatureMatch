import tempfile
import unittest
from pathlib import Path

import hard_pair_mining


class HardPairMiningTest(unittest.TestCase):
    def test_selects_low_precision_high_match_pairs_with_filters_and_dedup(self):
        rows = [
            {"pair_pt": "cache/source_a/pair_000010.pt", "sparse_matches": "4", "match_precision": "0.0"},
            {"pair_pt": "cache/source_b/pair_000003.pt", "sparse_matches": "80", "match_precision": "0.50"},
            {"pair_pt": "cache/source_c/pair_000007.pt", "sparse_matches": "120", "match_precision": "0.50"},
            {"pair_pt": "cache/source_d/pair_000008.pt", "sparse_matches": "60", "match_precision": "0.20"},
            {"pair_pt": "cache/source_e/pair_000003.pt", "sparse_matches": "90", "match_precision": "0.10"},
            {"pair_pt": "cache/source_f/pair_000009.pt", "sparse_matches": "55", "match_precision": "0.95"},
        ]

        selected = hard_pair_mining.select_hard_pairs(
            rows,
            limit=4,
            min_matches=50,
            max_precision=0.9,
        )

        self.assertEqual([entry.pair_index for entry in selected], [3, 8, 7])

    def test_reads_summary_csv_and_formats_cli_args(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = Path(temp_dir) / "summary.csv"
            summary.write_text(
                "pair_pt,sparse_matches,match_precision\n"
                "cache/source_a/pair_000004.pt,50,0.40\n"
                "cache/source_b/pair_000005.pt,70,0.30\n",
                encoding="utf-8",
            )

            selected = hard_pair_mining.read_and_select(
                summary,
                limit=2,
                min_matches=1,
                max_precision=0.9,
            )

        self.assertEqual(hard_pair_mining.format_cli_args(selected), "--hard-synthetic-pair-cache-index 5 --hard-synthetic-pair-cache-index 4")

    def test_reads_current_cache_eval_summary_columns(self):
        rows = [
            {"pair_pt": "cache/source_a/pair_000004.pt", "matches": "50", "precision": "0.40"},
            {"pair_pt": "cache/source_b/pair_000005.pt", "matches": "70", "precision": "0.95"},
            {"pair_pt": "cache/source_c/pair_000006.pt", "matches": "0", "precision": "0.00"},
        ]

        selected = hard_pair_mining.select_hard_pairs(
            rows,
            limit=4,
            min_matches=1,
            max_precision=0.9,
        )

        self.assertEqual([entry.pair_index for entry in selected], [4])


if __name__ == "__main__":
    unittest.main()
