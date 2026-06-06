import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_p1_viewpoint_retention_labels as p1


class P1RetentionLabelGenerationTest(unittest.TestCase):
    def test_select_kept_candidates_applies_policy_caps_and_filters(self):
        candidates = [
            p1.RetentionCandidate("numeric", "viewpoint", "source_a", "a0.pt", 80, 80, 0),
            p1.RetentionCandidate("numeric", "viewpoint", "source_a", "a1.pt", 70, 70, 0),
            p1.RetentionCandidate("numeric", "viewpoint", "source_a", "a2.pt", 90, 90, 0),
            p1.RetentionCandidate("numeric", "viewpoint", "source_b", "b0.pt", 49, 49, 0),
            p1.RetentionCandidate("numeric", "viewpoint", "source_c", "c0.pt", 100, 99, 1),
            p1.RetentionCandidate("numeric", "compound", "source_d", "d0.pt", 120, 120, 0),
            p1.RetentionCandidate("timestamp", "viewpoint", "source_e", "e0.pt", 120, 120, 0),
        ]

        selected = p1.select_kept_candidates(
            candidates,
            eligible_groups={("numeric", "viewpoint"), ("timestamp", "viewpoint")},
            label_cap_per_pair=12,
            label_cap_per_group=24,
            pair_min_inliers=50,
            pair_min_precision=0.995,
            pair_max_wrong=0,
            max_source_pairs=2,
        )

        selected_paths = [candidate.pair_pt for candidate in selected]
        self.assertEqual(selected_paths, ["a0.pt", "a1.pt", "e0.pt"])
        self.assertEqual(sum(candidate.retained_labels for candidate in selected), 36)

    def test_retention_candidate_reports_precision_safely(self):
        candidate = p1.RetentionCandidate("numeric", "viewpoint", "source", "pair.pt", 0, 0, 0)
        self.assertEqual(candidate.precision, 0.0)


if __name__ == "__main__":
    unittest.main()
