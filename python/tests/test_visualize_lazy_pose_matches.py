import csv
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import visualize_lazy_pose_matches as visual


def _make_spec(target_variant: str = "extreme_02"):
    return SimpleNamespace(
        pair_index=0,
        split="val",
        reference=SimpleNamespace(pose_id="ref_pose", base_id="ref_base", variant="nadir"),
        target=SimpleNamespace(pose_id="target_pose", base_id="target_base", variant=target_variant),
    )


def _make_visual(
    label: str,
    match_count: int,
    target_variant: str = "extreme_02",
    *,
    errors: list[float] | None = None,
    valid_fraction: float = 1.0,
):
    points_a = np.stack(
        [
            np.linspace(0.0, 100.0, match_count, dtype=np.float32),
            np.linspace(10.0, 110.0, match_count, dtype=np.float32),
        ],
        axis=1,
    )
    points_b = points_a + np.array([2.0, 3.0], dtype=np.float32)
    return visual.LazyMatchVisual(
        label=label,
        spec=_make_spec(target_variant),
        pair=SimpleNamespace(),
        valid_fraction=valid_fraction,
        points_a=points_a,
        points_b=points_b,
        scores=np.ones(match_count, dtype=np.float32),
        errors=np.array(errors if errors is not None else [0.0] * match_count, dtype=np.float32),
        correct=np.array(
            [error <= 5.0 for error in (errors if errors is not None else [0.0] * match_count)],
            dtype=bool,
        ),
    )


class VisualizeLazyPoseMatchesTest(unittest.TestCase):
    def test_parse_args_can_disable_html_report_generation(self):
        base_argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "state.pt",
            "--output-dir",
            "out",
        ]
        with mock.patch.object(sys, "argv", base_argv):
            default_args = visual.parse_args()
        self.assertTrue(default_args.html_report)

        with mock.patch.object(sys, "argv", [*base_argv, "--no-html-report"]):
            no_report_args = visual.parse_args()
        self.assertFalse(no_report_args.html_report)

    def test_parse_args_can_disable_unfiltered_all_match_details(self):
        base_argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "state.pt",
            "--output-dir",
            "out",
        ]
        with mock.patch.object(sys, "argv", base_argv):
            default_args = visual.parse_args()
        self.assertTrue(default_args.write_all_match_details)

        with mock.patch.object(sys, "argv", [*base_argv, "--no-write-all-match-details"]):
            no_all_detail_args = visual.parse_args()
        self.assertFalse(no_all_detail_args.write_all_match_details)

    def test_parse_args_accepts_unfiltered_all_match_detail_sampling_limits(self):
        base_argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "state.pt",
            "--output-dir",
            "out",
        ]
        with mock.patch.object(sys, "argv", base_argv):
            default_args = visual.parse_args()
        self.assertEqual(default_args.all_match_details_max_results, 0)
        self.assertEqual(default_args.all_match_details_max_matches_per_result, 0)

        with mock.patch.object(
            sys,
            "argv",
            [
                *base_argv,
                "--all-match-details-max-results",
                "12",
                "--all-match-details-max-matches-per-result",
                "4",
            ],
        ):
            limited_args = visual.parse_args()
        self.assertEqual(limited_args.all_match_details_max_results, 12)
        self.assertEqual(limited_args.all_match_details_max_matches_per_result, 4)

    def test_parse_args_can_override_matcher_accept_assignment_mode(self):
        base_argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "state.pt",
            "--output-dir",
            "out",
        ]
        with mock.patch.object(sys, "argv", base_argv):
            default_args = visual.parse_args()
        self.assertEqual(default_args.matcher_accept_assignment_mode, "")

        with mock.patch.object(sys, "argv", [*base_argv, "--matcher-accept-assignment-mode", "off"]):
            override_args = visual.parse_args()
        self.assertEqual(override_args.matcher_accept_assignment_mode, "off")

    def test_parse_args_can_override_matcher_final_accept_score_mode(self):
        base_argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "state.pt",
            "--output-dir",
            "out",
        ]
        with mock.patch.object(sys, "argv", base_argv):
            default_args = visual.parse_args()
        self.assertEqual(default_args.matcher_final_accept_score_mode, "")
        self.assertAlmostEqual(default_args.matcher_final_accept_score_alpha, -1.0)

        with mock.patch.object(
            sys,
            "argv",
            [
                *base_argv,
                "--matcher-final-accept-score-mode",
                "multiply",
                "--matcher-final-accept-score-alpha",
                "0.25",
            ],
        ):
            override_args = visual.parse_args()
        self.assertEqual(override_args.matcher_final_accept_score_mode, "multiply")
        self.assertAlmostEqual(override_args.matcher_final_accept_score_alpha, 0.25)

    def test_apply_matcher_calibration_overrides_passes_final_accept_score_mode(self):
        model = SimpleNamespace(config=object(), set_matcher_calibration=mock.Mock())
        args = SimpleNamespace(
            matcher_candidate_topk=-1,
            matcher_accept_assignment_mode="",
            matcher_final_accept_score_mode="multiply",
            matcher_final_accept_score_alpha=-1.0,
        )

        config = visual.apply_matcher_calibration_overrides(model, args)

        self.assertIs(config, model.config)
        model.set_matcher_calibration.assert_called_once_with(
            candidate_topk=None,
            final_accept_score_mode="multiply",
            accept_assignment_mode=None,
            final_accept_score_alpha=None,
        )

    def test_apply_matcher_calibration_overrides_keeps_checkpoint_when_unset(self):
        model = SimpleNamespace(config=object(), set_matcher_calibration=mock.Mock())
        args = SimpleNamespace(
            matcher_candidate_topk=-1,
            matcher_accept_assignment_mode="",
            matcher_final_accept_score_mode="",
            matcher_final_accept_score_alpha=-1.0,
        )

        config = visual.apply_matcher_calibration_overrides(model, args)

        self.assertIs(config, model.config)
        model.set_matcher_calibration.assert_not_called()

    def test_true_geometry_filter_keeps_only_matches_with_true_warp_error_under_threshold(self):
        result = _make_visual("base", 4, errors=[1.0, 4.9, 5.1, 9.0], valid_fraction=0.25)

        filtered = visual.filter_visual_matches(
            result,
            geometry_filter="true_geometry",
            threshold_px=5.0,
            true_geometry_min_valid_fraction=0.10,
            label="base / true-geometry",
        )

        self.assertEqual(filtered.label, "base / true-geometry")
        self.assertEqual(filtered.matches, 2)
        np.testing.assert_allclose(filtered.errors, np.array([1.0, 4.9], dtype=np.float32))
        self.assertEqual(filtered.correct_count, 2)
        self.assertEqual(filtered.filtered_reject_reason, "")

    def test_true_geometry_filter_rejects_low_overlap_pairs_before_match_thresholding(self):
        result = _make_visual("base", 3, errors=[1.0, 2.0, 3.0], valid_fraction=0.05)

        filtered = visual.filter_visual_matches(
            result,
            geometry_filter="true_geometry",
            threshold_px=5.0,
            true_geometry_min_valid_fraction=0.10,
            label="base / true-geometry",
        )

        self.assertEqual(filtered.matches, 0)
        self.assertEqual(filtered.filtered_reject_reason, "low_valid_fraction")

    def test_true_geometry_profile_sets_filtered_geometry_defaults(self):
        base_argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "state.pt",
            "--output-dir",
            "out",
            "--post-filter-profile",
            "true_geometry_error5_overlap10",
        ]

        with mock.patch.object(sys, "argv", base_argv):
            args = visual.parse_args()

        self.assertEqual(args.geometry_filter, "none")
        self.assertEqual(args.filtered_geometry_filter, "true_geometry")
        self.assertAlmostEqual(args.geometry_threshold_px, 5.0)
        self.assertAlmostEqual(args.true_geometry_min_valid_fraction, 0.10)
        self.assertEqual(args.filtered_min_matches, 0)

    def test_graph_magsac2_min24_profile_sets_pareto_defaults(self):
        base_argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "state.pt",
            "--output-dir",
            "out",
            "--post-filter-profile",
            "fov76_graph_magsac2_min24_balanced",
        ]

        with mock.patch.object(sys, "argv", base_argv):
            args = visual.parse_args()

        self.assertEqual(args.geometry_filter, "none")
        self.assertAlmostEqual(args.geometry_threshold_px, 2.0)
        self.assertEqual(args.filtered_geometry_filter, "magsac")
        self.assertAlmostEqual(args.filtered_min_score, 18.0)
        self.assertAlmostEqual(args.filtered_min_margin, 0.0)
        self.assertEqual(args.filtered_min_matches, 24)
        self.assertEqual(args.adaptive_geometry_rescue_variants, "")

    def test_skips_adaptive_rescue_filter_when_base_exceeds_rescue_match_cap(self):
        base = _make_visual("base", 20)
        rescue = _make_visual("base / adaptive-rescue-source", 30)

        def fake_filter(result, **_kwargs):
            if "adaptive-rescue-source" in result.label:
                raise AssertionError("rescue filtering should be skipped")
            return result

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(visual, "filter_visual_matches", side_effect=fake_filter):
                visual.write_visual_summary_artifacts(
                    Path(tmp_dir),
                    selected=[],
                    filtered_selected=[],
                    all_results=[base],
                    write_all_summary=True,
                    filtered_geometry_filter="magsac",
                    filtered_threshold_px=5.0,
                    adaptive_geometry_rescue_results={visual.visual_identity(base): rescue},
                    adaptive_geometry_rescue_config=visual.AdaptiveGeometryRescueConfig(
                        enabled=True,
                        target_variants=("extreme_02",),
                        rescue_threshold_px=10.0,
                        min_match_gain=5,
                        max_base_matches=16,
                    ),
                )

            self.assertTrue((Path(tmp_dir) / "all_filtered_summary.csv").exists())

    def test_can_write_filtered_match_details_without_unfiltered_all_details(self):
        base = _make_visual("base", 4, errors=[1.0, 8.0, 2.0, 9.0], valid_fraction=0.25)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            visual.write_visual_summary_artifacts(
                output_dir,
                selected=[],
                filtered_selected=[],
                all_results=[base],
                write_all_summary=True,
                filtered_geometry_filter="true_geometry",
                filtered_threshold_px=5.0,
                true_geometry_min_valid_fraction=0.10,
                write_match_details=True,
                write_all_match_details=False,
            )

            self.assertTrue((output_dir / "all_summary.csv").is_file())
            self.assertFalse((output_dir / "all_match_details.csv").exists())
            self.assertTrue((output_dir / "all_filtered_summary.csv").is_file())
            filtered_details = output_dir / "all_filtered_match_details.csv"
            self.assertTrue(filtered_details.is_file())
            with filtered_details.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)

    def test_can_limit_unfiltered_all_match_details_without_limiting_filtered_details(self):
        base = _make_visual("base", 6, errors=[1.0, 2.0, 3.0, 8.0, 9.0, 10.0], valid_fraction=0.25)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            visual.write_visual_summary_artifacts(
                output_dir,
                selected=[],
                filtered_selected=[],
                all_results=[base],
                write_all_summary=True,
                filtered_geometry_filter="true_geometry",
                filtered_threshold_px=5.0,
                true_geometry_min_valid_fraction=0.10,
                write_match_details=True,
                write_all_match_details=True,
                all_match_details_max_matches_per_result=4,
            )

            with (output_dir / "all_match_details.csv").open(encoding="utf-8", newline="") as handle:
                all_rows = list(csv.DictReader(handle))
            self.assertEqual(len(all_rows), 4)
            self.assertGreater(sum(1 for row in all_rows if row["correct"] == "1"), 0)
            self.assertGreater(sum(1 for row in all_rows if row["correct"] == "0"), 0)

            with (output_dir / "all_filtered_match_details.csv").open(encoding="utf-8", newline="") as handle:
                filtered_rows = list(csv.DictReader(handle))
            self.assertEqual(len(filtered_rows), 3)


if __name__ == "__main__":
    unittest.main()
