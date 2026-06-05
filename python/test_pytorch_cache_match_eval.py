import sys
import unittest
from argparse import Namespace
from unittest import mock
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytorch_cache_match_eval as eval_py
import pfm_model
from patch_descriptor_training import SyntheticPair


class PyTorchCacheMatchEvalTest(unittest.TestCase):
    def test_eval_csv_fieldnames_include_graph_adaptive_stats(self):
        self.assertIn("graph_executed_layers", eval_py.EVAL_CSV_FIELDNAMES)
        self.assertIn("graph_kept_keypoints_a", eval_py.EVAL_CSV_FIELDNAMES)
        self.assertIn("graph_pruned_keypoints_b", eval_py.EVAL_CSV_FIELDNAMES)

    def test_select_descriptor_keypoints_filters_dark_pixels_and_caps(self):
        image = torch.ones(1, 8, 8)
        image[:, :4, :4] = 0.0
        descriptors = torch.randn(1, 4, 4, 4)

        keypoints, selected = eval_py.select_descriptor_keypoints(
            image,
            descriptors,
            max_keypoints=3,
            min_intensity=0.5,
        )

        self.assertEqual(tuple(keypoints.shape), (3, 2))
        self.assertEqual(tuple(selected.shape), (3,))
        self.assertTrue((keypoints >= 0).all())
        self.assertTrue((selected < 16).all())

    def test_select_descriptor_keypoints_prefers_local_texture(self):
        image = torch.full((1, 8, 8), 0.5)
        image[:, 7, 7] = 1.0
        descriptors = torch.randn(1, 4, 4, 4)

        keypoints, selected = eval_py.select_descriptor_keypoints(
            image,
            descriptors,
            max_keypoints=1,
            min_intensity=0.01,
        )

        self.assertEqual(tuple(keypoints[0].tolist()), (3.0, 3.0))
        self.assertEqual(int(selected[0]), 15)

    def test_select_descriptor_keypoints_can_use_learned_scores(self):
        image = torch.full((1, 8, 8), 0.5)
        image[:, 7, 7] = 1.0
        descriptors = torch.randn(1, 4, 4, 4)
        learned_scores = torch.zeros(1, 1, 4, 4)
        learned_scores[0, 0, 0, 1] = 10.0

        keypoints, selected = eval_py.select_descriptor_keypoints(
            image,
            descriptors,
            max_keypoints=1,
            min_intensity=0.01,
            keypoint_scores=learned_scores,
        )

        self.assertEqual(tuple(keypoints[0].tolist()), (1.0, 0.0))
        self.assertEqual(int(selected[0]), 1)

    def test_select_descriptor_keypoints_can_mix_texture_and_uniform_coverage(self):
        image = torch.full((1, 8, 8), 0.5)
        image[:, 7, 7] = 1.0
        descriptors = torch.randn(1, 4, 4, 4)

        _, selected = eval_py.select_descriptor_keypoints(
            image,
            descriptors,
            max_keypoints=4,
            min_intensity=0.01,
            texture_fraction=0.5,
        )

        self.assertIn(15, selected.tolist())
        self.assertEqual(len(set(selected.tolist())), 4)

    def test_select_descriptor_keypoints_can_reserve_weak_texture_quota(self):
        image = torch.full((1, 8, 8), 0.5)
        image[:, 7, 7] = 1.0
        descriptors = torch.randn(1, 4, 4, 4)
        learned_scores = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)

        _, selected = eval_py.select_descriptor_keypoints(
            image,
            descriptors,
            max_keypoints=4,
            min_intensity=0.01,
            texture_fraction=0.5,
            weak_texture_fraction=0.25,
            keypoint_scores=learned_scores,
        )

        self.assertIn(15, selected.tolist())
        self.assertIn(0, selected.tolist())
        self.assertEqual(len(set(selected.tolist())), 4)

    def test_select_descriptor_keypoints_can_cap_dense_cells(self):
        image = torch.ones(1, 16, 16)
        descriptors = torch.randn(1, 4, 8, 8)
        learned_scores = torch.zeros(1, 1, 8, 8)
        learned_scores[0, 0, :4, :4] = torch.arange(16, dtype=torch.float32).view(4, 4) + 100.0
        learned_scores[0, 0, 4:, 4:] = torch.arange(16, dtype=torch.float32).view(4, 4)

        keypoints, _ = eval_py.select_descriptor_keypoints(
            image,
            descriptors,
            max_keypoints=8,
            min_intensity=0.01,
            texture_fraction=1.0,
            keypoint_scores=learned_scores,
            keypoint_cell_cap=2,
            spatial_bins=2,
        )
        upper_left = ((keypoints[:, 0] < 4) & (keypoints[:, 1] < 4)).sum()

        self.assertLessEqual(int(upper_left), 2)
        self.assertEqual(tuple(keypoints.shape), (8, 2))

    def test_select_spatially_distributed_indices_prefers_distinct_cells(self):
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        keypoints = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(torch.float32)
        scores = torch.zeros(16)
        scores[0] = 10.0
        scores[1] = 9.0
        scores[2] = 8.0

        selected = eval_py.select_spatially_distributed_indices(
            keypoints,
            scores,
            max_keypoints=2,
            spatial_bins=2,
            descriptor_height=4,
            descriptor_width=4,
        )

        self.assertEqual(selected.tolist(), [0, 2])

    def test_select_spatially_distributed_indices_fills_remaining_by_score(self):
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        keypoints = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(torch.float32)
        scores = torch.zeros(16)
        scores[0] = 10.0
        scores[1] = 9.0
        scores[2] = 8.0

        selected = eval_py.select_spatially_distributed_indices(
            keypoints,
            scores,
            max_keypoints=5,
            spatial_bins=2,
            descriptor_height=4,
            descriptor_width=4,
        )

        self.assertEqual(selected.tolist(), [0, 2, 8, 10, 1])

    def test_select_descriptor_keypoints_rejects_negative_spatial_bins(self):
        image = torch.ones(1, 8, 8)
        descriptors = torch.randn(1, 4, 4, 4)

        with self.assertRaises(ValueError):
            eval_py.select_descriptor_keypoints(
                image,
                descriptors,
                max_keypoints=3,
                min_intensity=0.01,
                spatial_bins=-1,
            )

    def test_parse_args_accepts_keypoint_spatial_bins(self):
        argv = [
            "pytorch_cache_match_eval.py",
            "--cache-dir",
            "cache",
            "--pytorch-state",
            "state.pt",
            "--output",
            "summary.csv",
            "--keypoint-spatial-bins",
            "16",
            "--weak-texture-keypoint-fraction",
            "0.25",
            "--keypoint-cell-cap",
            "12",
            "--keypoint-score-mode",
            "learned",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = eval_py.parse_args()

        self.assertEqual(args.keypoint_spatial_bins, 16)
        self.assertEqual(args.weak_texture_keypoint_fraction, 0.25)
        self.assertEqual(args.keypoint_cell_cap, 12)
        self.assertEqual(args.keypoint_score_mode, "learned")

    def test_parse_args_accepts_graph_matcher_mode(self):
        argv = [
            "pytorch_cache_match_eval.py",
            "--cache-dir",
            "cache",
            "--pytorch-state",
            "state.pt",
            "--output",
            "summary.csv",
            "--matcher-mode",
            "graph_matcher",
            "--graph-fallback-mode",
            "none",
            "--graph-dustbin-delta",
            "-0.5",
            "--graph-acceptance-margin",
            "0.02",
            "--graph-min-raw-score",
            "0.4",
            "--graph-min-raw-margin",
            "0.03",
            "--graph-width-prune-min-score",
            "0.35",
            "--graph-early-stop-min-confidence",
            "0.8",
            "--graph-inference-preset",
            "high_precision",
            "--graph-min-accept-probability",
            "0.75",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = eval_py.parse_args()

        self.assertEqual(args.matcher_mode, "graph_matcher")
        self.assertEqual(args.graph_fallback_mode, "none")
        self.assertEqual(args.graph_dustbin_delta, -0.5)
        self.assertEqual(args.graph_acceptance_margin, 0.02)
        self.assertEqual(args.graph_min_raw_score, 0.4)
        self.assertEqual(args.graph_min_raw_margin, 0.03)
        self.assertEqual(args.graph_width_prune_min_score, 0.35)
        self.assertEqual(args.graph_early_stop_min_confidence, 0.8)
        self.assertEqual(args.graph_inference_preset, "high_precision")
        self.assertEqual(args.graph_min_accept_probability, 0.75)

    def test_graph_inference_thresholds_resolve_lightglue_presets(self):
        self.assertEqual(eval_py.graph_inference_thresholds("off", -1.0, -1.0), (-1.0, -1.0))
        self.assertEqual(eval_py.graph_inference_thresholds("fast", -1.0, -1.0), (0.25, 0.85))
        self.assertEqual(eval_py.graph_inference_thresholds("high_precision", -1.0, -1.0), (0.5, 0.85))

    def test_graph_inference_thresholds_allow_numeric_override(self):
        self.assertEqual(eval_py.graph_inference_thresholds("fast", 0.7, -1.0), (0.7, 0.85))

    def test_sample_descriptor_rows_at_keypoints_interpolates_rows(self):
        descriptors = torch.zeros(1, 2, 2, 2)
        descriptors[0, 0] = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        descriptors[0, 1] = torch.tensor([[2.0, 4.0], [6.0, 8.0]])

        rows = eval_py.sample_descriptor_rows_at_keypoints(descriptors, torch.tensor([[0.5, 0.5]]))

        self.assertTrue(torch.allclose(rows, torch.tensor([[4.0, 5.0]]), atol=1.0e-5))

    def test_graph_metadata_from_raw_features_samples_geometry_fields(self):
        heatmap = torch.full((1, 1, 2, 2), 0.7)
        descriptors = torch.randn(1, 4, 2, 2)
        scale = torch.full((1, 1, 2, 2), 2.0)
        orientation = torch.zeros(1, 2, 2, 2)
        orientation[:, 0] = 1.0
        affine = torch.zeros(1, 4, 2, 2)
        affine[:, 0] = 1.0
        affine[:, 3] = 1.0
        raw = pfm_model.RawFeatureMaps(
            heatmap=heatmap,
            descriptors=descriptors,
            scale=scale,
            orientation=orientation,
            affine=affine,
            dense_confidence=heatmap,
            keypoint_offsets=torch.zeros(1, 2, 2, 2),
            quality=torch.full((1, 1, 2, 2), 0.9),
            local_contrast=torch.full((1, 1, 2, 2), 0.4),
        )

        meta = eval_py.graph_metadata_from_raw_features(raw, torch.tensor([[1.0, 1.0]]), meta_dim=16)

        self.assertEqual(tuple(meta.shape), (1, 16))
        self.assertAlmostEqual(float(meta[0, 4]), 0.7, places=5)
        self.assertAlmostEqual(float(meta[0, 5]), torch.tensor(2.0).log().item(), places=5)
        self.assertAlmostEqual(float(meta[0, 12]), 0.9, places=5)
        self.assertAlmostEqual(float(meta[0, 13]), 0.4, places=5)

    def test_parse_args_accepts_min_target_gradient(self):
        argv = [
            "pytorch_cache_match_eval.py",
            "--cache-dir",
            "cache",
            "--pytorch-state",
            "state.pt",
            "--output",
            "summary.csv",
            "--min-target-gradient",
            "20.25",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = eval_py.parse_args()

        self.assertEqual(args.min_target_gradient, 20.25)

    def test_parse_args_accepts_min_target_local_contrast(self):
        argv = [
            "pytorch_cache_match_eval.py",
            "--cache-dir",
            "cache",
            "--pytorch-state",
            "state.pt",
            "--output",
            "summary.csv",
            "--min-target-local-contrast",
            "5.32",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = eval_py.parse_args()

        self.assertEqual(args.min_target_local_contrast, 5.32)

    def test_target_texture_gradient_uses_uint8_like_sobel_scale(self):
        image = torch.zeros(1, 5, 5)
        image[:, :, 3:] = 1.0

        gradient = eval_py.target_texture_gradient_mean(image)

        self.assertGreater(gradient, 100.0)

    def test_target_local_contrast_uses_uint8_like_local_variance_scale(self):
        image = torch.zeros(1, 9, 9)
        image[:, 4, 4] = 1.0

        contrast = eval_py.target_local_contrast_mean(image)

        self.assertGreater(contrast, 20.0)

    def test_cyclic_descriptor_similarity_accepts_quarter_channel_shift(self):
        desc_a = torch.zeros(1, 8)
        desc_b = torch.zeros(1, 8)
        desc_a[0, 1] = 1.0
        desc_b[0, 3] = 1.0

        similarity = eval_py.cyclic_descriptor_similarity(desc_a, desc_b)

        self.assertAlmostEqual(float(similarity[0, 0]), 1.0, places=6)

    def test_match_pair_descriptor_maps_scores_identity_warp_matches(self):
        image = torch.ones(1, 4, 4)
        warp = torch.zeros(4, 4, 2)
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        pair = SyntheticPair(
            view_a=image,
            view_b=image,
            warp_a_to_b=warp,
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        descriptor_rows = torch.cat([torch.eye(4), torch.zeros(4, 1)], dim=1)
        descriptors = descriptor_rows.T.reshape(1, 5, 2, 2)

        result = eval_py.match_pair_descriptor_maps(
            pair,
            descriptors,
            descriptors,
            max_keypoints=4,
            min_intensity=0.0,
            threshold_px=0.01,
        )

        self.assertEqual(result.matches, 4)
        self.assertEqual(result.correct, 4)
        self.assertEqual(result.wrong, 0)
        self.assertEqual(result.precision, 1.0)

    def test_match_pair_descriptor_maps_affine_filter_updates_match_count(self):
        image = torch.ones(1, 6, 6)
        warp = torch.zeros(6, 6, 2)
        yy, xx = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        pair = SyntheticPair(
            view_a=image,
            view_b=image,
            warp_a_to_b=warp,
            valid_mask=torch.ones(6, 6, dtype=torch.bool),
        )
        descriptor_rows = torch.eye(9)
        descriptors_a = descriptor_rows.T.reshape(1, 9, 3, 3)
        swapped = descriptor_rows.clone()
        swapped[[6, 8]] = swapped[[8, 6]]
        descriptors_b = swapped.T.reshape(1, 9, 3, 3)

        unfiltered = eval_py.match_pair_descriptor_maps(
            pair,
            descriptors_a,
            descriptors_b,
            max_keypoints=9,
            min_intensity=0.0,
            threshold_px=0.01,
        )
        filtered = eval_py.match_pair_descriptor_maps(
            pair,
            descriptors_a,
            descriptors_b,
            max_keypoints=9,
            min_intensity=0.0,
            threshold_px=0.01,
            geometry_filter="affine",
        )

        self.assertLess(filtered.matches, unfiltered.matches)
        self.assertEqual(filtered.matches, filtered.correct)

    def test_match_pair_descriptor_maps_can_use_graph_matcher(self):
        image = torch.ones(1, 4, 4)
        warp = torch.zeros(4, 4, 2)
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        pair = SyntheticPair(
            view_a=image,
            view_b=image,
            warp_a_to_b=warp,
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        descriptors = torch.eye(4).T.reshape(1, 4, 2, 2)

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self):
                raise AssertionError("not used")

            def graph_matcher(self, desc_a, keypoints_a, desc_b, keypoints_b):
                return pfm_model.GraphMatcherOutput(
                    logits=torch.empty(5, 5),
                    matches=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
                    scores=torch.tensor([0.9, 0.8], dtype=torch.float32),
                )

        result = eval_py.match_pair_descriptor_maps(
            pair,
            descriptors,
            descriptors,
            model=DummyModel(),
            matcher_mode="graph_matcher",
            max_keypoints=4,
            min_intensity=0.0,
            threshold_px=0.01,
        )

        self.assertEqual(result.matches, 2)
        self.assertEqual(result.correct, 2)

    def test_match_pair_descriptor_maps_can_disable_graph_fallback(self):
        image = torch.ones(1, 4, 4)
        warp = torch.zeros(4, 4, 2)
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        pair = SyntheticPair(
            view_a=image,
            view_b=image,
            warp_a_to_b=warp,
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        descriptors = torch.eye(4).T.reshape(1, 4, 2, 2)

        class RejectingGraphModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def graph_matcher(self, desc_a, keypoints_a, desc_b, keypoints_b):
                return pfm_model.GraphMatcherOutput(
                    logits=torch.empty(5, 5),
                    matches=torch.empty(0, 2, dtype=torch.long),
                    scores=torch.empty(0, dtype=torch.float32),
                )

        fallback_result = eval_py.match_pair_descriptor_maps(
            pair,
            descriptors,
            descriptors,
            model=RejectingGraphModel(),
            matcher_mode="graph_matcher",
            max_keypoints=4,
            max_matches=4,
            min_intensity=0.0,
            threshold_px=0.01,
        )
        strict_result = eval_py.match_pair_descriptor_maps(
            pair,
            descriptors,
            descriptors,
            model=RejectingGraphModel(),
            matcher_mode="graph_matcher",
            graph_fallback_mode="none",
            max_keypoints=4,
            max_matches=4,
            min_intensity=0.0,
            threshold_px=0.01,
        )

        self.assertGreater(fallback_result.matches, 0)
        self.assertEqual(strict_result.matches, 0)

    def test_calibrated_graph_matches_can_lower_dustbin_rejection(self):
        logits = torch.tensor(
            [
                [5.0, -5.0, 6.0],
                [-5.0, 5.0, 6.0],
                [-5.0, -5.0, 0.0],
            ],
            dtype=torch.float32,
        )

        rejected_matches, _ = eval_py.calibrated_graph_matches_from_logits(logits, count_a=2, count_b=2)
        accepted_matches, _ = eval_py.calibrated_graph_matches_from_logits(
            logits,
            count_a=2,
            count_b=2,
            dustbin_delta=-2.0,
        )

        self.assertEqual(rejected_matches.tolist(), [])
        self.assertEqual(accepted_matches.tolist(), [[0, 0], [1, 1]])

    def test_graph_matcher_matches_can_filter_by_raw_margin(self):
        desc_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        desc_b = torch.tensor([[1.0, 0.0], [0.99, 0.01]])
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        logits = torch.tensor(
            [
                [5.0, -5.0, -5.0],
                [-5.0, 5.0, -5.0],
                [-5.0, -5.0, 0.0],
            ],
            dtype=torch.float32,
        )

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def graph_matcher(self, desc_a, keypoints_a, desc_b, keypoints_b):
                return pfm_model.GraphMatcherOutput(
                    logits=logits.to(desc_a.device),
                    matches=torch.empty(0, 2, dtype=torch.long),
                    scores=torch.empty(0),
                )

        matches, _ = eval_py.graph_matcher_matches(
            DummyModel(),
            desc_a,
            keypoints,
            desc_b,
            keypoints,
            max_matches=8,
            graph_dustbin_delta=-0.01,
            graph_min_raw_margin=0.05,
        )

        self.assertEqual(matches.tolist(), [])

    def test_graph_matcher_matches_passes_pruning_and_early_stop_thresholds(self):
        desc_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        desc_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.width_prune_min_score = None
                self.early_stop_min_confidence = None

            def graph_matcher(
                self,
                desc_a,
                keypoints_a,
                desc_b,
                keypoints_b,
                *,
                width_prune_min_score=-1.0,
                early_stop_min_confidence=-1.0,
            ):
                self.width_prune_min_score = width_prune_min_score
                self.early_stop_min_confidence = early_stop_min_confidence
                return pfm_model.GraphMatcherOutput(
                    logits=torch.empty(3, 3),
                    matches=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
                    scores=torch.tensor([0.9, 0.8], dtype=torch.float32),
                )

        model = DummyModel()
        matches, _ = eval_py.graph_matcher_matches(
            model,
            desc_a,
            keypoints,
            desc_b,
            keypoints,
            max_matches=8,
            graph_width_prune_min_score=0.25,
            graph_early_stop_min_confidence=0.75,
        )

        self.assertEqual(model.width_prune_min_score, 0.25)
        self.assertEqual(model.early_stop_min_confidence, 0.75)
        self.assertEqual(matches.tolist(), [[0, 0], [1, 1]])

    def test_graph_matcher_matches_filters_low_accept_probability(self):
        desc_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        desc_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def graph_matcher(self, desc_a, keypoints_a, desc_b, keypoints_b):
                return pfm_model.GraphMatcherOutput(
                    logits=torch.empty(3, 3),
                    matches=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
                    scores=torch.tensor([0.9, 0.8], dtype=torch.float32),
                    accept_logits=torch.tensor([[4.0, -4.0], [-4.0, 0.0]], dtype=torch.float32),
                )

        matches, scores = eval_py.graph_matcher_matches(
            DummyModel(),
            desc_a,
            keypoints,
            desc_b,
            keypoints,
            max_matches=8,
            graph_min_accept_probability=0.75,
        )

        self.assertEqual(matches.tolist(), [[0, 0]])
        self.assertTrue(torch.allclose(scores.cpu(), torch.tensor([0.9], dtype=torch.float32), atol=1.0e-6))

    def test_graph_matcher_matches_reports_adaptive_stats(self):
        desc_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        desc_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def graph_matcher(self, desc_a, keypoints_a, desc_b, keypoints_b):
                return pfm_model.GraphMatcherOutput(
                    logits=torch.empty(3, 3),
                    matches=torch.tensor([[0, 0]], dtype=torch.long),
                    scores=torch.tensor([0.9], dtype=torch.float32),
                    executed_layers=2,
                    input_keypoints_a=2,
                    input_keypoints_b=2,
                    kept_keypoints_a=1,
                    kept_keypoints_b=1,
                    pruned_keypoints_a=1,
                    pruned_keypoints_b=1,
                )

        graph_stats = {}
        matches, _ = eval_py.graph_matcher_matches(
            DummyModel(),
            desc_a,
            keypoints,
            desc_b,
            keypoints,
            max_matches=8,
            graph_stats=graph_stats,
        )

        self.assertEqual(matches.tolist(), [[0, 0]])
        self.assertEqual(graph_stats["graph_executed_layers"], 2)
        self.assertEqual(graph_stats["graph_pruned_keypoints_a"], 1)
        self.assertEqual(graph_stats["graph_kept_keypoints_b"], 1)

    def test_mutual_nearest_matches_reject_one_way_descriptor_candidates(self):
        desc_a = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        desc_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        matches, _ = eval_py.mutual_nearest_matches(desc_a, desc_b, max_matches=8, min_score=-1.0)

        self.assertEqual(matches.tolist(), [[0, 0], [2, 1]])

    def test_zero_max_matches_keeps_all_unique_descriptor_matches(self):
        desc_a = torch.eye(5, dtype=torch.float32)[:4]
        desc_b = torch.eye(5, dtype=torch.float32)[:4]

        matches, _ = eval_py.greedy_unique_matches(desc_a, desc_b, max_matches=0, min_score=-1.0)

        self.assertEqual(matches.tolist(), [[0, 0], [1, 1], [2, 2], [3, 3]])

    def test_mutual_nearest_matches_can_require_best_second_margin(self):
        desc_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        desc_b = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])

        matches, _ = eval_py.mutual_nearest_matches(
            desc_a,
            desc_b,
            max_matches=8,
            min_score=-1.0,
            min_margin=0.05,
        )

        self.assertEqual(matches.tolist(), [[1, 2]])

    def test_affine_consistency_filter_removes_geometric_outliers(self):
        points_a = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [9.0, 9.0]],
            dtype=torch.float32,
        )
        points_b = points_a + torch.tensor([5.0, 7.0])
        points_b[-1] = torch.tensor([-3.0, 4.0])
        matches = torch.stack([torch.arange(points_a.size(0)), torch.arange(points_a.size(0))], dim=1)
        scores = torch.linspace(1.0, 0.5, points_a.size(0))

        kept_matches, kept_scores = eval_py.filter_affine_consistent_matches(
            points_a,
            points_b,
            matches,
            scores,
            threshold_px=0.01,
            iterations=64,
            min_inliers=4,
        )

        self.assertEqual(kept_matches.size(0), 6)
        self.assertEqual(kept_scores.size(0), 6)
        self.assertNotIn([6, 6], kept_matches.tolist())

    def test_local_displacement_filter_removes_outlier_without_global_affine_assumption(self):
        points_a = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [8.0, 8.0]],
            dtype=torch.float32,
        )
        points_b = points_a + torch.tensor([3.0, 4.0])
        points_b[-1] = torch.tensor([50.0, -20.0])
        matches = torch.stack([torch.arange(points_a.size(0)), torch.arange(points_a.size(0))], dim=1)
        scores = torch.linspace(1.0, 0.5, points_a.size(0))

        kept_matches, kept_scores = eval_py.filter_local_displacement_consistent_matches(
            points_a,
            points_b,
            matches,
            scores,
            threshold_px=1.0,
            neighbors=4,
            min_inliers=3,
        )

        self.assertEqual(kept_matches.size(0), 4)
        self.assertEqual(kept_scores.size(0), 4)
        self.assertNotIn([4, 4], kept_matches.tolist())

    def test_limit_pair_paths_can_take_seeded_random_subset(self):
        paths = [Path(f"source_000001/pair_{index:06d}.pt") for index in range(10)]

        first = eval_py.limit_pair_paths(paths, limit_pairs=4, sample_seed=17)
        second = eval_py.limit_pair_paths(list(reversed(paths)), limit_pairs=4, sample_seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertNotEqual(first, paths[:4])

    def test_selected_pair_paths_discovers_all_pairs_before_seeded_limit(self):
        paths = [Path(f"source_000001/pair_{index:06d}.pt") for index in range(10)]
        args = Namespace(
            cache_dir=[Path("cache")],
            limit_pairs=4,
            sample_seed=23,
            exclude_self_pairs=False,
            hard_summary=[],
            hard_limit=64,
            hard_min_matches=4,
            hard_max_precision=0.9,
        )

        with mock.patch.object(eval_py, "discover_pair_archives", return_value=paths) as discover:
            selected = eval_py.selected_pair_paths(args)

        self.assertEqual(len(selected), 4)
        self.assertNotEqual(selected, paths[:4])
        discover.assert_called_once_with([Path("cache")], limit_pairs=0, exclude_self_pairs=False)

    def test_selected_pair_paths_passes_exclude_self_pairs_to_discovery(self):
        args = Namespace(
            cache_dir=[Path("cache")],
            limit_pairs=0,
            sample_seed=None,
            exclude_self_pairs=True,
            hard_summary=[],
            hard_limit=64,
            hard_min_matches=4,
            hard_max_precision=0.9,
        )

        with mock.patch.object(eval_py, "discover_pair_archives", return_value=[]) as discover:
            eval_py.selected_pair_paths(args)

        discover.assert_called_once_with([Path("cache")], limit_pairs=0, exclude_self_pairs=True)

    def test_evaluate_pair_path_skips_low_target_gradient_before_model_forward(self):
        image = torch.ones(1, 8, 8) * 0.5
        warp = torch.zeros(8, 8, 2)
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        pair = SyntheticPair(
            view_a=image,
            view_b=image,
            warp_a_to_b=warp,
            valid_mask=torch.ones(8, 8, dtype=torch.bool),
        )

        with mock.patch.object(eval_py, "load_libtorch_pair_archive", return_value=pair):
            result = eval_py.evaluate_pair_path(
                mock.Mock(),
                Path("pair.pt"),
                device=torch.device("cpu"),
                mode="blend",
                texture_blend_weight=1.0,
                max_keypoints=16,
                min_intensity=0.0,
                texture_fraction=1.0,
                threshold_px=5.0,
                topk=1,
                max_matches=16,
                min_score=-1.0,
                min_margin=0.0,
                min_target_gradient=1.0,
                min_target_local_contrast=0.0,
                mutual=True,
                geometry_filter="local",
                keypoint_spatial_bins=0,
            )

        self.assertEqual(result.matches, 0)
        self.assertEqual(result.correct, 0)
        self.assertEqual(result.precision, 0.0)

    def test_evaluate_pair_path_skips_low_target_local_contrast_before_model_forward(self):
        image = torch.ones(1, 8, 8) * 0.5
        warp = torch.zeros(8, 8, 2)
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        pair = SyntheticPair(
            view_a=image,
            view_b=image,
            warp_a_to_b=warp,
            valid_mask=torch.ones(8, 8, dtype=torch.bool),
        )

        with mock.patch.object(eval_py, "load_libtorch_pair_archive", return_value=pair):
            result = eval_py.evaluate_pair_path(
                mock.Mock(),
                Path("pair.pt"),
                device=torch.device("cpu"),
                mode="blend",
                texture_blend_weight=1.0,
                max_keypoints=16,
                min_intensity=0.0,
                texture_fraction=1.0,
                threshold_px=5.0,
                topk=1,
                max_matches=16,
                min_score=-1.0,
                min_margin=0.0,
                min_target_gradient=0.0,
                min_target_local_contrast=1.0,
                mutual=True,
                geometry_filter="local",
                keypoint_spatial_bins=0,
            )

        self.assertEqual(result.matches, 0)
        self.assertEqual(result.correct, 0)
        self.assertEqual(result.precision, 0.0)


if __name__ == "__main__":
    unittest.main()
