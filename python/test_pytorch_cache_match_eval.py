import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytorch_cache_match_eval as eval_py
from patch_descriptor_training import SyntheticPair


class PyTorchCacheMatchEvalTest(unittest.TestCase):
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

    def test_mutual_nearest_matches_reject_one_way_descriptor_candidates(self):
        desc_a = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        desc_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        matches, _ = eval_py.mutual_nearest_matches(desc_a, desc_b, max_matches=8, min_score=-1.0)

        self.assertEqual(matches.tolist(), [[0, 0], [2, 1]])

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


if __name__ == "__main__":
    unittest.main()
