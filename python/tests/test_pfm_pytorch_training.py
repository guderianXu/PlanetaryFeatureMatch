import sys
import unittest
import argparse
import random
import tempfile
from unittest import mock
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pfm_model
import pfm_pytorch_training as train
from patch_descriptor_training import SyntheticPair


class PFMPyTorchTrainingTest(unittest.TestCase):
    def test_epoch_shuffle_sampler_covers_each_pair_once_per_epoch(self):
        paths = [Path(f"pair_{idx:03d}.pt") for idx in range(5)]
        sampler = train.EpochShuffleSampler(paths, batch_pairs=1, seed=7)

        first_epoch = [sampler.batch_for_step(step)[0] for step in range(1, 6)]
        second_epoch = [sampler.batch_for_step(step)[0] for step in range(6, 11)]

        self.assertEqual(set(first_epoch), set(paths))
        self.assertEqual(set(second_epoch), set(paths))
        self.assertEqual(len(first_epoch), len(set(first_epoch)))
        self.assertEqual(len(second_epoch), len(set(second_epoch)))

    def test_pair_archive_cache_reuses_cpu_archive_and_moves_to_device(self):
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2) * 2.0,
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair_000001.pt"
            path.touch()
            with mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair) as loader:
                cache = train.PairArchiveCache(max_items=2)
                first = cache.get(path, device=torch.device("cpu"))
                second = cache.get(path, device=torch.device("cpu"))

        self.assertTrue(torch.equal(first.view_b, second.view_b))
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    def test_sample_feature_correspondences_scales_image_pixels_to_feature_grid(self):
        view = torch.ones(1, 5, 9)
        warp = torch.zeros(5, 9, 2)
        yy, xx = torch.meshgrid(torch.arange(5), torch.arange(9), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        valid = torch.zeros(5, 9, dtype=torch.bool)
        valid[2, 4] = True
        pair = SyntheticPair(view_a=view, view_b=view, warp_a_to_b=warp, valid_mask=valid)

        points_a, points_b = train.sample_feature_correspondences(
            pair,
            feature_height=3,
            feature_width=5,
            count=8,
            min_intensity=0.01,
            generator=torch.Generator().manual_seed(3),
        )

        self.assertEqual(tuple(points_a.shape), (1, 2))
        self.assertTrue(torch.allclose(points_a[0], torch.tensor([2.0, 1.0])))
        self.assertTrue(torch.allclose(points_b[0], torch.tensor([2.0, 1.0])))

    def test_sample_feature_correspondences_filters_dark_target_pixels(self):
        view_a = torch.ones(1, 5, 5)
        view_b = torch.ones(1, 5, 5)
        view_b[:, 2, 2] = 0.0
        warp = torch.zeros(5, 5, 2)
        yy, xx = torch.meshgrid(torch.arange(5), torch.arange(5), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        valid = torch.zeros(5, 5, dtype=torch.bool)
        valid[2, 2] = True
        pair = SyntheticPair(view_a=view_a, view_b=view_b, warp_a_to_b=warp, valid_mask=valid)

        points_a, points_b = train.sample_feature_correspondences(
            pair,
            feature_height=5,
            feature_width=5,
            count=8,
            min_intensity=0.01,
            generator=torch.Generator().manual_seed(3),
        )

        self.assertEqual(points_a.numel(), 0)
        self.assertEqual(points_b.numel(), 0)

    def test_sample_feature_correspondences_can_reserve_weak_texture_points(self):
        view_a = torch.full((1, 6, 6), 0.5)
        view_b = torch.full((1, 6, 6), 0.5)
        view_a[:, :, 4:] = torch.tensor([0.0, 1.0]).repeat(6, 1)
        view_b[:, :, 4:] = torch.tensor([0.0, 1.0]).repeat(6, 1)
        warp = torch.zeros(6, 6, 2)
        yy, xx = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        pair = SyntheticPair(view_a=view_a, view_b=view_b, warp_a_to_b=warp, valid_mask=torch.ones(6, 6, dtype=torch.bool))

        points_a, _ = train.sample_feature_correspondences(
            pair,
            feature_height=6,
            feature_width=6,
            count=8,
            min_intensity=0.0,
            weak_texture_fraction=0.5,
            generator=torch.Generator().manual_seed(11),
        )

        self.assertGreaterEqual(int((points_a[:, 0] < 4.0).sum()), 4)

    def test_sample_feature_correspondences_can_cover_spatial_bins(self):
        view = torch.ones(1, 8, 8)
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        warp = torch.stack([xx.to(torch.float32), yy.to(torch.float32)], dim=-1)
        pair = SyntheticPair(view_a=view, view_b=view, warp_a_to_b=warp, valid_mask=torch.ones(8, 8, dtype=torch.bool))

        points_a, _ = train.sample_feature_correspondences(
            pair,
            feature_height=8,
            feature_width=8,
            count=4,
            min_intensity=0.0,
            spatial_bins=2,
            generator=torch.Generator().manual_seed(17),
        )

        cell_ids = (points_a[:, 0] >= 4.0).to(torch.long) + 2 * (points_a[:, 1] >= 4.0).to(torch.long)
        self.assertEqual(set(cell_ids.tolist()), {0, 1, 2, 3})

    def test_sample_feature_correspondences_combines_weak_texture_and_spatial_bins(self):
        view_a = torch.full((1, 8, 8), 0.5)
        view_b = torch.full((1, 8, 8), 0.5)
        view_a[:, :, 4:] = torch.tensor([0.0, 1.0, 0.0, 1.0]).repeat(8, 1)
        view_b[:, :, 4:] = torch.tensor([0.0, 1.0, 0.0, 1.0]).repeat(8, 1)
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        warp = torch.stack([xx.to(torch.float32), yy.to(torch.float32)], dim=-1)
        pair = SyntheticPair(view_a=view_a, view_b=view_b, warp_a_to_b=warp, valid_mask=torch.ones(8, 8, dtype=torch.bool))

        points_a, _ = train.sample_feature_correspondences(
            pair,
            feature_height=8,
            feature_width=8,
            count=8,
            min_intensity=0.0,
            weak_texture_fraction=0.5,
            spatial_bins=2,
            generator=torch.Generator().manual_seed(19),
        )

        cell_ids = (points_a[:, 0] >= 4.0).to(torch.long) + 2 * (points_a[:, 1] >= 4.0).to(torch.long)
        self.assertEqual(set(cell_ids.tolist()), {0, 1, 2, 3})
        self.assertGreaterEqual(int((points_a[:, 0] < 4.0).sum()), 4)

    def test_resize_pair_for_training_scales_images_warp_and_mask(self):
        view = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        warp = torch.stack([xx.to(torch.float32), yy.to(torch.float32)], dim=-1)
        valid = torch.zeros(4, 4, dtype=torch.bool)
        valid[3, 3] = True
        pair = SyntheticPair(view_a=view, view_b=view + 1.0, warp_a_to_b=warp, valid_mask=valid)

        resized = train.resize_pair_for_training(pair, max_image_size=2)

        self.assertEqual(tuple(resized.view_a.shape), (1, 2, 2))
        self.assertEqual(tuple(resized.view_b.shape), (1, 2, 2))
        self.assertEqual(tuple(resized.warp_a_to_b.shape), (2, 2, 2))
        self.assertEqual(tuple(resized.valid_mask.shape), (2, 2))
        self.assertTrue(resized.valid_mask[1, 1])
        self.assertTrue(torch.allclose(resized.warp_a_to_b[1, 1], torch.tensor([1.0, 1.0])))

    def test_crop_pair_for_training_preserves_native_resolution_warp_coordinates(self):
        view = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        warp = torch.stack([xx.to(torch.float32), yy.to(torch.float32)], dim=-1)
        valid = torch.ones(4, 4, dtype=torch.bool)
        pair = SyntheticPair(view_a=view, view_b=view + 100.0, warp_a_to_b=warp, valid_mask=valid)

        cropped = train.crop_pair_for_training(pair, crop_size=2)

        self.assertTrue(torch.equal(cropped.view_a, view[:, 1:3, 1:3]))
        self.assertTrue(torch.equal(cropped.view_b, (view + 100.0)[:, 1:3, 1:3]))
        self.assertEqual(tuple(cropped.warp_a_to_b.shape), (2, 2, 2))
        self.assertTrue(torch.allclose(cropped.warp_a_to_b[0, 0], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.all(cropped.valid_mask))

    def test_read_pseudo_label_matches_groups_csv_rows_and_ignores_bad_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "pseudo_labels.csv"
            csv_path.write_text(
                "pair_pt,ax,ay,bx,by,matcher\n"
                "cache/source_a/pair_000001.pt,4,2,8,4,RootSIFT\n"
                "cache/source_a/pair_000001.pt,nan,2,8,4,RootSIFT\n"
                "cache/source_b/pair_000002.pt,1,3,5,7,RootSIFT\n",
                encoding="utf-8",
            )

            labels = train.read_pseudo_label_matches([csv_path])

        self.assertEqual(set(labels), {"cache/source_a/pair_000001.pt", "cache/source_b/pair_000002.pt"})
        self.assertTrue(
            torch.allclose(
                labels["cache/source_a/pair_000001.pt"].points_a_xy,
                torch.tensor([[4.0, 2.0]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                labels["cache/source_b/pair_000002.pt"].points_b_xy,
                torch.tensor([[5.0, 7.0]]),
            )
        )

    def test_read_false_match_labels_groups_valid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "false_matches.csv"
            csv_path.write_text(
                "pair_pt,ax,ay,bx,by,error_px,score,margin,style,gate\n"
                "cache/source_a/pair_000001.pt,4,2,8,4,12.5,0.9,0.1,numeric,viewpoint\n"
                "cache/source_a/pair_000001.pt,nan,2,8,4,12.5,0.9,0.1,numeric,viewpoint\n"
                "cache/source_b/pair_000002.pt,1,3,5,7,8.0,0.8,0.2,timestamp,compound\n",
                encoding="utf-8",
            )

            labels = train.read_false_match_labels([csv_path])

        self.assertEqual(set(labels), {"cache/source_a/pair_000001.pt", "cache/source_b/pair_000002.pt"})
        self.assertTrue(
            torch.allclose(
                labels["cache/source_a/pair_000001.pt"].points_a_xy,
                torch.tensor([[4.0, 2.0]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                labels["cache/source_b/pair_000002.pt"].points_b_xy,
                torch.tensor([[5.0, 7.0]]),
            )
        )

    def test_pseudo_label_feature_correspondences_scales_without_basename_fallback(self):
        view = torch.ones(1, 5, 9)
        pair = SyntheticPair(
            view_a=view,
            view_b=view,
            warp_a_to_b=torch.zeros(5, 9, 2),
            valid_mask=torch.ones(5, 9, dtype=torch.bool),
        )
        labels = {
            "cache/source_a/pair_000001.pt": train.PseudoLabelMatches(
                points_a_xy=torch.tensor([[4.0, 2.0]]),
                points_b_xy=torch.tensor([[8.0, 4.0]]),
            )
        }

        points_a, points_b = train.pseudo_label_feature_correspondences(
            Path("cache/source_a/pair_000001.pt"),
            pair,
            labels,
            feature_height=3,
            feature_width=5,
            max_points=8,
            generator=torch.Generator().manual_seed(3),
        )
        missing_a, missing_b = train.pseudo_label_feature_correspondences(
            Path("different/source_b/pair_000001.pt"),
            pair,
            labels,
            feature_height=3,
            feature_width=5,
            max_points=8,
            generator=torch.Generator().manual_seed(3),
        )

        self.assertTrue(torch.allclose(points_a, torch.tensor([[2.0, 1.0]])))
        self.assertTrue(torch.allclose(points_b, torch.tensor([[4.0, 2.0]])))
        self.assertEqual(missing_a.numel(), 0)
        self.assertEqual(missing_b.numel(), 0)

    def test_false_match_feature_correspondences_scales_without_basename_fallback(self):
        view = torch.ones(1, 5, 9)
        pair = SyntheticPair(
            view_a=view,
            view_b=view,
            warp_a_to_b=torch.zeros(5, 9, 2),
            valid_mask=torch.ones(5, 9, dtype=torch.bool),
        )
        labels = {
            "cache/source_a/pair_000001.pt": train.FalseMatchLabels(
                points_a_xy=torch.tensor([[4.0, 2.0]]),
                points_b_xy=torch.tensor([[8.0, 4.0]]),
            )
        }

        points_a, points_b = train.false_match_feature_correspondences(
            Path("cache/source_a/pair_000001.pt"),
            pair,
            labels,
            feature_height=3,
            feature_width=5,
            max_points=8,
            generator=torch.Generator().manual_seed(3),
        )
        missing_a, missing_b = train.false_match_feature_correspondences(
            Path("different/source_b/pair_000001.pt"),
            pair,
            labels,
            feature_height=3,
            feature_width=5,
            max_points=8,
            generator=torch.Generator().manual_seed(3),
        )

        self.assertTrue(torch.allclose(points_a, torch.tensor([[2.0, 1.0]])))
        self.assertTrue(torch.allclose(points_b, torch.tensor([[4.0, 2.0]])))
        self.assertEqual(missing_a.numel(), 0)
        self.assertEqual(missing_b.numel(), 0)

    def test_select_pseudo_labeled_training_pairs_uses_exact_path_keys(self):
        pair_paths = [
            Path("cache/source_a/pair_000001.pt"),
            Path("different/source_b/pair_000001.pt"),
            Path("cache/source_c/pair_000003.pt"),
        ]
        labels = {
            "cache/source_a/pair_000001.pt": train.PseudoLabelMatches(torch.ones(1, 2), torch.ones(1, 2)),
            "cache/source_c/pair_000003.pt": train.PseudoLabelMatches(torch.ones(1, 2), torch.ones(1, 2)),
        }

        selected = train.select_pseudo_labeled_training_pairs(pair_paths, labels)

        self.assertEqual(selected, [pair_paths[0], pair_paths[2]])

    def test_sample_descriptors_matches_grid_dtype_to_descriptor_map(self):
        descriptor_map = torch.ones(1, 2, 3, 3, dtype=torch.float16)
        points = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        sampled = torch.ones(1, 2, 1, 1, dtype=torch.float16)

        with mock.patch.object(train.F, "grid_sample", return_value=sampled) as grid_sample:
            result = train.sample_descriptors(descriptor_map, points)

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(grid_sample.call_args.args[1].dtype, torch.float16)

    def test_sample_descriptors_uses_xy_coordinates(self):
        descriptor_map = torch.zeros(1, 2, 3, 4)
        yy, xx = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
        descriptor_map[0, 0] = xx
        descriptor_map[0, 1] = yy
        points = torch.tensor([[2.0, 1.0], [3.0, 2.0]], dtype=torch.float32)

        descriptors = train.sample_descriptors(descriptor_map, points)

        self.assertTrue(torch.allclose(descriptors, torch.tensor([[2.0, 1.0], [3.0, 2.0]])))

    def test_descriptor_map_loss_is_low_for_identity_descriptors(self):
        descriptor_map = torch.zeros(1, 4, 4, 4)
        for index, (x, y) in enumerate([(0, 0), (1, 1), (2, 2), (3, 3)]):
            descriptor_map[0, index, y, x] = 1.0
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        loss, metrics = train.descriptor_map_pair_loss(
            descriptor_map,
            descriptor_map,
            points,
            points,
            temperature=0.05,
        )

        self.assertLess(float(loss), 0.2)
        self.assertGreater(metrics["top1_accuracy"], 0.99)

    def test_hard_negative_margin_penalizes_repeated_distractors(self):
        desc_a = torch.eye(4)
        desc_b_good = desc_a.clone()
        desc_b_bad = desc_a.clone()
        desc_b_bad[1] = desc_b_bad[0]

        good = train.hard_negative_margin_loss(desc_a, desc_b_good, margin=0.2)
        bad = train.hard_negative_margin_loss(desc_a, desc_b_bad, margin=0.2)

        self.assertGreater(float(bad), float(good))

    def test_normalize_descriptor_batch_keeps_zero_descriptor_gradients_finite(self):
        descriptors = torch.zeros(3, 4, requires_grad=True)

        normalized = train.normalize_descriptor_batch(descriptors, eps=1.0e-4)
        normalized.sum().backward()

        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.isfinite(descriptors.grad).all())
        self.assertLessEqual(float(descriptors.grad.abs().max()), 10000.0)

    def test_warp_aware_hard_negative_penalizes_far_distractor(self):
        descriptor_a = torch.zeros(1, 2, 1, 3)
        descriptor_b = torch.zeros(1, 2, 1, 3)
        descriptor_a[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 2] = torch.tensor([1.0, 0.0])
        points_a = torch.tensor([[0.0, 0.0]])
        points_b = torch.tensor([[0.0, 0.0]])

        loss = train.warp_aware_hard_negative_loss(
            descriptor_a,
            descriptor_b,
            points_a,
            points_b,
            negative_radius=0.25,
            margin=0.2,
        )

        self.assertGreater(float(loss), 0.03)

    def test_warp_aware_hard_negative_masks_near_positive_neighbor(self):
        descriptor_a = torch.zeros(1, 2, 1, 3)
        descriptor_b = torch.zeros(1, 2, 1, 3)
        descriptor_a[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 1] = torch.tensor([1.0, 0.0])
        points_a = torch.tensor([[0.0, 0.0]])
        points_b = torch.tensor([[0.0, 0.0]])

        masked = train.warp_aware_hard_negative_loss(
            descriptor_a,
            descriptor_b,
            points_a,
            points_b,
            negative_radius=1.25,
            margin=0.2,
        )
        unmasked = train.warp_aware_hard_negative_loss(
            descriptor_a,
            descriptor_b,
            points_a,
            points_b,
            negative_radius=0.25,
            margin=0.2,
        )

        self.assertLess(float(masked), float(unmasked))

    def test_reliability_supervision_losses_reward_expected_points(self):
        positive = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        negative = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        high_positive = torch.tensor([[[[0.9, 0.1]]]], dtype=torch.float32)
        high_negative = torch.tensor([[[[0.1, 0.9]]]], dtype=torch.float32)

        self.assertLess(
            float(train.matchability_supervision_loss(high_positive, positive, negative)),
            float(train.matchability_supervision_loss(high_negative, positive, negative)),
        )
        self.assertLess(
            float(train.no_match_prior_supervision_loss(high_negative, negative, positive)),
            float(train.no_match_prior_supervision_loss(high_positive, negative, positive)),
        )
        self.assertLess(
            float(train.descriptor_uncertainty_supervision_loss(high_negative, negative, positive)),
            float(train.descriptor_uncertainty_supervision_loss(high_positive, negative, positive)),
        )

    def test_rotation_consistency_losses_are_low_for_rot90_aligned_maps(self):
        descriptors = torch.zeros(1, 2, 2, 2)
        descriptors[0, 0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        descriptors[0, 1] = 1.0 - descriptors[0, 0]
        rotated_descriptors = torch.rot90(descriptors, k=1, dims=(-2, -1))
        orientation = torch.zeros(1, 2, 2, 2)
        orientation[:, 0] = 1.0
        rotated_orientation = torch.zeros_like(orientation)
        rotated_orientation[:, 1] = 1.0
        rotated_orientation = torch.rot90(rotated_orientation, k=1, dims=(-2, -1))
        scale = torch.ones(1, 1, 2, 2)
        affine = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, 2, 2)
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

        self.assertLess(
            float(train.rotation_descriptor_consistency_loss(descriptors, rotated_descriptors, points, 90)),
            1.0e-5,
        )
        self.assertLess(
            float(train.orientation_consistency_loss(orientation, rotated_orientation, points, 90)),
            1.0e-5,
        )
        self.assertLess(float(train.scale_consistency_loss(scale, scale, points, 90)), 1.0e-5)
        self.assertLess(float(train.affine_consistency_loss(affine, affine, points, 0)), 1.0e-5)

    def test_affine_regularization_penalizes_degenerate_local_geometry(self):
        identity = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, 2, 2)
        degenerate = torch.tensor([1.8, 0.0, 0.0, 0.2], dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, 2, 2)

        identity_loss, identity_metrics = train.affine_regularization_loss(identity, return_metrics=True)
        bad_loss, bad_metrics = train.affine_regularization_loss(degenerate, return_metrics=True)

        self.assertLess(float(identity_loss), 1.0e-6)
        self.assertGreater(float(bad_loss), 0.1)
        self.assertAlmostEqual(float(identity_metrics["affine_det_mean"]), 1.0)
        self.assertGreater(float(bad_metrics["affine_condition_mean"]), 1.0)

    def test_descriptor_false_match_suppression_penalizes_high_wrong_candidate(self):
        descriptor_a = torch.zeros(1, 2, 1, 3)
        descriptor_b = torch.zeros(1, 2, 1, 3)
        descriptor_a[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 0] = torch.tensor([0.0, 1.0])
        descriptor_b[0, :, 0, 2] = torch.tensor([1.0, 0.0])
        points_a = torch.tensor([[0.0, 0.0]])
        points_b = torch.tensor([[0.0, 0.0]])

        loss = train.descriptor_false_match_suppression_loss(
            descriptor_a,
            descriptor_b,
            points_a,
            points_b,
            negative_radius=0.25,
            max_false_score=0.2,
            topk=1,
        )

        self.assertGreater(float(loss), 0.5)

    def test_descriptor_false_match_suppression_masks_near_target_neighbor(self):
        descriptor_a = torch.zeros(1, 2, 1, 3)
        descriptor_b = torch.zeros(1, 2, 1, 3)
        descriptor_a[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 0] = torch.tensor([0.0, 1.0])
        descriptor_b[0, :, 0, 1] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 2] = torch.tensor([0.0, 1.0])
        points_a = torch.tensor([[0.0, 0.0]])
        points_b = torch.tensor([[0.0, 0.0]])

        masked = train.descriptor_false_match_suppression_loss(
            descriptor_a,
            descriptor_b,
            points_a,
            points_b,
            negative_radius=1.25,
            max_false_score=0.2,
            topk=1,
        )
        unmasked = train.descriptor_false_match_suppression_loss(
            descriptor_a,
            descriptor_b,
            points_a,
            points_b,
            negative_radius=0.25,
            max_false_score=0.2,
            topk=1,
        )

        self.assertLess(float(masked), float(unmasked))

    def test_paired_cyclic_similarity_rejects_quarter_channel_shift(self):
        desc_a = torch.zeros(1, 8)
        desc_b = torch.zeros(1, 8)
        desc_a[0, 1] = 1.0
        desc_b[0, 3] = 1.0

        similarity = train.paired_cyclic_similarity(desc_a, desc_b)

        self.assertAlmostEqual(float(similarity[0]), 0.0, places=6)

    def test_false_match_negative_loss_penalizes_high_wrong_pair(self):
        descriptor_a = torch.zeros(1, 8, 1, 1)
        descriptor_b = torch.zeros(1, 8, 1, 1)
        descriptor_a[0, 1, 0, 0] = 1.0
        descriptor_b[0, 1, 0, 0] = 1.0
        points = torch.tensor([[0.0, 0.0]])

        loss = train.false_match_negative_loss(
            descriptor_a,
            descriptor_b,
            points,
            points,
            max_false_score=0.2,
        )

        self.assertGreater(float(loss), 0.5)

    def test_online_false_match_feature_correspondences_mines_wrong_mutual_match(self):
        view = torch.ones(1, 4, 4)
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        warp = torch.stack([xx.to(torch.float32), yy.to(torch.float32)], dim=-1)
        pair = SyntheticPair(
            view_a=view,
            view_b=view,
            warp_a_to_b=warp,
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        descriptor_a = torch.zeros(1, 2, 2, 2)
        descriptor_b = torch.zeros(1, 2, 2, 2)
        descriptor_a[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_a[0, :, 0, 1] = torch.tensor([0.0, 1.0])
        descriptor_b[0, :, 0, 0] = torch.tensor([0.0, 1.0])
        descriptor_b[0, :, 0, 1] = torch.tensor([1.0, 0.0])

        false_a, false_b = train.online_false_match_feature_correspondences(
            pair,
            descriptor_a,
            descriptor_b,
            max_keypoints=4,
            max_matches=0,
            min_intensity=0.01,
            min_score=0.0,
            min_margin=0.0,
            threshold_px=0.5,
            max_points=8,
            generator=torch.Generator().manual_seed(7),
        )

        self.assertTrue(any(torch.allclose(point, torch.tensor([0.0, 0.0])) for point in false_a))
        self.assertTrue(any(torch.allclose(point, torch.tensor([1.0, 0.0])) for point in false_b))

    def test_descriptor_map_pair_loss_respects_warp_hard_negative_weight(self):
        descriptor_a = torch.zeros(1, 2, 1, 3)
        descriptor_b = torch.zeros(1, 2, 1, 3)
        descriptor_a[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 2] = torch.tensor([1.0, 0.0])
        points_a = torch.tensor([[0.0, 0.0]])
        points_b = torch.tensor([[0.0, 0.0]])

        baseline, _ = train.descriptor_map_pair_loss(
            descriptor_a,
            descriptor_b,
            points_a,
            points_b,
            hard_negative_weight=0.0,
            warp_hard_negative_weight=0.0,
        )
        penalized, _ = train.descriptor_map_pair_loss(
            descriptor_a,
            descriptor_b,
            points_a,
            points_b,
            hard_negative_weight=0.0,
            warp_hard_negative_weight=2.0,
            warp_hard_negative_radius=0.25,
        )

        self.assertGreater(float(penalized.detach()), float(baseline.detach()))

    def test_descriptor_map_pair_loss_respects_abstention_weight(self):
        descriptor_a = torch.zeros(1, 2, 1, 3)
        descriptor_b = torch.zeros(1, 2, 1, 3)
        descriptor_a[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_b[0, :, 0, 2] = torch.tensor([1.0, 0.0])
        points_a = torch.tensor([[0.0, 0.0]])
        points_b = torch.tensor([[0.0, 0.0]])

        baseline, _ = train.descriptor_map_pair_loss(
            descriptor_a,
            descriptor_b,
            points_a,
            points_b,
            hard_negative_weight=0.0,
            abstention_weight=0.0,
        )
        penalized, _ = train.descriptor_map_pair_loss(
            descriptor_a,
            descriptor_b,
            points_a,
            points_b,
            hard_negative_weight=0.0,
            abstention_weight=2.0,
            abstention_negative_radius=0.25,
            abstention_max_false_score=0.2,
            abstention_topk=1,
        )

        self.assertGreater(float(penalized.detach()), float(baseline.detach()))

    def test_descriptor_consistency_loss_penalizes_light_changed_descriptor_drift(self):
        descriptor_ref = torch.zeros(1, 2, 1, 2)
        descriptor_same = torch.zeros(1, 2, 1, 2)
        descriptor_drift = torch.zeros(1, 2, 1, 2)
        descriptor_ref[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        descriptor_ref[0, :, 0, 1] = torch.tensor([0.0, 1.0])
        descriptor_same.copy_(descriptor_ref)
        descriptor_drift[0, :, 0, 0] = torch.tensor([0.0, 1.0])
        descriptor_drift[0, :, 0, 1] = torch.tensor([1.0, 0.0])
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        stable = train.descriptor_consistency_loss(descriptor_ref, descriptor_same, points)
        drifting = train.descriptor_consistency_loss(descriptor_ref, descriptor_drift, points)

        self.assertLess(float(stable), 1.0e-6)
        self.assertGreater(float(drifting), 0.5)

    def test_teacher_guided_descriptor_loss_bootstraps_zero_student(self):
        student = torch.zeros(4, 4, requires_grad=True)
        teacher = torch.eye(4)

        loss = train.teacher_guided_descriptor_loss(student, teacher, temperature=0.05)
        loss.backward()

        self.assertGreater(float(student.grad.abs().max()), 0.0)

    def test_descriptor_map_pair_loss_respects_teacher_weight(self):
        student = torch.zeros(1, 4, 2, 2, requires_grad=True)
        teacher = torch.eye(4).T.reshape(1, 4, 2, 2)
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        no_teacher, _ = train.descriptor_map_pair_loss(
            student,
            student,
            points,
            points,
            teacher_descriptors_a=teacher,
            teacher_descriptors_b=teacher,
            teacher_weight=0.0,
        )
        with_teacher, _ = train.descriptor_map_pair_loss(
            student,
            student,
            points,
            points,
            teacher_descriptors_a=teacher,
            teacher_descriptors_b=teacher,
            teacher_weight=1.0,
        )

        self.assertGreater(float(with_teacher.detach()), float(no_teacher.detach()))

    def test_descriptor_map_pair_loss_respects_hard_negative_weight(self):
        descriptor_a = torch.eye(4).T.reshape(1, 4, 2, 2)
        descriptor_b = descriptor_a.clone()
        descriptor_b[:, :, 0, 1] = descriptor_b[:, :, 0, 0]
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        low_weight, _ = train.descriptor_map_pair_loss(
            descriptor_a,
            descriptor_b,
            points,
            points,
            hard_negative_weight=0.0,
        )
        high_weight, _ = train.descriptor_map_pair_loss(
            descriptor_a,
            descriptor_b,
            points,
            points,
            hard_negative_weight=2.0,
        )

        self.assertGreater(float(high_weight.detach()), float(low_weight.detach()))

    def test_compute_descriptor_maps_uses_fast_descriptor_forward(self):
        class FastOnlyModel:
            def __init__(self):
                self.fast_calls = 0

            def descriptor_map_single(self, image):
                self.fast_calls += 1
                return image[:, :1]

            def forward_single(self, image):
                raise AssertionError("descriptor fine-tune should not call full forward_single")

        model = FastOnlyModel()
        pair = SyntheticPair(
            view_a=torch.ones(1, 4, 4),
            view_b=torch.ones(1, 4, 4) * 2.0,
            warp_a_to_b=torch.zeros(4, 4, 2),
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )

        desc_a, desc_b = train.compute_descriptor_maps(model, pair)

        self.assertEqual(model.fast_calls, 2)
        self.assertEqual(tuple(desc_a.shape), (1, 1, 4, 4))
        self.assertEqual(tuple(desc_b.shape), (1, 1, 4, 4))

    def test_student_teacher_descriptor_maps_can_use_final_blended_descriptor(self):
        model = train.pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=2,
            descriptor_dim=4,
            graph_hidden_dim=8,
            graph_attention_layers=1,
        )
        model.eval()
        pair = SyntheticPair(
            view_a=torch.ones(1, 8, 8),
            view_b=torch.ones(1, 8, 8) * 0.5,
            warp_a_to_b=torch.zeros(8, 8, 2),
            valid_mask=torch.ones(8, 8, dtype=torch.bool),
        )

        learned_a, _, _, _ = train.compute_student_teacher_descriptor_maps(
            model,
            pair,
            train_blended_descriptors=False,
        )
        blended_a, _, _, _ = train.compute_student_teacher_descriptor_maps(
            model,
            pair,
            train_blended_descriptors=True,
            texture_blend_weight=1.0,
        )

        self.assertTrue(torch.allclose(learned_a, model.learned_descriptor_map_single(pair.view_a.unsqueeze(0))))
        self.assertTrue(torch.allclose(blended_a, model.descriptor_map_single(pair.view_a.unsqueeze(0), texture_blend_weight=1.0)))

    def test_training_descriptor_path_uses_dual_fpn_when_available(self):
        model = train.pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=8,
            graph_attention_layers=1,
        )
        pair = SyntheticPair(
            view_a=torch.rand(1, 32, 32),
            view_b=torch.rand(1, 32, 32),
            warp_a_to_b=torch.zeros(32, 32, 2),
            valid_mask=torch.ones(32, 32, dtype=torch.bool),
        )
        calls = {"count": 0}
        original_forward = model.dual_fpn.forward

        def counted_forward(features):
            calls["count"] += 1
            return original_forward(features)

        model.dual_fpn.forward = counted_forward

        train.compute_student_teacher_descriptor_maps(model, pair)

        self.assertEqual(calls["count"], 2)

    def test_raw_quality_score_mode_skips_dense_quality_path_for_training_sparse_maps(self):
        model = train.pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=8,
            graph_attention_layers=1,
            quality_score_mode="raw",
        )

        with mock.patch.object(model.dense_head, "forward", side_effect=AssertionError("dense quality path called")):
            sparse_maps = train.learned_training_sparse_maps_single(
                model,
                torch.rand(1, 1, 32, 32),
            )

        self.assertIsNone(sparse_maps.quality)
        self.assertEqual(tuple(sparse_maps.heatmap.shape), (1, 1, 8, 8))

    def test_descriptor_parameters_can_include_sparse_context_for_feature_extractor_tuning(self):
        model = train.pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=2,
            descriptor_dim=4,
            graph_hidden_dim=8,
            graph_attention_layers=1,
        )

        descriptor_only = train.descriptor_parameters(model)
        descriptor_names = {
            name for name, parameter in model.named_parameters() if any(parameter is selected for selected in descriptor_only)
        }
        self.assertTrue(any(name.startswith("sparse_head.descriptor") for name in descriptor_names))
        self.assertFalse(any(name.startswith("sparse_head.context") for name in descriptor_names))

        with_context = train.descriptor_parameters(model, train_sparse_context=True)
        context_names = {
            name for name, parameter in model.named_parameters() if any(parameter is selected for selected in with_context)
        }

        self.assertTrue(any(name.startswith("sparse_head.context") for name in context_names))
        self.assertFalse(any(name.startswith("backbone") for name in context_names))

    def test_aggregate_descriptor_metrics_weights_by_points(self):
        rows = [
            {
                "loss": 2.0,
                "top1_accuracy": 0.25,
                "mean_positive_score": 0.8,
                "mean_negative_score": 0.5,
                "points": 10.0,
            },
            {
                "loss": 4.0,
                "top1_accuracy": 0.75,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.3,
                "points": 30.0,
            },
        ]

        metrics = train.aggregate_descriptor_metrics(rows)

        self.assertAlmostEqual(metrics["loss"], 3.5)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.625)
        self.assertAlmostEqual(metrics["mean_positive_score"], 0.95)
        self.assertAlmostEqual(metrics["mean_negative_score"], 0.35)
        self.assertEqual(metrics["points"], 40.0)

    def test_aggregate_graph_matcher_loss_metrics_weights_by_points(self):
        rows = [
            {
                "graph_matcher_ce_loss": 2.0,
                "graph_matcher_accept_loss": 0.4,
                "graph_matcher_prune_ranking_loss": 0.2,
                "graph_matcher_no_match_loss": 0.1,
                "true_match_in_topk@64": 0.25,
                "true_match_in_topk@256": 0.5,
                "points": 8.0,
            },
            {
                "graph_matcher_ce_loss": 4.0,
                "graph_matcher_accept_loss": 0.8,
                "graph_matcher_prune_ranking_loss": 0.6,
                "graph_matcher_no_match_loss": 0.3,
                "true_match_in_topk@64": 0.75,
                "true_match_in_topk@256": 1.0,
                "points": 24.0,
            },
        ]

        metrics = train.aggregate_graph_matcher_loss_metrics(rows)

        self.assertAlmostEqual(metrics["graph_matcher_ce_loss"], 3.5)
        self.assertAlmostEqual(metrics["graph_matcher_accept_loss"], 0.7)
        self.assertAlmostEqual(metrics["graph_matcher_prune_ranking_loss"], 0.5)
        self.assertAlmostEqual(metrics["graph_matcher_no_match_loss"], 0.25)
        self.assertAlmostEqual(metrics["true_match_in_topk@64"], 0.625)
        self.assertAlmostEqual(metrics["true_match_in_topk@256"], 0.875)

    def test_split_train_eval_pairs_uses_tail_as_held_out(self):
        paths = [Path(f"pair_{index:06d}.pt") for index in range(6)]

        train_paths, eval_paths = train.split_train_eval_pairs(paths, eval_pairs=2)

        self.assertEqual(train_paths, paths[:4])
        self.assertEqual(eval_paths, paths[4:])

    def test_split_train_eval_pairs_keeps_training_nonempty(self):
        paths = [Path("pair_000000.pt")]

        train_paths, eval_paths = train.split_train_eval_pairs(paths, eval_pairs=4)

        self.assertEqual(train_paths, paths)
        self.assertEqual(eval_paths, [])

    def test_split_train_eval_pairs_balances_multiple_cache_roots(self):
        paths = [
            Path("img/CompoundViewpoint/source_000000/pair_000000.pt"),
            Path("img/CompoundViewpoint/source_000001/pair_000001.pt"),
            Path("img/CompoundViewpoint/source_000002/pair_000002.pt"),
            Path("img/Rotate/source_000000/pair_000000.pt"),
            Path("img/Rotate/source_000001/pair_000001.pt"),
            Path("img/Rotate/source_000002/pair_000002.pt"),
            Path("img/Viewpoint/source_000000/pair_000000.pt"),
            Path("img/Viewpoint/source_000001/pair_000001.pt"),
            Path("img/Viewpoint/source_000002/pair_000002.pt"),
        ]

        train_paths, eval_paths = train.split_train_eval_pairs(paths, eval_pairs=3)

        self.assertEqual(
            eval_paths,
            [
                Path("img/CompoundViewpoint/source_000002/pair_000002.pt"),
                Path("img/Rotate/source_000002/pair_000002.pt"),
                Path("img/Viewpoint/source_000002/pair_000002.pt"),
            ],
        )
        self.assertEqual(len(train_paths), 6)

    def test_resolve_training_and_eval_paths_uses_explicit_validation_cache(self):
        paths = [
            Path("train/source_001_a/pair_000001.pt"),
            Path("train/source_001_a/pair_000002.pt"),
            Path("val/source_002_b/pair_000003.pt"),
            Path("val/source_002_b/pair_000004.pt"),
        ]

        with mock.patch.object(train, "discover_pair_archives") as discover:
            discover.side_effect = [paths[:2], paths[2:]]
            train_paths, eval_paths = train.resolve_training_and_eval_pair_paths(
                [Path("train")],
                [Path("val")],
                limit_pairs=0,
                eval_pairs=1,
            )

        self.assertEqual(train_paths, paths[:2])
        self.assertEqual(eval_paths, [paths[2]])
        self.assertEqual(discover.call_args_list[0].kwargs["limit_pairs"], 0)
        self.assertEqual(discover.call_args_list[1].kwargs["limit_pairs"], 1)

    def test_resolve_training_and_eval_paths_can_exclude_self_pairs(self):
        paths = [
            Path("train/source_001_a/pair_000002.pt"),
            Path("val/source_002_b/pair_000003.pt"),
        ]

        with mock.patch.object(train, "discover_pair_archives") as discover:
            discover.side_effect = [[paths[0]], [paths[1]]]
            train_paths, eval_paths = train.resolve_training_and_eval_pair_paths(
                [Path("train")],
                [Path("val")],
                limit_pairs=0,
                eval_pairs=0,
                exclude_self_pairs=True,
            )

        self.assertEqual(train_paths, [paths[0]])
        self.assertEqual(eval_paths, [paths[1]])
        self.assertTrue(discover.call_args_list[0].kwargs["exclude_self_pairs"])
        self.assertTrue(discover.call_args_list[1].kwargs["exclude_self_pairs"])

    def test_parse_args_accepts_validation_cache_dirs(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--validation-cache-dir",
            "val_numeric",
            "--validation-cache-dir",
            "val_timestamp",
            "--exclude-self-pairs",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertEqual(args.validation_cache_dir, [Path("val_numeric"), Path("val_timestamp")])
        self.assertTrue(args.exclude_self_pairs)

    def test_repeat_hard_pairs_from_summary_after_base_training_paths(self):
        paths = [
            Path("cache/source_a/pair_000001.pt"),
            Path("cache/source_b/pair_000002.pt"),
            Path("cache/source_c/pair_000003.pt"),
        ]
        summary = Path("/tmp/pfm_hard_repeat_summary.csv")
        summary.write_text(
            "pair_pt,sparse_matches,match_precision\n"
            "cache/source_a/pair_000001.pt,3,0.00\n"
            "cache/source_b/pair_000002.pt,7,0.25\n"
            "cache/source_c/pair_000003.pt,40,0.95\n",
            encoding="utf-8",
        )

        expanded, selected = train.repeat_hard_training_pairs(
            paths,
            [summary],
            limit=8,
            min_matches=4,
            max_precision=0.9,
            repeat=3,
        )

        self.assertEqual(selected, [Path("cache/source_b/pair_000002.pt")])
        self.assertEqual(expanded, paths + [Path("cache/source_b/pair_000002.pt")] * 3)

    def test_hard_pair_probability_warms_up_and_clamps(self):
        self.assertEqual(train.hard_pair_probability(0, max_probability=0.4, warmup_steps=100), 0.0)
        self.assertAlmostEqual(train.hard_pair_probability(50, max_probability=0.4, warmup_steps=100), 0.2)
        self.assertAlmostEqual(train.hard_pair_probability(100, max_probability=0.4, warmup_steps=100), 0.4)
        self.assertAlmostEqual(train.hard_pair_probability(200, max_probability=0.4, warmup_steps=100), 0.4)
        self.assertEqual(train.hard_pair_probability(10, max_probability=3.0, warmup_steps=0), 1.0)

    def test_scheduled_value_interpolates_to_final_weight(self):
        self.assertAlmostEqual(train.scheduled_value(0, start=1.0, final=0.5, schedule_steps=100), 1.0)
        self.assertAlmostEqual(train.scheduled_value(50, start=1.0, final=0.5, schedule_steps=100), 0.75)
        self.assertAlmostEqual(train.scheduled_value(100, start=1.0, final=0.5, schedule_steps=100), 0.5)
        self.assertAlmostEqual(train.scheduled_value(200, start=1.0, final=0.5, schedule_steps=100), 0.5)
        self.assertAlmostEqual(train.scheduled_value(10, start=1.0, final=0.25, schedule_steps=0), 0.25)

    def test_sample_curriculum_training_pairs_mixes_base_and_hard_pairs(self):
        base = [Path(f"base_{index}.pt") for index in range(8)]
        hard = [Path(f"hard_{index}.pt") for index in range(4)]

        selected = train.sample_curriculum_training_pairs(
            base,
            hard,
            batch_pairs=4,
            hard_probability=0.5,
            rng=random.Random(7),
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(sum(path in hard for path in selected), 2)
        self.assertEqual(sum(path in base for path in selected), 2)

    def test_sample_cache_balanced_training_pairs_prefers_distinct_cache_roots(self):
        paths = [
            Path("train/numeric/rotate/source_000001/pair_000010.pt"),
            Path("train/numeric/rotate/source_000001/pair_000011.pt"),
            Path("train/numeric/viewpoint/source_000001/pair_000010.pt"),
            Path("train/timestamp/compound/source_000001/pair_000010.pt"),
        ]

        selected = train.sample_cache_balanced_training_pairs(paths, batch_pairs=3, rng=random.Random(5))

        self.assertEqual(len(selected), 3)
        self.assertEqual(len({path.parent.parent for path in selected}), 3)

    def test_sample_curriculum_training_pairs_can_balance_base_cache_roots(self):
        base = [
            Path("train/numeric/rotate/source_000001/pair_000010.pt"),
            Path("train/numeric/rotate/source_000001/pair_000011.pt"),
            Path("train/numeric/viewpoint/source_000001/pair_000010.pt"),
            Path("train/timestamp/compound/source_000001/pair_000010.pt"),
        ]

        selected = train.sample_curriculum_training_pairs(
            base,
            [],
            batch_pairs=3,
            hard_probability=0.0,
            rng=random.Random(5),
            balanced_cache_sampling=True,
        )

        self.assertEqual(len({path.parent.parent for path in selected}), 3)

    def test_sample_training_pairs_with_pseudo_labels_injects_labeled_pairs(self):
        base = [Path(f"base_{index}.pt") for index in range(8)]
        pseudo = [Path(f"pseudo_{index}.pt") for index in range(4)]

        selected = train.sample_training_pairs_with_pseudo_labels(
            base,
            [],
            pseudo,
            batch_pairs=4,
            hard_probability=0.0,
            pseudo_label_probability=0.5,
            rng=random.Random(7),
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(sum(path in pseudo for path in selected), 2)

    def test_sample_training_pairs_with_pseudo_and_false_uses_separate_quotas(self):
        base = [Path(f"base_{index}.pt") for index in range(8)]
        pseudo = [Path(f"pseudo_{index}.pt") for index in range(4)]
        false = [Path(f"false_{index}.pt") for index in range(4)]

        selected = train.sample_training_pairs_with_pseudo_labels(
            base,
            [],
            pseudo,
            batch_pairs=2,
            hard_probability=0.0,
            pseudo_label_probability=1.0,
            false_match_pair_paths=false,
            false_match_probability=1.0,
            rng=random.Random(7),
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(sum(path in pseudo for path in selected), 1)
        self.assertEqual(sum(path in false for path in selected), 1)

    def test_sample_training_pairs_with_pseudo_and_false_splits_larger_batch(self):
        base = [Path(f"base_{index}.pt") for index in range(8)]
        pseudo = [Path(f"pseudo_{index}.pt") for index in range(8)]
        false = [Path(f"false_{index}.pt") for index in range(8)]

        selected = train.sample_training_pairs_with_pseudo_labels(
            base,
            [],
            pseudo,
            batch_pairs=4,
            hard_probability=0.0,
            pseudo_label_probability=1.0,
            false_match_pair_paths=false,
            false_match_probability=1.0,
            rng=random.Random(11),
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(sum(path in pseudo for path in selected), 2)
        self.assertEqual(sum(path in false for path in selected), 2)

    def test_sample_training_pairs_with_low_pseudo_and_false_probability_keeps_base_pairs(self):
        base = [Path(f"base_{index}.pt") for index in range(8)]
        pseudo = [Path(f"pseudo_{index}.pt") for index in range(8)]
        false = [Path(f"false_{index}.pt") for index in range(8)]

        selected = train.sample_training_pairs_with_pseudo_labels(
            base,
            [],
            pseudo,
            batch_pairs=4,
            hard_probability=0.0,
            pseudo_label_probability=0.25,
            false_match_pair_paths=false,
            false_match_probability=0.25,
            rng=random.Random(7),
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(sum(path in pseudo for path in selected), 1)
        self.assertEqual(sum(path in false for path in selected), 1)
        self.assertEqual(sum(path in base for path in selected), 2)

    def test_sample_pose_balanced_training_pairs_covers_difficulty_buckets(self):
        paths = [Path(f"pair_{index}.pt") for index in range(6)]
        metadata = {
            paths[0].as_posix(): mock.Mock(difficulty="easy"),
            paths[1].as_posix(): mock.Mock(difficulty="easy"),
            paths[2].as_posix(): mock.Mock(difficulty="medium"),
            paths[3].as_posix(): mock.Mock(difficulty="hard"),
            paths[4].as_posix(): mock.Mock(difficulty="hard"),
        }

        selected = train.sample_pose_balanced_training_pairs(paths, metadata, batch_pairs=4, rng=random.Random(2))
        selected_difficulties = [metadata[path.as_posix()].difficulty if path.as_posix() in metadata else "unknown" for path in selected]

        self.assertEqual(len(selected), 4)
        self.assertIn("easy", selected_difficulties)
        self.assertIn("medium", selected_difficulties)
        self.assertIn("hard", selected_difficulties)
        self.assertIn("unknown", selected_difficulties)

    def test_sample_pose_balanced_training_pairs_with_single_pair_eventually_samples_hard(self):
        paths = [Path(f"pair_{index}.pt") for index in range(4)]
        metadata = {
            paths[0].as_posix(): mock.Mock(difficulty="medium"),
            paths[1].as_posix(): mock.Mock(difficulty="medium"),
            paths[2].as_posix(): mock.Mock(difficulty="hard"),
            paths[3].as_posix(): mock.Mock(difficulty="hard"),
        }
        rng = random.Random(17)

        selected_difficulties = []
        for _ in range(40):
            selected = train.sample_pose_balanced_training_pairs(paths, metadata, batch_pairs=1, rng=rng)
            selected_difficulties.append(metadata[selected[0].as_posix()].difficulty)

        self.assertIn("medium", selected_difficulties)
        self.assertIn("hard", selected_difficulties)

    def test_pose_difficulty_loss_weight_scales_synthetic_loss(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_hard.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        metadata = {pair_path.as_posix(): mock.Mock(difficulty="hard", difficulty_score=1.0)}

        def fake_descriptor_loss(*_args, **_kwargs):
            return parameter * 2.0, {
                "top1_accuracy": 1.0,
                "top5_accuracy": 1.0,
                "top10_accuracy": 1.0,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.0,
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.0,
                pose_metadata=metadata,
                pose_difficulty_loss_weight=1.0,
            )

        self.assertAlmostEqual(float(parameter.detach()), 0.6, places=5)
        self.assertEqual(metrics["pose_hard_pairs"], 1.0)
        self.assertAlmostEqual(metrics["pose_mean_loss_weight"], 2.0)

    def test_heatmap_point_loss_rewards_pseudo_label_locations(self):
        points = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        low = torch.full((1, 1, 4, 4), 0.1)
        high = low.clone()
        high[0, 0, 1, 1] = 0.9
        high[0, 0, 2, 2] = 0.9

        low_loss = train.heatmap_point_loss(low, points)
        high_loss = train.heatmap_point_loss(high, points)

        self.assertLess(float(high_loss), float(low_loss))

    def test_descriptor_parameters_can_include_keypoint_head(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=8, graph_attention_layers=1)

        selected = train.descriptor_parameters(model, train_keypoint_head=True)

        selected_ids = {id(parameter) for parameter in selected}
        self.assertIn(id(model.sparse_head.heatmap.weight), selected_ids)
        self.assertTrue(model.sparse_head.heatmap.weight.requires_grad)
        self.assertFalse(model.backbone.stage1[0].weight.requires_grad)

    def test_descriptor_parameters_can_include_texture_adapter(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=8, graph_attention_layers=1)

        selected = train.descriptor_parameters(model, train_texture_adapter=True)

        selected_ids = {id(parameter) for parameter in selected}
        self.assertIn(id(model.texture_adapter.residual.weight), selected_ids)
        self.assertTrue(model.texture_adapter.residual.weight.requires_grad)
        self.assertFalse(model.backbone.stage1[0].weight.requires_grad)

    def test_descriptor_parameters_can_train_texture_adapter_without_descriptor_head(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=8, graph_attention_layers=1)

        selected = train.descriptor_parameters(
            model,
            train_descriptor_head=False,
            train_texture_adapter=True,
        )

        selected_ids = {id(parameter) for parameter in selected}
        self.assertIn(id(model.texture_adapter.residual.weight), selected_ids)
        self.assertTrue(model.texture_adapter.residual.weight.requires_grad)
        self.assertNotIn(id(model.sparse_head.descriptor_skip.weight), selected_ids)
        self.assertFalse(model.sparse_head.descriptor_skip.weight.requires_grad)

    def test_descriptor_parameters_can_include_descriptor_fusion_without_descriptor_head(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=8, graph_attention_layers=1)

        selected = train.descriptor_parameters(
            model,
            train_descriptor_head=False,
            train_descriptor_fusion=True,
        )

        selected_ids = {id(parameter) for parameter in selected}
        self.assertIn(id(model.descriptor_fusion.output.weight), selected_ids)
        self.assertTrue(model.descriptor_fusion.output.weight.requires_grad)
        self.assertNotIn(id(model.sparse_head.descriptor_skip.weight), selected_ids)
        self.assertFalse(model.sparse_head.descriptor_skip.weight.requires_grad)

    def test_descriptor_parameters_can_include_full_v21_modules(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=8, graph_attention_layers=1)

        selected = train.descriptor_parameters(
            model,
            train_descriptor_head=False,
            train_backbone=True,
            train_dual_fpn=True,
            train_geometry_head=True,
            train_quality_head=True,
            train_graph_matcher=True,
        )

        selected_ids = {id(parameter) for parameter in selected}
        self.assertIn(id(model.backbone.stage1[0].weight), selected_ids)
        self.assertIn(id(model.dual_fpn.descriptor_from_stage3.weight), selected_ids)
        self.assertIn(id(model.sparse_head.affine.weight), selected_ids)
        self.assertIn(id(model.quality_head.predictor[-1].weight), selected_ids)
        self.assertIn(id(model.graph_matcher.descriptor_projection.weight), selected_ids)
        self.assertNotIn(id(model.texture_adapter.residual.weight), selected_ids)

    def test_descriptor_parameters_can_train_graph_calibration_without_attention(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=8, graph_attention_layers=1)

        selected = train.descriptor_parameters(
            model,
            train_descriptor_head=False,
            train_graph_matcher=True,
            train_graph_calibration_only=True,
        )

        selected_ids = {id(parameter) for parameter in selected}
        self.assertIn(id(model.graph_matcher.accept_head[-1].weight), selected_ids)
        self.assertIn(id(model.graph_matcher.geometry_bias[-1].weight), selected_ids)
        self.assertIn(id(model.graph_matcher.dustbin_bias), selected_ids)
        self.assertTrue(model.graph_matcher.accept_head[-1].weight.requires_grad)
        self.assertTrue(model.graph_matcher.geometry_bias[-1].weight.requires_grad)
        self.assertTrue(model.graph_matcher.dustbin_bias.requires_grad)
        self.assertNotIn(id(model.graph_matcher.attention_layers[0].self_query.weight), selected_ids)
        self.assertNotIn(id(model.graph_matcher.descriptor_projection.weight), selected_ids)
        self.assertFalse(model.graph_matcher.attention_layers[0].self_query.weight.requires_grad)
        self.assertFalse(model.graph_matcher.descriptor_projection.weight.requires_grad)

    def test_graph_matcher_correspondence_loss_backpropagates_to_matcher(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=1)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)

        loss = train.graph_matcher_correspondence_loss(model, descriptors_a, descriptors_b, points, points)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.graph_matcher.descriptor_projection.weight.grad)

    def test_descriptor_parameters_can_include_reliability_head(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )

        selected = train.descriptor_parameters(
            model,
            train_descriptor_head=False,
            train_reliability_head=True,
        )

        selected_ids = {id(parameter) for parameter in selected}
        self.assertIn(id(model.sparse_head.matchability.weight), selected_ids)
        self.assertIn(id(model.sparse_head.descriptor_uncertainty.weight), selected_ids)
        self.assertIn(id(model.sparse_head.no_match_prior.weight), selected_ids)
        self.assertNotIn(id(model.sparse_head.descriptor_skip.weight), selected_ids)

    def test_graph_matcher_correspondence_loss_default_decouples_reliability_maps(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
        matchability_a = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        matchability_b = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        uncertainty_a = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        uncertainty_b = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        no_match_a = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        no_match_b = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)

        loss = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            no_match_points=2,
            no_match_weight=0.5,
            matchability_a=matchability_a,
            matchability_b=matchability_b,
            descriptor_uncertainty_a=uncertainty_a,
            descriptor_uncertainty_b=uncertainty_b,
            no_match_prior_a=no_match_a,
            no_match_prior_b=no_match_b,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        for tensor in (matchability_a, matchability_b, uncertainty_a, uncertainty_b, no_match_a, no_match_b):
            if tensor.grad is not None:
                self.assertAlmostEqual(float(tensor.grad.abs().sum()), 0.0, places=7)

    def test_graph_matcher_correspondence_loss_full_metadata_backpropagates_to_reliability_maps(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
        matchability_a = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        matchability_b = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        uncertainty_a = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        uncertainty_b = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        no_match_a = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        no_match_b = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)

        loss = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            no_match_points=2,
            no_match_weight=0.5,
            matchability_a=matchability_a,
            matchability_b=matchability_b,
            descriptor_uncertainty_a=uncertainty_a,
            descriptor_uncertainty_b=uncertainty_b,
            no_match_prior_a=no_match_a,
            no_match_prior_b=no_match_b,
            metadata_mode="full",
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        for tensor in (matchability_a, matchability_b, uncertainty_a, uncertainty_b, no_match_a, no_match_b):
            self.assertIsNotNone(tensor.grad)
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0)

    def test_graph_matcher_correspondence_loss_disables_candidate_mask_for_supervision(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        model.eval()
        model.graph_matcher.candidate_topk = 1
        descriptors_a = torch.zeros(1, 8, 1, 3)
        descriptors_b = torch.zeros(1, 8, 1, 3)
        descriptors_a[0, 0, 0, 0] = 1.0
        descriptors_a[0, 0, 0, 1] = -1.0
        descriptors_a[0, 0, 0, 2] = -1.0
        descriptors_b[0, 0, 0, 0] = -1.0
        descriptors_b[0, 0, 0, 1] = 1.0
        descriptors_b[0, 0, 0, 2] = 1.0
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

        loss = train.graph_matcher_correspondence_loss(model, descriptors_a, descriptors_b, points, points)

        self.assertLess(float(loss.detach()), 100.0)

    def test_graph_matcher_correspondence_loss_candidate_topk_preserves_positive_diagonal(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        model.eval()
        descriptors_a = torch.zeros(1, 8, 1, 3)
        descriptors_b = torch.zeros(1, 8, 1, 3)
        descriptors_a[0, 0, 0, 0] = 1.0
        descriptors_a[0, 0, 0, 1] = -1.0
        descriptors_a[0, 0, 0, 2] = -1.0
        descriptors_b[0, 0, 0, 0] = -1.0
        descriptors_b[0, 0, 0, 1] = 1.0
        descriptors_b[0, 0, 0, 2] = 1.0
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

        loss, components = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            train_candidate_topk=1,
            return_components=True,
        )

        self.assertLess(float(loss.detach()), 100.0)
        self.assertEqual(int(components["graph_matcher_train_candidate_topk"].item()), 1)
        self.assertIn("true_match_in_topk@64", components)
        self.assertIn("true_match_in_topk@256", components)
        self.assertAlmostEqual(float(components["true_match_in_topk@64"].item()), 1.0)
        self.assertAlmostEqual(float(components["true_match_in_topk@256"].item()), 1.0)

    def test_graph_matcher_correspondence_loss_can_train_dustbin_negatives(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=1)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 8, 8))
        descriptors_b = pfm_model.normalize_channels_stable(torch.randn(1, 8, 8, 8))
        points = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=torch.float32)

        loss = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            metadata_mode="descriptor_only",
            no_match_points=4,
            no_match_weight=0.5,
            no_match_min_distance=1.0,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.graph_matcher.dustbin_bias.grad)

    def test_graph_matcher_correspondence_loss_can_add_context_distractors_without_dustbin_loss(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=1)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 8, 8))
        descriptors_b = pfm_model.normalize_channels_stable(torch.randn(1, 8, 8, 8))
        points = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=torch.float32)
        captured_sizes = []
        original_forward = model.graph_matcher.forward

        def wrapped_forward(desc_a, meta_a, desc_b, meta_b, **kwargs):
            captured_sizes.append((desc_a.size(0), desc_b.size(0)))
            return original_forward(desc_a, meta_a, desc_b, meta_b, **kwargs)

        model.graph_matcher.forward = wrapped_forward
        with mock.patch.object(
            train,
            "sample_unmatched_feature_points",
            side_effect=[
                torch.tensor([[0.0, 6.0], [7.0, 1.0]], dtype=torch.float32),
                torch.tensor([[6.0, 0.0], [1.0, 7.0]], dtype=torch.float32),
            ],
        ):
            loss, components = train.graph_matcher_correspondence_loss(
                model,
                descriptors_a,
                descriptors_b,
                points,
                points,
                metadata_mode="descriptor_only",
                no_match_points=2,
                no_match_weight=0.0,
                assignment_weight=0.2,
                return_components=True,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(captured_sizes[-1], (5, 5))
        self.assertEqual(float(components["graph_matcher_no_match_loss"].detach()), 0.0)

    def test_graph_matcher_correspondence_loss_can_use_extra_online_no_match_points(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=1)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 8, 8))
        descriptors_b = pfm_model.normalize_channels_stable(torch.randn(1, 8, 8, 8))
        points = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=torch.float32)
        false_a = torch.tensor([[0.0, 6.0], [7.0, 1.0]], dtype=torch.float32)
        false_b = torch.tensor([[6.0, 0.0], [1.0, 7.0]], dtype=torch.float32)

        loss, components = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            metadata_mode="descriptor_only",
            no_match_weight=0.5,
            assignment_weight=0.2,
            extra_no_match_points_a_xy=false_a,
            extra_no_match_points_b_xy=false_b,
            return_components=True,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(components["graph_matcher_no_match_loss"].detach()), 0.0)
        self.assertEqual(int(components["graph_matcher_extra_no_match_points"].item()), 4)
        self.assertIsNotNone(model.graph_matcher.dustbin_bias.grad)

    def test_graph_matcher_correspondence_loss_can_use_semi_dense_no_match_candidates(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=1)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 8, 8))
        descriptors_b = pfm_model.normalize_channels_stable(torch.randn(1, 8, 8, 8))
        points = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=torch.float32)

        loss = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            metadata_mode="descriptor_only",
            no_match_weight=0.5,
            semi_dense_no_match_points=4,
            semi_dense_min_score=0.0,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.graph_matcher.dustbin_bias.grad)

    def test_graph_matcher_correspondence_loss_can_train_accept_head(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=1)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)

        loss = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            accept_weight=0.5,
            accept_negative_topk=2,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.graph_matcher.accept_head[-1].weight.grad)

    def test_graph_matcher_correspondence_loss_can_train_with_attention_layer_budget(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=3)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)

        loss = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            max_attention_layers=1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.graph_matcher.last_executed_attention_layers, 1)

    def test_graph_matcher_correspondence_loss_reports_ransac_consistency_components(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=1)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points_a = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        points_b = points_a + torch.tensor([1.0, 0.0], dtype=torch.float32)

        loss, components = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points_a,
            points_b,
            ransac_consistency_weight=0.2,
            ransac_consistency_topk=2,
            ransac_consistency_residual_threshold_px=0.5,
            return_components=True,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("graph_matcher_ransac_consistency_loss", components)
        self.assertIn("graph_matcher_ransac_consistency_edges", components)
        self.assertIn("graph_matcher_ransac_consistency_residual_mean_px", components)

    def test_graph_matcher_positive_dustbin_margin_loss_penalizes_true_match_rejection(self):
        logits = torch.zeros(3, 3)
        logits[0, 0] = 0.5
        logits[1, 1] = 0.4
        logits[0, 2] = 1.0
        logits[2, 1] = 1.2
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )

        loss = train.graph_matcher_positive_dustbin_margin_loss(output, positive_count=2, margin=0.25)

        self.assertGreater(float(loss), 0.0)

    def test_graph_matcher_final_false_match_loss_mines_high_confidence_wrong_edge(self):
        logits = torch.full((4, 4), -5.0)
        logits[0, 0] = 3.0
        logits[1, 1] = 3.0
        logits[2, 2] = 3.0
        logits[0, 1] = 8.0
        logits[3, :] = -4.0
        logits[:, 3] = -4.0
        accept_logits = torch.zeros(3, 3)
        accept_logits[0, 1] = 6.0
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
            accept_logits=accept_logits,
        )
        points = torch.tensor([[0.0, 0.0], [8.0, 0.0], [16.0, 0.0]], dtype=torch.float32)

        loss, metrics = train.graph_matcher_final_false_match_loss(
            output,
            positive_count=3,
            points_b_xy=points,
            topk=1,
            min_score=0.01,
            margin=0.25,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(float(metrics["edges"]), 1.0)
        self.assertGreater(float(metrics["score_mean"]), 0.0)
        self.assertGreater(float(metrics["accept_mean"]), 0.0)

    def test_graph_matcher_final_false_match_loss_returns_zero_without_selected_edges(self):
        logits = torch.full((4, 4), -5.0)
        logits[0, 0] = 6.0
        logits[1, 1] = 6.0
        logits[2, 2] = 6.0
        logits[3, 3] = 0.0
        logits[:3, 3] = 0.0
        logits[3, :3] = 0.0
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
            accept_logits=torch.zeros(3, 3),
        )
        points = torch.tensor([[0.0, 0.0], [8.0, 0.0], [16.0, 0.0]], dtype=torch.float32)

        loss, metrics = train.graph_matcher_final_false_match_loss(
            output,
            positive_count=3,
            points_b_xy=points,
            topk=2,
            min_score=0.9,
            margin=0.25,
        )

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(metrics["edges"]), 0.0)

    def test_graph_matcher_mined_false_match_loss_penalizes_false_edge_without_dustbin_gradient(self):
        logits = torch.full((5, 5), -5.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0] = 5.0
            logits[1, 1] = 5.0
            logits[2, 2] = 7.0
            logits[3, 3] = -4.0
            logits[2, 4] = -4.0
            logits[4, 2] = -4.0
            logits[3, 4] = 5.0
            logits[4, 3] = 5.0
        accept_logits = torch.zeros(4, 4, requires_grad=True)
        with torch.no_grad():
            accept_logits[2, 2] = 6.0
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
            accept_logits=accept_logits,
        )

        loss, metrics = train.graph_matcher_mined_false_match_loss(
            output,
            positive_count=2,
            false_a_start=2,
            false_b_start=2,
            false_pair_count=2,
            topk=1,
            min_score=0.01,
            margin=0.25,
        )
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(float(metrics["edges"]), 1.0)
        self.assertGreater(float(metrics["score_mean"]), 0.0)
        self.assertGreater(float(metrics["accept_mean"]), 0.0)
        self.assertGreater(float(logits.grad[2, 2]), 0.0)
        self.assertEqual(float(logits.grad[2, 4]), 0.0)
        self.assertEqual(float(logits.grad[4, 2]), 0.0)

    def test_graph_matcher_mined_false_match_loss_uses_final_score_margin_without_dustbin_gradient(self):
        logits = torch.full((4, 4), -5.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0] = 5.0
            logits[1, 1] = 5.0
            logits[2, 2] = 4.5
            logits[0, 3] = 4.0
            logits[1, 3] = 4.0
            logits[3, 0] = 4.0
            logits[3, 1] = 4.0
            logits[2, 3] = -4.0
            logits[3, 2] = -4.0
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )

        loss, metrics = train.graph_matcher_mined_false_match_loss(
            output,
            positive_count=2,
            false_a_start=2,
            false_b_start=2,
            false_pair_count=1,
            topk=1,
            min_score=0.01,
            margin=0.25,
        )
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(float(metrics["edges"]), 1.0)
        self.assertGreater(float(metrics["score_mean"]), 0.0)
        self.assertGreater(float(logits.grad[2, 2]), 0.0)
        self.assertEqual(float(logits.grad[2, 3]), 0.0)
        self.assertEqual(float(logits.grad[3, 2]), 0.0)

    def test_graph_matcher_ransac_consistency_loss_penalizes_geometric_outlier_without_dustbin_gradient(self):
        logits = torch.full((4, 4), -5.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0] = 5.0
            logits[1, 1] = 5.0
            logits[2, 2] = 5.0
            logits[0, 1] = 14.0
            logits[0, 3] = 4.0
            logits[3, 1] = 4.0
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )
        points_a = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
        points_b = torch.tensor([[5.0, 0.0], [15.0, 0.0], [5.0, 10.0]], dtype=torch.float32)

        loss, metrics = train.graph_matcher_ransac_consistency_loss(
            output,
            positive_count=3,
            points_a_xy=points_a,
            points_b_xy=points_b,
            topk=1,
            residual_threshold_px=3.0,
            min_score=0.01,
            margin=0.25,
        )
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(float(metrics["edges"]), 1.0)
        self.assertGreater(float(metrics["residual_mean_px"]), 3.0)
        self.assertGreater(float(logits.grad[0, 1]), 0.0)
        self.assertEqual(float(logits.grad[0, 3]), 0.0)
        self.assertEqual(float(logits.grad[3, 1]), 0.0)

    def test_graph_matcher_mined_false_match_loss_can_cap_extreme_spikes(self):
        logits = torch.full((4, 4), -5.0)
        logits[0, 0] = 2.0
        logits[1, 1] = 2.0
        logits[2, 2] = 30.0
        logits[2, 3] = -4.0
        logits[3, 2] = -4.0
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )

        uncapped_loss, _ = train.graph_matcher_mined_false_match_loss(
            output,
            positive_count=2,
            false_a_start=2,
            false_b_start=2,
            false_pair_count=1,
            topk=1,
            min_score=0.01,
            margin=0.25,
        )
        capped_loss, metrics = train.graph_matcher_mined_false_match_loss(
            output,
            positive_count=2,
            false_a_start=2,
            false_b_start=2,
            false_pair_count=1,
            topk=1,
            min_score=0.01,
            margin=0.25,
            loss_cap=4.0,
        )

        self.assertGreater(float(uncapped_loss), 100.0)
        self.assertLessEqual(float(capped_loss), 4.0)
        self.assertEqual(float(metrics["edges"]), 1.0)

    def test_graph_matcher_mined_false_match_loss_skips_safe_false_edges_below_positive_reference(self):
        logits = torch.full((5, 5), -5.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0] = 5.0
            logits[1, 1] = 5.0
            logits[2, 2] = 4.8
            logits[3, 3] = 0.1
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )

        loss, metrics = train.graph_matcher_mined_false_match_loss(
            output,
            positive_count=2,
            false_a_start=2,
            false_b_start=2,
            false_pair_count=2,
            topk=2,
            min_score=0.01,
            margin=0.25,
            reference_margin=0.25,
        )

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(float(metrics["edges"]), 1.0)
        self.assertEqual(float(metrics["reference_filtered_edges"]), 1.0)

    def test_graph_matcher_true_match_margin_loss_penalizes_missed_true_pair(self):
        logits = torch.full((4, 4), -4.0)
        logits[0, 0] = 3.0
        logits[1, 1] = 1.0
        logits[2, 2] = 3.0
        logits[1, 0] = 2.5
        logits[0, 1] = 2.2
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )

        loss, metrics = train.graph_matcher_true_match_margin_loss(
            output,
            positive_count=3,
            margin=0.5,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(int(metrics["violations"].item()), 1)
        safe_logits = logits.clone()
        safe_logits[1, 1] = 4.0
        safe_loss, safe_metrics = train.graph_matcher_true_match_margin_loss(
            pfm_model.GraphMatcherOutput(
                logits=safe_logits,
                matches=torch.empty((0, 2), dtype=torch.long),
                scores=torch.empty((0,), dtype=torch.float32),
            ),
            positive_count=3,
            margin=0.5,
        )
        self.assertAlmostEqual(float(safe_loss), 0.0)
        self.assertEqual(int(safe_metrics["violations"].item()), 0)

    def test_graph_matcher_depth_distillation_loss_penalizes_full_depth_drift(self):
        teacher_logits = torch.tensor(
            [
                [4.0, -1.0, -2.0],
                [-1.0, 4.0, -2.0],
                [-2.0, -2.0, 0.0],
            ],
            dtype=torch.float32,
        )
        student_logits = torch.tensor(
            [
                [-1.0, 4.0, -2.0],
                [4.0, -1.0, -2.0],
                [-2.0, -2.0, 0.0],
            ],
            dtype=torch.float32,
        )
        teacher = pfm_model.GraphMatcherOutput(teacher_logits, torch.empty((0, 2), dtype=torch.long), torch.empty((0,)))
        student = pfm_model.GraphMatcherOutput(student_logits, torch.empty((0, 2), dtype=torch.long), torch.empty((0,)))

        loss = train.graph_matcher_depth_distillation_loss(student, teacher, positive_count=2, temperature=1.0)
        identical = train.graph_matcher_depth_distillation_loss(teacher, teacher, positive_count=2, temperature=1.0)

        self.assertGreater(float(loss), 1.0)
        self.assertAlmostEqual(float(identical), 0.0, places=6)

    def test_graph_matcher_teacher_guard_loss_penalizes_positive_margin_regression(self):
        teacher_logits = torch.tensor(
            [
                [5.0, -1.0, -3.0],
                [-1.0, 5.0, -3.0],
                [-3.0, -3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        student_logits = torch.tensor(
            [
                [2.0, 4.0, -3.0],
                [-1.0, 5.0, -3.0],
                [-3.0, -3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        teacher = pfm_model.GraphMatcherOutput(teacher_logits, torch.empty((0, 2), dtype=torch.long), torch.empty((0,)))
        student = pfm_model.GraphMatcherOutput(student_logits, torch.empty((0, 2), dtype=torch.long), torch.empty((0,)))

        loss, metrics = train.graph_matcher_teacher_guard_loss(
            student,
            teacher,
            positive_count=2,
            positive_margin_tolerance=0.0,
            false_margin_tolerance=0.0,
        )
        identical, identical_metrics = train.graph_matcher_teacher_guard_loss(
            teacher,
            teacher,
            positive_count=2,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(metrics["positive_margin_loss"]), 0.0)
        self.assertGreater(float(metrics["false_edge_loss"]), 0.0)
        self.assertGreater(float(metrics["positive_violations"]), 0.0)
        self.assertGreater(float(metrics["false_edges"]), 0.0)
        self.assertAlmostEqual(float(identical), 0.0, places=6)
        self.assertEqual(float(identical_metrics["positive_violations"]), 0.0)

    def test_graph_matcher_teacher_score_floor_loss_penalizes_positive_score_regression(self):
        teacher_logits = torch.tensor(
            [
                [5.0, 4.0, -5.0],
                [4.0, 5.0, -5.0],
                [-5.0, -5.0, 0.0],
            ],
            dtype=torch.float32,
        )
        student_logits = torch.tensor(
            [
                [4.0, 3.0, 1.0],
                [3.0, 4.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        teacher = pfm_model.GraphMatcherOutput(teacher_logits, torch.empty((0, 2), dtype=torch.long), torch.empty((0,)))
        student = pfm_model.GraphMatcherOutput(student_logits, torch.empty((0, 2), dtype=torch.long), torch.empty((0,)))

        margin_guard, _margin_metrics = train.graph_matcher_teacher_guard_loss(
            student,
            teacher,
            positive_count=2,
        )
        score_floor, score_metrics = train.graph_matcher_teacher_score_floor_loss(
            student,
            teacher,
            positive_count=2,
            tolerance=0.5,
            min_teacher_score=0.0,
        )
        identical, identical_metrics = train.graph_matcher_teacher_score_floor_loss(
            teacher,
            teacher,
            positive_count=2,
            tolerance=0.5,
            min_teacher_score=0.0,
        )

        self.assertAlmostEqual(float(margin_guard), 0.0, places=6)
        self.assertGreater(float(score_floor), 0.0)
        self.assertEqual(float(score_metrics["violations"]), 2.0)
        self.assertLess(float(score_metrics["score_delta_mean"]), 0.0)
        self.assertAlmostEqual(float(identical), 0.0, places=6)
        self.assertEqual(float(identical_metrics["violations"]), 0.0)

    def test_graph_matcher_correspondence_loss_can_distill_external_teacher_distribution(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        teacher_logits = torch.tensor(
            [
                [5.0, -1.0, -3.0],
                [-1.0, 5.0, -3.0],
                [-3.0, -3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        teacher = pfm_model.GraphMatcherOutput(
            teacher_logits,
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

        loss, components = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            teacher_guard_output=teacher,
            teacher_distillation_weight=0.5,
            teacher_distillation_temperature=1.25,
            return_components=True,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("graph_matcher_teacher_distillation_loss", components)
        self.assertGreaterEqual(float(components["graph_matcher_teacher_distillation_loss"].detach()), 0.0)

    def test_positive_dustbin_guard_disables_negative_dustbin_losses(self):
        diagnostics = {
            "true_match_rejected_by_dustbin_ratio": 0.35,
            "positive_vs_dustbin_margin_mean": 0.5,
        }

        self.assertTrue(
            train.should_apply_positive_dustbin_guard(
                diagnostics,
                reject_threshold=0.2,
                margin_threshold=1.0,
            )
        )
        self.assertFalse(
            train.should_apply_positive_dustbin_guard(
                diagnostics,
                reject_threshold=0.8,
                margin_threshold=-1.0,
            )
        )

    def test_extractor_freeze_warmup_preserves_original_trainable_mask(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=2,
        )
        original = {}
        for name, parameter in model.named_parameters():
            trainable = name.startswith("backbone.") or name.startswith("graph_matcher.")
            parameter.requires_grad_(trainable)
            original[name] = trainable

        train.apply_extractor_freeze_warmup(model, original, freeze_extractor=True)

        for name, parameter in model.named_parameters():
            if name.startswith("graph_matcher."):
                self.assertEqual(parameter.requires_grad, original[name])
            else:
                self.assertFalse(parameter.requires_grad)

        train.apply_extractor_freeze_warmup(model, original, freeze_extractor=False)

        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, original[name])

    def test_graph_matcher_dustbin_weight_schedule_warms_up_and_ramps(self):
        self.assertEqual(train.scheduled_graph_matcher_weight(0.4, step=5, warmup_steps=10, ramp_steps=20), 0.0)
        self.assertAlmostEqual(
            train.scheduled_graph_matcher_weight(0.4, step=20, warmup_steps=10, ramp_steps=20),
            0.2,
        )
        self.assertAlmostEqual(
            train.scheduled_graph_matcher_weight(0.4, step=31, warmup_steps=10, ramp_steps=20),
            0.4,
        )

    def test_parse_graph_supervision_depths_accepts_comma_list(self):
        self.assertEqual(train.parse_graph_supervision_depths("1,2,4"), [1, 2, 4])
        self.assertEqual(train.parse_graph_supervision_depths(""), [])
        with self.assertRaises(argparse.ArgumentTypeError):
            train.parse_graph_supervision_depths("1,0,2")

    def test_graph_matcher_correspondence_loss_uses_deep_supervision_depths(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=4)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        calls = []

        def fake_forward(desc_a, meta_a, desc_b, meta_b, **kwargs):
            depth = int(kwargs.get("max_attention_layers", 0))
            calls.append(depth)
            model.graph_matcher.last_executed_attention_layers = depth
            logits = desc_a.new_zeros((desc_a.size(0) + 1, desc_b.size(0) + 1))
            logits[0, 0] = 3.0
            logits[1, 1] = 3.0
            return pfm_model.GraphMatcherOutput(
                logits=logits,
                matches=torch.empty((0, 2), dtype=torch.long),
                scores=torch.empty((0,), dtype=torch.float32),
                accept_logits=desc_a.new_zeros((desc_a.size(0), desc_b.size(0))),
                executed_layers=depth,
                attention_work_fraction=1.0,
            )

        with mock.patch.object(model.graph_matcher, "forward", side_effect=fake_forward):
            loss, components = train.graph_matcher_correspondence_loss(
                model,
                descriptors_a,
                descriptors_b,
                points,
                points,
                max_attention_layers=3,
                deep_supervision_depths=[1, 2],
                deep_supervision_weight=0.5,
                return_components=True,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(calls, [1, 2, 3])
        self.assertIn("graph_matcher_deep_supervision_loss", components)
        self.assertGreaterEqual(float(components["graph_matcher_deep_supervision_loss"].detach()), 0.0)

    def test_graph_matcher_correspondence_loss_can_randomize_attention_layer_budget(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=3)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
        generator = torch.Generator().manual_seed(2)

        loss = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            random_attention_layers=True,
            generator=generator,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.graph_matcher.last_executed_attention_layers, 1)

    def test_graph_matcher_correspondence_loss_can_train_with_attention_work_budget(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=2,
        )
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)

        loss, components = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            max_attention_work_fraction=0.5,
            return_components=True,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.graph_matcher.last_executed_attention_layers, 1)
        self.assertAlmostEqual(float(components["graph_matcher_attention_work_fraction"].item()), 0.5)

    def test_graph_matcher_correspondence_loss_can_train_with_width_dropout(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
        )
        generator = torch.Generator().manual_seed(20260606)

        loss, components = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            width_keep_ratio=0.5,
            generator=generator,
            return_components=True,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(components["graph_matcher_positive_pairs"].item()), 3)

    def test_graph_matcher_assignment_loss_uses_dual_softmax_and_dustbin(self):
        good_logits = torch.zeros(4, 4)
        good_logits[0, 0] = 10.0
        good_logits[1, 1] = 10.0
        good_logits[2, 3] = 10.0
        good_logits[3, 2] = 10.0
        bad_logits = torch.zeros(4, 4)
        bad_logits[0, 1] = 10.0
        bad_logits[1, 0] = 10.0
        bad_logits[2, 2] = 10.0
        bad_logits[3, 3] = 10.0
        good = pfm_model.GraphMatcherOutput(
            logits=good_logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )
        bad = pfm_model.GraphMatcherOutput(
            logits=bad_logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )

        good_loss = train.graph_matcher_assignment_loss(good, positive_count=2)
        bad_loss = train.graph_matcher_assignment_loss(bad, positive_count=2)

        self.assertLess(float(good_loss), 0.1)
        self.assertGreater(float(bad_loss), 5.0)

    def test_graph_matcher_prune_ranking_loss_penalizes_hard_negative_accept_scores(self):
        logits = torch.zeros(4, 4)
        accept_logits = torch.tensor(
            [
                [0.1, 1.2, -2.0],
                [1.1, 0.0, -2.0],
                [1.4, 1.3, -2.0],
            ],
            dtype=torch.float32,
        )
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
            accept_logits=accept_logits,
        )

        loss = train.graph_matcher_prune_ranking_loss(output, positive_count=2, margin=0.5)

        self.assertGreater(float(loss), 0.0)

    def test_graph_matcher_stop_confidence_loss_penalizes_confident_wrong_assignment(self):
        logits = torch.zeros(4, 4)
        logits[:3, :3] = torch.tensor(
            [
                [0.1, 5.0, -2.0],
                [4.8, 0.2, -2.0],
                [-2.0, -2.0, 0.0],
            ],
            dtype=torch.float32,
        )
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )

        loss = train.graph_matcher_stop_confidence_loss(output, positive_count=2, safe_margin=0.5)

        self.assertGreater(float(loss), 1.0)

    def test_graph_matcher_dustbin_diagnostics_detects_rejected_true_matches(self):
        logits = torch.zeros(4, 4)
        logits[:3, :3] = torch.tensor(
            [
                [0.1, -1.0, -1.0],
                [-1.0, 0.2, -1.0],
                [-1.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        logits[0, 3] = 2.0
        logits[1, 3] = 2.0
        logits[3, 0] = 1.5
        logits[3, 1] = 1.5
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )

        metrics = train.graph_matcher_dustbin_diagnostics(output, positive_count=2)

        self.assertAlmostEqual(metrics["true_match_rejected_by_dustbin_ratio"], 1.0)
        self.assertLess(metrics["positive_vs_dustbin_margin_mean"], -3.0)
        self.assertIn("dustbin_logit_for_true_match_mean", metrics)
        self.assertAlmostEqual(metrics["dustbin_logit_for_true_match_mean"], 3.5)
        self.assertAlmostEqual(metrics["dustbin_logit_mean"], 3.5)
        self.assertGreater(metrics["dustbin_prob_for_true_match_mean"], metrics["true_pair_prob_mean"])

    def test_graph_matcher_dustbin_diagnostics_reports_false_accept_ratio(self):
        logits = torch.zeros(3, 3)
        logits[:2, :2] = torch.tensor(
            [
                [5.0, 2.0],
                [-1.0, 5.0],
            ],
            dtype=torch.float32,
        )
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )

        metrics = train.graph_matcher_dustbin_diagnostics(output, positive_count=2)

        self.assertAlmostEqual(metrics["false_match_accepted_ratio"], 0.5)

    def test_graph_matcher_dustbin_diagnostics_reports_accept_logit_mean(self):
        logits = torch.zeros(3, 3)
        accept_logits = torch.tensor(
            [
                [1.0, -2.0],
                [-3.0, 3.0],
            ],
            dtype=torch.float32,
        )
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
            accept_logits=accept_logits,
        )

        metrics = train.graph_matcher_dustbin_diagnostics(output, positive_count=2)

        self.assertAlmostEqual(metrics["accept_logit_mean"], 2.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA autocast is required for this regression test")
    def test_graph_matcher_stop_confidence_loss_is_safe_under_cuda_autocast(self):
        logits = torch.zeros(4, 4, device="cuda")
        logits[:3, :3] = torch.tensor(
            [
                [3.0, 0.1, -2.0],
                [0.2, 3.2, -2.0],
                [-2.0, -2.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long, device="cuda"),
            scores=torch.empty((0,), dtype=torch.float32, device="cuda"),
        )

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss = train.graph_matcher_stop_confidence_loss(output, positive_count=2, safe_margin=0.5)

        self.assertTrue(torch.isfinite(loss))

    def test_graph_matcher_correspondence_loss_can_return_lightglue_components(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=1)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)

        loss, components = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            accept_weight=0.5,
            accept_negative_topk=2,
            assignment_weight=0.25,
            prune_ranking_weight=0.25,
            prune_ranking_margin=0.5,
            stop_confidence_weight=0.1,
            stop_confidence_margin=0.5,
            return_components=True,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(components["graph_matcher_ce_loss"].detach()), 0.0)
        self.assertGreater(float(components["graph_matcher_assignment_loss"].detach()), 0.0)
        self.assertGreater(float(components["graph_matcher_accept_loss"].detach()), 0.0)
        self.assertIn("graph_matcher_prune_ranking_loss", components)
        self.assertIn("graph_matcher_stop_confidence_loss", components)
        self.assertIn("graph_matcher_total_loss", components)
        self.assertIn("true_match_rejected_by_dustbin_ratio", components)
        self.assertIn("positive_vs_dustbin_margin_mean", components)
        self.assertIn("dustbin_logit_mean", components)
        self.assertIn("dustbin_logit_for_true_match_mean", components)
        self.assertIn("false_match_accepted_ratio", components)
        self.assertIn("accept_logit_mean", components)
        self.assertIn("true_match_in_topk@64", components)
        self.assertIn("true_match_in_topk@256", components)

    def test_graph_matcher_correspondence_loss_returns_final_false_match_components(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)

        _loss, components = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            final_false_match_weight=0.5,
            final_false_match_topk=2,
            final_false_match_min_score=0.0,
            final_false_match_margin=0.25,
            return_components=True,
        )

        self.assertIn("graph_matcher_final_false_match_loss", components)
        self.assertIn("graph_matcher_final_false_match_edges", components)
        self.assertIn("graph_matcher_final_false_match_score_mean", components)
        self.assertIn("graph_matcher_final_false_match_accept_mean", components)
        self.assertIn("graph_matcher_mined_false_match_loss", components)
        self.assertIn("graph_matcher_mined_false_match_edges", components)
        self.assertIn("graph_matcher_mined_false_match_reference_filtered_edges", components)
        self.assertIn("graph_matcher_extra_false_match_pairs", components)

    def test_graph_matcher_correspondence_loss_returns_teacher_guard_components(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        teacher_logits = torch.tensor(
            [
                [5.0, -1.0, -3.0],
                [-1.0, 5.0, -3.0],
                [-3.0, -3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        teacher = pfm_model.GraphMatcherOutput(
            teacher_logits,
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

        loss, components = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            teacher_guard_output=teacher,
            teacher_guard_weight=0.5,
            teacher_guard_positive_margin_tolerance=0.1,
            teacher_guard_false_margin_tolerance=0.1,
            return_components=True,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("graph_matcher_teacher_guard_loss", components)
        self.assertIn("graph_matcher_teacher_guard_positive_margin_loss", components)
        self.assertIn("graph_matcher_teacher_guard_false_edge_loss", components)
        self.assertIn("graph_matcher_teacher_guard_positive_violations", components)
        self.assertIn("graph_matcher_teacher_guard_false_edges", components)

    def test_graph_matcher_correspondence_loss_returns_teacher_score_floor_components(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        teacher_logits = torch.tensor(
            [
                [5.0, 4.0, -5.0],
                [4.0, 5.0, -5.0],
                [-5.0, -5.0, 0.0],
            ],
            dtype=torch.float32,
        )
        teacher = pfm_model.GraphMatcherOutput(
            teacher_logits,
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

        loss, components = train.graph_matcher_correspondence_loss(
            model,
            descriptors_a,
            descriptors_b,
            points,
            points,
            teacher_guard_output=teacher,
            teacher_score_floor_weight=0.5,
            teacher_score_floor_tolerance=0.25,
            teacher_score_floor_min_score=0.0,
            return_components=True,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("graph_matcher_teacher_score_floor_loss", components)
        self.assertIn("graph_matcher_teacher_score_floor_violations", components)
        self.assertIn("graph_matcher_teacher_score_floor_delta_mean", components)
        self.assertIn("graph_matcher_teacher_score_floor_teacher_score_mean", components)

    def test_graph_matcher_correspondence_loss_can_use_frozen_teacher_guard_model(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        teacher = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

        with mock.patch.object(teacher.graph_matcher, "forward", wraps=teacher.graph_matcher.forward) as forward:
            loss, components = train.graph_matcher_correspondence_loss(
                model,
                descriptors_a,
                descriptors_b,
                points,
                points,
                teacher_guard_model=teacher,
                teacher_guard_weight=0.5,
                teacher_guard_positive_margin_tolerance=0.1,
                teacher_guard_false_margin_tolerance=0.1,
                return_components=True,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(forward.call_count, 1)
        self.assertIn("graph_matcher_teacher_guard_loss", components)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_train_step_passes_teacher_guard_model_to_graph_loss(self):
        pair_path = Path("pair_000001.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 4, 4),
            view_b=torch.ones(1, 4, 4),
            warp_a_to_b=torch.zeros(4, 4, 2),
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        teacher = object()
        graph_calls: list[dict] = []

        def fake_graph_loss(*_args, **kwargs):
            graph_calls.append(kwargs)
            return parameter * 2.0, {
                "graph_matcher_total_loss": torch.tensor(2.0),
                "graph_matcher_teacher_guard_loss": torch.tensor(0.5),
                "graph_matcher_teacher_guard_positive_margin_loss": torch.tensor(0.25),
                "graph_matcher_teacher_guard_false_edge_loss": torch.tensor(0.25),
                "graph_matcher_teacher_guard_positive_violations": torch.tensor(1.0),
                "graph_matcher_teacher_guard_false_edges": torch.tensor(2.0),
                "graph_matcher_teacher_score_floor_loss": torch.tensor(0.125),
                "graph_matcher_teacher_score_floor_violations": torch.tensor(3.0),
                "graph_matcher_teacher_score_floor_delta_mean": torch.tensor(-0.4),
                "graph_matcher_teacher_score_floor_teacher_score_mean": torch.tensor(1.2),
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 4, 4, 4),
                    torch.ones(1, 4, 4, 4),
                    torch.ones(1, 4, 4, 4),
                    torch.ones(1, 4, 4, 4),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "graph_matcher_correspondence_loss", side_effect=fake_graph_loss),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.0,
                synthetic_loss_weight=0.0,
                graph_matcher_loss_weight=1.0,
                graph_matcher_teacher_guard_model=teacher,
                graph_matcher_teacher_guard_weight=0.5,
                graph_matcher_teacher_guard_positive_margin_tolerance=0.1,
                graph_matcher_teacher_guard_false_margin_tolerance=0.2,
                graph_matcher_teacher_score_floor_weight=0.25,
                graph_matcher_teacher_score_floor_tolerance=0.3,
                graph_matcher_teacher_score_floor_min_score=0.4,
            )

        self.assertEqual(len(graph_calls), 1)
        self.assertIs(graph_calls[0]["teacher_guard_model"], teacher)
        self.assertAlmostEqual(graph_calls[0]["teacher_guard_weight"], 0.5)
        self.assertAlmostEqual(graph_calls[0]["teacher_guard_positive_margin_tolerance"], 0.1)
        self.assertAlmostEqual(graph_calls[0]["teacher_guard_false_margin_tolerance"], 0.2)
        self.assertAlmostEqual(graph_calls[0]["teacher_score_floor_weight"], 0.25)
        self.assertAlmostEqual(graph_calls[0]["teacher_score_floor_tolerance"], 0.3)
        self.assertAlmostEqual(graph_calls[0]["teacher_score_floor_min_score"], 0.4)
        self.assertAlmostEqual(graph_calls[0]["teacher_distillation_weight"], 0.0)
        self.assertAlmostEqual(metrics["graph_matcher_teacher_guard_loss"], 0.5)
        self.assertAlmostEqual(metrics["graph_matcher_teacher_score_floor_loss"], 0.125)

    def test_train_step_passes_teacher_distillation_to_graph_loss(self):
        pair_path = Path("pair_000001.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 4, 4),
            view_b=torch.ones(1, 4, 4),
            warp_a_to_b=torch.zeros(4, 4, 2),
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        teacher = object()
        graph_calls: list[dict] = []

        def fake_graph_loss(*_args, **kwargs):
            graph_calls.append(kwargs)
            return parameter * 2.0, {
                "graph_matcher_total_loss": torch.tensor(2.0),
                "graph_matcher_teacher_distillation_loss": torch.tensor(0.75),
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 4, 4, 4),
                    torch.ones(1, 4, 4, 4),
                    torch.ones(1, 4, 4, 4),
                    torch.ones(1, 4, 4, 4),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "graph_matcher_correspondence_loss", side_effect=fake_graph_loss),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.0,
                synthetic_loss_weight=0.0,
                graph_matcher_loss_weight=1.0,
                graph_matcher_teacher_guard_model=teacher,
                graph_matcher_teacher_distillation_weight=0.35,
                graph_matcher_teacher_distillation_temperature=1.75,
            )

        self.assertEqual(len(graph_calls), 1)
        self.assertIs(graph_calls[0]["teacher_guard_model"], teacher)
        self.assertAlmostEqual(graph_calls[0]["teacher_distillation_weight"], 0.35)
        self.assertAlmostEqual(graph_calls[0]["teacher_distillation_temperature"], 1.75)
        self.assertAlmostEqual(metrics["graph_matcher_teacher_distillation_loss"], 0.75)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA autocast is required for this regression test")
    def test_matchability_supervision_loss_is_safe_under_cuda_autocast(self):
        score_map = torch.sigmoid(torch.randn(1, 1, 8, 8, device="cuda"))
        positive_points = torch.tensor([[2.0, 3.0], [5.0, 6.0]], dtype=torch.float32, device="cuda")
        negative_points = torch.tensor([[1.0, 1.0], [6.0, 2.0]], dtype=torch.float32, device="cuda")

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss = train.matchability_supervision_loss(score_map, positive_points, negative_points)

        self.assertTrue(torch.isfinite(loss))

    def test_graph_matcher_raw_preservation_loss_penalizes_degraded_raw_margin(self):
        desc = torch.eye(3)
        logits = torch.zeros(4, 4)
        logits[:3, :3] = torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [0.0, 1.0, 2.0],
                [2.0, 0.0, 1.0],
            ]
        )

        loss = train.graph_matcher_raw_preservation_loss(
            logits,
            desc,
            desc,
            target_margin=1.0,
            raw_margin_threshold=0.1,
        )

        self.assertGreater(float(loss), 0.0)

    def test_graph_matcher_hard_negative_dustbin_loss_penalizes_raw_confusable_edges(self):
        desc_a = torch.eye(3)
        desc_b = torch.eye(3)
        desc_b[1] = torch.tensor([0.96, 0.28, 0.0])
        desc_b = torch.nn.functional.normalize(desc_b, dim=1)
        logits = torch.zeros(4, 4)
        logits[0, 0] = 3.0
        logits[1, 1] = 3.0
        logits[2, 2] = 3.0
        logits[0, 1] = 2.0
        logits[:3, 3] = 1.0
        logits[3, :3] = 1.0

        loss = train.graph_matcher_hard_negative_dustbin_loss(
            logits,
            desc_a,
            desc_b,
            positive_count=3,
            negative_topk=1,
            margin=0.25,
        )

        self.assertGreater(float(loss), 0.0)
        safer_logits = logits.clone()
        safer_logits[0, 1] = 0.2
        self.assertAlmostEqual(
            float(
                train.graph_matcher_hard_negative_dustbin_loss(
                    safer_logits,
                    desc_a,
                    desc_b,
                    positive_count=3,
                    negative_topk=1,
                    margin=0.25,
                )
            ),
            0.0,
        )

    def test_graph_matcher_raw_false_match_loss_penalizes_confusable_pair_logits(self):
        desc_a = torch.eye(3)
        desc_b = torch.eye(3)
        desc_b[1] = torch.tensor([0.96, 0.28, 0.0])
        desc_b = torch.nn.functional.normalize(desc_b, dim=1)
        logits = torch.zeros(4, 4)
        logits[0, 0] = 2.0
        logits[1, 1] = 2.0
        logits[2, 2] = 2.0
        logits[0, 1] = 1.9

        loss, metrics = train.graph_matcher_raw_false_match_loss(
            logits,
            desc_a,
            desc_b,
            positive_count=3,
            negative_topk=1,
            min_raw_similarity=0.8,
            margin=0.25,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(int(metrics["edges"].item()), 1)
        safer_logits = logits.clone()
        safer_logits[0, 1] = 0.2
        safer_loss, safer_metrics = train.graph_matcher_raw_false_match_loss(
            safer_logits,
            desc_a,
            desc_b,
            positive_count=3,
            negative_topk=1,
            min_raw_similarity=0.8,
            margin=0.25,
        )
        self.assertAlmostEqual(float(safer_loss), 0.0)
        self.assertEqual(int(safer_metrics["edges"].item()), 1)

    def test_graph_matcher_hard_negative_dustbin_loss_can_ignore_nearby_target_candidates(self):
        desc_a = torch.eye(4)
        desc_b = torch.eye(4)
        desc_b[1] = torch.tensor([0.95, 0.31, 0.0, 0.0])
        desc_b[2] = torch.tensor([0.94, 0.0, 0.34, 0.0])
        desc_b = torch.nn.functional.normalize(desc_b, dim=1)
        logits = torch.zeros(5, 5)
        logits[:4, 4] = 1.0
        logits[4, :4] = 1.0
        logits[0, 1] = 2.0
        logits[0, 2] = 2.0
        points_b = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [16.0, 0.0],
                [0.0, 16.0],
            ],
            dtype=torch.float32,
        )

        loss = train.graph_matcher_hard_negative_dustbin_loss(
            logits,
            desc_a,
            desc_b,
            positive_count=4,
            negative_topk=2,
            margin=0.25,
            points_b_xy=points_b,
            spatial_min_distance=4.0,
        )

        self.assertGreater(float(loss), 0.0)
        safer_logits = logits.clone()
        safer_logits[0, 2] = 0.2
        self.assertAlmostEqual(
            float(
                train.graph_matcher_hard_negative_dustbin_loss(
                    safer_logits,
                    desc_a,
                    desc_b,
                    positive_count=4,
                    negative_topk=2,
                    margin=0.25,
                    points_b_xy=points_b,
                    spatial_min_distance=4.0,
                )
            ),
            0.0,
        )

    def test_apply_graph_metadata_mode_can_remove_xy_prior(self):
        metadata = torch.arange(28, dtype=torch.float32).view(2, 14)

        adjusted = train.apply_graph_metadata_mode(metadata, "no_xy")

        self.assertTrue(torch.equal(adjusted[:, :4], torch.zeros_like(adjusted[:, :4])))
        self.assertTrue(torch.equal(adjusted[:, 4:], metadata[:, 4:]))

    def test_apply_graph_metadata_mode_calibrated_removes_reliability_shortcuts(self):
        metadata = torch.arange(32, dtype=torch.float32).view(2, 16)

        adjusted = train.apply_graph_metadata_mode(metadata, "calibrated")

        self.assertTrue(torch.equal(adjusted[:, :12], metadata[:, :12]))
        self.assertTrue(torch.equal(adjusted[:, 13:14], metadata[:, 13:14]))
        self.assertTrue(torch.equal(adjusted[:, 12:13], torch.zeros_like(adjusted[:, 12:13])))
        self.assertTrue(torch.equal(adjusted[:, 14:16], torch.zeros_like(adjusted[:, 14:16])))

    def test_parse_args_accepts_balanced_cache_sampling(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--balanced-cache-sampling",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertTrue(args.balanced_cache_sampling)

    def test_parse_args_defaults_graph_metadata_to_calibrated(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--train-graph-matcher",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertEqual(args.graph_matcher_metadata_mode, "calibrated")

    def test_parse_args_accepts_graph_matcher_no_match_options(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--train-graph-matcher",
            "--graph-matcher-metadata-mode",
            "no_xy",
            "--graph-matcher-no-match-points",
            "32",
            "--graph-matcher-no-match-weight",
            "0.25",
            "--graph-matcher-train-max-attention-layers",
            "2",
            "--graph-matcher-train-random-attention-layers",
            "--graph-matcher-train-max-attention-work-fraction",
            "0.5",
            "--graph-matcher-train-width-keep-ratio",
            "0.5",
            "--graph-matcher-deep-supervision-depths",
            "1,2",
            "--graph-matcher-deep-supervision-weight",
            "0.4",
            "--matcher-reliability-pair-bias",
            "off",
            "--matcher-reliability-dustbin-bias",
            "matchability",
            "--matcher-final-accept-score-mode",
            "add",
            "--matcher-accept-assignment-mode",
            "off",
            "--matcher-final-accept-score-alpha",
            "0.07",
            "--matcher-geometry-bias-scale",
            "0.25",
            "--matcher-geometry-bias-clamp",
            "1.5",
            "--matcher-attention-residual-gate-init",
            "0.05",
            "--matcher-attention-residual-gate-start-layer",
            "5",
            "--matcher-candidate-topk",
            "96",
            "--graph-matcher-accept-weight",
            "0.2",
            "--graph-matcher-assignment-weight",
            "0.35",
            "--graph-matcher-train-candidate-topk",
            "64",
            "--graph-matcher-accept-negative-topk",
            "6",
            "--graph-matcher-prune-ranking-weight",
            "0.15",
            "--graph-matcher-prune-ranking-margin",
            "0.4",
            "--graph-matcher-stop-confidence-weight",
            "0.07",
            "--graph-matcher-stop-confidence-margin",
            "0.6",
            "--graph-matcher-raw-preservation-weight",
            "0.1",
            "--graph-matcher-raw-preservation-margin",
            "1.5",
            "--graph-matcher-raw-preservation-raw-margin",
            "0.04",
            "--graph-matcher-hard-negative-dustbin-weight",
            "0.3",
            "--graph-matcher-hard-negative-dustbin-topk",
            "12",
            "--graph-matcher-hard-negative-dustbin-margin",
            "0.35",
            "--graph-matcher-hard-negative-dustbin-spatial-min-distance",
            "4.5",
            "--graph-matcher-dustbin-warmup-steps",
            "100",
            "--graph-matcher-dustbin-ramp-steps",
            "300",
            "--graph-matcher-positive-dustbin-margin-weight",
            "0.45",
            "--graph-matcher-positive-dustbin-margin",
            "0.2",
            "--graph-matcher-true-match-margin-weight",
            "0.06",
            "--graph-matcher-true-match-margin",
            "0.4",
            "--graph-matcher-final-false-match-weight",
            "0.05",
            "--graph-matcher-mined-false-match-weight",
            "0.025",
            "--graph-matcher-mined-false-match-loss-cap",
            "3.5",
            "--graph-matcher-mined-false-match-reference-margin",
            "0.5",
            "--graph-matcher-final-false-match-topk",
            "4",
            "--graph-matcher-final-false-match-min-score",
            "0.02",
            "--graph-matcher-final-false-match-margin",
            "0.3",
            "--graph-matcher-final-false-match-spatial-min-distance",
            "5.0",
            "--graph-matcher-raw-false-match-weight",
            "0.04",
            "--graph-matcher-raw-false-match-topk",
            "3",
            "--graph-matcher-raw-false-match-min-similarity",
            "0.82",
            "--graph-matcher-raw-false-match-margin",
            "0.45",
            "--graph-matcher-raw-false-match-spatial-min-distance",
            "6.5",
            "--graph-matcher-ransac-consistency-weight",
            "0.07",
            "--graph-matcher-ransac-consistency-topk",
            "5",
            "--graph-matcher-ransac-consistency-residual-threshold-px",
            "2.5",
            "--graph-matcher-ransac-consistency-min-score",
            "0.03",
            "--graph-matcher-ransac-consistency-margin",
            "0.4",
            "--graph-matcher-depth-distillation-weight",
            "0.6",
            "--graph-matcher-depth-distillation-teacher-layers",
            "4",
            "--graph-matcher-depth-distillation-temperature",
            "1.5",
            "--graph-matcher-teacher-guard-state",
            "/tmp/best_teacher.pt",
            "--graph-matcher-teacher-guard-weight",
            "0.55",
            "--graph-matcher-teacher-guard-positive-margin-tolerance",
            "0.15",
            "--graph-matcher-teacher-guard-false-margin-tolerance",
            "0.25",
            "--graph-matcher-teacher-score-floor-weight",
            "0.45",
            "--graph-matcher-teacher-score-floor-tolerance",
            "0.35",
            "--graph-matcher-teacher-score-floor-min-score",
            "0.2",
            "--graph-matcher-teacher-distillation-weight",
            "0.35",
            "--graph-matcher-teacher-distillation-temperature",
            "1.75",
            "--graph-matcher-positive-dustbin-guard-reject-threshold",
            "0.2",
            "--graph-matcher-positive-dustbin-guard-margin-threshold",
            "1.0",
            "--freeze-extractor-warmup-steps",
            "600",
            "--graph-matcher-semi-dense-no-match-points",
            "24",
            "--graph-matcher-semi-dense-min-score",
            "0.02",
            "--graph-matcher-online-false-no-match",
            "--training-weak-texture-fraction",
            "0.25",
            "--training-spatial-bins",
            "8",
            "--hard-pair-glob",
            "*pair_004541*.pt",
            "--train-reliability-head",
            "--matchability-weight",
            "0.11",
            "--descriptor-uncertainty-weight",
            "0.12",
            "--no-match-prior-weight",
            "0.13",
            "--reliability-negative-points",
            "9",
            "--reliability-negative-min-distance",
            "3.5",
            "--rotation-descriptor-consistency-weight",
            "0.21",
            "--orientation-consistency-weight",
            "0.22",
            "--scale-consistency-weight",
            "0.23",
            "--affine-consistency-weight",
            "0.24",
            "--affine-regularization-weight",
            "0.25",
            "--descriptor-geometry-mode",
            "orientation_scale",
            "--descriptor-geometry-blend-weight",
            "0.35",
            "--descriptor-scale-log-clamp-min",
            "-0.7",
            "--descriptor-scale-log-clamp-max",
            "0.7",
            "--descriptor-geometry-safety-schedule",
            "phase4",
            "--quality-score-mode",
            "soft",
            "--rotation-consistency-degrees",
            "90,270",
            "--amp",
            "--amp-dtype",
            "float16",
            "--activation-checkpointing",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertEqual(args.graph_matcher_metadata_mode, "no_xy")
        self.assertEqual(args.graph_matcher_no_match_points, 32)
        self.assertAlmostEqual(args.graph_matcher_no_match_weight, 0.25)
        self.assertEqual(args.graph_matcher_train_max_attention_layers, 2)
        self.assertTrue(args.graph_matcher_train_random_attention_layers)
        self.assertAlmostEqual(args.graph_matcher_train_max_attention_work_fraction, 0.5)
        self.assertAlmostEqual(args.graph_matcher_train_width_keep_ratio, 0.5)
        self.assertEqual(args.graph_matcher_deep_supervision_depths, [1, 2])
        self.assertAlmostEqual(args.graph_matcher_deep_supervision_weight, 0.4)
        self.assertEqual(args.matcher_reliability_pair_bias, "off")
        self.assertEqual(args.matcher_reliability_dustbin_bias, "matchability")
        self.assertEqual(args.matcher_final_accept_score_mode, "add")
        self.assertEqual(args.matcher_accept_assignment_mode, "off")
        self.assertAlmostEqual(args.matcher_final_accept_score_alpha, 0.07)
        self.assertAlmostEqual(args.matcher_geometry_bias_scale, 0.25)
        self.assertAlmostEqual(args.matcher_geometry_bias_clamp, 1.5)
        self.assertAlmostEqual(args.matcher_attention_residual_gate_init, 0.05)
        self.assertEqual(args.matcher_attention_residual_gate_start_layer, 5)
        self.assertEqual(args.matcher_candidate_topk, 96)
        self.assertAlmostEqual(args.graph_matcher_accept_weight, 0.2)
        self.assertAlmostEqual(args.graph_matcher_assignment_weight, 0.35)
        self.assertEqual(args.graph_matcher_train_candidate_topk, 64)
        self.assertEqual(args.graph_matcher_accept_negative_topk, 6)
        self.assertAlmostEqual(args.graph_matcher_prune_ranking_weight, 0.15)
        self.assertAlmostEqual(args.graph_matcher_prune_ranking_margin, 0.4)
        self.assertAlmostEqual(args.graph_matcher_stop_confidence_weight, 0.07)
        self.assertAlmostEqual(args.graph_matcher_stop_confidence_margin, 0.6)
        self.assertAlmostEqual(args.graph_matcher_raw_preservation_weight, 0.1)
        self.assertAlmostEqual(args.graph_matcher_raw_preservation_margin, 1.5)
        self.assertAlmostEqual(args.graph_matcher_raw_preservation_raw_margin, 0.04)
        self.assertAlmostEqual(args.graph_matcher_hard_negative_dustbin_weight, 0.3)
        self.assertEqual(args.graph_matcher_hard_negative_dustbin_topk, 12)
        self.assertAlmostEqual(args.graph_matcher_hard_negative_dustbin_margin, 0.35)
        self.assertAlmostEqual(args.graph_matcher_hard_negative_dustbin_spatial_min_distance, 4.5)
        self.assertEqual(args.graph_matcher_dustbin_warmup_steps, 100)
        self.assertEqual(args.graph_matcher_dustbin_ramp_steps, 300)
        self.assertAlmostEqual(args.graph_matcher_positive_dustbin_margin_weight, 0.45)
        self.assertAlmostEqual(args.graph_matcher_positive_dustbin_margin, 0.2)
        self.assertAlmostEqual(args.graph_matcher_true_match_margin_weight, 0.06)
        self.assertAlmostEqual(args.graph_matcher_true_match_margin, 0.4)
        self.assertAlmostEqual(args.graph_matcher_final_false_match_weight, 0.05)
        self.assertAlmostEqual(args.graph_matcher_mined_false_match_weight, 0.025)
        self.assertAlmostEqual(args.graph_matcher_mined_false_match_loss_cap, 3.5)
        self.assertAlmostEqual(args.graph_matcher_mined_false_match_reference_margin, 0.5)
        self.assertEqual(args.graph_matcher_final_false_match_topk, 4)
        self.assertAlmostEqual(args.graph_matcher_final_false_match_min_score, 0.02)
        self.assertAlmostEqual(args.graph_matcher_final_false_match_margin, 0.3)
        self.assertAlmostEqual(args.graph_matcher_final_false_match_spatial_min_distance, 5.0)
        self.assertAlmostEqual(args.graph_matcher_raw_false_match_weight, 0.04)
        self.assertEqual(args.graph_matcher_raw_false_match_topk, 3)
        self.assertAlmostEqual(args.graph_matcher_raw_false_match_min_similarity, 0.82)
        self.assertAlmostEqual(args.graph_matcher_raw_false_match_margin, 0.45)
        self.assertAlmostEqual(args.graph_matcher_raw_false_match_spatial_min_distance, 6.5)
        self.assertAlmostEqual(args.graph_matcher_ransac_consistency_weight, 0.07)
        self.assertEqual(args.graph_matcher_ransac_consistency_topk, 5)
        self.assertAlmostEqual(args.graph_matcher_ransac_consistency_residual_threshold_px, 2.5)
        self.assertAlmostEqual(args.graph_matcher_ransac_consistency_min_score, 0.03)
        self.assertAlmostEqual(args.graph_matcher_ransac_consistency_margin, 0.4)
        self.assertAlmostEqual(args.graph_matcher_depth_distillation_weight, 0.6)
        self.assertEqual(args.graph_matcher_depth_distillation_teacher_layers, 4)
        self.assertAlmostEqual(args.graph_matcher_depth_distillation_temperature, 1.5)
        self.assertEqual(args.graph_matcher_teacher_guard_state, Path("/tmp/best_teacher.pt"))
        self.assertAlmostEqual(args.graph_matcher_teacher_guard_weight, 0.55)
        self.assertAlmostEqual(args.graph_matcher_teacher_guard_positive_margin_tolerance, 0.15)
        self.assertAlmostEqual(args.graph_matcher_teacher_guard_false_margin_tolerance, 0.25)
        self.assertAlmostEqual(args.graph_matcher_teacher_score_floor_weight, 0.45)
        self.assertAlmostEqual(args.graph_matcher_teacher_score_floor_tolerance, 0.35)
        self.assertAlmostEqual(args.graph_matcher_teacher_score_floor_min_score, 0.2)
        self.assertAlmostEqual(args.graph_matcher_teacher_distillation_weight, 0.35)
        self.assertAlmostEqual(args.graph_matcher_teacher_distillation_temperature, 1.75)
        self.assertAlmostEqual(args.graph_matcher_positive_dustbin_guard_reject_threshold, 0.2)
        self.assertAlmostEqual(args.graph_matcher_positive_dustbin_guard_margin_threshold, 1.0)
        self.assertEqual(args.freeze_extractor_warmup_steps, 600)
        self.assertEqual(args.graph_matcher_semi_dense_no_match_points, 24)
        self.assertAlmostEqual(args.graph_matcher_semi_dense_min_score, 0.02)
        self.assertTrue(args.graph_matcher_online_false_no_match)
        self.assertAlmostEqual(args.training_weak_texture_fraction, 0.25)
        self.assertEqual(args.training_spatial_bins, 8)
        self.assertEqual(args.hard_pair_glob, ["*pair_004541*.pt"])
        self.assertTrue(args.train_reliability_head)
        self.assertAlmostEqual(args.matchability_weight, 0.11)
        self.assertAlmostEqual(args.descriptor_uncertainty_weight, 0.12)
        self.assertAlmostEqual(args.no_match_prior_weight, 0.13)
        self.assertEqual(args.reliability_negative_points, 9)
        self.assertAlmostEqual(args.reliability_negative_min_distance, 3.5)
        self.assertAlmostEqual(args.rotation_descriptor_consistency_weight, 0.21)
        self.assertAlmostEqual(args.orientation_consistency_weight, 0.22)
        self.assertAlmostEqual(args.scale_consistency_weight, 0.23)
        self.assertAlmostEqual(args.affine_consistency_weight, 0.24)
        self.assertAlmostEqual(args.affine_regularization_weight, 0.25)
        self.assertEqual(args.descriptor_geometry_mode, "orientation_scale")
        self.assertAlmostEqual(args.descriptor_geometry_blend_weight, 0.35)
        self.assertAlmostEqual(args.descriptor_scale_log_clamp_min, -0.7)
        self.assertAlmostEqual(args.descriptor_scale_log_clamp_max, 0.7)
        self.assertEqual(args.descriptor_geometry_safety_schedule, "phase4")
        self.assertEqual(args.quality_score_mode, "soft")
        self.assertEqual(args.rotation_consistency_degrees, [90, 270])
        self.assertTrue(args.amp)
        self.assertEqual(args.amp_dtype, "float16")
        self.assertTrue(args.activation_checkpointing)

    def test_parse_args_enables_lightglue_accept_loss_for_graph_training_by_default(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--train-graph-matcher",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertAlmostEqual(args.graph_matcher_accept_weight, 0.2)
        self.assertAlmostEqual(args.graph_matcher_prune_ranking_weight, 0.1)
        self.assertAlmostEqual(args.graph_matcher_stop_confidence_weight, 0.05)
        self.assertAlmostEqual(args.graph_matcher_stop_confidence_margin, 0.5)

    def test_parse_args_enable_rejection_training_expands_safe_defaults(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--enable-rejection-training",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertTrue(args.enable_rejection_training)
        self.assertTrue(args.train_graph_matcher)
        self.assertTrue(args.graph_matcher_online_false_no_match)
        self.assertAlmostEqual(args.graph_matcher_loss_weight, 0.75)
        self.assertEqual(args.graph_matcher_no_match_points, 128)
        self.assertAlmostEqual(args.graph_matcher_no_match_weight, 0.45)
        self.assertAlmostEqual(args.graph_matcher_assignment_weight, 0.35)
        self.assertAlmostEqual(args.graph_matcher_accept_weight, 0.30)
        self.assertAlmostEqual(args.graph_matcher_prune_ranking_weight, 0.10)
        self.assertAlmostEqual(args.graph_matcher_stop_confidence_weight, 0.05)
        self.assertAlmostEqual(args.graph_matcher_hard_negative_dustbin_weight, 0.075)
        self.assertEqual(args.graph_matcher_hard_negative_dustbin_topk, 16)
        self.assertAlmostEqual(args.graph_matcher_hard_negative_dustbin_margin, 0.35)
        self.assertEqual(args.graph_matcher_semi_dense_no_match_points, 128)
        self.assertAlmostEqual(args.false_match_weight, 0.15)
        self.assertEqual(args.false_match_max_points, 192)
        self.assertAlmostEqual(args.keypoint_weight, 0.05)
        self.assertAlmostEqual(args.keypoint_negative_weight, 0.02)
        self.assertEqual(args.reliability_negative_points, 128)
        self.assertAlmostEqual(args.matchability_weight, 0.08)
        self.assertAlmostEqual(args.descriptor_uncertainty_weight, 0.05)
        self.assertAlmostEqual(args.no_match_prior_weight, 0.08)
        self.assertAlmostEqual(args.rotation_descriptor_consistency_weight, 0.03)
        self.assertEqual(args.report_matcher_mode, "graph_matcher")
        self.assertEqual(args.report_graph_inference_preset, "fast")

    def test_parse_args_accepts_gradient_accumulation_steps(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--gradient-accumulation-steps",
            "2",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertEqual(args.gradient_accumulation_steps, 2)

    def test_parse_args_skips_nonfinite_steps_by_default(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertTrue(args.skip_nonfinite_steps)

        with mock.patch.object(sys, "argv", [*argv, "--no-skip-nonfinite-steps"]):
            args = train.parse_args()

        self.assertFalse(args.skip_nonfinite_steps)

    def test_parse_args_accepts_pose_metadata_options(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--pose-metadata-root",
            "pose_root",
            "--pose-balanced-sampling",
            "--pose-min-overlap",
            "0.4",
            "--pose-difficulty-loss-weight",
            "0.75",
            "--training-max-image-size",
            "1024",
            "--training-crop-size",
            "1024",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertEqual(args.pose_metadata_root, [Path("pose_root")])
        self.assertTrue(args.pose_balanced_sampling)
        self.assertEqual(args.pose_min_overlap, 0.4)
        self.assertEqual(args.pose_difficulty_loss_weight, 0.75)
        self.assertEqual(args.training_max_image_size, 1024)
        self.assertEqual(args.training_crop_size, 1024)

    def test_parse_args_accepts_training_report_options(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--validation-cache-dir",
            "val",
            "--generate-training-report",
            "--report-output-dir",
            "report",
            "--report-sample-count",
            "6",
            "--report-max-keypoints",
            "1536",
            "--report-max-matches",
            "384",
            "--report-draw-matches",
            "96",
            "--report-min-margin",
            "0.01",
            "--report-graph-width-prune-min-score",
            "0.25",
            "--report-graph-early-stop-min-confidence",
            "0.85",
            "--report-graph-inference-preset",
            "fast",
            "--report-graph-min-accept-probability",
            "0.7",
            "--report-graph-max-attention-work-fraction",
            "0.55",
            "--report-graph-width-prune-keep-ratio",
            "0.4",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertTrue(args.generate_training_report)
        self.assertEqual(args.report_output_dir, Path("report"))
        self.assertEqual(args.report_sample_count, 6)
        self.assertEqual(args.report_max_keypoints, 1536)
        self.assertEqual(args.report_max_matches, 384)
        self.assertEqual(args.report_draw_matches, 96)
        self.assertEqual(args.report_min_margin, 0.01)
        self.assertEqual(args.report_graph_width_prune_min_score, 0.25)
        self.assertEqual(args.report_graph_early_stop_min_confidence, 0.85)
        self.assertEqual(args.report_graph_inference_preset, "fast")
        self.assertEqual(args.report_graph_min_accept_probability, 0.7)
        self.assertEqual(args.report_graph_max_attention_work_fraction, 0.55)
        self.assertEqual(args.report_graph_width_prune_keep_ratio, 0.4)

    def test_training_report_command_forwards_lightglue_graph_options(self):
        with tempfile.TemporaryDirectory() as temp:
            args = train.argparse.Namespace(
                validation_cache_dir=[Path("val")],
                output_dir=Path(temp),
                report_output_dir=None,
                report_matcher_mode="graph_matcher",
                device="cuda",
                report_sample_count=6,
                training_crop_size=1024,
                training_max_image_size=768,
                report_max_keypoints=512,
                report_max_matches=128,
                report_draw_matches=64,
                report_texture_keypoint_fraction=0.8,
                report_weak_texture_keypoint_fraction=0.2,
                report_keypoint_spatial_bins=8,
                report_keypoint_cell_cap=4,
                report_coverage_bins=8,
                report_min_margin=0.01,
                report_graph_inference_preset="fast",
                report_graph_width_prune_min_score=0.25,
                report_graph_early_stop_min_confidence=0.85,
                report_graph_min_accept_probability=0.7,
                report_graph_max_attention_work_fraction=0.55,
                report_graph_width_prune_keep_ratio=0.4,
                min_intensity=0.01,
                report_required_sample_glob=[],
                pose_metadata_root=[],
            )

            with mock.patch.object(train.subprocess, "run") as run:
                train.run_training_report(args, pytorch_state=Path("state.pt"))

        command = run.call_args.args[0]
        self.assertIn("--graph-inference-preset", command)
        self.assertIn("fast", command)
        self.assertIn("--graph-width-prune-min-score", command)
        self.assertIn("0.25", command)
        self.assertIn("--graph-early-stop-min-confidence", command)
        self.assertIn("0.85", command)
        self.assertIn("--graph-min-accept-probability", command)
        self.assertIn("0.7", command)
        self.assertIn("--graph-max-attention-work-fraction", command)
        self.assertIn("0.55", command)
        self.assertIn("--graph-width-prune-keep-ratio", command)
        self.assertIn("0.4", command)

    def test_parse_args_accepts_pseudo_label_options(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--pseudo-label-csv",
            "pseudo_a.csv",
            "--pseudo-label-csv",
            "pseudo_b.csv",
            "--pseudo-label-weight",
            "0.25",
            "--pseudo-label-max-points",
            "64",
            "--pseudo-keypoint-weight",
            "0.75",
            "--pseudo-keypoint-negative-weight",
            "0.02",
            "--synthetic-loss-weight",
            "0.25",
            "--pseudo-label-curriculum-max-probability",
            "0.5",
            "--pseudo-label-curriculum-warmup-steps",
            "20",
            "--abstention-weight",
            "0.3",
            "--abstention-negative-radius",
            "3.0",
            "--abstention-max-false-score",
            "0.4",
            "--abstention-topk",
            "6",
            "--abstention-candidates",
            "2048",
            "--false-match-csv",
            "false_a.csv",
            "--false-match-csv",
            "false_b.csv",
            "--false-match-weight",
            "0.4",
            "--false-match-max-points",
            "32",
            "--false-match-max-score",
            "0.25",
            "--false-match-curriculum-max-probability",
            "0.75",
            "--false-match-curriculum-warmup-steps",
            "10",
            "--train-texture-adapter",
            "--train-descriptor-fusion",
            "--freeze-descriptor-head",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertEqual(args.pseudo_label_csv, [Path("pseudo_a.csv"), Path("pseudo_b.csv")])
        self.assertEqual(args.pseudo_label_weight, 0.25)
        self.assertEqual(args.pseudo_label_max_points, 64)
        self.assertEqual(args.pseudo_keypoint_weight, 0.75)
        self.assertEqual(args.pseudo_keypoint_negative_weight, 0.02)
        self.assertEqual(args.synthetic_loss_weight, 0.25)
        self.assertEqual(args.pseudo_label_curriculum_max_probability, 0.5)
        self.assertEqual(args.pseudo_label_curriculum_warmup_steps, 20)
        self.assertEqual(args.abstention_weight, 0.3)
        self.assertEqual(args.abstention_negative_radius, 3.0)
        self.assertEqual(args.abstention_max_false_score, 0.4)
        self.assertEqual(args.abstention_topk, 6)
        self.assertEqual(args.abstention_candidates, 2048)
        self.assertEqual(args.false_match_csv, [Path("false_a.csv"), Path("false_b.csv")])
        self.assertEqual(args.false_match_weight, 0.4)
        self.assertEqual(args.false_match_max_points, 32)
        self.assertEqual(args.false_match_max_score, 0.25)
        self.assertEqual(args.false_match_curriculum_max_probability, 0.75)
        self.assertEqual(args.false_match_curriculum_warmup_steps, 10)
        self.assertTrue(args.train_texture_adapter)
        self.assertTrue(args.train_descriptor_fusion)
        self.assertTrue(args.freeze_descriptor_head)

    def test_parse_args_rejects_nonpositive_gradient_accumulation_steps(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "train",
            "--gradient-accumulation-steps",
            "0",
        ]

        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                train.parse_args()

    def test_train_step_accumulates_micro_batches_before_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        optimizer.step = mock.Mock(wraps=optimizer.step)
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        metric_rows = [
            {
                "top1_accuracy": 0.25,
                "top5_accuracy": 0.50,
                "top10_accuracy": 0.75,
                "mean_positive_rank": 3.0,
                "mean_positive_score": 0.80,
                "mean_negative_score": 0.30,
            },
            {
                "top1_accuracy": 0.75,
                "top5_accuracy": 1.00,
                "top10_accuracy": 1.00,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 0.90,
                "mean_negative_score": 0.20,
            },
        ]
        loss_scales = [2.0, 4.0]

        def fake_descriptor_loss(*_args, **_kwargs):
            scale = loss_scales.pop(0)
            metric = metric_rows[1 if scale == 4.0 else 0]
            return parameter * scale, metric

        with (
            mock.patch.object(
                train,
                "sample_curriculum_training_pairs",
                side_effect=[[Path("pair_a.pt")], [Path("pair_b.pt")]],
            ) as sample_pairs,
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [Path("pair_a.pt"), Path("pair_b.pt")],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                gradient_accumulation_steps=2,
            )

        self.assertEqual(sample_pairs.call_count, 2)
        self.assertEqual(optimizer.step.call_count, 1)
        self.assertAlmostEqual(float(parameter.detach()), 0.7, places=5)
        self.assertAlmostEqual(metrics["loss"], 3.0)
        self.assertEqual(metrics["points"], 4.0)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.5)

    def test_train_step_reports_graph_matcher_accept_metrics(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_graph.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )

        def fake_graph_loss(*_args, **kwargs):
            self.assertTrue(kwargs.get("return_components"))
            return parameter * 3.0, {
                "graph_matcher_total_loss": torch.tensor(3.0),
                "graph_matcher_ce_loss": torch.tensor(2.0),
                "graph_matcher_no_match_loss": torch.tensor(0.0),
                "graph_matcher_accept_loss": torch.tensor(0.5),
                "graph_matcher_prune_ranking_loss": torch.tensor(0.25),
                "graph_matcher_raw_preservation_loss": torch.tensor(0.0),
                "graph_matcher_hard_negative_dustbin_loss": torch.tensor(0.0),
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "graph_matcher_correspondence_loss", side_effect=fake_graph_loss),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.0,
                synthetic_loss_weight=0.0,
                graph_matcher_loss_weight=1.0,
                graph_matcher_accept_weight=0.2,
                graph_matcher_prune_ranking_weight=0.1,
            )

        self.assertAlmostEqual(metrics["graph_matcher_ce_loss"], 2.0)
        self.assertAlmostEqual(metrics["graph_matcher_accept_loss"], 0.5)
        self.assertAlmostEqual(metrics["graph_matcher_prune_ranking_loss"], 0.25)

    def test_train_step_adds_illumination_consistency_loss(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_a.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        changed_pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2) * 0.25,
            view_b=torch.ones(1, 2, 2) * 0.75,
            warp_a_to_b=pair.warp_a_to_b,
            valid_mask=pair.valid_mask,
        )
        descriptors = (
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
        )

        def fake_descriptor_loss(*_args, **_kwargs):
            return parameter * 2.0, {
                "top1_accuracy": 1.0,
                "top5_accuracy": 1.0,
                "top10_accuracy": 1.0,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.0,
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(train, "compute_student_teacher_descriptor_maps", return_value=descriptors),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_map_single",
                return_value=(descriptors[1], descriptors[3]),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss),
            mock.patch.object(train, "compute_training_descriptor_map", return_value=torch.ones(1, 1, 2, 2)),
            mock.patch.object(train, "descriptor_consistency_loss", return_value=parameter * 3.0) as consistency_loss,
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                illumination_consistency_pairs={pair_path.resolve(strict=False): changed_pair},
                illumination_consistency_weight=0.5,
                illumination_consistency_max_points=1,
            )

        self.assertEqual(consistency_loss.call_count, 2)
        self.assertAlmostEqual(float(parameter.detach()), 0.65, places=5)
        self.assertEqual(metrics["illumination_consistency_pairs"], 1.0)
        self.assertEqual(metrics["illumination_consistency_points"], 2.0)

    def test_train_step_uses_lightweight_descriptor_path_for_illumination_consistency(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_a.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        changed_pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2) * 0.25,
            view_b=torch.ones(1, 2, 2) * 0.75,
            warp_a_to_b=pair.warp_a_to_b,
            valid_mask=pair.valid_mask,
        )
        descriptors = (
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
        )

        def fake_descriptor_loss(*_args, **_kwargs):
            return parameter * 2.0, {
                "top1_accuracy": 1.0,
                "top5_accuracy": 1.0,
                "top10_accuracy": 1.0,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.0,
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(train, "compute_student_teacher_descriptor_maps", return_value=descriptors) as descriptor_maps,
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss),
            mock.patch.object(train, "compute_training_descriptor_map", return_value=torch.ones(1, 1, 2, 2)) as lightweight,
            mock.patch.object(train, "descriptor_consistency_loss", return_value=parameter * 1.0),
        ):
            train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                illumination_consistency_pairs={pair_path.resolve(strict=False): changed_pair},
                illumination_consistency_weight=0.5,
                illumination_consistency_max_points=1,
            )

        self.assertEqual(descriptor_maps.call_count, 1)
        self.assertEqual(lightweight.call_count, 2)

    def test_train_step_adds_illumination_match_pair_loss(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_a.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        changed_pair = SyntheticPair(
            view_a=pair.view_a,
            view_b=torch.ones(1, 2, 2) * 0.25,
            warp_a_to_b=pair.warp_a_to_b,
            valid_mask=pair.valid_mask,
        )
        descriptors = (
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
        )

        def fake_descriptor_loss(*_args, **_kwargs):
            fake_descriptor_loss.calls += 1
            return parameter * 2.0, {
                "top1_accuracy": 1.0,
                "top5_accuracy": 1.0,
                "top10_accuracy": 1.0,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.0,
            }

        fake_descriptor_loss.calls = 0

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(train, "compute_student_teacher_descriptor_maps", return_value=descriptors),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_map_single",
                return_value=(descriptors[1], descriptors[3]),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                illumination_match_pairs={pair_path.resolve(strict=False): changed_pair},
                illumination_match_weight=0.5,
                illumination_match_probability=1.0,
            )

        self.assertEqual(fake_descriptor_loss.calls, 2)
        self.assertAlmostEqual(float(parameter.detach()), 0.7, places=5)
        self.assertEqual(metrics["illumination_match_pairs"], 1.0)
        self.assertEqual(metrics["illumination_match_points"], 2.0)

    def test_train_step_reuses_unchanged_source_view_for_illumination_match_pair_loss(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_a.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        changed_pair = SyntheticPair(
            view_a=pair.view_a,
            view_b=torch.ones(1, 2, 2) * 0.25,
            warp_a_to_b=pair.warp_a_to_b,
            valid_mask=pair.valid_mask,
        )
        base_descriptors = (
            torch.full((1, 1, 2, 2), 1.0),
            torch.full((1, 1, 2, 2), 2.0),
            torch.full((1, 1, 2, 2), 3.0),
            torch.full((1, 1, 2, 2), 4.0),
        )
        changed_b_descriptors = (
            torch.full((1, 1, 2, 2), 5.0),
            torch.full((1, 1, 2, 2), 6.0),
        )

        def fake_descriptor_loss(*_args, **_kwargs):
            return parameter * 2.0, {
                "top1_accuracy": 1.0,
                "top5_accuracy": 1.0,
                "top10_accuracy": 1.0,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.0,
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(train, "compute_student_teacher_descriptor_maps", return_value=base_descriptors) as full_maps,
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_map_single",
                return_value=changed_b_descriptors,
                create=True,
            ) as single_map,
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss) as descriptor_loss,
        ):
            train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                illumination_match_pairs={pair_path.resolve(strict=False): changed_pair},
                illumination_match_weight=0.5,
                illumination_match_probability=1.0,
            )

        self.assertEqual(full_maps.call_count, 1)
        self.assertEqual(single_map.call_count, 1)
        self.assertTrue(torch.equal(single_map.call_args.args[1], changed_pair.view_b.unsqueeze(0)))
        self.assertIs(descriptor_loss.call_args.kwargs["teacher_descriptors_a"], base_descriptors[2])
        self.assertIs(descriptor_loss.call_args.kwargs["teacher_descriptors_b"], changed_b_descriptors[1])

    def test_train_step_adds_weighted_pseudo_label_loss(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )

        def fake_descriptor_loss(*_args, **_kwargs):
            scale = 2.0 if fake_descriptor_loss.calls == 0 else 5.0
            fake_descriptor_loss.calls += 1
            return parameter * scale, {
                "top1_accuracy": 1.0,
                "top5_accuracy": 1.0,
                "top10_accuracy": 1.0,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.0,
            }

        fake_descriptor_loss.calls = 0

        with (
            mock.patch.object(train, "sample_curriculum_training_pairs", return_value=[Path("pair_a.pt")]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(
                train,
                "pseudo_label_feature_correspondences",
                return_value=(torch.ones(3, 2), torch.ones(3, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [Path("pair_a.pt")],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                pseudo_labels={"pair_a.pt": train.PseudoLabelMatches(torch.ones(1, 2), torch.ones(1, 2))},
                pseudo_label_weight=0.5,
                pseudo_label_max_points=3,
            )

        self.assertEqual(fake_descriptor_loss.calls, 2)
        self.assertAlmostEqual(float(parameter.detach()), 0.55, places=5)
        self.assertEqual(metrics["points"], 5.0)
        self.assertEqual(metrics["pseudo_label_points"], 3.0)

    def test_train_step_can_disable_synthetic_loss_for_pseudo_only_updates(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )

        def fake_descriptor_loss(*_args, **_kwargs):
            fake_descriptor_loss.calls += 1
            return parameter * 5.0, {
                "top1_accuracy": 1.0,
                "top5_accuracy": 1.0,
                "top10_accuracy": 1.0,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.0,
            }

        fake_descriptor_loss.calls = 0

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[Path("pair_a.pt")]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(
                train,
                "pseudo_label_feature_correspondences",
                return_value=(torch.ones(3, 2), torch.ones(3, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [Path("pair_a.pt")],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                synthetic_loss_weight=0.0,
                pseudo_labels={"pair_a.pt": train.PseudoLabelMatches(torch.ones(1, 2), torch.ones(1, 2))},
                pseudo_label_weight=1.0,
                pseudo_label_max_points=3,
                pseudo_label_pair_paths=[Path("pair_a.pt")],
                pseudo_label_probability=1.0,
            )

        self.assertEqual(fake_descriptor_loss.calls, 1)
        self.assertAlmostEqual(float(parameter.detach()), 0.5, places=5)
        self.assertEqual(metrics["points"], 3.0)
        self.assertEqual(metrics["pseudo_label_points"], 3.0)

    def test_train_step_adds_weighted_pseudo_keypoint_loss(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )

        def fake_descriptor_loss(*_args, **_kwargs):
            return parameter * 5.0, {
                "top1_accuracy": 1.0,
                "top5_accuracy": 1.0,
                "top10_accuracy": 1.0,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.0,
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[Path("pair_a.pt")]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2) * parameter,
                    torch.ones(1, 1, 2, 2) * parameter,
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(
                train,
                "pseudo_label_feature_correspondences",
                return_value=(torch.ones(3, 2), torch.ones(3, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss),
            mock.patch.object(train, "heatmap_point_loss", return_value=parameter * 2.0),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [Path("pair_a.pt")],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                synthetic_loss_weight=0.0,
                pseudo_labels={"pair_a.pt": train.PseudoLabelMatches(torch.ones(1, 2), torch.ones(1, 2))},
                pseudo_label_weight=1.0,
                pseudo_keypoint_weight=0.5,
                pseudo_label_max_points=3,
                pseudo_label_pair_paths=[Path("pair_a.pt")],
                pseudo_label_probability=1.0,
            )

        self.assertAlmostEqual(float(parameter.detach()), 0.3, places=5)
        self.assertEqual(metrics["pseudo_keypoint_points"], 6.0)

    def test_train_step_allows_pseudo_keypoint_only_updates(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[Path("pair_a.pt")]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2) * parameter,
                    torch.ones(1, 1, 2, 2) * parameter,
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(
                train,
                "pseudo_label_feature_correspondences",
                return_value=(torch.ones(3, 2), torch.ones(3, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss") as descriptor_loss,
            mock.patch.object(train, "heatmap_point_loss", return_value=parameter * 2.0),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [Path("pair_a.pt")],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                synthetic_loss_weight=0.0,
                pseudo_labels={"pair_a.pt": train.PseudoLabelMatches(torch.ones(1, 2), torch.ones(1, 2))},
                pseudo_label_weight=0.0,
                pseudo_keypoint_weight=0.5,
                pseudo_label_max_points=3,
                pseudo_label_pair_paths=[Path("pair_a.pt")],
                pseudo_label_probability=1.0,
            )

        descriptor_loss.assert_not_called()
        self.assertAlmostEqual(float(parameter.detach()), 0.8, places=5)
        self.assertEqual(metrics["points"], 0.0)
        self.assertEqual(metrics["pseudo_label_points"], 0.0)
        self.assertEqual(metrics["pseudo_keypoint_points"], 6.0)

    def test_train_step_allows_synthetic_keypoint_only_updates(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[Path("pair_a.pt")]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2) * parameter,
                    torch.ones(1, 1, 2, 2) * parameter,
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss") as descriptor_loss,
            mock.patch.object(train, "heatmap_point_loss", return_value=parameter * 2.0) as heatmap_loss,
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [Path("pair_a.pt")],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                synthetic_loss_weight=0.0,
                keypoint_weight=0.5,
                keypoint_negative_weight=0.02,
            )

        descriptor_loss.assert_not_called()
        self.assertEqual(heatmap_loss.call_count, 2)
        self.assertAlmostEqual(float(parameter.detach()), 0.8, places=5)
        self.assertEqual(metrics["points"], 2.0)
        self.assertEqual(metrics["keypoint_points"], 4.0)
        self.assertAlmostEqual(metrics["keypoint_loss"], 4.0)

    def test_train_step_adds_weighted_false_match_loss(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[Path("pair_a.pt")]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(
                train,
                "false_match_feature_correspondences",
                return_value=(torch.ones(3, 2), torch.ones(3, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss") as descriptor_loss,
            mock.patch.object(train, "false_match_negative_loss", return_value=parameter * 4.0),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [Path("pair_a.pt")],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                synthetic_loss_weight=0.0,
                false_matches={"pair_a.pt": train.FalseMatchLabels(torch.ones(3, 2), torch.ones(3, 2))},
                false_match_weight=0.5,
                false_match_max_points=3,
                false_match_pair_paths=[Path("pair_a.pt")],
                false_match_probability=1.0,
            )

        descriptor_loss.assert_not_called()
        self.assertAlmostEqual(float(parameter.detach()), 0.8, places=5)
        self.assertEqual(metrics["points"], 0.0)
        self.assertEqual(metrics["false_match_points"], 3.0)
        self.assertEqual(metrics["false_match_pairs"], 1.0)

    def test_train_step_adds_inline_online_false_match_loss(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[Path("pair_a.pt")]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(
                train,
                "online_false_match_feature_correspondences",
                return_value=(torch.ones(4, 2), torch.ones(4, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss") as descriptor_loss,
            mock.patch.object(train, "false_match_negative_loss", return_value=parameter * 4.0),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [Path("pair_a.pt")],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.25,
                synthetic_loss_weight=0.0,
                online_false_match_weight=0.5,
                online_false_match_max_points=4,
                online_false_match_max_keypoints=8,
            )

        descriptor_loss.assert_not_called()
        self.assertAlmostEqual(float(parameter.detach()), 0.8, places=5)
        self.assertEqual(metrics["points"], 0.0)
        self.assertEqual(metrics["online_false_match_points"], 4.0)
        self.assertEqual(metrics["online_false_match_pairs"], 1.0)
        self.assertEqual(metrics["false_match_points"], 4.0)

    def test_train_step_passes_online_false_matches_to_graph_no_match(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_graph_false.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        false_a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        false_b = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

        def fake_graph_loss(*_args, **kwargs):
            self.assertTrue(kwargs.get("return_components"))
            self.assertTrue(torch.equal(kwargs["extra_no_match_points_a_xy"], false_a))
            self.assertTrue(torch.equal(kwargs["extra_no_match_points_b_xy"], false_b))
            return parameter * 3.0, {
                "graph_matcher_total_loss": torch.tensor(3.0),
                "graph_matcher_ce_loss": torch.tensor(0.0),
                "graph_matcher_assignment_loss": torch.tensor(1.0),
                "graph_matcher_no_match_loss": torch.tensor(2.0),
                "graph_matcher_accept_loss": torch.tensor(0.0),
                "graph_matcher_prune_ranking_loss": torch.tensor(0.0),
                "graph_matcher_stop_confidence_loss": torch.tensor(0.0),
                "graph_matcher_raw_preservation_loss": torch.tensor(0.0),
                "graph_matcher_hard_negative_dustbin_loss": torch.tensor(0.0),
                "graph_matcher_extra_no_match_points": torch.tensor(4.0),
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(
                train,
                "online_false_match_feature_correspondences",
                return_value=(false_a, false_b),
            ),
            mock.patch.object(train, "graph_matcher_correspondence_loss", side_effect=fake_graph_loss),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.0,
                synthetic_loss_weight=0.0,
                graph_matcher_loss_weight=1.0,
                graph_matcher_assignment_weight=0.25,
                graph_matcher_online_false_no_match=True,
                online_false_match_max_points=2,
                online_false_match_max_keypoints=8,
            )

        self.assertAlmostEqual(float(parameter.detach()), 0.7, places=5)
        self.assertEqual(metrics["online_false_match_points"], 2.0)
        self.assertEqual(metrics["online_false_match_pairs"], 1.0)
        self.assertEqual(metrics["graph_matcher_extra_no_match_points"], 4.0)

    def test_train_step_passes_static_false_matches_to_graph_final_false_loss(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_graph_static_false.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        false_a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        false_b = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

        def fake_graph_loss(*_args, **kwargs):
            self.assertTrue(kwargs.get("return_components"))
            self.assertTrue(torch.equal(kwargs["extra_false_match_points_a_xy"], false_a))
            self.assertTrue(torch.equal(kwargs["extra_false_match_points_b_xy"], false_b))
            self.assertIsNone(kwargs["extra_no_match_points_a_xy"])
            self.assertIsNone(kwargs["extra_no_match_points_b_xy"])
            return parameter * 2.0, {
                "graph_matcher_total_loss": torch.tensor(2.0),
                "graph_matcher_ce_loss": torch.tensor(0.0),
                "graph_matcher_final_false_match_loss": torch.tensor(0.5),
                "graph_matcher_mined_false_match_loss": torch.tensor(0.5),
                "graph_matcher_mined_false_match_edges": torch.tensor(2.0),
                "graph_matcher_extra_false_match_pairs": torch.tensor(2.0),
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(
                train,
                "false_match_feature_correspondences",
                return_value=(false_a, false_b),
            ),
            mock.patch.object(train, "graph_matcher_correspondence_loss", side_effect=fake_graph_loss),
            mock.patch.object(train, "false_match_negative_loss", return_value=parameter * 0.0),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.0,
                synthetic_loss_weight=0.0,
                false_matches={str(pair_path): train.FalseMatchLabels(false_a, false_b)},
                false_match_weight=0.0,
                false_match_max_points=2,
                false_match_pair_paths=[pair_path],
                false_match_probability=1.0,
                graph_matcher_loss_weight=1.0,
                graph_matcher_final_false_match_weight=0.25,
            )

        self.assertAlmostEqual(float(parameter.detach()), 0.8, places=5)
        self.assertEqual(metrics["graph_matcher_mined_false_match_edges"], 2.0)
        self.assertEqual(metrics["graph_matcher_extra_false_match_pairs"], 2.0)

    def test_train_step_passes_static_false_matches_with_independent_mined_false_weight(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_graph_static_false_independent.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        false_a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        false_b = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

        def fake_graph_loss(*_args, **kwargs):
            self.assertTrue(kwargs.get("return_components"))
            self.assertEqual(kwargs["final_false_match_weight"], 0.0)
            self.assertAlmostEqual(kwargs["mined_false_match_weight"], 0.25)
            self.assertAlmostEqual(kwargs["mined_false_match_reference_margin"], 0.5)
            self.assertTrue(torch.equal(kwargs["extra_false_match_points_a_xy"], false_a))
            self.assertTrue(torch.equal(kwargs["extra_false_match_points_b_xy"], false_b))
            return parameter * 2.0, {
                "graph_matcher_total_loss": torch.tensor(2.0),
                "graph_matcher_ce_loss": torch.tensor(0.0),
                "graph_matcher_final_false_match_loss": torch.tensor(0.0),
                "graph_matcher_mined_false_match_loss": torch.tensor(0.5),
                "graph_matcher_mined_false_match_edges": torch.tensor(2.0),
                "graph_matcher_extra_false_match_pairs": torch.tensor(2.0),
            }

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(
                train,
                "false_match_feature_correspondences",
                return_value=(false_a, false_b),
            ),
            mock.patch.object(train, "graph_matcher_correspondence_loss", side_effect=fake_graph_loss),
            mock.patch.object(train, "false_match_negative_loss", return_value=parameter * 0.0),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.0,
                synthetic_loss_weight=0.0,
                false_matches={str(pair_path): train.FalseMatchLabels(false_a, false_b)},
                false_match_weight=0.0,
                false_match_max_points=2,
                false_match_pair_paths=[pair_path],
                false_match_probability=1.0,
                graph_matcher_loss_weight=1.0,
                graph_matcher_final_false_match_weight=0.0,
                graph_matcher_mined_false_match_weight=0.25,
                graph_matcher_mined_false_match_reference_margin=0.5,
            )

        self.assertAlmostEqual(float(parameter.detach()), 0.8, places=5)
        self.assertEqual(metrics["graph_matcher_mined_false_match_edges"], 2.0)
        self.assertEqual(metrics["graph_matcher_extra_false_match_pairs"], 2.0)

    def test_train_step_passes_pseudo_and_false_pools_separately_to_sampler(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        pseudo_pair = Path("pseudo_pair.pt")
        false_pair = Path("false_pair.pt")

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pseudo_pair, false_pair]) as sampler,
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(train, "sample_feature_correspondences", return_value=(torch.zeros(2, 2), torch.zeros(2, 2))),
            mock.patch.object(train, "pseudo_label_feature_correspondences", return_value=(torch.ones(1, 2), torch.ones(1, 2))),
            mock.patch.object(train, "false_match_feature_correspondences", return_value=(torch.ones(1, 2), torch.ones(1, 2))),
            mock.patch.object(
                train,
                "descriptor_map_pair_loss",
                return_value=(
                    parameter * 1.0,
                    {
                        "top1_accuracy": 1.0,
                        "top5_accuracy": 1.0,
                        "top10_accuracy": 1.0,
                        "mean_positive_rank": 1.0,
                        "mean_positive_score": 1.0,
                        "mean_negative_score": 0.0,
                    },
                ),
            ),
            mock.patch.object(train, "false_match_negative_loss", return_value=parameter * 1.0),
        ):
            train.train_step(
                object(),
                optimizer,
                [pseudo_pair, false_pair],
                device=torch.device("cpu"),
                batch_pairs=2,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.0,
                synthetic_loss_weight=0.0,
                pseudo_labels={"pseudo_pair.pt": train.PseudoLabelMatches(torch.ones(1, 2), torch.ones(1, 2))},
                pseudo_label_weight=1.0,
                pseudo_label_pair_paths=[pseudo_pair],
                pseudo_label_probability=1.0,
                false_matches={"false_pair.pt": train.FalseMatchLabels(torch.ones(1, 2), torch.ones(1, 2))},
                false_match_weight=1.0,
                false_match_pair_paths=[false_pair],
                false_match_probability=1.0,
            )

        self.assertEqual(sampler.call_args.kwargs["pseudo_label_pair_paths"], [pseudo_pair])
        self.assertEqual(sampler.call_args.kwargs["false_match_pair_paths"], [false_pair])

    def test_make_torch_generator_is_reproducible(self):
        first = train.make_torch_generator(torch.device("cpu"), seed=17)
        second = train.make_torch_generator(torch.device("cpu"), seed=17)

        self.assertTrue(torch.equal(torch.randperm(8, generator=first), torch.randperm(8, generator=second)))

    def test_gradient_l2_norm_sums_available_gradients(self):
        first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        second = torch.nn.Parameter(torch.tensor([3.0]))
        first.grad = torch.tensor([3.0, 4.0])
        second.grad = None

        self.assertAlmostEqual(train.gradient_l2_norm([first, second]), 5.0)

    def test_require_finite_scalar_rejects_nan_loss(self):
        with self.assertRaises(FloatingPointError):
            train.require_finite_scalar(torch.tensor(float("nan")), name="loss")

    def test_clip_and_measure_gradients_limits_large_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        parameter.grad = torch.tensor([30.0, 40.0])

        norm = train.clip_and_measure_gradients([parameter], max_grad_norm=5.0)

        self.assertLessEqual(norm, 5.0001)

    def test_clip_and_measure_gradients_rejects_nonfinite_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        parameter.grad = torch.tensor([float("nan"), 1.0])

        with self.assertRaises(FloatingPointError):
            train.clip_and_measure_gradients([parameter], max_grad_norm=5.0)

    def test_skipped_step_metrics_marks_nonfinite_step_without_optimizer_update(self):
        rows = [
            {
                "top1_accuracy": 0.25,
                "top5_accuracy": 0.5,
                "top10_accuracy": 0.75,
                "mean_positive_rank": 4.0,
                "mean_positive_score": 0.8,
                "mean_negative_score": 0.3,
            }
        ]

        metrics = train.skipped_step_metrics(torch.tensor(7.0), rows, sampled_count=16)

        self.assertEqual(metrics["skipped"], 1.0)
        self.assertEqual(metrics["points"], 16.0)
        self.assertEqual(metrics["grad_l2"], 0.0)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.25)

    def test_train_step_uses_grad_scaler_for_amp_before_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_amp.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        calls: list[str] = []

        class RecordingScaler:
            def scale(self, loss):
                calls.append("scale")
                return loss

            def unscale_(self, optimizer):
                calls.append("unscale")

            def step(self, optimizer):
                calls.append("step")
                optimizer.step()

            def update(self):
                calls.append("update")

            def get_scale(self):
                return 128.0

        def fake_descriptor_loss(*_args, **_kwargs):
            return parameter * 2.0, {
                "top1_accuracy": 1.0,
                "top5_accuracy": 1.0,
                "top10_accuracy": 1.0,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.0,
            }

        original_clip = train.clip_and_measure_gradients

        def record_clip(parameters, *, max_grad_norm=0.0):
            calls.append("clip")
            return original_clip(parameters, max_grad_norm=max_grad_norm)

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss),
            mock.patch.object(train, "clip_and_measure_gradients", side_effect=record_clip),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.0,
                amp_enabled=True,
                amp_dtype=torch.bfloat16,
                grad_scaler=RecordingScaler(),
                activation_checkpointing=True,
            )

        self.assertEqual(calls, ["scale", "unscale", "clip", "step", "update"])
        self.assertAlmostEqual(float(parameter.detach()), 0.8, places=5)
        self.assertEqual(metrics["amp_enabled"], 1.0)
        self.assertEqual(metrics["amp_scale"], 128.0)
        self.assertEqual(metrics["activation_checkpointing"], 1.0)

    def test_train_step_updates_grad_scaler_after_nonfinite_gradient_skip(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        pair_path = Path("pair_amp_skip.pt")
        pair = SyntheticPair(
            view_a=torch.ones(1, 2, 2),
            view_b=torch.ones(1, 2, 2),
            warp_a_to_b=torch.zeros(2, 2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        calls: list[str] = []

        class RecordingScaler:
            def scale(self, loss):
                calls.append("scale")
                return loss

            def unscale_(self, optimizer):
                calls.append("unscale")

            def step(self, optimizer):
                calls.append("step")
                optimizer.step()

            def update(self):
                calls.append("update")

            def get_scale(self):
                return 64.0

        def fake_descriptor_loss(*_args, **_kwargs):
            return parameter * 2.0, {
                "top1_accuracy": 1.0,
                "top5_accuracy": 1.0,
                "top10_accuracy": 1.0,
                "mean_positive_rank": 1.0,
                "mean_positive_score": 1.0,
                "mean_negative_score": 0.0,
            }

        def raise_nonfinite_gradient(_parameters, *, max_grad_norm=0.0):
            calls.append("clip")
            raise FloatingPointError("nonfinite gradients")

        with (
            mock.patch.object(train, "sample_training_pairs_with_pseudo_labels", return_value=[pair_path]),
            mock.patch.object(train, "load_libtorch_pair_archive", return_value=pair),
            mock.patch.object(
                train,
                "compute_student_teacher_descriptor_maps",
                return_value=(
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 2, 2),
                ),
            ),
            mock.patch.object(
                train,
                "sample_feature_correspondences",
                return_value=(torch.zeros(2, 2), torch.zeros(2, 2)),
            ),
            mock.patch.object(train, "descriptor_map_pair_loss", side_effect=fake_descriptor_loss),
            mock.patch.object(train, "clip_and_measure_gradients", side_effect=raise_nonfinite_gradient),
        ):
            metrics = train.train_step(
                object(),
                optimizer,
                [pair_path],
                device=torch.device("cpu"),
                batch_pairs=1,
                samples_per_pair=2,
                min_intensity=0.01,
                generator=torch.Generator().manual_seed(7),
                temperature=0.07,
                teacher_weight=0.0,
                skip_nonfinite_steps=True,
                amp_enabled=True,
                amp_dtype=torch.bfloat16,
                grad_scaler=RecordingScaler(),
            )

        self.assertEqual(calls, ["scale", "unscale", "clip", "update"])
        self.assertEqual(metrics["skipped"], 1.0)
        self.assertAlmostEqual(float(parameter.detach()), 1.0, places=5)

    def test_load_training_model_prefers_pytorch_state_when_provided(self):
        class Args:
            checkpoint = Path("missing_libtorch.pt")
            init_pytorch_state = Path("/tmp/pfm_training_state_preferred.pt")
            init_random = False
            device = "cpu"

        model = train.pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=2,
            descriptor_dim=4,
            graph_hidden_dim=8,
            graph_attention_layers=1,
        )
        torch.save({"config": model.config.__dict__, "model": model.state_dict()}, Args.init_pytorch_state)

        loaded, config = train.load_training_model(Args)

        self.assertEqual(config.base_channels, 2)
        self.assertEqual(loaded.config.descriptor_dim, 4)

    def test_parse_args_allows_random_initialization_without_checkpoint(self):
        argv = [
            "pfm_pytorch_training.py",
            "--init-random",
            "--cache-dir",
            "img/Rotate_1024",
            "--steps",
            "1",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = train.parse_args()

        self.assertTrue(args.init_random)
        self.assertIsNone(args.checkpoint)

    def test_load_training_model_can_initialize_default_random_model(self):
        class Args:
            checkpoint = None
            init_pytorch_state = None
            init_random = True
            device = "cpu"

        loaded, config = train.load_training_model(Args)

        self.assertEqual(config.base_channels, 64)
        self.assertEqual(config.descriptor_dim, 256)
        self.assertEqual(config.graph_hidden_dim, 512)
        self.assertEqual(config.graph_keypoint_meta_dim, 16)
        self.assertEqual(loaded.config.graph_attention_layers, 8)

    def test_load_graph_matcher_teacher_guard_model_freezes_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                graph_matcher_teacher_guard_state=Path(tmp) / "pfm_teacher_guard_state.pt",
                device="cpu",
                matcher_reliability_pair_bias="off",
                matcher_reliability_dustbin_bias="off",
                matcher_final_accept_score_mode="none",
                matcher_geometry_bias_scale=0.25,
                matcher_accept_assignment_mode="add",
                matcher_final_accept_score_alpha=0.05,
                matcher_geometry_bias_clamp=1.5,
                matcher_attention_residual_gate_init=None,
                matcher_attention_residual_gate_start_layer=1,
                matcher_candidate_topk=96,
            )
            model = train.pfm_model.PlanetaryFeatureMatcher(
                input_channels=1,
                base_channels=2,
                descriptor_dim=4,
                graph_hidden_dim=8,
                graph_attention_layers=1,
            )
            torch.save(
                {"config": model.config.__dict__, "model": model.state_dict()},
                args.graph_matcher_teacher_guard_state,
            )

            loaded = train.load_graph_matcher_teacher_guard_model(args)

        self.assertFalse(loaded.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
        self.assertEqual(loaded.config.matcher_candidate_topk, 96)
        self.assertAlmostEqual(loaded.config.matcher_geometry_bias_scale, 0.25)


if __name__ == "__main__":
    unittest.main()
