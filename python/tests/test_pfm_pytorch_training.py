import sys
import unittest
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

    def test_paired_cyclic_similarity_accepts_quarter_channel_shift(self):
        desc_a = torch.zeros(1, 8)
        desc_b = torch.zeros(1, 8)
        desc_a[0, 1] = 1.0
        desc_b[0, 3] = 1.0

        similarity = train.paired_cyclic_similarity(desc_a, desc_b)

        self.assertAlmostEqual(float(similarity[0]), 1.0, places=6)

    def test_false_match_negative_loss_penalizes_cyclic_wrong_pair(self):
        descriptor_a = torch.zeros(1, 8, 1, 1)
        descriptor_b = torch.zeros(1, 8, 1, 1)
        descriptor_a[0, 1, 0, 0] = 1.0
        descriptor_b[0, 3, 0, 0] = 1.0
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
                "points": 8.0,
            },
            {
                "graph_matcher_ce_loss": 4.0,
                "graph_matcher_accept_loss": 0.8,
                "graph_matcher_prune_ranking_loss": 0.6,
                "graph_matcher_no_match_loss": 0.3,
                "points": 24.0,
            },
        ]

        metrics = train.aggregate_graph_matcher_loss_metrics(rows)

        self.assertAlmostEqual(metrics["graph_matcher_ce_loss"], 3.5)
        self.assertAlmostEqual(metrics["graph_matcher_accept_loss"], 0.7)
        self.assertAlmostEqual(metrics["graph_matcher_prune_ranking_loss"], 0.5)
        self.assertAlmostEqual(metrics["graph_matcher_no_match_loss"], 0.25)

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

    def test_graph_matcher_correspondence_loss_backpropagates_to_matcher(self):
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=1)
        descriptors_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 4, 4))
        descriptors_b = descriptors_a.clone()
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)

        loss = train.graph_matcher_correspondence_loss(model, descriptors_a, descriptors_b, points, points)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.graph_matcher.descriptor_projection.weight.grad)

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
            "--graph-matcher-accept-weight",
            "0.2",
            "--graph-matcher-assignment-weight",
            "0.35",
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
        self.assertAlmostEqual(args.graph_matcher_accept_weight, 0.2)
        self.assertAlmostEqual(args.graph_matcher_assignment_weight, 0.35)
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
        self.assertEqual(args.graph_matcher_semi_dense_no_match_points, 24)
        self.assertAlmostEqual(args.graph_matcher_semi_dense_min_score, 0.02)
        self.assertTrue(args.graph_matcher_online_false_no_match)
        self.assertAlmostEqual(args.training_weak_texture_fraction, 0.25)
        self.assertEqual(args.training_spatial_bins, 8)
        self.assertEqual(args.hard_pair_glob, ["*pair_004541*.pt"])

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
        self.assertEqual(args.graph_matcher_no_match_points, 64)
        self.assertAlmostEqual(args.graph_matcher_no_match_weight, 0.15)
        self.assertAlmostEqual(args.graph_matcher_assignment_weight, 0.25)
        self.assertGreater(args.graph_matcher_accept_weight, 0.0)
        self.assertGreater(args.graph_matcher_prune_ranking_weight, 0.0)
        self.assertAlmostEqual(args.graph_matcher_hard_negative_dustbin_weight, 0.05)
        self.assertAlmostEqual(args.false_match_weight, 0.05)
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


if __name__ == "__main__":
    unittest.main()
