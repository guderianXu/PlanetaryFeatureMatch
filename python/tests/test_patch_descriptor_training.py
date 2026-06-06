import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_pair_cache import make_compact_pair_payload, save_shared_image

import patch_descriptor_training as pdt


class PatchDescriptorTrainingTest(unittest.TestCase):
    def test_extract_patches_uses_xy_pixel_coordinates(self):
        image = torch.arange(25, dtype=torch.float32).reshape(1, 5, 5)
        points = torch.tensor([[2.0, 2.0], [1.0, 3.0]], dtype=torch.float32)

        patches = pdt.extract_patches(image, points, patch_size=3)

        self.assertEqual(tuple(patches.shape), (2, 1, 3, 3))
        self.assertTrue(torch.equal(patches[0, 0], image[0, 1:4, 1:4]))
        self.assertTrue(torch.equal(patches[1, 0], image[0, 2:5, 0:3]))

    def test_sample_correspondences_rejects_zero_intensity_and_out_of_bounds_warp(self):
        view_a = torch.ones(1, 6, 6)
        view_b = torch.ones(1, 6, 6)
        view_a[:, 2, 2] = 0.0
        view_b[:, 4, 4] = 0.0
        warp = torch.zeros(6, 6, 2)
        yy, xx = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
        warp[..., 0] = xx
        warp[..., 1] = yy
        warp[1, 1] = torch.tensor([99.0, 99.0])
        warp[3, 3] = torch.tensor([4.0, 4.0])
        valid = torch.ones(6, 6, dtype=torch.bool)
        pair = pdt.SyntheticPair(view_a=view_a, view_b=view_b, warp_a_to_b=warp, valid_mask=valid)

        points_a, points_b = pdt.sample_valid_correspondences(
            pair,
            count=64,
            patch_size=3,
            min_intensity=0.05,
            generator=torch.Generator().manual_seed(7),
        )

        point_set_a = {(int(x), int(y)) for x, y in points_a.tolist()}
        point_set_b = {(round(float(x)), round(float(y))) for x, y in points_b.tolist()}
        self.assertNotIn((1, 1), point_set_a)
        self.assertNotIn((2, 2), point_set_a)
        self.assertNotIn((4, 4), point_set_b)
        self.assertGreater(points_a.shape[0], 0)

    def test_descriptor_loss_is_lower_for_matching_pairs_than_shuffled_pairs(self):
        desc_a = torch.eye(4, dtype=torch.float32)
        desc_b = desc_a.clone()

        matched = pdt.paired_descriptor_loss(desc_a, desc_b, temperature=0.05)
        shuffled = pdt.paired_descriptor_loss(desc_a, desc_b.roll(shifts=1, dims=0), temperature=0.05)

        self.assertLess(float(matched), float(shuffled))

    def test_descriptor_loss_keeps_zero_descriptor_gradients_finite(self):
        desc_a = torch.zeros(4, 8, requires_grad=True)
        desc_b = torch.eye(4, 8)

        loss = pdt.paired_descriptor_loss(desc_a, desc_b, temperature=0.07)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(desc_a.grad).all())
        self.assertLessEqual(float(desc_a.grad.abs().max()), 10000.0)

    def test_descriptor_diversity_loss_penalizes_collapsed_descriptors(self):
        collapsed = torch.ones(4, 8)
        separated = torch.eye(4, 8)

        collapsed_loss = pdt.descriptor_diversity_loss(collapsed, margin=0.2)
        separated_loss = pdt.descriptor_diversity_loss(separated, margin=0.2)

        self.assertGreater(float(collapsed_loss), float(separated_loss))

    def test_descriptor_metrics_report_top1_accuracy_and_negative_score(self):
        desc_a = torch.eye(3, dtype=torch.float32)
        desc_b = desc_a.clone()

        metrics = pdt.paired_descriptor_metrics(desc_a, desc_b)

        self.assertAlmostEqual(metrics["top1_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["top5_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["top10_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_positive_rank"], 1.0)
        self.assertLess(metrics["mean_negative_score"], metrics["mean_positive_score"])

    def test_normalize_patches_is_per_patch_and_handles_constants(self):
        patches = torch.tensor(
            [
                [[
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]],
                [[
                    [5.0, 5.0],
                    [5.0, 5.0],
                ]],
            ]
        )

        normalized = pdt.normalize_patches(patches)

        self.assertAlmostEqual(float(normalized[0].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[0].std(unbiased=False)), 1.0, places=5)
        self.assertTrue(torch.equal(normalized[1], torch.zeros_like(normalized[1])))

    def test_rotation_invariant_descriptor_shape_and_quarter_turn_stability(self):
        model = pdt.RotationInvariantPatchDescriptor(descriptor_dim=16, base_channels=8)
        patch = torch.randn(2, 1, 17, 17)

        desc = model(patch)
        rotated_desc = model(torch.rot90(patch, 1, dims=(-2, -1)))

        self.assertEqual(tuple(desc.shape), (2, 16))
        self.assertTrue(torch.allclose(desc.norm(dim=1), torch.ones(2), atol=1.0e-5))
        self.assertLess(float((desc - rotated_desc).detach().abs().max()), 1.0e-5)

    def test_load_libtorch_pair_archive_reads_expected_tensors_when_available(self):
        pair_path = Path(
            "img/Rotate/source_000077_20260514T065316672_NAS_PAN_L2b/pair_000077.pt"
        )
        if not pair_path.exists():
            self.skipTest("local synthetic pair cache is unavailable")

        pair = pdt.load_libtorch_pair_archive(pair_path)

        self.assertEqual(pair.view_a.dim(), 3)
        self.assertEqual(pair.view_b.dim(), 3)
        self.assertEqual(pair.warp_a_to_b.shape[-1], 2)
        self.assertEqual(pair.valid_mask.dtype, torch.bool)

    def test_load_libtorch_pair_archive_reads_compact_pair_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_path = root / "cache" / "train" / "source_000001" / "pair_000001.pt"
            image_store = root / "image_store"
            pair_path.parent.mkdir(parents=True)
            view_a = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 16.0
            view_b = torch.flip(view_a, dims=[2])
            warp = torch.zeros(4, 4, 2, dtype=torch.float32)
            valid = torch.ones(4, 4, dtype=torch.bool)
            image_a_path = save_shared_image(view_a, image_store)
            image_b_path = save_shared_image(view_b, image_store)
            torch.save(
                make_compact_pair_payload(
                    pair_path=pair_path,
                    image_a_path=image_a_path,
                    image_b_path=image_b_path,
                    warp_a_to_b=warp,
                    valid_mask=valid,
                ),
                pair_path,
            )

            pair = pdt.load_libtorch_pair_archive(pair_path)

        self.assertTrue(torch.equal(pair.view_a, view_a))
        self.assertTrue(torch.equal(pair.view_b, view_b))
        self.assertTrue(torch.equal(pair.warp_a_to_b, warp))
        self.assertTrue(torch.equal(pair.valid_mask, valid))

    def test_is_self_pair_archive_matches_source_index_and_pair_index(self):
        self.assertTrue(
            pdt.is_self_pair_archive(
                Path("cache/source_000088_20260514T070226673_NAS_PAN_L2b/pair_000088.pt")
            )
        )
        self.assertTrue(pdt.is_self_pair_archive(Path("cache/source_000001_10/pair_000001.pt")))
        self.assertFalse(pdt.is_self_pair_archive(Path("cache/source_000001_10/pair_000694.pt")))
        self.assertFalse(pdt.is_self_pair_archive(Path("cache/source_without_index/pair_000001.pt")))

    def test_discover_pair_archives_can_exclude_self_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_000001_10"
            source.mkdir()
            self_pair = source / "pair_000001.pt"
            other_pair = source / "pair_000694.pt"
            self_pair.touch()
            other_pair.touch()

            discovered = pdt.discover_pair_archives([root], exclude_self_pairs=True)

        self.assertEqual(discovered, [other_pair])


if __name__ == "__main__":
    unittest.main()
