import sys
import unittest
import random
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pfm_pytorch_training as train
from patch_descriptor_training import SyntheticPair


class PFMPyTorchTrainingTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
