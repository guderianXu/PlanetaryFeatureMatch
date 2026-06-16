import sys
import unittest
from unittest import mock
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pfm_model
import pfm_model_descriptors


class PFMModelTest(unittest.TestCase):
    def test_descriptor_geometry_helpers_live_in_descriptor_module(self):
        self.assertIs(pfm_model.geometry_aware_descriptor_pool, pfm_model_descriptors.geometry_aware_descriptor_pool)
        self.assertIs(pfm_model.make_xy_grid, pfm_model_descriptors.make_xy_grid)

    def test_backbone_matches_libtorch_feature_pyramid_shapes(self):
        model = pfm_model.Backbone(input_channels=1, base_channels=4)
        features = model(torch.randn(2, 1, 64, 80))

        self.assertEqual([tuple(feature.shape) for feature in features], [
            (2, 4, 32, 40),
            (2, 8, 16, 20),
            (2, 16, 8, 10),
            (2, 32, 4, 5),
        ])

    def test_activation_checkpointing_preserves_python_forward_backward(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        image = torch.randn(1, 1, 32, 40)

        with mock.patch.object(
            pfm_model,
            "activation_checkpoint",
            wraps=pfm_model.activation_checkpoint,
        ) as checkpoint:
            descriptors = model.learned_descriptor_map_single(image, activation_checkpointing=True)
            loss = descriptors.square().mean()
            loss.backward()

        self.assertGreater(checkpoint.call_count, 0)
        self.assertIsNotNone(model.sparse_head.descriptor_skip.weight.grad)

    def test_sparse_head_outputs_libtorch_maps_and_normalized_descriptors(self):
        head = pfm_model.SparseHead(input_channels=8, descriptor_dim=16)
        output = head(torch.randn(2, 8, 16, 20))

        self.assertEqual(tuple(output.heatmap.shape), (2, 1, 16, 20))
        self.assertEqual(tuple(output.descriptors.shape), (2, 16, 16, 20))
        self.assertEqual(tuple(output.scale.shape), (2, 1, 16, 20))
        self.assertEqual(tuple(output.orientation.shape), (2, 2, 16, 20))
        self.assertEqual(tuple(output.affine.shape), (2, 4, 16, 20))
        self.assertEqual(tuple(output.keypoint_offsets.shape), (2, 2, 16, 20))
        self.assertLessEqual(float(output.keypoint_offsets.detach().abs().max()), 0.5)
        self.assertTrue(torch.allclose(output.descriptors.norm(dim=1), torch.ones(2, 16, 20), atol=1.0e-5))

    def test_sparse_head_outputs_reliability_maps(self):
        head = pfm_model.SparseHead(input_channels=8, descriptor_dim=16)
        output = head(torch.randn(2, 8, 16, 20))

        self.assertEqual(tuple(output.matchability.shape), tuple(output.heatmap.shape))
        self.assertEqual(tuple(output.descriptor_uncertainty.shape), tuple(output.heatmap.shape))
        self.assertEqual(tuple(output.no_match_prior.shape), tuple(output.heatmap.shape))
        for tensor in (output.matchability, output.descriptor_uncertainty, output.no_match_prior):
            self.assertTrue(bool(torch.all(tensor >= 0.0)))
            self.assertTrue(bool(torch.all(tensor <= 1.0)))

    def test_sparse_head_accepts_separate_keypoint_and_descriptor_features(self):
        head = pfm_model.SparseHead(input_channels=8, descriptor_dim=16)
        output = head(torch.randn(2, 8, 16, 20), torch.randn(2, 8, 16, 20))

        self.assertEqual(tuple(output.heatmap.shape), (2, 1, 16, 20))
        self.assertEqual(tuple(output.descriptors.shape), (2, 16, 16, 20))

    def test_sparse_head_no_longer_uses_c4_rotated_branches(self):
        head = pfm_model.SparseHead(input_channels=8, descriptor_dim=16)

        output = head(torch.randn(1, 8, 16, 20), torch.randn(1, 8, 16, 20))

        self.assertEqual(tuple(output.descriptors.shape), (1, 16, 16, 20))
        self.assertFalse(hasattr(pfm_model, "_rotate_feature_map"))
        self.assertFalse(hasattr(pfm_model, "_align_descriptor_orientation_channels"))
        self.assertFalse(hasattr(head, "descriptor_branch_quality"))
        self.assertFalse(hasattr(head, "descriptor_rotation_fusion"))

    def test_canonical_descriptor_pool_changes_with_orientation(self):
        descriptors = pfm_model.normalize_channels_stable(torch.randn(1, 8, 9, 9))
        scale = torch.ones(1, 1, 9, 9)
        affine = torch.tensor([1.0, 0.0, 0.0, 1.0]).view(1, 4, 1, 1).expand(1, 4, 9, 9)
        orientation_x = torch.zeros(1, 2, 9, 9)
        orientation_x[:, 0] = 1.0
        orientation_y = torch.zeros(1, 2, 9, 9)
        orientation_y[:, 1] = 1.0

        pooled_x = pfm_model.geometry_aware_descriptor_pool(descriptors, orientation_x, scale, affine)
        pooled_y = pfm_model.geometry_aware_descriptor_pool(descriptors, orientation_y, scale, affine)

        self.assertEqual(tuple(pooled_x.shape), tuple(descriptors.shape))
        self.assertTrue(torch.allclose(pooled_x.norm(dim=1), torch.ones(1, 9, 9), atol=1.0e-5))
        self.assertFalse(torch.allclose(pooled_x, pooled_y, atol=1.0e-4))

    def test_dual_fpn_lite_returns_separate_p2_features(self):
        backbone = pfm_model.Backbone(input_channels=1, base_channels=4)
        fpn = pfm_model.DualFPNLite(base_channels=4)
        features = backbone(torch.randn(2, 1, 64, 80))

        p2_keypoint, p2_descriptor = fpn(features)

        self.assertEqual(tuple(p2_keypoint.shape), (2, 8, 16, 20))
        self.assertEqual(tuple(p2_descriptor.shape), (2, 8, 16, 20))

    def test_dual_fpn_lite_stage1_skip_can_affect_keypoint_feature(self):
        fpn = pfm_model.DualFPNLite(base_channels=2)
        with torch.no_grad():
            fpn.keypoint_from_stage1.weight.fill_(1.0)
            if fpn.keypoint_from_stage1.bias is not None:
                fpn.keypoint_from_stage1.bias.zero_()
        stage1 = torch.ones(1, 2, 8, 8)
        stage2 = torch.zeros(1, 4, 4, 4)
        stage3 = torch.zeros(1, 8, 2, 2)
        stage4 = torch.zeros(1, 16, 1, 1)

        p2_keypoint, p2_descriptor = fpn([stage1, stage2, stage3, stage4])

        self.assertGreater(float(p2_keypoint.detach().abs().mean()), 0.5)
        self.assertLess(float(p2_descriptor.detach().abs().max()), 1.0e-6)

    def test_soft_quality_score_modulates_without_erasing_heatmap(self):
        heatmap = torch.tensor([[[[0.8, 0.4]]]])
        quality = torch.tensor([[[[0.0, 1.0]]]])

        soft = pfm_model.apply_quality_score_mode(heatmap, quality, mode="soft")
        legacy = pfm_model.apply_quality_score_mode(heatmap, quality, mode="multiply")
        raw = pfm_model.apply_quality_score_mode(heatmap, quality, mode="raw")

        self.assertTrue(torch.allclose(soft, torch.tensor([[[[0.4, 0.4]]]])))
        self.assertTrue(torch.allclose(legacy, torch.tensor([[[[0.0, 0.4]]]])))
        self.assertTrue(torch.allclose(raw, heatmap))

    def test_sparse_head_plain_geometry_mode_skips_canonical_pooling(self):
        head = pfm_model.SparseHead(input_channels=8, descriptor_dim=16, descriptor_geometry_mode="plain")

        with mock.patch.object(pfm_model, "geometry_aware_descriptor_pool", side_effect=AssertionError):
            output = head(torch.randn(1, 8, 8, 9))

        self.assertEqual(tuple(output.descriptors.shape), (1, 16, 8, 9))
        self.assertTrue(torch.allclose(output.descriptors.norm(dim=1), torch.ones(1, 8, 9), atol=1.0e-5))

    def test_sparse_head_orientation_scale_mode_uses_identity_affine_for_pooling(self):
        captured = {}

        def fake_pool(descriptors, orientation, scale, affine):
            captured["affine"] = affine.detach().clone()
            return descriptors

        head = pfm_model.SparseHead(
            input_channels=8,
            descriptor_dim=16,
            descriptor_geometry_mode="orientation_scale",
        )

        with mock.patch.object(pfm_model, "geometry_aware_descriptor_pool", side_effect=fake_pool):
            head(torch.randn(1, 8, 8, 9))

        affine = captured["affine"]
        self.assertTrue(torch.allclose(affine[:, 0], torch.ones_like(affine[:, 0])))
        self.assertTrue(torch.allclose(affine[:, 1], torch.zeros_like(affine[:, 1])))
        self.assertTrue(torch.allclose(affine[:, 2], torch.zeros_like(affine[:, 2])))
        self.assertTrue(torch.allclose(affine[:, 3], torch.ones_like(affine[:, 3])))

    def test_sparse_head_can_blend_original_and_pooled_descriptors(self):
        captured = {}

        def fake_pool(descriptors, orientation, scale, affine):
            del orientation, scale, affine
            pooled = torch.zeros_like(descriptors)
            pooled[:, 0] = 1.0
            captured["original"] = descriptors.detach().clone()
            captured["pooled"] = pooled.detach().clone()
            return pooled

        head = pfm_model.SparseHead(
            input_channels=8,
            descriptor_dim=16,
            descriptor_geometry_mode="full",
            descriptor_geometry_blend_weight=0.25,
        )

        with mock.patch.object(pfm_model, "geometry_aware_descriptor_pool", side_effect=fake_pool):
            output = head(torch.randn(1, 8, 8, 9))

        expected = pfm_model.normalize_channels_stable(
            0.75 * captured["original"] + 0.25 * captured["pooled"],
        )
        self.assertTrue(torch.allclose(output.descriptors, expected, atol=1.0e-6))

    def test_sparse_head_clamps_scale_log_with_configured_bounds(self):
        captured = {}

        def fake_pool(descriptors, orientation, scale, affine):
            del orientation, affine
            captured["scale"] = scale.detach().clone()
            return descriptors

        head = pfm_model.SparseHead(
            input_channels=8,
            descriptor_dim=16,
            descriptor_geometry_mode="full",
            descriptor_scale_log_clamp_min=-0.7,
            descriptor_scale_log_clamp_max=0.7,
        )
        with torch.no_grad():
            head.scale.weight.zero_()
            head.scale.bias.fill_(10.0)

        with mock.patch.object(pfm_model, "geometry_aware_descriptor_pool", side_effect=fake_pool):
            head(torch.randn(1, 8, 8, 9))

        self.assertLessEqual(float(captured["scale"].max()), torch.exp(torch.tensor(0.7)).item() + 1.0e-6)

    def test_phase4_descriptor_geometry_safety_schedule(self):
        self.assertIsNone(pfm_model.descriptor_geometry_safety_for_progress("off", 0.5))

        start = pfm_model.descriptor_geometry_safety_for_progress("phase4", 0.0)
        mid_ramp = pfm_model.descriptor_geometry_safety_for_progress("phase4", 0.4)
        late_ramp = pfm_model.descriptor_geometry_safety_for_progress("phase4", 0.8)
        final = pfm_model.descriptor_geometry_safety_for_progress("phase4", 1.0)

        self.assertEqual(start, (0.0, -0.7, 0.7))
        self.assertAlmostEqual(mid_ramp[0], 0.15)
        self.assertEqual(mid_ramp[1:], (-0.7, 0.7))
        self.assertAlmostEqual(late_ramp[0], 0.4)
        self.assertEqual(late_ramp[1:], (-1.2, 1.2))
        self.assertEqual(final, (0.5, -1.2, 1.2))

    def test_geometry_aware_descriptor_pool_preserves_shape_and_normalization(self):
        descriptors = pfm_model.normalize_channels_stable(torch.randn(1, 8, 6, 7))
        orientation = pfm_model.normalize_channels_stable(torch.randn(1, 2, 6, 7))
        scale = torch.ones(1, 1, 6, 7)
        affine = torch.tensor([1.0, 0.0, 0.0, 1.0]).view(1, 4, 1, 1).expand(1, 4, 6, 7)

        pooled = pfm_model.geometry_aware_descriptor_pool(descriptors, orientation, scale, affine)

        self.assertEqual(tuple(pooled.shape), tuple(descriptors.shape))
        self.assertTrue(torch.allclose(pooled.norm(dim=1), torch.ones(1, 6, 7), atol=1.0e-5))

    def test_dense_head_and_matcher_shapes_match_cpp_contracts(self):
        dense = pfm_model.DenseHead(feature_channels=4)
        dense_output = dense(torch.randn(2, 4, 8, 9), torch.randn(2, 4, 8, 9))

        self.assertEqual(tuple(dense_output.confidence.shape), (2, 1, 8, 9))
        self.assertEqual(tuple(dense_output.offsets.shape), (2, 2, 8, 9))

        matcher = pfm_model.DescriptorMatcher(descriptor_dim=8)
        scores = matcher(torch.randn(3, 5, 8), torch.randn(3, 7, 8))
        self.assertEqual(tuple(scores.shape), (3, 5, 7))

    def test_graph_matcher_outputs_logits_with_dustbin_and_mutual_matches(self):
        graph = pfm_model.PlanetaryGraphMatcher(descriptor_dim=4, hidden_dim=8, attention_layers=1, keypoint_meta_dim=4)
        desc = torch.eye(4)
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        output = graph(desc, keypoints, desc, keypoints)

        self.assertEqual(tuple(output.logits.shape), (5, 5))
        self.assertEqual(tuple(output.accept_logits.shape), (4, 4))
        self.assertEqual(output.matches.dim(), 2)
        self.assertEqual(output.matches.size(1), 2)
        self.assertEqual(output.scores.dim(), 1)

    def test_graph_matcher_match_scores_include_accept_probability(self):
        graph = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=4,
            hidden_dim=8,
            attention_layers=1,
            keypoint_meta_dim=4,
            final_accept_score_mode="multiply",
        )
        desc = torch.eye(4)
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        output = graph(desc, keypoints, desc, keypoints, apply_candidate_mask=False)

        self.assertGreater(output.matches.size(0), 0)
        margin_scores = (
            output.logits[:4, :4]
            - output.logits[:4, 4][:, None]
            - output.logits[4, :4][None, :]
        )
        source = output.matches[:, 0]
        target = output.matches[:, 1]
        expected = margin_scores[source, target] * torch.sigmoid(output.accept_logits[source, target])

        self.assertTrue(torch.allclose(output.scores, expected.cpu(), atol=1.0e-6))

    def test_graph_matcher_can_disable_final_accept_score_gate(self):
        graph = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=4,
            hidden_dim=8,
            attention_layers=1,
            keypoint_meta_dim=4,
            final_accept_score_mode="none",
        )
        desc = torch.eye(4)
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        output = graph(desc, keypoints, desc, keypoints, apply_candidate_mask=False)

        self.assertGreater(output.matches.size(0), 0)
        margin_scores = (
            output.logits[:4, :4]
            - output.logits[:4, 4][:, None]
            - output.logits[4, :4][None, :]
        )
        source = output.matches[:, 0]
        target = output.matches[:, 1]
        self.assertTrue(torch.allclose(output.scores, margin_scores[source, target].cpu(), atol=1.0e-6))

    def test_graph_matcher_defaults_to_phase1_candidate_width(self):
        graph = pfm_model.PlanetaryGraphMatcher(descriptor_dim=4, hidden_dim=8, attention_layers=1, keypoint_meta_dim=4)
        model = pfm_model.PlanetaryFeatureMatcher(base_channels=4, descriptor_dim=8, graph_hidden_dim=16, graph_attention_layers=1)

        self.assertEqual(graph.candidate_topk, 256)
        self.assertEqual(model.config.matcher_candidate_topk, 256)

    def test_graph_attention_residual_gate_can_start_as_identity(self):
        layer = pfm_model.PlanetaryGraphAttentionLayer(hidden_dim=8, residual_gate_init=0.0)
        features_a = torch.randn(4, 8)
        features_b = torch.randn(5, 8)

        refined_a, refined_b = layer(features_a, features_b)

        self.assertTrue(torch.allclose(refined_a, features_a, atol=1.0e-6))
        self.assertTrue(torch.allclose(refined_b, features_b, atol=1.0e-6))
        self.assertAlmostEqual(float(layer.self_residual_gate.detach()), 0.0)
        self.assertAlmostEqual(float(layer.cross_residual_gate.detach()), 0.0)
        self.assertAlmostEqual(float(layer.feed_forward_residual_gate.detach()), 0.0)

    def test_matcher_calibration_can_initialize_only_new_attention_layers(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=16,
            graph_attention_layers=6,
        )
        with torch.no_grad():
            for index, layer in enumerate(model.graph_matcher.attention_layers):
                value = 0.6 + 0.05 * index
                layer.self_residual_gate.fill_(value)
                layer.cross_residual_gate.fill_(value)
                layer.feed_forward_residual_gate.fill_(value)

        model.set_matcher_calibration(
            attention_residual_gate_init=0.05,
            attention_residual_gate_start_layer=5,
        )

        for index, layer in enumerate(model.graph_matcher.attention_layers):
            expected = 0.6 + 0.05 * index if index < 4 else 0.05
            self.assertAlmostEqual(float(layer.self_residual_gate.detach()), expected)
            self.assertAlmostEqual(float(layer.cross_residual_gate.detach()), expected)
            self.assertAlmostEqual(float(layer.feed_forward_residual_gate.detach()), expected)

    def test_graph_matcher_geometry_bias_is_clamped_after_scaling(self):
        graph = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=4,
            hidden_dim=16,
            attention_layers=1,
            keypoint_meta_dim=16,
            geometry_bias_scale=3.0,
            geometry_bias_clamp=0.5,
        )
        with torch.no_grad():
            graph.geometry_bias[-1].bias.fill_(100.0)
        metadata = torch.zeros(3, 16)
        metadata[:, 12] = 1.0

        bias = graph._geometry_compatibility_bias(metadata, metadata)

        self.assertLessEqual(float(bias.detach().max()), 0.5 + 1.0e-6)
        self.assertGreaterEqual(float(bias.detach().min()), -0.5 - 1.0e-6)

    def test_graph_matcher_can_decouple_accept_logits_from_assignment_logits(self):
        graph = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=4,
            hidden_dim=8,
            attention_layers=1,
            keypoint_meta_dim=4,
            accept_assignment_mode="off",
            final_accept_score_mode="none",
        )
        with torch.no_grad():
            graph.graph_delta_scale.fill_(0.0)
            graph.raw_score_temperature.fill_(0.1)
            graph.accept_logit_scale.fill_(2.0)
        desc = torch.eye(4)
        keypoints = torch.zeros(4, 4)
        fake_accept = torch.full((4, 4), 1000.0)

        with mock.patch.object(graph, "_acceptance_logits", return_value=fake_accept):
            output = graph(desc, keypoints, desc, keypoints, apply_candidate_mask=False)

        expected_pair_logits = (desc @ desc.T) / graph.raw_score_temperature.abs().clamp(0.03, 1.0)
        self.assertTrue(torch.allclose(output.logits[:4, :4], expected_pair_logits, atol=1.0e-5))

    def test_graph_matcher_final_additive_accept_score_calibrates_scores_only(self):
        graph = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=4,
            hidden_dim=8,
            attention_layers=1,
            keypoint_meta_dim=4,
            accept_assignment_mode="off",
            final_accept_score_mode="add",
            final_accept_score_alpha=0.1,
        )
        with torch.no_grad():
            graph.graph_delta_scale.fill_(0.0)
            graph.raw_score_temperature.fill_(0.1)
        desc = torch.eye(4)
        keypoints = torch.zeros(4, 4)

        output = graph(desc, keypoints, desc, keypoints, apply_candidate_mask=False)

        self.assertGreater(output.matches.size(0), 0)
        margin_scores = (
            output.logits[:4, :4]
            - output.logits[:4, 4][:, None]
            - output.logits[4, :4][None, :]
        )
        source = output.matches[:, 0]
        target = output.matches[:, 1]
        expected = (
            margin_scores[source, target]
            + 0.1 * torch.sigmoid(output.accept_logits[source, target])
        )
        self.assertTrue(torch.allclose(output.scores, expected.cpu(), atol=1.0e-6))

    def test_graph_matcher_residual_logits_preserve_raw_descriptor_signal(self):
        graph = pfm_model.PlanetaryGraphMatcher(descriptor_dim=4, hidden_dim=8, attention_layers=1, keypoint_meta_dim=4)
        with torch.no_grad():
            graph.graph_delta_scale.fill_(0.0)
            graph.raw_score_temperature.fill_(0.1)
        desc_a = torch.eye(4)
        desc_b = desc_a[[2, 0, 3, 1]]
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        output = graph(desc_a, keypoints, desc_b, keypoints)
        pair_logits = output.logits[:4, :4]

        self.assertEqual(pair_logits.argmax(dim=1).tolist(), [1, 3, 0, 2])
        self.assertGreater(float(pair_logits.max(dim=1).values.min().detach()), 5.0)

    def test_graph_matcher_width_pruning_restores_full_logits(self):
        graph = pfm_model.PlanetaryGraphMatcher(descriptor_dim=2, hidden_dim=8, attention_layers=1, keypoint_meta_dim=4)
        with torch.no_grad():
            graph.graph_delta_scale.fill_(0.0)
            graph.accept_logit_scale.fill_(0.0)
            graph.raw_score_temperature.fill_(0.1)
        desc_a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        desc_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        keypoints_a = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        keypoints_b = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        output = graph(desc_a, keypoints_a, desc_b, keypoints_b, width_prune_min_score=0.5)

        self.assertEqual(tuple(output.logits.shape), (4, 3))
        self.assertEqual(tuple(output.accept_logits.shape), (3, 2))
        self.assertNotIn(2, output.matches[:, 0].tolist())
        self.assertTrue(torch.all(output.logits[2, :2] < -9000.0))
        self.assertTrue(torch.all(output.accept_logits[2] < -9000.0))

    def test_graph_matcher_width_pruning_uses_layer_acceptance(self):
        graph = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=5,
            hidden_dim=16,
            attention_layers=3,
            keypoint_meta_dim=16,
            candidate_topk=0,
        )
        with torch.no_grad():
            for parameter in graph.accept_head.parameters():
                parameter.zero_()
            graph.accept_head[0].weight[0, 4] = 10.0
            graph.accept_head[0].bias[0] = -7.5
            graph.accept_head[2].weight[0, 0] = 4.0
            graph.accept_head[2].bias[0] = -4.0
        descriptors = torch.eye(5)
        metadata = torch.zeros(5, 16)
        metadata[:, 12] = 1.0
        metadata[4, 12] = 0.0

        output = graph(descriptors, metadata, descriptors, metadata, width_prune_min_score=0.8)

        self.assertTrue(torch.any(output.accept_logits[:4, :4] > -100.0))
        self.assertTrue(torch.all(output.accept_logits[4, :] < -9000.0))
        self.assertTrue(torch.all(output.accept_logits[:, 4] < -9000.0))
        self.assertEqual(output.executed_layers, 3)
        self.assertEqual(output.input_keypoints_a, 5)
        self.assertEqual(output.input_keypoints_b, 5)
        self.assertEqual(output.kept_keypoints_a, 4)
        self.assertEqual(output.kept_keypoints_b, 4)
        self.assertEqual(output.pruned_keypoints_a, 1)
        self.assertEqual(output.pruned_keypoints_b, 1)
        self.assertEqual(output.attention_work_units, 57)
        self.assertEqual(output.full_attention_work_units, 75)
        self.assertAlmostEqual(output.attention_work_fraction, 57 / 75)

    def test_graph_matcher_width_pruning_can_keep_top_ratio(self):
        graph = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=5,
            hidden_dim=16,
            attention_layers=3,
            keypoint_meta_dim=16,
            candidate_topk=0,
        )
        with torch.no_grad():
            for parameter in graph.accept_head.parameters():
                parameter.zero_()
            graph.accept_head[0].weight[0, 4] = 10.0
            graph.accept_head[0].bias[0] = -7.5
            graph.accept_head[2].weight[0, 0] = 4.0
            graph.accept_head[2].bias[0] = -4.0
        descriptors = torch.eye(5)
        metadata = torch.zeros(5, 16)
        metadata[:, 12] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.1])

        output = graph(descriptors, metadata, descriptors, metadata, width_prune_keep_ratio=0.4)

        self.assertEqual(tuple(output.logits.shape), (6, 6))
        self.assertEqual(tuple(output.accept_logits.shape), (5, 5))
        self.assertTrue(torch.any(output.accept_logits[:2, :2] > -100.0))
        self.assertTrue(torch.all(output.accept_logits[2:, :] < -9000.0))
        self.assertTrue(torch.all(output.accept_logits[:, 2:] < -9000.0))
        self.assertEqual(output.executed_layers, 3)
        self.assertEqual(output.input_keypoints_a, 5)
        self.assertEqual(output.input_keypoints_b, 5)
        self.assertEqual(output.kept_keypoints_a, 2)
        self.assertEqual(output.kept_keypoints_b, 2)
        self.assertEqual(output.pruned_keypoints_a, 3)
        self.assertEqual(output.pruned_keypoints_b, 3)
        self.assertEqual(output.attention_work_units, 33)
        self.assertEqual(output.full_attention_work_units, 75)
        self.assertAlmostEqual(output.attention_work_fraction, 33 / 75)

    def test_graph_matcher_top_ratio_survives_over_strict_accept_threshold(self):
        graph = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=5,
            hidden_dim=16,
            attention_layers=3,
            keypoint_meta_dim=16,
            candidate_topk=0,
        )
        with torch.no_grad():
            for parameter in graph.accept_head.parameters():
                parameter.zero_()
            graph.accept_head[2].bias.fill_(-4.0)
        descriptors = torch.eye(5)
        metadata = torch.zeros(5, 16)
        metadata[:, 12] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.1])

        output = graph(
            descriptors,
            metadata,
            descriptors,
            metadata,
            width_prune_min_score=0.8,
            width_prune_keep_ratio=0.4,
        )

        self.assertEqual(output.kept_keypoints_a, 2)
        self.assertEqual(output.kept_keypoints_b, 2)
        self.assertEqual(output.pruned_keypoints_a, 3)
        self.assertEqual(output.pruned_keypoints_b, 3)
        self.assertLess(output.attention_work_fraction, 1.0)

    def test_graph_matcher_can_stop_attention_layers_when_confident(self):
        graph = pfm_model.PlanetaryGraphMatcher(descriptor_dim=2, hidden_dim=8, attention_layers=3, keypoint_meta_dim=4)
        desc = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        output = graph(desc, keypoints, desc, keypoints, early_stop_min_confidence=0.0)

        self.assertEqual(tuple(output.logits.shape), (3, 3))
        self.assertEqual(graph.last_executed_attention_layers, 1)

    def test_graph_matcher_respects_max_attention_layers(self):
        graph = pfm_model.PlanetaryGraphMatcher(descriptor_dim=2, hidden_dim=8, attention_layers=4, keypoint_meta_dim=4)
        desc = torch.eye(3, dtype=torch.float32)[:, :2]
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

        output = graph(desc, keypoints, desc, keypoints, max_attention_layers=2)

        self.assertEqual(tuple(output.logits.shape), (4, 4))
        self.assertEqual(output.executed_layers, 2)
        self.assertEqual(graph.last_executed_attention_layers, 2)
        self.assertEqual(output.attention_work_units, 18)
        self.assertEqual(output.full_attention_work_units, 36)
        self.assertAlmostEqual(output.attention_work_fraction, 0.5)

    def test_graph_matcher_respects_attention_work_fraction_budget(self):
        graph = pfm_model.PlanetaryGraphMatcher(descriptor_dim=2, hidden_dim=8, attention_layers=4, keypoint_meta_dim=4)
        desc = torch.eye(3, dtype=torch.float32)[:, :2]
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

        output = graph(desc, keypoints, desc, keypoints, max_attention_work_fraction=0.5)

        self.assertEqual(tuple(output.logits.shape), (4, 4))
        self.assertEqual(output.executed_layers, 2)
        self.assertEqual(graph.last_executed_attention_layers, 2)
        self.assertEqual(output.attention_work_units, 18)
        self.assertEqual(output.full_attention_work_units, 36)
        self.assertAlmostEqual(output.attention_work_fraction, 0.5)

    def test_graph_matcher_early_stop_tolerates_single_uncertain_keypoint(self):
        graph = pfm_model.PlanetaryGraphMatcher(descriptor_dim=5, hidden_dim=8, attention_layers=3, keypoint_meta_dim=4)
        with torch.no_grad():
            graph.graph_delta_scale.fill_(0.0)
            graph.accept_logit_scale.fill_(0.0)
            graph.raw_score_temperature.fill_(0.03)
        desc_a = torch.eye(5)
        desc_b = torch.eye(5)
        desc_a[4] = 0.0
        desc_b[4] = 0.0
        keypoints = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            dtype=torch.float32,
        )

        output = graph(desc_a, keypoints, desc_b, keypoints, early_stop_min_confidence=0.8)

        self.assertEqual(tuple(output.logits.shape), (6, 6))
        self.assertEqual(graph.last_executed_attention_layers, 1)

    def test_current_libtorch_checkpoint_loads_strictly_when_available(self):
        checkpoint = Path("runs/rotation_clean_2026-05-25/rotation_clean_ft_e1_b2.pt")
        if not checkpoint.exists():
            self.skipTest("current rotation checkpoint is unavailable")

        model, config = pfm_model.load_libtorch_checkpoint(checkpoint, device="cpu")

        self.assertEqual(config.input_channels, 1)
        self.assertGreater(config.base_channels, 0)
        self.assertGreater(config.descriptor_dim, 0)
        self.assertGreater(config.graph_keypoint_meta_dim, 0)
        self.assertEqual(model.sparse_head.descriptor_dim, config.descriptor_dim)
        image = torch.randn(1, config.input_channels, 64, 64)
        with torch.no_grad():
            raw = model.forward_single(image)
        self.assertEqual(tuple(raw.heatmap.shape[-2:]), tuple(raw.descriptors.shape[-2:]))
        self.assertEqual(raw.descriptors.size(1), config.descriptor_dim)
        self.assertEqual(tuple(raw.quality.shape), tuple(raw.heatmap.shape))

    def test_pytorch_state_round_trip_loads_config_and_weights(self):
        with torch.no_grad():
            model = pfm_model.PlanetaryFeatureMatcher(
                input_channels=1,
                base_channels=4,
                descriptor_dim=16,
                graph_hidden_dim=16,
                graph_attention_layers=1,
            )
            model.sparse_head.descriptor_skip.bias.fill_(0.25)
        path = Path("/tmp/pfm_pytorch_state_roundtrip.pt")
        torch.save({"config": model.config.__dict__, "model": model.state_dict()}, path)

        loaded, config = pfm_model.load_pytorch_state(path, device="cpu")

        self.assertEqual(config.base_channels, 4)
        self.assertEqual(config.graph_keypoint_meta_dim, 16)
        self.assertTrue(torch.allclose(loaded.sparse_head.descriptor_skip.bias, torch.full((16,), 0.25)))

    def test_pytorch_state_load_can_override_graph_architecture_shape_safely(self):
        with torch.no_grad():
            model = pfm_model.PlanetaryFeatureMatcher(
                input_channels=1,
                base_channels=4,
                descriptor_dim=16,
                graph_hidden_dim=16,
                graph_attention_layers=1,
            )
            model.sparse_head.descriptor_skip.bias.fill_(0.25)
            model.graph_matcher.descriptor_projection.bias.fill_(0.5)
        path = Path("/tmp/pfm_pytorch_state_graph_override.pt")
        torch.save({"config": model.config.__dict__, "model": model.state_dict()}, path)

        loaded, config = pfm_model.load_pytorch_state(
            path,
            device="cpu",
            strict=False,
            graph_hidden_dim=8,
            graph_attention_layers=2,
        )

        self.assertEqual(config.graph_hidden_dim, 8)
        self.assertEqual(config.graph_attention_layers, 2)
        self.assertEqual(loaded.config.graph_hidden_dim, 8)
        self.assertEqual(loaded.config.graph_attention_layers, 2)
        self.assertTrue(torch.allclose(loaded.sparse_head.descriptor_skip.bias, torch.full((16,), 0.25)))
        self.assertEqual(tuple(loaded.graph_matcher.descriptor_projection.bias.shape), (8,))

    def test_interpolate_pytorch_state_payloads_blends_floating_weights_only(self):
        first = {
            "config": {"base_channels": 4},
            "model": {
                "weight": torch.tensor([0.0, 2.0]),
                "counter": torch.tensor([3], dtype=torch.long),
            },
        }
        second = {
            "config": {"base_channels": 4},
            "model": {
                "weight": torch.tensor([2.0, 6.0]),
                "counter": torch.tensor([9], dtype=torch.long),
            },
        }

        mixed = pfm_model.interpolate_pytorch_state_payloads(first, second, alpha=0.25)

        self.assertTrue(torch.allclose(mixed["model"]["weight"], torch.tensor([0.5, 3.0])))
        self.assertTrue(torch.equal(mixed["model"]["counter"], torch.tensor([3], dtype=torch.long)))
        self.assertEqual(mixed["interpolation"]["alpha"], 0.25)

    def test_descriptor_map_single_matches_forward_single_descriptor_output(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=16,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        model.eval()
        image = torch.randn(1, 1, 32, 32)

        with torch.no_grad():
            descriptors = model.descriptor_map_single(image)
            raw = model.forward_single(image)

        self.assertTrue(torch.allclose(descriptors, raw.descriptors, atol=1.0e-6))

    def test_default_texture_blend_matches_cpp_inference_default(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=16,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        model.eval()
        image = torch.randn(1, 1, 32, 32)

        with torch.no_grad():
            default_descriptors = model.descriptor_map_single(image)
            explicit_descriptors = model.descriptor_map_single(image, texture_blend_weight=1.0)

        self.assertTrue(torch.allclose(default_descriptors, explicit_descriptors, atol=1.0e-6))

    def test_learned_and_texture_descriptor_maps_are_available_separately(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=16,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        image = torch.randn(1, 1, 31, 33)

        learned = model.learned_descriptor_map_single(image)
        texture = model.texture_descriptor_map_single(image)
        blended = model.descriptor_map_single(image, texture_blend_weight=0.0)

        self.assertEqual(tuple(learned.shape), (1, 16, 8, 9))
        self.assertEqual(tuple(texture.shape), tuple(learned.shape))
        self.assertTrue(torch.allclose(learned, blended, atol=1.0e-6))

    def test_texture_descriptor_adapter_is_identity_at_initialization(self):
        adapter = pfm_model.TextureDescriptorAdapter(descriptor_dim=8)
        texture = torch.randn(2, 8, 5, 7)

        adapted = adapter(texture)

        self.assertTrue(torch.allclose(adapted, pfm_model.normalize_channels_stable(texture), atol=1.0e-6))
        self.assertTrue(torch.allclose(adapted.norm(dim=1), torch.ones(2, 5, 7), atol=1.0e-5))

    def test_descriptor_fusion_adapter_is_identity_at_initialization(self):
        fusion = pfm_model.DescriptorFusionAdapter(descriptor_dim=8)
        learned = pfm_model.normalize_channels_stable(torch.randn(2, 8, 5, 7))
        texture = pfm_model.normalize_channels_stable(torch.randn(2, 8, 5, 7))

        fused = fusion(learned, texture, blend_weight=1.5)
        expected = pfm_model.normalize_channels_stable(learned + texture * 1.5)

        self.assertTrue(torch.allclose(fused, expected, atol=1.0e-6))
        self.assertGreater(
            sum(parameter.numel() for parameter in fusion.parameters()),
            sum(parameter.numel() for parameter in pfm_model.TextureDescriptorAdapter(8).parameters()),
        )

    def test_descriptor_map_uses_trainable_texture_adapter_without_changing_initial_output(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=8,
            graph_attention_layers=1,
        )
        image = torch.randn(1, 1, 32, 32)

        with torch.no_grad():
            learned = model.learned_descriptor_map_single(image)
            raw_texture = model.raw_texture_descriptor_map_single(image)
            initial = model.descriptor_map_single(image, texture_blend_weight=1.0)
            expected_initial = pfm_model.normalize_channels_stable(learned + raw_texture)
            model.texture_adapter.residual.weight.fill_(0.05)
            changed = model.descriptor_map_single(image, texture_blend_weight=1.0)

        self.assertTrue(torch.allclose(initial, expected_initial, atol=1.0e-6))
        self.assertFalse(torch.allclose(changed, expected_initial, atol=1.0e-4))

    def test_descriptor_map_uses_trainable_fusion_adapter_without_changing_initial_output(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=8,
            graph_attention_layers=1,
        )
        image = torch.randn(1, 1, 32, 32)

        with torch.no_grad():
            initial = model.descriptor_map_single(image, texture_blend_weight=1.0)
            model.descriptor_fusion.output.bias.fill_(0.05)
            changed = model.descriptor_map_single(image, texture_blend_weight=1.0)

        self.assertFalse(torch.allclose(changed, initial, atol=1.0e-4))

    def test_legacy_pytorch_state_without_new_descriptor_modules_loads_strictly(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=8,
            graph_hidden_dim=8,
            graph_attention_layers=1,
        )
        legacy_state = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("texture_adapter.") and not key.startswith("descriptor_fusion.")
        }
        path = Path("/tmp/pfm_legacy_without_texture_adapter.pt")
        torch.save({"config": model.config.__dict__, "model": legacy_state}, path)

        loaded, _ = pfm_model.load_pytorch_state(path, device="cpu", strict=True)

        self.assertTrue(hasattr(loaded, "texture_adapter"))
        self.assertTrue(hasattr(loaded, "descriptor_fusion"))
        self.assertTrue(hasattr(loaded.backbone, "stage2_refine"))
        self.assertTrue(hasattr(loaded.sparse_head, "keypoint_offsets"))
        for parameter in loaded.texture_adapter.parameters():
            self.assertTrue(torch.allclose(parameter, torch.zeros_like(parameter)))
        self.assertTrue(torch.allclose(loaded.descriptor_fusion.output.weight, torch.zeros_like(loaded.descriptor_fusion.output.weight)))

    def test_default_model_uses_v2_full_capacity(self):
        model = pfm_model.PlanetaryFeatureMatcher()

        self.assertEqual(model.config.base_channels, 64)
        self.assertEqual(model.config.descriptor_dim, 256)
        self.assertEqual(model.config.graph_hidden_dim, 512)
        self.assertEqual(model.config.graph_keypoint_meta_dim, 16)

    def test_graph_matcher_keypoint_embedding_can_include_xy_metadata(self):
        keypoints = torch.tensor([[0.0, 0.0], [4.0, 2.0], [8.0, 6.0]], dtype=torch.float32)

        meta = pfm_model.prepare_keypoints_for_embedding(keypoints, meta_dim=4)

        self.assertEqual(tuple(meta.shape), (3, 4))
        self.assertTrue(torch.all(meta[:, 0].abs() <= 1.0))
        self.assertTrue(torch.all(meta[:, 1].abs() <= 1.0))

    def test_prepare_graph_keypoint_metadata_builds_v21_fields(self):
        keypoints = torch.tensor([[0.0, 0.0], [4.0, 2.0]], dtype=torch.float32)
        scores = torch.tensor([0.8, 0.3])

        meta = pfm_model.prepare_graph_keypoint_metadata(keypoints, meta_dim=16, scores=scores, quality=scores)

        self.assertEqual(tuple(meta.shape), (2, 16))
        self.assertTrue(torch.allclose(meta[:, 4], scores))
        self.assertTrue(torch.allclose(meta[:, 12], scores))

    def test_prepare_graph_keypoint_metadata_uses_reliability_columns(self):
        keypoints = torch.tensor([[0.0, 0.0], [4.0, 2.0]], dtype=torch.float32)
        matchability = torch.tensor([[0.8], [0.2]], dtype=torch.float32)
        descriptor_uncertainty = torch.tensor([[0.1], [0.7]], dtype=torch.float32)
        no_match_prior = torch.tensor([[0.05], [0.9]], dtype=torch.float32)

        meta = pfm_model.prepare_graph_keypoint_metadata(
            keypoints,
            meta_dim=16,
            matchability=matchability,
            descriptor_uncertainty=descriptor_uncertainty,
            no_match_prior=no_match_prior,
        )

        self.assertTrue(torch.allclose(meta[:, 12:13], matchability))
        self.assertTrue(torch.allclose(meta[:, 14:15], descriptor_uncertainty))
        self.assertTrue(torch.allclose(meta[:, 15:16], no_match_prior))

    def test_graph_matcher_legacy_full_dustbin_bias_ignores_no_match_prior(self):
        graph = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=2,
            hidden_dim=8,
            attention_layers=1,
            keypoint_meta_dim=16,
            reliability_dustbin_bias_mode="full",
        )
        descriptors = torch.eye(2, dtype=torch.float32)
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        metadata = pfm_model.prepare_graph_keypoint_metadata(
            keypoints,
            meta_dim=16,
            matchability=torch.full((2, 1), 0.5),
            descriptor_uncertainty=torch.full((2, 1), 0.5),
            no_match_prior=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        )

        output = graph(descriptors, metadata, descriptors, metadata, apply_candidate_mask=False)

        self.assertAlmostEqual(float((output.logits[1, -1] - output.logits[0, -1]).detach()), 0.0, places=6)
        self.assertAlmostEqual(float((output.logits[-1, 1] - output.logits[-1, 0]).detach()), 0.0, places=6)

    def test_graph_matcher_default_dustbin_bias_ignores_no_match_prior(self):
        graph = pfm_model.PlanetaryGraphMatcher(descriptor_dim=2, hidden_dim=8, attention_layers=1, keypoint_meta_dim=16)
        descriptors = torch.eye(2, dtype=torch.float32)
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        metadata = pfm_model.prepare_graph_keypoint_metadata(
            keypoints,
            meta_dim=16,
            matchability=torch.full((2, 1), 0.5),
            descriptor_uncertainty=torch.full((2, 1), 0.5),
            no_match_prior=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        )

        output = graph(descriptors, metadata, descriptors, metadata, apply_candidate_mask=False)

        self.assertAlmostEqual(float((output.logits[1, -1] - output.logits[0, -1]).detach()), 0.0, places=6)
        self.assertAlmostEqual(float((output.logits[-1, 1] - output.logits[-1, 0]).detach()), 0.0, places=6)

    def test_graph_matcher_legacy_full_pair_bias_is_noop(self):
        descriptors = torch.eye(2, dtype=torch.float32)
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        metadata = pfm_model.prepare_graph_keypoint_metadata(
            keypoints,
            meta_dim=16,
            matchability=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
            descriptor_uncertainty=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            no_match_prior=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        )
        full = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=2,
            hidden_dim=8,
            attention_layers=1,
            keypoint_meta_dim=16,
            reliability_pair_bias_mode="full",
        )
        off = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=2,
            hidden_dim=8,
            attention_layers=1,
            keypoint_meta_dim=16,
            reliability_pair_bias_mode="off",
        )
        for graph in (full, off):
            with torch.no_grad():
                graph.graph_delta_scale.fill_(0.0)
                graph.accept_logit_scale.fill_(0.0)
                graph.raw_score_temperature.fill_(0.1)

        full_output = full(descriptors, metadata, descriptors, metadata, apply_candidate_mask=False)
        off_output = off(descriptors, metadata, descriptors, metadata, apply_candidate_mask=False)

        self.assertLess(float((full_output.logits[0, 0] - full_output.logits[1, 1]).detach().abs()), 1.0e-5)
        self.assertLess(float((off_output.logits[0, 0] - off_output.logits[1, 1]).detach().abs()), 1.0e-5)

    def test_graph_matcher_legacy_matchability_dustbin_bias_is_noop(self):
        graph = pfm_model.PlanetaryGraphMatcher(
            descriptor_dim=2,
            hidden_dim=8,
            attention_layers=1,
            keypoint_meta_dim=16,
            reliability_dustbin_bias_mode="matchability",
        )
        descriptors = torch.eye(2, dtype=torch.float32)
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        metadata = pfm_model.prepare_graph_keypoint_metadata(
            keypoints,
            meta_dim=16,
            matchability=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            descriptor_uncertainty=torch.full((2, 1), 0.5),
            no_match_prior=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        )

        output = graph(descriptors, metadata, descriptors, metadata, apply_candidate_mask=False)

        self.assertLess(float((output.logits[1, -1] - output.logits[0, -1]).detach().abs()), 1.0e-5)
        self.assertLess(float((output.logits[-1, 1] - output.logits[-1, 0]).detach().abs()), 1.0e-5)

    def test_normalize_channels_handles_large_finite_values_without_zeroing(self):
        tensor = torch.tensor([[[[1.0e30]], [[2.0e30]], [[-3.0e30]]]], dtype=torch.float32)

        normalized = pfm_model.normalize_channels_stable(tensor)

        self.assertTrue(torch.isfinite(normalized).all())
        self.assertGreater(float(normalized.abs().max()), 0.0)
        self.assertTrue(torch.allclose(normalized.norm(dim=1), torch.ones(1, 1, 1), atol=1.0e-5))

    def test_normalize_channels_keeps_zero_input_gradients_bounded(self):
        tensor = torch.zeros(1, 4, 2, 2, requires_grad=True)

        normalized = pfm_model.normalize_channels_stable(tensor)
        normalized.sum().backward()

        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertLessEqual(float(tensor.grad.abs().max()), 10000.0)

    def test_descriptor_training_gradients_ignore_nonfinite_image_pixels(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=16,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        descriptor_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("sparse_head.descriptor") or name.startswith("sparse_head.descriptors")
        ]
        for parameter in descriptor_parameters:
            parameter.requires_grad_(True)
        image = torch.randn(1, 1, 32, 32)
        image[:, :, 5, 7] = float("nan")
        image[:, :, 9, 11] = float("inf")

        descriptors = model.descriptor_map_single(image, texture_blend_weight=1.0)
        descriptors.square().mean().backward()

        self.assertTrue(torch.isfinite(descriptors).all())
        for parameter in descriptor_parameters:
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_texture_descriptor_sanitizes_nonfinite_image_pixels_before_filtering(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=16,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )
        image = torch.randn(1, 1, 32, 32)
        image[:, :, 5, 7] = float("nan")
        image[:, :, 9, 11] = float("inf")
        sanitized = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        actual = model.texture_descriptor_map_single(image)
        expected = model.texture_descriptor_map_single(sanitized)

        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6))

    def test_forward_single_returns_v21_quality_maps(self):
        model = pfm_model.PlanetaryFeatureMatcher(
            input_channels=1,
            base_channels=4,
            descriptor_dim=16,
            graph_hidden_dim=16,
            graph_attention_layers=1,
        )

        with torch.no_grad():
            raw = model.forward_single(torch.randn(1, 1, 32, 32))

        self.assertEqual(tuple(raw.quality.shape), tuple(raw.heatmap.shape))
        self.assertEqual(tuple(raw.local_contrast.shape), tuple(raw.heatmap.shape))
        self.assertTrue(torch.all((raw.quality >= 0.0) & (raw.quality <= 1.0)))

    def test_semi_dense_candidate_branch_outputs_feature_grid_keypoints(self):
        branch = pfm_model.SemiDenseCandidateBranch(descriptor_dim=8, projection_dim=4, max_grid=8)
        desc_a = pfm_model.normalize_channels_stable(torch.randn(1, 8, 12, 10))
        desc_b = pfm_model.normalize_channels_stable(torch.randn(1, 8, 12, 10))

        output = branch(desc_a, desc_b, max_candidates=6)

        self.assertEqual(tuple(output.keypoints_a.shape), (6, 2))
        self.assertEqual(tuple(output.keypoints_b.shape), (6, 2))
        self.assertEqual(tuple(output.scores.shape), (6,))
        self.assertTrue(torch.all(output.keypoints_a[:, 0] >= 0.0))
        self.assertTrue(torch.all(output.keypoints_a[:, 0] <= 9.0))
        self.assertTrue(torch.all(output.keypoints_a[:, 1] <= 11.0))


if __name__ == "__main__":
    unittest.main()
