import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pfm_model


class PFMModelTest(unittest.TestCase):
    def test_backbone_matches_libtorch_feature_pyramid_shapes(self):
        model = pfm_model.Backbone(input_channels=1, base_channels=4)
        features = model(torch.randn(2, 1, 64, 80))

        self.assertEqual([tuple(feature.shape) for feature in features], [
            (2, 4, 32, 40),
            (2, 8, 16, 20),
            (2, 16, 8, 10),
            (2, 32, 4, 5),
        ])

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

    def test_sparse_head_accepts_separate_keypoint_and_descriptor_features(self):
        head = pfm_model.SparseHead(input_channels=8, descriptor_dim=16)
        output = head(torch.randn(2, 8, 16, 20), torch.randn(2, 8, 16, 20))

        self.assertEqual(tuple(output.heatmap.shape), (2, 1, 16, 20))
        self.assertEqual(tuple(output.descriptors.shape), (2, 16, 16, 20))

    def test_dual_fpn_lite_returns_separate_p2_features(self):
        backbone = pfm_model.Backbone(input_channels=1, base_channels=4)
        fpn = pfm_model.DualFPNLite(base_channels=4)
        features = backbone(torch.randn(2, 1, 64, 80))

        p2_keypoint, p2_descriptor = fpn(features)

        self.assertEqual(tuple(p2_keypoint.shape), (2, 8, 16, 20))
        self.assertEqual(tuple(p2_descriptor.shape), (2, 8, 16, 20))

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
        self.assertEqual(output.matches.dim(), 2)
        self.assertEqual(output.matches.size(1), 2)
        self.assertEqual(output.scores.dim(), 1)

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
