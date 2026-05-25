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
        self.assertTrue(torch.allclose(output.descriptors.norm(dim=1), torch.ones(2, 16, 20), atol=1.0e-5))

    def test_dense_head_and_matcher_shapes_match_cpp_contracts(self):
        dense = pfm_model.DenseHead(feature_channels=4)
        dense_output = dense(torch.randn(2, 4, 8, 9), torch.randn(2, 4, 8, 9))

        self.assertEqual(tuple(dense_output.confidence.shape), (2, 1, 8, 9))
        self.assertEqual(tuple(dense_output.offsets.shape), (2, 2, 8, 9))

        matcher = pfm_model.DescriptorMatcher(descriptor_dim=8)
        scores = matcher(torch.randn(3, 5, 8), torch.randn(3, 7, 8))
        self.assertEqual(tuple(scores.shape), (3, 5, 7))

    def test_graph_matcher_outputs_logits_with_dustbin_and_mutual_matches(self):
        graph = pfm_model.PlanetaryGraphMatcher(descriptor_dim=4, hidden_dim=8, attention_layers=1)
        desc = torch.eye(4)
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        output = graph(desc, keypoints, desc, keypoints)

        self.assertEqual(tuple(output.logits.shape), (5, 5))
        self.assertEqual(output.matches.dim(), 2)
        self.assertEqual(output.matches.size(1), 2)
        self.assertEqual(output.scores.dim(), 1)

    def test_current_libtorch_checkpoint_loads_strictly_when_available(self):
        checkpoint = Path("runs/rotation_clean_2026-05-25/rotation_clean_ft_e1_b2.pt")
        if not checkpoint.exists():
            self.skipTest("current rotation checkpoint is unavailable")

        model, config = pfm_model.load_libtorch_checkpoint(checkpoint, device="cpu")

        self.assertEqual(config.input_channels, 1)
        self.assertGreater(config.base_channels, 0)
        self.assertGreater(config.descriptor_dim, 0)
        self.assertEqual(model.sparse_head.descriptor_dim, config.descriptor_dim)
        image = torch.randn(1, config.input_channels, 64, 64)
        with torch.no_grad():
            raw = model.forward_single(image)
        self.assertEqual(tuple(raw.heatmap.shape[-2:]), tuple(raw.descriptors.shape[-2:]))
        self.assertEqual(raw.descriptors.size(1), config.descriptor_dim)

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

    def test_normalize_channels_handles_large_finite_values_without_zeroing(self):
        tensor = torch.tensor([[[[1.0e30]], [[2.0e30]], [[-3.0e30]]]], dtype=torch.float32)

        normalized = pfm_model.normalize_channels_stable(tensor)

        self.assertTrue(torch.isfinite(normalized).all())
        self.assertGreater(float(normalized.abs().max()), 0.0)
        self.assertTrue(torch.allclose(normalized.norm(dim=1), torch.ones(1, 1, 1), atol=1.0e-5))


if __name__ == "__main__":
    unittest.main()
