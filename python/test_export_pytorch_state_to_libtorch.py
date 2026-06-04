import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_pytorch_state_to_libtorch as exporter
import pfm_model


class ExportPytorchStateToLibtorchTest(unittest.TestCase):
    def test_exported_archive_loads_as_libtorch_checkpoint(self):
        with torch.no_grad():
            model = pfm_model.PlanetaryFeatureMatcher(
                input_channels=1,
                base_channels=4,
                descriptor_dim=16,
                graph_hidden_dim=16,
                graph_attention_layers=1,
                graph_keypoint_meta_dim=16,
            )
            model.sparse_head.descriptor_skip.bias.fill_(0.375)
            model.graph_matcher.raw_score_temperature.fill_(0.125)

        source = Path("/tmp/pfm_export_pytorch_state_source.pt")
        output = Path("/tmp/pfm_export_pytorch_state_libtorch.pt")
        torch.save({"config": model.config.__dict__, "model": model.state_dict()}, source)

        exporter.export_pytorch_state_to_libtorch(source, output)
        archive = torch.jit.load(str(output), map_location="cpu")
        loaded, config = pfm_model.load_libtorch_checkpoint(output, device="cpu")

        self.assertIn("backbone.stage1.2", dict(archive.named_modules()))
        self.assertEqual(config.base_channels, 4)
        self.assertEqual(config.descriptor_dim, 16)
        self.assertEqual(config.graph_keypoint_meta_dim, 16)
        self.assertTrue(torch.allclose(loaded.sparse_head.descriptor_skip.bias, torch.full((16,), 0.375)))
        self.assertTrue(torch.allclose(loaded.graph_matcher.raw_score_temperature, torch.tensor(0.125)))


if __name__ == "__main__":
    unittest.main()
