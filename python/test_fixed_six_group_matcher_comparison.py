import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "fixed_six_group_matcher_comparison.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fixed_six_group_matcher_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixedSixGroupMatcherComparisonTests(unittest.TestCase):
    def test_pair_path_falls_back_to_img_root_when_split_missing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_pair = root / "img" / "Rotate_1024" / "source_000001" / "pair_000123.pt"
            img_pair.parent.mkdir(parents=True)
            img_pair.write_text("pair", encoding="utf-8")
            pair = module.FixedPair("numeric", "rotate", "rot90", "source_000001", "pair_000123.pt", "90")

            resolved = module.pair_path(pair, split_root=root / "missing_split", img_root=root / "img")

            self.assertEqual(resolved, img_pair)

    def test_missing_route_uses_direct_pfm_state_for_all_groups(self):
        module = load_module()
        args = argparse.Namespace(
            pfm_texture_blend_weight=0.75,
            pfm_keypoint_score_mode="learned",
            pfm_min_margin=0.05,
            pfm_min_target_gradient=1.0,
            pfm_min_target_local_contrast=2.0,
            pfm_state=Path("runs/model.pt"),
        )

        params = module.load_pfm_route_params(Path("/tmp/does_not_exist"), args)

        self.assertEqual(len(params), 6)
        item = params[("numeric", "rotate")]
        self.assertEqual(item.texture_blend_weight, 0.75)
        self.assertEqual(item.keypoint_score_mode, "learned")
        self.assertEqual(item.min_margin, 0.05)
        self.assertEqual(item.min_target_gradient, 1.0)
        self.assertEqual(item.min_target_local_contrast, 2.0)
        self.assertEqual(item.pytorch_state, PROJECT_ROOT / "runs" / "model.pt")


if __name__ == "__main__":
    unittest.main()
