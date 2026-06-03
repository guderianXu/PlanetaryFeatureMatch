import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "verify_pair_cache_dataset.py"
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from compact_pair_cache import make_compact_pair_payload, save_shared_image  # noqa: E402


class VerifyPairCacheDatasetTests(unittest.TestCase):
    def _write_compact_pair(self, root: Path, split: str, index: int) -> None:
        pair_dir = root / "cache" / split / f"source_{index:05d}"
        pair_dir.mkdir(parents=True)
        image = torch.full((1, 4, 4), float(index), dtype=torch.float32)
        image_path = save_shared_image(image, root / "image_store")
        warp = torch.zeros((4, 4, 2), dtype=torch.float32)
        valid_mask = torch.ones((4, 4), dtype=torch.bool)
        pair_path = pair_dir / f"pair_{index:06d}.pt"
        payload = make_compact_pair_payload(
            pair_path=pair_path,
            image_a_path=image_path,
            image_b_path=image_path,
            warp_a_to_b=warp,
            valid_mask=valid_mask,
        )
        torch.save(payload, pair_path)

    def test_verify_compact_cache_counts_ratio_and_loads_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            layout = {"train": 7, "val": 2, "test": 1}
            index = 0
            for split, count in layout.items():
                for _ in range(count):
                    self._write_compact_pair(root, split, index)
                    index += 1

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-root",
                    str(root),
                    "--expected-total",
                    "10",
                    "--expected-ratio",
                    "7:2:1",
                    "--samples-per-split",
                    "2",
                ],
                text=True,
                check=True,
                capture_output=True,
            )
            report = json.loads(result.stdout)
            self.assertTrue(report["ok"])
            self.assertEqual(report["counts"], layout)
            self.assertEqual(report["expected_ratio_counts"], layout)
            self.assertEqual(len(report["loaded_samples"]["train"]), 2)
            self.assertEqual(len(report["loaded_samples"]["test"]), 1)


if __name__ == "__main__":
    unittest.main()
