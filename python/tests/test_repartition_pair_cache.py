import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "repartition_pair_cache.py"


class RepartitionPairCacheTests(unittest.TestCase):
    def _create_input_dataset(self, root: Path) -> None:
        for split in ("train", "val", "test"):
            for index in range({"train": 5, "val": 3, "test": 2}[split]):
                source = root / "cache" / split / f"source_{index:05d}_TrackA"
                source.mkdir(parents=True)
                pair = source / f"pair_{index:06d}.pt"
                pair.write_text(f"{split}-{index}", encoding="utf-8")
                pair.with_suffix(".json").write_text("{}", encoding="utf-8")
        (root / "image_store").mkdir()
        (root / "image_store" / "shared.pt").write_text("image", encoding="utf-8")
        (root / "tsai_tracks").mkdir()
        (root / "tsai_tracks" / "camera.tsai").write_text("camera", encoding="utf-8")
        (root / "dataset_metadata.json").write_text("{}", encoding="utf-8")

    def test_copy_mode_creates_self_contained_ratio_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            output = Path(tmp) / "output"
            self._create_input_dataset(root)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-root",
                    str(root),
                    "--output-root",
                    str(output),
                    "--ratio",
                    "7:2:1",
                    "--link-mode",
                    "copy",
                    "--workers",
                    "2",
                ],
                check=True,
            )

            counts = {
                split: len(list((output / "cache" / split).glob("source_*/pair_*.pt")))
                for split in ("train", "val", "test")
            }
            self.assertEqual(counts, {"train": 7, "val": 2, "test": 1})
            self.assertTrue((output / "image_store" / "shared.pt").is_file())
            self.assertFalse((output / "image_store").is_symlink())
            self.assertTrue((output / "tsai_tracks" / "camera.tsai").is_file())
            metadata = json.loads((output / "repartition_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["link_type"], "copy")
            self.assertEqual(metadata["workers"], 2)

    def test_skip_existing_resumes_complete_files_and_recopies_truncated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            output = Path(tmp) / "output"
            self._create_input_dataset(root)

            base_command = [
                sys.executable,
                str(SCRIPT),
                "--input-root",
                str(root),
                "--output-root",
                str(output),
                "--ratio",
                "7:2:1",
                "--link-mode",
                "copy",
                "--workers",
                "2",
            ]
            subprocess.run(base_command, check=True)
            pair_paths = sorted((output / "cache").glob("**/pair_*.pt"))
            self.assertGreaterEqual(len(pair_paths), 2)
            keep_path = pair_paths[0]
            repair_path = pair_paths[1]
            keep_original_size = keep_path.stat().st_size
            keep_path.write_text("K" * keep_original_size, encoding="utf-8")
            repair_path.write_text("x", encoding="utf-8")

            subprocess.run(base_command + ["--skip-existing"], check=True)

            self.assertEqual(keep_path.read_text(encoding="utf-8"), "K" * keep_original_size)
            self.assertNotEqual(repair_path.read_text(encoding="utf-8"), "x")
            metadata = json.loads((output / "repartition_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["skip_existing"])


if __name__ == "__main__":
    unittest.main()
