from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import annotator_data


def write_image(path: Path, value: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 16), value, dtype=np.uint16)).save(path)


class AnnotatorDataTest(unittest.TestCase):
    def test_discovers_pair_from_two_images_in_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "case_001" / "left.tif", 20)
            write_image(root / "case_001" / "right.tif", 80)

            pairs = annotator_data.discover_pairs(root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].image_a, "case_001/left.tif")
            self.assertEqual(pairs[0].image_b, "case_001/right.tif")
            self.assertEqual(pairs[0].size_a, (16, 12))

    def test_discovers_pair_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "images" / "a.tif", 10)
            write_image(root / "images" / "b.tif", 90)
            (root / "pairs.json").write_text(
                json.dumps({"pairs": [{"id": "hard_case", "image_a": "images/a.tif", "image_b": "images/b.tif"}]}),
                encoding="utf-8",
            )

            pairs = annotator_data.discover_pairs(root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].pair_id, "hard_case")
            self.assertEqual(pairs[0].image_a, "images/a.tif")
            self.assertEqual(pairs[0].image_b, "images/b.tif")

    def test_makes_pair_from_explicit_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "manual" / "mars_left.tif", 10)
            write_image(root / "manual" / "mars_right.tif", 90)

            pair = annotator_data.make_image_pair(root, "manual/mars_left.tif", "manual/mars_right.tif")

            self.assertEqual(pair.image_a, "manual/mars_left.tif")
            self.assertEqual(pair.image_b, "manual/mars_right.tif")
            self.assertEqual(pair.name, "manual")
            self.assertEqual(pair.annotation_count, 0)
            self.assertTrue(pair.pair_id.startswith("manual_"))

    def test_discovers_saved_explicit_pair_from_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "a.tif", 10)
            write_image(root / "b.tif", 20)
            write_image(root / "c.tif", 30)
            annotator_data.save_annotation(
                root,
                {
                    "pair_id": "manual_ac",
                    "image_a": "a.tif",
                    "image_b": "c.tif",
                    "matches": [],
                },
            )

            pairs = annotator_data.discover_pairs(root)

            discovered = {(pair.pair_id, pair.image_a, pair.image_b) for pair in pairs}
            self.assertIn(("manual_ac", "a.tif", "c.tif"), discovered)

    def test_candidate_images_skip_annotation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "a.tif")
            write_image(root / "annotations" / "hidden.tif")

            images = [annotator_data.relative_path(root, path) for path in annotator_data.candidate_image_files(root)]

            self.assertEqual(images, ["a.tif"])

    def test_saves_annotation_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "a.tif")
            write_image(root / "b.tif")

            path = annotator_data.save_annotation(
                root,
                {
                    "pair_id": "pair_a",
                    "image_a": "a.tif",
                    "image_b": "b.tif",
                    "matches": [{"id": 7, "a": {"x": 1, "y": 2}, "b": {"x": 3.5, "y": 4.5}}],
                },
            )

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["pair_id"], "pair_a")
            self.assertEqual(saved["image_size_a"], [16, 12])
            self.assertEqual(saved["matches"][0]["b"]["x"], 3.5)

    def test_converts_16bit_tiff_for_display(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "source.tif", 2048)

            with Image.open(root / "source.tif") as image:
                display = annotator_data.to_display_image(image, max_side=64)

            self.assertEqual(display.mode, "L")
            self.assertLessEqual(max(display.size), 64)


if __name__ == "__main__":
    unittest.main()
