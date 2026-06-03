import tempfile
import unittest
from pathlib import Path

import cache_split


class CacheSplitTest(unittest.TestCase):
    def test_source_name_from_dir_strips_prefix_and_preserves_underscores(self):
        self.assertEqual(
            cache_split.source_name_from_dir(Path("source_000108_20260514T143135909_NAS_PAN_L2b")),
            "20260514T143135909_NAS_PAN_L2b",
        )
        self.assertEqual(cache_split.source_name_from_dir(Path("source_000010_108")), "108")

    def test_source_style_separates_numeric_and_timestamp_sources(self):
        self.assertEqual(cache_split.source_style("108"), "numeric")
        self.assertEqual(cache_split.source_style("20260514T064636672_NAS_PAN_L2b"), "timestamp")
        self.assertEqual(cache_split.source_style("custom_scene_name"), "timestamp")

    def test_split_source_names_is_deterministic_and_source_disjoint(self):
        sources = [f"{index:03d}" for index in range(20)]

        first = cache_split.split_source_names(sources, train_ratio=0.6, val_ratio=0.2, seed=7)
        second = cache_split.split_source_names(reversed(sources), train_ratio=0.6, val_ratio=0.2, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(sources))
        self.assertEqual(sum(1 for split in first.values() if split == "train"), 12)
        self.assertEqual(sum(1 for split in first.values() if split == "val"), 4)
        self.assertEqual(sum(1 for split in first.values() if split == "test"), 4)

    def test_discover_cache_sources_and_create_split_dirs_group_by_split_style_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rotate = root / "Rotate_1024"
            viewpoint = root / "Viewpoint_1024"
            for cache_root in (rotate, viewpoint):
                for source_dir_name in ("source_000001_1", "source_000002_20260514T064636672_NAS_PAN_L2b"):
                    source_dir = cache_root / source_dir_name
                    source_dir.mkdir(parents=True)
                    (source_dir / "pair_000001.pt").write_text("pair", encoding="utf-8")

            sources = cache_split.discover_cache_sources([rotate, viewpoint])
            assignments = {
                "1": "train",
                "20260514T064636672_NAS_PAN_L2b": "test",
            }
            created = cache_split.create_split_cache_dirs(
                sources,
                assignments,
                root / "splits",
            )

            train_rotate = created[("train", "numeric", "rotate")]
            test_viewpoint = created[("test", "timestamp", "viewpoint")]
            self.assertTrue((train_rotate / "source_000001_1").is_symlink())
            self.assertTrue((test_viewpoint / "source_000002_20260514T064636672_NAS_PAN_L2b").is_symlink())
            self.assertEqual(
                sorted(path.name for path in train_rotate.glob("source_*/pair_*.pt")),
                ["pair_000001.pt"],
            )
            self.assertEqual(
                sorted(path.name for path in test_viewpoint.glob("source_*/pair_*.pt")),
                ["pair_000001.pt"],
            )


if __name__ == "__main__":
    unittest.main()
