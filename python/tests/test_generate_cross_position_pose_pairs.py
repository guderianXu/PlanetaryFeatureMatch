import tempfile
import unittest
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import generate_cross_position_pose_pairs as cross_pairs  # noqa: E402
from compact_pair_cache import make_compact_pair_payload, save_shared_image  # noqa: E402


class CrossPositionSourceDiscoveryTest(unittest.TestCase):
    def test_discover_sources_accepts_repartition_dirs_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for repart_index in (1, 2):
                source = (
                    root
                    / "cache"
                    / "train"
                    / f"source_repart_{repart_index:06d}_val_source_00150_TrackA"
                )
                source.mkdir(parents=True)
                (source / f"pair_{repart_index:06d}_basic_xp5.pt").write_bytes(b"archive")
            tsai = root / "tsai_tracks" / "TrackA_gap30_views10_2048" / "tsai" / "CameraA" / "00150.tsai"
            tsai.parent.mkdir(parents=True)
            tsai.write_text(
                "\n".join(
                    [
                        "VERSION_3",
                        "fu = 1000",
                        "fv = 1000",
                        "cu = 1024",
                        "cv = 1024",
                        "C = 0 0 0",
                        "R = 1 0 0 0 1 0 0 0 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            records = cross_pairs.discover_sources(root, "train")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].seq, 150)
        self.assertEqual(records[0].track, "TrackA")

    def test_load_archive_reads_compact_pair_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_path = root / "cache" / "train" / "source_000001_TrackA" / "pair_000001_basic_xp5.pt"
            pair_path.parent.mkdir(parents=True)
            image_store = root / "image_store"
            view_a = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 16.0
            view_b = torch.flip(view_a, dims=[2])
            warp = torch.zeros(4, 4, 2, dtype=torch.float32)
            valid = torch.ones(4, 4, dtype=torch.bool)
            image_a = save_shared_image(view_a, image_store)
            image_b = save_shared_image(view_b, image_store)
            torch.save(
                make_compact_pair_payload(
                    pair_path=pair_path,
                    image_a_path=image_a,
                    image_b_path=image_b,
                    warp_a_to_b=warp,
                    valid_mask=valid,
                ),
                pair_path,
            )

            archive = cross_pairs.load_archive(pair_path)

        self.assertTrue(torch.equal(archive.view_a, view_a))
        self.assertTrue(torch.equal(archive.view_b, view_b))
        self.assertTrue(torch.equal(archive.warp_a_to_b, warp))
        self.assertTrue(torch.equal(archive.valid_mask, valid))


if __name__ == "__main__":
    unittest.main()
