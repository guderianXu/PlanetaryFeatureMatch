import tempfile
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_cross_position_pose_pairs as cross_pairs  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
