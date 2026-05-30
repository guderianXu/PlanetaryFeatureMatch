import tempfile
import unittest
from pathlib import Path

import pose_pair_metadata as pose


def write_tsai(path: Path, *, fu: float, center: tuple[float, float, float], r_values: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "VERSION_3",
                f"fu = {fu}",
                f"fv = {fu}",
                "cu = 1024",
                "cv = 1024",
                "u_direction = 1 0 0",
                "v_direction = 0 1 0",
                "w_direction = 0 0 1",
                f"C = {center[0]} {center[1]} {center[2]}",
                f"R = {r_values}",
                "pitch = 1",
                "NULL",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class PosePairMetadataTest(unittest.TestCase):
    def test_load_pose_metadata_index_reads_manifest_and_tsai_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_path = root / "cache" / "train" / "source_00001_TrackA" / "pair_000000_basic_xp5.pt"
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            pair_path.write_bytes(b"archive")
            (root / "manifests").mkdir()
            (root / "manifests" / "first.csv").write_text(
                "candidate_index,split,track,seq,sim_view,pair_path,valid_pair_fraction,valid_pixels\n"
                f"0,train,TrackA,1,A_basic_xp5,{pair_path},0.75,123\n",
                encoding="utf-8",
            )
            track_root = root / "tsai_tracks" / "TrackA_gap30_views10_2048" / "tsai"
            write_tsai(
                track_root / "CameraA" / "00001.tsai",
                fu=1000.0,
                center=(0.0, 0.0, 0.0),
                r_values="1 0 0 0 1 0 0 0 1",
            )
            write_tsai(
                track_root / "CameraA_basic_xp5" / "00001.tsai",
                fu=1200.0,
                center=(3.0, 4.0, 0.0),
                r_values="1 0 0 0 0.996194698 0.087155743 0 -0.087155743 0.996194698",
            )

            index = pose.load_pose_metadata_index(root)
            metadata = pose.lookup_pose_metadata(index, pair_path)
            copied_path = (
                Path("/local/copy/pose_sim")
                / "cache"
                / "train"
                / "source_00001_TrackA"
                / "pair_000000_basic_xp5.pt"
            )
            copied_metadata = pose.lookup_pose_metadata(index, copied_path)

        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertIs(metadata, copied_metadata)
        self.assertAlmostEqual(metadata.baseline_m, 5.0)
        self.assertAlmostEqual(metadata.focal_ratio, 1.2)
        self.assertAlmostEqual(metadata.overlap_fraction, 0.75)
        self.assertAlmostEqual(metadata.view_angle_deg, 5.0, places=3)
        self.assertEqual(metadata.difficulty, "medium")
        self.assertEqual(metadata.difficulty_score, 0.5)

    def test_infer_pose_metadata_roots_from_split_cache_dirs(self):
        root = Path("/dataset/pose_sim")

        inferred = pose.infer_pose_metadata_roots([root / "cache" / "train"], [])

        self.assertEqual(inferred, [root])


if __name__ == "__main__":
    unittest.main()
