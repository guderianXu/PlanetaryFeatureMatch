import importlib.util
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "辅助软件" / "数据模拟" / "batch_pose_sim_dataset.py"


def load_batch_module():
    spec = importlib.util.spec_from_file_location("batch_pose_sim_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BatchPoseSimDatasetTests(unittest.TestCase):
    def test_ratio_split_assigns_exact_7_2_1_counts(self):
        module = load_batch_module()
        track = module.TrackInfo(
            stem="TrackA",
            split="train",
            csv_path=Path("TrackA.csv"),
            tsai_dir=Path("tsai"),
            frames=1,
        )
        candidates = [
            module.Candidate(index=index, track=track, seq=index + 1, sim_view="A_basic_xp5", split=track.split)
            for index in range(10)
        ]

        repartitioned = module.apply_ratio_split(candidates, ratio=(7, 2, 1), seed=123)

        counts = {split: sum(1 for candidate in repartitioned if candidate.split == split) for split in ("train", "val", "test")}
        self.assertEqual(counts, {"train": 7, "val": 2, "test": 1})

    def test_output_pair_path_uses_candidate_split(self):
        module = load_batch_module()
        track = module.TrackInfo(
            stem="TrackA",
            split="train",
            csv_path=Path("TrackA.csv"),
            tsai_dir=Path("tsai"),
            frames=1,
        )
        candidate = module.Candidate(index=3, track=track, seq=7, sim_view="A_ext_diag", split="test")
        with tempfile.TemporaryDirectory() as tmp:
            path = module.output_pair_path(Path(tmp), candidate)
        self.assertIn("/cache/test/", path.as_posix())
        self.assertTrue(path.name.startswith("pair_000003_ext_diag"))

    def test_select_candidates_applies_ratio_after_range_limit(self):
        module = load_batch_module()
        track = module.TrackInfo(
            stem="TrackA",
            split="train",
            csv_path=Path("TrackA.csv"),
            tsai_dir=Path("tsai"),
            frames=1,
        )
        candidates = [
            module.Candidate(index=index, track=track, seq=index + 1, sim_view="A_basic_xp5", split=track.split)
            for index in range(100)
        ]

        selected = module.select_generation_candidates(
            candidates,
            start_pair_index=0,
            max_pairs=10,
            split_mode="ratio",
            split_ratio=(7, 2, 1),
            split_seed=123,
        )

        counts = {split: sum(1 for candidate in selected if candidate.split == split) for split in ("train", "val", "test")}
        self.assertEqual(len(selected), 10)
        self.assertEqual(counts, {"train": 7, "val": 2, "test": 1})

    def test_parse_args_accepts_frame_workers_and_ratio_split(self):
        module = load_batch_module()
        original_argv = sys.argv
        try:
            sys.argv = [
                "batch_pose_sim_dataset.py",
                "--output-root",
                "/tmp/out",
                "--split-mode",
                "ratio",
                "--split-ratio",
                "7:2:1",
                "--frame-workers",
                "3",
            ]
            args = module.parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.split_mode, "ratio")
        self.assertEqual(args.split_ratio, (7, 2, 1))
        self.assertEqual(args.frame_workers, 3)

    def test_parse_args_accepts_explicit_dem_and_dom(self):
        module = load_batch_module()
        original_argv = sys.argv
        try:
            sys.argv = [
                "batch_pose_sim_dataset.py",
                "--output-root",
                "/tmp/out",
                "--dem",
                "/tmp/dem/mar.tif",
                "--dom",
                "/tmp/dom/mar_dom.tif",
            ]
            args = module.parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.dem, Path("/tmp/dem/mar.tif"))
        self.assertEqual(args.dom, Path("/tmp/dom/mar_dom.tif"))


if __name__ == "__main__":
    unittest.main()
