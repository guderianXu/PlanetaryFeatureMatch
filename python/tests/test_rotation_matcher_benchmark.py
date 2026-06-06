import csv
from unittest import mock
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rotation_matcher_benchmark as bench


class RotationMatcherBenchmarkTest(unittest.TestCase):
    def test_rotate_points_matches_numpy_rot90_geometry(self):
        points = np.array(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [0.0, 2.0],
                [3.0, 2.0],
            ],
            dtype=np.float32,
        )

        rotated_90 = bench.rotate_points(points, width=4, height=3, angle=90)
        rotated_180 = bench.rotate_points(points, width=4, height=3, angle=180)
        rotated_270 = bench.rotate_points(points, width=4, height=3, angle=270)

        np.testing.assert_allclose(rotated_90, np.array([[0, 3], [0, 0], [2, 3], [2, 0]], dtype=np.float32))
        np.testing.assert_allclose(rotated_180, np.array([[3, 2], [0, 2], [3, 0], [0, 0]], dtype=np.float32))
        np.testing.assert_allclose(rotated_270, np.array([[2, 0], [2, 3], [0, 0], [0, 3]], dtype=np.float32))

    def test_write_metrics_csv_uses_stable_fields(self):
        row = bench.ResultRow(
            image_style="numeric",
            image_path="img/1.tif",
            angle=90,
            matcher="SIFT",
            status="ok",
            keypoints_a=12,
            keypoints_b=14,
            matches=5,
            correct=4,
            wrong=1,
            precision=0.8,
            mean_error_px=1.25,
            median_error_px=1.0,
            visualization="visualizations/numeric_90_SIFT.png",
            message="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            bench.write_metrics_csv(path, [row])

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(bench.CSV_FIELDS, list(rows[0].keys()))
        self.assertEqual(rows[0]["matcher"], "SIFT")
        self.assertEqual(rows[0]["precision"], "0.800000")
        self.assertEqual(rows[0]["mean_error_px"], "1.250000")

    def test_rootsift_descriptor_normalization_l1_then_sqrt(self):
        descriptors = np.array([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

        rootsift = bench.normalize_sift_descriptors_to_rootsift(descriptors)

        expected = np.array([[0.5, np.sqrt(0.75), 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(rootsift, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(rootsift.dtype, np.float32)

    def test_filter_points_with_homography_usac_rejects_outlier(self):
        if bench.importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is not installed")
        cv2 = __import__("cv2")
        if not hasattr(cv2, "USAC_MAGSAC"):
            self.skipTest("OpenCV USAC_MAGSAC is not available")
        inliers_a = np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
                [10.0, 10.0],
                [5.0, 5.0],
                [20.0, 0.0],
                [0.0, 20.0],
                [20.0, 20.0],
            ],
            dtype=np.float32,
        )
        points_a = np.concatenate([inliers_a, np.array([[30.0, 30.0], [40.0, 40.0]], dtype=np.float32)], axis=0)
        points_b = np.concatenate(
            [inliers_a + np.array([[2.0, 3.0]], dtype=np.float32), np.array([[100.0, 100.0], [120.0, 10.0]], dtype=np.float32)],
            axis=0,
        )

        filtered_a, filtered_b = bench.filter_points_with_homography_usac(
            points_a,
            points_b,
            method=cv2.USAC_MAGSAC,
            reprojection_threshold_px=1.0,
            max_matches=16,
        )

        self.assertEqual(filtered_a.shape, (8, 2))
        self.assertEqual(filtered_b.shape, (8, 2))
        np.testing.assert_allclose(filtered_b - filtered_a, np.tile([[2.0, 3.0]], (8, 1)), atol=1e-4)

    def test_make_opencv_matchers_includes_rootsift_flann_ransac_when_sift_available(self):
        if bench.importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is not installed")

        matchers = bench.make_opencv_matchers(max_matches=16)

        by_name = {matcher.name: matcher for matcher in matchers}
        self.assertIn("RootSIFT-FLANN-RANSAC", by_name)
        self.assertNotIsInstance(by_name["RootSIFT-FLANN-RANSAC"], bench.UnavailableMatcher)

    def test_make_opencv_matchers_includes_rootsift_usac_when_available(self):
        if bench.importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is not installed")
        cv2 = __import__("cv2")
        if not hasattr(cv2, "SIFT_create") or not hasattr(cv2, "USAC_MAGSAC"):
            self.skipTest("OpenCV SIFT/USAC_MAGSAC is not available")

        matchers = bench.make_opencv_matchers(max_matches=16)

        by_name = {matcher.name: matcher for matcher in matchers}
        self.assertIn("RootSIFT-FLANN-USAC-MAGSAC", by_name)
        self.assertNotIsInstance(by_name["RootSIFT-FLANN-USAC-MAGSAC"], bench.UnavailableMatcher)

    def test_make_opencv_matchers_includes_affine_sift_when_available(self):
        if bench.importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is not installed")
        cv2 = __import__("cv2")
        if not hasattr(cv2, "AffineFeature_create") or not hasattr(cv2, "SIFT_create"):
            self.skipTest("OpenCV AffineFeature/SIFT is not available")

        matchers = bench.make_opencv_matchers(max_matches=16)

        by_name = {matcher.name: matcher for matcher in matchers}
        self.assertIn("AffineSIFT-BF", by_name)
        self.assertNotIsInstance(by_name["AffineSIFT-BF"], bench.UnavailableMatcher)

    def test_unavailable_matcher_returns_row_without_raising(self):
        matcher = bench.UnavailableMatcher("LightGlue", "module not installed")
        image = np.zeros((8, 8), dtype=np.uint8)

        row, points_a, points_b = bench.evaluate_matcher_on_rotation(
            matcher,
            image,
            image,
            image_style="numeric",
            image_path=Path("img/1.tif"),
            angle=90,
            output_dir=Path("runs/test"),
            threshold_px=3.0,
        )

        self.assertEqual(row.matcher, "LightGlue")
        self.assertEqual(row.status, "unavailable")
        self.assertEqual(row.matches, 0)
        self.assertEqual(row.message, "module not installed")
        self.assertEqual(points_a.shape, (0, 2))
        self.assertEqual(points_b.shape, (0, 2))

    def test_make_optional_deep_matchers_uses_lightglue_when_installed(self):
        def fake_find_spec(module):
            if module == "lightglue":
                return object()
            return None

        with mock.patch.object(bench.importlib.util, "find_spec", side_effect=fake_find_spec):
            matchers = bench.make_optional_deep_matchers(device="cpu", max_keypoints=128, max_matches=32)

        by_name = {matcher.name: matcher for matcher in matchers}
        self.assertIn("LightGlue-SIFT", by_name)
        self.assertIn("LightGlue-SuperPoint", by_name)
        self.assertIn("LightGlue-DISK", by_name)
        self.assertIn("LightGlue-ALIKED", by_name)
        self.assertNotIsInstance(by_name["LightGlue-SIFT"], bench.UnavailableMatcher)
        self.assertNotIsInstance(by_name["LightGlue-SuperPoint"], bench.UnavailableMatcher)
        self.assertNotIsInstance(by_name["LightGlue-DISK"], bench.UnavailableMatcher)
        self.assertNotIsInstance(by_name["LightGlue-ALIKED"], bench.UnavailableMatcher)
        self.assertIsInstance(by_name["LoFTR"], bench.UnavailableMatcher)
        self.assertIsInstance(by_name["SuperGlue"], bench.UnavailableMatcher)

    def test_lightglue_disk_and_aliked_load_use_feature_specific_entrypoints(self):
        calls = []

        class FakeModule:
            def eval(self):
                return self

            def to(self, device):
                self.device = device
                return self

        class FakeDisk(FakeModule):
            def __init__(self, *, max_num_keypoints):
                self.max_num_keypoints = max_num_keypoints
                calls.append(("DISK", max_num_keypoints))

        class FakeAliked(FakeModule):
            def __init__(self, *, max_num_keypoints):
                self.max_num_keypoints = max_num_keypoints
                calls.append(("ALIKED", max_num_keypoints))

        class FakeLightGlue(FakeModule):
            def __init__(self, *, features):
                self.features = features
                calls.append(("LightGlue", features))

        fake_lightglue = types.SimpleNamespace(DISK=FakeDisk, ALIKED=FakeAliked, LightGlue=FakeLightGlue)
        fake_torch = types.SimpleNamespace()

        with mock.patch.dict(sys.modules, {"lightglue": fake_lightglue, "torch": fake_torch}):
            disk_matcher = bench.LightGlueDiskMatcher(device="cpu", max_keypoints=123, max_matches=5)
            disk_extractor, disk_lightglue = disk_matcher._load()
            aliked_matcher = bench.LightGlueAlikedMatcher(device="cpu", max_keypoints=456, max_matches=7)
            aliked_extractor, aliked_lightglue = aliked_matcher._load()

        self.assertEqual(disk_extractor.max_num_keypoints, 123)
        self.assertEqual(disk_lightglue.features, "disk")
        self.assertEqual(aliked_extractor.max_num_keypoints, 456)
        self.assertEqual(aliked_lightglue.features, "aliked")
        self.assertEqual(
            calls,
            [
                ("DISK", 123),
                ("LightGlue", "disk"),
                ("ALIKED", 456),
                ("LightGlue", "aliked"),
            ],
        )

    def test_make_optional_deep_matchers_uses_loftr_when_kornia_is_installed(self):
        def fake_find_spec(module):
            if module == "kornia.feature":
                return object()
            return None

        with mock.patch.object(bench.importlib.util, "find_spec", side_effect=fake_find_spec):
            matchers = bench.make_optional_deep_matchers(device="cpu", max_keypoints=128, max_matches=32)

        by_name = {matcher.name: matcher for matcher in matchers}
        self.assertIn("LoFTR", by_name)
        self.assertNotIsInstance(by_name["LoFTR"], bench.UnavailableMatcher)
        self.assertIsInstance(by_name["LightGlue-SIFT"], bench.UnavailableMatcher)
        self.assertIsInstance(by_name["SuperGlue"], bench.UnavailableMatcher)

    def test_loftr_matcher_sorts_by_confidence_and_limits_matches(self):
        if bench.importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")
        torch = __import__("torch")

        class FakeLoFTR:
            def __init__(self):
                self.seen = None

            def __call__(self, data):
                self.seen = data
                return {
                    "keypoints0": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                    "keypoints1": torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]),
                    "confidence": torch.tensor([0.2, 0.9, 0.4]),
                    "batch_indexes": torch.tensor([0, 0, 0]),
                }

        fake = FakeLoFTR()
        matcher = bench.KorniaLoFTRMatcher(device="cpu", max_matches=2, pretrained="outdoor")
        with mock.patch.object(matcher, "_load", return_value=fake):
            output = matcher.match(np.zeros((8, 9), dtype=np.uint8), np.ones((8, 9), dtype=np.uint8))

        self.assertEqual(fake.seen["image0"].shape, (1, 1, 8, 9))
        self.assertEqual(output.keypoints_a, 3)
        self.assertEqual(output.keypoints_b, 3)
        np.testing.assert_allclose(output.points_a, np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
        np.testing.assert_allclose(output.points_b, np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float32))

    def test_lightglue_superpoint_matcher_sorts_by_score_and_limits_matches(self):
        if bench.importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")
        torch = __import__("torch")

        class FakeExtractor:
            def __init__(self):
                self.seen_shapes = []
                self.outputs = [
                    {"keypoints": torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])},
                    {"keypoints": torch.tensor([[[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]])},
                ]

            def extract(self, image):
                self.seen_shapes.append(tuple(image.shape))
                return self.outputs.pop(0)

        class FakeLightGlue:
            def __init__(self):
                self.seen = None

            def __call__(self, data):
                self.seen = data
                return {
                    "matches": [torch.tensor([[0, 0], [1, 1], [2, 2]])],
                    "scores": [torch.tensor([0.2, 0.9, 0.4])],
                }

        extractor = FakeExtractor()
        lightglue = FakeLightGlue()
        fake_lightglue = types.ModuleType("lightglue")
        fake_lightglue_utils = types.ModuleType("lightglue.utils")
        fake_lightglue_utils.numpy_image_to_torch = lambda image: torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
        matcher = bench.LightGlueSuperPointMatcher(device="cpu", max_keypoints=128, max_matches=2)
        with mock.patch.dict(
            sys.modules,
            {
                "lightglue": fake_lightglue,
                "lightglue.utils": fake_lightglue_utils,
            },
        ), mock.patch.object(matcher, "_load", return_value=(extractor, lightglue)):
            output = matcher.match(np.zeros((8, 9), dtype=np.uint8), np.ones((8, 9), dtype=np.uint8))

        self.assertEqual(extractor.seen_shapes, [(1, 8, 9), (1, 8, 9)])
        self.assertIn("image0", lightglue.seen)
        self.assertIn("image1", lightglue.seen)
        self.assertEqual(output.keypoints_a, 3)
        self.assertEqual(output.keypoints_b, 3)
        np.testing.assert_allclose(output.points_a, np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
        np.testing.assert_allclose(output.points_b, np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
