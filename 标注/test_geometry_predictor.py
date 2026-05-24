from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry_predictor


def close_point(testcase: unittest.TestCase, actual: dict[str, float], expected: dict[str, float], tolerance: float = 1.0e-4) -> None:
    testcase.assertAlmostEqual(actual["x"], expected["x"], delta=tolerance)
    testcase.assertAlmostEqual(actual["y"], expected["y"], delta=tolerance)


def apply_homography(point: dict[str, float]) -> dict[str, float]:
    denominator = 0.001 * point["x"] + 0.0007 * point["y"] + 1.0
    return {
        "x": (1.2 * point["x"] + 0.15 * point["y"] + 20.0) / denominator,
        "y": (0.08 * point["x"] + 0.9 * point["y"] + 35.0) / denominator,
    }


class GeometryPredictorTest(unittest.TestCase):
    def test_predicts_by_scale_without_existing_matches(self) -> None:
        prediction = geometry_predictor.predict_match_point({"x": 25, "y": 20}, [], (100, 80), (200, 160))
        close_point(self, prediction.point, {"x": 50, "y": 40})
        self.assertEqual(prediction.method, "scale")

    def test_predicts_by_translation_from_one_match(self) -> None:
        prediction = geometry_predictor.predict_match_point(
            {"x": 8, "y": 9},
            [{"a": {"x": 2, "y": 3}, "b": {"x": 12, "y": 23}}],
            (100, 100),
            (100, 100),
        )
        close_point(self, prediction.point, {"x": 18, "y": 29})
        self.assertEqual(prediction.method, "translation")

    def test_predicts_by_affine_from_three_matches(self) -> None:
        matches = [
            {"a": {"x": 0, "y": 0}, "b": {"x": 10, "y": 20}},
            {"a": {"x": 10, "y": 0}, "b": {"x": 30, "y": 20}},
            {"a": {"x": 0, "y": 10}, "b": {"x": 10, "y": 50}},
        ]
        prediction = geometry_predictor.predict_match_point({"x": 4, "y": 5}, matches, (100, 100), (200, 200))
        close_point(self, prediction.point, {"x": 18, "y": 35})
        self.assertEqual(prediction.method, "affine")

    def test_predicts_by_homography_from_four_or_more_matches(self) -> None:
        sources = [
            {"x": 0, "y": 0},
            {"x": 100, "y": 0},
            {"x": 0, "y": 100},
            {"x": 100, "y": 100},
            {"x": 50, "y": 20},
            {"x": 20, "y": 70},
        ]
        matches = [{"a": point, "b": apply_homography(point)} for point in sources]
        query = {"x": 40, "y": 30}
        prediction = geometry_predictor.predict_match_point(query, matches, (120, 120), (200, 200))
        close_point(self, prediction.point, apply_homography(query), tolerance=2.0e-2)
        self.assertEqual(prediction.method, "homography")


if __name__ == "__main__":
    unittest.main()
