import math
import unittest

from pfm_training_stability import RollingMetricWindow, StabilityThresholds, TrainingStabilityTracker


class PFMTrainingStabilityTest(unittest.TestCase):
    def test_rolling_metric_window_ignores_nonfinite_values(self):
        window = RollingMetricWindow(size=3)
        window.add({"loss": 5.0, "top1_accuracy": 0.9})
        window.add({"loss": float("nan"), "top1_accuracy": 0.1})
        window.add({"loss": 4.0, "top1_accuracy": 0.8})
        window.add({"loss": 3.0, "top1_accuracy": 0.7})

        self.assertEqual(window.count, 3)
        self.assertEqual(window.nonfinite_count("loss"), 1)
        self.assertAlmostEqual(window.mean("loss"), 3.5)
        self.assertAlmostEqual(window.mean("top1_accuracy"), (0.1 + 0.8 + 0.7) / 3.0)

    def test_match_score_penalizes_dustbin_rejection_and_nan(self):
        tracker = TrainingStabilityTracker(
            thresholds=StabilityThresholds(
                min_steps_before_early_stop=100,
                rolling_window=5,
                max_nan_in_window=3,
                max_loss_multiplier=3.0,
                min_top1_mean=0.25,
            )
        )

        score = tracker.match_score(
            {
                "top1_accuracy": 0.8,
                "true_match_rejected_by_dustbin_ratio": 0.6,
                "false_match_accepted_ratio": 0.2,
                "positive_vs_dustbin_margin_mean": -0.5,
                "loss": float("nan"),
            }
        )

        self.assertLess(score, 0.0)

    def test_tracker_requests_stop_after_warmup_when_loss_and_top1_collapse(self):
        tracker = TrainingStabilityTracker(
            thresholds=StabilityThresholds(
                min_steps_before_early_stop=5,
                rolling_window=4,
                max_nan_in_window=2,
                max_loss_multiplier=2.0,
                min_top1_mean=0.4,
            )
        )
        for step in range(1, 6):
            tracker.update(step, {"loss": 4.0, "top1_accuracy": 0.9})
        for step in range(6, 10):
            decision = tracker.update(step, {"loss": 20.0, "top1_accuracy": 0.1})

        self.assertTrue(decision.should_stop)
        self.assertIn("top1", decision.reason)

    def test_tracker_marks_last_good_only_for_finite_reasonable_steps(self):
        tracker = TrainingStabilityTracker(
            thresholds=StabilityThresholds(
                min_steps_before_early_stop=5,
                rolling_window=3,
                max_nan_in_window=2,
                max_loss_multiplier=3.0,
                min_top1_mean=0.2,
            )
        )

        good = tracker.update(1, {"loss": 5.0, "top1_accuracy": 0.8})
        bad = tracker.update(2, {"loss": math.nan, "top1_accuracy": 0.0})

        self.assertTrue(good.is_last_good)
        self.assertFalse(bad.is_last_good)


if __name__ == "__main__":
    unittest.main()
