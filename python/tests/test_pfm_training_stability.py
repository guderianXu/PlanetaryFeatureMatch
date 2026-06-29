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

    def test_tracker_requests_stop_on_dustbin_rejection_spike(self):
        tracker = TrainingStabilityTracker(
            thresholds=StabilityThresholds(
                min_steps_before_early_stop=3,
                rolling_window=3,
                max_nan_in_window=3,
                max_loss_multiplier=3.0,
                min_top1_mean=0.4,
                max_dustbin_rejection_ratio=0.85,
            )
        )

        decision = None
        for step in range(1, 4):
            decision = tracker.update(
                step,
                {
                    "loss": 4.0,
                    "top1_accuracy": 0.9,
                    "true_match_rejected_by_dustbin_ratio": 0.9,
                },
            )

        self.assertIsNotNone(decision)
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.reason, "dustbin_rejection_spike")

    def test_tracker_requests_stop_when_filtered_matches_collapse(self):
        tracker = TrainingStabilityTracker(
            thresholds=StabilityThresholds(
                min_steps_before_early_stop=3,
                rolling_window=3,
                max_nan_in_window=3,
                max_loss_multiplier=3.0,
                min_top1_mean=0.4,
                min_num_filtered_matches=16,
            )
        )

        decision = None
        for step in range(1, 4):
            decision = tracker.update(
                step,
                {
                    "loss": 4.0,
                    "top1_accuracy": 0.9,
                    "num_filtered_matches": 4,
                },
            )

        self.assertIsNotNone(decision)
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.reason, "num_filtered_matches_collapse")

    def test_tracker_does_not_stop_on_small_absolute_loss_spikes_when_matching_is_healthy(self):
        tracker = TrainingStabilityTracker(
            thresholds=StabilityThresholds(
                min_steps_before_early_stop=5,
                rolling_window=4,
                max_nan_in_window=3,
                max_loss_multiplier=3.0,
                min_top1_mean=0.4,
            )
        )

        for step in range(1, 5):
            tracker.update(
                step,
                {
                    "loss": 0.001,
                    "top1_accuracy": 0.98,
                    "true_match_rejected_by_dustbin_ratio": 0.0,
                    "positive_vs_dustbin_margin_mean": 12.0,
                },
            )

        decision = None
        for step in range(5, 9):
            decision = tracker.update(
                step,
                {
                    "loss": 0.02,
                    "top1_accuracy": 0.95,
                    "true_match_rejected_by_dustbin_ratio": 0.0,
                    "positive_vs_dustbin_margin_mean": 12.0,
                },
            )

        self.assertIsNotNone(decision)
        self.assertFalse(decision.should_stop)

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

    def test_tracker_requests_last_good_save_only_on_score_improvement(self):
        tracker = TrainingStabilityTracker(
            thresholds=StabilityThresholds(
                min_steps_before_early_stop=5,
                rolling_window=3,
                max_nan_in_window=2,
                max_loss_multiplier=3.0,
                min_top1_mean=0.2,
            )
        )

        first = tracker.update(1, {"loss": 5.0, "top1_accuracy": 0.8})
        same_score = tracker.update(2, {"loss": 4.5, "top1_accuracy": 0.8})
        worse_score = tracker.update(3, {"loss": 4.0, "top1_accuracy": 0.7})
        better_score = tracker.update(4, {"loss": 4.0, "top1_accuracy": 0.9})

        self.assertTrue(first.should_save_last_good)
        self.assertFalse(same_score.should_save_last_good)
        self.assertFalse(worse_score.should_save_last_good)
        self.assertTrue(better_score.should_save_last_good)

    def test_tracker_exposes_rolling_diagnostics_for_training_logs(self):
        tracker = TrainingStabilityTracker(
            thresholds=StabilityThresholds(
                min_steps_before_early_stop=10,
                rolling_window=3,
                max_nan_in_window=2,
                max_loss_multiplier=3.0,
                min_top1_mean=0.2,
            )
        )
        tracker.update(1, {"loss": 3.0, "top1_accuracy": 0.8})
        tracker.update(2, {"loss": math.nan, "top1_accuracy": 0.2})
        tracker.update(3, {"loss": 1.0, "top1_accuracy": 0.5})

        diagnostics = tracker.rolling_diagnostics()

        self.assertEqual(diagnostics["nan_count"], 1)
        self.assertAlmostEqual(diagnostics["recent_loss_mean"], 2.0)
        self.assertAlmostEqual(diagnostics["recent_top1_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
