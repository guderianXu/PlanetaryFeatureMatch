# Stable Graph Matcher Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the next PFM training run trustworthy by adding best-checkpoint selection, bad-step protection, graph-depth consistency, reduced rejection defaults, and diagnostics for true-match-vs-dustbin failure.

**Architecture:** First stabilize the existing lazy training pipeline instead of changing the extractor. Add a small training-stability module for rolling metrics and checkpoint decisions, add GraphMatcher diagnostics at the loss source, then expose explicit model graph-depth/hidden-dim controls so a 2-layer training run is also a 2-layer inference/eval run. Geometry pooling, reliability, texture fusion, and extractor changes are left as later ablation phases after this foundation is reliable.

**Tech Stack:** Python 3.12, PyTorch, argparse, csv/json/pathlib, existing `scripts/benchmark_lazy_pose_pairs.py`, `scripts/visualize_lazy_pose_matches.py`, `python/pfm_model.py`, `python/pfm_pytorch_training.py`, unittest.

---

## File Structure

- Create `python/pfm_training_stability.py`
  - Rolling metric windows.
  - Finite / NaN counters.
  - Match-score computation.
  - Early-stop and rollback decisions.
  - JSON-serializable crash/stability summaries.
- Create `python/tests/test_pfm_training_stability.py`
  - Unit tests for match-score, rolling windows, NaN protection, and early-stop decisions.
- Modify `python/pfm_pytorch_training.py`
  - Add GraphMatcher dustbin diagnostics from the same logits used by the loss.
  - Return diagnostics in `train_step()` metrics.
  - Keep changes model-agnostic so both direct cache training and lazy training can use them.
- Modify `python/pfm_model.py`
  - Add checkpoint loading with graph architecture overrides and shape-safe partial loading.
  - Allow initializing a 2-layer/256-hidden GraphMatcher while transferring compatible extractor weights from an older checkpoint.
- Modify `scripts/benchmark_lazy_pose_pairs.py`
  - Add CLI args for graph architecture override, stability thresholds, reduced rejection preset, best checkpoint saving, and visual eval on best checkpoint.
  - Add new metric fields.
  - Save `latest.pt`, `last_good.pt`, `best_by_loss.pt`, `best_by_match_score.pt`, and `crash_report.json`.
- Modify `scripts/visualize_lazy_pose_matches.py`
  - Add `--graph-max-attention-layers`, `--graph-max-attention-work-fraction`, and `--graph-width-prune-keep-ratio` to evaluate the same checkpoint at different inference depths.
- Create `scripts/run_graph_depth_ablation.py`
  - Run P1-A inference-depth sweep on a fixed checkpoint and fixed lazy pair set.
  - Aggregate raw and filtered visual report CSVs into one HTML/CSV summary.
- Modify `scripts/README.md`
  - Document the new depth ablation script and stable training controls.
- Modify `runs/train_h100_fov090_spatial_cross_camera_sample_20260608.sh`
  - Add environment-controlled stable defaults for the next run without hard-coding local one-off values into source code.

---

## Phase 1 Scope

Implement now:

- P0: best checkpoint, bad-step protection, auto visual eval on best checkpoint, key diagnostics.
- P1-lite: enforce training/inference graph-depth consistency and support 2-layer/256-hidden model loading.
- P2: reduced rejection/dustbin stable preset and optional ramp.

Do not implement yet:

- Geometry canonical pooling ablation.
- Reliability signal ablation.
- Texture fusion ablation.
- Stage1 keypoint skip / extractor redesign.

Those require stable checkpointing and diagnostics first.

---

### Task 1: Add Training Stability Unit Tests

**Files:**
- Create: `python/tests/test_pfm_training_stability.py`
- Later implementation target: `python/pfm_training_stability.py`

- [ ] **Step 1: Write tests for rolling metrics and finite handling**

Create `python/tests/test_pfm_training_stability.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_training_stability
```

Expected: fail with `ModuleNotFoundError: No module named 'pfm_training_stability'`.

---

### Task 2: Implement Training Stability Helper

**Files:**
- Create: `python/pfm_training_stability.py`
- Test: `python/tests/test_pfm_training_stability.py`

- [ ] **Step 1: Add minimal implementation**

Create `python/pfm_training_stability.py`:

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import fmean
from typing import Deque


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class StabilityThresholds:
    min_steps_before_early_stop: int = 1000
    rolling_window: int = 200
    max_nan_in_window: int = 20
    max_loss_multiplier: float = 3.0
    min_top1_mean: float = 0.35
    min_match_score: float = -0.5


@dataclass(frozen=True)
class StabilityDecision:
    should_stop: bool
    should_save_latest: bool
    should_save_last_good: bool
    is_last_good: bool
    reason: str = ""


class RollingMetricWindow:
    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self._rows: Deque[dict[str, object]] = deque(maxlen=int(size))

    @property
    def count(self) -> int:
        return len(self._rows)

    def add(self, row: dict[str, object]) -> None:
        self._rows.append(dict(row))

    def values(self, key: str) -> list[float]:
        values: list[float] = []
        for row in self._rows:
            value = finite_float(row.get(key))
            if value is not None:
                values.append(value)
        return values

    def mean(self, key: str) -> float:
        values = self.values(key)
        return fmean(values) if values else float("nan")

    def nonfinite_count(self, key: str) -> int:
        return sum(1 for row in self._rows if finite_float(row.get(key)) is None)


class TrainingStabilityTracker:
    def __init__(self, *, thresholds: StabilityThresholds | None = None) -> None:
        self.thresholds = thresholds or StabilityThresholds()
        self.window = RollingMetricWindow(self.thresholds.rolling_window)
        self.best_recent_loss = float("inf")
        self.best_match_score = -float("inf")

    def match_score(self, metrics: dict[str, object]) -> float:
        top1 = finite_float(metrics.get("top1_accuracy")) or 0.0
        rejected = finite_float(metrics.get("true_match_rejected_by_dustbin_ratio")) or 0.0
        false_accept = finite_float(metrics.get("false_match_accepted_ratio")) or 0.0
        margin = finite_float(metrics.get("positive_vs_dustbin_margin_mean")) or 0.0
        loss = finite_float(metrics.get("loss"))
        nan_penalty = 1.0 if loss is None else 0.0
        return top1 + 0.25 * margin - 0.75 * rejected - 0.25 * false_accept - nan_penalty

    def update(self, step: int, metrics: dict[str, object]) -> StabilityDecision:
        self.window.add(metrics)
        loss = finite_float(metrics.get("loss"))
        top1 = finite_float(metrics.get("top1_accuracy"))
        score = self.match_score(metrics)
        if score > self.best_match_score:
            self.best_match_score = score
        recent_loss = self.window.mean("loss")
        if math.isfinite(recent_loss):
            self.best_recent_loss = min(self.best_recent_loss, recent_loss)
        is_last_good = loss is not None and top1 is not None and top1 >= self.thresholds.min_top1_mean
        should_stop = False
        reason = ""
        if step >= self.thresholds.min_steps_before_early_stop:
            if self.window.nonfinite_count("loss") > self.thresholds.max_nan_in_window:
                should_stop = True
                reason = "too_many_nonfinite_loss_values"
            elif (
                math.isfinite(recent_loss)
                and math.isfinite(self.best_recent_loss)
                and recent_loss > self.best_recent_loss * self.thresholds.max_loss_multiplier
            ):
                should_stop = True
                reason = "recent_loss_exceeded_best_window"
            elif math.isfinite(self.window.mean("top1_accuracy")) and self.window.mean("top1_accuracy") < self.thresholds.min_top1_mean:
                should_stop = True
                reason = "top1_mean_below_threshold"
            elif score < self.thresholds.min_match_score:
                should_stop = True
                reason = "match_score_below_threshold"
        return StabilityDecision(
            should_stop=should_stop,
            should_save_latest=True,
            should_save_last_good=is_last_good,
            is_last_good=is_last_good,
            reason=reason,
        )
```

- [ ] **Step 2: Run GREEN test**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_training_stability
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add python/pfm_training_stability.py python/tests/test_pfm_training_stability.py
git commit -m "Add training stability tracker"
```

---

### Task 3: Add True-Match-vs-Dustbin Diagnostics

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Modify: `python/tests/test_pfm_pytorch_training.py`

- [ ] **Step 1: Write failing diagnostics test**

Add to `PFMPyTorchTrainingTest` in `python/tests/test_pfm_pytorch_training.py`:

```python
    def test_graph_matcher_dustbin_diagnostics_detects_rejected_true_matches(self):
        logits = torch.zeros(4, 4)
        logits[:3, :3] = torch.tensor(
            [
                [0.1, -1.0, -1.0],
                [-1.0, 0.2, -1.0],
                [-1.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        logits[0, 3] = 2.0
        logits[1, 3] = 2.0
        logits[3, 0] = 1.5
        logits[3, 1] = 1.5
        output = pfm_model.GraphMatcherOutput(
            logits=logits,
            matches=torch.empty((0, 2), dtype=torch.long),
            scores=torch.empty((0,), dtype=torch.float32),
        )

        metrics = train.graph_matcher_dustbin_diagnostics(output, positive_count=2)

        self.assertAlmostEqual(metrics["true_match_rejected_by_dustbin_ratio"], 1.0)
        self.assertLess(metrics["positive_vs_dustbin_margin_mean"], 0.0)
        self.assertGreater(metrics["dustbin_prob_for_true_match_mean"], metrics["true_pair_prob_mean"])
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_pytorch_training.PFMPyTorchTrainingTest.test_graph_matcher_dustbin_diagnostics_detects_rejected_true_matches
```

Expected: fail with `AttributeError: module 'pfm_pytorch_training' has no attribute 'graph_matcher_dustbin_diagnostics'`.

- [ ] **Step 3: Implement diagnostics**

Add near the GraphMatcher loss helpers in `python/pfm_pytorch_training.py`:

```python
def graph_matcher_dustbin_diagnostics(
    output: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
) -> dict[str, float]:
    count = min(int(positive_count), output.logits.size(0) - 1, output.logits.size(1) - 1)
    if count <= 0:
        return {
            "true_match_rejected_by_dustbin_ratio": 0.0,
            "positive_pair_logit_mean": 0.0,
            "positive_dustbin_logit_mean": 0.0,
            "positive_vs_dustbin_margin_mean": 0.0,
            "positive_vs_dustbin_margin_median": 0.0,
            "positive_vs_dustbin_margin_p10": 0.0,
            "positive_vs_dustbin_margin_below0_ratio": 0.0,
            "true_pair_prob_mean": 0.0,
            "dustbin_prob_for_true_match_mean": 0.0,
        }
    pair_logits = output.logits[:count, :count]
    true_logits = pair_logits.diagonal().to(torch.float32)
    row_dustbin = output.logits[:count, output.logits.size(1) - 1].to(torch.float32)
    col_dustbin = output.logits[output.logits.size(0) - 1, :count].to(torch.float32)
    strongest_dustbin = torch.maximum(row_dustbin, col_dustbin)
    margin = true_logits - strongest_dustbin
    row_prob = torch.softmax(output.logits[:count, :], dim=1)
    col_prob = torch.softmax(output.logits[:, :count], dim=0)
    true_pair_prob = row_prob[torch.arange(count, device=output.logits.device), torch.arange(count, device=output.logits.device)]
    dustbin_prob = torch.maximum(row_prob[:, -1], col_prob[-1, :])
    sorted_margin = margin.sort().values
    p10_index = min(sorted_margin.numel() - 1, max(0, int(math.floor((sorted_margin.numel() - 1) * 0.10))))
    return {
        "true_match_rejected_by_dustbin_ratio": float(margin.lt(0.0).to(torch.float32).mean().detach().cpu()),
        "positive_pair_logit_mean": float(true_logits.mean().detach().cpu()),
        "positive_dustbin_logit_mean": float(strongest_dustbin.mean().detach().cpu()),
        "positive_vs_dustbin_margin_mean": float(margin.mean().detach().cpu()),
        "positive_vs_dustbin_margin_median": float(margin.median().detach().cpu()),
        "positive_vs_dustbin_margin_p10": float(sorted_margin[p10_index].detach().cpu()),
        "positive_vs_dustbin_margin_below0_ratio": float(margin.lt(0.0).to(torch.float32).mean().detach().cpu()),
        "true_pair_prob_mean": float(true_pair_prob.to(torch.float32).mean().detach().cpu()),
        "dustbin_prob_for_true_match_mean": float(dustbin_prob.to(torch.float32).mean().detach().cpu()),
    }
```

Add these values to the GraphMatcher component metrics returned by `graph_matcher_correspondence_loss()` and surfaced by `train_step()`.

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_pytorch_training.PFMPyTorchTrainingTest.test_graph_matcher_dustbin_diagnostics_detects_rejected_true_matches
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add python/pfm_pytorch_training.py python/tests/test_pfm_pytorch_training.py
git commit -m "Add graph matcher dustbin diagnostics"
```

---

### Task 4: Integrate Best Checkpoints and Bad-Step Protection in Lazy Training

**Files:**
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Modify: `python/tests/test_benchmark_lazy_pose_pairs.py`
- Uses: `python/pfm_training_stability.py`

- [ ] **Step 1: Add parser tests for stability args**

Add to `python/tests/test_benchmark_lazy_pose_pairs.py`:

```python
    def test_parse_args_accepts_training_stability_controls(self):
        args = lazy_bench.parse_args(
            [
                "--render-manifest",
                "render.csv",
                "--uint8-manifest",
                "uint8.csv",
                "--output-dir",
                "out",
                "--mode",
                "train",
                "--stability-window",
                "300",
                "--stability-min-steps",
                "1500",
                "--stability-max-nan-in-window",
                "10",
                "--stability-min-top1-mean",
                "0.35",
                "--stability-max-loss-multiplier",
                "2.5",
                "--save-best-checkpoints",
            ]
        )

        self.assertEqual(args.stability_window, 300)
        self.assertEqual(args.stability_min_steps, 1500)
        self.assertEqual(args.stability_max_nan_in_window, 10)
        self.assertAlmostEqual(args.stability_min_top1_mean, 0.35)
        self.assertAlmostEqual(args.stability_max_loss_multiplier, 2.5)
        self.assertTrue(args.save_best_checkpoints)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs.PFMBenchmarkLazyPosePairsTest.test_parse_args_accepts_training_stability_controls
```

Expected: fail because args do not exist.

- [ ] **Step 3: Add parser args**

In `scripts/benchmark_lazy_pose_pairs.py`, add:

```python
    parser.add_argument("--save-best-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stability-window", type=int, default=200)
    parser.add_argument("--stability-min-steps", type=int, default=1000)
    parser.add_argument("--stability-max-nan-in-window", type=int, default=20)
    parser.add_argument("--stability-min-top1-mean", type=float, default=0.35)
    parser.add_argument("--stability-max-loss-multiplier", type=float, default=3.0)
    parser.add_argument("--stability-min-match-score", type=float, default=-0.5)
```

Add validation:

```python
    if args.stability_window <= 0:
        parser.error("--stability-window must be positive")
    if args.stability_min_steps < 0:
        parser.error("--stability-min-steps must be nonnegative")
    if args.stability_max_nan_in_window < 0:
        parser.error("--stability-max-nan-in-window must be nonnegative")
    if args.stability_max_loss_multiplier <= 1.0:
        parser.error("--stability-max-loss-multiplier must be greater than 1")
```

- [ ] **Step 4: Add checkpoint save decisions**

In `run_train()`, construct:

```python
from pfm_training_stability import StabilityThresholds, TrainingStabilityTracker

stability = TrainingStabilityTracker(
    thresholds=StabilityThresholds(
        min_steps_before_early_stop=args.stability_min_steps,
        rolling_window=args.stability_window,
        max_nan_in_window=args.stability_max_nan_in_window,
        max_loss_multiplier=args.stability_max_loss_multiplier,
        min_top1_mean=args.stability_min_top1_mean,
        min_match_score=args.stability_min_match_score,
    )
)
best_match_score = -float("inf")
best_loss = float("inf")
```

After each row is assembled:

```python
decision = stability.update(step, row)
current_score = stability.match_score(row)
loss_value = _finite_float(row["loss"])
if args.save_best_checkpoints:
    _save_training_state(args.output_dir / "checkpoints" / "latest_pytorch_pfm_state.pt", model, args, step)
    if decision.should_save_last_good:
        _save_training_state(args.output_dir / "checkpoints" / "last_good_pytorch_pfm_state.pt", model, args, step)
    if current_score > best_match_score:
        best_match_score = current_score
        _save_training_state(args.output_dir / "checkpoints" / "best_by_match_score_pytorch_pfm_state.pt", model, args, step)
    if loss_value is not None and loss_value < best_loss:
        best_loss = loss_value
        _save_training_state(args.output_dir / "checkpoints" / "best_by_loss_pytorch_pfm_state.pt", model, args, step)
if decision.should_stop:
    crash_report = {
        "step": step,
        "reason": decision.reason,
        "best_match_score": best_match_score,
        "best_loss": best_loss,
        "latest_row": row,
    }
    (args.output_dir / "crash_report.json").write_text(json.dumps(crash_report, ensure_ascii=False, indent=2), encoding="utf-8")
    break
```

Add helper near `_summarize_float()` if no equivalent exists:

```python
def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
```

- [ ] **Step 5: Run parser and unit tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs python.tests.test_pfm_training_stability
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark_lazy_pose_pairs.py python/tests/test_benchmark_lazy_pose_pairs.py python/pfm_training_stability.py python/tests/test_pfm_training_stability.py
git commit -m "Add stable checkpoint selection to lazy training"
```

---

### Task 5: Add Shape-Safe Graph Architecture Overrides

**Files:**
- Modify: `python/pfm_model.py`
- Modify: `python/tests/test_pfm_pytorch_training.py`
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Modify: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Write model-loading test**

Add to `python/tests/test_pfm_pytorch_training.py`:

```python
    def test_load_pytorch_state_can_override_graph_architecture_and_keep_extractor(self):
        model = pfm_model.PlanetaryFeatureMatcher(graph_hidden_dim=512, graph_attention_layers=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            torch.save(
                {
                    "config": {
                        "input_channels": model.config.input_channels,
                        "base_channels": model.config.base_channels,
                        "descriptor_dim": model.config.descriptor_dim,
                        "graph_hidden_dim": model.config.graph_hidden_dim,
                        "graph_attention_layers": model.config.graph_attention_layers,
                        "graph_keypoint_meta_dim": model.config.graph_keypoint_meta_dim,
                    },
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                },
                path,
            )

            loaded, config = pfm_model.load_pytorch_state(
                path,
                device="cpu",
                strict=False,
                graph_hidden_dim_override=256,
                graph_attention_layers_override=2,
                skip_mismatched_shapes=True,
            )

        self.assertEqual(config.graph_hidden_dim, 256)
        self.assertEqual(config.graph_attention_layers, 2)
        self.assertEqual(len(loaded.graph_matcher.attention_layers), 2)
        self.assertEqual(loaded.graph_matcher.hidden_dim, 256)
        self.assertTrue(torch.equal(loaded.backbone.stage1[0].weight, model.backbone.stage1[0].weight))
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_pytorch_training.PFMPyTorchTrainingTest.test_load_pytorch_state_can_override_graph_architecture_and_keep_extractor
```

Expected: fail with unexpected keyword args.

- [ ] **Step 3: Implement shape-safe override**

In `python/pfm_model.py`, extend `_with_default_compatible_state()`:

```python
def _with_default_compatible_state(
    model: PlanetaryFeatureMatcher,
    state: dict[str, torch.Tensor],
    *,
    skip_mismatched_shapes: bool = False,
) -> dict[str, torch.Tensor]:
    defaults = model.state_dict()
    patched: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key not in defaults:
            continue
        if skip_mismatched_shapes and tuple(value.shape) != tuple(defaults[key].shape):
            continue
        patched[key] = value
    for key, value in defaults.items():
        if key not in patched:
            patched[key] = value.detach().clone()
    return patched
```

Extend `load_pytorch_state()` signature:

```python
def load_pytorch_state(
    checkpoint: Path | str,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
    graph_hidden_dim_override: int | None = None,
    graph_attention_layers_override: int | None = None,
    skip_mismatched_shapes: bool = False,
) -> tuple[PlanetaryFeatureMatcher, CheckpointConfig]:
```

After reading `config_dict`, compute:

```python
graph_hidden_dim = int(graph_hidden_dim_override or config_dict["graph_hidden_dim"])
graph_attention_layers = int(graph_attention_layers_override or config_dict["graph_attention_layers"])
config = CheckpointConfig(
    input_channels=int(config_dict["input_channels"]),
    base_channels=int(config_dict["base_channels"]),
    descriptor_dim=int(config_dict["descriptor_dim"]),
    graph_hidden_dim=graph_hidden_dim,
    graph_attention_layers=graph_attention_layers,
    graph_keypoint_meta_dim=int(config_dict.get("graph_keypoint_meta_dim", 2)),
)
```

Call:

```python
model_state = _with_default_compatible_state(model, payload["model"], skip_mismatched_shapes=skip_mismatched_shapes)
```

- [ ] **Step 4: Add lazy training parser args**

In `scripts/benchmark_lazy_pose_pairs.py` parser:

```python
    parser.add_argument("--model-graph-hidden-dim", type=int, default=0)
    parser.add_argument("--model-graph-attention-layers", type=int, default=0)
    parser.add_argument("--skip-mismatched-checkpoint-shapes", action=argparse.BooleanOptionalAction, default=False)
```

Validate:

```python
    if args.model_graph_hidden_dim < 0:
        parser.error("--model-graph-hidden-dim must be nonnegative")
    if args.model_graph_attention_layers < 0:
        parser.error("--model-graph-attention-layers must be nonnegative")
```

Update `_load_model()` to call:

```python
pfm_model.load_pytorch_state(
    args.init_pytorch_state,
    device=device,
    strict=not args.skip_mismatched_checkpoint_shapes,
    graph_hidden_dim_override=args.model_graph_hidden_dim or None,
    graph_attention_layers_override=args.model_graph_attention_layers or None,
    skip_mismatched_shapes=args.skip_mismatched_checkpoint_shapes,
)
```

- [ ] **Step 5: Add parser test**

Add to `python/tests/test_benchmark_lazy_pose_pairs.py`:

```python
    def test_parse_args_accepts_model_graph_architecture_overrides(self):
        args = lazy_bench.parse_args(
            [
                "--render-manifest",
                "render.csv",
                "--uint8-manifest",
                "uint8.csv",
                "--output-dir",
                "out",
                "--mode",
                "train",
                "--model-graph-hidden-dim",
                "256",
                "--model-graph-attention-layers",
                "2",
                "--skip-mismatched-checkpoint-shapes",
            ]
        )

        self.assertEqual(args.model_graph_hidden_dim, 256)
        self.assertEqual(args.model_graph_attention_layers, 2)
        self.assertTrue(args.skip_mismatched_checkpoint_shapes)
```

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_pytorch_training python.tests.test_benchmark_lazy_pose_pairs
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add python/pfm_model.py python/tests/test_pfm_pytorch_training.py scripts/benchmark_lazy_pose_pairs.py python/tests/test_benchmark_lazy_pose_pairs.py
git commit -m "Allow graph matcher architecture overrides"
```

---

### Task 6: Add Reduced Rejection Stable Preset

**Files:**
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Modify: `python/pfm_pytorch_training.py`
- Modify: `python/tests/test_benchmark_lazy_pose_pairs.py`
- Modify: `python/tests/test_pfm_pytorch_training.py`

- [ ] **Step 1: Add tests for stable rejection preset**

Add to both parser test files:

```python
    def test_stable_rejection_preset_uses_low_dustbin_weights(self):
        args = lazy_bench.parse_args(
            [
                "--render-manifest",
                "render.csv",
                "--uint8-manifest",
                "uint8.csv",
                "--output-dir",
                "out",
                "--mode",
                "train",
                "--enable-stable-rejection-training",
            ]
        )

        self.assertTrue(args.train_graph_matcher)
        self.assertGreater(args.graph_matcher_no_match_weight, 0.0)
        self.assertLessEqual(args.graph_matcher_no_match_weight, 0.15)
        self.assertLessEqual(args.graph_matcher_hard_negative_dustbin_weight, 0.02)
        self.assertEqual(args.graph_matcher_stop_confidence_weight, 0.0)
        self.assertEqual(args.no_match_prior_weight, 0.0)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs.PFMBenchmarkLazyPosePairsTest.test_stable_rejection_preset_uses_low_dustbin_weights
```

Expected: fail because `--enable-stable-rejection-training` does not exist.

- [ ] **Step 3: Implement stable preset**

Add constants:

```python
STABLE_REJECTION_TRAINING_DEFAULTS = {
    "graph_matcher_loss_weight": 0.50,
    "graph_matcher_no_match_points": 64,
    "graph_matcher_no_match_weight": 0.10,
    "graph_matcher_assignment_weight": 0.30,
    "graph_matcher_accept_weight": 0.10,
    "graph_matcher_prune_ranking_weight": 0.02,
    "graph_matcher_stop_confidence_weight": 0.0,
    "graph_matcher_hard_negative_dustbin_weight": 0.01,
    "graph_matcher_hard_negative_dustbin_topk": 8,
    "graph_matcher_hard_negative_dustbin_margin": 0.20,
    "graph_matcher_semi_dense_no_match_points": 32,
    "false_match_weight": 0.05,
    "false_match_max_points": 96,
    "keypoint_weight": 0.03,
    "keypoint_negative_weight": 0.01,
    "matchability_weight": 0.03,
    "descriptor_uncertainty_weight": 0.02,
    "no_match_prior_weight": 0.0,
    "reliability_negative_points": 32,
    "rotation_descriptor_consistency_weight": 0.02,
}
```

Add parser arg:

```python
    parser.add_argument("--enable-stable-rejection-training", action=argparse.BooleanOptionalAction, default=False)
```

Apply before or instead of strong preset:

```python
def apply_stable_rejection_training_defaults(args: argparse.Namespace) -> None:
    if not getattr(args, "enable_stable_rejection_training", False):
        return
    args.train_graph_matcher = True
    args.inline_false_match_mining = True
    args.graph_matcher_online_false_no_match = False
    args.visual_matcher_mode = "graph_matcher"
    if args.visual_keypoint_score_mode == "texture":
        args.visual_keypoint_score_mode = "learned"
    for name, value in STABLE_REJECTION_TRAINING_DEFAULTS.items():
        current = getattr(args, name)
        if current <= 0 or current == REJECTION_TRAINING_BASE_DEFAULTS.get(name):
            setattr(args, name, value)
```

Call:

```python
apply_stable_rejection_training_defaults(args)
apply_rejection_training_defaults(args)
```

Guard against both strong and stable:

```python
    if args.enable_stable_rejection_training and args.enable_rejection_training:
        parser.error("--enable-stable-rejection-training cannot be combined with --enable-rejection-training")
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs python.tests.test_pfm_pytorch_training
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_lazy_pose_pairs.py python/pfm_pytorch_training.py python/tests/test_benchmark_lazy_pose_pairs.py python/tests/test_pfm_pytorch_training.py
git commit -m "Add stable rejection training preset"
```

---

### Task 7: Add Inference-Depth Controls to Visual Evaluation

**Files:**
- Modify: `scripts/visualize_lazy_pose_matches.py`
- Modify: `python/tests/test_stress_eval_scripts.py`
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Modify: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Add parser test**

In `python/tests/test_stress_eval_scripts.py`, add:

```python
    def test_visualize_lazy_pose_matches_accepts_graph_depth_controls(self):
        args = visualize_lazy_pose_matches.parse_args(
            [
                "--render-manifest",
                "render.csv",
                "--uint8-manifest",
                "uint8.csv",
                "--pytorch-state",
                "state.pt",
                "--output-dir",
                "out",
                "--graph-max-attention-layers",
                "2",
                "--graph-max-attention-work-fraction",
                "1.0",
                "--graph-width-prune-keep-ratio",
                "1.0",
            ]
        )

        self.assertEqual(args.graph_max_attention_layers, 2)
        self.assertAlmostEqual(args.graph_max_attention_work_fraction, 1.0)
        self.assertAlmostEqual(args.graph_width_prune_keep_ratio, 1.0)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_stress_eval_scripts
```

Expected: fail because args do not exist.

- [ ] **Step 3: Wire visualizer args into matcher calls**

Add parser args:

```python
    parser.add_argument("--graph-max-attention-layers", type=int, default=0)
    parser.add_argument("--graph-max-attention-work-fraction", type=float, default=1.0)
    parser.add_argument("--graph-width-prune-keep-ratio", type=float, default=1.0)
```

Validate:

```python
    if args.graph_max_attention_layers < 0:
        raise ValueError("--graph-max-attention-layers must be nonnegative")
    if args.graph_max_attention_work_fraction < 0.0 or args.graph_max_attention_work_fraction > 1.0:
        raise ValueError("--graph-max-attention-work-fraction must be in [0, 1]")
    if args.graph_width_prune_keep_ratio < 0.0 or args.graph_width_prune_keep_ratio > 1.0:
        raise ValueError("--graph-width-prune-keep-ratio must be in [0, 1]")
```

Pass into all `model.graph_matcher(...)` calls:

```python
max_attention_layers=args.graph_max_attention_layers,
max_attention_work_fraction=args.graph_max_attention_work_fraction,
width_prune_keep_ratio=args.graph_width_prune_keep_ratio,
```

- [ ] **Step 4: Pass controls from auto visual report**

In `scripts/benchmark_lazy_pose_pairs.py`, add training parser args:

```python
    parser.add_argument("--visual-graph-max-attention-layers", type=int, default=0)
    parser.add_argument("--visual-graph-max-attention-work-fraction", type=float, default=1.0)
    parser.add_argument("--visual-graph-width-prune-keep-ratio", type=float, default=1.0)
```

Add to `_run_visual_report()` command:

```python
        "--graph-max-attention-layers",
        str(args.visual_graph_max_attention_layers),
        "--graph-max-attention-work-fraction",
        str(args.visual_graph_max_attention_work_fraction),
        "--graph-width-prune-keep-ratio",
        str(args.visual_graph_width_prune_keep_ratio),
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_stress_eval_scripts python.tests.test_benchmark_lazy_pose_pairs
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add scripts/visualize_lazy_pose_matches.py scripts/benchmark_lazy_pose_pairs.py python/tests/test_stress_eval_scripts.py python/tests/test_benchmark_lazy_pose_pairs.py
git commit -m "Expose graph inference depth controls"
```

---

### Task 8: Add P1-A Graph Inference Depth Ablation Script

**Files:**
- Create: `scripts/run_graph_depth_ablation.py`
- Modify: `scripts/README.md`
- Test: `python/tests/test_stress_eval_scripts.py`

- [ ] **Step 1: Write parser test**

Add to `python/tests/test_stress_eval_scripts.py`:

```python
    def test_run_graph_depth_ablation_parse_args(self):
        import run_graph_depth_ablation

        args = run_graph_depth_ablation.parse_args(
            [
                "--render-manifest",
                "render.csv",
                "--uint8-manifest",
                "uint8.csv",
                "--pytorch-state",
                "state.pt",
                "--output-dir",
                "out",
                "--layers",
                "1,2,4,8",
                "--keypoints",
                "256,512",
                "--pair-spec-manifest",
                "pairs.csv",
            ]
        )

        self.assertEqual(args.layers, [1, 2, 4, 8])
        self.assertEqual(args.keypoints, [256, 512])
        self.assertEqual(args.pair_spec_manifest, Path("pairs.csv"))
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_stress_eval_scripts
```

Expected: fail because module does not exist.

- [ ] **Step 3: Implement script**

Create `scripts/run_graph_depth_ablation.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return items


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--uint8-manifest", type=Path, required=True)
    parser.add_argument("--pytorch-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-spec-manifest", type=Path, default=None)
    parser.add_argument("--layers", type=parse_int_list, default=[1, 2, 4, 6, 8])
    parser.add_argument("--keypoints", type=parse_int_list, default=[256, 512, 1024])
    parser.add_argument("--candidate-pairs", type=int, default=24)
    parser.add_argument("--select-count", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args(argv)
    if any(value <= 0 for value in args.layers):
        parser.error("--layers values must be positive")
    if any(value <= 0 for value in args.keypoints):
        parser.error("--keypoints values must be positive")
    return args


def read_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    numeric: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            try:
                numeric.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                pass
    return {f"{key}_mean": f"{sum(values) / len(values):.6f}" for key, values in numeric.items() if values}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for layer_count in args.layers:
        for keypoint_count in args.keypoints:
            run_dir = args.output_dir / f"layers_{layer_count:02d}_keypoints_{keypoint_count:04d}"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "visualize_lazy_pose_matches.py"),
                "--render-manifest",
                str(args.render_manifest),
                "--uint8-manifest",
                str(args.uint8_manifest),
                "--pytorch-state",
                str(args.pytorch_state),
                "--output-dir",
                str(run_dir),
                "--matcher-mode",
                "graph_matcher",
                "--keypoint-score-mode",
                "learned",
                "--max-keypoints",
                str(keypoint_count),
                "--candidate-pairs",
                str(args.candidate_pairs),
                "--select-count",
                str(args.select_count),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--graph-max-attention-layers",
                str(layer_count),
                "--graph-max-attention-work-fraction",
                "1.0",
                "--graph-width-prune-keep-ratio",
                "1.0",
            ]
            if args.pair_spec_manifest is not None:
                command.extend(["--pair-spec-manifest", str(args.pair_spec_manifest)])
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            row = {
                "layers": str(layer_count),
                "keypoints": str(keypoint_count),
                "report_dir": str(run_dir),
            }
            row.update({f"raw_{key}": value for key, value in read_summary(run_dir / "summary.csv").items()})
            row.update({f"filtered_{key}": value for key, value in read_summary(run_dir / "filtered_summary.csv").items()})
            rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_dir / "depth_ablation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Update scripts README**

In `scripts/README.md`, add row:

```markdown
| `run_graph_depth_ablation.py` | 对同一 checkpoint 扫 `graph max attention layers` 与 keypoint 数，汇总 raw/filtered visual report。 | P1-A 诊断 2 层训练和多层推理不一致时使用。 |
```

- [ ] **Step 5: Run tests and compile**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_stress_eval_scripts
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m compileall -q scripts/run_graph_depth_ablation.py scripts/visualize_lazy_pose_matches.py
```

Expected: `OK`, then no compile output.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_graph_depth_ablation.py scripts/README.md python/tests/test_stress_eval_scripts.py
git commit -m "Add graph depth ablation script"
```

---

### Task 9: Update Stable h100 fov90 Launcher

**Files:**
- Modify: `runs/train_h100_fov090_spatial_cross_camera_sample_20260608.sh`

- [ ] **Step 1: Add stable environment defaults**

Change launcher defaults to allow:

```bash
PFM_RUN_STAMP="${PFM_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_h100_fov090_stable_graph2_hidden256}"
PFM_STEPS="${PFM_STEPS:-12000}"
PFM_GRAPH_LAYERS="${PFM_GRAPH_LAYERS:-2}"
PFM_GRAPH_HIDDEN_DIM="${PFM_GRAPH_HIDDEN_DIM:-256}"
PFM_GRAPH_TRAIN_MAX_LAYERS="${PFM_GRAPH_TRAIN_MAX_LAYERS:-2}"
PFM_GRAPH_WIDTH_KEEP_RATIO="${PFM_GRAPH_WIDTH_KEEP_RATIO:-1.0}"
PFM_STABILITY_MIN_STEPS="${PFM_STABILITY_MIN_STEPS:-1500}"
PFM_STABILITY_WINDOW="${PFM_STABILITY_WINDOW:-300}"
```

Add training args:

```bash
--model-graph-hidden-dim "$PFM_GRAPH_HIDDEN_DIM"
--model-graph-attention-layers "$PFM_GRAPH_LAYERS"
--skip-mismatched-checkpoint-shapes
--graph-matcher-train-max-attention-layers "$PFM_GRAPH_TRAIN_MAX_LAYERS"
--graph-matcher-train-width-keep-ratio "$PFM_GRAPH_WIDTH_KEEP_RATIO"
--enable-stable-rejection-training
--save-best-checkpoints
--stability-min-steps "$PFM_STABILITY_MIN_STEPS"
--stability-window "$PFM_STABILITY_WINDOW"
--visual-graph-max-attention-layers "$PFM_GRAPH_LAYERS"
--visual-graph-width-prune-keep-ratio 1.0
```

Remove strong rejection args from the stable default command or gate them behind:

```bash
PFM_ENABLE_STRONG_REJECTION="${PFM_ENABLE_STRONG_REJECTION:-0}"
```

- [ ] **Step 2: Validate shell syntax**

Run:

```bash
bash -n runs/train_h100_fov090_spatial_cross_camera_sample_20260608.sh
```

Expected: no output and exit code 0.

- [ ] **Step 3: Do not commit runs script unless requested**

`runs/` is ignored and local-run specific. Leave it uncommitted unless the user explicitly asks for this launcher to be versioned elsewhere.

---

### Task 10: Run Verification Suite

**Files:** no source edits.

- [ ] **Step 1: Run Python unit tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_pfm_training_stability \
  python.tests.test_benchmark_lazy_pose_pairs \
  python.tests.test_pfm_pytorch_training \
  python.tests.test_stress_eval_scripts
```

Expected: all tests pass.

- [ ] **Step 2: Run compile checks**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m compileall -q \
  python/pfm_training_stability.py \
  python/pfm_model.py \
  python/pfm_pytorch_training.py \
  scripts/benchmark_lazy_pose_pairs.py \
  scripts/visualize_lazy_pose_matches.py \
  scripts/run_graph_depth_ablation.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Commit verification updates if any**

If Task 10 required fixes:

```bash
git add python scripts docs
git commit -m "Stabilize graph matcher training pipeline"
```

---

## Next Stable Training Run

After implementation, launch one stability verification run, not a final performance run.

Use:

```bash
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py' || true

RUN_STAMP="$(date +%Y%m%d_%H%M%S)_h100_fov090_stable_graph2_hidden256"
LOG="runs/train_${RUN_STAMP}.log"
PID="runs/train_${RUN_STAMP}.pid"
SNAPSHOT="/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260608_223338_h100_fov090_mid2_extreme3_overlap_train/training_snapshot_20260609_100315_amp_safe_strong_rejection_learned_kp_overlap_edges.csv"

setsid env \
  PFM_RUN_STAMP="$RUN_STAMP" \
  PFM_PAIR_SPEC_MANIFEST="$SNAPSHOT" \
  PFM_STEPS=12000 \
  PFM_LIMIT_PAIRS=50000 \
  PFM_WORKERS=8 \
  PFM_PREFETCH_BATCHES=24 \
  PFM_WORKER_CACHE_ITEMS=48 \
  PFM_GRAPH_LAYERS=2 \
  PFM_GRAPH_HIDDEN_DIM=256 \
  PFM_GRAPH_TRAIN_MAX_LAYERS=2 \
  PFM_GRAPH_WIDTH_KEEP_RATIO=1.0 \
  PFM_AUTO_VISUAL_REPORT=1 \
  runs/train_h100_fov090_spatial_cross_camera_sample_20260608.sh > "$LOG" 2>&1 &
echo $! > "$PID"
```

Expected:

- Training and inference graph layers both equal 2.
- Hidden dim equals 256.
- `no_match_prior_weight=0`.
- `graph_matcher_stop_confidence_weight=0`.
- `graph_matcher_hard_negative_dustbin_weight<=0.02`.
- `train_metrics.csv` contains true-match-vs-dustbin diagnostics.
- `checkpoints/best_by_match_score_pytorch_pfm_state.pt` exists before the run ends.
- Auto visual report uses the best checkpoint and `--graph-max-attention-layers 2`.

Success criteria:

- `nan_loss_total` is near zero after warmup.
- `top1_accuracy` does not collapse after 10k steps.
- `true_match_rejected_by_dustbin_ratio < 0.3` in rolling windows.
- `positive_vs_dustbin_margin_mean > 0`.
- `num_filtered_matches` is not consistently near zero.
- filtered visual precision and recall improve together, not precision-only via over-rejection.

---

## Later Ablation Phases

Run only after the stable graph2 run produces reliable checkpoints.

1. P1-A current-checkpoint inference depth sweep:
   - `layers=1,2,4,6,8`
   - `keypoints=256,512,1024`
   - same checkpoint, same pair snapshot.

2. P1-B architecture ablation:
   - separately train `layers=1,2,4,6,8`
   - small budget `2000-4000` steps first.

3. P3 geometry pooling ablation:
   - A: orientation + scale + affine canonical pooling.
   - B: orientation + scale only.
   - C: plain bilinear descriptor sampling.
   - Compare descriptor recall@1/5, positive cosine, negative cosine, positive-negative margin.

4. P4 reliability ablation:
   - no reliability bias.
   - matchability only.
   - uncertainty only as loss weighting.
   - no_match_prior disabled until true-match-vs-dustbin is healthy.

5. P5 texture fusion ablation:
   - learned only.
   - texture only.
   - learned + texture fusion.
   - learned + texture + quality.
   - Record fusion gate/norm distributions.

6. P6 extractor changes:
   - stage1 skip for keypoint branch.
   - `score = heatmap * (0.5 + 0.5 * quality)`.
   - descriptor dim 128 only if memory still blocks cleaner graph training.

---

## Self-Review

- Spec coverage:
  - Best checkpoint and bad-step protection: Tasks 1, 2, 4.
  - Graph depth consistency: Tasks 5, 7, 8, 9.
  - Reduced rejection/dustbin: Task 6 and next-run command.
  - True-match-vs-dustbin diagnostics: Task 3 and metric success criteria.
  - Geometry/reliability/texture/extractor ablations: later phases, deliberately not mixed into first implementation.
- Placeholder scan:
  - No `TBD`, `TODO`, or unspecified test commands are present.
- Type consistency:
  - `StabilityThresholds`, `TrainingStabilityTracker`, `graph_matcher_dustbin_diagnostics`, and graph override arg names are consistent across tests and implementation tasks.
