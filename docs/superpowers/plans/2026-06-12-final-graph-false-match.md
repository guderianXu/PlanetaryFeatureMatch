# Final Graph False-Match Mining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional final GraphMatcher output false-match mining to directly penalize high-confidence wrong matcher edges during training.

**Architecture:** Keep the loss local to `python/pfm_pytorch_training.py`. The helper consumes `GraphMatcherOutput` and positive feature points, mines off-diagonal positive-block edges from final logits, and returns a scalar loss plus diagnostics. CLI and CSV metrics are plumbed through the existing train-step and training-loop argument path.

**Tech Stack:** Python, PyTorch, `unittest`, existing PFM training scripts.

---

### Task 1: Add Failing Tests

**Files:**
- Modify: `python/tests/test_pfm_pytorch_training.py`

- [ ] **Step 1: Add direct helper tests**

Add tests near existing GraphMatcher loss tests:

```python
def test_graph_matcher_final_false_match_loss_mines_high_confidence_wrong_edge(self):
    logits = torch.full((4, 4), -5.0)
    logits[0, 0] = 3.0
    logits[1, 1] = 3.0
    logits[2, 2] = 3.0
    logits[0, 1] = 8.0
    logits[3, :] = -4.0
    logits[:, 3] = -4.0
    accept_logits = torch.zeros(3, 3)
    accept_logits[0, 1] = 6.0
    output = pfm_model.GraphMatcherOutput(
        matches=torch.empty((0, 2), dtype=torch.long),
        scores=torch.empty(0),
        logits=logits,
        accept_logits=accept_logits,
    )
    points = torch.tensor([[0.0, 0.0], [8.0, 0.0], [16.0, 0.0]])

    loss, metrics = train.graph_matcher_final_false_match_loss(
        output,
        positive_count=3,
        points_b_xy=points,
        topk=2,
        min_score=0.01,
        margin=0.25,
    )

    self.assertGreater(float(loss), 0.0)
    self.assertEqual(float(metrics["edges"]), 1.0)
    self.assertGreater(float(metrics["score_mean"]), 0.0)
    self.assertGreater(float(metrics["accept_mean"]), 0.0)
```

```python
def test_graph_matcher_final_false_match_loss_returns_zero_without_selected_edges(self):
    logits = torch.eye(4) * 6.0
    logits[3, :] = -5.0
    logits[:, 3] = -5.0
    output = pfm_model.GraphMatcherOutput(
        matches=torch.empty((0, 2), dtype=torch.long),
        scores=torch.empty(0),
        logits=logits,
        accept_logits=torch.zeros(3, 3),
    )
    points = torch.tensor([[0.0, 0.0], [8.0, 0.0], [16.0, 0.0]])

    loss, metrics = train.graph_matcher_final_false_match_loss(
        output,
        positive_count=3,
        points_b_xy=points,
        topk=2,
        min_score=0.9,
        margin=0.25,
    )

    self.assertEqual(float(loss), 0.0)
    self.assertEqual(float(metrics["edges"]), 0.0)
```

- [ ] **Step 2: Add graph loss component test**

Create a tiny `PlanetaryFeatureMatcher` with three sampled positive correspondences, call `graph_matcher_correspondence_loss(..., final_false_match_weight=0.5, return_components=True)`, and assert these keys exist:

```python
self.assertIn("graph_matcher_final_false_match_loss", components)
self.assertIn("graph_matcher_final_false_match_edges", components)
```

- [ ] **Step 3: Extend parse-args test**

In `test_parse_args_accepts_graph_matcher_no_match_options`, add:

```python
"--graph-matcher-final-false-match-weight", "0.05",
"--graph-matcher-final-false-match-topk", "4",
"--graph-matcher-final-false-match-min-score", "0.02",
"--graph-matcher-final-false-match-margin", "0.3",
"--graph-matcher-final-false-match-spatial-min-distance", "5.0",
```

Then assert:

```python
self.assertAlmostEqual(args.graph_matcher_final_false_match_weight, 0.05)
self.assertEqual(args.graph_matcher_final_false_match_topk, 4)
self.assertAlmostEqual(args.graph_matcher_final_false_match_min_score, 0.02)
self.assertAlmostEqual(args.graph_matcher_final_false_match_margin, 0.3)
self.assertAlmostEqual(args.graph_matcher_final_false_match_spatial_min_distance, 5.0)
```

- [ ] **Step 4: Run tests and confirm RED**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_pfm_pytorch_training.TestPfmPytorchTraining.test_graph_matcher_final_false_match_loss_mines_high_confidence_wrong_edge \
  python.tests.test_pfm_pytorch_training.TestPfmPytorchTraining.test_graph_matcher_final_false_match_loss_returns_zero_without_selected_edges \
  python.tests.test_pfm_pytorch_training.TestPfmPytorchTraining.test_parse_args_accepts_graph_matcher_no_match_options
```

Expected: failure because `graph_matcher_final_false_match_loss` and CLI args do not exist yet.

### Task 2: Implement Helper and Components

**Files:**
- Modify: `python/pfm_pytorch_training.py`

- [ ] **Step 1: Add diagnostic metric names**

Extend `GRAPH_MATCHER_DIAGNOSTIC_METRIC_FIELDS` with:

```python
"graph_matcher_final_false_match_loss",
"graph_matcher_final_false_match_edges",
"graph_matcher_final_false_match_score_mean",
"graph_matcher_final_false_match_accept_mean",
```

- [ ] **Step 2: Implement helper**

Add `graph_matcher_final_false_match_loss()` near existing GraphMatcher loss helpers. It should:

- validate nonnegative `topk`, `min_score`, `margin`, and `spatial_min_distance`;
- return zero loss and zero metrics when `positive_count <= 1` or `topk <= 0`;
- compute row softmax over `output.logits[:count, :]`;
- compute column softmax over `output.logits[:, :count]`;
- compute dual scores in the positive block;
- mask diagonal entries and spatially-near target points;
- select up to `topk` entries with dual score at least `min_score`;
- apply negative BCE to selected `accept_logits` when available;
- apply squared margin loss `margin - (true_logit - wrong_logit)`;
- return the mean selected dual score and mean selected accept probability as detached diagnostics.

- [ ] **Step 3: Wire into `graph_matcher_correspondence_loss()`**

Add keyword arguments:

```python
final_false_match_weight: float = 0.0
final_false_match_topk: int = 8
final_false_match_min_score: float = 0.0
final_false_match_margin: float = 0.25
final_false_match_spatial_min_distance: float = 0.0
```

Compute the helper after the final `output` exists. Apply the loss only when `final_false_match_weight > 0.0` and `dustbin_guard_enabled` is false.

- [ ] **Step 4: Include components**

Return zero-valued metrics when disabled and real metrics when enabled.

### Task 3: Wire CLI and Train Step

**Files:**
- Modify: `python/pfm_pytorch_training.py`

- [ ] **Step 1: Add train_step parameters**

Add the five `graph_matcher_final_false_match_*` parameters to `train_step()` and forward them into `graph_matcher_correspondence_loss()`.

- [ ] **Step 2: Add parser arguments and validation**

Add parser arguments near the other GraphMatcher rejection options and reject invalid negative values.

- [ ] **Step 3: Add training loop CSV columns and row values**

Add the new weight and diagnostics to the CSV header and writer row. Use zero values when `--train-graph-matcher` is off.

### Task 4: Update Benchmark Pass-Through

**Files:**
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Mirror metric names and argument pass-through**

Add the same diagnostic field names and pass the new args into `train.train_step()` when this script trains or benchmarks.

- [ ] **Step 2: Add or update benchmark parser tests**

Extend the existing benchmark parser tests to ensure the new options are accepted if that script exposes the GraphMatcher training options.

### Task 5: Verify

**Files:**
- Read: test output

- [ ] **Step 1: Run targeted tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_pfm_pytorch_training
```

- [ ] **Step 2: Run benchmark tests if script changed**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs
```

### Task 6: Short Training Smoke

**Files:**
- Create: `runs/final_false_match_smoke_20260612.sh`
- Create: `runs/final_false_match_smoke_20260612.html`

- [ ] **Step 1: Launch a short run**

Use the fov76 internal cache and current safe baseline checkpoint. Start with:

```bash
--steps 400
--graph-matcher-final-false-match-weight 0.03
--graph-matcher-final-false-match-topk 8
--graph-matcher-final-false-match-min-score 0.01
--graph-matcher-final-false-match-margin 0.25
--graph-matcher-final-false-match-spatial-min-distance 4.0
```

- [ ] **Step 2: Inspect metrics**

Check these fields in `metrics.csv`:

```text
graph_matcher_final_false_match_loss
graph_matcher_final_false_match_edges
true_match_rejected_by_dustbin_ratio
positive_vs_dustbin_margin_mean
```

The run is acceptable if final false-match edges appear without sustained growth in true-match dustbin rejection.
