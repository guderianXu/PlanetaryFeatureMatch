# fov76 Phase6c Matcher Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the next fov76 matcher optimization round after phase6b was rejected, focusing on increasing valid extreme_02/extreme_03 matches without reducing formal val/test correct count or weakening protected variants.

**Architecture:** Keep the current active model path as `phase5g + phase6a selector` and treat phase6b only as diagnostic evidence. Add a reusable delta analysis script, create safer train-only replay manifests from clean gain and regression patterns, add a match-count preservation loss to GraphMatcher training, then run short gated matcher-only experiments before any longer training.

**Tech Stack:** Python 3, PyTorch, existing `benchmark_lazy_pose_pairs.py`, `pfm_pytorch_training.py`, fov76 lazy manifests, promotion pipeline, HTML/CSV experiment records.

---

## Current Evidence

Use these paths as the authoritative starting point:

```text
Project:
/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch

Data root:
/media/w24/D/xjw深度学习训练数据

Pair root:
/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal

Active baseline checkpoint:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

Current accepted rescue checkpoint:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6a_fov76_phase5g_residual_pattern_replay_4l384_20260616_105709/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt

Rejected phase6b run:
/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6b_fov76_phase6a_gain3_failure_gain_replay_4l384_20260616_194909
```

Observed phase6b result:

```text
promotion: REJECT
failed reason:
formal_target_total/all correct_delta=0 < required_gain=1

Direct phase6b vs phase5g:
formal val:  filtered_correct 2603 vs 2630, delta -27
formal test: filtered_correct 1954 vs 2010, delta -56

Selector phase5g + phase6b:
selected phase5g for every row, no net gain.
```

Interpretation:

```text
phase6b is not worth extending.
The next run must not continue from phase6b blindly.
The main failure mode is match-count/correct-count drop, not dustbin over-rejection.
```

## Files And Responsibilities

Create:

```text
scripts/analyze_fov76_checkpoint_delta.py
```

Reusable report script. It reads `formal_summary.csv`, `formal_variant_summary.csv`, and `dual_checkpoint_rescue_selector/combined_filtered_summary.csv`; writes per-split, per-variant, and per-pair gain/loss CSV/JSON/HTML. This replaces one-off Python snippets.

Create:

```text
python/tests/test_analyze_fov76_checkpoint_delta.py
```

Unit tests for the new delta analyzer.

Modify:

```text
python/pfm_pytorch_training.py
```

Add `graph_matcher_teacher_match_count_floor_loss` and wire it into `train_step`. The loss protects teacher-accepted true matches by comparing student final score against the teacher-selected positive score threshold, without increasing dustbin/no-match supervision.

Modify:

```text
scripts/benchmark_lazy_pose_pairs.py
```

Expose CLI flags for the match-count floor loss, pass them into `train_step`, log metrics into `train_metrics.csv`, and include them in checkpoint input summaries.

Modify:

```text
python/tests/test_pfm_pytorch_training.py
python/tests/test_benchmark_lazy_pose_pairs.py
```

Add tests for the new loss and CLI wiring.

Modify:

```text
scripts/README.md
```

Document the delta analyzer and the match-count floor training flags.

Create:

```text
runs/train_h100_fov076_phase6c_match_count_floor_20260616.sh
runs/eval_h100_fov076_phase6c_expanded_20260616.sh
```

Local run scripts for the short training and expanded validation. These are run artifacts and may remain untracked if `runs/` is ignored.

## Task 1: Delta Analyzer Script

**Files:**
- Create: `scripts/analyze_fov76_checkpoint_delta.py`
- Test: `python/tests/test_analyze_fov76_checkpoint_delta.py`
- Modify: `scripts/README.md`

- [ ] **Step 1: Write analyzer tests**

Create `python/tests/test_analyze_fov76_checkpoint_delta.py`:

```python
import tempfile
import unittest
from pathlib import Path

import analyze_fov76_checkpoint_delta as mod


class AnalyzeFov76CheckpointDeltaTest(unittest.TestCase):
    def test_summarizes_direct_and_selector_deltas(self):
        rows = [
            {
                "source": "formal",
                "split": "val",
                "base_id": "b1",
                "target_variant": "extreme_02",
                "match_delta": "5",
                "correct_delta": "5",
                "wrong_delta": "0",
                "selected_model": "phase5g_active",
                "selector_reason": "blocked_homography_p90:3.8>3.2",
            },
            {
                "source": "formal",
                "split": "test",
                "base_id": "b2",
                "target_variant": "mid_02",
                "match_delta": "-8",
                "correct_delta": "-8",
                "wrong_delta": "0",
                "selected_model": "phase5g_active",
                "selector_reason": "blocked_target_variant:mid_02",
            },
        ]

        summary = mod.summarize_combined_rows(rows)

        self.assertEqual(summary["formal"]["val"]["correct_delta_sum"], 5)
        self.assertEqual(summary["formal"]["test"]["correct_delta_sum"], -8)
        self.assertEqual(summary["formal"]["val"]["gain_rows"], 1)
        self.assertEqual(summary["formal"]["test"]["loss_rows"], 1)
        self.assertEqual(summary["selector_reason_counts"]["blocked_target_variant:mid_02"], 1)

    def test_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            combined = output / "combined_filtered_summary.csv"
            combined.write_text(
                "source,split,base_id,target_variant,match_delta,correct_delta,wrong_delta,selected_model,selector_reason\n"
                "formal,val,b1,extreme_02,5,5,0,phase5g_active,blocked_homography_p90:3.8>3.2\n",
                encoding="utf-8",
            )

            result = mod.run_analysis(combined_csv=combined, output_dir=output)

            self.assertTrue((output / "delta_summary.json").exists())
            self.assertTrue((output / "delta_by_variant.csv").exists())
            self.assertTrue((output / "delta_top_gains.csv").exists())
            self.assertTrue((output / "delta_top_losses.csv").exists())
            self.assertTrue((output / "index.html").exists())
            self.assertEqual(result["formal"]["val"]["correct_delta_sum"], 5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_analyze_fov76_checkpoint_delta
```

Expected before implementation:

```text
ImportError: No module named 'analyze_fov76_checkpoint_delta'
```

- [ ] **Step 3: Implement the analyzer**

Create `scripts/analyze_fov76_checkpoint_delta.py` with:

```python
#!/usr/bin/env python3
"""Summarize fov76 checkpoint delta reports into reusable CSV/HTML diagnostics."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


def _int_value(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(round(float(value)))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def summarize_combined_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    summary: dict[str, dict[str, dict[str, int]]] = {}
    by_variant: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {
            "rows": 0,
            "gain_rows": 0,
            "loss_rows": 0,
            "match_delta_sum": 0,
            "correct_delta_sum": 0,
            "wrong_delta_sum": 0,
        }
    )
    selector_reasons: Counter[str] = Counter()
    for row in rows:
        source = row.get("source", "unknown")
        split = row.get("split", "unknown")
        variant = row.get("target_variant", "unknown")
        match_delta = _int_value(row, "match_delta")
        correct_delta = _int_value(row, "correct_delta")
        wrong_delta = _int_value(row, "wrong_delta")
        source_summary = summary.setdefault(source, {})
        split_summary = source_summary.setdefault(
            split,
            {
                "rows": 0,
                "gain_rows": 0,
                "loss_rows": 0,
                "match_delta_sum": 0,
                "correct_delta_sum": 0,
                "wrong_delta_sum": 0,
            },
        )
        split_summary["rows"] += 1
        split_summary["gain_rows"] += int(correct_delta > 0)
        split_summary["loss_rows"] += int(correct_delta < 0)
        split_summary["match_delta_sum"] += match_delta
        split_summary["correct_delta_sum"] += correct_delta
        split_summary["wrong_delta_sum"] += wrong_delta

        variant_summary = by_variant[(source, split, variant)]
        variant_summary["rows"] += 1
        variant_summary["gain_rows"] += int(correct_delta > 0)
        variant_summary["loss_rows"] += int(correct_delta < 0)
        variant_summary["match_delta_sum"] += match_delta
        variant_summary["correct_delta_sum"] += correct_delta
        variant_summary["wrong_delta_sum"] += wrong_delta

        reason = row.get("selector_reason", "")
        if reason:
            selector_reasons[reason] += 1

    return {
        **summary,
        "by_variant": {
            "|".join(key): value for key, value in sorted(by_variant.items())
        },
        "selector_reason_counts": dict(selector_reasons),
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _top_rows(rows: list[dict[str, str]], *, reverse: bool) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _int_value(row, "correct_delta"),
            -_int_value(row, "wrong_delta"),
            _int_value(row, "match_delta"),
        ),
        reverse=reverse,
    )
    selected = []
    for row in ordered[:50]:
        selected.append(
            {
                "source": row.get("source", ""),
                "split": row.get("split", ""),
                "base_id": row.get("base_id", ""),
                "target_variant": row.get("target_variant", ""),
                "match_delta": _int_value(row, "match_delta"),
                "correct_delta": _int_value(row, "correct_delta"),
                "wrong_delta": _int_value(row, "wrong_delta"),
                "selected_model": row.get("selected_model", ""),
                "selector_reason": row.get("selector_reason", ""),
            }
        )
    return selected


def run_analysis(*, combined_csv: Path, output_dir: Path) -> dict[str, object]:
    rows = _read_rows(combined_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_combined_rows(rows)
    (output_dir / "delta_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    variant_rows = []
    for packed_key, values in summary["by_variant"].items():
        source, split, variant = packed_key.split("|")
        variant_rows.append({"source": source, "split": split, "target_variant": variant, **values})
    _write_csv(
        output_dir / "delta_by_variant.csv",
        variant_rows,
        [
            "source",
            "split",
            "target_variant",
            "rows",
            "gain_rows",
            "loss_rows",
            "match_delta_sum",
            "correct_delta_sum",
            "wrong_delta_sum",
        ],
    )
    top_fields = [
        "source",
        "split",
        "base_id",
        "target_variant",
        "match_delta",
        "correct_delta",
        "wrong_delta",
        "selected_model",
        "selector_reason",
    ]
    _write_csv(output_dir / "delta_top_gains.csv", _top_rows(rows, reverse=True), top_fields)
    _write_csv(output_dir / "delta_top_losses.csv", _top_rows(rows, reverse=False), top_fields)
    html_body = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_dir / "index.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>fov76 delta analysis</title></head>"
        f"<body><h1>fov76 delta analysis</h1><pre>{html_body}</pre></body></html>\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(combined_csv=args.combined_csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run analyzer tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_analyze_fov76_checkpoint_delta
```

Expected:

```text
OK
```

- [ ] **Step 5: Run analyzer on phase6b**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/analyze_fov76_checkpoint_delta.py \
  --combined-csv "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6b_fov76_phase6a_gain3_failure_gain_replay_4l384_20260616_194909/promotion_phase5g_profile/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --output-dir "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase6b_fov76_phase6a_gain3_failure_gain_replay_4l384_20260616_194909/phase6b_delta_analysis"
```

Expected:

```text
delta_summary.json exists
delta_by_variant.csv exists
delta_top_gains.csv exists
delta_top_losses.csv exists
index.html exists
```

- [ ] **Step 6: Update scripts README**

Add this row under "评估与诊断":

```markdown
| `analyze_fov76_checkpoint_delta.py` | 读取 fov76 promotion/selector 的 `combined_filtered_summary.csv`，汇总 direct candidate 相对 baseline 的 per split、per variant、per pair gain/loss，并输出 `delta_summary.json`、`delta_by_variant.csv`、`delta_top_gains.csv`、`delta_top_losses.csv` 和 HTML。 | 用于 phase6b 这类“局部有收益但整体 correct/match count 下降”的候选诊断，决定下一轮 hard replay 和 match-count preservation 训练方向。 |
```

- [ ] **Step 7: Commit analyzer**

Run:

```bash
git add scripts/analyze_fov76_checkpoint_delta.py python/tests/test_analyze_fov76_checkpoint_delta.py scripts/README.md
git commit -m "Add fov76 checkpoint delta analyzer"
```

## Task 2: Build Phase6c Train-Only Replay

**Files:**
- Use existing: `scripts/build_train_replay_from_pair_deltas.py`
- Use existing: `scripts/mine_selector_disagreement_pairs.py`
- Create run artifacts under: `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/hard_mining/phase6c_match_count_floor_YYYYMMDD_HHMMSS/`

- [ ] **Step 1: Define paths**

Use:

```bash
export DATA_ROOT="/media/w24/D/xjw深度学习训练数据"
export PAIR_ROOT="${DATA_ROOT}/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal"
export ACTIVE_RUN="${DATA_ROOT}/pfm_runs/phase6a_fov76_phase5g_residual_pattern_replay_4l384_20260616_105709"
export PHASE6B_RUN="${DATA_ROOT}/pfm_runs/phase6b_fov76_phase6a_gain3_failure_gain_replay_4l384_20260616_194909"
export OUT_ROOT="${PAIR_ROOT}/hard_mining/phase6c_match_count_floor_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"
```

- [ ] **Step 2: Mine active-vs-phase6b regressions**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/mine_selector_disagreement_pairs.py \
  --active-combined-csv "${ACTIVE_RUN}/promotion_phase5g_profile_gain3_expanded200_20260616_192526/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --candidate-combined-csv "${PHASE6B_RUN}/promotion_phase5g_profile/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --pair-root "${PAIR_ROOT}" \
  --mine-mode active_regressions \
  --include-non-target-regressions \
  --output-manifest "${OUT_ROOT}/active_vs_phase6b_regressions.csv" \
  --output-summary-json "${OUT_ROOT}/active_vs_phase6b_regressions_summary.json" \
  --output-html "${OUT_ROOT}/active_vs_phase6b_regressions.html"
```

Expected:

```text
active_vs_phase6b_regressions.csv exists
summary JSON reports at least one candidate_match_drop or candidate_missed_active_correct reason
```

- [ ] **Step 3: Mine clean candidate gains only as patterns**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/mine_selector_disagreement_pairs.py \
  --active-combined-csv "${ACTIVE_RUN}/promotion_phase5g_profile_gain3_expanded200_20260616_192526/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --candidate-combined-csv "${PHASE6B_RUN}/promotion_phase5g_profile/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --pair-root "${PAIR_ROOT}" \
  --mine-mode candidate_gains \
  --max-candidate-wrong-increase 0 \
  --output-manifest "${OUT_ROOT}/phase6b_clean_extreme_gains.csv" \
  --output-summary-json "${OUT_ROOT}/phase6b_clean_extreme_gains_summary.json" \
  --output-html "${OUT_ROOT}/phase6b_clean_extreme_gains.html"
```

Expected:

```text
phase6b_clean_extreme_gains.csv exists
all output rows target only extreme_02 or extreme_03
wrong_delta <= 0 for selected gain rows
```

- [ ] **Step 4: Build train-only replay from patterns**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/build_train_replay_from_pair_deltas.py \
  --train-manifest "${PAIR_ROOT}/overlap_edges_train.csv" \
  --regression-delta-csv "${OUT_ROOT}/active_vs_phase6b_regressions.csv" \
  --gain-delta-csv "${OUT_ROOT}/phase6b_clean_extreme_gains.csv" \
  --output-manifest "${OUT_ROOT}/train_phase6c_match_count_floor_replay.csv" \
  --mixed-output-manifest "${OUT_ROOT}/train_phase6c_match_count_floor_replay_mix10.csv" \
  --mixed-base-manifest "${PAIR_ROOT}/overlap_edges_train.csv" \
  --mixed-replay-fraction 0.10 \
  --max-per-pattern 128 \
  --seed 20260616 \
  --output-html "${OUT_ROOT}/train_phase6c_match_count_floor_replay.html"
```

Expected:

```text
train_phase6c_match_count_floor_replay.csv exists
train_phase6c_match_count_floor_replay_mix10.csv exists
HTML report exists
No val/test pair identity is copied into train; only train manifest rows are sampled.
```

- [ ] **Step 5: Validate replay manifest size**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python - <<'PY'
import csv
from pathlib import Path
path = Path("/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/hard_mining")
latest = sorted(path.glob("phase6c_match_count_floor_*/train_phase6c_match_count_floor_replay_mix10.csv"))[-1]
with latest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
print(latest)
print(len(rows))
assert len(rows) >= 1000
PY
```

Expected:

```text
The printed row count is >= 1000.
```

## Task 3: Add Match-Count Floor Loss

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_pfm_pytorch_training.py`
- Test: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Write loss tests**

Append tests to `python/tests/test_pfm_pytorch_training.py`:

```python
def test_graph_matcher_teacher_match_count_floor_penalizes_lost_teacher_acceptance(self):
    import torch
    import pfm_pytorch_training as training
    from types import SimpleNamespace

    teacher_logits = torch.full((4, 4), -5.0)
    student_logits = torch.full((4, 4), -5.0)
    for i in range(3):
        teacher_logits[i, i] = 5.0
        student_logits[i, i] = 5.0
    student_logits[1, 1] = -2.0
    teacher = SimpleNamespace(logits=teacher_logits)
    student = SimpleNamespace(logits=student_logits)

    loss, metrics = training.graph_matcher_teacher_match_count_floor_loss(
        student,
        teacher,
        positive_count=3,
        teacher_score_threshold=4.0,
        student_score_margin=0.5,
    )

    self.assertGreater(float(loss), 0.0)
    self.assertEqual(float(metrics["teacher_kept"]), 3.0)
    self.assertEqual(float(metrics["student_kept"]), 2.0)
    self.assertEqual(float(metrics["violations"]), 1.0)


def test_graph_matcher_teacher_match_count_floor_zero_when_student_keeps_teacher_matches(self):
    import torch
    import pfm_pytorch_training as training
    from types import SimpleNamespace

    logits = torch.full((3, 3), -5.0)
    logits[0, 0] = 6.0
    logits[1, 1] = 6.0
    student = SimpleNamespace(logits=logits.clone())
    teacher = SimpleNamespace(logits=logits.clone())

    loss, metrics = training.graph_matcher_teacher_match_count_floor_loss(
        student,
        teacher,
        positive_count=2,
        teacher_score_threshold=4.0,
        student_score_margin=0.5,
    )

    self.assertEqual(float(loss), 0.0)
    self.assertEqual(float(metrics["teacher_kept"]), 2.0)
    self.assertEqual(float(metrics["student_kept"]), 2.0)
```

- [ ] **Step 2: Run failing loss tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_pfm_pytorch_training.PfmPytorchTrainingTest.test_graph_matcher_teacher_match_count_floor_penalizes_lost_teacher_acceptance \
  python.tests.test_pfm_pytorch_training.PfmPytorchTrainingTest.test_graph_matcher_teacher_match_count_floor_zero_when_student_keeps_teacher_matches
```

Expected before implementation:

```text
AttributeError: module 'pfm_pytorch_training' has no attribute 'graph_matcher_teacher_match_count_floor_loss'
```

- [ ] **Step 3: Implement loss function**

Add to `python/pfm_pytorch_training.py` near `graph_matcher_teacher_score_floor_loss`:

```python
def graph_matcher_teacher_match_count_floor_loss(
    student: pfm_model.GraphMatcherOutput,
    teacher: pfm_model.GraphMatcherOutput,
    *,
    positive_count: int,
    teacher_score_threshold: float = 0.0,
    student_score_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Protect teacher-kept true matches so calibration does not reduce match count."""

    if student_score_margin < 0.0:
        raise ValueError("student_score_margin must be nonnegative")
    if not math.isfinite(float(teacher_score_threshold)):
        raise ValueError("teacher_score_threshold must be finite")
    count = min(
        int(positive_count),
        student.logits.size(0) - 1,
        student.logits.size(1) - 1,
        teacher.logits.size(0) - 1,
        teacher.logits.size(1) - 1,
    )
    zero = student.logits.new_zeros(())
    metrics = {
        "teacher_kept": zero,
        "student_kept": zero,
        "violations": zero,
        "student_score_mean": zero,
        "teacher_score_mean": zero,
    }
    if count <= 0:
        return zero, metrics
    device = student.logits.device
    indices = torch.arange(count, device=device)
    student_logits = student.logits
    teacher_logits = teacher.logits.detach().to(device=device, dtype=student_logits.dtype)

    def true_final_scores(logits: torch.Tensor) -> torch.Tensor:
        true_logits = logits[:count, :count][indices, indices]
        row_dustbin = logits[:count, logits.size(1) - 1]
        col_dustbin = logits[logits.size(0) - 1, :count]
        return true_logits - row_dustbin - col_dustbin

    teacher_scores = true_final_scores(teacher_logits)
    student_scores = true_final_scores(student_logits)
    protected = teacher_scores >= float(teacher_score_threshold)
    if not bool(protected.any()):
        return zero, metrics

    selected_teacher = teacher_scores[protected]
    selected_student = student_scores[protected]
    required_student = selected_teacher - float(student_score_margin)
    deficits = (required_student - selected_student).clamp_min(0.0)
    loss = deficits.pow(2).mean()
    metrics = {
        "teacher_kept": protected.to(student_logits.dtype).sum().detach(),
        "student_kept": (selected_student >= required_student).to(student_logits.dtype).sum().detach(),
        "violations": deficits.gt(0.0).to(student_logits.dtype).sum().detach(),
        "student_score_mean": selected_student.detach().mean(),
        "teacher_score_mean": selected_teacher.detach().mean(),
    }
    return loss, metrics
```

- [ ] **Step 4: Wire loss into train_step**

Add parameters to `train_step`:

```python
graph_matcher_teacher_match_count_floor_weight: float = 0.0,
graph_matcher_teacher_match_count_floor_threshold: float = 0.0,
graph_matcher_teacher_match_count_floor_margin: float = 0.0,
```

Validate them:

```python
if graph_matcher_teacher_match_count_floor_weight < 0.0:
    raise ValueError("graph_matcher_teacher_match_count_floor_weight must be nonnegative")
if graph_matcher_teacher_match_count_floor_margin < 0.0:
    raise ValueError("graph_matcher_teacher_match_count_floor_margin must be nonnegative")
if not math.isfinite(float(graph_matcher_teacher_match_count_floor_threshold)):
    raise ValueError("graph_matcher_teacher_match_count_floor_threshold must be finite")
```

When `teacher_output` is available, add:

```python
teacher_match_count_floor_loss = output.logits.new_zeros(())
teacher_match_count_floor_metrics = None
if graph_matcher_teacher_match_count_floor_weight > 0.0 and teacher_output is not None:
    teacher_match_count_floor_loss, teacher_match_count_floor_metrics = (
        graph_matcher_teacher_match_count_floor_loss(
            output,
            teacher_output,
            positive_count=count,
            teacher_score_threshold=graph_matcher_teacher_match_count_floor_threshold,
            student_score_margin=graph_matcher_teacher_match_count_floor_margin,
        )
    )
    loss = loss + float(graph_matcher_teacher_match_count_floor_weight) * teacher_match_count_floor_loss
```

Add component metrics:

```python
"graph_matcher_teacher_match_count_floor_loss"
"graph_matcher_teacher_match_count_floor_teacher_kept"
"graph_matcher_teacher_match_count_floor_student_kept"
"graph_matcher_teacher_match_count_floor_violations"
```

- [ ] **Step 5: Add CLI tests**

Append to `python/tests/test_benchmark_lazy_pose_pairs.py`:

```python
def test_parse_graph_matcher_teacher_match_count_floor_args(self):
    args = benchmark_lazy_pose_pairs.parse_args_from_list(
        [
            "train",
            "--graph-matcher-teacher-match-count-floor-weight",
            "0.12",
            "--graph-matcher-teacher-match-count-floor-threshold",
            "18.0",
            "--graph-matcher-teacher-match-count-floor-margin",
            "0.75",
        ]
    )

    self.assertAlmostEqual(args.graph_matcher_teacher_match_count_floor_weight, 0.12)
    self.assertAlmostEqual(args.graph_matcher_teacher_match_count_floor_threshold, 18.0)
    self.assertAlmostEqual(args.graph_matcher_teacher_match_count_floor_margin, 0.75)
```

- [ ] **Step 6: Run failing CLI test**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_benchmark_lazy_pose_pairs.BenchmarkLazyPosePairsTest.test_parse_graph_matcher_teacher_match_count_floor_args
```

Expected before implementation:

```text
unrecognized arguments: --graph-matcher-teacher-match-count-floor-weight
```

- [ ] **Step 7: Implement CLI flags and pass-through**

Add parser flags in `scripts/benchmark_lazy_pose_pairs.py`:

```python
parser.add_argument("--graph-matcher-teacher-match-count-floor-weight", type=float, default=0.0)
parser.add_argument("--graph-matcher-teacher-match-count-floor-threshold", type=float, default=0.0)
parser.add_argument("--graph-matcher-teacher-match-count-floor-margin", type=float, default=0.0)
```

Pass them into `pfm_pytorch_training.train_step`:

```python
graph_matcher_teacher_match_count_floor_weight=args.graph_matcher_teacher_match_count_floor_weight,
graph_matcher_teacher_match_count_floor_threshold=args.graph_matcher_teacher_match_count_floor_threshold,
graph_matcher_teacher_match_count_floor_margin=args.graph_matcher_teacher_match_count_floor_margin,
```

Add metrics to the training metrics field list and HTML/input summary:

```text
graph_matcher_teacher_match_count_floor_loss
graph_matcher_teacher_match_count_floor_teacher_kept
graph_matcher_teacher_match_count_floor_student_kept
graph_matcher_teacher_match_count_floor_violations
```

- [ ] **Step 8: Run focused tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_pfm_pytorch_training \
  python.tests.test_benchmark_lazy_pose_pairs
```

Expected:

```text
OK
```

- [ ] **Step 9: Update README**

Add to `scripts/README.md` under the `benchmark_lazy_pose_pairs.py` notes:

```markdown
备注：`--graph-matcher-teacher-match-count-floor-weight` 使用 teacher checkpoint 的 true-pair final score 作为 match-count floor，只保护 teacher 高置信保留的真匹配，惩罚 student 把这些真匹配分数压低导致 filtered match/correct count 下降；它不增加 dustbin/no-match/rejection 监督，适合 phase6b 这类 precision 近似但 correct/match count 下降的 matcher-only 校准。
```

- [ ] **Step 10: Commit match-count floor**

Run:

```bash
git add python/pfm_pytorch_training.py scripts/benchmark_lazy_pose_pairs.py python/tests/test_pfm_pytorch_training.py python/tests/test_benchmark_lazy_pose_pairs.py scripts/README.md
git commit -m "Add GraphMatcher match-count floor loss"
```

## Task 4: Phase6c Short Training

**Files:**
- Create: `runs/train_h100_fov076_phase6c_match_count_floor_20260616.sh`

- [ ] **Step 1: Check for long tasks**

Run:

```bash
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
```

Expected:

```text
Only the pgrep command itself appears, or no output.
```

- [ ] **Step 2: Create training script**

Create `runs/train_h100_fov076_phase6c_match_count_floor_20260616.sh` from the phase6b script with these changes:

```text
RUN_ROOT="${DATA_ROOT}/pfm_runs/phase6c_fov76_phase6a_match_count_floor_4l384_${STAMP}"
TRAIN_MANIFEST="${OUT_ROOT}/train_phase6c_match_count_floor_replay_mix10.csv"
INIT_STATE="${PHASE6A_STATE}"
TEACHER_STATE="${PHASE5G_STATE}"
--steps 96
--visual-eval-every-steps 48
--learning-rate 1e-11
--graph-matcher-teacher-score-floor-weight 0.10
--graph-matcher-teacher-score-floor-tolerance 0.010
--graph-matcher-teacher-score-floor-min-score 18.0
--graph-matcher-teacher-match-count-floor-weight 0.02
--graph-matcher-teacher-match-count-floor-threshold 18.0
--graph-matcher-teacher-match-count-floor-margin 0.5
--graph-matcher-true-match-margin-weight 0.00035
--graph-matcher-ransac-consistency-weight 0.002
--graph-matcher-ransac-consistency-margin 0.15
--graph-matcher-ransac-consistency-residual-threshold-px 3.0
--graph-matcher-no-match-weight 0.0
--graph-matcher-hard-negative-dustbin-weight 0.0
--no-graph-matcher-online-false-no-match
```

Keep:

```text
--train-graph-matcher
--train-graph-calibration-only
--graph-hidden-dim 384
--graph-attention-layers 4
--matcher-candidate-topk 256
--graph-matcher-train-candidate-topk 256
--matcher-reliability-pair-bias off
--matcher-reliability-dustbin-bias off
--matcher-final-accept-score-mode none
--visual-post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard
```

- [ ] **Step 3: Syntax check**

Run:

```bash
bash -n runs/train_h100_fov076_phase6c_match_count_floor_20260616.sh
```

Expected:

```text
exit code 0
```

- [ ] **Step 4: Start training**

Run:

```bash
setsid bash runs/train_h100_fov076_phase6c_match_count_floor_20260616.sh >/dev/null 2>&1 &
```

Expected:

```text
PID printed by shell or available in runs/train_h100_fov076_phase6c_match_count_floor_*.pid
```

- [ ] **Step 5: Monitor training**

Run every 30-60 seconds:

```bash
tail -n 100 runs/train_h100_fov076_phase6c_match_count_floor_*.log
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
```

Expected:

```text
No NaN.
No early stop.
data_wait median near 0 after warmup.
graph_matcher_teacher_match_count_floor_violations is logged.
```

## Task 5: Phase6c Quick Promotion

**Files:**
- Generated by training script: `${RUN_ROOT}/promotion_phase5g_profile`

- [ ] **Step 1: Inspect quick promotion decision**

Run:

```bash
/home/w24/anaconda3/envs/cppTorch/bin/python -m json.tool \
  "${RUN_ROOT}/promotion_phase5g_profile/promotion_decision.json"
```

Expected pass condition:

```text
"promote": true
formal_target_total/all correct_delta >= 1
regression_guard val/test correct_delta >= 0
regression_guard val/test wrong_delta <= 0
protected mid_01/mid_02/extreme_01/nadir no precision/correct/wrong regression
```

- [ ] **Step 2: If quick promotion rejects**

Stop the phase6c branch and do not run expanded validation.

Record:

```text
promotion_decision.json
formal_summary.csv
formal_variant_summary.csv
dual_checkpoint_rescue_selector/combined_filtered_summary.csv
```

Run the delta analyzer:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/analyze_fov76_checkpoint_delta.py \
  --combined-csv "${RUN_ROOT}/promotion_phase5g_profile/dual_checkpoint_rescue_selector/combined_filtered_summary.csv" \
  --output-dir "${RUN_ROOT}/phase6c_delta_analysis"
```

Decision:

```text
If correct_delta is still negative or zero, keep active selector unchanged.
If correct_delta improves but guard fails, mine guard regression patterns for phase6d.
```

- [ ] **Step 3: If quick promotion passes**

Proceed to Task 6 expanded validation.

## Task 6: Expanded 200/200 Validation

**Files:**
- Create: `runs/eval_h100_fov076_phase6c_expanded_20260616.sh`

- [ ] **Step 1: Create expanded eval script**

Create a script that runs:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/run_fov76_checkpoint_promotion_pipeline.py \
  --pair-root "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal" \
  --guard-root "/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/hard_mining/phase3d_diff_guard_20260614" \
  --output-dir "${RUN_ROOT}/promotion_phase5g_profile_expanded200" \
  --baseline-state "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt" \
  --baseline-run-dir "/media/w24/D/xjw深度学习训练数据/pfm_runs/phase5g_fov76_phase5d_clean_gain_replay_4l384_20260615_122433/train_output" \
  --candidate-state "${RUN_ROOT}/train_output/checkpoints/best_by_match_score_pytorch_pfm_state.pt" \
  --candidate-run-dir "${RUN_ROOT}/train_output" \
  --baseline-label phase5g_active \
  --candidate-label phase6c_match_count_floor \
  --guard-baseline-label phase5g_active \
  --guard-candidate-label phase6c_match_count_floor \
  --dual-checkpoint-rescue-selector \
  --dual-checkpoint-rescue-label phase5g_phase6c_selector \
  --python-executable "/home/w24/anaconda3/envs/cppTorch/bin/python" \
  --post-filter-profile fov76_geo5_geo10_extreme_rescue_lowmatch_guard \
  --formal-candidate-pairs 200 \
  --guard-candidate-pairs 200
```

- [ ] **Step 2: Run expanded eval**

Run:

```bash
setsid bash runs/eval_h100_fov076_phase6c_expanded_20260616.sh >/dev/null 2>&1 &
```

Expected:

```text
promotion_phase5g_profile_expanded200/promotion_decision.json exists
```

- [ ] **Step 3: Expanded promotion gate**

Promotion is accepted only if:

```text
formal_target_total/all correct_delta >= 1
formal target wrong_delta <= 1
formal target precision_delta >= 0
protected variants no regression
regression_guard val/test no regression
extreme_gain val/test no worse than active selector
```

If any condition fails:

```text
Do not upgrade active selector.
Archive phase6c as diagnostic branch.
```

## Task 7: Active Selector Update And Validation

**Files:**
- Modify if phase6c expanded passes: `runs/fov76_active_mainline_config_*.json` or the current active config file used by `validate_fov76_active_selector.py`
- Use: `scripts/validate_fov76_active_selector.py`

- [ ] **Step 1: Locate active config**

Run:

```bash
ls -t runs/fov76_active_mainline_config_*.json | head -n 5
```

Expected:

```text
Current active config file is listed.
```

- [ ] **Step 2: Update active config only if expanded passes**

Set the active selector to:

```text
baseline: phase5g_active
rescue: phase6c_match_count_floor
selector profile: fov76_geo5_geo10_extreme_rescue_lowmatch_guard
min_match_gain: 3
min_rescue_matches: 16
```

Do not remove phase6a config until phase6c passes validation.

- [ ] **Step 3: Validate active selector**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/validate_fov76_active_selector.py \
  --config runs/fov76_active_mainline_config_<updated>.json \
  --output-json runs/fov76_active_mainline_validation_phase6c.json
```

Expected:

```text
validation JSON reports active mainline pass.
```

- [ ] **Step 4: Commit active selector update**

Run:

```bash
git add runs/fov76_active_mainline_config_<updated>.json runs/fov76_active_mainline_validation_phase6c.json
git commit -m "Promote fov76 phase6c selector"
```

Skip this task if phase6c expanded validation does not pass.

## Task 8: Final Verification And GitHub Push

**Files:**
- All tracked code/docs touched above.

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest \
  python.tests.test_analyze_fov76_checkpoint_delta \
  python.tests.test_pfm_pytorch_training \
  python.tests.test_benchmark_lazy_pose_pairs
```

Expected:

```text
OK
```

- [ ] **Step 2: Check long tasks are done**

Run:

```bash
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|run_fov76_checkpoint_promotion_pipeline.py|run_graph_filter_sweep.py|visualize_lazy_pose_matches.py' || true
```

Expected:

```text
No training/eval process remains except the pgrep command itself.
```

- [ ] **Step 3: Check git state**

Run:

```bash
git status -sb
```

Expected:

```text
Only intentional tracked changes staged/committed.
Untracked file `0` remains untouched.
```

- [ ] **Step 4: Push**

Run:

```bash
git push
```

Expected:

```text
origin/main receives analyzer and match-count floor commits.
```

## Decision Rules

Do not promote a model just because visual eval looks good.

Promote only when all of these are true:

```text
1. quick promotion passes.
2. expanded 200/200 promotion passes.
3. formal target extreme_02/extreme_03 total correct increases.
4. protected variants do not regress.
5. regression guard does not regress.
6. wrong count does not increase beyond the gate.
7. match count/correct count are not lower than phase5g on formal val/test.
```

Reject and archive when:

```text
1. correct gain is zero or negative.
2. match count drops materially while precision only slightly improves.
3. protected mid_01/mid_02/extreme_01 regress.
4. selector chooses baseline for all rows.
5. any guard set shows wrong increase.
```

## Expected Outcome

Best case:

```text
phase6c gives +correct on extreme_02/extreme_03 without losing protected variants.
Expanded 200/200 passes.
Active selector is upgraded to phase5g + phase6c.
```

Acceptable outcome:

```text
phase6c still rejects, but delta analyzer and match-count floor remain useful infrastructure.
Active selector stays phase5g + phase6a.
Next plan mines only the clean gain rows that failed selector gates due homography/score thresholds.
```

Bad outcome:

```text
match-count floor prevents all learning, selected model stays phase5g everywhere.
In that case do not raise dustbin/rejection; instead reduce teacher threshold from 18.0 to 16.0 and lower floor weight from 0.02 to 0.01.
```

## Self-Review

Spec coverage:

```text
Uses phase6b failure as evidence.
Keeps active phase5g + phase6a selector unchanged unless stronger evidence appears.
Builds reusable delta diagnostics.
Builds train-only replay from patterns, not val/test rows.
Adds a model-level optimization that directly addresses phase6b match-count loss.
Runs quick promotion before expanded validation.
Pushes code only after tests and result gates.
```

Red-flag scan:

```text
No unresolved filler markers.
All paths, commands, and gates are explicit.
```

Type consistency:

```text
New loss function name:
graph_matcher_teacher_match_count_floor_loss

CLI names:
--graph-matcher-teacher-match-count-floor-weight
--graph-matcher-teacher-match-count-floor-threshold
--graph-matcher-teacher-match-count-floor-margin

Metric prefixes:
graph_matcher_teacher_match_count_floor_*
```
