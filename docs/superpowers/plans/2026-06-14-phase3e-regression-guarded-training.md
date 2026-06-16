# Phase3e Regression-Guarded Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a smaller regression-guarded fov76 matcher-only experiment that tries to keep phase3d's extreme_02/extreme_03 gains without degrading val or mid variants.

**Architecture:** Do not change model code or increase model capacity. Start from the current phase2h best checkpoint, freeze extractor/detector/descriptor, train only the existing 4-layer/384 GraphMatcher with lower false-edge weights, then evaluate against phase2h on full val/test plus guard manifests.

**Tech Stack:** Bash run scripts, Python/PyTorch training via `scripts/benchmark_lazy_pose_pairs.py`, existing MAGSAC eval via `scripts/run_graph_filter_sweep.py`.

---

### Task 1: Create Phase3e Training Script

**Files:**
- Create: `runs/train_h100_fov076_phase3e_regression_guarded_20260614.sh`
- Reference: `runs/train_h100_fov076_phase3d_residual_falseedge_20260614.sh`

- [ ] **Step 1: Create the script from the phase3d template**

Use the same dataset, manifest, checkpoint, model dimensions, AMP, checkpointing, visual eval, and stability settings as phase3d.

- [ ] **Step 2: Apply phase3e changes**

Use these exact differences:

```text
RUN_ROOT suffix: phase3e_fov76_regression_guarded_4l384
steps: 240
graph_matcher_final_false_match_weight: 0.001
graph_matcher_mined_false_match_weight: 0.0005
false_match_mine_every: 6
HTML title: fov76 phase3e regression-guarded matcher training
```

Keep these invariants:

```text
graph_hidden_dim = 384
graph_attention_layers = 4
matcher_candidate_topk = 256
freeze_extractor_warmup_steps = 999999
no_match_prior_weight = 0
graph_matcher_no_match_weight = 0
graph_matcher_hard_negative_dustbin_weight = 0
matcher_reliability_pair_bias = off
matcher_reliability_dustbin_bias = off
```

- [ ] **Step 3: Validate script syntax**

Run:

```bash
bash -n runs/train_h100_fov076_phase3e_regression_guarded_20260614.sh
```

Expected: no output and exit code 0.

### Task 2: Launch and Monitor Phase3e

**Files:**
- Run: `runs/train_h100_fov076_phase3e_regression_guarded_20260614.sh`
- Output: `/media/w24/D/xjw深度学习训练数据/pfm_runs/phase3e_fov76_regression_guarded_4l384_<STAMP>/train_output`

- [ ] **Step 1: Check no conflicting long tasks**

Run:

```bash
pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py|run_graph_filter_sweep.py' || true
```

Expected: no real training or eval process, ignoring the `pgrep` command itself.

- [ ] **Step 2: Launch training**

Run:

```bash
setsid bash runs/train_h100_fov076_phase3e_regression_guarded_20260614.sh > runs/train_h100_fov076_phase3e_regression_guarded_launch.log 2>&1 &
```

- [ ] **Step 3: Monitor until completion**

Tail the generated `runs/train_h100_fov076_phase3e_regression_guarded_<STAMP>.log`.

Expected:

```text
finished run_root=/media/w24/D/xjw深度学习训练数据/pfm_runs/phase3e_fov76_regression_guarded_4l384_<STAMP>
```

Also check:

```text
true_match_rejected_by_dustbin_ratio remains near 0
no NaN or early stop
checkpoints include best_by_ransac_inlier, best_by_match_score, best_by_extreme_score
```

### Task 3: Formal Full Eval

**Files:**
- Reuse: `runs/eval_phase3b_fov76_vs_phase2h_ransac_20260614.sh`

- [ ] **Step 1: Evaluate `best_by_ransac_inlier`**

Run the existing eval script with:

```bash
PHASE3B_RUN=/media/w24/D/xjw深度学习训练数据/pfm_runs/phase3e_fov76_regression_guarded_4l384_<STAMP>
PHASE3B_STATE=/media/w24/D/xjw深度学习训练数据/pfm_runs/phase3e_fov76_regression_guarded_4l384_<STAMP>/train_output/checkpoints/best_by_ransac_inlier_pytorch_pfm_state.pt
PHASE3B_LABEL=phase3e_ransac
```

- [ ] **Step 2: Evaluate `best_by_match_score` if needed**

Run the same script with `PHASE3B_LABEL=phase3e_match`.

- [ ] **Step 3: Promote only if full val/test passes**

Promotion gate:

```text
val correct >= phase2h val correct
test correct >= phase2h test correct
val precision not lower by more than 0.001
mid_01/mid_02 precision not lower
extreme_02/extreme_03 correct improves or zero-count decreases
```

### Task 4: Guard Eval

**Files:**
- Use guard manifests:
  - `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/hard_mining/phase3d_diff_guard_20260614/regression_guard_val.csv`
  - `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/hard_mining/phase3d_diff_guard_20260614/regression_guard_test.csv`
  - `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/hard_mining/phase3d_diff_guard_20260614/extreme_gain_val.csv`
  - `/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/hard_mining/phase3d_diff_guard_20260614/extreme_gain_test.csv`

- [ ] **Step 1: Run phase2h and phase3e on guard manifests**

Use `scripts/run_graph_filter_sweep.py` with `--pair-spec-manifest`, `--candidate-pairs 0`, `--no-shuffle`, local10 prefilter, MAGSAC filtered, and min matches 16.

- [ ] **Step 2: Compare guard metrics**

Regression guard must not get worse than phase2h. Extreme gain set should keep at least most of phase3d's benefit without increasing wrong matches.

### Task 5: Decision Record

**Files:**
- Create: `runs/phase3e_fov76_checkpoint_decision_20260614.html`

- [ ] **Step 1: Record formal eval and guard eval metrics**

Include full val/test metrics, variant metrics, guard metrics, checkpoint paths, and final decision.

- [ ] **Step 2: Select next action**

If phase3e passes gates, promote it as the new best. Otherwise keep phase2h and move to post-filter calibration or training-side guard hook.
