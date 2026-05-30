# Abstention-Aware Descriptor Training Design

## Goal

Reduce PFM over-activation on cross-height viewpoint and compound pairs by adding an explicit loss term that suppresses high descriptor similarity to geometrically wrong targets.

## Current Problem

Existing pseudo-label and synthetic descriptor losses mostly say "these points should match." They do not strongly say "distant non-corresponding targets should stay below a safe similarity threshold." Prior P1 probes showed this failure mode: correct matches increased slightly, but false matches increased more, lowering precision and making checkpoints non-routeable.

## Chosen Approach

Add a `descriptor_false_match_suppression_loss` to `python/pfm_pytorch_training.py`.

- Inputs: descriptor maps A/B and known feature-grid correspondence points.
- Candidate set: sampled grid descriptors from B, reusing the existing deterministic candidate helper.
- Mask: candidates within a configurable radius of the known target point are excluded, so nearby positives are not punished.
- Penalty: for each A query, take the highest `topk` false-candidate similarities and penalize values above an absolute `max_false_score`.
- Integration: `descriptor_map_pair_loss()` accepts an `abstention_weight` and adds this loss for both synthetic correspondences and pseudo-label correspondences.

This is intentionally conservative: it does not add a new model head, does not change inference, and can be enabled or disabled entirely through CLI flags.

## CLI

Add these options to `pfm_pytorch_training.py` and pass them through `cross_view_experiment.py`:

- `--abstention-weight`
- `--abstention-negative-radius`
- `--abstention-max-false-score`
- `--abstention-topk`
- `--abstention-candidates`

Default weight is `0.0`, preserving existing behavior.

## Evaluation Plan

First run focused tests. Then run a short viewpoint-only P1-style experiment with conservative settings, checking whether validation and sparse evaluation avoid the previous over-activation pattern. Do not promote the checkpoint unless both fixed64 and full-val behavior are non-regressive.
