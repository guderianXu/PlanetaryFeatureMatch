# Mined False-Match Training Design

## Goal

Improve PFM raw matching precision on viewpoint and compound cases by training directly against false matches the current PFM evaluator actually emits.

## Problem

The previous abstention loss reduced random high-similarity descriptor candidates, but sparse raw matching did not improve. The likely mismatch is that evaluation uses `cyclic_descriptor_similarity()` plus mutual nearest matching, while the loss used ordinary pairwise dot similarity over sampled grid candidates. Training should target accepted false matches, not generic distant descriptors.

## Chosen Approach

Add a mined false-match loop:

- A mining script runs the current PFM checkpoint over train-split cache pairs using the same raw evaluator settings used for reports.
- For every accepted raw match, the script samples the synthetic warp and writes only wrong matches to a CSV.
- Training reads that CSV, scales the image-space false match points to descriptor-grid coordinates, samples both descriptors, and applies a cyclic negative loss to suppress the exact false A/B descriptor pairs.
- The loss is paired with existing P1 positive pseudo labels in a short probe. Positive labels preserve useful descriptor matches; mined false labels suppress the red-line pairs.

## Interfaces

New CLI options in `pfm_pytorch_training.py`:

- `--false-match-csv`
- `--false-match-weight`
- `--false-match-max-points`
- `--false-match-max-score`
- `--false-match-curriculum-max-probability`
- `--false-match-curriculum-warmup-steps`

New pass-through options in `cross_view_experiment.py` use the same names.

New mining script:

- `scripts/mine_pfm_false_matches.py`
- Output CSV columns: `pair_pt,ax,ay,bx,by,error_px,score,margin,style,gate`

## Evaluation

Run focused unit tests first. Then mine train-split false matches from viewpoint/compound groups and run a short 80-step probe from the current routed base checkpoint. Promotion requires same-split raw sparse evaluation to improve precision without losing meaningful correct matches.
