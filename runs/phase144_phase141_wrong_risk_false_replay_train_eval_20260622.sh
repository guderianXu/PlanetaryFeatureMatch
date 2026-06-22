#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/scripts"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${PFM_TMPDIR:-/media/w24/D/xjw深度学习训练数据/tmp}"
mkdir -p "${TMPDIR}"

PHASE141_ROOT="${PFM_PHASE141_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase141_extreme01_02_gap_replay_conservative_train_eval_20260622}"
PHASE119_ROOT="${PFM_PHASE119_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase119_phase118_boundary_false_replay_train_eval_20260621}"
PHASE140_PREP_ROOT="${PFM_PHASE140_PREP_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase140_extreme01_02_gap_replay_manifest_20260622}"
PHASE141_SUBDIR="${PFM_PHASE141_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase141_gap_replay_conservative}"
RUN_ROOT="${PFM_PHASE144_RUN_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase144_phase141_wrong_risk_false_replay_train_eval_20260622}"

for required in \
  "${PHASE141_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt" \
  "${PHASE141_ROOT}/eval/dev_pairs.csv" \
  "${PHASE141_ROOT}/eval/val_pairs.csv" \
  "${PHASE141_ROOT}/eval/dev/${PHASE141_SUBDIR}/all_filtered_match_details.csv" \
  "${PHASE141_ROOT}/eval/val/${PHASE141_SUBDIR}/all_filtered_match_details.csv" \
  "${PHASE119_ROOT}/eval/dev_pairs.csv" \
  "${PHASE119_ROOT}/eval/val_pairs.csv" \
  "${PHASE119_ROOT}/eval/lockbox_pairs.csv" \
  "${PHASE140_PREP_ROOT}/phase140_gap_replay_mixed_train.csv"
do
  if [[ ! -f "${required}" ]]; then
    echo "[phase144] missing required input: ${required}" >&2
    exit 1
  fi
done

export PFM_PHASE122_ROOT="${PFM_PHASE122_ROOT:-${PHASE141_ROOT}}"
export PFM_PHASE56_MANIFEST_ROOT="${PFM_PHASE56_MANIFEST_ROOT:-${PHASE119_ROOT}}"
export PFM_PHASE123_RUN_ROOT="${PFM_PHASE123_RUN_ROOT:-${RUN_ROOT}}"
export PFM_PHASE123_SOURCE_SUBDIR="${PFM_PHASE123_SOURCE_SUBDIR:-${PHASE141_SUBDIR}}"
export PFM_PHASE123_BASE_TRAIN_MANIFEST="${PFM_PHASE123_BASE_TRAIN_MANIFEST:-${PHASE140_PREP_ROOT}/phase140_gap_replay_mixed_train.csv}"
export PFM_PHASE123_TARGET_HARD_FRACTION="${PFM_PHASE123_TARGET_HARD_FRACTION:-0.04}"
export PFM_PHASE123_MIN_ERROR_PX="${PFM_PHASE123_MIN_ERROR_PX:-5.0}"
export PFM_PHASE123_MAX_ERROR_PX="${PFM_PHASE123_MAX_ERROR_PX:-14.0}"
export PFM_PHASE123_MIN_FALSE_SCORE="${PFM_PHASE123_MIN_FALSE_SCORE:-16.0}"
export PFM_PHASE123_MIN_ACCEPT_PROBABILITY="${PFM_PHASE123_MIN_ACCEPT_PROBABILITY:-0.70}"
export PFM_PHASE123_MAX_FALSE_PER_PAIR="${PFM_PHASE123_MAX_FALSE_PER_PAIR:-10}"

export PFM_PHASE41_TITLE="${PFM_PHASE41_TITLE:-Phase144 Phase141 Wrong-Risk False Replay Train/Eval}"
export PFM_PHASE41_GOAL="${PFM_PHASE41_GOAL:-Reduce the Phase141 pure-PFM +108 wrong-match excess by replaying train-side hard false edges mined from Phase141 dev/val wrong-risk behavior.}"
export PFM_PHASE41_NOTE="${PFM_PHASE41_NOTE:-Uses true-geometry wrong edges from Phase141 match details; no LightGlue labels are used for training.}"
export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-30}"
export PFM_PHASE41_SAVE_EVERY="${PFM_PHASE41_SAVE_EVERY:-15}"
export PFM_PHASE41_TRAIN_SEED="${PFM_PHASE41_TRAIN_SEED:-20260744}"
export PFM_PHASE41_LEARNING_RATE="${PFM_PHASE41_LEARNING_RATE:-1e-7}"
export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase144_wrong_risk_false_replay}"

exec bash runs/phase123_phase122_targeted_false_replay_train_eval_20260621.sh
