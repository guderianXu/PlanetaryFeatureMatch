#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/scripts"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${PFM_TMPDIR:-/media/w24/D/xjw深度学习训练数据/tmp}"
mkdir -p "${TMPDIR}"

PY="${PFM_PHASE145_PYTHON:-/home/w24/anaconda3/envs/cppTorch/bin/python}"
PHASE144_ROOT="${PFM_PHASE144_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase144_phase141_wrong_risk_false_replay_train_eval_20260622}"
PHASE119_ROOT="${PFM_PHASE119_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase119_phase118_boundary_false_replay_train_eval_20260621}"
SOURCE_SUBDIR="${PFM_PHASE145_SOURCE_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase144_wrong_risk_false_replay}"
RUN_ROOT="${PFM_PHASE145_RUN_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase145_extreme01_precision_guard_train_eval_20260622}"
FALSE_ROOT="${RUN_ROOT}/false_edges"
DEV_FALSE_CSV="${FALSE_ROOT}/dev_phase144_extreme01_false_matches.csv"
VAL_FALSE_CSV="${FALSE_ROOT}/val_phase144_extreme01_false_matches.csv"
BASE_TRAIN_MANIFEST="${PFM_PHASE145_BASE_TRAIN_MANIFEST:-${PHASE144_ROOT}/manifests/phase122_targeted_false_replay_mixed_train.csv}"

MIN_ERROR_PX="${PFM_PHASE145_MIN_ERROR_PX:-5.0}"
MAX_ERROR_PX="${PFM_PHASE145_MAX_ERROR_PX:-14.0}"
MIN_FALSE_SCORE="${PFM_PHASE145_MIN_FALSE_SCORE:-16.0}"
MIN_ACCEPT_PROBABILITY="${PFM_PHASE145_MIN_ACCEPT_PROBABILITY:-0.70}"
MAX_FALSE_PER_PAIR="${PFM_PHASE145_MAX_FALSE_PER_PAIR:-10}"

for required in \
  "${PHASE144_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt" \
  "${PHASE144_ROOT}/eval/dev_pairs.csv" \
  "${PHASE144_ROOT}/eval/val_pairs.csv" \
  "${PHASE144_ROOT}/eval/dev/${SOURCE_SUBDIR}/all_filtered_match_details.csv" \
  "${PHASE144_ROOT}/eval/val/${SOURCE_SUBDIR}/all_filtered_match_details.csv" \
  "${PHASE119_ROOT}/eval/dev_pairs.csv" \
  "${PHASE119_ROOT}/eval/val_pairs.csv" \
  "${PHASE119_ROOT}/eval/lockbox_pairs.csv" \
  "${BASE_TRAIN_MANIFEST}"
do
  if [[ ! -f "${required}" ]]; then
    echo "[phase145] missing required input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${FALSE_ROOT}"

if [[ "${PFM_PHASE145_REBUILD_FALSE_CSV:-0}" == "1" || ! -f "${DEV_FALSE_CSV}" ]]; then
  "${PY}" scripts/build_lazy_false_match_csv.py \
    --pair-manifest "${PHASE144_ROOT}/eval/dev_pairs.csv" \
    --match-details "${PHASE144_ROOT}/eval/dev/${SOURCE_SUBDIR}/all_filtered_match_details.csv" \
    --output-csv "${DEV_FALSE_CSV}" \
    --summary-json "${FALSE_ROOT}/dev_phase144_extreme01_false_matches_summary.json" \
    --report-html "${FALSE_ROOT}/dev_phase144_extreme01_false_matches_summary.html" \
    --min-error-px "${MIN_ERROR_PX}" \
    --max-error-px "${MAX_ERROR_PX}" \
    --target-variant extreme_01 \
    --min-score "${MIN_FALSE_SCORE}" \
    --min-accept-probability "${MIN_ACCEPT_PROBABILITY}" \
    --max-per-pair "${MAX_FALSE_PER_PAIR}" \
    --matcher graph_matcher \
    --mine-source phase144_dev_extreme01_filtered_true_geometry_wrong
fi

if [[ "${PFM_PHASE145_REBUILD_FALSE_CSV:-0}" == "1" || ! -f "${VAL_FALSE_CSV}" ]]; then
  "${PY}" scripts/build_lazy_false_match_csv.py \
    --pair-manifest "${PHASE144_ROOT}/eval/val_pairs.csv" \
    --match-details "${PHASE144_ROOT}/eval/val/${SOURCE_SUBDIR}/all_filtered_match_details.csv" \
    --output-csv "${VAL_FALSE_CSV}" \
    --summary-json "${FALSE_ROOT}/val_phase144_extreme01_false_matches_summary.json" \
    --report-html "${FALSE_ROOT}/val_phase144_extreme01_false_matches_summary.html" \
    --min-error-px "${MIN_ERROR_PX}" \
    --max-error-px "${MAX_ERROR_PX}" \
    --target-variant extreme_01 \
    --min-score "${MIN_FALSE_SCORE}" \
    --min-accept-probability "${MIN_ACCEPT_PROBABILITY}" \
    --max-per-pair "${MAX_FALSE_PER_PAIR}" \
    --matcher graph_matcher \
    --mine-source phase144_val_extreme01_filtered_true_geometry_wrong
fi

export PFM_PHASE122_ROOT="${PFM_PHASE122_ROOT:-${PHASE144_ROOT}}"
export PFM_PHASE56_MANIFEST_ROOT="${PFM_PHASE56_MANIFEST_ROOT:-${PHASE119_ROOT}}"
export PFM_PHASE123_RUN_ROOT="${PFM_PHASE123_RUN_ROOT:-${RUN_ROOT}}"
export PFM_PHASE123_SOURCE_SUBDIR="${PFM_PHASE123_SOURCE_SUBDIR:-${SOURCE_SUBDIR}}"
export PFM_PHASE123_BASE_TRAIN_MANIFEST="${PFM_PHASE123_BASE_TRAIN_MANIFEST:-${BASE_TRAIN_MANIFEST}}"
export PFM_PHASE123_DEV_FALSE_CSV="${PFM_PHASE123_DEV_FALSE_CSV:-${DEV_FALSE_CSV}}"
export PFM_PHASE123_VAL_FALSE_CSV="${PFM_PHASE123_VAL_FALSE_CSV:-${VAL_FALSE_CSV}}"
export PFM_PHASE123_REBUILD_FALSE_CSV="${PFM_PHASE123_REBUILD_FALSE_CSV:-0}"
export PFM_PHASE123_TARGET_HARD_FRACTION="${PFM_PHASE123_TARGET_HARD_FRACTION:-0.03}"

export PFM_PHASE41_TITLE="${PFM_PHASE41_TITLE:-Phase145 Extreme01 Precision Guard Train/Eval}"
export PFM_PHASE41_GOAL="${PFM_PHASE41_GOAL:-Apply an extreme_01-only precision guard after Phase144 so extreme_01 can move beyond LightGlue without importing extreme_02/03 replay pressure.}"
export PFM_PHASE41_NOTE="${PFM_PHASE41_NOTE:-Only extreme_01 true-geometry wrong edges are mined for this phase; LightGlue labels are not used for training.}"
export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-30}"
export PFM_PHASE41_SAVE_EVERY="${PFM_PHASE41_SAVE_EVERY:-15}"
export PFM_PHASE41_TRAIN_SEED="${PFM_PHASE41_TRAIN_SEED:-20260745}"
export PFM_PHASE41_LEARNING_RATE="${PFM_PHASE41_LEARNING_RATE:-8e-8}"
export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase145_extreme01_guard}"

exec bash runs/phase123_phase122_targeted_false_replay_train_eval_20260621.sh
