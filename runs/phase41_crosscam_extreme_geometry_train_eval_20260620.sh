#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PFM_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/scripts"

PY="${PFM_PYTHON:-${PY:-/home/w24/anaconda3/envs/cppTorch/bin/python}}"
RUN_ROOT="${PFM_PHASE41_TRAIN_ROOT:-runs/phase41_crosscam_extreme_geometry_train_20260620}"
DATA_ROOT="${PFM_PHASE41_DATA_ROOT:-runs/phase41_crosscam_extreme_geometry_20260620}"
DEFAULT_INIT_STATE="runs/phase40_crosscam_extreme_geometry_formal_train_20260620/train_output/checkpoints/last_good_pytorch_pfm_state.pt"
if [[ ! -f "${DEFAULT_INIT_STATE}" ]]; then
  DEFAULT_INIT_STATE="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase39a_devlock_failure_replay_main_4l384_20260620_110647/train_output/checkpoints/last_good_pytorch_pfm_state.pt"
fi
INIT_STATE="${PFM_PHASE41_INIT_STATE:-${DEFAULT_INIT_STATE}}"
RENDER_MANIFEST="${PFM_PHASE41_RENDER_MANIFEST:-${PFM_RENDER_MANIFEST:-/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/manifests/h100km_fov076_render_manifest.csv}}"
UINT8_MANIFEST="${PFM_PHASE41_UINT8_MANIFEST:-${PFM_UINT8_MANIFEST:-/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/manifests/h100km_fov076_uint8_manifest.csv}}"
STEPS="${PFM_PHASE41_STEPS:-400}"
SAVE_EVERY="${PFM_PHASE41_SAVE_EVERY:-100}"
SEED="${PFM_PHASE41_TRAIN_SEED:-20260621}"
LEARNING_RATE="${PFM_PHASE41_LEARNING_RATE:-4e-6}"
TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-1}"
FREEZE_EXTRACTOR_WARMUP_STEPS="${PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS:-0}"
TEACHER_WEIGHT="${PFM_PHASE41_TEACHER_WEIGHT:-1.0}"
SYNTHETIC_LOSS_WEIGHT="${PFM_PHASE41_SYNTHETIC_LOSS_WEIGHT:-1.0}"
HARD_NEGATIVE_WEIGHT="${PFM_PHASE41_HARD_NEGATIVE_WEIGHT:-0.6}"
WARP_HARD_NEGATIVE_WEIGHT="${PFM_PHASE41_WARP_HARD_NEGATIVE_WEIGHT:-0.15}"
SELECTED_KEYPOINT_OFFSET_WEIGHT="${PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT:-0.03}"
PAIR_ACCEPT_LOSS_WEIGHT="${PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT:-0.20}"
GRAPH_MATCHER_ACCEPT_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT:-0.12}"
GRAPH_MATCHER_PRUNE_RANKING_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_PRUNE_RANKING_WEIGHT:-0.06}"
GRAPH_MATCHER_STOP_CONFIDENCE_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_STOP_CONFIDENCE_WEIGHT:-0.03}"
GRAPH_MATCHER_POSITIVE_DUSTBIN_MARGIN_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_POSITIVE_DUSTBIN_MARGIN_WEIGHT:-0.006}"
GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT:-0.08}"
GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT:-0.0}"
GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_THRESHOLD="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_THRESHOLD:-0.0}"
GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_MARGIN="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_MARGIN:-0.0}"
FINAL_FALSE_MATCH_WEIGHT="${PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT:-0.015}"
FINAL_FALSE_MATCH_TOPK="${PFM_PHASE41_FINAL_FALSE_MATCH_TOPK:-12}"
MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.0}"
MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.0}"
MINED_FALSE_MATCH_REFERENCE_MARGIN="${PFM_PHASE41_MINED_FALSE_MATCH_REFERENCE_MARGIN:--1.0}"
FALSE_MATCH_CSV="${PFM_PHASE41_FALSE_MATCH_CSV:-}"
RAW_FALSE_MATCH_WEIGHT="${PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT:-0.006}"
RAW_FALSE_MATCH_TOPK="${PFM_PHASE41_RAW_FALSE_MATCH_TOPK:-8}"
WARP_OUTLIER_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_WEIGHT:-0.08}"
WARP_OUTLIER_TOPK="${PFM_PHASE41_WARP_OUTLIER_TOPK:-12}"
WARP_OUTLIER_RESIDUAL_THRESHOLD_PX="${PFM_PHASE41_WARP_OUTLIER_RESIDUAL_THRESHOLD_PX:-2.0}"
WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.15}"
WARP_OUTLIER_ACCEPT_TOPK="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_TOPK:-12}"
WARP_OUTLIER_ACCEPT_RESIDUAL_THRESHOLD_PX="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_RESIDUAL_THRESHOLD_PX:-2.0}"
WARP_SOFT_BOUNDARY_WEIGHT="${PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT:-0.0}"
WARP_SOFT_BOUNDARY_TOPK="${PFM_PHASE41_WARP_SOFT_BOUNDARY_TOPK:-12}"
WARP_SOFT_BOUNDARY_LOWER_RESIDUAL_PX="${PFM_PHASE41_WARP_SOFT_BOUNDARY_LOWER_RESIDUAL_PX:-5.0}"
WARP_SOFT_BOUNDARY_UPPER_RESIDUAL_PX="${PFM_PHASE41_WARP_SOFT_BOUNDARY_UPPER_RESIDUAL_PX:-8.0}"
WARP_SOFT_BOUNDARY_MIN_SCORE="${PFM_PHASE41_WARP_SOFT_BOUNDARY_MIN_SCORE:-0.0}"
FALSE_CLUSTER_REPLAY_MULTIPLIER="${PFM_PHASE41_FALSE_CLUSTER_REPLAY_MULTIPLIER:-1.0}"
TRAIN_SAMPLES_PER_PAIR="${PFM_PHASE41_TRAIN_SAMPLES_PER_PAIR:-256}"
TRAIN_SPATIAL_BINS="${PFM_PHASE41_TRAIN_SPATIAL_BINS:-8}"
TRAIN_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_TRAIN_MATCHER_CANDIDATE_TOPK:-256}"
TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK:-${TRAIN_MATCHER_CANDIDATE_TOPK}}"
TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE41_TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE:-none}"
TRAIN_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${PFM_PHASE41_TRAIN_MATCHER_FINAL_ACCEPT_SCORE_ALPHA:-0.05}"
TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${DATA_ROOT}/train_pairs_geometry_accept.csv}"
TRAIN_OUT="${RUN_ROOT}/train_output"
EVAL_ROOT="${RUN_ROOT}/eval"
EVAL_MAX_KEYPOINTS="${PFM_PHASE41_EVAL_MAX_KEYPOINTS:-4096}"
EVAL_KEYPOINT_SPATIAL_BINS="${PFM_PHASE41_EVAL_KEYPOINT_SPATIAL_BINS:-16}"
EVAL_KEYPOINT_CELL_CAP="${PFM_PHASE41_EVAL_KEYPOINT_CELL_CAP:-8}"
EVAL_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK:-256}"
EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE:-}"
EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA:--1.0}"
PFM_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp${EVAL_MAX_KEYPOINTS}}"
RUN_TITLE="${PFM_PHASE41_TITLE:-Phase41 Cross-Camera Extreme Geometry Gate Train}"
RUN_GOAL="${PFM_PHASE41_GOAL:-Improve PFM on extreme cross-camera pairs using dense warp supervision and true-geometry pair acceptance labels.}"
RUN_NOTE="${PFM_PHASE41_NOTE:-LightGlue is used only after training as a baseline metric, not as labels or distillation.}"
GATE_THRESHOLDS="${PFM_PHASE41_GATE_THRESHOLDS:-0.3,0.5,0.7}"
GEOMETRY_OVERLAP_GATE_THRESHOLD="${PFM_PHASE41_GEOMETRY_OVERLAP_GATE_THRESHOLD:-}"
GEOMETRY_OVERLAP_GATE_THRESHOLDS="${PFM_PHASE41_GEOMETRY_OVERLAP_GATE_THRESHOLDS:-0.02,0.08,0.10,0.12,0.15,0.20,0.25,0.30}"

for required in \
  "${INIT_STATE}" \
  "${RENDER_MANIFEST}" \
  "${UINT8_MANIFEST}" \
  "${TRAIN_MANIFEST}" \
  "${DATA_ROOT}/dev_pairs.csv" \
  "${DATA_ROOT}/val_pairs.csv" \
  "${DATA_ROOT}/lockbox_pairs.csv"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}" "${TRAIN_OUT}" "${EVAL_ROOT}"
cp "${DATA_ROOT}/dev_pairs.csv" "${EVAL_ROOT}/dev_pairs.csv"
cp "${DATA_ROOT}/val_pairs.csv" "${EVAL_ROOT}/val_pairs.csv"
cp "${DATA_ROOT}/lockbox_pairs.csv" "${EVAL_ROOT}/lockbox_pairs.csv"

read -r INIT_SHA256 _ < <(sha256sum "${INIT_STATE}")
TRAIN_ROWS="$(($(wc -l < "${TRAIN_MANIFEST}") - 1))"
DEV_ROWS="$(($(wc -l < "${DATA_ROOT}/dev_pairs.csv") - 1))"
VAL_ROWS="$(($(wc -l < "${DATA_ROOT}/val_pairs.csv") - 1))"
LOCKBOX_ROWS="$(($(wc -l < "${DATA_ROOT}/lockbox_pairs.csv") - 1))"
DISK_FREE_PROJECT="$(df -h . | awk 'NR == 2 {print $4}')"
ACTIVE_TASKS="$(pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py|run_lightglue' || true)"
TRAIN_EXTRA_FLAGS=()
if [[ "${PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY:-0}" == "1" ]]; then
  TRAIN_EXTRA_FLAGS+=(--train-pair-accept-head-only)
fi
if [[ "${PFM_PHASE41_TRAIN_KEYPOINT_OFFSET_HEAD_ONLY:-0}" == "1" ]]; then
  TRAIN_EXTRA_FLAGS+=(--train-keypoint-offset-head-only)
fi
if [[ "${PFM_PHASE41_TRAIN_GRAPH_CALIBRATION_ONLY:-0}" == "1" ]]; then
  TRAIN_EXTRA_FLAGS+=(--train-graph-calibration-only)
fi
TRAIN_DESCRIPTOR_FLAGS=()
if [[ "${TRAIN_DESCRIPTOR_HEAD}" == "0" ]]; then
  TRAIN_DESCRIPTOR_FLAGS+=(--no-train-descriptor-head)
elif [[ "${TRAIN_DESCRIPTOR_HEAD}" != "1" ]]; then
  echo "PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD must be 0 or 1, got ${TRAIN_DESCRIPTOR_HEAD}" >&2
  exit 1
fi
FALSE_MATCH_FLAGS=()
if [[ -n "${FALSE_MATCH_CSV}" ]]; then
  IFS=':' read -r -a FALSE_MATCH_CSV_ITEMS <<< "${FALSE_MATCH_CSV}"
  for false_match_csv in "${FALSE_MATCH_CSV_ITEMS[@]}"; do
    if [[ -z "${false_match_csv}" ]]; then
      continue
    fi
    if [[ ! -f "${false_match_csv}" ]]; then
      echo "missing false match CSV: ${false_match_csv}" >&2
      exit 1
    fi
    FALSE_MATCH_FLAGS+=(--false-match-csv "${false_match_csv}")
  done
fi

cat > "${RUN_ROOT}/record.html" <<HTML
<!doctype html><meta charset="utf-8">
<title>${RUN_TITLE}</title>
<h1>${RUN_TITLE}</h1>
<p>stage=<code>running</code></p>
<p>goal=<code>${RUN_GOAL}</code></p>
<p>note=<code>${RUN_NOTE}</code></p>
<p>init_state=<code>${INIT_STATE}</code></p>
<p>init_sha256=<code>${INIT_SHA256}</code></p>
<p>train_manifest=<code>${TRAIN_MANIFEST}</code></p>
<p>train_rows=<code>${TRAIN_ROWS}</code></p>
<p>dev_rows=<code>${DEV_ROWS}</code> val_rows=<code>${VAL_ROWS}</code> lockbox_rows=<code>${LOCKBOX_ROWS}</code></p>
<p>steps=<code>${STEPS}</code> save_every=<code>${SAVE_EVERY}</code> seed=<code>${SEED}</code></p>
<p>learning_rate=<code>${LEARNING_RATE}</code></p>
<p>train_descriptor_head=<code>${TRAIN_DESCRIPTOR_HEAD}</code></p>
<p>train_keypoint_offset_head_only=<code>${PFM_PHASE41_TRAIN_KEYPOINT_OFFSET_HEAD_ONLY:-0}</code></p>
<p>train_graph_calibration_only=<code>${PFM_PHASE41_TRAIN_GRAPH_CALIBRATION_ONLY:-0}</code></p>
<p>freeze_extractor_warmup_steps=<code>${FREEZE_EXTRACTOR_WARMUP_STEPS}</code></p>
<p>descriptor_losses teacher/synthetic/hard/warp_hard=<code>${TEACHER_WEIGHT}/${SYNTHETIC_LOSS_WEIGHT}/${HARD_NEGATIVE_WEIGHT}/${WARP_HARD_NEGATIVE_WEIGHT}</code></p>
<p>train_samples_per_pair=<code>${TRAIN_SAMPLES_PER_PAIR}</code> train_spatial_bins=<code>${TRAIN_SPATIAL_BINS}</code></p>
<p>train_matcher_candidate_topk=<code>${TRAIN_MATCHER_CANDIDATE_TOPK}</code> train_graph_matcher_candidate_topk=<code>${TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK}</code></p>
<p>train_matcher_final_accept_score=<code>${TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE}/${TRAIN_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}</code></p>
<p>selected_keypoint_offset_weight=<code>${SELECTED_KEYPOINT_OFFSET_WEIGHT}</code></p>
<p>pair_accept_loss_weight=<code>${PAIR_ACCEPT_LOSS_WEIGHT}</code> graph_accept/prune/stop=<code>${GRAPH_MATCHER_ACCEPT_WEIGHT}/${GRAPH_MATCHER_PRUNE_RANKING_WEIGHT}/${GRAPH_MATCHER_STOP_CONFIDENCE_WEIGHT}</code></p>
<p>positive_dustbin/true_match_margin=<code>${GRAPH_MATCHER_POSITIVE_DUSTBIN_MARGIN_WEIGHT}/${GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT}</code></p>
<p>true_geometry_match_count_floor weight/threshold/margin=<code>${GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT}/${GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_THRESHOLD}/${GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_MARGIN}</code></p>
<p>false_match_csv=<code>${FALSE_MATCH_CSV:-disabled}</code></p>
<p>precision_guard_final_false=<code>${FINAL_FALSE_MATCH_WEIGHT}/${FINAL_FALSE_MATCH_TOPK}</code> mined_false=<code>${MINED_FALSE_MATCH_WEIGHT}/${MINED_FALSE_MATCH_LOSS_CAP}/${MINED_FALSE_MATCH_REFERENCE_MARGIN}</code> raw_false=<code>${RAW_FALSE_MATCH_WEIGHT}/${RAW_FALSE_MATCH_TOPK}</code></p>
<p>precision_guard_warp_outlier=<code>${WARP_OUTLIER_WEIGHT}/${WARP_OUTLIER_TOPK}/${WARP_OUTLIER_RESIDUAL_THRESHOLD_PX}</code> warp_outlier_accept=<code>${WARP_OUTLIER_ACCEPT_WEIGHT}/${WARP_OUTLIER_ACCEPT_TOPK}/${WARP_OUTLIER_ACCEPT_RESIDUAL_THRESHOLD_PX}</code></p>
<p>warp_soft_boundary weight/topk/lower/upper/min_score=<code>${WARP_SOFT_BOUNDARY_WEIGHT}/${WARP_SOFT_BOUNDARY_TOPK}/${WARP_SOFT_BOUNDARY_LOWER_RESIDUAL_PX}/${WARP_SOFT_BOUNDARY_UPPER_RESIDUAL_PX}/${WARP_SOFT_BOUNDARY_MIN_SCORE}</code></p>
<p>false_cluster_replay_multiplier=<code>${FALSE_CLUSTER_REPLAY_MULTIPLIER}</code></p>
<p>eval_max_keypoints=<code>${EVAL_MAX_KEYPOINTS}</code> eval_bins/cellcap=<code>${EVAL_KEYPOINT_SPATIAL_BINS}/${EVAL_KEYPOINT_CELL_CAP}</code></p>
<p>eval_matcher_candidate_topk=<code>${EVAL_MATCHER_CANDIDATE_TOPK}</code> eval_subdir=<code>${PFM_EVAL_SUBDIR}</code></p>
<p>eval_matcher_final_accept_score_mode=<code>${EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE}</code></p>
<p>eval_matcher_final_accept_score_alpha=<code>${EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}</code></p>
<p>gate_thresholds=<code>${GATE_THRESHOLDS}</code></p>
<p>geometry_overlap_gate_threshold=<code>${GEOMETRY_OVERLAP_GATE_THRESHOLD:-disabled}</code></p>
<p>geometry_overlap_gate_thresholds=<code>${GEOMETRY_OVERLAP_GATE_THRESHOLDS}</code></p>
<p>disk_free_project=<code>${DISK_FREE_PROJECT}</code></p>
<p>active_tasks_at_start=<pre>${ACTIVE_TASKS}</pre></p>
HTML

"${PY}" scripts/benchmark_lazy_pose_pairs.py \
  --render-manifest "${RENDER_MANIFEST}" \
  --uint8-manifest "${UINT8_MANIFEST}" \
  --output-dir "${TRAIN_OUT}" \
  --mode train \
  --split all \
  --pair-spec-manifest "${TRAIN_MANIFEST}" \
  --pair-mode cross-camera \
  --pair-type-weights same_position_view=0.0,cross_camera=1.0,cross_fov=0.0 \
  --image-source uint8 \
  --pairs "${TRAIN_ROWS}" \
  --steps "${STEPS}" \
  --workers 2 \
  --prefetch-batches 4 \
  --worker-cache-items 24 \
  --crop-size 2048 \
  --max-attempts 4 \
  --min-valid-fraction 0.02 \
  --absolute-depth-tolerance-m 100.0 \
  --relative-depth-tolerance 0.005 \
  --seed "${SEED}" \
  --shuffle \
  --progress-every 10 \
  --save-every-steps "${SAVE_EVERY}" \
  --device cuda \
  --amp \
  --activation-checkpointing \
  --init-pytorch-state "${INIT_STATE}" \
  --graph-hidden-dim 384 \
  --graph-attention-layers 4 \
  --batch-pairs 1 \
  --samples-per-pair "${TRAIN_SAMPLES_PER_PAIR}" \
  --training-spatial-bins "${TRAIN_SPATIAL_BINS}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay 1e-4 \
  --freeze-extractor-warmup-steps "${FREEZE_EXTRACTOR_WARMUP_STEPS}" \
  --teacher-weight "${TEACHER_WEIGHT}" \
  --synthetic-loss-weight "${SYNTHETIC_LOSS_WEIGHT}" \
  --hard-negative-weight "${HARD_NEGATIVE_WEIGHT}" \
  --warp-hard-negative-weight "${WARP_HARD_NEGATIVE_WEIGHT}" \
  --warp-hard-negative-radius 2.0 \
  --warp-hard-negative-margin 0.25 \
  --warp-hard-negative-candidates 4096 \
  --input-local-contrast \
  --input-local-contrast-strength 0.35 \
  --input-local-contrast-kernel 31 \
  --selected-keypoint-offset-weight "${SELECTED_KEYPOINT_OFFSET_WEIGHT}" \
  --selected-keypoint-offset-max-points 256 \
  --selected-keypoint-offset-inverse-radius-px 1.5 \
  "${TRAIN_DESCRIPTOR_FLAGS[@]}" \
  "${FALSE_MATCH_FLAGS[@]}" \
  --train-graph-matcher \
  --graph-matcher-loss-weight 1.0 \
  --graph-matcher-metadata-mode calibrated \
  --matcher-reliability-pair-bias off \
  --matcher-reliability-dustbin-bias off \
  --matcher-final-accept-score-mode "${TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE}" \
  --matcher-final-accept-score-alpha "${TRAIN_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}" \
  --matcher-candidate-topk "${TRAIN_MATCHER_CANDIDATE_TOPK}" \
  --graph-matcher-train-candidate-topk "${TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK}" \
  --graph-matcher-accept-weight "${GRAPH_MATCHER_ACCEPT_WEIGHT}" \
  --graph-matcher-accept-negative-topk 8 \
  --graph-matcher-prune-ranking-weight "${GRAPH_MATCHER_PRUNE_RANKING_WEIGHT}" \
  --graph-matcher-prune-ranking-margin 0.25 \
  --graph-matcher-stop-confidence-weight "${GRAPH_MATCHER_STOP_CONFIDENCE_WEIGHT}" \
  --graph-matcher-stop-confidence-margin 0.5 \
  --graph-matcher-positive-dustbin-margin-weight "${GRAPH_MATCHER_POSITIVE_DUSTBIN_MARGIN_WEIGHT}" \
  --graph-matcher-positive-dustbin-margin 0.15 \
  --graph-matcher-true-match-margin-weight "${GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT}" \
  --graph-matcher-true-match-margin 0.45 \
  --graph-matcher-true-geometry-match-count-floor-weight "${GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT}" \
  --graph-matcher-true-geometry-match-count-floor-threshold "${GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_THRESHOLD}" \
  --graph-matcher-true-geometry-match-count-floor-margin "${GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_MARGIN}" \
  --graph-matcher-final-false-match-weight "${FINAL_FALSE_MATCH_WEIGHT}" \
  --graph-matcher-final-false-match-topk "${FINAL_FALSE_MATCH_TOPK}" \
  --graph-matcher-mined-false-match-weight "${MINED_FALSE_MATCH_WEIGHT}" \
  --graph-matcher-mined-false-match-loss-cap "${MINED_FALSE_MATCH_LOSS_CAP}" \
  --graph-matcher-mined-false-match-reference-margin "${MINED_FALSE_MATCH_REFERENCE_MARGIN}" \
  --graph-matcher-final-false-match-min-score 0.0 \
  --graph-matcher-final-false-match-margin 0.25 \
  --graph-matcher-final-false-match-spatial-min-distance 4.0 \
  --graph-matcher-raw-false-match-weight "${RAW_FALSE_MATCH_WEIGHT}" \
  --graph-matcher-raw-false-match-topk "${RAW_FALSE_MATCH_TOPK}" \
  --graph-matcher-raw-false-match-min-similarity 0.70 \
  --graph-matcher-raw-false-match-margin 0.20 \
  --graph-matcher-raw-false-match-spatial-min-distance 4.0 \
  --graph-matcher-warp-outlier-weight "${WARP_OUTLIER_WEIGHT}" \
  --graph-matcher-warp-outlier-topk "${WARP_OUTLIER_TOPK}" \
  --graph-matcher-warp-outlier-residual-threshold-px "${WARP_OUTLIER_RESIDUAL_THRESHOLD_PX}" \
  --graph-matcher-warp-outlier-min-score 0.0 \
  --graph-matcher-warp-outlier-margin 0.30 \
  --graph-matcher-warp-outlier-accept-weight "${WARP_OUTLIER_ACCEPT_WEIGHT}" \
  --graph-matcher-warp-outlier-accept-topk "${WARP_OUTLIER_ACCEPT_TOPK}" \
  --graph-matcher-warp-outlier-accept-residual-threshold-px "${WARP_OUTLIER_ACCEPT_RESIDUAL_THRESHOLD_PX}" \
  --graph-matcher-warp-outlier-accept-min-score 0.0 \
  --graph-matcher-warp-soft-boundary-weight "${WARP_SOFT_BOUNDARY_WEIGHT}" \
  --graph-matcher-warp-soft-boundary-topk "${WARP_SOFT_BOUNDARY_TOPK}" \
  --graph-matcher-warp-soft-boundary-lower-residual-px "${WARP_SOFT_BOUNDARY_LOWER_RESIDUAL_PX}" \
  --graph-matcher-warp-soft-boundary-upper-residual-px "${WARP_SOFT_BOUNDARY_UPPER_RESIDUAL_PX}" \
  --graph-matcher-warp-soft-boundary-min-score "${WARP_SOFT_BOUNDARY_MIN_SCORE}" \
  --graph-matcher-pair-acceptance-loss-weight "${PAIR_ACCEPT_LOSS_WEIGHT}" \
  --graph-matcher-positive-dustbin-guard-reject-threshold 1.1 \
  --false-cluster-replay-loss-multiplier "${FALSE_CLUSTER_REPLAY_MULTIPLIER}" \
  --max-grad-norm 1.0 \
  --skip-bad-pairs \
  --skip-nonfinite-steps \
  --no-gpu-monitor \
  --no-auto-visual-report \
  "${TRAIN_EXTRA_FLAGS[@]}"

CANDIDATE_STATE="${TRAIN_OUT}/checkpoints/last_good_pytorch_pfm_state.pt"
if [[ ! -s "${CANDIDATE_STATE}" ]]; then
  CANDIDATE_STATE="${TRAIN_OUT}/pytorch_pfm_state.pt"
fi
if [[ ! -s "${CANDIDATE_STATE}" ]]; then
  echo "missing trained candidate state under ${TRAIN_OUT}" >&2
  exit 1
fi

PFM_PHASE40_ROOT="${EVAL_ROOT}" \
PFM_CANDIDATE_STATE="${CANDIDATE_STATE}" \
PFM_PHASE40_RENDER_MANIFEST="${RENDER_MANIFEST}" \
PFM_PHASE40_UINT8_MANIFEST="${UINT8_MANIFEST}" \
PFM_EVAL_SUBDIR="${PFM_EVAL_SUBDIR}" \
PFM_MAX_KEYPOINTS="${EVAL_MAX_KEYPOINTS}" \
PFM_KEYPOINT_SPATIAL_BINS="${EVAL_KEYPOINT_SPATIAL_BINS}" \
PFM_KEYPOINT_CELL_CAP="${EVAL_KEYPOINT_CELL_CAP}" \
PFM_PHASE40_MATCHER_CANDIDATE_TOPK="${EVAL_MATCHER_CANDIDATE_TOPK}" \
PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_MODE="${EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE}" \
PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}" \
  bash runs/phase40_crosscam_extreme_baseline_20260620.sh

if [[ -n "${GEOMETRY_OVERLAP_GATE_THRESHOLD}" ]]; then
  GEOMETRY_GATE_DIR="${EVAL_ROOT}/geometry_overlap_gate"
  mkdir -p "${GEOMETRY_GATE_DIR}"
  "${PY}" scripts/sweep_geometry_overlap_gate.py \
    --source "dev,${EVAL_ROOT}/dev/${PFM_EVAL_SUBDIR}/all_filtered_summary.csv,${EVAL_ROOT}/dev/lightglue/lightglue_sift_metrics.csv" \
    --source "val,${EVAL_ROOT}/val/${PFM_EVAL_SUBDIR}/all_filtered_summary.csv,${EVAL_ROOT}/val/lightglue/lightglue_sift_metrics.csv" \
    --thresholds "${GEOMETRY_OVERLAP_GATE_THRESHOLDS}" \
    --output-csv "${GEOMETRY_GATE_DIR}/dev_val_threshold_summary.csv" \
    --summary-json "${GEOMETRY_GATE_DIR}/dev_val_threshold_summary.json" \
    --report-html "${GEOMETRY_GATE_DIR}/dev_val_threshold_summary.html" \
    --selected-threshold "${GEOMETRY_OVERLAP_GATE_THRESHOLD}" \
    --selected-summary-json "${GEOMETRY_GATE_DIR}/dev_val_selected_threshold_summary.json" \
    --selected-report-html "${GEOMETRY_GATE_DIR}/dev_val_selected_threshold_summary.html"
  "${PY}" scripts/sweep_geometry_overlap_gate.py \
    --source "lockbox,${EVAL_ROOT}/lockbox/${PFM_EVAL_SUBDIR}/all_filtered_summary.csv,${EVAL_ROOT}/lockbox/lightglue/lightglue_sift_metrics.csv" \
    --thresholds "${GEOMETRY_OVERLAP_GATE_THRESHOLDS}" \
    --output-csv "${GEOMETRY_GATE_DIR}/lockbox_threshold_summary.csv" \
    --summary-json "${GEOMETRY_GATE_DIR}/lockbox_threshold_summary.json" \
    --report-html "${GEOMETRY_GATE_DIR}/lockbox_threshold_summary.html" \
    --selected-threshold "${GEOMETRY_OVERLAP_GATE_THRESHOLD}" \
    --selected-summary-json "${GEOMETRY_GATE_DIR}/lockbox_selected_threshold_summary.json" \
    --selected-report-html "${GEOMETRY_GATE_DIR}/lockbox_selected_threshold_summary.html"
  "${PY}" scripts/sweep_geometry_overlap_gate.py \
    --source "dev,${EVAL_ROOT}/dev/${PFM_EVAL_SUBDIR}/all_filtered_summary.csv,${EVAL_ROOT}/dev/lightglue/lightglue_sift_metrics.csv" \
    --source "val,${EVAL_ROOT}/val/${PFM_EVAL_SUBDIR}/all_filtered_summary.csv,${EVAL_ROOT}/val/lightglue/lightglue_sift_metrics.csv" \
    --source "lockbox,${EVAL_ROOT}/lockbox/${PFM_EVAL_SUBDIR}/all_filtered_summary.csv,${EVAL_ROOT}/lockbox/lightglue/lightglue_sift_metrics.csv" \
    --thresholds "${GEOMETRY_OVERLAP_GATE_THRESHOLDS}" \
    --output-csv "${GEOMETRY_GATE_DIR}/aggregate_threshold_summary.csv" \
    --summary-json "${GEOMETRY_GATE_DIR}/aggregate_threshold_summary.json" \
    --report-html "${GEOMETRY_GATE_DIR}/aggregate_threshold_summary.html" \
    --selected-threshold "${GEOMETRY_OVERLAP_GATE_THRESHOLD}" \
    --selected-summary-json "${GEOMETRY_GATE_DIR}/aggregate_selected_threshold_summary.json" \
    --selected-report-html "${GEOMETRY_GATE_DIR}/aggregate_selected_threshold_summary.html"
fi

"${PY}" - "${EVAL_ROOT}" "${PFM_EVAL_SUBDIR}" "${GATE_THRESHOLDS}" <<'PY'
import csv
import html
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
pfm_subdir = sys.argv[2]
thresholds = [float(item) for item in sys.argv[3].split(",") if item.strip()]
splits = ("dev", "val", "lockbox")


def _int(row: dict[str, str], key: str) -> int:
    return int(round(float(row.get(key, "0") or 0.0)))


def _probability(row: dict[str, str]) -> float:
    value = row.get("pair_accept_probability", "")
    if value == "":
        raise ValueError("missing pair_accept_probability in PFM summary")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"invalid pair_accept_probability: {value!r}")
    return parsed


def _lightglue_rows(split: str) -> list[dict[str, str]]:
    path = root / split / "lightglue" / "lightglue_sift_metrics.csv"
    return [
        row
        for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
        if row.get("label") == "LightGlue-SIFT-MAGSAC-min16"
    ]


def _pfm_rows(split: str) -> list[dict[str, str]]:
    path = root / split / pfm_subdir / "all_filtered_summary.csv"
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def _finalize(item: dict[str, object]) -> dict[str, object]:
    matches = int(item["matches"])
    lg_matches = int(item["lightglue_matches"])
    item["precision"] = int(item["correct"]) / matches if matches else 0.0
    item["lightglue_precision"] = int(item["lightglue_correct"]) / lg_matches if lg_matches else 0.0
    item["correct_delta_vs_lightglue"] = int(item["correct"]) - int(item["lightglue_correct"])
    item["wrong_delta_vs_lightglue"] = int(item["wrong"]) - int(item["lightglue_wrong"])
    item["precision_delta_vs_lightglue"] = float(item["precision"]) - float(item["lightglue_precision"])
    return item


threshold_rows = []
by_split: dict[str, list[dict[str, object]]] = {}
for threshold in thresholds:
    aggregate = {
        "threshold": threshold,
        "split": "aggregate",
        "rows": 0,
        "kept_pairs": 0,
        "rejected_pairs": 0,
        "matches": 0,
        "correct": 0,
        "wrong": 0,
        "lightglue_matches": 0,
        "lightglue_correct": 0,
        "lightglue_wrong": 0,
    }
    split_rows = []
    for split in splits:
        pfm_rows = _pfm_rows(split)
        lightglue_rows = _lightglue_rows(split)
        if len(pfm_rows) != len(lightglue_rows):
            raise ValueError(f"row count mismatch for {split}: PFM={len(pfm_rows)} LightGlue={len(lightglue_rows)}")
        item = {
            "threshold": threshold,
            "split": split,
            "rows": len(pfm_rows),
            "kept_pairs": 0,
            "rejected_pairs": 0,
            "matches": 0,
            "correct": 0,
            "wrong": 0,
            "lightglue_matches": sum(_int(row, "matches") for row in lightglue_rows),
            "lightglue_correct": sum(_int(row, "correct") for row in lightglue_rows),
            "lightglue_wrong": sum(_int(row, "wrong") for row in lightglue_rows),
        }
        for row in pfm_rows:
            if _probability(row) >= threshold:
                item["kept_pairs"] += 1
                item["matches"] += _int(row, "matches")
                item["correct"] += _int(row, "correct")
                item["wrong"] += _int(row, "wrong")
            else:
                item["rejected_pairs"] += 1
        item = _finalize(item)
        split_rows.append(item)
        for key in ("rows", "kept_pairs", "rejected_pairs", "matches", "correct", "wrong", "lightglue_matches", "lightglue_correct", "lightglue_wrong"):
            aggregate[key] += int(item[key])
    aggregate = _finalize(aggregate)
    threshold_rows.append(aggregate)
    by_split[f"{threshold:.6f}"] = split_rows

best = max(threshold_rows, key=lambda row: (int(row["correct"]), -int(row["wrong"]), int(row["kept_pairs"])))
payload = {
    "root": str(root),
    "pfm_eval_subdir": pfm_subdir,
    "thresholds": thresholds,
    "selection_policy": "diagnostic only: maximize PFM correct, then minimize PFM wrong",
    "best_threshold": best,
    "aggregate": threshold_rows,
    "by_split": by_split,
}
(root / "pair_accept_gate_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
fields = list(threshold_rows[0].keys()) if threshold_rows else []
table_rows = [
    "<tr>" + "".join(f"<td>{html.escape(str(row[field]))}</td>" for field in fields) + "</tr>"
    for row in threshold_rows
]
(root / "pair_accept_gate_summary.html").write_text(
    "\n".join(
        [
            "<!doctype html>",
            '<meta charset="utf-8">',
            "<title>Phase41 Pair Accept Gate Summary</title>",
            "<h1>Phase41 Pair Accept Gate Summary</h1>",
            f"<p>pfm_eval_subdir=<code>{html.escape(pfm_subdir)}</code></p>",
            f"<p>best_threshold=<code>{html.escape(str(best['threshold']))}</code></p>",
            '<table border="1" cellspacing="0" cellpadding="4">',
            "<tr>" + "".join(f"<th>{html.escape(field)}</th>" for field in fields) + "</tr>",
            *table_rows,
            "</table>",
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)
PY

cat >> "${RUN_ROOT}/record.html" <<HTML
<p>stage=<code>completed</code></p>
<p>candidate_state=<code>${CANDIDATE_STATE}</code></p>
<p>eval_summary=<code>${EVAL_ROOT}/baseline_summary.json</code></p>
<p>pair_accept_gate_summary=<code>${EVAL_ROOT}/pair_accept_gate_summary.json</code></p>
<p>geometry_overlap_gate_summary=<code>${EVAL_ROOT}/geometry_overlap_gate/aggregate_selected_threshold_summary.json</code></p>
HTML

echo "phase41_crosscam_extreme_geometry_train_eval_complete root=${RUN_ROOT} state=${CANDIDATE_STATE}"
