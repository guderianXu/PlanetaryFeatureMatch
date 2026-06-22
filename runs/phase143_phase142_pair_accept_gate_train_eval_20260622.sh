#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch"
cd "${PROJECT_ROOT}"

ACTIVE_PATTERN='batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py|run_lightglue|phase59_true_geometry_selector_multiseed_eval|train_match_detail_filter_calibrator.py|apply_match_detail_filter_calibrator.py|apply_observable_pair_gate_match_filter.py|train_match_set_rejection_calibrator.py|apply_match_set_rejection_calibrator.py|build_cluster_gate_dataset.py'
ACTIVE_TASKS="$(pgrep -af "${ACTIVE_PATTERN}" | grep -v -E 'pgrep -af|grep -v' || true)"
if [[ -n "${ACTIVE_TASKS}" ]]; then
  echo "[phase143] active long-running PFM task detected; refusing to start:" >&2
  echo "${ACTIVE_TASKS}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/scripts"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${PFM_TMPDIR:-/media/w24/D/xjw深度学习训练数据/tmp}"
mkdir -p "${TMPDIR}"

PY="${PFM_PHASE143_PYTHON:-/home/w24/anaconda3/envs/cppTorch/bin/python}"
PHASE141_ROOT="${PFM_PHASE141_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase141_extreme01_02_gap_replay_conservative_train_eval_20260622}"
PHASE142_ROOT="${PFM_PHASE142_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase142_phase141_observable_gate_sweep_20260622}"
SAFE_GATE_ROOT="${PFM_PHASE143_SAFE_GATE_ROOT:-${PHASE142_ROOT}/apply_all_split_variant_safe_valid0356314_hmed1195}"
RUN_ROOT="${PFM_PHASE143_RUN_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase143_phase142_pair_accept_gate_train_eval_20260622}"
PREP_ROOT="${RUN_ROOT}/prep"

SOURCE_DATASET="${PFM_PHASE143_SOURCE_DATASET:-${PHASE142_ROOT}/phase141_observable_gate_dataset.csv}"
SAFE_HYBRID_ROWS="${PFM_PHASE143_SAFE_HYBRID_ROWS:-${PHASE142_ROOT}/apply_all_split_variant_safe_valid0356314_hmed1195/hybrid_rows.csv}"
REJECTION_DATASET="${PREP_ROOT}/phase143_gate_rejection_dataset.csv"
CALIBRATOR_ROOT="${PREP_ROOT}/learned_reject_calibrator"
CALIBRATOR_APPLY_ROOT="${PREP_ROOT}/learned_reject_apply"
DEV_SAFE_HYBRID="${PREP_ROOT}/dev_safe_hybrid_rows.csv"
VAL_SAFE_HYBRID="${PREP_ROOT}/val_safe_hybrid_rows.csv"
PAIR_ACCEPT_MANIFEST="${PREP_ROOT}/phase143_pair_accept_gate_train_manifest.csv"

for required in \
  "${SOURCE_DATASET}" \
  "${SAFE_HYBRID_ROWS}" \
  "${PHASE141_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt" \
  "${PHASE141_ROOT}/eval/dev_pairs.csv" \
  "${PHASE141_ROOT}/eval/val_pairs.csv" \
  "${PHASE141_ROOT}/eval/lockbox_pairs.csv"
do
  if [[ ! -f "${required}" ]]; then
    echo "[phase143] missing required input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${PREP_ROOT}" "${CALIBRATOR_ROOT}" "${CALIBRATOR_APPLY_ROOT}"

"${PY}" - "${SOURCE_DATASET}" "${SAFE_HYBRID_ROWS}" "${REJECTION_DATASET}" "${DEV_SAFE_HYBRID}" "${VAL_SAFE_HYBRID}" <<'PY'
import csv
import sys
from pathlib import Path

source_dataset = Path(sys.argv[1])
safe_hybrid_rows = Path(sys.argv[2])
rejection_dataset = Path(sys.argv[3])
dev_safe_hybrid = Path(sys.argv[4])
val_safe_hybrid = Path(sys.argv[5])


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


source_fields, source_rows = read_csv(source_dataset)
hybrid_fields, hybrid_rows = read_csv(safe_hybrid_rows)
hybrid_by_key = {
    (
        row.get("source_name", ""),
        row.get("split", ""),
        row.get("pair_index", ""),
        row.get("base_id", ""),
        row.get("target_variant", ""),
    ): row
    for row in hybrid_rows
}

output_fields = list(source_fields)
for field in ["reject_label", "reject_reasons", "keep_label", "gate_selected_pfm", "gate_chosen_source"]:
    if field not in output_fields:
        output_fields.append(field)

output_rows = []
for row in source_rows:
    key = (
        row.get("source_name", ""),
        row.get("split", ""),
        row.get("pair_index", ""),
        row.get("base_id", ""),
        row.get("target_variant", ""),
    )
    hybrid = hybrid_by_key.get(key)
    if hybrid is None:
        raise ValueError(f"missing safe hybrid row for key={key}")
    selected = hybrid.get("gate_selected_pfm", "") == "1" and hybrid.get("chosen_source", "") == "pfm"
    item = dict(row)
    item["reject_label"] = "0" if selected else "1"
    item["keep_label"] = "1" if selected else "0"
    item["reject_reasons"] = "safe_gate_accept_pfm" if selected else "safe_gate_fallback_lightglue"
    item["gate_selected_pfm"] = "1" if selected else "0"
    item["gate_chosen_source"] = hybrid.get("chosen_source", "")
    output_rows.append(item)

rejection_dataset.parent.mkdir(parents=True, exist_ok=True)
with rejection_dataset.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(output_rows)

for split, path in [("dev", dev_safe_hybrid), ("val", val_safe_hybrid)]:
    rows = [row for row in hybrid_rows if row.get("split", "") == split]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=hybrid_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

print(
    f"phase143_sources source_rows={len(source_rows)} hybrid_rows={len(hybrid_rows)} "
    f"reject_dataset={rejection_dataset} dev_hybrid={dev_safe_hybrid} val_hybrid={val_safe_hybrid}",
    flush=True,
)
PY

"${PY}" scripts/train_match_set_rejection_calibrator.py \
  --dataset-csv "${REJECTION_DATASET}" \
  --output-dir "${CALIBRATOR_ROOT}" \
  --train-split dev \
  --eval-split val \
  --threshold-objective hybrid_lightglue_wrong_cap \
  --threshold-selection-source eval \
  --max-hybrid-wrong-delta-vs-lightglue 0 \
  --epochs 2000 \
  --learning-rate 0.05 \
  --l2 0.001

"${PY}" scripts/apply_match_set_rejection_calibrator.py \
  --dataset-csv "${REJECTION_DATASET}" \
  --model-json "${CALIBRATOR_ROOT}/model.json" \
  --output-csv "${CALIBRATOR_APPLY_ROOT}/hybrid_rows.csv" \
  --summary-json "${CALIBRATOR_APPLY_ROOT}/summary.json" \
  --output-html "${CALIBRATOR_APPLY_ROOT}/index.html" \
  --reject-action lightglue

"${PY}" scripts/build_gate_acceptance_training_manifest.py \
  --source "phase143_dev,${PHASE141_ROOT}/eval/dev_pairs.csv,${DEV_SAFE_HYBRID}" \
  --source "phase143_val,${PHASE141_ROOT}/eval/val_pairs.csv,${VAL_SAFE_HYBRID}" \
  --output-manifest "${PAIR_ACCEPT_MANIFEST}" \
  --summary-json "${PREP_ROOT}/phase143_pair_accept_gate_train_manifest_summary.json" \
  --report-html "${PREP_ROOT}/phase143_pair_accept_gate_train_manifest_summary.html" \
  --accept-weight 1.0 \
  --reject-weight 4.0 \
  --min-accept-precision 0.999 \
  --max-accept-wrong 0 \
  --target-accept-fraction 0.25

if [[ "${PFM_PHASE143_PREP_ONLY:-0}" == "1" ]]; then
  echo "phase143_prep_only_complete rejection_dataset=${REJECTION_DATASET} pair_accept_manifest=${PAIR_ACCEPT_MANIFEST}"
  exit 0
fi

export PFM_PHASE41_TITLE="${PFM_PHASE41_TITLE:-Phase143 Phase142 Pair-Accept Gate Train/Eval}"
export PFM_PHASE41_GOAL="${PFM_PHASE41_GOAL:-Train a learnable pair accept head from the Phase142 safe observable hybrid gate while keeping lockbox held out for evaluation.}"
export PFM_PHASE41_NOTE="${PFM_PHASE41_NOTE:-Accept/reject labels come from the dev/val Phase142 safe gate; lockbox rows are not used for pair-accept training labels.}"
export PFM_PHASE41_TRAIN_ROOT="${PFM_PHASE41_TRAIN_ROOT:-${RUN_ROOT}}"
export PFM_PHASE41_DATA_ROOT="${PFM_PHASE41_DATA_ROOT:-${PHASE141_ROOT}/eval}"
export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE141_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"
export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${PAIR_ACCEPT_MANIFEST}}"
export PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY="${PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY:-1}"
export PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-0}"
export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-80}"
export PFM_PHASE41_SAVE_EVERY="${PFM_PHASE41_SAVE_EVERY:-40}"
export PFM_PHASE41_TRAIN_SEED="${PFM_PHASE41_TRAIN_SEED:-20260743}"
export PFM_PHASE41_LEARNING_RATE="${PFM_PHASE41_LEARNING_RATE:-1e-4}"
export PFM_PHASE41_TRAIN_SAMPLES_PER_PAIR="${PFM_PHASE41_TRAIN_SAMPLES_PER_PAIR:-512}"
export PFM_PHASE41_TRAIN_SPATIAL_BINS="${PFM_PHASE41_TRAIN_SPATIAL_BINS:-8}"
export PFM_PHASE41_TRAIN_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_TRAIN_MATCHER_CANDIDATE_TOPK:-512}"
export PFM_PHASE41_TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK:-512}"

export PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT="${PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT:-1.0}"
export PFM_PHASE41_TEACHER_WEIGHT="${PFM_PHASE41_TEACHER_WEIGHT:-0.0}"
export PFM_PHASE41_SYNTHETIC_LOSS_WEIGHT="${PFM_PHASE41_SYNTHETIC_LOSS_WEIGHT:-0.0}"
export PFM_PHASE41_HARD_NEGATIVE_WEIGHT="${PFM_PHASE41_HARD_NEGATIVE_WEIGHT:-0.0}"
export PFM_PHASE41_WARP_HARD_NEGATIVE_WEIGHT="${PFM_PHASE41_WARP_HARD_NEGATIVE_WEIGHT:-0.0}"
export PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT="${PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT:-0.0}"
export PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT:-0.0}"
export PFM_PHASE41_GRAPH_MATCHER_PRUNE_RANKING_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_PRUNE_RANKING_WEIGHT:-0.0}"
export PFM_PHASE41_GRAPH_MATCHER_STOP_CONFIDENCE_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_STOP_CONFIDENCE_WEIGHT:-0.0}"
export PFM_PHASE41_GRAPH_MATCHER_POSITIVE_DUSTBIN_MARGIN_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_POSITIVE_DUSTBIN_MARGIN_WEIGHT:-0.0}"
export PFM_PHASE41_GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT:-0.0}"
export PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT:-0.0}"
export PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT="${PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT:-0.0}"
export PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT="${PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT:-0.0}"
export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.0}"
export PFM_PHASE41_WARP_OUTLIER_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_WEIGHT:-0.0}"
export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.0}"
export PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT="${PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT:-0.0}"

export PFM_PHASE41_GATE_THRESHOLDS="${PFM_PHASE41_GATE_THRESHOLDS:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95}"
export PFM_PHASE41_EVAL_MAX_KEYPOINTS="${PFM_PHASE41_EVAL_MAX_KEYPOINTS:-6144}"
export PFM_PHASE41_EVAL_KEYPOINT_SPATIAL_BINS="${PFM_PHASE41_EVAL_KEYPOINT_SPATIAL_BINS:-16}"
export PFM_PHASE41_EVAL_KEYPOINT_CELL_CAP="${PFM_PHASE41_EVAL_KEYPOINT_CELL_CAP:-12}"
export PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK:-512}"
export PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE:-multiply}"
export PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA:--1.0}"
export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase143_pair_accept_gate}"
export PFM_PAIR_ACCEPT_MIN_PROBABILITY="${PFM_PAIR_ACCEPT_MIN_PROBABILITY:--1.0}"

exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh
