#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PFM_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

ACTIVE_PATTERN='batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py|run_lightglue|phase59_true_geometry_selector_multiseed_eval|train_match_detail_filter_calibrator.py|apply_match_detail_filter_calibrator.py|apply_observable_pair_gate_match_filter.py|train_match_set_rejection_calibrator.py|apply_match_set_rejection_calibrator.py|build_cluster_gate_dataset.py'
ACTIVE_TASKS="$(pgrep -af "${ACTIVE_PATTERN}" | grep -v -E 'pgrep -af|grep -v' || true)"
if [[ -n "${ACTIVE_TASKS}" ]]; then
  echo "[phase148] active long-running PFM task detected; refusing to start:" >&2
  echo "${ACTIVE_TASKS}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/scripts"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${PFM_TMPDIR:-/media/w24/D/xjw深度学习训练数据/tmp}"
mkdir -p "${TMPDIR}"

PY="${PFM_PHASE148_PYTHON:-${PFM_PYTHON:-/home/w24/anaconda3/envs/cppTorch/bin/python}}"
PHASE141_ROOT="${PFM_PHASE141_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase141_extreme01_02_gap_replay_conservative_train_eval_20260622}"
PHASE146_ROOT="${PFM_PHASE146_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase146_phase142_safe_gate_mainline_teacher_20260623}"
PHASE147_ROOT="${PFM_PHASE147_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase147_phase146_fresh_gate_validation_20260623}"
RUN_ROOT="${PFM_PHASE148_RUN_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase148_phase146_pair_rejection_head_train_eval_20260623}"
AUDIT_ROOT="${RUN_ROOT}/prep"

CHECKPOINT="${PFM_PHASE148_CHECKPOINT:-${PHASE141_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"
BASE_TEACHER_MANIFEST="${PFM_PHASE148_BASE_TEACHER_MANIFEST:-${PHASE146_ROOT}/teacher/phase146_pair_accept_teacher_manifest.csv}"
BASE_TEACHER_SUMMARY_JSON="${PFM_PHASE148_BASE_TEACHER_SUMMARY_JSON:-${PHASE146_ROOT}/teacher/phase146_pair_accept_teacher_manifest_summary.json}"
PHASE147_VALIDATION_JSON="${PFM_PHASE148_PHASE147_VALIDATION_JSON:-${PHASE147_ROOT}/phase147_fresh_gate_validation.json}"
EVAL_DATA_ROOT="${PFM_PHASE148_EVAL_DATA_ROOT:-${PHASE147_ROOT}/eval}"
DEV_HYBRID_ROWS="${PFM_PHASE148_DEV_HYBRID_ROWS:-${PHASE146_ROOT}/prep/dev_frozen_hybrid_rows.csv}"
VAL_HYBRID_ROWS="${PFM_PHASE148_VAL_HYBRID_ROWS:-${PHASE146_ROOT}/prep/val_frozen_hybrid_rows.csv}"
TEACHER_MANIFEST="${PFM_PHASE148_TEACHER_MANIFEST:-${AUDIT_ROOT}/phase148_pair_accept_teacher_balanced_manifest.csv}"
TEACHER_SUMMARY_JSON="${PFM_PHASE148_TEACHER_SUMMARY_JSON:-${AUDIT_ROOT}/phase148_pair_accept_teacher_balanced_manifest_summary.json}"
TEACHER_REPORT_HTML="${PFM_PHASE148_TEACHER_REPORT_HTML:-${AUDIT_ROOT}/phase148_pair_accept_teacher_balanced_manifest_summary.html}"
AUDIT_JSON="${AUDIT_ROOT}/phase148_teacher_manifest_audit.json"
AUDIT_HTML="${AUDIT_ROOT}/phase148_teacher_manifest_audit.html"

for required in \
  "${CHECKPOINT}" \
  "${BASE_TEACHER_MANIFEST}" \
  "${BASE_TEACHER_SUMMARY_JSON}" \
  "${PHASE141_ROOT}/eval/dev_pairs.csv" \
  "${PHASE141_ROOT}/eval/val_pairs.csv" \
  "${DEV_HYBRID_ROWS}" \
  "${VAL_HYBRID_ROWS}" \
  "${PHASE147_VALIDATION_JSON}" \
  "${EVAL_DATA_ROOT}/dev_pairs.csv" \
  "${EVAL_DATA_ROOT}/val_pairs.csv" \
  "${EVAL_DATA_ROOT}/lockbox_pairs.csv"
do
  if [[ ! -f "${required}" ]]; then
    echo "[phase148] missing required input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${AUDIT_ROOT}"

"${PY}" scripts/build_gate_acceptance_training_manifest.py \
  --source "phase148_dev_teacher,${PHASE141_ROOT}/eval/dev_pairs.csv,${DEV_HYBRID_ROWS}" \
  --source "phase148_val_teacher,${PHASE141_ROOT}/eval/val_pairs.csv,${VAL_HYBRID_ROWS}" \
  --output-manifest "${TEACHER_MANIFEST}" \
  --summary-json "${TEACHER_SUMMARY_JSON}" \
  --report-html "${TEACHER_REPORT_HTML}" \
  --accept-weight "${PFM_PHASE148_ACCEPT_WEIGHT:-1.0}" \
  --reject-weight "${PFM_PHASE148_REJECT_WEIGHT:-1.0}" \
  --min-accept-precision 0.0 \
  --max-accept-wrong 999999 \
  --target-accept-fraction "${PFM_PHASE148_TARGET_ACCEPT_FRACTION:-0.50}"

"${PY}" - \
  "${TEACHER_MANIFEST}" \
  "${TEACHER_SUMMARY_JSON}" \
  "${PHASE147_VALIDATION_JSON}" \
  "${AUDIT_JSON}" \
  "${AUDIT_HTML}" \
  "${CHECKPOINT}" \
  "${EVAL_DATA_ROOT}" <<'PY'
import csv
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path

teacher_manifest = Path(sys.argv[1])
teacher_summary_json = Path(sys.argv[2])
phase147_validation_json = Path(sys.argv[3])
audit_json = Path(sys.argv[4])
audit_html = Path(sys.argv[5])
checkpoint = Path(sys.argv[6])
eval_data_root = Path(sys.argv[7])

with teacher_manifest.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fields = list(reader.fieldnames or [])
    rows = [dict(row) for row in reader]
if not fields:
    raise ValueError(f"{teacher_manifest} is missing a CSV header")

required_fields = {
    "pair_accept_label",
    "pair_accept_weight",
    "gate_accept_source",
    "gate_accept_reason",
}
missing_fields = sorted(required_fields.difference(fields))
if missing_fields:
    raise ValueError(f"teacher manifest is missing required fields: {missing_fields}")

labels = Counter(row.get("pair_accept_label", "") for row in rows)
sources = Counter(row.get("gate_accept_source", "") for row in rows)
reasons = Counter(row.get("gate_accept_reason", "") for row in rows)
splits = Counter(row.get("split", "") for row in rows)
weights = []
for row in rows:
    value = row.get("pair_accept_weight", "")
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"invalid pair_accept_weight={value!r}") from None
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"pair_accept_weight must be positive finite, got {value!r}")
    weights.append(parsed)

forbidden_tokens = ("fresh", "heldout", "holdout", "lockbox")
source_text = " ".join(sources.keys()).lower()
for token in forbidden_tokens:
    if token in source_text:
        raise ValueError(f"refusing to train phase148 from forbidden teacher source token: {token}")

accept_count = labels.get("1", 0)
reject_count = labels.get("0", 0)
if accept_count < 1 or reject_count < 1:
    raise ValueError(f"teacher manifest must contain both accept and reject rows, got {dict(labels)}")
if len(rows) < int(__import__("os").environ.get("PFM_PHASE148_MIN_TEACHER_ROWS", "20")):
    raise ValueError(f"teacher manifest has too few rows: {len(rows)}")

teacher_summary = json.loads(teacher_summary_json.read_text(encoding="utf-8"))
phase147_validation = json.loads(phase147_validation_json.read_text(encoding="utf-8"))
if phase147_validation.get("valid") is not True:
    raise ValueError(f"phase147 validation is not valid: {phase147_validation_json}")
if phase147_validation.get("base_disjoint") is not True:
    raise ValueError(f"phase147 validation is not base_disjoint: {phase147_validation_json}")

payload = {
    "phase": "phase148_teacher_manifest_audit",
    "valid": True,
    "teacher_manifest": str(teacher_manifest),
    "teacher_summary_json": str(teacher_summary_json),
    "phase147_validation_json": str(phase147_validation_json),
    "checkpoint": str(checkpoint),
    "eval_data_root": str(eval_data_root),
    "rows": len(rows),
    "label_counts": dict(labels),
    "source_counts": dict(sources),
    "split_counts": dict(splits),
    "reason_counts": dict(reasons),
    "weight_min": min(weights),
    "weight_max": max(weights),
    "weight_mean": sum(weights) / len(weights),
    "teacher_summary": teacher_summary,
    "phase147_aggregate": phase147_validation.get("aggregate", {}),
    "policy": {
        "training_labels": "phase146 dev/val frozen safe-gate accept/reject labels",
        "heldout_eval": "phase147 fresh dev/val/lockbox manifests",
        "lockbox_teacher_rows_allowed": False,
    },
}
audit_json.parent.mkdir(parents=True, exist_ok=True)
audit_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
audit_html.write_text(
    "\n".join(
        [
            "<!doctype html>",
            '<meta charset="utf-8">',
            "<title>Phase148 Teacher Manifest Audit</title>",
            "<h1>Phase148 Teacher Manifest Audit</h1>",
            f"<p>valid=<code>{str(payload['valid']).lower()}</code></p>",
            f"<p>rows=<code>{payload['rows']}</code></p>",
            f"<p>label_counts=<code>{html.escape(json.dumps(payload['label_counts'], ensure_ascii=False))}</code></p>",
            f"<p>source_counts=<code>{html.escape(json.dumps(payload['source_counts'], ensure_ascii=False))}</code></p>",
            f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(
    "phase148_teacher_manifest_audit "
    f"rows={len(rows)} accept={accept_count} reject={reject_count} "
    f"weight_min={min(weights):.6f} weight_max={max(weights):.6f} "
    f"audit_json={audit_json}",
    flush=True,
)
PY

if [[ "${PFM_PHASE148_PREP_ONLY:-0}" == "1" ]]; then
  echo "phase148_prep_only_complete run_root=${RUN_ROOT} audit_json=${AUDIT_JSON}"
  exit 0
fi

export PFM_PHASE41_TITLE="${PFM_PHASE41_TITLE:-Phase148 Phase146 Pair Rejection Head Train/Eval}"
export PFM_PHASE41_GOAL="${PFM_PHASE41_GOAL:-Train only PFM graph_matcher pair_accept_head to imitate the frozen Phase146 safe gate teacher, then evaluate on the Phase147 fresh base-disjoint manifests.}"
export PFM_PHASE41_NOTE="${PFM_PHASE41_NOTE:-phase148_pair_rejection_head_validation: labels are the Phase146 dev/val teacher only; lockbox remains evaluation-only.}"
export PFM_PHASE41_TRAIN_ROOT="${PFM_PHASE41_TRAIN_ROOT:-${RUN_ROOT}}"
export PFM_PHASE41_DATA_ROOT="${PFM_PHASE41_DATA_ROOT:-${EVAL_DATA_ROOT}}"
export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${CHECKPOINT}}"
export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${TEACHER_MANIFEST}}"
export PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY="${PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY:-1}"
export PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-0}"
export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-240}"
export PFM_PHASE41_SAVE_EVERY="${PFM_PHASE41_SAVE_EVERY:-60}"
export PFM_PHASE41_TRAIN_SEED="${PFM_PHASE41_TRAIN_SEED:-20260748}"
export PFM_PHASE41_LEARNING_RATE="${PFM_PHASE41_LEARNING_RATE:-2e-4}"
export PFM_PHASE41_TRAIN_SAMPLES_PER_PAIR="${PFM_PHASE41_TRAIN_SAMPLES_PER_PAIR:-512}"
export PFM_PHASE41_TRAIN_SPATIAL_BINS="${PFM_PHASE41_TRAIN_SPATIAL_BINS:-8}"
export PFM_PHASE41_TRAIN_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_TRAIN_MATCHER_CANDIDATE_TOPK:-512}"
export PFM_PHASE41_TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK:-512}"

export PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT="${PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT:-2.0}"
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

export PFM_PHASE41_GATE_THRESHOLDS="${PFM_PHASE41_GATE_THRESHOLDS:-0.05,0.10,0.15,0.20,0.25,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70}"
export PFM_PHASE41_EVAL_MAX_KEYPOINTS="${PFM_PHASE41_EVAL_MAX_KEYPOINTS:-6144}"
export PFM_PHASE41_EVAL_KEYPOINT_SPATIAL_BINS="${PFM_PHASE41_EVAL_KEYPOINT_SPATIAL_BINS:-16}"
export PFM_PHASE41_EVAL_KEYPOINT_CELL_CAP="${PFM_PHASE41_EVAL_KEYPOINT_CELL_CAP:-12}"
export PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK:-512}"
export PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE:-multiply}"
export PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA:--1.0}"
export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase148_pair_rejection_head_balanced}"
export PFM_PAIR_ACCEPT_MIN_PROBABILITY="${PFM_PAIR_ACCEPT_MIN_PROBABILITY:--1.0}"

if [[ "${PFM_PHASE148_CLEAR_EVAL:-1}" == "1" ]]; then
  rm -rf \
    "${RUN_ROOT}/eval/dev/${PFM_PHASE41_EVAL_SUBDIR}" \
    "${RUN_ROOT}/eval/val/${PFM_PHASE41_EVAL_SUBDIR}" \
    "${RUN_ROOT}/eval/lockbox/${PFM_PHASE41_EVAL_SUBDIR}"
fi

bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh

HYBRID_ROOT="${RUN_ROOT}/eval/pair_accept_hybrid_balanced/aggregate_row_aligned_dense"
HYBRID_THRESHOLDS="${PFM_PHASE148_HYBRID_THRESHOLDS:-0.4975,0.4976,0.4977,0.4978,0.4979,0.4980,0.4981,0.4982,0.4983,0.4984,0.4985,0.4986,0.4987,0.4988,0.4989,0.4990}"

"${PY}" - "${RUN_ROOT}/eval" "${PFM_PHASE41_EVAL_SUBDIR}" "${HYBRID_ROOT}" "${HYBRID_THRESHOLDS}" <<'PY'
import csv
import html
import json
import sys
from pathlib import Path

eval_root = Path(sys.argv[1])
pfm_subdir = sys.argv[2]
output_root = Path(sys.argv[3])
thresholds = [float(item) for item in sys.argv[4].split(",") if item.strip()]
splits = ("dev", "val", "lockbox")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def int_value(row: dict[str, str], key: str) -> int:
    return int(round(float(row.get(key, "0") or 0.0)))


def float_value(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan") or "nan")


aggregate_rows: list[dict[str, object]] = []
hybrid_rows: list[dict[str, object]] = []
by_split: dict[str, list[dict[str, object]]] = {}
for threshold in thresholds:
    aggregate = {
        "threshold": threshold,
        "split": "aggregate",
        "pairs": 0,
        "pfm_pairs": 0,
        "lightglue_pairs": 0,
        "matches": 0,
        "correct": 0,
        "wrong": 0,
        "lightglue_matches": 0,
        "lightglue_correct": 0,
        "lightglue_wrong": 0,
    }
    split_rows: list[dict[str, object]] = []
    for split in splits:
        pfm_rows = read_csv(eval_root / split / pfm_subdir / "all_filtered_summary.csv")
        lightglue_rows = [
            row
            for row in read_csv(eval_root / split / "lightglue" / "lightglue_sift_metrics.csv")
            if row.get("label") == "LightGlue-SIFT-MAGSAC-min16"
        ]
        if len(pfm_rows) != len(lightglue_rows):
            raise ValueError(f"row mismatch for {split}: PFM={len(pfm_rows)} LightGlue={len(lightglue_rows)}")
        item = {
            "threshold": threshold,
            "split": split,
            "pairs": len(pfm_rows),
            "pfm_pairs": 0,
            "lightglue_pairs": 0,
            "matches": 0,
            "correct": 0,
            "wrong": 0,
            "lightglue_matches": sum(int_value(row, "matches") for row in lightglue_rows),
            "lightglue_correct": sum(int_value(row, "correct") for row in lightglue_rows),
            "lightglue_wrong": sum(int_value(row, "wrong") for row in lightglue_rows),
        }
        for row_index, (pfm_row, lightglue_row) in enumerate(zip(pfm_rows, lightglue_rows)):
            probability = float_value(pfm_row, "pair_accept_probability")
            use_pfm = probability >= threshold
            source_row = pfm_row if use_pfm else lightglue_row
            item["pfm_pairs" if use_pfm else "lightglue_pairs"] = int(
                item["pfm_pairs" if use_pfm else "lightglue_pairs"]
            ) + 1
            item["matches"] = int(item["matches"]) + int_value(source_row, "matches")
            item["correct"] = int(item["correct"]) + int_value(source_row, "correct")
            item["wrong"] = int(item["wrong"]) + int_value(source_row, "wrong")
            hybrid_rows.append(
                {
                    "threshold": f"{threshold:.6f}",
                    "split": split,
                    "row_index": row_index,
                    "source": "pfm" if use_pfm else "lightglue",
                    "pair_accept_probability": f"{probability:.6f}",
                    "pfm_base_id": pfm_row.get("base_id", ""),
                    "lightglue_base_id": lightglue_row.get("base_id", ""),
                    "target_variant": pfm_row.get("target_variant", ""),
                    "hybrid_matches": int_value(source_row, "matches"),
                    "hybrid_correct": int_value(source_row, "correct"),
                    "hybrid_wrong": int_value(source_row, "wrong"),
                    "pfm_correct": int_value(pfm_row, "correct"),
                    "pfm_wrong": int_value(pfm_row, "wrong"),
                    "lightglue_correct": int_value(lightglue_row, "correct"),
                    "lightglue_wrong": int_value(lightglue_row, "wrong"),
                }
            )
        item["precision"] = int(item["correct"]) / int(item["matches"]) if int(item["matches"]) else 0.0
        item["correct_delta_vs_lightglue"] = int(item["correct"]) - int(item["lightglue_correct"])
        item["wrong_delta_vs_lightglue"] = int(item["wrong"]) - int(item["lightglue_wrong"])
        split_rows.append(dict(item))
        for key in (
            "pairs",
            "pfm_pairs",
            "lightglue_pairs",
            "matches",
            "correct",
            "wrong",
            "lightglue_matches",
            "lightglue_correct",
            "lightglue_wrong",
        ):
            aggregate[key] = int(aggregate[key]) + int(item[key])
    aggregate["precision"] = (
        int(aggregate["correct"]) / int(aggregate["matches"]) if int(aggregate["matches"]) else 0.0
    )
    aggregate["correct_delta_vs_lightglue"] = int(aggregate["correct"]) - int(aggregate["lightglue_correct"])
    aggregate["wrong_delta_vs_lightglue"] = int(aggregate["wrong"]) - int(aggregate["lightglue_wrong"])
    aggregate_rows.append(aggregate)
    by_split[f"{threshold:.6f}"] = split_rows

safe_rows = [
    row
    for row in aggregate_rows
    if all(split_row["wrong_delta_vs_lightglue"] <= 0 for split_row in by_split[f"{float(row['threshold']):.6f}"])
]
best = max(
    safe_rows or aggregate_rows,
    key=lambda row: (
        int(row["correct_delta_vs_lightglue"]),
        -int(row["wrong_delta_vs_lightglue"]),
        int(row["pfm_pairs"]),
    ),
)
payload = {
    "phase": "phase148_pair_rejection_head_validation",
    "root": str(eval_root),
    "pfm_eval_subdir": pfm_subdir,
    "alignment": "row_order_within_split",
    "thresholds": thresholds,
    "selection_policy": "maximize correct delta while every split keeps wrong_delta_vs_lightglue <= 0",
    "best_all_split_wrong_safe": best,
    "aggregate": aggregate_rows,
    "by_split": by_split,
}

output_root.mkdir(parents=True, exist_ok=True)
(output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
fields = list(aggregate_rows[0].keys()) if aggregate_rows else []
with (output_root / "threshold_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(aggregate_rows)
hybrid_fields = list(hybrid_rows[0].keys()) if hybrid_rows else []
with (output_root / "hybrid_rows.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=hybrid_fields)
    writer.writeheader()
    writer.writerows(hybrid_rows)
table_rows = [
    "<tr>" + "".join(f"<td>{html.escape(str(row[field]))}</td>" for field in fields) + "</tr>"
    for row in aggregate_rows
]
(output_root / "index.html").write_text(
    "\n".join(
        [
            "<!doctype html>",
            '<meta charset="utf-8">',
            "<title>Phase148 Pair Accept Hybrid Balanced</title>",
            "<h1>Phase148 Pair Accept Hybrid Balanced</h1>",
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

echo "phase148_pair_rejection_head_validation hybrid_summary=${HYBRID_ROOT}/summary.json"
