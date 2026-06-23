#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch"
cd "${PROJECT_ROOT}"

ACTIVE_PATTERN='batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py|run_lightglue|phase59_true_geometry_selector_multiseed_eval|train_match_detail_filter_calibrator.py|apply_match_detail_filter_calibrator.py|apply_observable_pair_gate_match_filter.py|train_match_set_rejection_calibrator.py|apply_match_set_rejection_calibrator.py|build_cluster_gate_dataset.py'
ACTIVE_TASKS="$(pgrep -af "${ACTIVE_PATTERN}" | grep -v -E 'pgrep -af|grep -v' || true)"
if [[ -n "${ACTIVE_TASKS}" ]]; then
  echo "[phase147] active long-running PFM task detected; refusing to start:" >&2
  echo "${ACTIVE_TASKS}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/scripts"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${PFM_TMPDIR:-/media/w24/D/xjw深度学习训练数据/tmp}"
mkdir -p "${TMPDIR}"

PY="${PFM_PHASE147_PYTHON:-/home/w24/anaconda3/envs/cppTorch/bin/python}"
PHASE141_ROOT="${PFM_PHASE141_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase141_extreme01_02_gap_replay_conservative_train_eval_20260622}"
PHASE146_ROOT="${PFM_PHASE146_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase146_phase142_safe_gate_mainline_teacher_20260623}"
FRESH_ROOT="${PFM_PHASE147_FRESH_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase56_large_fresh_true_geometry_selector_eval_20260621}"
FRESH_MANIFEST_ROOT="${PFM_PHASE147_FRESH_MANIFEST_ROOT:-${FRESH_ROOT}/phase54_large_manifests}"
FRESH_EVAL_MANIFEST_ROOT="${PFM_PHASE147_FRESH_EVAL_MANIFEST_ROOT:-${FRESH_MANIFEST_ROOT}/eval}"
MANIFEST_VALIDATION_JSON="${PFM_PHASE147_MANIFEST_VALIDATION_JSON:-${FRESH_MANIFEST_ROOT}/fresh_manifest_validation.json}"
RUN_ROOT="${PFM_PHASE147_RUN_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase147_phase146_fresh_gate_validation_20260623}"
EVAL_ROOT="${RUN_ROOT}/eval"
DATASET_ROOT="${RUN_ROOT}/dataset"
GATE_ROOT="${RUN_ROOT}/frozen_phase146_gate"

CHECKPOINT="${PFM_PHASE147_CHECKPOINT:-${PHASE141_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"
FROZEN_GATE="${PFM_PHASE147_FROZEN_GATE:-feature_valid_fraction >= 0.356314 AND feature_homography_residual_median_px >= 1.195}"
PFM_EVAL_SUBDIR="${PFM_PHASE147_PFM_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase147_fresh_phase141}"
REJECTION_DATASET="${DATASET_ROOT}/phase147_fresh_gate_dataset.csv"
DATASET_SUMMARY_JSON="${DATASET_ROOT}/summary.json"
DATASET_REPORT_HTML="${DATASET_ROOT}/index.html"
GATE_SUMMARY_JSON="${GATE_ROOT}/summary.json"
VALIDATION_JSON="${RUN_ROOT}/phase147_fresh_gate_validation.json"
VALIDATION_HTML="${RUN_ROOT}/phase147_fresh_gate_validation.html"
RUN_RECORD="${RUN_ROOT}/record.html"

for required in \
  "${CHECKPOINT}" \
  "${PHASE146_ROOT}/gate_config.json" \
  "${MANIFEST_VALIDATION_JSON}" \
  "${FRESH_EVAL_MANIFEST_ROOT}/dev_pairs.csv" \
  "${FRESH_EVAL_MANIFEST_ROOT}/val_pairs.csv" \
  "${FRESH_EVAL_MANIFEST_ROOT}/lockbox_pairs.csv"
do
  if [[ ! -f "${required}" ]]; then
    echo "[phase147] missing required input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}" "${EVAL_ROOT}" "${DATASET_ROOT}" "${GATE_ROOT}"
cp "${FRESH_EVAL_MANIFEST_ROOT}/dev_pairs.csv" "${EVAL_ROOT}/dev_pairs.csv"
cp "${FRESH_EVAL_MANIFEST_ROOT}/val_pairs.csv" "${EVAL_ROOT}/val_pairs.csv"
cp "${FRESH_EVAL_MANIFEST_ROOT}/lockbox_pairs.csv" "${EVAL_ROOT}/lockbox_pairs.csv"

cat > "${RUN_RECORD}" <<HTML
<!doctype html><meta charset="utf-8">
<title>Phase147 Phase146 Fresh Gate Validation</title>
<h1>Phase147 Phase146 Fresh Gate Validation</h1>
<p>stage=<code>running</code></p>
<p>phase146_root=<code>${PHASE146_ROOT}</code></p>
<p>fresh_root=<code>${FRESH_ROOT}</code></p>
<p>manifest_validation_json=<code>${MANIFEST_VALIDATION_JSON}</code></p>
<p>checkpoint=<code>${CHECKPOINT}</code></p>
<p>frozen_gate=<code>${FROZEN_GATE}</code></p>
<p>pfm_eval_subdir=<code>${PFM_EVAL_SUBDIR}</code></p>
<p>require_valid=<code>${PFM_PHASE147_REQUIRE_VALID:-0}</code></p>
HTML

PFM_PHASE40_ROOT="${EVAL_ROOT}" \
PFM_CANDIDATE_STATE="${CHECKPOINT}" \
PFM_EVAL_SUBDIR="${PFM_EVAL_SUBDIR}" \
PFM_MAX_KEYPOINTS="${PFM_PHASE147_MAX_KEYPOINTS:-6144}" \
PFM_KEYPOINT_SPATIAL_BINS="${PFM_PHASE147_KEYPOINT_SPATIAL_BINS:-16}" \
PFM_KEYPOINT_CELL_CAP="${PFM_PHASE147_KEYPOINT_CELL_CAP:-12}" \
PFM_PHASE40_MATCHER_CANDIDATE_TOPK="${PFM_PHASE147_MATCHER_CANDIDATE_TOPK:-512}" \
PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE147_MATCHER_FINAL_ACCEPT_SCORE_MODE:-multiply}" \
PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${PFM_PHASE147_MATCHER_FINAL_ACCEPT_SCORE_ALPHA:--1.0}" \
  bash runs/phase40_crosscam_extreme_baseline_20260620.sh

"${PY}" scripts/build_match_set_rejection_dataset.py \
  --source "dev,${EVAL_ROOT}/dev_pairs.csv,${EVAL_ROOT}/dev/${PFM_EVAL_SUBDIR}/all_filtered_summary.csv,${EVAL_ROOT}/dev/lightglue/lightglue_sift_metrics.csv,${EVAL_ROOT}/dev/${PFM_EVAL_SUBDIR}/all_filtered_match_details.csv" \
  --source "val,${EVAL_ROOT}/val_pairs.csv,${EVAL_ROOT}/val/${PFM_EVAL_SUBDIR}/all_filtered_summary.csv,${EVAL_ROOT}/val/lightglue/lightglue_sift_metrics.csv,${EVAL_ROOT}/val/${PFM_EVAL_SUBDIR}/all_filtered_match_details.csv" \
  --source "lockbox,${EVAL_ROOT}/lockbox_pairs.csv,${EVAL_ROOT}/lockbox/${PFM_EVAL_SUBDIR}/all_filtered_summary.csv,${EVAL_ROOT}/lockbox/lightglue/lightglue_sift_metrics.csv,${EVAL_ROOT}/lockbox/${PFM_EVAL_SUBDIR}/all_filtered_match_details.csv" \
  --source-name phase147_fresh_gate_validation \
  --output-csv "${REJECTION_DATASET}" \
  --summary-json "${DATASET_SUMMARY_JSON}" \
  --output-html "${DATASET_REPORT_HTML}"

"${PY}" scripts/apply_observable_pair_gate.py \
  --dataset-csv "${REJECTION_DATASET}" \
  --gate "${FROZEN_GATE}" \
  --output-dir "${GATE_ROOT}"

"${PY}" - \
  "${MANIFEST_VALIDATION_JSON}" \
  "${EVAL_ROOT}/baseline_summary.json" \
  "${DATASET_SUMMARY_JSON}" \
  "${GATE_SUMMARY_JSON}" \
  "${VALIDATION_JSON}" \
  "${VALIDATION_HTML}" \
  "${CHECKPOINT}" \
  "${FROZEN_GATE}" \
  "${PHASE146_ROOT}" <<'PY'
import html
import json
import os
import sys
from pathlib import Path

manifest_validation_json = Path(sys.argv[1])
baseline_summary_json = Path(sys.argv[2])
dataset_summary_json = Path(sys.argv[3])
gate_summary_json = Path(sys.argv[4])
validation_json = Path(sys.argv[5])
validation_html = Path(sys.argv[6])
checkpoint = sys.argv[7]
frozen_gate = sys.argv[8]
phase146_root = sys.argv[9]

manifest = json.loads(manifest_validation_json.read_text(encoding="utf-8"))
baseline = json.loads(baseline_summary_json.read_text(encoding="utf-8"))
dataset = json.loads(dataset_summary_json.read_text(encoding="utf-8"))
gate = json.loads(gate_summary_json.read_text(encoding="utf-8"))
required_splits = [item for item in os.environ.get("PFM_PHASE147_REQUIRED_SPLITS", "dev,val,lockbox").split(",") if item]
min_correct_delta = int(os.environ.get("PFM_PHASE147_MIN_CORRECT_DELTA_VS_LIGHTGLUE", "1"))
max_wrong_delta = int(os.environ.get("PFM_PHASE147_MAX_WRONG_DELTA_VS_LIGHTGLUE", "0"))
min_split_rows = int(os.environ.get("PFM_PHASE147_MIN_SPLIT_ROWS", "1"))

errors: list[str] = []
if manifest.get("base_disjoint") is not True:
    errors.append("manifest_validation.base_disjoint is not true")

manifest_counts = manifest.get("counts", {})
if not isinstance(manifest_counts, dict):
    errors.append("manifest_validation.counts is missing")
    manifest_counts = {}

gate_by_split = gate.get("by_split", {})
baseline_by_split = baseline.get("by_split", {})
dataset_by_split = dataset.get("by_split", {})
if not isinstance(gate_by_split, dict):
    errors.append("gate summary missing by_split")
    gate_by_split = {}
if not isinstance(baseline_by_split, dict):
    errors.append("baseline summary missing by_split")
    baseline_by_split = {}
if not isinstance(dataset_by_split, dict):
    errors.append("dataset summary missing by_split")
    dataset_by_split = {}

split_results = {}
for split in required_splits:
    gate_item = gate_by_split.get(split)
    baseline_item = baseline_by_split.get(split)
    dataset_item = dataset_by_split.get(split)
    if not isinstance(gate_item, dict):
        errors.append(f"{split}: missing gate by_split entry")
        continue
    rows = int(round(float(gate_item.get("rows", 0) or 0)))
    correct_delta = int(round(float(gate_item.get("correct_delta_vs_lightglue", 0) or 0)))
    wrong_delta = int(round(float(gate_item.get("wrong_delta_vs_lightglue", 0) or 0)))
    if rows < min_split_rows:
        errors.append(f"{split}: rows {rows} < {min_split_rows}")
    expected_rows = int(round(float(manifest_counts.get(split, rows) or rows)))
    if rows != expected_rows:
        errors.append(f"{split}: gate rows {rows} != manifest rows {expected_rows}")
    if correct_delta < min_correct_delta:
        errors.append(f"{split}: correct_delta_vs_lightglue {correct_delta} < {min_correct_delta}")
    if wrong_delta > max_wrong_delta:
        errors.append(f"{split}: wrong_delta_vs_lightglue {wrong_delta} > {max_wrong_delta}")
    split_results[split] = {
        "rows": rows,
        "expected_rows": expected_rows,
        "correct_delta_vs_lightglue": correct_delta,
        "wrong_delta_vs_lightglue": wrong_delta,
        "gate": gate_item,
        "baseline": baseline_item if isinstance(baseline_item, dict) else {},
        "dataset": dataset_item if isinstance(dataset_item, dict) else {},
    }

aggregate = gate
aggregate_correct_delta = int(round(float(aggregate.get("correct_delta_vs_lightglue", 0) or 0)))
aggregate_wrong_delta = int(round(float(aggregate.get("wrong_delta_vs_lightglue", 0) or 0)))
if aggregate_correct_delta < min_correct_delta:
    errors.append(f"aggregate: correct_delta_vs_lightglue {aggregate_correct_delta} < {min_correct_delta}")
if aggregate_wrong_delta > max_wrong_delta:
    errors.append(f"aggregate: wrong_delta_vs_lightglue {aggregate_wrong_delta} > {max_wrong_delta}")

payload = {
    "phase": "phase147_fresh_gate_validation",
    "valid": not errors,
    "errors": errors,
    "phase146_root": phase146_root,
    "checkpoint": checkpoint,
    "frozen_gate": frozen_gate,
    "manifest_validation_json": str(manifest_validation_json),
    "base_disjoint": bool(manifest.get("base_disjoint") is True),
    "manifest_counts": manifest_counts,
    "required_splits": required_splits,
    "thresholds": {
        "min_correct_delta_vs_lightglue": min_correct_delta,
        "max_wrong_delta_vs_lightglue": max_wrong_delta,
        "min_split_rows": min_split_rows,
    },
    "aggregate": {
        "rows": gate.get("rows", 0),
        "kept_pfm_rows": gate.get("kept_pfm_rows", 0),
        "fallback_lightglue_rows": gate.get("fallback_lightglue_rows", 0),
        "hybrid_correct": gate.get("hybrid_correct", 0),
        "hybrid_wrong": gate.get("hybrid_wrong", 0),
        "hybrid_precision": gate.get("hybrid_precision", 0.0),
        "lightglue_correct": gate.get("lightglue_correct", 0),
        "lightglue_wrong": gate.get("lightglue_wrong", 0),
        "correct_delta_vs_lightglue": aggregate_correct_delta,
        "wrong_delta_vs_lightglue": aggregate_wrong_delta,
    },
    "split_results": split_results,
    "inputs": {
        "baseline_summary_json": str(baseline_summary_json),
        "dataset_summary_json": str(dataset_summary_json),
        "gate_summary_json": str(gate_summary_json),
    },
}

validation_json.parent.mkdir(parents=True, exist_ok=True)
validation_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

split_rows = []
for split, item in split_results.items():
    split_rows.append(
        "<tr>"
        f"<td>{html.escape(split)}</td>"
        f"<td>{item['rows']}</td>"
        f"<td>{item['correct_delta_vs_lightglue']}</td>"
        f"<td>{item['wrong_delta_vs_lightglue']}</td>"
        "</tr>"
    )
validation_html.write_text(
    "\n".join(
        [
            "<!doctype html>",
            '<meta charset="utf-8">',
            "<title>Phase147 Fresh Gate Validation</title>",
            "<h1>Phase147 Fresh Gate Validation</h1>",
            f"<p>valid=<code>{str(payload['valid']).lower()}</code></p>",
            f"<p>base_disjoint=<code>{str(payload['base_disjoint']).lower()}</code></p>",
            f"<p>checkpoint=<code>{html.escape(checkpoint)}</code></p>",
            f"<p>frozen_gate=<code>{html.escape(frozen_gate)}</code></p>",
            f"<p>errors=<pre>{html.escape(json.dumps(errors, ensure_ascii=False, indent=2))}</pre></p>",
            '<table border="1" cellspacing="0" cellpadding="4">',
            "<tr><th>split</th><th>rows</th><th>correct delta</th><th>wrong delta</th></tr>",
            *split_rows,
            "</table>",
            "<h2>Payload</h2>",
            f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
        ]
    )
    + "\n",
    encoding="utf-8",
)

print(
    "phase147_fresh_gate_validation "
    f"valid={str(payload['valid']).lower()} "
    f"base_disjoint={str(payload['base_disjoint']).lower()} "
    f"correct_delta_vs_lightglue={aggregate_correct_delta} "
    f"wrong_delta_vs_lightglue={aggregate_wrong_delta} "
    f"validation={validation_json}",
    flush=True,
)
PY

cat >> "${RUN_RECORD}" <<HTML
<p>stage=<code>complete</code></p>
<p>baseline_summary=<code>${EVAL_ROOT}/baseline_summary.json</code></p>
<p>rejection_dataset=<code>${REJECTION_DATASET}</code></p>
<p>gate_summary=<code>${GATE_SUMMARY_JSON}</code></p>
<p>validation=<code>${VALIDATION_JSON}</code></p>
HTML

if [[ "${PFM_PHASE147_REQUIRE_VALID:-0}" == "1" ]]; then
  "${PY}" - "${VALIDATION_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not payload.get("valid", False):
    raise SystemExit("phase147 validation failed: " + "; ".join(payload.get("errors", [])))
PY
fi
