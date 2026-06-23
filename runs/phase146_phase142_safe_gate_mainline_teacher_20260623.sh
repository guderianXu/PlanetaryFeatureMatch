#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/w24/xjw_code/deeplearning/PlanetaryFeatureMatch"
cd "${PROJECT_ROOT}"

ACTIVE_PATTERN='batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda|benchmark_lazy_pose_pairs.py|visualize_lazy_pose_matches.py|run_lightglue|phase59_true_geometry_selector_multiseed_eval|train_match_detail_filter_calibrator.py|apply_match_detail_filter_calibrator.py|apply_observable_pair_gate_match_filter.py|train_match_set_rejection_calibrator.py|apply_match_set_rejection_calibrator.py|build_cluster_gate_dataset.py'
ACTIVE_TASKS="$(pgrep -af "${ACTIVE_PATTERN}" | grep -v -E 'pgrep -af|grep -v' || true)"
if [[ -n "${ACTIVE_TASKS}" ]]; then
  echo "[phase146] active long-running PFM task detected; refusing to start:" >&2
  echo "${ACTIVE_TASKS}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/scripts"
export TMPDIR="${PFM_TMPDIR:-/media/w24/D/xjw深度学习训练数据/tmp}"
mkdir -p "${TMPDIR}"

PY="${PFM_PHASE146_PYTHON:-/home/w24/anaconda3/envs/cppTorch/bin/python}"
PHASE141_ROOT="${PFM_PHASE141_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase141_extreme01_02_gap_replay_conservative_train_eval_20260622}"
PHASE142_ROOT="${PFM_PHASE142_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase142_phase141_observable_gate_sweep_20260622}"
RUN_ROOT="${PFM_PHASE146_RUN_ROOT:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase146_phase142_safe_gate_mainline_teacher_20260623}"
PREP_ROOT="${RUN_ROOT}/prep"
FROZEN_APPLY_ROOT="${RUN_ROOT}/frozen_phase142_safe_gate"
TEACHER_ROOT="${RUN_ROOT}/teacher"

SOURCE_DATASET="${PFM_PHASE146_SOURCE_DATASET:-${PHASE142_ROOT}/phase141_observable_gate_dataset.csv}"
FROZEN_GATE="${PFM_PHASE146_FROZEN_GATE:-feature_valid_fraction >= 0.356314 AND feature_homography_residual_median_px >= 1.195}"
FROZEN_HYBRID_ROWS="${FROZEN_APPLY_ROOT}/hybrid_rows.csv"
FROZEN_SUMMARY_JSON="${FROZEN_APPLY_ROOT}/summary.json"
DEV_HYBRID_ROWS="${PREP_ROOT}/dev_frozen_hybrid_rows.csv"
VAL_HYBRID_ROWS="${PREP_ROOT}/val_frozen_hybrid_rows.csv"
LOCKBOX_HYBRID_ROWS="${PREP_ROOT}/lockbox_frozen_hybrid_rows.csv"
TEACHER_MANIFEST="${TEACHER_ROOT}/phase146_pair_accept_teacher_manifest.csv"
TEACHER_SUMMARY_JSON="${TEACHER_ROOT}/phase146_pair_accept_teacher_manifest_summary.json"
TEACHER_REPORT_HTML="${TEACHER_ROOT}/phase146_pair_accept_teacher_manifest_summary.html"
GATE_CONFIG_JSON="${RUN_ROOT}/gate_config.json"
MAINLINE_SUMMARY_HTML="${RUN_ROOT}/phase146_mainline_summary.html"

for required in \
  "${SOURCE_DATASET}" \
  "${PHASE141_ROOT}/eval/dev_pairs.csv" \
  "${PHASE141_ROOT}/eval/val_pairs.csv" \
  "${PHASE141_ROOT}/eval/lockbox_pairs.csv"
do
  if [[ ! -f "${required}" ]]; then
    echo "[phase146] missing required input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${PREP_ROOT}" "${FROZEN_APPLY_ROOT}" "${TEACHER_ROOT}"

"${PY}" scripts/apply_observable_pair_gate.py \
  --dataset-csv "${SOURCE_DATASET}" \
  --gate "${FROZEN_GATE}" \
  --output-dir "${FROZEN_APPLY_ROOT}"

"${PY}" - \
  "${FROZEN_SUMMARY_JSON}" \
  "${FROZEN_HYBRID_ROWS}" \
  "${DEV_HYBRID_ROWS}" \
  "${VAL_HYBRID_ROWS}" \
  "${LOCKBOX_HYBRID_ROWS}" \
  "${GATE_CONFIG_JSON}" \
  "${MAINLINE_SUMMARY_HTML}" \
  "${SOURCE_DATASET}" \
  "${FROZEN_GATE}" \
  "${TEACHER_MANIFEST}" <<'PY'
import csv
import html
import json
import os
import sys
from pathlib import Path

summary_json = Path(sys.argv[1])
hybrid_rows_csv = Path(sys.argv[2])
dev_rows_csv = Path(sys.argv[3])
val_rows_csv = Path(sys.argv[4])
lockbox_rows_csv = Path(sys.argv[5])
gate_config_json = Path(sys.argv[6])
mainline_summary_html = Path(sys.argv[7])
source_dataset = Path(sys.argv[8])
frozen_gate = sys.argv[9]
teacher_manifest = Path(sys.argv[10])

summary = json.loads(summary_json.read_text(encoding="utf-8"))
required_summary_keys = [
    "rows",
    "kept_pfm_rows",
    "fallback_lightglue_rows",
    "correct_delta_vs_lightglue",
    "wrong_delta_vs_lightglue",
    "hybrid_precision",
]
missing = [key for key in required_summary_keys if key not in summary]
if missing:
    raise ValueError(f"summary is missing required keys: {missing}")

expected_exact = {
    "rows": os.environ.get("PFM_PHASE146_EXPECTED_ROWS", "78"),
    "kept_pfm_rows": os.environ.get("PFM_PHASE146_EXPECTED_KEPT_PFM_ROWS", "11"),
    "fallback_lightglue_rows": os.environ.get("PFM_PHASE146_EXPECTED_FALLBACK_LIGHTGLUE_ROWS", "67"),
    "correct_delta_vs_lightglue": os.environ.get("PFM_PHASE146_EXPECTED_CORRECT_DELTA_VS_LIGHTGLUE", "174"),
    "wrong_delta_vs_lightglue": os.environ.get("PFM_PHASE146_EXPECTED_WRONG_DELTA_VS_LIGHTGLUE", "-9"),
}
for key, raw_expected in expected_exact.items():
    if raw_expected == "":
        continue
    expected = int(raw_expected)
    actual = int(round(float(summary[key])))
    if actual != expected:
        raise ValueError(f"{key}={actual} does not match frozen expected value {expected}")

min_correct_delta = int(os.environ.get("PFM_PHASE146_MIN_CORRECT_DELTA_VS_LIGHTGLUE", "1"))
max_wrong_delta = int(os.environ.get("PFM_PHASE146_MAX_WRONG_DELTA_VS_LIGHTGLUE", "0"))
if int(round(float(summary["correct_delta_vs_lightglue"]))) < min_correct_delta:
    raise ValueError("frozen gate no longer increases correct matches over LightGlue")
if int(round(float(summary["wrong_delta_vs_lightglue"]))) > max_wrong_delta:
    raise ValueError("frozen gate no longer keeps wrong matches at or below LightGlue")

with hybrid_rows_csv.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fieldnames = list(reader.fieldnames or [])
    rows = [dict(row) for row in reader]
if not fieldnames:
    raise ValueError(f"{hybrid_rows_csv} is missing a CSV header")

split_outputs = {
    "dev": dev_rows_csv,
    "val": val_rows_csv,
    "lockbox": lockbox_rows_csv,
}
split_counts: dict[str, int] = {}
for split, path in split_outputs.items():
    split_rows = [row for row in rows if row.get("split", "") == split]
    split_counts[split] = len(split_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(split_rows)

if split_counts != {"dev": 26, "val": 26, "lockbox": 26}:
    raise ValueError(f"unexpected split counts for frozen gate rows: {split_counts}")

payload = {
    "phase": "phase146_phase142_safe_gate_mainline_teacher",
    "source_dataset": str(source_dataset),
    "frozen_gate": frozen_gate,
    "frozen_apply_root": str(summary_json.parent),
    "frozen_summary_json": str(summary_json),
    "frozen_hybrid_rows": str(hybrid_rows_csv),
    "split_hybrid_rows": {split: str(path) for split, path in split_outputs.items()},
    "teacher_manifest": str(teacher_manifest),
    "summary": summary,
    "teacher_policy": {
        "label_definition": "pair_accept_label=1 iff the frozen phase142 gate chooses PFM; otherwise 0",
        "training_splits": ["dev", "val"],
        "heldout_split": "lockbox",
        "uses_lightglue_as_runtime_fallback": True,
    },
}
gate_config_json.parent.mkdir(parents=True, exist_ok=True)
gate_config_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

metric_rows = "".join(
    f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(summary.get(key, '')))}</td></tr>"
    for key in required_summary_keys
)
mainline_summary_html.write_text(
    "\n".join(
        [
            "<!doctype html>",
            '<meta charset="utf-8">',
            "<title>Phase146 Safe Gate Mainline</title>",
            "<h1>Phase146 Safe Gate Mainline</h1>",
            f"<p>source_dataset={html.escape(str(source_dataset))}</p>",
            f"<p>frozen_gate={html.escape(frozen_gate)}</p>",
            f"<p>teacher_manifest={html.escape(str(teacher_manifest))}</p>",
            '<table border="1" cellspacing="0" cellpadding="4">',
            metric_rows,
            "</table>",
            "<h2>Gate Config</h2>",
            f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
        ]
    )
    + "\n",
    encoding="utf-8",
)

print(
    "phase146_frozen_gate_validated "
    f"rows={summary['rows']} "
    f"kept_pfm_rows={summary['kept_pfm_rows']} "
    f"correct_delta_vs_lightglue={summary['correct_delta_vs_lightglue']} "
    f"wrong_delta_vs_lightglue={summary['wrong_delta_vs_lightglue']} "
    f"gate_config={gate_config_json}",
    flush=True,
)
PY

"${PY}" scripts/build_gate_acceptance_training_manifest.py \
  --source "phase146_dev_teacher,${PHASE141_ROOT}/eval/dev_pairs.csv,${DEV_HYBRID_ROWS}" \
  --source "phase146_val_teacher,${PHASE141_ROOT}/eval/val_pairs.csv,${VAL_HYBRID_ROWS}" \
  --output-manifest "${TEACHER_MANIFEST}" \
  --summary-json "${TEACHER_SUMMARY_JSON}" \
  --report-html "${TEACHER_REPORT_HTML}" \
  --accept-weight "${PFM_PHASE146_TEACHER_ACCEPT_WEIGHT:-1.0}" \
  --reject-weight "${PFM_PHASE146_TEACHER_REJECT_WEIGHT:-3.0}" \
  --min-accept-precision 0.0 \
  --max-accept-wrong 999999 \
  --target-accept-fraction "${PFM_PHASE146_TEACHER_TARGET_ACCEPT_FRACTION:-0.25}"

echo "phase146_prep_only_complete run_root=${RUN_ROOT} gate_config=${GATE_CONFIG_JSON} teacher_manifest=${TEACHER_MANIFEST}"
