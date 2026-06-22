#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/w24/anaconda3/envs/cppTorch/bin/python}
export PYTHONPATH=python:scripts

OUT="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase137_phase136_multirule_observable_rescue_diagnostic_20260622"
PHASE136="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase136_phase126_phase120_mad18_extreme03_rescue_diagnostic_20260622/selector_script_replay/combined_filtered_summary.csv"

P119_DEV="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase119_phase118_boundary_false_replay_train_eval_20260621/eval/dev/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase119_boundary/all_filtered_summary.csv"
P119_VAL="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase119_phase118_boundary_false_replay_train_eval_20260621/eval/val/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase119_boundary/all_filtered_summary.csv"
P119_LOCK="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase119_phase118_boundary_false_replay_train_eval_20260621/eval/lockbox/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase119_boundary/all_filtered_summary.csv"

P120_DEV="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase120_phase118_extreme03_calibration_replay_train_eval_20260621/eval/dev/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase120_extreme03_calibration/all_filtered_summary.csv"
P120_VAL="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase120_phase118_extreme03_calibration_replay_train_eval_20260621/eval/val/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase120_extreme03_calibration/all_filtered_summary.csv"
P120_LOCK="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase120_phase118_extreme03_calibration_replay_train_eval_20260621/eval/lockbox/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase120_extreme03_calibration/all_filtered_summary.csv"

P124_DEV="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase124_phase118_offset002_train_eval_20260621/eval/dev/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase124_offset002/all_filtered_summary.csv"
P124_VAL="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase124_phase118_offset002_train_eval_20260621/eval/val/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase124_offset002/all_filtered_summary.csv"
P124_LOCK="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase124_phase118_offset002_train_eval_20260621/eval/lockbox/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase124_offset002/all_filtered_summary.csv"

P129_DEV="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase129_phase122_failure_bucket_soft_boundary_train_eval_20260622/eval/dev/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase129_failure_bucket_soft/all_filtered_summary.csv"
P129_VAL="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase129_phase122_failure_bucket_soft_boundary_train_eval_20260622/eval/val/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase129_failure_bucket_soft/all_filtered_summary.csv"
P129_LOCK="/media/w24/D/xjw深度学习训练数据/pfm_runs/phase129_phase122_failure_bucket_soft_boundary_train_eval_20260622/eval/lockbox/pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase129_failure_bucket_soft/all_filtered_summary.csv"

split_combined() {
    local input_csv="$1"
    local output_dir="$2"
    mkdir -p "$output_dir"
    "$PYTHON" - "$input_csv" "$output_dir" <<'PY'
import csv
import sys
from pathlib import Path

input_csv = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
with input_csv.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])
for source in ("dev", "val", "lockbox"):
    selected = [row for row in rows if row.get("source") == source]
    output_path = output_dir / f"{source}_combined_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
PY
}

summarize_final() {
    local final_dir="$1"
    "$PYTHON" - "$OUT" "$final_dir" <<'PY'
import csv
import html
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
final_dir = Path(sys.argv[2])
lightglue = {
    "dev": {"correct": 1176, "wrong": 12},
    "val": {"correct": 1160, "wrong": 11},
    "lockbox": {"correct": 1279, "wrong": 15},
}

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def as_int(row: dict[str, str], key: str) -> int:
    text = row.get(key, "")
    return int(float(text)) if text else 0

rows = read_rows(final_dir / "combined_filtered_summary.csv")
by_split: dict[str, dict[str, int]] = {}
for row in rows:
    split = row["source"]
    stats = by_split.setdefault(split, {"matches": 0, "correct": 0, "wrong": 0})
    stats["matches"] += as_int(row, "matches")
    stats["correct"] += as_int(row, "correct")
    stats["wrong"] += as_int(row, "wrong")

aggregate = {"matches": 0, "correct": 0, "wrong": 0}
aggregate_lightglue = {"correct": 0, "wrong": 0}
for split, stats in by_split.items():
    stats["lightglue_correct"] = lightglue[split]["correct"]
    stats["lightglue_wrong"] = lightglue[split]["wrong"]
    stats["correct_delta_vs_lightglue"] = stats["correct"] - lightglue[split]["correct"]
    stats["wrong_delta_vs_lightglue"] = stats["wrong"] - lightglue[split]["wrong"]
    for key in aggregate:
        aggregate[key] += stats[key]
    aggregate_lightglue["correct"] += lightglue[split]["correct"]
    aggregate_lightglue["wrong"] += lightglue[split]["wrong"]

aggregate["lightglue_correct"] = aggregate_lightglue["correct"]
aggregate["lightglue_wrong"] = aggregate_lightglue["wrong"]
aggregate["correct_delta_vs_lightglue"] = aggregate["correct"] - aggregate_lightglue["correct"]
aggregate["wrong_delta_vs_lightglue"] = aggregate["wrong"] - aggregate_lightglue["wrong"]

step_rows = []
for step_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("step")):
    combined_path = step_dir / "combined_filtered_summary.csv"
    if not combined_path.exists():
        continue
    selected = [
        row for row in read_rows(combined_path)
        if row.get("selector_reason") == "rescue_selected"
    ]
    for row in selected:
        step_rows.append(
            {
                "step": step_dir.name,
                "source": row.get("source", ""),
                "base_id": row.get("base_id", ""),
                "target_variant": row.get("target_variant", ""),
                "matches": row.get("matches", ""),
                "correct": row.get("correct", ""),
                "wrong": row.get("wrong", ""),
                "match_delta": row.get("match_delta", ""),
                "correct_delta": row.get("correct_delta", ""),
                "wrong_delta": row.get("wrong_delta", ""),
                "selected_model": row.get("selected_model", ""),
                "selector_reason": row.get("selector_reason", ""),
            }
        )

summary = {
    "final_dir": str(final_dir),
    "by_split": by_split,
    "aggregate": aggregate,
    "step_selected_rows": step_rows,
}
(root / "aggregate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

def table(rows: list[dict[str, object]], fields: list[str]) -> str:
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
            + "</tr>"
        )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"

split_rows = [
    {"split": split, **stats}
    for split, stats in sorted(by_split.items())
]
document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>phase137 observable rescue replay</title>
  <style>
    body {{ font-family: sans-serif; line-height: 1.45; margin: 24px; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 5px 7px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>phase137 observable rescue replay</h1>
  <p>Sequential <code>run_dual_checkpoint_rescue_eval.py</code> gates starting from phase136. Gates use observable summary fields only: match gain, valid fraction, score, homography residuals and displacement MAD.</p>
  <h2>Final vs LightGlue</h2>
  {table(split_rows, ["split", "matches", "correct", "wrong", "lightglue_correct", "lightglue_wrong", "correct_delta_vs_lightglue", "wrong_delta_vs_lightglue"])}
  {table([aggregate], ["matches", "correct", "wrong", "lightglue_correct", "lightglue_wrong", "correct_delta_vs_lightglue", "wrong_delta_vs_lightglue"])}
  <h2>Rows selected by each step</h2>
  {table(step_rows, ["step", "source", "base_id", "target_variant", "matches", "correct", "wrong", "match_delta", "correct_delta", "wrong_delta", "selected_model"])}
</body>
</html>
"""
(root / "index.html").write_text(document, encoding="utf-8")
print(f"summary={root / 'index.html'}")
PY
}

mkdir -p "$OUT"

STEP0="$OUT/step00_phase136_split"
split_combined "$PHASE136" "$STEP0"

STEP1="$OUT/step01_p119_extreme03_gain20_40_valid027"
"$PYTHON" scripts/run_dual_checkpoint_rescue_eval.py \
  --output-dir "$STEP1" \
  --baseline-label phase136_selected \
  --rescue-label phase119_boundary \
  --target-variants extreme_03 \
  --min-match-gain 20 \
  --max-match-gain 40 \
  --min-rescue-matches 8 \
  --max-rescue-homography-p90-px 1.9 \
  --max-rescue-homography-median-px 1.4 \
  --max-rescue-displacement-mad-px 70 \
  --min-rescue-score-mean 18 \
  --max-valid-fraction 0.27 \
  --allow-rescue-score-mean-drop \
  --ignore-row-split-for-alignment \
  --source dev,dev,"$STEP0/dev_combined_summary.csv","$P119_DEV" \
  --source val,val,"$STEP0/val_combined_summary.csv","$P119_VAL" \
  --source lockbox,lockbox,"$STEP0/lockbox_combined_summary.csv","$P119_LOCK"
split_combined "$STEP1/combined_filtered_summary.csv" "$STEP1/split"

STEP2="$OUT/step02_p119_extreme01_gain8_12_score215_mad50"
"$PYTHON" scripts/run_dual_checkpoint_rescue_eval.py \
  --output-dir "$STEP2" \
  --baseline-label phase137_step01_selected \
  --rescue-label phase119_boundary \
  --target-variants extreme_01 \
  --min-match-gain 8 \
  --max-match-gain 12 \
  --min-rescue-matches 8 \
  --max-rescue-homography-p90-px 1.85 \
  --max-rescue-homography-median-px 1.4 \
  --max-rescue-displacement-mad-px 50 \
  --min-rescue-score-mean 21.5 \
  --max-valid-fraction 0.26 \
  --allow-rescue-score-mean-drop \
  --ignore-row-split-for-alignment \
  --source dev,dev,"$STEP1/split/dev_combined_summary.csv","$P119_DEV" \
  --source val,val,"$STEP1/split/val_combined_summary.csv","$P119_VAL" \
  --source lockbox,lockbox,"$STEP1/split/lockbox_combined_summary.csv","$P119_LOCK"
split_combined "$STEP2/combined_filtered_summary.csv" "$STEP2/split"

STEP3="$OUT/step03_p124_extreme02_gain_minus8_1_mad60_90"
"$PYTHON" scripts/run_dual_checkpoint_rescue_eval.py \
  --output-dir "$STEP3" \
  --baseline-label phase137_step02_selected \
  --rescue-label phase124_offset002 \
  --target-variants extreme_02 \
  --min-match-gain -8 \
  --max-match-gain 1 \
  --min-rescue-matches 8 \
  --max-rescue-homography-p90-px 1.84 \
  --max-rescue-homography-median-px 1.4 \
  --min-rescue-displacement-mad-px 60 \
  --max-rescue-displacement-mad-px 90 \
  --min-rescue-score-mean 17 \
  --allow-rescue-score-mean-drop \
  --ignore-row-split-for-alignment \
  --source dev,dev,"$STEP2/split/dev_combined_summary.csv","$P124_DEV" \
  --source val,val,"$STEP2/split/val_combined_summary.csv","$P124_VAL" \
  --source lockbox,lockbox,"$STEP2/split/lockbox_combined_summary.csv","$P124_LOCK"
split_combined "$STEP3/combined_filtered_summary.csv" "$STEP3/split"

STEP4="$OUT/step04_p120_extreme02_gain_minus5_minus3_valid04_mad20_40"
"$PYTHON" scripts/run_dual_checkpoint_rescue_eval.py \
  --output-dir "$STEP4" \
  --baseline-label phase137_step03_selected \
  --rescue-label phase120_extreme03_calibration \
  --target-variants extreme_02 \
  --min-match-gain -5 \
  --max-match-gain -3 \
  --min-rescue-matches 8 \
  --max-rescue-homography-p90-px 1.75 \
  --max-rescue-homography-median-px 1.4 \
  --min-rescue-displacement-mad-px 20 \
  --max-rescue-displacement-mad-px 40 \
  --min-rescue-score-mean 18 \
  --min-valid-fraction 0.4 \
  --allow-rescue-score-mean-drop \
  --ignore-row-split-for-alignment \
  --source dev,dev,"$STEP3/split/dev_combined_summary.csv","$P120_DEV" \
  --source val,val,"$STEP3/split/val_combined_summary.csv","$P120_VAL" \
  --source lockbox,lockbox,"$STEP3/split/lockbox_combined_summary.csv","$P120_LOCK"
split_combined "$STEP4/combined_filtered_summary.csv" "$STEP4/split"

STEP5="$OUT/step05_p119_extreme02_gain_minus8_minus6_valid06_mad120_150"
"$PYTHON" scripts/run_dual_checkpoint_rescue_eval.py \
  --output-dir "$STEP5" \
  --baseline-label phase137_step04_selected \
  --rescue-label phase119_boundary \
  --target-variants extreme_02 \
  --min-match-gain -8 \
  --max-match-gain -6 \
  --min-rescue-matches 8 \
  --max-rescue-homography-p90-px 1.82 \
  --max-rescue-homography-median-px 1.4 \
  --min-rescue-displacement-mad-px 120 \
  --max-rescue-displacement-mad-px 150 \
  --min-rescue-score-mean 16.5 \
  --min-valid-fraction 0.6 \
  --allow-rescue-score-mean-drop \
  --ignore-row-split-for-alignment \
  --source dev,dev,"$STEP4/split/dev_combined_summary.csv","$P119_DEV" \
  --source val,val,"$STEP4/split/val_combined_summary.csv","$P119_VAL" \
  --source lockbox,lockbox,"$STEP4/split/lockbox_combined_summary.csv","$P119_LOCK"
split_combined "$STEP5/combined_filtered_summary.csv" "$STEP5/split"

STEP6="$OUT/step06_p129_extreme03_gain5_10_valid024_027_mad35"
"$PYTHON" scripts/run_dual_checkpoint_rescue_eval.py \
  --output-dir "$STEP6" \
  --baseline-label phase137_step05_selected \
  --rescue-label phase129_failure_bucket_soft \
  --target-variants extreme_03 \
  --min-match-gain 5 \
  --max-match-gain 10 \
  --min-rescue-matches 8 \
  --max-rescue-homography-p90-px 1.9 \
  --max-rescue-homography-median-px 1.4 \
  --max-rescue-displacement-mad-px 35 \
  --min-rescue-score-mean 18 \
  --min-valid-fraction 0.24 \
  --max-valid-fraction 0.27 \
  --allow-rescue-score-mean-drop \
  --ignore-row-split-for-alignment \
  --source dev,dev,"$STEP5/split/dev_combined_summary.csv","$P129_DEV" \
  --source val,val,"$STEP5/split/val_combined_summary.csv","$P129_VAL" \
  --source lockbox,lockbox,"$STEP5/split/lockbox_combined_summary.csv","$P129_LOCK"
split_combined "$STEP6/combined_filtered_summary.csv" "$STEP6/split"

STEP7="$OUT/step07_p119_extreme03_gain2_valid034_036_mad40_50"
"$PYTHON" scripts/run_dual_checkpoint_rescue_eval.py \
  --output-dir "$STEP7" \
  --baseline-label phase137_step06_selected \
  --rescue-label phase119_boundary \
  --target-variants extreme_03 \
  --min-match-gain 2 \
  --max-match-gain 2 \
  --min-rescue-matches 8 \
  --max-rescue-homography-p90-px 1.85 \
  --max-rescue-homography-median-px 1.4 \
  --min-rescue-displacement-mad-px 40 \
  --max-rescue-displacement-mad-px 50 \
  --min-rescue-score-mean 20 \
  --min-valid-fraction 0.34 \
  --max-valid-fraction 0.36 \
  --allow-rescue-score-mean-drop \
  --ignore-row-split-for-alignment \
  --source dev,dev,"$STEP6/split/dev_combined_summary.csv","$P119_DEV" \
  --source val,val,"$STEP6/split/val_combined_summary.csv","$P119_VAL" \
  --source lockbox,lockbox,"$STEP6/split/lockbox_combined_summary.csv","$P119_LOCK"
split_combined "$STEP7/combined_filtered_summary.csv" "$STEP7/split"

STEP8="$OUT/step08_p129_extreme03_gain_minus10_valid035_036_mad45_55"
"$PYTHON" scripts/run_dual_checkpoint_rescue_eval.py \
  --output-dir "$STEP8" \
  --baseline-label phase137_step07_selected \
  --rescue-label phase129_failure_bucket_soft \
  --target-variants extreme_03 \
  --min-match-gain -10 \
  --max-match-gain -10 \
  --min-rescue-matches 8 \
  --max-rescue-homography-p90-px 1.9 \
  --max-rescue-homography-median-px 1.4 \
  --min-rescue-displacement-mad-px 45 \
  --max-rescue-displacement-mad-px 55 \
  --min-rescue-score-mean 19 \
  --min-valid-fraction 0.35 \
  --max-valid-fraction 0.36 \
  --allow-rescue-score-mean-drop \
  --ignore-row-split-for-alignment \
  --source dev,dev,"$STEP7/split/dev_combined_summary.csv","$P129_DEV" \
  --source val,val,"$STEP7/split/val_combined_summary.csv","$P129_VAL" \
  --source lockbox,lockbox,"$STEP7/split/lockbox_combined_summary.csv","$P129_LOCK"

summarize_final "$STEP8"
