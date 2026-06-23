#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PFM_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/scripts"

PY="${PFM_PYTHON:-${PY:-/home/w24/anaconda3/envs/cppTorch/bin/python}}"
ROOT="${PFM_PHASE40_ROOT:-runs/phase40_crosscam_extreme_geometry_20260620}"
CHECKPOINT="${PFM_CANDIDATE_STATE:-/media/w24/D/xjw深度学习训练数据/pfm_runs/phase39a_devlock_failure_replay_main_4l384_20260620_110647/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"
RENDER_MANIFEST="${PFM_PHASE40_RENDER_MANIFEST:-${PFM_RENDER_MANIFEST:-/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/manifests/h100km_fov076_render_manifest.csv}}"
UINT8_MANIFEST="${PFM_PHASE40_UINT8_MANIFEST:-${PFM_UINT8_MANIFEST:-/media/w24/D/xjw深度学习训练数据/pfm_overlap_graphs/20260611_234228_h100_fov076_dom76_lat60_0m2e3_internal/manifests/h100km_fov076_uint8_manifest.csv}}"
LIGHTGLUE_RUNNER="${PFM_PHASE40_LIGHTGLUE_RUNNER:-${PFM_LIGHTGLUE_RUNNER:-${PROJECT_ROOT}/runs/fov76_phase24_dev_expansion_20260618/run_lightglue_for_manifest.py}}"
PFM_EVAL_SUBDIR="${PFM_EVAL_SUBDIR:-pfm_eval}"
PFM_MAX_KEYPOINTS="${PFM_MAX_KEYPOINTS:-1536}"
PFM_KEYPOINT_SPATIAL_BINS="${PFM_KEYPOINT_SPATIAL_BINS:-12}"
PFM_KEYPOINT_CELL_CAP="${PFM_KEYPOINT_CELL_CAP:-6}"
PFM_MATCHER_CANDIDATE_TOPK="${PFM_PHASE40_MATCHER_CANDIDATE_TOPK:-256}"
PFM_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_MODE:-}"
PFM_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_ALPHA:--1.0}"
PFM_PAIR_ACCEPT_MIN_PROBABILITY="${PFM_PAIR_ACCEPT_MIN_PROBABILITY:--1.0}"

declare -a SPLITS=(dev val lockbox)

mkdir -p "${ROOT}"
cat > "${ROOT}/baseline_record.html" <<HTML
<!doctype html><meta charset="utf-8">
<title>Phase40 Cross-Camera Extreme Baseline</title>
<h1>Phase40 Cross-Camera Extreme Baseline</h1>
<p>stage=<code>running</code></p>
<p>checkpoint=<code>${CHECKPOINT}</code></p>
<p>root=<code>${ROOT}</code></p>
<p>pfm_eval_subdir=<code>${PFM_EVAL_SUBDIR}</code></p>
<p>pfm_max_keypoints=<code>${PFM_MAX_KEYPOINTS}</code></p>
<p>pfm_matcher_candidate_topk=<code>${PFM_MATCHER_CANDIDATE_TOPK}</code></p>
<p>pfm_matcher_final_accept_score_mode=<code>${PFM_MATCHER_FINAL_ACCEPT_SCORE_MODE}</code></p>
<p>pfm_matcher_final_accept_score_alpha=<code>${PFM_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}</code></p>
<p>pfm_pair_accept_min_probability=<code>${PFM_PAIR_ACCEPT_MIN_PROBABILITY}</code></p>
<p>note=<code>PFM-only geo20 no-rescue versus LightGlue baseline. LightGlue is not used as a training label.</code></p>
HTML

run_pfm() {
  local split="$1"
  local manifest="${ROOT}/${split}_pairs.csv"
  local out_dir="${ROOT}/${split}/${PFM_EVAL_SUBDIR}"
  local summary="${out_dir}/all_filtered_summary.csv"
  local row_count
  row_count="$(($(wc -l < "${manifest}") - 1))"
  if [[ -s "${summary}" ]]; then
    echo "skip_pfm split=${split} summary=${summary}"
    return 0
  fi
  mkdir -p "${out_dir}"
  "${PY}" scripts/visualize_lazy_pose_matches.py \
    --render-manifest "${RENDER_MANIFEST}" \
    --uint8-manifest "${UINT8_MANIFEST}" \
    --pytorch-state "${CHECKPOINT}" \
    --output-dir "${out_dir}" \
    --split all --reference-variant nadir \
    --pair-spec-manifest "${manifest}" \
    --pair-mode cross-camera \
    --pair-type-weights same_position_view=0.0,cross_camera=1.0,cross_fov=0.0 \
    --image-source uint8 \
    --limit-pairs 0 --no-shuffle --candidate-pairs "${row_count}" --select-count 0 --seed 20260620 \
    --crop-size 2048 --max-image-size 768 \
    --max-attempts 4 --min-valid-fraction 0.02 \
    --absolute-depth-tolerance-m 100.0 --relative-depth-tolerance 0.005 \
    --device cuda \
    --descriptor-mode learned --texture-blend-weight 1.0 \
    --keypoint-score-mode learned --matcher-mode graph_matcher \
    --max-keypoints "${PFM_MAX_KEYPOINTS}" --min-intensity 0.01 \
    --texture-fraction 0.85 --weak-texture-fraction 0.05 \
    --keypoint-spatial-bins "${PFM_KEYPOINT_SPATIAL_BINS}" --keypoint-cell-cap "${PFM_KEYPOINT_CELL_CAP}" \
    --use-keypoint-offsets \
    --input-local-contrast --input-local-contrast-strength 0.35 --input-local-contrast-kernel 31 \
    --topk 1 --max-matches 0 --draw-matches 0 \
    --min-score -1.0 --min-margin 0.0 \
    --graph-dustbin-delta 0.0 --graph-acceptance-margin 0.0 \
    --graph-min-raw-score -1.0 --graph-min-raw-margin 0.0 \
    --graph-min-accept-probability -1.0 \
    --pair-accept-min-probability "${PFM_PAIR_ACCEPT_MIN_PROBABILITY}" \
    --graph-width-prune-min-score -1.0 --graph-early-stop-min-confidence -1.0 \
    --graph-max-attention-layers 0 --graph-max-attention-work-fraction 1.0 \
    --graph-width-prune-keep-ratio 1.0 --matcher-candidate-topk "${PFM_MATCHER_CANDIDATE_TOPK}" \
    --matcher-final-accept-score-mode "${PFM_MATCHER_FINAL_ACCEPT_SCORE_MODE}" \
    --matcher-final-accept-score-alpha "${PFM_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}" \
    --no-mutual \
    --geometry-filter local --geometry-threshold-px 2.0 \
    --filtered-report --filtered-mutual --filtered-geometry-filter magsac \
    --write-all-summary --write-match-details \
    --filtered-max-matches 0 --filtered-draw-matches 0 \
    --filtered-min-score -1.0 --filtered-min-margin 0.0 --filtered-min-matches 16 \
    --threshold-px 5.0 --no-illumination-stress --no-html-report
}

run_lightglue() {
  local split="$1"
  local manifest="${ROOT}/${split}_pairs.csv"
  local out_dir="${ROOT}/${split}/lightglue"
  local metrics="${out_dir}/lightglue_sift_metrics.csv"
  local summary="${out_dir}/lightglue_sift_summary.json"
  if [[ -s "${metrics}" && -s "${summary}" ]]; then
    echo "skip_lightglue split=${split} metrics=${metrics}"
    return 0
  fi
  mkdir -p "${out_dir}"
  "${PY}" "${LIGHTGLUE_RUNNER}" \
    --root "${out_dir}" \
    --pair-manifest "${manifest}" \
    --metrics-csv "${metrics}" \
    --summary-json "${summary}" \
    --seed 20260620
}

for split in "${SPLITS[@]}"; do
  run_pfm "${split}"
  run_lightglue "${split}"
done

"${PY}" - "${ROOT}" "${CHECKPOINT}" "${SPLITS[@]}" <<'PY'
import csv
import html
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = sys.argv[2]
splits = sys.argv[3:]
pfm_eval_subdir = os.environ.get("PFM_EVAL_SUBDIR", "pfm_eval")
pfm_max_keypoints = int(os.environ.get("PFM_MAX_KEYPOINTS", "1536"))
pfm_matcher_candidate_topk = int(os.environ.get("PFM_PHASE40_MATCHER_CANDIDATE_TOPK", "256"))
pfm_matcher_final_accept_score_mode = os.environ.get("PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_MODE", "")
pfm_matcher_final_accept_score_alpha = float(os.environ.get("PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_ALPHA", "-1.0"))
pfm_pair_accept_min_probability = float(os.environ.get("PFM_PAIR_ACCEPT_MIN_PROBABILITY", "-1.0"))


def read_pfm(split: str) -> dict[str, object]:
    path = root / split / pfm_eval_subdir / "all_filtered_summary.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    matches = sum(int(float(row.get("matches", 0) or 0)) for row in rows)
    correct = sum(int(float(row.get("correct", 0) or 0)) for row in rows)
    wrong = sum(int(float(row.get("wrong", 0) or 0)) for row in rows)
    return {"path": str(path), "rows": len(rows), "matches": matches, "correct": correct, "wrong": wrong}


def read_lightglue(split: str) -> dict[str, object]:
    path = root / split / "lightglue" / "lightglue_sift_metrics.csv"
    rows = [
        row
        for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
        if row.get("label") == "LightGlue-SIFT-MAGSAC-min16"
    ]
    matches = sum(int(float(row.get("matches", 0) or 0)) for row in rows)
    correct = sum(int(float(row.get("correct", 0) or 0)) for row in rows)
    wrong = sum(int(float(row.get("wrong", 0) or 0)) for row in rows)
    return {"path": str(path), "rows": len(rows), "matches": matches, "correct": correct, "wrong": wrong}


def finalize(item: dict[str, object]) -> dict[str, object]:
    pfm_matches = int(item["pfm_matches"])
    lg_matches = int(item["lightglue_matches"])
    item["pfm_precision"] = int(item["pfm_correct"]) / pfm_matches if pfm_matches else 0.0
    item["lightglue_precision"] = int(item["lightglue_correct"]) / lg_matches if lg_matches else 0.0
    item["correct_delta_vs_lightglue"] = int(item["pfm_correct"]) - int(item["lightglue_correct"])
    item["wrong_delta_vs_lightglue"] = int(item["pfm_wrong"]) - int(item["lightglue_wrong"])
    item["precision_delta_vs_lightglue"] = float(item["pfm_precision"]) - float(item["lightglue_precision"])
    return item


by_split = {}
aggregate = {
    "rows": 0,
    "pfm_matches": 0,
    "pfm_correct": 0,
    "pfm_wrong": 0,
    "lightglue_matches": 0,
    "lightglue_correct": 0,
    "lightglue_wrong": 0,
}
for split in splits:
    pfm = read_pfm(split)
    lg = read_lightglue(split)
    item = {
        "rows": int(pfm["rows"]),
        "pfm_matches": int(pfm["matches"]),
        "pfm_correct": int(pfm["correct"]),
        "pfm_wrong": int(pfm["wrong"]),
        "lightglue_matches": int(lg["matches"]),
        "lightglue_correct": int(lg["correct"]),
        "lightglue_wrong": int(lg["wrong"]),
        "pfm_path": pfm["path"],
        "lightglue_path": lg["path"],
    }
    by_split[split] = finalize(item)
    for key in ("rows", "pfm_matches", "pfm_correct", "pfm_wrong", "lightglue_matches", "lightglue_correct", "lightglue_wrong"):
        aggregate[key] += int(item[key])
aggregate = finalize(aggregate)

payload = {
    "checkpoint": checkpoint,
    "root": str(root),
    "pfm_eval_subdir": pfm_eval_subdir,
    "pfm_max_keypoints": pfm_max_keypoints,
    "pfm_matcher_candidate_topk": pfm_matcher_candidate_topk,
    "pfm_matcher_final_accept_score_mode": pfm_matcher_final_accept_score_mode,
    "pfm_matcher_final_accept_score_alpha": pfm_matcher_final_accept_score_alpha,
    "pfm_pair_accept_min_probability": pfm_pair_accept_min_probability,
    "aggregate": aggregate,
    "by_split": by_split,
}
(root / "baseline_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

rows = []
for split, item in [("aggregate", aggregate), *by_split.items()]:
    rows.append(
        "<tr>"
        f"<td>{html.escape(split)}</td>"
        f"<td>{item['rows']}</td>"
        f"<td>{item['pfm_correct']}</td>"
        f"<td>{item['pfm_wrong']}</td>"
        f"<td>{float(item['pfm_precision']):.6f}</td>"
        f"<td>{item['lightglue_correct']}</td>"
        f"<td>{item['lightglue_wrong']}</td>"
        f"<td>{float(item['lightglue_precision']):.6f}</td>"
        f"<td>{item['correct_delta_vs_lightglue']}</td>"
        f"<td>{item['wrong_delta_vs_lightglue']}</td>"
        f"<td>{float(item['precision_delta_vs_lightglue']):+.6f}</td>"
        "</tr>"
    )
(root / "baseline_summary.html").write_text(
    "\n".join(
        [
            "<!doctype html>",
            '<meta charset="utf-8">',
            "<title>Phase40 Cross-Camera Extreme Baseline Summary</title>",
            "<h1>Phase40 Cross-Camera Extreme Baseline Summary</h1>",
            f"<p>checkpoint=<code>{html.escape(checkpoint)}</code></p>",
            '<table border="1" cellspacing="0" cellpadding="4">',
            "<tr><th>split</th><th>rows</th><th>PFM correct</th><th>PFM wrong</th><th>PFM precision</th><th>LightGlue correct</th><th>LightGlue wrong</th><th>LightGlue precision</th><th>delta correct</th><th>delta wrong</th><th>delta precision</th></tr>",
            *rows,
            "</table>",
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2), flush=True)
PY

cat > "${ROOT}/baseline_record.html" <<HTML
<!doctype html><meta charset="utf-8">
<title>Phase40 Cross-Camera Extreme Baseline</title>
<h1>Phase40 Cross-Camera Extreme Baseline</h1>
<p>stage=<code>complete</code></p>
<p>checkpoint=<code>${CHECKPOINT}</code></p>
<p>summary=<code>${ROOT}/baseline_summary.html</code></p>
HTML

echo "phase40_crosscam_extreme_baseline_complete root=${ROOT}"
