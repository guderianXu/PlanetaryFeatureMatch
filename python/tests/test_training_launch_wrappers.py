import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TrainingLaunchWrappersTest(unittest.TestCase):
    def read_script(self, relative_path: str) -> str:
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_phase41_crosscam_launcher_exposes_phase42_candidate_pressure_overrides(self) -> None:
        text = self.read_script("runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh")

        self.assertIn('TRAIN_SAMPLES_PER_PAIR="${PFM_PHASE41_TRAIN_SAMPLES_PER_PAIR:-256}"', text)
        self.assertIn('TRAIN_SPATIAL_BINS="${PFM_PHASE41_TRAIN_SPATIAL_BINS:-8}"', text)
        self.assertIn('TRAIN_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_TRAIN_MATCHER_CANDIDATE_TOPK:-256}"', text)
        self.assertIn(
            'TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK:-${TRAIN_MATCHER_CANDIDATE_TOPK}}"',
            text,
        )
        self.assertIn('TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${DATA_ROOT}/train_pairs_geometry_accept.csv}"', text)
        self.assertIn('--samples-per-pair "${TRAIN_SAMPLES_PER_PAIR}"', text)
        self.assertIn('--training-spatial-bins "${TRAIN_SPATIAL_BINS}"', text)
        self.assertIn('--matcher-candidate-topk "${TRAIN_MATCHER_CANDIDATE_TOPK}"', text)
        self.assertIn('--graph-matcher-train-candidate-topk "${TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK}"', text)

    def test_phase40_crosscam_eval_launcher_allows_matcher_candidate_topk_override(self) -> None:
        text = self.read_script("runs/phase40_crosscam_extreme_baseline_20260620.sh")

        self.assertIn('PFM_MATCHER_CANDIDATE_TOPK="${PFM_PHASE40_MATCHER_CANDIDATE_TOPK:-256}"', text)
        self.assertIn('pfm_matcher_candidate_topk=<code>${PFM_MATCHER_CANDIDATE_TOPK}</code>', text)
        self.assertIn('--matcher-candidate-topk "${PFM_MATCHER_CANDIDATE_TOPK}"', text)

    def test_phase40_crosscam_eval_launcher_allows_final_accept_score_override(self) -> None:
        text = self.read_script("runs/phase40_crosscam_extreme_baseline_20260620.sh")

        self.assertIn('PFM_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_MODE:-}"', text)
        self.assertIn(
            'PFM_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_ALPHA:--1.0}"',
            text,
        )
        self.assertIn('pfm_matcher_final_accept_score_mode=<code>${PFM_MATCHER_FINAL_ACCEPT_SCORE_MODE}</code>', text)
        self.assertIn('pfm_matcher_final_accept_score_alpha=<code>${PFM_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}</code>', text)
        self.assertIn('--matcher-final-accept-score-mode "${PFM_MATCHER_FINAL_ACCEPT_SCORE_MODE}"', text)
        self.assertIn('--matcher-final-accept-score-alpha "${PFM_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}"', text)

    def test_phase41_crosscam_launcher_allows_eval_topk_and_subdir_override(self) -> None:
        text = self.read_script("runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh")

        self.assertIn('EVAL_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK:-256}"', text)
        self.assertIn('PFM_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp${EVAL_MAX_KEYPOINTS}}"', text)
        self.assertIn('eval_matcher_candidate_topk=<code>${EVAL_MATCHER_CANDIDATE_TOPK}</code>', text)
        self.assertIn('PFM_PHASE40_MATCHER_CANDIDATE_TOPK="${EVAL_MATCHER_CANDIDATE_TOPK}"', text)

    def test_phase41_crosscam_launcher_allows_eval_final_accept_score_override(self) -> None:
        text = self.read_script("runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh")

        self.assertIn('EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE:-}"', text)
        self.assertIn(
            'EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA:--1.0}"',
            text,
        )
        self.assertIn(
            'eval_matcher_final_accept_score_mode=<code>${EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE}</code>',
            text,
        )
        self.assertIn(
            'eval_matcher_final_accept_score_alpha=<code>${EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}</code>',
            text,
        )
        self.assertIn(
            'PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_MODE="${EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE}"',
            text,
        )
        self.assertIn(
            'PFM_PHASE40_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${EVAL_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}"',
            text,
        )

    def test_phase41_crosscam_launcher_allows_train_final_accept_score_override(self) -> None:
        text = self.read_script("runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh")

        self.assertIn('TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE41_TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE:-none}"', text)
        self.assertIn(
            'TRAIN_MATCHER_FINAL_ACCEPT_SCORE_ALPHA="${PFM_PHASE41_TRAIN_MATCHER_FINAL_ACCEPT_SCORE_ALPHA:-0.05}"',
            text,
        )
        self.assertIn(
            'train_matcher_final_accept_score=<code>${TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE}/${TRAIN_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}</code>',
            text,
        )
        self.assertIn('--matcher-final-accept-score-mode "${TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE}"', text)
        self.assertIn('--matcher-final-accept-score-alpha "${TRAIN_MATCHER_FINAL_ACCEPT_SCORE_ALPHA}"', text)

    def test_phase41_crosscam_launcher_exposes_phase42_precision_guard_overrides(self) -> None:
        text = self.read_script("runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh")

        self.assertIn('FINAL_FALSE_MATCH_WEIGHT="${PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT:-0.015}"', text)
        self.assertIn('FINAL_FALSE_MATCH_TOPK="${PFM_PHASE41_FINAL_FALSE_MATCH_TOPK:-12}"', text)
        self.assertIn('MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.0}"', text)
        self.assertIn(
            'MINED_FALSE_MATCH_REFERENCE_MARGIN="${PFM_PHASE41_MINED_FALSE_MATCH_REFERENCE_MARGIN:--1.0}"',
            text,
        )
        self.assertIn('FALSE_MATCH_CSV="${PFM_PHASE41_FALSE_MATCH_CSV:-}"', text)
        self.assertIn('FALSE_MATCH_FLAGS=()', text)
        self.assertIn('FALSE_MATCH_FLAGS+=(--false-match-csv "${false_match_csv}")', text)
        self.assertIn('RAW_FALSE_MATCH_WEIGHT="${PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT:-0.006}"', text)
        self.assertIn('RAW_FALSE_MATCH_TOPK="${PFM_PHASE41_RAW_FALSE_MATCH_TOPK:-8}"', text)
        self.assertIn('WARP_OUTLIER_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_WEIGHT:-0.08}"', text)
        self.assertIn('WARP_OUTLIER_TOPK="${PFM_PHASE41_WARP_OUTLIER_TOPK:-12}"', text)
        self.assertIn('WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.15}"', text)
        self.assertIn('WARP_OUTLIER_ACCEPT_TOPK="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_TOPK:-12}"', text)
        self.assertIn('FALSE_CLUSTER_REPLAY_MULTIPLIER="${PFM_PHASE41_FALSE_CLUSTER_REPLAY_MULTIPLIER:-1.0}"', text)
        self.assertIn('"${FALSE_MATCH_FLAGS[@]}"', text)
        self.assertIn('--graph-matcher-final-false-match-weight "${FINAL_FALSE_MATCH_WEIGHT}"', text)
        self.assertIn('--graph-matcher-final-false-match-topk "${FINAL_FALSE_MATCH_TOPK}"', text)
        self.assertIn('--graph-matcher-mined-false-match-weight "${MINED_FALSE_MATCH_WEIGHT}"', text)
        self.assertIn('--graph-matcher-mined-false-match-loss-cap "${MINED_FALSE_MATCH_LOSS_CAP}"', text)
        self.assertIn(
            '--graph-matcher-mined-false-match-reference-margin "${MINED_FALSE_MATCH_REFERENCE_MARGIN}"',
            text,
        )
        self.assertIn('--graph-matcher-raw-false-match-weight "${RAW_FALSE_MATCH_WEIGHT}"', text)
        self.assertIn('--graph-matcher-raw-false-match-topk "${RAW_FALSE_MATCH_TOPK}"', text)
        self.assertIn('--graph-matcher-warp-outlier-weight "${WARP_OUTLIER_WEIGHT}"', text)
        self.assertIn('--graph-matcher-warp-outlier-topk "${WARP_OUTLIER_TOPK}"', text)
        self.assertIn('--graph-matcher-warp-outlier-accept-weight "${WARP_OUTLIER_ACCEPT_WEIGHT}"', text)
        self.assertIn('--graph-matcher-warp-outlier-accept-topk "${WARP_OUTLIER_ACCEPT_TOPK}"', text)
        self.assertIn('--false-cluster-replay-loss-multiplier "${FALSE_CLUSTER_REPLAY_MULTIPLIER}"', text)

    def test_phase41_crosscam_launcher_can_apply_fixed_geometry_overlap_gate(self) -> None:
        text = self.read_script("runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh")

        self.assertIn('GEOMETRY_OVERLAP_GATE_THRESHOLD="${PFM_PHASE41_GEOMETRY_OVERLAP_GATE_THRESHOLD:-}"', text)
        self.assertIn(
            'GEOMETRY_OVERLAP_GATE_THRESHOLDS="${PFM_PHASE41_GEOMETRY_OVERLAP_GATE_THRESHOLDS:-0.02,0.08,0.10,0.12,0.15,0.20,0.25,0.30}"',
            text,
        )
        self.assertIn("scripts/sweep_geometry_overlap_gate.py", text)
        self.assertIn('--selected-threshold "${GEOMETRY_OVERLAP_GATE_THRESHOLD}"', text)
        self.assertIn("geometry_overlap_gate/aggregate_selected_threshold_summary.json", text)
        self.assertIn("geometry_overlap_gate_summary=<code>${EVAL_ROOT}/geometry_overlap_gate/aggregate_selected_threshold_summary.json</code>", text)

    def test_phase41_crosscam_launcher_allows_learning_rate_override(self) -> None:
        text = self.read_script("runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh")

        self.assertIn('LEARNING_RATE="${PFM_PHASE41_LEARNING_RATE:-4e-6}"', text)
        self.assertIn('learning_rate=<code>${LEARNING_RATE}</code>', text)
        self.assertIn('--learning-rate "${LEARNING_RATE}"', text)

    def test_phase41_crosscam_launcher_allows_pure_pair_accept_training_overrides(self) -> None:
        text = self.read_script("runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh")

        self.assertIn('TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-1}"', text)
        self.assertIn('FREEZE_EXTRACTOR_WARMUP_STEPS="${PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS:-0}"', text)
        self.assertIn('TRAIN_DESCRIPTOR_FLAGS=()', text)
        self.assertIn('TRAIN_DESCRIPTOR_FLAGS+=(--no-train-descriptor-head)', text)
        self.assertIn('TEACHER_WEIGHT="${PFM_PHASE41_TEACHER_WEIGHT:-1.0}"', text)
        self.assertIn('SYNTHETIC_LOSS_WEIGHT="${PFM_PHASE41_SYNTHETIC_LOSS_WEIGHT:-1.0}"', text)
        self.assertIn('HARD_NEGATIVE_WEIGHT="${PFM_PHASE41_HARD_NEGATIVE_WEIGHT:-0.6}"', text)
        self.assertIn('WARP_HARD_NEGATIVE_WEIGHT="${PFM_PHASE41_WARP_HARD_NEGATIVE_WEIGHT:-0.15}"', text)
        self.assertIn('SELECTED_KEYPOINT_OFFSET_WEIGHT="${PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT:-0.03}"', text)
        self.assertIn('GRAPH_MATCHER_ACCEPT_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT:-0.12}"', text)
        self.assertIn('GRAPH_MATCHER_PRUNE_RANKING_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_PRUNE_RANKING_WEIGHT:-0.06}"', text)
        self.assertIn('GRAPH_MATCHER_STOP_CONFIDENCE_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_STOP_CONFIDENCE_WEIGHT:-0.03}"', text)
        self.assertIn(
            'GRAPH_MATCHER_POSITIVE_DUSTBIN_MARGIN_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_POSITIVE_DUSTBIN_MARGIN_WEIGHT:-0.006}"',
            text,
        )
        self.assertIn('GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT:-0.08}"', text)
        self.assertIn('--teacher-weight "${TEACHER_WEIGHT}"', text)
        self.assertIn('--synthetic-loss-weight "${SYNTHETIC_LOSS_WEIGHT}"', text)
        self.assertIn('--hard-negative-weight "${HARD_NEGATIVE_WEIGHT}"', text)
        self.assertIn('--warp-hard-negative-weight "${WARP_HARD_NEGATIVE_WEIGHT}"', text)
        self.assertIn('--selected-keypoint-offset-weight "${SELECTED_KEYPOINT_OFFSET_WEIGHT}"', text)
        self.assertIn('--graph-matcher-accept-weight "${GRAPH_MATCHER_ACCEPT_WEIGHT}"', text)
        self.assertIn('--graph-matcher-prune-ranking-weight "${GRAPH_MATCHER_PRUNE_RANKING_WEIGHT}"', text)
        self.assertIn('--graph-matcher-stop-confidence-weight "${GRAPH_MATCHER_STOP_CONFIDENCE_WEIGHT}"', text)
        self.assertIn('--graph-matcher-positive-dustbin-margin-weight "${GRAPH_MATCHER_POSITIVE_DUSTBIN_MARGIN_WEIGHT}"', text)
        self.assertIn('--graph-matcher-true-match-margin-weight "${GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT}"', text)
        self.assertIn('--freeze-extractor-warmup-steps "${FREEZE_EXTRACTOR_WARMUP_STEPS}"', text)
        self.assertIn('"${TRAIN_DESCRIPTOR_FLAGS[@]}"', text)

    def test_phase41_crosscam_launcher_allows_keypoint_offset_head_only_override(self) -> None:
        text = self.read_script("runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh")

        self.assertIn('if [[ "${PFM_PHASE41_TRAIN_KEYPOINT_OFFSET_HEAD_ONLY:-0}" == "1" ]]; then', text)
        self.assertIn('TRAIN_EXTRA_FLAGS+=(--train-keypoint-offset-head-only)', text)
        self.assertIn('train_keypoint_offset_head_only=<code>${PFM_PHASE41_TRAIN_KEYPOINT_OFFSET_HEAD_ONLY:-0}</code>', text)
        self.assertIn('"${TRAIN_EXTRA_FLAGS[@]}"', text)

    def test_phase41_crosscam_launcher_allows_graph_calibration_only_override(self) -> None:
        text = self.read_script("runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh")

        self.assertIn('if [[ "${PFM_PHASE41_TRAIN_GRAPH_CALIBRATION_ONLY:-0}" == "1" ]]; then', text)
        self.assertIn('TRAIN_EXTRA_FLAGS+=(--train-graph-calibration-only)', text)
        self.assertIn('train_graph_calibration_only=<code>${PFM_PHASE41_TRAIN_GRAPH_CALIBRATION_ONLY:-0}</code>', text)
        self.assertIn('"${TRAIN_EXTRA_FLAGS[@]}"', text)

    def test_phase45_true_geometry_visual_eval_can_run_train_without_lightglue(self) -> None:
        text = self.read_script("runs/phase45_true_geometry_visual_eval_20260621.sh")

        self.assertIn('SPLITS_TEXT="${PFM_PHASE45_SPLITS:-dev val lockbox}"', text)
        self.assertIn('read -r -a SPLITS <<< "${SPLITS_TEXT}"', text)
        self.assertIn('REQUIRE_LIGHTGLUE="${PFM_PHASE45_REQUIRE_LIGHTGLUE:-1}"', text)
        self.assertIn('if [[ "${REQUIRE_LIGHTGLUE}" == "1" && ! -f "${SOURCE_ROOT}/eval/${split}/lightglue/lightglue_sift_metrics.csv" ]]; then', text)
        self.assertIn('"${REQUIRE_LIGHTGLUE}"', text)
        self.assertIn("if not require_lightglue:", text)

    def test_phase45_true_geometry_visual_eval_skips_unfiltered_all_match_details(self) -> None:
        text = self.read_script("runs/phase45_true_geometry_visual_eval_20260621.sh")

        self.assertIn("--write-match-details", text)
        self.assertIn("--no-write-all-match-details", text)

    def test_phase57_true_geometry_supervised_smoke_uses_generated_train_manifest(self) -> None:
        text = self.read_script("runs/phase57_true_geometry_supervised_smoke_20260621.sh")

        self.assertIn("phase57_true_geometry_supervision_prepare_20260621", text)
        self.assertIn("true_geometry_supervision_train.csv", text)
        self.assertIn('export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-80}"', text)
        self.assertIn('export PFM_PHASE41_SAVE_EVERY="${PFM_PHASE41_SAVE_EVERY:-40}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_TRAIN_MATCHER_CANDIDATE_TOPK:-512}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_TRAIN_GRAPH_MATCHER_CANDIDATE_TOPK:-512}"', text)
        self.assertIn('export PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT="${PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT:-0.20}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_WEIGHT:-0.08}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.15}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase58_pair_accept_head_only_smoke_uses_reject_weighted_true_geometry_manifest(self) -> None:
        text = self.read_script("runs/phase58_pair_accept_head_only_smoke_20260621.sh")

        self.assertIn("phase58_pair_accept_head_only_smoke_20260621", text)
        self.assertIn("phase58_pair_accept_head_only_smoke_20260621", text)
        self.assertIn("true_geometry_supervision_train_reject4.csv", text)
        self.assertIn("phase42_crosscam_extreme_gap_mixed_formal_20260621", text)
        self.assertIn('export PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY="${PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY:-1}"', text)
        self.assertIn('export PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT="${PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT="${PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_GEOMETRY_OVERLAP_GATE_THRESHOLD="${PFM_PHASE41_GEOMETRY_OVERLAP_GATE_THRESHOLD:-0.10}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase76_false_match_mining_wrapper_keeps_raw_match_details(self) -> None:
        text = self.read_script("runs/phase76_phase73_train_raw_false_mining_20260621.sh")

        self.assertIn("phase73_phase72_true_geometry_large_train_eval_20260621", text)
        self.assertIn("phase72_true_geometry_supervision_prepare_20260621/source/eval/train_pairs.csv", text)
        self.assertIn("--write-all-summary --write-match-details", text)
        self.assertNotIn("--no-write-all-match-details", text)
        self.assertIn("scripts/build_lazy_false_match_csv.py", text)
        self.assertIn('--match-details "${MATCH_DETAILS}"', text)
        self.assertIn('--output-csv "${FALSE_MATCH_CSV}"', text)
        self.assertIn('--min-error-px "${MIN_ERROR_PX}"', text)
        self.assertIn('--max-per-pair "${MAX_FALSE_PER_PAIR}"', text)

    def test_phase77_hard_false_edge_launcher_uses_phase76_static_false_csv(self) -> None:
        text = self.read_script("runs/phase77_phase76_hard_false_edge_train_eval_20260621.sh")

        self.assertIn("PHASE73_ROOT=", text)
        self.assertIn("phase73_phase72_true_geometry_large_train_eval_20260621", text)
        self.assertIn('export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE73_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"', text)
        self.assertIn("phase76_phase73_train_raw_false_mining_20260621", text)
        self.assertIn("hard_false_edges/phase76_phase73_train_raw_false_matches.csv", text)
        self.assertIn('export PFM_PHASE41_FALSE_MATCH_CSV="${PFM_PHASE41_FALSE_MATCH_CSV:-${PHASE76_FALSE_MATCH_CSV}}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.035}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.12}"', text)
        self.assertIn('export PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT="${PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT:-0.008}"', text)
        self.assertIn('export PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT="${PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT:-0.004}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase78_matcher_only_false_edge_launcher_freezes_extractor(self) -> None:
        text = self.read_script("runs/phase78_matcher_only_hard_false_edge_train_eval_20260621.sh")

        self.assertIn("phase73_phase72_true_geometry_large_train_eval_20260621", text)
        self.assertIn("phase76_phase73_train_raw_false_mining_20260621", text)
        self.assertIn('export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-80}"', text)
        self.assertIn('export PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS="${PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS:-80}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-0}"', text)
        self.assertIn('export PFM_PHASE41_TEACHER_WEIGHT="${PFM_PHASE41_TEACHER_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_SYNTHETIC_LOSS_WEIGHT="${PFM_PHASE41_SYNTHETIC_LOSS_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.015}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.06}"', text)
        self.assertIn('export PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT="${PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT="${PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_WEIGHT:-0.0}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase92_residual_false_edge_launcher_uses_raw_and_filtered_false_csvs(self) -> None:
        text = self.read_script("runs/phase92_residual_filtered_false_edge_train_eval_20260621.sh")

        self.assertIn("phase76_phase73_train_raw_false_mining_20260621", text)
        self.assertIn("phase91_geometry_edge_supervision_dataset_20260621", text)
        self.assertIn("phase76_phase73_train_raw_false_matches.csv", text)
        self.assertIn("phase76_train_filtered_residual_false_matches.csv", text)
        self.assertIn('${PHASE76_FALSE_MATCH_CSV}:${PHASE91_RESIDUAL_FALSE_MATCH_CSV}', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.028}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.08}"', text)
        self.assertIn('export PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS="${PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS:-100}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase93_conservative_residual_false_edge_launcher_limits_false_pressure(self) -> None:
        text = self.read_script("runs/phase93_conservative_residual_false_edge_train_eval_20260621.sh")

        self.assertIn("phase76_phase73_train_raw_false_mining_20260621", text)
        self.assertIn("phase91_geometry_edge_supervision_dataset_20260621", text)
        self.assertIn("phase76_phase73_train_raw_false_matches.csv", text)
        self.assertIn("phase76_train_filtered_residual_false_matches.csv", text)
        self.assertIn('${PHASE76_FALSE_MATCH_CSV}:${PHASE91_RESIDUAL_FALSE_MATCH_CSV}', text)
        self.assertIn('export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-80}"', text)
        self.assertIn('export PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS="${PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS:-80}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.008}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.025}"', text)
        self.assertIn(
            'export PFM_PHASE41_MINED_FALSE_MATCH_REFERENCE_MARGIN="${PFM_PHASE41_MINED_FALSE_MATCH_REFERENCE_MARGIN:-0.15}"',
            text,
        )
        self.assertIn('export PFM_PHASE41_GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT:-0.24}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase94_low_pressure_residual_false_edge_launcher_keeps_false_edges_active(self) -> None:
        text = self.read_script("runs/phase94_low_pressure_residual_false_edge_train_eval_20260621.sh")

        self.assertIn("phase76_phase73_train_raw_false_mining_20260621", text)
        self.assertIn("phase91_geometry_edge_supervision_dataset_20260621", text)
        self.assertIn("phase76_phase73_train_raw_false_matches.csv", text)
        self.assertIn("phase76_train_filtered_residual_false_matches.csv", text)
        self.assertIn('${PHASE76_FALSE_MATCH_CSV}:${PHASE91_RESIDUAL_FALSE_MATCH_CSV}', text)
        self.assertIn('export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-80}"', text)
        self.assertIn('export PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS="${PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS:-80}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.006}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.020}"', text)
        self.assertIn(
            'export PFM_PHASE41_MINED_FALSE_MATCH_REFERENCE_MARGIN="${PFM_PHASE41_MINED_FALSE_MATCH_REFERENCE_MARGIN:--1.0}"',
            text,
        )
        self.assertIn('export PFM_PHASE41_GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT:-0.18}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase95_failure_bucket_replay_launcher_uses_devval_replay_manifest(self) -> None:
        text = self.read_script("runs/phase95_failure_bucket_replay_soft_boundary_train_eval_20260621.sh")

        self.assertIn("phase94_low_pressure_residual_false_edge_train_eval_20260621", text)
        self.assertIn("failure_bucket_replay_devval_high_precision_thresholds", text)
        self.assertIn("failure_bucket_replay_mixed_train.csv", text)
        self.assertIn('export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE94_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${REPLAY_MIXED_MANIFEST}}"', text)
        self.assertIn('export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-80}"', text)
        self.assertIn('export PFM_PHASE41_FALSE_CLUSTER_REPLAY_MULTIPLIER="${PFM_PHASE41_FALSE_CLUSTER_REPLAY_MULTIPLIER:-1.50}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.004}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.015}"', text)
        self.assertIn(
            'export PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT:-0.04}"',
            text,
        )
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase96_offset_boundary_calibration_reuses_phase94_without_replay(self) -> None:
        text = self.read_script("runs/phase96_offset_boundary_calibration_train_eval_20260621.sh")

        self.assertIn("phase94_low_pressure_residual_false_edge_train_eval_20260621", text)
        self.assertIn("phase72_true_geometry_supervision_prepare_20260621", text)
        self.assertIn('export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE94_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${PHASE72_PREP_ROOT}/true_geometry_supervision_train.csv}"', text)
        self.assertNotIn("failure_bucket_replay_mixed_train.csv", text)
        self.assertIn('export PFM_PHASE41_TRAIN_KEYPOINT_OFFSET_HEAD_ONLY="${PFM_PHASE41_TRAIN_KEYPOINT_OFFSET_HEAD_ONLY:-1}"', text)
        self.assertIn('export PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT="${PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT:-0.08}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.003}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.012}"', text)
        self.assertIn(
            'export PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT:-0.03}"',
            text,
        )
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase98_soft_boundary_geometry_launcher_uses_true_geometry_without_replay(self) -> None:
        text = self.read_script("runs/phase98_soft_boundary_geometry_train_eval_20260621.sh")

        self.assertIn("phase94_low_pressure_residual_false_edge_train_eval_20260621", text)
        self.assertIn("phase72_true_geometry_supervision_prepare_20260621", text)
        self.assertIn(
            'export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE94_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"',
            text,
        )
        self.assertIn(
            'export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${PHASE72_PREP_ROOT}/true_geometry_supervision_train.csv}"',
            text,
        )
        self.assertNotIn("failure_bucket_replay_mixed_train.csv", text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_RESIDUAL_THRESHOLD_PX="${PFM_PHASE41_WARP_OUTLIER_RESIDUAL_THRESHOLD_PX:-8.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_RESIDUAL_THRESHOLD_PX="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_RESIDUAL_THRESHOLD_PX:-8.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT="${PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT:-0.045}"', text)
        self.assertIn('export PFM_PHASE41_WARP_SOFT_BOUNDARY_LOWER_RESIDUAL_PX="${PFM_PHASE41_WARP_SOFT_BOUNDARY_LOWER_RESIDUAL_PX:-5.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_SOFT_BOUNDARY_UPPER_RESIDUAL_PX="${PFM_PHASE41_WARP_SOFT_BOUNDARY_UPPER_RESIDUAL_PX:-8.0}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase99_match_detail_filter_calibration_reuses_phase94_without_retraining(self) -> None:
        text = self.read_script("runs/phase99_phase94_match_detail_filter_calibration_20260621.sh")

        self.assertIn("phase94_low_pressure_residual_false_edge_train_eval_20260621", text)
        self.assertIn("scripts/train_match_detail_filter_calibrator.py", text)
        self.assertIn('--train-match-details "${DEV_DETAILS}"', text)
        self.assertIn('--eval-match-details "${VAL_DETAILS}"', text)
        self.assertIn("--threshold-objective pfm_wrong_cap", text)
        self.assertIn("--threshold-selection-source eval", text)
        self.assertIn('--max-kept-wrong "${VAL_LIGHTGLUE_WRONG}"', text)
        self.assertIn("scripts/apply_match_detail_filter_calibrator.py", text)
        self.assertIn("scripts/sweep_match_filter_thresholds.py", text)
        self.assertIn(
            '--select-source "val,${APPLY_VAL}/match_predictions.csv,${VAL_LIGHTGLUE_CORRECT},${VAL_LIGHTGLUE_WRONG}"',
            text,
        )
        self.assertIn(
            '--validation-source "lockbox,${APPLY_LOCKBOX}/match_predictions.csv,${LOCKBOX_LIGHTGLUE_CORRECT},${LOCKBOX_LIGHTGLUE_WRONG}"',
            text,
        )
        self.assertIn('"model_threshold"', text)
        self.assertIn('"per_variant_sweep"', text)
        self.assertIn('"recommended_policy"', text)
        self.assertNotIn("pfm_pytorch_training.py", text)
        self.assertNotIn("phase98_soft_boundary_geometry_train_eval_20260621", text)

    def test_phase100_strict_boundary_hard_negative_reuses_phase94_without_soft_boundary(self) -> None:
        text = self.read_script("runs/phase100_strict_boundary_hard_negative_train_eval_20260621.sh")

        self.assertIn("phase94_low_pressure_residual_false_edge_train_eval_20260621", text)
        self.assertIn("phase72_true_geometry_supervision_prepare_20260621", text)
        self.assertIn(
            'export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE94_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"',
            text,
        )
        self.assertIn(
            'export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${PHASE72_PREP_ROOT}/true_geometry_supervision_train.csv}"',
            text,
        )
        self.assertIn("phase76_phase73_train_raw_false_matches.csv", text)
        self.assertIn("phase76_train_filtered_residual_false_matches.csv", text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_RESIDUAL_THRESHOLD_PX="${PFM_PHASE41_WARP_OUTLIER_RESIDUAL_THRESHOLD_PX:-5.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_RESIDUAL_THRESHOLD_PX="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_RESIDUAL_THRESHOLD_PX:-5.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT="${PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT:-0.0}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("failure_bucket_replay_mixed_train.csv", text)
        self.assertNotIn("phase98_soft_boundary_geometry_train_eval_20260621", text)

    def test_phase101_pair_gate_uses_true_geometry_labels_with_lightglue_eval_only(self) -> None:
        text = self.read_script("runs/phase101_phase94_pair_gate_diagnostic_20260621.sh")

        self.assertIn("phase94_low_pressure_residual_false_edge_train_eval_20260621", text)
        self.assertIn("scripts/build_match_set_rejection_dataset.py", text)
        self.assertIn("scripts/train_match_set_rejection_calibrator.py", text)
        self.assertIn("scripts/apply_match_set_rejection_calibrator.py", text)
        self.assertIn('--source "dev,${EVAL_ROOT}/dev_pairs.csv,${DEV_SUMMARY},${DEV_LIGHTGLUE},${DEV_DETAILS}"', text)
        self.assertIn("--teacher-wrong-excess-threshold 999999", text)
        self.assertIn("--teacher-precision-advantage-threshold 999999", text)
        self.assertIn("--threshold-selection-source eval", text)
        self.assertIn("--threshold-objective pfm_wrong_cap", text)
        self.assertIn('--max-kept-pfm-wrong "${VAL_LIGHTGLUE_WRONG}"', text)
        self.assertIn("--reject-action zero", text)
        self.assertIn("--reject-action lightglue", text)
        self.assertNotIn("pfm_pytorch_training.py", text)
        self.assertNotIn("phase98_soft_boundary_geometry_train_eval_20260621", text)

    def test_phase103_high_recall_accept_head_launcher_uses_phase94_and_kp6144_eval(self) -> None:
        text = self.read_script("runs/phase103_high_recall_accept_head_train_eval_20260621.sh")

        self.assertIn("phase94_low_pressure_residual_false_edge_train_eval_20260621", text)
        self.assertIn("phase72_true_geometry_supervision_prepare_20260621", text)
        self.assertIn(
            'export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE94_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"',
            text,
        )
        self.assertIn('export PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-0}"', text)
        self.assertIn('export PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS="${PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS:-100}"', text)
        self.assertIn('export PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT:-0.09}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.10}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.004}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_MAX_KEYPOINTS="${PFM_PHASE41_EVAL_MAX_KEYPOINTS:-6144}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_KEYPOINT_SPATIAL_BINS="${PFM_PHASE41_EVAL_KEYPOINT_SPATIAL_BINS:-16}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_KEYPOINT_CELL_CAP="${PFM_PHASE41_EVAL_KEYPOINT_CELL_CAP:-12}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK:-512}"', text)
        self.assertIn(
            'export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512}"',
            text,
        )
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("phase98_soft_boundary_geometry_train_eval_20260621", text)

    def test_phase104_high_recall_false_replay_launcher_uses_phase103_devval_false_edges(self) -> None:
        text = self.read_script("runs/phase104_phase103_high_recall_false_replay_train_eval_20260621.sh")

        self.assertIn("phase103_high_recall_accept_head_train_eval_20260621", text)
        self.assertIn("phase104_phase103_high_recall_false_edges_20260621", text)
        self.assertIn("dev_phase103_kp6144_filtered_false_matches.csv", text)
        self.assertIn("val_phase103_kp6144_filtered_false_matches.csv", text)
        self.assertIn("phase103_devval_false_pair_replay_train.csv", text)
        self.assertIn("csv.DictReader", text)
        self.assertIn(
            'export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE103_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"',
            text,
        )
        self.assertIn('export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${REPLAY_MANIFEST}}"', text)
        self.assertIn('export PFM_PHASE41_FALSE_MATCH_CSV="${PFM_PHASE41_FALSE_MATCH_CSV:-${DEV_FALSE_CSV}:${VAL_FALSE_CSV}}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-0}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.020}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.060}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_MAX_KEYPOINTS="${PFM_PHASE41_EVAL_MAX_KEYPOINTS:-6144}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK:-512}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("phase98_soft_boundary_geometry_train_eval_20260621", text)

    def test_phase106_final_accept_fusion_launcher_uses_phase103_and_multiply_training(self) -> None:
        text = self.read_script("runs/phase106_phase103_final_accept_fusion_train_eval_20260621.sh")

        self.assertIn("phase103_high_recall_accept_head_train_eval_20260621", text)
        self.assertIn(
            'export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE103_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"',
            text,
        )
        self.assertIn('export PFM_PHASE41_TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE41_TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE:-multiply}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE:-multiply}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-0}"', text)
        self.assertIn('export PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS="${PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS:-100}"', text)
        self.assertIn('export PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT:-0.12}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.16}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK:-512}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("phase104_phase103_high_recall_false_replay_train_eval_20260621", text)

    def test_phase108_true_geometry_false_edge_launcher_mixes_phase106_hard_pairs_with_train(self) -> None:
        text = self.read_script("runs/phase108_phase106_true_geometry_false_edge_train_eval_20260621.sh")

        self.assertIn("phase106_phase103_final_accept_fusion_train_eval_20260621", text)
        self.assertIn("phase108_phase106_final_accept_false_edges_20260621", text)
        self.assertIn("dev_phase106_filtered_false_matches.csv", text)
        self.assertIn("val_phase106_filtered_false_matches.csv", text)
        self.assertIn("phase106_devval_false_edge_mixed_train.csv", text)
        self.assertIn("false_cluster_reasons", text)
        self.assertIn("false_cluster_wrong_sum", text)
        self.assertIn("target_hard_fraction", text)
        self.assertIn(
            'export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE106_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"',
            text,
        )
        self.assertIn('export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${MIXED_MANIFEST}}"', text)
        self.assertIn('export PFM_PHASE41_FALSE_MATCH_CSV="${PFM_PHASE41_FALSE_MATCH_CSV:-${DEV_FALSE_CSV}:${VAL_FALSE_CSV}}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE41_TRAIN_MATCHER_FINAL_ACCEPT_SCORE_MODE:-multiply}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE="${PFM_PHASE41_EVAL_MATCHER_FINAL_ACCEPT_SCORE_MODE:-multiply}"', text)
        self.assertIn('export PFM_PHASE41_FALSE_CLUSTER_REPLAY_MULTIPLIER="${PFM_PHASE41_FALSE_CLUSTER_REPLAY_MULTIPLIER:-1.50}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.008}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.024}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.16}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK="${PFM_PHASE41_EVAL_MATCHER_CANDIDATE_TOPK:-512}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("phase104_phase103_high_recall_false_replay_train_eval_20260621", text)

    def test_phase109_conservative_false_replay_preserves_phase106_train_false_edges(self) -> None:
        text = self.read_script("runs/phase109_phase106_conservative_false_replay_train_eval_20260621.sh")

        self.assertIn("phase108_phase106_true_geometry_false_edge_train_eval_20260621.sh", text)
        self.assertIn("phase106_phase103_final_accept_fusion_train_eval_20260621", text)
        self.assertIn("phase76_phase73_train_raw_false_mining_20260621", text)
        self.assertIn("phase91_geometry_edge_supervision_dataset_20260621", text)
        self.assertIn("phase76_phase73_train_raw_false_matches.csv", text)
        self.assertIn("phase76_train_filtered_residual_false_matches.csv", text)
        self.assertIn("dev_phase106_filtered_false_matches.csv", text)
        self.assertIn("val_phase106_filtered_false_matches.csv", text)
        self.assertIn(
            'export PFM_PHASE41_FALSE_MATCH_CSV="${PFM_PHASE41_FALSE_MATCH_CSV:-${PHASE76_FALSE_MATCH_CSV}:${PHASE91_RESIDUAL_FALSE_MATCH_CSV}:${DEV_FALSE_CSV}:${VAL_FALSE_CSV}}"',
            text,
        )
        self.assertIn('export PFM_PHASE108_TARGET_HARD_FRACTION="${PFM_PHASE108_TARGET_HARD_FRACTION:-0.08}"', text)
        self.assertIn('export PFM_PHASE41_FALSE_CLUSTER_REPLAY_MULTIPLIER="${PFM_PHASE41_FALSE_CLUSTER_REPLAY_MULTIPLIER:-1.20}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.006}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.020}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_ROOT="${PFM_PHASE41_TRAIN_ROOT:-${PFM_PHASE108_RUN_ROOT}}"', text)
        self.assertIn('exec bash runs/phase108_phase106_true_geometry_false_edge_train_eval_20260621.sh', text)
        self.assertNotIn("phase104_phase103_high_recall_false_replay_train_eval_20260621", text)

    def test_phase111_geometry_pair_accept_head_trains_rejection_probe_without_lightglue_labels(self) -> None:
        text = self.read_script("runs/phase111_phase106_geometry_pair_accept_head_train_eval_20260621.sh")

        self.assertIn("phase106_phase103_final_accept_fusion_train_eval_20260621", text)
        self.assertIn("phase72_true_geometry_supervision_prepare_20260621", text)
        self.assertIn("scripts/build_geometry_acceptance_training_manifest.py", text)
        self.assertIn("geometry_pair_accept_train.csv", text)
        self.assertIn('--reject-below-valid-fraction "${REJECT_BELOW_VALID_FRACTION}"', text)
        self.assertIn('--accept-at-valid-fraction "${ACCEPT_AT_VALID_FRACTION}"', text)
        self.assertIn('REJECT_BELOW_VALID_FRACTION="${PFM_PHASE111_REJECT_BELOW_VALID_FRACTION:-0.20}"', text)
        self.assertIn('ACCEPT_AT_VALID_FRACTION="${PFM_PHASE111_ACCEPT_AT_VALID_FRACTION:-0.20}"', text)
        self.assertIn(
            'export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE106_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"',
            text,
        )
        self.assertIn('export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${GEOMETRY_ACCEPT_MANIFEST}}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY="${PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY:-1}"', text)
        self.assertIn('export PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT="${PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT:-1.0}"', text)
        self.assertIn('export PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT="${PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PAIR_ACCEPT_MIN_PROBABILITY="${PFM_PAIR_ACCEPT_MIN_PROBABILITY:--1.0}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_pairaccept_probe}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("lightglue", text.split("PFM_PHASE41_NOTE", 1)[0].lower())

    def test_phase115_true_geometry_raw_filter_replays_phase113_without_lightglue_labels(self) -> None:
        text = self.read_script("runs/phase115_phase113_true_geometry_raw_filter_eval_20260621.sh")

        self.assertIn("phase113_phase106_geometry_pair_accept_benchmark_bnfix_probe_20260621", text)
        self.assertIn("scripts/apply_true_geometry_match_filter.py", text)
        self.assertIn("all_match_details.csv", text)
        self.assertIn("lightglue_sift_metrics.csv", text)
        self.assertIn('VALID_FRACTIONS="${PFM_PHASE115_VALID_FRACTIONS:-0 0.05 0.10 0.15 0.20 0.25 0.30}"', text)
        self.assertIn("--max-error-px \"${MAX_ERROR_PX}\"", text)
        self.assertIn("--min-valid-fraction \"${VF}\"", text)
        self.assertIn("true_geometry_raw_filter_summary.html", text)
        self.assertNotIn("--include-true-geometry-features", text)

    def test_phase116_true_geometry_mlp_filter_trains_from_raw_matches_without_lightglue_labels(self) -> None:
        text = self.read_script("runs/phase116_phase113_true_geometry_mlp_filter_train_eval_20260621.sh")

        self.assertIn("phase113_phase106_geometry_pair_accept_benchmark_bnfix_probe_20260621", text)
        self.assertIn("scripts/train_match_detail_mlp_filter_calibrator.py", text)
        self.assertIn("all_match_details.csv", text)
        self.assertIn("--include-true-geometry-features", text)
        self.assertIn("--threshold-objective pfm_wrong_cap", text)
        self.assertIn("--max-kept-wrong \"${MAX_KEPT_WRONG}\"", text)
        self.assertIn("lightglue_sift_metrics.csv", text)
        self.assertIn("phase116_true_geometry_mlp_summary.html", text)
        self.assertNotIn("distill", text.lower())

    def test_phase117_true_geometry_mlp_apply_profile_reuses_frozen_phase116_model(self) -> None:
        text = self.read_script("runs/phase117_phase116_true_geometry_mlp_apply_profile_20260621.sh")

        self.assertIn("phase113_phase106_geometry_pair_accept_benchmark_bnfix_probe_20260621", text)
        self.assertIn("phase116_phase113_true_geometry_mlp_filter_train_eval_20260621", text)
        self.assertIn("mlp_true_geometry_filter/model.json", text)
        self.assertIn("scripts/apply_match_detail_filter_calibrator.py", text)
        self.assertIn('MODEL_JSON="${PFM_PHASE117_MODEL_JSON:-${PHASE116_ROOT}/mlp_true_geometry_filter/model.json}"', text)
        self.assertIn('for SPLIT in dev val lockbox; do', text)
        self.assertIn('--match-details "${MATCH_DETAILS}"', text)
        self.assertIn('--model-json "${MODEL_JSON}"', text)
        self.assertIn('--output-dir "${APPLY_DIR}"', text)
        self.assertIn("LightGlue-SIFT-MAGSAC-min16", text)
        self.assertIn("phase117_true_geometry_mlp_apply_profile_summary.json", text)
        self.assertIn("phase117_true_geometry_mlp_apply_profile_summary.html", text)
        self.assertIn('"include_true_geometry_features"', text)
        self.assertNotIn("train_match_detail_mlp_filter_calibrator.py", text)
        self.assertNotIn("scripts/pfm_pytorch_training.py", text)
        self.assertNotIn("distill", text.lower())

    def test_phase118_raw_false_edge_launcher_uses_phase113_raw_geometry_without_lightglue_labels(self) -> None:
        text = self.read_script("runs/phase118_phase113_raw_false_accept_head_train_eval_20260621.sh")

        self.assertIn("phase113_phase106_geometry_pair_accept_benchmark_bnfix_probe_20260621", text)
        self.assertIn("phase106_phase103_final_accept_fusion_train_eval_20260621", text)
        self.assertIn("phase72_true_geometry_supervision_prepare_20260621", text)
        self.assertIn("scripts/build_lazy_false_match_csv.py", text)
        self.assertIn("all_match_details.csv", text)
        self.assertIn("dev_phase113_raw_false_matches.csv", text)
        self.assertIn("val_phase113_raw_false_matches.csv", text)
        self.assertIn("phase113_dev_raw_true_geometry_wrong_high_accept", text)
        self.assertIn("phase113_val_raw_true_geometry_wrong_high_accept", text)
        self.assertIn('MIN_FALSE_SCORE="${PFM_PHASE118_MIN_FALSE_SCORE:-16.0}"', text)
        self.assertIn('MIN_ACCEPT_PROBABILITY="${PFM_PHASE118_MIN_ACCEPT_PROBABILITY:-0.68}"', text)
        self.assertIn('TARGET_HARD_FRACTION="${PFM_PHASE118_TARGET_HARD_FRACTION:-0.25}"', text)
        self.assertIn('"phase118_manifest_origin"', text)
        self.assertIn('"phase113_raw_true_geometry_false_edge"', text)
        self.assertIn('export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE106_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${MIXED_MANIFEST}}"', text)
        self.assertIn('export PFM_PHASE41_FALSE_MATCH_CSV="${PFM_PHASE41_FALSE_MATCH_CSV:-${DEV_FALSE_CSV}:${VAL_FALSE_CSV}}"', text)
        self.assertIn('export PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT:-0.20}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.30}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.018}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-0}"', text)
        self.assertIn('export PFM_PAIR_ACCEPT_MIN_PROBABILITY="${PFM_PAIR_ACCEPT_MIN_PROBABILITY:--1.0}"', text)
        self.assertIn('if [[ "${PFM_PHASE118_PREP_ONLY:-0}" == "1" ]]; then', text)
        self.assertIn("phase118_prep_only_complete", text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("train_match_detail_mlp_filter_calibrator.py", text)
        self.assertNotIn("lightglue", text.split("PFM_PHASE41_NOTE", 1)[0].lower())
        self.assertNotIn("distill", text.lower())

    def test_phase119_boundary_false_replay_launcher_focuses_extreme_true_geometry_errors(self) -> None:
        text = self.read_script("runs/phase119_phase118_boundary_false_replay_train_eval_20260621.sh")

        self.assertIn("phase118_phase113_raw_false_accept_head_train_eval_20260621", text)
        self.assertIn("phase72_true_geometry_supervision_prepare_20260621", text)
        self.assertIn("scripts/build_lazy_false_match_csv.py", text)
        self.assertIn("all_filtered_match_details.csv", text)
        self.assertIn("dev_phase118_boundary_false_matches.csv", text)
        self.assertIn("val_phase118_boundary_false_matches.csv", text)
        self.assertIn("--target-variant extreme_01", text)
        self.assertIn("--target-variant extreme_03", text)
        self.assertIn('MIN_ERROR_PX="${PFM_PHASE119_MIN_ERROR_PX:-5.0}"', text)
        self.assertIn('MAX_ERROR_PX="${PFM_PHASE119_MAX_ERROR_PX:-8.0}"', text)
        self.assertIn('MIN_FALSE_SCORE="${PFM_PHASE119_MIN_FALSE_SCORE:-14.0}"', text)
        self.assertIn('TARGET_HARD_FRACTION="${PFM_PHASE119_TARGET_HARD_FRACTION:-0.30}"', text)
        self.assertIn('"phase119_manifest_origin"', text)
        self.assertIn('"phase118_boundary_true_geometry_false_edge"', text)
        self.assertIn('export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE118_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"', text)
        self.assertIn('export PFM_PHASE41_FALSE_MATCH_CSV="${PFM_PHASE41_FALSE_MATCH_CSV:-${DEV_FALSE_CSV}:${VAL_FALSE_CSV}}"', text)
        self.assertIn('export PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT="${PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT:-0.035}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.34}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.024}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-0}"', text)
        self.assertIn('if [[ "${PFM_PHASE119_PREP_ONLY:-0}" == "1" ]]; then', text)
        self.assertIn("phase119_prep_only_complete", text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("train_match_detail_mlp_filter_calibrator.py", text)
        self.assertNotIn("distill", text.lower())

    def test_phase120_extreme03_calibration_replay_launcher_is_conservative(self) -> None:
        text = self.read_script("runs/phase120_phase118_extreme03_calibration_replay_train_eval_20260621.sh")

        self.assertIn("phase118_phase113_raw_false_accept_head_train_eval_20260621", text)
        self.assertIn("phase72_true_geometry_supervision_prepare_20260621", text)
        self.assertIn("scripts/build_lazy_false_match_csv.py", text)
        self.assertIn("all_filtered_match_details.csv", text)
        self.assertIn("dev_phase118_extreme03_boundary_false_matches.csv", text)
        self.assertIn("val_phase118_extreme03_boundary_false_matches.csv", text)
        self.assertIn("--target-variant extreme_03", text)
        self.assertNotIn("--target-variant extreme_01", text)
        self.assertIn('MIN_ERROR_PX="${PFM_PHASE120_MIN_ERROR_PX:-5.0}"', text)
        self.assertIn('MAX_ERROR_PX="${PFM_PHASE120_MAX_ERROR_PX:-8.0}"', text)
        self.assertIn('MIN_FALSE_SCORE="${PFM_PHASE120_MIN_FALSE_SCORE:-16.0}"', text)
        self.assertIn('TARGET_HARD_FRACTION="${PFM_PHASE120_TARGET_HARD_FRACTION:-0.12}"', text)
        self.assertIn('"phase120_manifest_origin"', text)
        self.assertIn('"phase118_extreme03_boundary_true_geometry_false_edge"', text)
        self.assertIn('export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE118_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_GRAPH_CALIBRATION_ONLY="${PFM_PHASE41_TRAIN_GRAPH_CALIBRATION_ONLY:-1}"', text)
        self.assertIn('export PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT="${PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.12}"', text)
        self.assertIn('export PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT="${PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT:-0.004}"', text)
        self.assertIn('export PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT="${PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-60}"', text)
        self.assertIn('if [[ "${PFM_PHASE120_PREP_ONLY:-0}" == "1" ]]; then', text)
        self.assertIn("phase120_prep_only_complete", text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("train_match_detail_mlp_filter_calibrator.py", text)
        self.assertNotIn("distill", text.lower())

    def test_phase121_phase118_offset_head_launcher_trains_only_localization(self) -> None:
        text = self.read_script("runs/phase121_phase118_offset_head_localization_train_eval_20260621.sh")

        self.assertIn("phase118_phase113_raw_false_accept_head_train_eval_20260621", text)
        self.assertIn("phase72_true_geometry_supervision_prepare_20260621", text)
        self.assertIn('export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE118_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${PHASE72_PREP_ROOT}/true_geometry_supervision_train.csv}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_KEYPOINT_OFFSET_HEAD_ONLY="${PFM_PHASE41_TRAIN_KEYPOINT_OFFSET_HEAD_ONLY:-1}"', text)
        self.assertIn('export PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT="${PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT:-0.08}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-0}"', text)
        self.assertIn('export PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_ACCEPT_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT="${PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT="${PFM_PHASE41_RAW_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_MAX_KEYPOINTS="${PFM_PHASE41_EVAL_MAX_KEYPOINTS:-6144}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase121_offset}"', text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("build_lazy_false_match_csv.py", text)
        self.assertNotIn("distill", text.lower())

    def test_phase123_phase122_targeted_false_replay_uses_true_geometry_only(self) -> None:
        text = self.read_script("runs/phase123_phase122_targeted_false_replay_train_eval_20260621.sh")

        self.assertIn("phase122_phase118_low_weight_offset_head_train_eval_20260621", text)
        self.assertIn("pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase122_offset003", text)
        self.assertIn("scripts/build_lazy_false_match_csv.py", text)
        self.assertIn("all_filtered_match_details.csv", text)
        self.assertIn("dev_phase122_targeted_false_matches.csv", text)
        self.assertIn("val_phase122_targeted_false_matches.csv", text)
        self.assertIn("--target-variant extreme_01", text)
        self.assertIn("--target-variant extreme_02", text)
        self.assertIn("--target-variant extreme_03", text)
        self.assertIn('MAX_FALSE_PER_PAIR="${PFM_PHASE123_MAX_FALSE_PER_PAIR:-8}"', text)
        self.assertIn('TARGET_HARD_FRACTION="${PFM_PHASE123_TARGET_HARD_FRACTION:-0.05}"', text)
        self.assertIn('"phase123_manifest_origin"', text)
        self.assertIn('"phase122_targeted_true_geometry_false_edge"', text)
        self.assertIn('export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE122_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_GRAPH_CALIBRATION_ONLY="${PFM_PHASE41_TRAIN_GRAPH_CALIBRATION_ONLY:-1}"', text)
        self.assertIn('export PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT="${PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT="${PFM_PHASE41_FINAL_FALSE_MATCH_WEIGHT:-0.002}"', text)
        self.assertIn('export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.04}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-30}"', text)
        self.assertIn('if [[ "${PFM_PHASE123_PREP_ONLY:-0}" == "1" ]]; then', text)
        self.assertIn("phase123_prep_only_complete", text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("lightglue", text.lower())
        self.assertNotIn("distill", text.lower())

    def test_phase125_phase122_match_detail_filter_uses_true_geometry_labels_with_lightglue_eval_only(self) -> None:
        text = self.read_script("runs/phase125_phase122_match_detail_filter_calibration_20260621.sh")

        self.assertIn("phase122_phase118_low_weight_offset_head_train_eval_20260621", text)
        self.assertIn("pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase122_offset003", text)
        self.assertIn("scripts/train_match_detail_filter_calibrator.py", text)
        self.assertIn("scripts/apply_match_detail_filter_calibrator.py", text)
        self.assertIn("scripts/sweep_match_filter_thresholds.py", text)
        self.assertIn('PFM_PHASE125_USE_TRUE_GEOMETRY_FEATURES:-0', text)
        self.assertIn("--threshold-objective pfm_wrong_cap", text)
        self.assertIn("--threshold-selection-source eval", text)
        self.assertIn("--balance-sampling-key target_variant", text)
        self.assertIn("LightGlue is eval baseline only", text)
        self.assertIn("all_filtered_match_details.csv", text)
        self.assertIn("selected_apply_lockbox", text)
        self.assertIn("summary.json", text)
        self.assertNotIn("distill", text.lower())

    def test_phase130_phase122_coordinate_mlp_filter_uses_inference_features_only(self) -> None:
        text = self.read_script("runs/phase130_phase122_coordinate_mlp_match_detail_filter_calibration_20260622.sh")

        self.assertIn("phase122_phase118_low_weight_offset_head_train_eval_20260621", text)
        self.assertIn("pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase122_offset003", text)
        self.assertIn("scripts/train_match_detail_mlp_filter_calibrator.py", text)
        self.assertIn("scripts/apply_match_detail_filter_calibrator.py", text)
        self.assertIn("scripts/sweep_match_filter_thresholds.py", text)
        self.assertIn("feature_point_a_x_norm", text)
        self.assertIn("feature_displacement_magnitude_px", text)
        self.assertIn("PFM_PHASE130_FEATURE_NAME_REGEX", text)
        self.assertIn("--feature-name-regex", text)
        self.assertIn("--threshold-objective pfm_wrong_cap", text)
        self.assertIn("--threshold-selection-source eval", text)
        self.assertIn("--balance-sampling-key target_variant", text)
        self.assertIn("devval_cap17_summary.json", text)
        self.assertIn("LightGlue is eval baseline only", text)
        self.assertIn("all_filtered_match_details.csv", text)
        self.assertIn("phase130_phase122_coordinate_mlp_match_detail_filter_calibration_20260622", text)
        self.assertNotIn("--include-true-geometry-features", text)
        self.assertNotIn("distill", text.lower())

    def test_phase129_phase122_failure_bucket_soft_boundary_replay_is_train_only(self) -> None:
        text = self.read_script("runs/phase129_phase122_failure_bucket_soft_boundary_train_eval_20260622.sh")

        self.assertIn("phase122_phase118_low_weight_offset_head_train_eval_20260621", text)
        self.assertIn("pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase122_offset003", text)
        self.assertIn("scripts/analyze_pfm_failure_buckets.py", text)
        self.assertIn("scripts/build_failure_bucket_replay_manifest.py", text)
        self.assertIn("dev_failure_buckets/pair_failure_summary.csv", text)
        self.assertIn("val_failure_buckets/pair_failure_summary.csv", text)
        self.assertIn("phase122_failure_bucket_replay_mixed_train.csv", text)
        self.assertIn("--mixed-replay-fraction \"${MIXED_REPLAY_FRACTION}\"", text)
        self.assertIn("phase122_dev", text)
        self.assertIn("phase122_val", text)
        self.assertNotIn("lockbox_failure_buckets", text)
        self.assertIn('export PFM_PHASE41_INIT_STATE="${PFM_PHASE41_INIT_STATE:-${PHASE122_ROOT}/train_output/checkpoints/last_good_pytorch_pfm_state.pt}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${MIXED_MANIFEST}}"', text)
        self.assertIn('export PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT="${PFM_PHASE41_WARP_SOFT_BOUNDARY_WEIGHT:-0.025}"', text)
        self.assertIn('export PFM_PHASE41_WARP_SOFT_BOUNDARY_LOWER_RESIDUAL_PX="${PFM_PHASE41_WARP_SOFT_BOUNDARY_LOWER_RESIDUAL_PX:-5.0}"', text)
        self.assertIn('export PFM_PHASE41_WARP_SOFT_BOUNDARY_UPPER_RESIDUAL_PX="${PFM_PHASE41_WARP_SOFT_BOUNDARY_UPPER_RESIDUAL_PX:-8.0}"', text)
        self.assertIn('export PFM_PHASE41_FALSE_CLUSTER_REPLAY_MULTIPLIER="${PFM_PHASE41_FALSE_CLUSTER_REPLAY_MULTIPLIER:-1.20}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase129_failure_bucket_soft}"', text)
        self.assertIn('if [[ "${PFM_PHASE129_PREP_ONLY:-0}" == "1" ]]; then', text)
        self.assertIn("phase129_prep_only_complete", text)
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)
        self.assertNotIn("distill", text.lower())

    def test_phase83_true_geometry_filter_eval_uses_phase79_active_profile(self) -> None:
        text = self.read_script("runs/phase83_phase79_true_geometry_filter_eval_20260621.sh")

        self.assertIn("phase79_phase78_active_profile_eval_20260621", text)
        self.assertIn("phase78_matcher_only_hard_false_edge_train_eval_20260621/eval", text)
        self.assertIn("scripts/apply_true_geometry_match_filter.py", text)
        self.assertIn('--source "dev,${PHASE79}/dev/${SUB}/all_filtered_match_details.csv,${LG}/dev/lightglue/lightglue_sift_metrics.csv"', text)
        self.assertIn("--max-error-px 5.0", text)
        self.assertIn("--min-valid-fraction 0.10", text)
        self.assertIn("summary.html", text)
        self.assertIn("PFM_PHASE83_MANIFEST_VALIDATION_JSON", text)
        self.assertIn("fresh_manifest_validation.json", text)
        self.assertIn("scripts/validate_true_geometry_selector.py", text)
        self.assertIn("true_geometry_filter_validation.json", text)
        self.assertIn("--required-variant extreme_01", text)
        self.assertIn("--required-variant extreme_02", text)
        self.assertIn("--required-variant extreme_03", text)

    def test_phase84_offset_head_smoke_trains_only_keypoint_offsets_and_reuses_phase81_filter(self) -> None:
        text = self.read_script("runs/phase84_extreme01_offset_head_smoke_20260621.sh")

        self.assertIn("phase78_matcher_only_hard_false_edge_train_eval_20260621", text)
        self.assertIn('export PFM_PHASE41_TRAIN_KEYPOINT_OFFSET_HEAD_ONLY="${PFM_PHASE41_TRAIN_KEYPOINT_OFFSET_HEAD_ONLY:-1}"', text)
        self.assertIn('export PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT="${PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT:-0.12}"', text)
        self.assertIn('export PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD="${PFM_PHASE41_TRAIN_DESCRIPTOR_HEAD:-0}"', text)
        self.assertIn("bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh", text)
        self.assertIn("runs/phase79_phase78_active_profile_eval_20260621.sh", text)
        self.assertIn("scripts/apply_match_detail_filter_calibrator.py", text)
        self.assertIn("--variant-threshold extreme_01=0.259406001", text)

    def test_phase85_extreme01_false_edge_uses_near_boundary_static_false_csv(self) -> None:
        text = self.read_script("runs/phase85_extreme01_near_boundary_false_edge_train_eval_20260621.sh")

        self.assertIn("phase85_extreme01_near_boundary_false_edge_20260621", text)
        self.assertIn("scripts/build_lazy_false_match_csv.py", text)
        self.assertIn("--min-error-px 5.0", text)
        self.assertIn("--max-error-px 10.0", text)
        self.assertIn("--target-variant extreme_01", text)
        self.assertIn("extreme01_near_boundary_false_matches.csv", text)
        self.assertIn('export PFM_PHASE41_FALSE_MATCH_CSV="${PFM_PHASE41_FALSE_MATCH_CSV:-${FALSE_MATCH_CSV}}"', text)
        self.assertIn('export PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS="${PFM_PHASE41_FREEZE_EXTRACTOR_WARMUP_STEPS:-80}"', text)
        self.assertIn('export PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT="${PFM_PHASE41_SELECTED_KEYPOINT_OFFSET_WEIGHT:-0.0}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.02}"', text)
        self.assertIn("runs/phase79_phase78_active_profile_eval_20260621.sh", text)
        self.assertIn("scripts/apply_match_detail_filter_calibrator.py", text)

    def test_phase86_extreme01_false_pair_replay_filters_train_manifest_to_static_false_pairs(self) -> None:
        text = self.read_script("runs/phase86_extreme01_false_pair_replay_train_eval_20260621.sh")

        self.assertIn("phase86_extreme01_false_pair_replay_20260621", text)
        self.assertIn("phase85_extreme01_near_boundary_false_edge_20260621", text)
        self.assertIn("extreme01_near_boundary_false_matches.csv", text)
        self.assertIn("extreme01_false_pair_train_manifest.csv", text)
        self.assertIn("reference_pose_id", text)
        self.assertIn("target_pose_id", text)
        self.assertIn('export PFM_PHASE41_TRAIN_MANIFEST="${PFM_PHASE41_TRAIN_MANIFEST:-${FALSE_PAIR_TRAIN_MANIFEST}}"', text)
        self.assertIn('export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-120}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.06}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.16}"', text)
        self.assertIn("runs/phase79_phase78_active_profile_eval_20260621.sh", text)
        self.assertIn("scripts/apply_match_detail_filter_calibrator.py", text)

    def test_phase88_match_set_rejection_diagnostic_uses_phase86_filtered_pairs(self) -> None:
        text = self.read_script("runs/phase88_phase86_match_set_rejection_diagnostic_20260621.sh")

        self.assertIn("phase88_phase86_match_set_rejection_diagnostic_20260621", text)
        self.assertIn("phase86_extreme01_false_pair_replay_20260621", text)
        self.assertIn("scripts/build_match_set_rejection_dataset.py", text)
        self.assertIn("scripts/train_match_set_rejection_calibrator.py", text)
        self.assertIn("scripts/apply_match_set_rejection_calibrator.py", text)
        self.assertIn("phase81_per_variant_filter/apply_dev/pair_summary.csv", text)
        self.assertIn("phase81_per_variant_filter/apply_dev/kept_match_details.csv", text)
        self.assertIn("--train-split dev", text)
        self.assertIn("--eval-split val", text)
        self.assertIn("--threshold-selection-source eval", text)
        self.assertIn("--threshold-objective pfm_wrong_cap", text)
        self.assertIn("--reject-action zero", text)
        self.assertIn("--reject-action lightglue", text)
        self.assertIn("by_variant_summary.json", text)

    def test_phase89_extreme01_recall_floor_reuses_phase86_false_pair_manifest(self) -> None:
        text = self.read_script("runs/phase89_extreme01_recall_floor_false_pair_train_eval_20260621.sh")

        self.assertIn("phase89_extreme01_recall_floor_false_pair_20260621", text)
        self.assertIn("phase86_extreme01_false_pair_replay_20260621", text)
        self.assertIn("near_boundary_false_edge_train/train_output/checkpoints/last_good_pytorch_pfm_state.pt", text)
        self.assertIn("manifests/extreme01_false_pair_train_manifest.csv", text)
        self.assertIn("hard_false_edges/extreme01_near_boundary_false_matches.csv", text)
        self.assertIn('export PFM_PHASE85_TRAIN_MANIFEST="${PFM_PHASE85_TRAIN_MANIFEST:-${FALSE_PAIR_TRAIN_MANIFEST}}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.06}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.16}"', text)
        self.assertIn(
            'export PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT:-0.05}"',
            text,
        )
        self.assertIn(
            'export PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_THRESHOLD="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_THRESHOLD:-0.0}"',
            text,
        )
        self.assertIn(
            'export PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_MARGIN="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_MARGIN:-0.25}"',
            text,
        )
        self.assertIn("runs/phase85_extreme01_near_boundary_false_edge_train_eval_20260621.sh", text)

    def test_phase90_strong_recall_floor_uses_score_scale_threshold(self) -> None:
        text = self.read_script("runs/phase90_extreme01_strong_recall_floor_train_eval_20260621.sh")

        self.assertIn("phase90_extreme01_strong_recall_floor_20260621", text)
        self.assertIn("phase86_extreme01_false_pair_replay_20260621", text)
        self.assertIn("manifests/extreme01_false_pair_train_manifest.csv", text)
        self.assertIn('export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-80}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT="${PFM_PHASE41_MINED_FALSE_MATCH_WEIGHT:-0.04}"', text)
        self.assertIn('export PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP="${PFM_PHASE41_MINED_FALSE_MATCH_LOSS_CAP:-0.12}"', text)
        self.assertIn(
            'export PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT:-0.01}"',
            text,
        )
        self.assertIn(
            'export PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_THRESHOLD="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_THRESHOLD:-12.0}"',
            text,
        )
        self.assertIn(
            'export PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_MARGIN="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_MARGIN:-0.5}"',
            text,
        )
        self.assertIn("runs/phase85_extreme01_near_boundary_false_edge_train_eval_20260621.sh", text)

    def test_phase141_gap_replay_conservative_launcher_limits_phase140_recall_pressure(self) -> None:
        text = self.read_script("runs/phase141_extreme01_02_gap_replay_conservative_train_eval_20260622.sh")

        self.assertIn("pgrep -af 'batch_pose_sim_dataset.py|pfm_pytorch_training.py|sat_sim_cuda", text)
        self.assertIn('export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"', text)
        self.assertIn("phase140_gap_replay_mixed_train.csv", text)
        self.assertIn('export PFM_PHASE41_STEPS="${PFM_PHASE41_STEPS:-30}"', text)
        self.assertIn(
            'export PFM_PHASE41_GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_MATCH_MARGIN_WEIGHT:-0.060}"',
            text,
        )
        self.assertIn(
            'export PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT="${PFM_PHASE41_GRAPH_MATCHER_TRUE_GEOMETRY_MATCH_COUNT_FLOOR_WEIGHT:-0.015}"',
            text,
        )
        self.assertIn(
            'export PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT="${PFM_PHASE41_WARP_OUTLIER_ACCEPT_WEIGHT:-0.010}"',
            text,
        )
        self.assertIn(
            'export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase141_gap_replay_conservative}"',
            text,
        )
        self.assertIn('exec bash runs/phase41_crosscam_extreme_geometry_train_eval_20260620.sh', text)

    def test_phase143_pair_accept_gate_launcher_trains_calibrator_and_pair_accept_head(self) -> None:
        text = self.read_script("runs/phase143_phase142_pair_accept_gate_train_eval_20260622.sh")

        self.assertIn("phase142_phase141_observable_gate_sweep_20260622", text)
        self.assertIn("apply_all_split_variant_safe_valid0356314_hmed1195/hybrid_rows.csv", text)
        self.assertIn("scripts/train_match_set_rejection_calibrator.py", text)
        self.assertIn("--threshold-objective hybrid_lightglue_wrong_cap", text)
        self.assertIn("scripts/apply_match_set_rejection_calibrator.py", text)
        self.assertIn("scripts/build_gate_acceptance_training_manifest.py", text)
        self.assertIn('export PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY="${PFM_PHASE41_TRAIN_PAIR_ACCEPT_HEAD_ONLY:-1}"', text)
        self.assertIn('export PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT="${PFM_PHASE41_PAIR_ACCEPT_LOSS_WEIGHT:-1.0}"', text)
        self.assertIn("phase143_phase142_pair_accept_gate_train_eval_20260622", text)

    def test_phase144_phase141_wrong_risk_launcher_reuses_phase123_false_replay_on_phase141(self) -> None:
        text = self.read_script("runs/phase144_phase141_wrong_risk_false_replay_train_eval_20260622.sh")

        self.assertIn("phase141_extreme01_02_gap_replay_conservative_train_eval_20260622", text)
        self.assertIn("phase140_gap_replay_mixed_train.csv", text)
        self.assertIn('export PFM_PHASE123_TARGET_HARD_FRACTION="${PFM_PHASE123_TARGET_HARD_FRACTION:-0.04}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase144_wrong_risk_false_replay}"', text)
        self.assertIn("exec bash runs/phase123_phase122_targeted_false_replay_train_eval_20260621.sh", text)

    def test_phase145_extreme01_guard_launcher_mines_only_extreme01_false_edges(self) -> None:
        text = self.read_script("runs/phase145_extreme01_precision_guard_train_eval_20260622.sh")

        self.assertIn("phase144_phase141_wrong_risk_false_replay_train_eval_20260622", text)
        self.assertIn("--target-variant extreme_01", text)
        self.assertNotIn("--target-variant extreme_02", text)
        self.assertNotIn("--target-variant extreme_03", text)
        self.assertIn('export PFM_PHASE123_DEV_FALSE_CSV="${PFM_PHASE123_DEV_FALSE_CSV:-${DEV_FALSE_CSV}}"', text)
        self.assertIn('export PFM_PHASE41_EVAL_SUBDIR="${PFM_PHASE41_EVAL_SUBDIR:-pfm_eval_kp6144_bins16_cap12_top512_accept_multiply_phase145_extreme01_guard}"', text)
        self.assertIn("exec bash runs/phase123_phase122_targeted_false_replay_train_eval_20260621.sh", text)


if __name__ == "__main__":
    unittest.main()
