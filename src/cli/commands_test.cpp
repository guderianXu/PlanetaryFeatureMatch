#include <string>
#include <vector>

#include "CLI11.hpp"
#include "cli/commands.h"
#include "tests/test_harness.h"

static void parse_missing_subcommand_throws()
{
    const std::vector<std::string> args = {"pfm"};

    PFM_REQUIRE_THROWS_AS(pfm::parse_cli(args), CLI::ParseError);
}

static void parse_extract_missing_required_option_throws()
{
    const std::vector<std::string> args = {
        "pfm", "extract", "--image", "a.png", "--checkpoint", "model.pt",
    };

    PFM_REQUIRE_THROWS_AS(pfm::parse_cli(args), CLI::ParseError);
}

static void parse_extract_command()
{
    const auto parsed = pfm::parse_cli({
        "pfm",
        "extract",
        "--image",
        "a.png",
        "--checkpoint",
        "model.pt",
        "--output",
        "a.pfm",
        "--visualization-dir",
        "vis",
        "--min-keypoint-intensity",
        "0.08",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Extract);
    PFM_REQUIRE(parsed.image == "a.png");
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.output == "a.pfm");
    PFM_REQUIRE(parsed.visualization_dir == "vis");
    PFM_REQUIRE_CLOSE(parsed.min_keypoint_intensity, 0.08, 1.0e-6);
}

static void parse_extract_keypoint_distribution_options()
{
    const auto options = pfm::parse_cli({"pfm",
                                         "extract",
                                         "--image",
                                         "a.png",
                                         "--checkpoint",
                                         "model.pt",
                                         "--output",
                                         "features.pt",
                                         "--keypoint-grid-rows",
                                         "4",
                                         "--keypoint-grid-cols",
                                         "6",
                                         "--keypoints-per-cell",
                                         "3",
                                         "--min-keypoints",
                                         "32",
                                         "--nms-radius",
                                         "2",
                                         "--descriptor-pool-radius",
                                         "2",
                                         "--disable-descriptor-orientation-canonicalization"});

    PFM_REQUIRE(options.keypoint_grid_rows == 4);
    PFM_REQUIRE(options.keypoint_grid_cols == 6);
    PFM_REQUIRE(options.keypoints_per_cell == 3);
    PFM_REQUIRE(options.min_keypoints == 32);
    PFM_REQUIRE(options.nms_radius == 2);
    PFM_REQUIRE(options.descriptor_pool_radius == 2);
    PFM_REQUIRE(options.disable_descriptor_orientation_canonicalization);
}

static void parse_invalid_keypoint_distribution_options_throw()
{
    PFM_REQUIRE_THROWS_AS(pfm::parse_cli({"pfm", "extract", "--image", "a.png", "--checkpoint", "model.pt", "--output",
                                          "features.pt", "--keypoint-grid-rows", "0"}),
                          CLI::ParseError);
    PFM_REQUIRE_THROWS_AS(pfm::parse_cli({"pfm", "match", "--image-a", "a.png", "--image-b", "b.png", "--checkpoint",
                                          "model.pt", "--output", "matches.pt", "--nms-radius", "-1"}),
                          CLI::ParseError);
    PFM_REQUIRE_THROWS_AS(pfm::parse_cli({"pfm", "eval", "--pairs", "pairs.txt", "--checkpoint", "model.pt", "--output",
                                          "report.pt", "--descriptor-pool-radius", "-1"}),
                          CLI::ParseError);
}

static void parse_train_defaults_to_bounded_resize()
{
    const auto parsed = pfm::parse_cli({
        "pfm",
        "train",
        "--image-dir",
        "images",
        "--checkpoint",
        "model.pt",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Train);
    PFM_REQUIRE(parsed.resize == 512);
    PFM_REQUIRE_CLOSE(parsed.min_keypoint_intensity, 0.08, 1.0e-6);
    PFM_REQUIRE_CLOSE(parsed.train_ratio, 1.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.val_ratio, 0.0, 1.0e-12);
}

static void parse_train_command()
{
    const auto parsed = pfm::parse_cli({
        "pfm",
        "train",
        "--image-dir",
        "images",
        "--checkpoint",
        "model.pt",
        "--init-checkpoint",
        "base.pt",
        "--epochs",
        "7",
        "--batch-size",
        "4",
        "--device",
        "cuda:0",
        "--resize",
        "512",
        "--training-crop-size",
        "384",
        "--pairs-per-image",
        "3",
        "--max-train-batches",
        "5",
        "--augmentation-profile",
        "rotation-only",
        "--augmentation-curriculum",
        "--rotation-step-degrees",
        "30",
        "--extreme-pair-ratio",
        "0.35",
        "--learning-rate",
        "0.00003",
        "--lr-warmup-steps",
        "12",
        "--min-learning-rate-ratio",
        "0.05",
        "--weight-decay",
        "0.08",
        "--max-grad-norm",
        "0.75",
        "--graph-keypoint-meta-dim",
        "12",
        "--training-profile",
        "smoke",
        "--train-backbone",
        "--train-dual-fpn",
        "--freeze-descriptor-head",
        "--train-sparse-context",
        "--train-keypoint-head",
        "--train-geometry-head",
        "--train-blended-descriptors",
        "--train-texture-adapter",
        "--train-descriptor-fusion",
        "--train-quality-head",
        "--train-graph-matcher",
        "--graph-matcher-accept-weight",
        "0.35",
        "--graph-matcher-accept-negative-topk",
        "6",
        "--graph-matcher-no-match-points",
        "24",
        "--graph-matcher-no-match-min-distance",
        "5.5",
        "--graph-matcher-metadata-mode",
        "no_xy",
        "--graph-matcher-train-max-attention-layers",
        "2",
        "--graph-matcher-train-random-attention-layers",
        "--graph-matcher-train-max-attention-work-fraction",
        "0.5",
        "--graph-matcher-train-width-keep-ratio",
        "0.5",
        "--graph-matcher-prune-ranking-weight",
        "0.15",
        "--graph-matcher-prune-ranking-margin",
        "0.4",
        "--graph-matcher-stop-confidence-weight",
        "0.07",
        "--graph-matcher-stop-confidence-margin",
        "0.6",
        "--training-texture-blend-weight",
        "0.75",
        "--synthetic-pair-cache-dir",
        "pair_cache",
        "--cache-only",
        "--extra-synthetic-pair-cache-dir",
        "rotate_cache",
        "--extra-synthetic-pair-cache-dir",
        "compound_cache",
        "--hard-synthetic-pair-cache-dir",
        "compound_hard_cache",
        "--hard-synthetic-pair-cache-dir",
        "compound_extreme_cache",
        "--hard-synthetic-pair-cache-repeats",
        "4",
        "--hard-synthetic-pair-cache-index",
        "3",
        "--hard-synthetic-pair-cache-index",
        "8",
        "--pair-cache-dir",
        "sim_cache_train",
        "--pair-cache-limit",
        "9",
        "--memory-cache-items",
        "11",
        "--synthetic-pair-cache-rebuild",
        "--log-csv",
        "metrics.csv",
        "--dataloader-workers",
        "2",
        "--prefetch-batches",
        "3",
        "--pin-memory",
        "--descriptor-only-finetune",
        "--viewpoint-head-only-finetune",
        "--graph-only-finetune",
        "--disable-descriptor-orientation-canonicalization",
        "--visualization-dir",
        "train_vis",
        "--visualization-samples",
        "6",
        "--min-keypoint-intensity",
        "0.08",
        "--max-keypoints",
        "2048",
        "--min-keypoints",
        "512",
        "--keypoint-grid-rows",
        "4",
        "--keypoint-grid-cols",
        "6",
        "--keypoints-per-cell",
        "8",
        "--nms-radius",
        "2",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Train);
    PFM_REQUIRE(parsed.image_dir == "images");
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.init_checkpoint == "base.pt");
    PFM_REQUIRE(parsed.epochs == 7);
    PFM_REQUIRE(parsed.batch_size == 4);
    PFM_REQUIRE(parsed.device == "cuda:0");
    PFM_REQUIRE(parsed.resize == 512);
    PFM_REQUIRE(parsed.training_crop_size == 384);
    PFM_REQUIRE(parsed.pairs_per_image == 3);
    PFM_REQUIRE(parsed.max_train_batches == 5);
    PFM_REQUIRE(parsed.augmentation_profile == "rotation-only");
    PFM_REQUIRE(parsed.augmentation_curriculum);
    PFM_REQUIRE_CLOSE(parsed.rotation_step_degrees, 30.0, 1.0e-6);
    PFM_REQUIRE_CLOSE(parsed.extreme_pair_ratio, 0.35, 1.0e-6);
    PFM_REQUIRE_CLOSE(parsed.learning_rate, 3.0e-5, 1.0e-12);
    PFM_REQUIRE(parsed.lr_warmup_steps == 12);
    PFM_REQUIRE_CLOSE(parsed.min_learning_rate_ratio, 0.05, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.weight_decay, 0.08, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.gradient_clip_norm, 0.75, 1.0e-12);
    PFM_REQUIRE(parsed.graph_keypoint_meta_dim == 12);
    PFM_REQUIRE(parsed.training_profile == "smoke");
    PFM_REQUIRE(parsed.train_backbone);
    PFM_REQUIRE(parsed.train_dual_fpn);
    PFM_REQUIRE(parsed.freeze_descriptor_head);
    PFM_REQUIRE(parsed.train_sparse_context);
    PFM_REQUIRE(parsed.train_keypoint_head);
    PFM_REQUIRE(parsed.train_geometry_head);
    PFM_REQUIRE(parsed.train_blended_descriptors);
    PFM_REQUIRE(parsed.train_texture_adapter);
    PFM_REQUIRE(parsed.train_descriptor_fusion);
    PFM_REQUIRE(parsed.train_quality_head);
    PFM_REQUIRE(parsed.train_graph_matcher);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_accept_weight, 0.35, 1.0e-12);
    PFM_REQUIRE(parsed.graph_matcher_accept_negative_topk == 6);
    PFM_REQUIRE(parsed.graph_matcher_no_match_points == 24);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_no_match_min_distance, 5.5, 1.0e-12);
    PFM_REQUIRE(parsed.graph_matcher_metadata_mode == "no_xy");
    PFM_REQUIRE(parsed.graph_matcher_train_max_attention_layers == 2);
    PFM_REQUIRE(parsed.graph_matcher_train_random_attention_layers);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_train_max_attention_work_fraction, 0.5, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_train_width_keep_ratio, 0.5, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_prune_ranking_weight, 0.15, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_prune_ranking_margin, 0.4, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_stop_confidence_weight, 0.07, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_stop_confidence_margin, 0.6, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.training_texture_blend_weight, 0.75, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.min_keypoint_intensity, 0.08, 1.0e-6);
    PFM_REQUIRE(parsed.synthetic_pair_cache_dir == "pair_cache");
    PFM_REQUIRE(parsed.cache_only);
    PFM_REQUIRE(parsed.extra_synthetic_pair_cache_dirs.size() == 2);
    PFM_REQUIRE(parsed.extra_synthetic_pair_cache_dirs[0] == "rotate_cache");
    PFM_REQUIRE(parsed.extra_synthetic_pair_cache_dirs[1] == "compound_cache");
    PFM_REQUIRE(parsed.hard_synthetic_pair_cache_dirs.size() == 2);
    PFM_REQUIRE(parsed.hard_synthetic_pair_cache_dirs[0] == "compound_hard_cache");
    PFM_REQUIRE(parsed.hard_synthetic_pair_cache_dirs[1] == "compound_extreme_cache");
    PFM_REQUIRE(parsed.hard_synthetic_pair_cache_repeats == 4);
    PFM_REQUIRE(parsed.hard_synthetic_pair_cache_indices == std::vector<int64_t>({3, 8}));
    PFM_REQUIRE(parsed.pair_cache_dirs == std::vector<std::string>({"sim_cache_train"}));
    PFM_REQUIRE(parsed.pair_cache_limit == 9);
    PFM_REQUIRE(parsed.pair_memory_cache_size == 11);
    PFM_REQUIRE(parsed.synthetic_pair_cache_rebuild);
    PFM_REQUIRE(parsed.log_csv == "metrics.csv");
    PFM_REQUIRE(parsed.dataloader_workers == 2);
    PFM_REQUIRE(parsed.prefetch_batches == 3);
    PFM_REQUIRE(parsed.pin_memory);
    PFM_REQUIRE(parsed.descriptor_only_finetune);
    PFM_REQUIRE(parsed.viewpoint_head_only_finetune);
    PFM_REQUIRE(parsed.graph_only_finetune);
    PFM_REQUIRE(parsed.disable_descriptor_orientation_canonicalization);
    PFM_REQUIRE(parsed.visualization_dir == "train_vis");
    PFM_REQUIRE(parsed.visualization_samples == 6);
    PFM_REQUIRE(!parsed.visualization_samples_all);
    PFM_REQUIRE(parsed.max_keypoints == 2048);
    PFM_REQUIRE(parsed.min_keypoints == 512);
    PFM_REQUIRE(parsed.keypoint_grid_rows == 4);
    PFM_REQUIRE(parsed.keypoint_grid_cols == 6);
    PFM_REQUIRE(parsed.keypoints_per_cell == 8);
    PFM_REQUIRE(parsed.nms_radius == 2);
}

static void parse_train_full_v21_sets_large_model_dimensions()
{
    const auto parsed = pfm::parse_cli({
        "pfm",
        "train",
        "--image-dir",
        "images",
        "--checkpoint",
        "model.pt",
        "--full-v21",
    });

    PFM_REQUIRE(parsed.full_v21);
    PFM_REQUIRE(parsed.base_channels == 64);
    PFM_REQUIRE(parsed.descriptor_dim == 256);
    PFM_REQUIRE(parsed.graph_hidden_dim == 512);
    PFM_REQUIRE(parsed.graph_attention_layers == 8);
    PFM_REQUIRE(parsed.graph_keypoint_meta_dim == 16);
}

static void parse_train_python_compare_profile_options()
{
    const auto parsed = pfm::parse_cli({
        "pfm",
        "train",
        "--pair-cache-dir",
        "cache/train",
        "--checkpoint",
        "model.pt",
        "--training-profile",
        "python-compare",
        "--samples-per-pair",
        "512",
        "--synthetic-loss-weight",
        "0.1",
        "--graph-matcher-loss-weight",
        "1.0",
        "--temperature",
        "0.07",
        "--seed",
        "20260603",
    });

    PFM_REQUIRE(parsed.training_profile == "python-compare");
    PFM_REQUIRE(parsed.samples_per_pair == 512);
    PFM_REQUIRE_CLOSE(parsed.synthetic_loss_weight, 0.1, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_loss_weight, 1.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_accept_weight, 0.2, 1.0e-12);
    PFM_REQUIRE(parsed.graph_matcher_accept_negative_topk == 8);
    PFM_REQUIRE(parsed.graph_matcher_no_match_points == 0);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_no_match_min_distance, 4.0, 1.0e-12);
    PFM_REQUIRE(parsed.graph_matcher_metadata_mode == "full");
    PFM_REQUIRE(parsed.graph_matcher_train_max_attention_layers == 0);
    PFM_REQUIRE(!parsed.graph_matcher_train_random_attention_layers);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_train_max_attention_work_fraction, 1.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_train_width_keep_ratio, 1.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_prune_ranking_weight, 0.1, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_prune_ranking_margin, 0.25, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_stop_confidence_weight, 0.05, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_matcher_stop_confidence_margin, 0.5, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.temperature, 0.07, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.min_learning_rate_ratio, 1.0, 1.0e-12);
    PFM_REQUIRE(parsed.seed == 20260603);
}

static void parse_train_python_compare_allows_explicit_min_learning_rate_ratio()
{
    const auto parsed = pfm::parse_cli({
        "pfm",
        "train",
        "--pair-cache-dir",
        "cache/train",
        "--checkpoint",
        "model.pt",
        "--training-profile",
        "python-compare",
        "--min-learning-rate-ratio",
        "0.25",
    });

    PFM_REQUIRE(parsed.training_profile == "python-compare");
    PFM_REQUIRE_CLOSE(parsed.min_learning_rate_ratio, 0.25, 1.0e-12);
}

static void parse_train_visualization_defaults_to_four_samples()
{
    const auto parsed = pfm::parse_cli(
        {"pfm", "train", "--image-dir", "images", "--checkpoint", "model.pt", "--visualization-dir", "train_vis"});

    PFM_REQUIRE(parsed.visualization_dir == "train_vis");
    PFM_REQUIRE(parsed.visualization_samples == 4);
    PFM_REQUIRE(!parsed.visualization_samples_all);
}

static void parse_train_visualization_samples_all()
{
    const auto parsed = pfm::parse_cli({"pfm", "train", "--image-dir", "images", "--checkpoint", "model.pt",
                                        "--visualization-dir", "train_vis", "--visualization-samples", "all"});

    PFM_REQUIRE(parsed.visualization_samples_all);
}

static void parse_train_visualization_samples_zero_disables_output()
{
    const auto parsed = pfm::parse_cli({"pfm", "train", "--image-dir", "images", "--checkpoint", "model.pt",
                                        "--visualization-dir", "train_vis", "--visualization-samples", "0"});

    PFM_REQUIRE(parsed.visualization_samples == 0);
    PFM_REQUIRE(!parsed.visualization_samples_all);
}

static void parse_train_visualization_invalid_samples_throw()
{
    PFM_REQUIRE_THROWS_AS(pfm::parse_cli({"pfm", "train", "--image-dir", "images", "--checkpoint", "model.pt",
                                          "--visualization-dir", "train_vis", "--visualization-samples", "invalid"}),
                          CLI::ParseError);
}

static void parse_match_command()
{
    const auto parsed = pfm::parse_cli({
        "pfm",
        "match",
        "--image-a",
        "a.png",
        "--image-b",
        "b.png",
        "--feature-a",
        "a_features.pt",
        "--feature-b",
        "b_features.pt",
        "--match-mode",
        "sparse",
        "--sparse-match-strategy",
        "python-raw-mutual",
        "--max-matches",
        "256",
        "--checkpoint",
        "model.pt",
        "--output",
        "matches.json",
        "--warp-a-to-b",
        "pair_000000.pt",
        "--match-correct-threshold-pixels",
        "3.5",
        "--max-keypoints",
        "2048",
        "--min-keypoints",
        "512",
        "--semi-dense-threshold",
        "0.5",
        "--visualization-dir",
        "vis",
        "--min-keypoint-intensity",
        "0.08",
        "--sparse-geometry-filter",
        "rotation-only",
        "--graph-width-prune-min-score",
        "0.5",
        "--graph-early-stop-min-confidence",
        "0.85",
        "--graph-inference-preset",
        "high_precision",
        "--graph-min-accept-probability",
        "0.75",
        "--graph-max-attention-layers",
        "2",
        "--graph-max-attention-work-fraction",
        "0.5",
        "--graph-width-prune-keep-ratio",
        "0.4",
        "--graph-fallback-mode",
        "none",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Match);
    PFM_REQUIRE(parsed.image_a == "a.png");
    PFM_REQUIRE(parsed.image_b == "b.png");
    PFM_REQUIRE(parsed.feature_a == "a_features.pt");
    PFM_REQUIRE(parsed.feature_b == "b_features.pt");
    PFM_REQUIRE(parsed.match_mode == "sparse");
    PFM_REQUIRE(parsed.sparse_match_strategy == "python-raw-mutual");
    PFM_REQUIRE(parsed.max_matches == 256);
    PFM_REQUIRE(parsed.warp_a_to_b == "pair_000000.pt");
    PFM_REQUIRE_CLOSE(parsed.match_correct_threshold_pixels, 3.5, 1.0e-6);
    PFM_REQUIRE(parsed.max_keypoints == 2048);
    PFM_REQUIRE(parsed.min_keypoints == 512);
    PFM_REQUIRE_CLOSE(parsed.semi_dense_threshold, 0.5, 1.0e-6);
    PFM_REQUIRE_CLOSE(parsed.min_keypoint_intensity, 0.08, 1.0e-6);
    PFM_REQUIRE(parsed.visualization_dir == "vis");
    PFM_REQUIRE(parsed.sparse_geometry_filter == "rotation-only");
    PFM_REQUIRE_CLOSE(parsed.graph_width_prune_min_score, 0.5, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_early_stop_min_confidence, 0.85, 1.0e-12);
    PFM_REQUIRE(parsed.graph_inference_preset == "high_precision");
    PFM_REQUIRE_CLOSE(parsed.graph_min_accept_probability, 0.75, 1.0e-12);
    PFM_REQUIRE(parsed.graph_max_attention_layers == 2);
    PFM_REQUIRE_CLOSE(parsed.graph_max_attention_work_fraction, 0.5, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_width_prune_keep_ratio, 0.4, 1.0e-12);
    PFM_REQUIRE(parsed.graph_fallback_mode == "none");
}

static void parse_match_defaults_to_sparse_mode()
{
    const auto parsed = pfm::parse_cli({"pfm", "match", "--image-a", "a.png", "--image-b", "b.png", "--checkpoint",
                                        "model.pt", "--output", "matches.pt"});

    PFM_REQUIRE(parsed.match_mode == "sparse");
    PFM_REQUIRE(parsed.sparse_match_strategy == "learned");
    PFM_REQUIRE(parsed.max_matches == 512);
}

static void parse_match_accepts_adaptive_and_local_sparse_geometry_filters()
{
    for (const std::string mode : {"adaptive", "local", "projective", "rotation-only"})
    {
        const auto parsed = pfm::parse_cli({"pfm", "match", "--image-a", "a.png", "--image-b", "b.png", "--checkpoint",
                                            "model.pt", "--output", "matches.pt", "--sparse-geometry-filter", mode});

        PFM_REQUIRE(parsed.sparse_geometry_filter == mode);
    }
}

static void parse_eval_command()
{
    const auto parsed = pfm::parse_cli({
        "pfm",
        "eval",
        "--pairs",
        "pairs.txt",
        "--checkpoint",
        "model.pt",
        "--output",
        "report.json",
        "--device",
        "cuda:1",
        "--semi-dense-threshold",
        "0.25",
        "--max-keypoints",
        "1024",
        "--min-keypoints",
        "256",
        "--sparse-match-strategy",
        "python-raw-mutual",
        "--max-matches",
        "128",
        "--min-keypoint-intensity",
        "0.08",
        "--graph-width-prune-min-score",
        "0.25",
        "--graph-early-stop-min-confidence",
        "0.9",
        "--graph-inference-preset",
        "fast",
        "--graph-min-accept-probability",
        "0.7",
        "--graph-max-attention-layers",
        "3",
        "--graph-max-attention-work-fraction",
        "0.4",
        "--graph-width-prune-keep-ratio",
        "0.5",
        "--graph-fallback-mode",
        "none",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Eval);
    PFM_REQUIRE(parsed.pairs == "pairs.txt");
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.output == "report.json");
    PFM_REQUIRE(parsed.device == "cuda:1");
    PFM_REQUIRE(parsed.max_keypoints == 1024);
    PFM_REQUIRE(parsed.min_keypoints == 256);
    PFM_REQUIRE(parsed.sparse_match_strategy == "python-raw-mutual");
    PFM_REQUIRE(parsed.max_matches == 128);
    PFM_REQUIRE_CLOSE(parsed.semi_dense_threshold, 0.25, 1.0e-6);
    PFM_REQUIRE_CLOSE(parsed.min_keypoint_intensity, 0.08, 1.0e-6);
    PFM_REQUIRE_CLOSE(parsed.graph_width_prune_min_score, 0.25, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_early_stop_min_confidence, 0.9, 1.0e-12);
    PFM_REQUIRE(parsed.graph_inference_preset == "fast");
    PFM_REQUIRE_CLOSE(parsed.graph_min_accept_probability, 0.7, 1.0e-12);
    PFM_REQUIRE(parsed.graph_max_attention_layers == 3);
    PFM_REQUIRE_CLOSE(parsed.graph_max_attention_work_fraction, 0.4, 1.0e-12);
    PFM_REQUIRE_CLOSE(parsed.graph_width_prune_keep_ratio, 0.5, 1.0e-12);
    PFM_REQUIRE(parsed.graph_fallback_mode == "none");
}

static void parse_export_command()
{
    const auto parsed = pfm::parse_cli({
        "pfm",
        "export",
        "--checkpoint",
        "model.pt",
        "--output",
        "exported.pt",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Export);
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.output == "exported.pt");
}

static void parse_min_keypoint_intensity_out_of_range_throws()
{
    PFM_REQUIRE_THROWS_AS(pfm::parse_cli({"pfm", "extract", "--image", "a.tif", "--checkpoint", "model.pt", "--output",
                                          "features.pt", "--min-keypoint-intensity", "1.5"}),
                          CLI::ParseError);
    PFM_REQUIRE_THROWS_AS(pfm::parse_cli({"pfm", "train", "--image-dir", "images", "--checkpoint", "model.pt",
                                          "--min-keypoint-intensity", "-0.1"}),
                          CLI::ParseError);
}

static void parse_match_invalid_max_keypoints_throws()
{
    const std::vector<std::string> args = {
        "pfm",          "match",    "--image-a", "a.png",        "--image-b",       "b.png",
        "--checkpoint", "model.pt", "--output",  "matches.json", "--max-keypoints", "invalid",
    };

    PFM_REQUIRE_THROWS_AS(pfm::parse_cli(args), CLI::ParseError);
}

static void parseMatchInvalidModeThrows()
{
    PFM_REQUIRE_THROWS_AS(pfm::parse_cli({"pfm", "match", "--image-a", "a.png", "--image-b", "b.png", "--checkpoint",
                                          "model.pt", "--output", "matches.pt", "--match-mode", "invalid"}),
                          CLI::ParseError);
}

static void parse_match_invalid_sparse_geometry_filter_throws()
{
    PFM_REQUIRE_THROWS_AS(pfm::parse_cli({"pfm", "match", "--image-a", "a.png", "--image-b", "b.png", "--checkpoint",
                                          "model.pt", "--output", "matches.pt", "--sparse-geometry-filter", "invalid"}),
                          CLI::ParseError);
}

static void parse_match_invalid_sparse_strategy_throws()
{
    PFM_REQUIRE_THROWS_AS(pfm::parse_cli({"pfm", "match", "--image-a", "a.png", "--image-b", "b.png", "--checkpoint",
                                          "model.pt", "--output", "matches.pt", "--sparse-match-strategy", "invalid"}),
                          CLI::ParseError);
}

static void top_level_help_lists_subcommand_options()
{
    pfm::CliOptions options;
    auto app = pfm::build_cli_app(options);
    const auto help = app->help();

    PFM_REQUIRE(help.find("train --image-dir") != std::string::npos);
    PFM_REQUIRE(help.find("--resize") != std::string::npos);
    PFM_REQUIRE(help.find("--pairs-per-image") != std::string::npos);
    PFM_REQUIRE(help.find("--augmentation-profile") != std::string::npos);
    PFM_REQUIRE(help.find("--augmentation-curriculum") != std::string::npos);
    PFM_REQUIRE(help.find("--extreme-pair-ratio") != std::string::npos);
    PFM_REQUIRE(help.find("--max-training-images-per-epoch") == std::string::npos);
    PFM_REQUIRE(help.find("--synthetic-pair-cache-dir") != std::string::npos);
    PFM_REQUIRE(help.find("--visualization-dir") != std::string::npos);
    PFM_REQUIRE(help.find("--min-keypoint-intensity") != std::string::npos);
    PFM_REQUIRE(help.find("--keypoint-grid-rows") != std::string::npos);
    PFM_REQUIRE(help.find("--keypoint-grid-cols") != std::string::npos);
    PFM_REQUIRE(help.find("--keypoints-per-cell") != std::string::npos);
    PFM_REQUIRE(help.find("--nms-radius") != std::string::npos);
    PFM_REQUIRE(help.find("--feature-a") != std::string::npos);
    PFM_REQUIRE(help.find("--feature-b") != std::string::npos);
    PFM_REQUIRE(help.find("--match-mode") != std::string::npos);
    PFM_REQUIRE(help.find("--sparse-match-strategy") != std::string::npos);
    PFM_REQUIRE(help.find("--max-matches") != std::string::npos);
    PFM_REQUIRE(help.find("--sparse-geometry-filter") != std::string::npos);
    PFM_REQUIRE(help.find("extract --image") != std::string::npos);
    PFM_REQUIRE(help.find("match --image-a") != std::string::npos);
    PFM_REQUIRE(help.find("eval --pairs") != std::string::npos);
    PFM_REQUIRE(help.find("export --checkpoint") != std::string::npos);
}

static void run_cli_help_returns_zero()
{
    const char* argv[] = {"pfm", "--help"};

    PFM_REQUIRE(pfm::run_cli(2, const_cast<char**>(argv)) == 0);
}

static void run_extract_without_checkpoint_path_fails_cleanly()
{
    const char* argv[] = {"pfm", "extract", "--image", "a.png", "--checkpoint", "", "--output", "a.pfm"};

    PFM_REQUIRE(pfm::run_cli(8, const_cast<char**>(argv)) != 0);
}

static void run_extract_with_required_paths_fails_without_loadable_checkpoint()
{
    const char* argv[] = {"pfm", "extract", "--image", "a.png", "--checkpoint", "model.pt", "--output", "a.pfm"};

    PFM_REQUIRE(pfm::run_cli(8, const_cast<char**>(argv)) != 0);
}

static void run_match_with_required_paths_returns_task_8_failure()
{
    const char* argv[] = {
        "pfm",   "match",        "--image-a", "a.png",    "--image-b",
        "b.png", "--checkpoint", "model.pt",  "--output", "matches.json",
    };

    PFM_REQUIRE(pfm::run_cli(10, const_cast<char**>(argv)) != 0);
}

static void run_eval_with_required_paths_returns_task_8_failure()
{
    const char* argv[] = {
        "pfm", "eval", "--pairs", "pairs.txt", "--checkpoint", "model.pt", "--output", "report.json",
    };

    PFM_REQUIRE(pfm::run_cli(8, const_cast<char**>(argv)) != 0);
}

static void run_train_with_required_paths_fails_without_image_directory()
{
    const char* argv[] = {
        "pfm", "train", "--image-dir", "images", "--checkpoint", "model.pt", "--epochs", "1", "--batch-size", "1",
    };

    PFM_REQUIRE(pfm::run_cli(10, const_cast<char**>(argv)) != 0);
}

static void run_export_with_required_paths_fails_without_loadable_checkpoint()
{
    const char* argv[] = {"pfm", "export", "--checkpoint", "model.pt", "--output", "exported.pt"};

    PFM_REQUIRE(pfm::run_cli(6, const_cast<char**>(argv)) != 0);
}

void register_cli_tests()
{
    register_test("parse_missing_subcommand_throws", parse_missing_subcommand_throws);
    register_test("parse_extract_missing_required_option_throws", parse_extract_missing_required_option_throws);
    register_test("parse_extract_command", parse_extract_command);
    register_test("parse_extract_keypoint_distribution_options", parse_extract_keypoint_distribution_options);
    register_test("parse_invalid_keypoint_distribution_options_throw",
                  parse_invalid_keypoint_distribution_options_throw);
    register_test("parse_train_defaults_to_bounded_resize", parse_train_defaults_to_bounded_resize);
    register_test("parse_train_command", parse_train_command);
    register_test("parse_train_full_v21_sets_large_model_dimensions", parse_train_full_v21_sets_large_model_dimensions);
    register_test("parse_train_python_compare_profile_options", parse_train_python_compare_profile_options);
    register_test("parse_train_python_compare_allows_explicit_min_learning_rate_ratio",
                  parse_train_python_compare_allows_explicit_min_learning_rate_ratio);
    register_test("parse_train_visualization_defaults_to_four_samples",
                  parse_train_visualization_defaults_to_four_samples);
    register_test("parse_train_visualization_samples_all", parse_train_visualization_samples_all);
    register_test("parse_train_visualization_samples_zero_disables_output",
                  parse_train_visualization_samples_zero_disables_output);
    register_test("parse_train_visualization_invalid_samples_throw", parse_train_visualization_invalid_samples_throw);
    register_test("parse_match_command", parse_match_command);
    register_test("parse_match_defaults_to_sparse_mode", parse_match_defaults_to_sparse_mode);
    register_test("parse_match_accepts_adaptive_and_local_sparse_geometry_filters",
                  parse_match_accepts_adaptive_and_local_sparse_geometry_filters);
    register_test("parse_eval_command", parse_eval_command);
    register_test("parse_export_command", parse_export_command);
    register_test("parse_min_keypoint_intensity_out_of_range_throws", parse_min_keypoint_intensity_out_of_range_throws);
    register_test("parse_match_invalid_max_keypoints_throws", parse_match_invalid_max_keypoints_throws);
    register_test("parse_match_invalid_mode_throws", parseMatchInvalidModeThrows);
    register_test("parse_match_invalid_sparse_geometry_filter_throws",
                  parse_match_invalid_sparse_geometry_filter_throws);
    register_test("parse_match_invalid_sparse_strategy_throws", parse_match_invalid_sparse_strategy_throws);
    register_test("top_level_help_lists_subcommand_options", top_level_help_lists_subcommand_options);
    register_test("run_cli_help_returns_zero", run_cli_help_returns_zero);
    register_test("run_extract_without_checkpoint_path_fails_cleanly",
                  run_extract_without_checkpoint_path_fails_cleanly);
    register_test("run_extract_with_required_paths_fails_without_loadable_checkpoint",
                  run_extract_with_required_paths_fails_without_loadable_checkpoint);
    register_test("run_match_with_required_paths_returns_task_8_failure",
                  run_match_with_required_paths_returns_task_8_failure);
    register_test("run_eval_with_required_paths_returns_task_8_failure",
                  run_eval_with_required_paths_returns_task_8_failure);
    register_test("run_train_with_required_paths_fails_without_image_directory",
                  run_train_with_required_paths_fails_without_image_directory);
    register_test("run_export_with_required_paths_fails_without_loadable_checkpoint",
                  run_export_with_required_paths_fails_without_loadable_checkpoint);
}
