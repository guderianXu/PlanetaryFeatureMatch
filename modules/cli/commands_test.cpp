#include "tests/test_harness.h"

#include <string>
#include <vector>

#include "CLI11.hpp"

#include "cli/commands.h"

static void parse_missing_subcommand_throws() {
    const std::vector<std::string> args = {"pfm"};

    PFM_REQUIRE_THROWS_AS(pfm::parse_cli(args), CLI::ParseError);
}

static void parse_extract_missing_required_option_throws() {
    const std::vector<std::string> args = {
        "pfm",
        "extract",
        "--image",
        "a.png",
        "--checkpoint",
        "model.pt",
    };

    PFM_REQUIRE_THROWS_AS(pfm::parse_cli(args), CLI::ParseError);
}

static void parse_extract_command() {
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

static void parse_train_defaults_to_bounded_resize() {
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
}

static void parse_train_command() {
    const auto parsed = pfm::parse_cli({
        "pfm",
        "train",
        "--image-dir",
        "images",
        "--checkpoint",
        "model.pt",
        "--epochs",
        "7",
        "--batch-size",
        "4",
        "--device",
        "cuda:0",
        "--resize",
        "512",
        "--pairs-per-image",
        "3",
        "--augmentation-profile",
        "hard",
        "--extreme-pair-ratio",
        "0.35",
        "--synthetic-pair-cache-dir",
        "pair_cache",
        "--synthetic-pair-cache-rebuild",
        "--min-keypoint-intensity",
        "0.08",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Train);
    PFM_REQUIRE(parsed.image_dir == "images");
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.epochs == 7);
    PFM_REQUIRE(parsed.batch_size == 4);
    PFM_REQUIRE(parsed.device == "cuda:0");
    PFM_REQUIRE(parsed.resize == 512);
    PFM_REQUIRE(parsed.pairs_per_image == 3);
    PFM_REQUIRE(parsed.augmentation_profile == "hard");
    PFM_REQUIRE_CLOSE(parsed.extreme_pair_ratio, 0.35, 1.0e-6);
    PFM_REQUIRE_CLOSE(parsed.min_keypoint_intensity, 0.08, 1.0e-6);
    PFM_REQUIRE(parsed.synthetic_pair_cache_dir == "pair_cache");
    PFM_REQUIRE(parsed.synthetic_pair_cache_rebuild);
}

static void parse_match_command() {
    const auto parsed = pfm::parse_cli({
        "pfm",
        "match",
        "--image-a",
        "a.png",
        "--image-b",
        "b.png",
        "--checkpoint",
        "model.pt",
        "--output",
        "matches.json",
        "--max-keypoints",
        "2048",
        "--semi-dense-threshold",
        "0.5",
        "--visualization-dir",
        "vis",
        "--min-keypoint-intensity",
        "0.08",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Match);
    PFM_REQUIRE(parsed.image_a == "a.png");
    PFM_REQUIRE(parsed.image_b == "b.png");
    PFM_REQUIRE(parsed.max_keypoints == 2048);
    PFM_REQUIRE_CLOSE(parsed.semi_dense_threshold, 0.5, 1.0e-6);
    PFM_REQUIRE_CLOSE(parsed.min_keypoint_intensity, 0.08, 1.0e-6);
    PFM_REQUIRE(parsed.visualization_dir == "vis");
}

static void parse_eval_command() {
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
        "--min-keypoint-intensity",
        "0.08",
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Eval);
    PFM_REQUIRE(parsed.pairs == "pairs.txt");
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.output == "report.json");
    PFM_REQUIRE(parsed.device == "cuda:1");
    PFM_REQUIRE_CLOSE(parsed.semi_dense_threshold, 0.25, 1.0e-6);
    PFM_REQUIRE_CLOSE(parsed.min_keypoint_intensity, 0.08, 1.0e-6);
}

static void parse_export_command() {
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

static void parse_min_keypoint_intensity_out_of_range_throws() {
    PFM_REQUIRE_THROWS_AS(
        pfm::parse_cli({"pfm",
                        "extract",
                        "--image",
                        "a.tif",
                        "--checkpoint",
                        "model.pt",
                        "--output",
                        "features.pt",
                        "--min-keypoint-intensity",
                        "1.5"}),
        CLI::ParseError);
    PFM_REQUIRE_THROWS_AS(
        pfm::parse_cli({"pfm",
                        "train",
                        "--image-dir",
                        "images",
                        "--checkpoint",
                        "model.pt",
                        "--min-keypoint-intensity",
                        "-0.1"}),
        CLI::ParseError);
}

static void parse_match_invalid_max_keypoints_throws() {
    const std::vector<std::string> args = {
        "pfm",
        "match",
        "--image-a",
        "a.png",
        "--image-b",
        "b.png",
        "--checkpoint",
        "model.pt",
        "--output",
        "matches.json",
        "--max-keypoints",
        "invalid",
    };

    PFM_REQUIRE_THROWS_AS(pfm::parse_cli(args), CLI::ParseError);
}

static void top_level_help_lists_subcommand_options() {
    pfm::CliOptions options;
    auto app = pfm::build_cli_app(options);
    const auto help = app->help();

    PFM_REQUIRE(help.find("train --image-dir") != std::string::npos);
    PFM_REQUIRE(help.find("--resize") != std::string::npos);
    PFM_REQUIRE(help.find("--pairs-per-image") != std::string::npos);
    PFM_REQUIRE(help.find("--augmentation-profile") != std::string::npos);
    PFM_REQUIRE(help.find("--extreme-pair-ratio") != std::string::npos);
    PFM_REQUIRE(help.find("--max-training-images-per-epoch") == std::string::npos);
    PFM_REQUIRE(help.find("--synthetic-pair-cache-dir") != std::string::npos);
    PFM_REQUIRE(help.find("--visualization-dir") != std::string::npos);
    PFM_REQUIRE(help.find("--min-keypoint-intensity") != std::string::npos);
    PFM_REQUIRE(help.find("extract --image") != std::string::npos);
    PFM_REQUIRE(help.find("match --image-a") != std::string::npos);
    PFM_REQUIRE(help.find("eval --pairs") != std::string::npos);
    PFM_REQUIRE(help.find("export --checkpoint") != std::string::npos);
}

static void run_cli_help_returns_zero() {
    const char* argv[] = {"pfm", "--help"};

    PFM_REQUIRE(pfm::run_cli(2, const_cast<char**>(argv)) == 0);
}

static void run_extract_without_checkpoint_path_fails_cleanly() {
    const char* argv[] = {"pfm", "extract", "--image", "a.png", "--checkpoint", "", "--output", "a.pfm"};

    PFM_REQUIRE(pfm::run_cli(8, const_cast<char**>(argv)) != 0);
}

static void run_extract_with_required_paths_fails_without_loadable_checkpoint() {
    const char* argv[] = {"pfm", "extract", "--image", "a.png", "--checkpoint", "model.pt", "--output", "a.pfm"};

    PFM_REQUIRE(pfm::run_cli(8, const_cast<char**>(argv)) != 0);
}

static void run_match_with_required_paths_returns_task_8_failure() {
    const char* argv[] = {
        "pfm",
        "match",
        "--image-a",
        "a.png",
        "--image-b",
        "b.png",
        "--checkpoint",
        "model.pt",
        "--output",
        "matches.json",
    };

    PFM_REQUIRE(pfm::run_cli(10, const_cast<char**>(argv)) != 0);
}

static void run_eval_with_required_paths_returns_task_8_failure() {
    const char* argv[] = {
        "pfm",
        "eval",
        "--pairs",
        "pairs.txt",
        "--checkpoint",
        "model.pt",
        "--output",
        "report.json",
    };

    PFM_REQUIRE(pfm::run_cli(8, const_cast<char**>(argv)) != 0);
}

static void run_train_with_required_paths_fails_without_image_directory() {
    const char* argv[] = {
        "pfm",
        "train",
        "--image-dir",
        "images",
        "--checkpoint",
        "model.pt",
        "--epochs",
        "1",
        "--batch-size",
        "1",
    };

    PFM_REQUIRE(pfm::run_cli(10, const_cast<char**>(argv)) != 0);
}

static void run_export_with_required_paths_fails_without_loadable_checkpoint() {
    const char* argv[] = {"pfm", "export", "--checkpoint", "model.pt", "--output", "exported.pt"};

    PFM_REQUIRE(pfm::run_cli(6, const_cast<char**>(argv)) != 0);
}

void register_cli_tests() {
    register_test("parse_missing_subcommand_throws", parse_missing_subcommand_throws);
    register_test("parse_extract_missing_required_option_throws", parse_extract_missing_required_option_throws);
    register_test("parse_extract_command", parse_extract_command);
    register_test("parse_train_defaults_to_bounded_resize", parse_train_defaults_to_bounded_resize);
    register_test("parse_train_command", parse_train_command);
    register_test("parse_match_command", parse_match_command);
    register_test("parse_eval_command", parse_eval_command);
    register_test("parse_export_command", parse_export_command);
    register_test("parse_min_keypoint_intensity_out_of_range_throws", parse_min_keypoint_intensity_out_of_range_throws);
    register_test("parse_match_invalid_max_keypoints_throws", parse_match_invalid_max_keypoints_throws);
    register_test("top_level_help_lists_subcommand_options", top_level_help_lists_subcommand_options);
    register_test("run_cli_help_returns_zero", run_cli_help_returns_zero);
    register_test(
        "run_extract_without_checkpoint_path_fails_cleanly",
        run_extract_without_checkpoint_path_fails_cleanly
    );
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
