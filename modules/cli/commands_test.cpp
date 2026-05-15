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
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Extract);
    PFM_REQUIRE(parsed.image == "a.png");
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.output == "a.pfm");
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
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Train);
    PFM_REQUIRE(parsed.image_dir == "images");
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.epochs == 7);
    PFM_REQUIRE(parsed.batch_size == 4);
    PFM_REQUIRE(parsed.device == "cuda:0");
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
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Match);
    PFM_REQUIRE(parsed.image_a == "a.png");
    PFM_REQUIRE(parsed.image_b == "b.png");
    PFM_REQUIRE(parsed.max_keypoints == 2048);
    PFM_REQUIRE_CLOSE(parsed.semi_dense_threshold, 0.5, 1.0e-6);
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
    });

    PFM_REQUIRE(parsed.command == pfm::Command::Eval);
    PFM_REQUIRE(parsed.pairs == "pairs.txt");
    PFM_REQUIRE(parsed.checkpoint == "model.pt");
    PFM_REQUIRE(parsed.output == "report.json");
    PFM_REQUIRE(parsed.device == "cuda:1");
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
    register_test("parse_train_command", parse_train_command);
    register_test("parse_match_command", parse_match_command);
    register_test("parse_eval_command", parse_eval_command);
    register_test("parse_export_command", parse_export_command);
    register_test("parse_match_invalid_max_keypoints_throws", parse_match_invalid_max_keypoints_throws);
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
