#include "infer/pipeline.h"

#include <iostream>
#include <string>

namespace pfm {
namespace {

bool require_path(const std::string& value, const char* option_name) {
    if (!value.empty()) {
        return true;
    }

    std::cerr << "missing required option " << option_name << '\n';
    return false;
}

}  // namespace

int run_train_command(const CliOptions& options) {
    if (!require_path(options.image_dir, "--image-dir") || !require_path(options.checkpoint, "--checkpoint")) {
        return 1;
    }

    std::cout << "train command accepted\n";
    return 0;
}

int run_extract_command(const CliOptions& options) {
    if (!require_path(options.image, "--image") || !require_path(options.checkpoint, "--checkpoint") ||
        !require_path(options.output, "--output")) {
        return 1;
    }

    std::cout << "extract command accepted\n";
    return 0;
}

int run_match_command(const CliOptions& options) {
    if (!require_path(options.image_a, "--image-a") || !require_path(options.image_b, "--image-b") ||
        !require_path(options.checkpoint, "--checkpoint") || !require_path(options.output, "--output")) {
        return 1;
    }

    std::cout << "match command accepted\n";
    return 0;
}

int run_eval_command(const CliOptions& options) {
    if (!require_path(options.pairs, "--pairs") || !require_path(options.checkpoint, "--checkpoint") ||
        !require_path(options.output, "--output")) {
        return 1;
    }

    std::cout << "eval command accepted\n";
    return 0;
}

int run_export_command(const CliOptions& options) {
    if (!require_path(options.checkpoint, "--checkpoint") || !require_path(options.output, "--output")) {
        return 1;
    }

    std::cout << "export command accepted\n";
    return 0;
}

}  // namespace pfm
