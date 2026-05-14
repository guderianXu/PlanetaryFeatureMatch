#pragma once

#include "cli/commands.h"

namespace pfm {

/// Run the training command validation stub.
/// @param options Parsed CLI options containing image_dir and checkpoint.
/// @return Zero when required paths are present, nonzero otherwise.
int run_train_command(const CliOptions& options);

/// Run the feature extraction command validation stub.
/// @param options Parsed CLI options containing image, checkpoint, and output.
/// @return Zero when required paths are present, nonzero otherwise.
int run_extract_command(const CliOptions& options);

/// Run the image matching command validation stub.
/// @param options Parsed CLI options containing image_a, image_b, checkpoint, and output.
/// @return Zero when required paths are present, nonzero otherwise.
int run_match_command(const CliOptions& options);

/// Run the evaluation command validation stub.
/// @param options Parsed CLI options containing pairs, checkpoint, and output.
/// @return Zero when required paths are present, nonzero otherwise.
int run_eval_command(const CliOptions& options);

/// Run the model export command validation stub.
/// @param options Parsed CLI options containing checkpoint and output.
/// @return Zero when required paths are present, nonzero otherwise.
int run_export_command(const CliOptions& options);

}  // namespace pfm
