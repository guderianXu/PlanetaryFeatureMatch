#pragma once

#include "cli/commands.h"

namespace pfm {

/// Run the training command.
/// @param options Parsed CLI options containing image_dir, checkpoint, device, epochs, and batch_size.
/// @return Zero after writing a loadable checkpoint, nonzero when validation or training fails.
int run_train_command(const CliOptions& options);

/// Run the feature extraction command.
/// @param options Parsed CLI options containing image, checkpoint, output, device, and decode settings.
/// @return Zero after writing a loadable feature file, nonzero when validation or extraction fails.
int run_extract_command(const CliOptions& options);

/// Validate image matching command paths and report that matching is deferred to Task 8.
/// @param options Parsed CLI options containing image_a, image_b, checkpoint, and output.
/// @return Always nonzero after required paths are present because matching is deferred.
int run_match_command(const CliOptions& options);

/// Validate evaluation command paths and report that evaluation is deferred to Task 8.
/// @param options Parsed CLI options containing pairs, checkpoint, and output.
/// @return Always nonzero after required paths are present because evaluation is deferred.
int run_eval_command(const CliOptions& options);

/// Run the model export command.
/// @param options Parsed CLI options containing checkpoint and output.
/// @return Zero after writing a loadable checkpoint copy, nonzero when validation or export fails.
int run_export_command(const CliOptions& options);

}  // namespace pfm
