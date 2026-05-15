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

/// Run the image matching command.
/// @param options Parsed CLI options containing image_a, image_b, checkpoint, output, device, and decode settings.
/// @return Zero after writing a loadable match file, nonzero when validation, extraction, or matching fails.
int run_match_command(const CliOptions& options);

/// Run the image-pair evaluation command.
/// @param options Parsed CLI options containing pairs, checkpoint, output, device, and decode settings.
/// @return Zero after writing a LibTorch report archive, nonzero when validation, extraction, or matching fails.
int run_eval_command(const CliOptions& options);

/// Run the model export command.
/// @param options Parsed CLI options containing checkpoint and output.
/// @return Zero after writing a loadable checkpoint copy, nonzero when validation or export fails.
int run_export_command(const CliOptions& options);

}  // namespace pfm
