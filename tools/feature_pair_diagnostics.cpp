#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "feature_io/feature_codec.h"
#include "infer/match_metrics.h"

namespace
{

struct Options
{
    std::string feature_a;
    std::string feature_b;
    std::string warp;
    double threshold = 5.0;
};

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        auto require_value = [&](const char* name) -> std::string
        {
            if (index + 1 >= argc)
            {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if (arg == "--feature-a")
        {
            options.feature_a = require_value("--feature-a");
        }
        else if (arg == "--feature-b")
        {
            options.feature_b = require_value("--feature-b");
        }
        else if (arg == "--warp-a-to-b")
        {
            options.warp = require_value("--warp-a-to-b");
        }
        else if (arg == "--threshold-px")
        {
            options.threshold = std::stod(require_value("--threshold-px"));
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: pfm_feature_pair_diagnostics --feature-a a.pt --feature-b b.pt "
                         "--warp-a-to-b pair.pt [--threshold-px 5]\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.feature_a.empty() || options.feature_b.empty() || options.warp.empty())
    {
        throw std::invalid_argument("feature-a, feature-b, and warp-a-to-b are required");
    }
    return options;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        const auto features_a = pfm::load_feature_set(options.feature_a);
        const auto features_b = pfm::load_feature_set(options.feature_b);
        const auto warp = pfm::load_warp_a_to_b_tensor(options.warp);
        const auto metrics =
            pfm::compute_warp_feature_coverage_metrics(features_a, features_b, warp, options.threshold);
        std::cout << "source_total=" << metrics.source_total << " valid_warp_total=" << metrics.valid_warp_total
                  << " covered_by_target_keypoint=" << metrics.covered_by_target_keypoint
                  << " coverage_fraction=" << metrics.coverage_fraction
                  << " mean_nearest_target_distance_px=" << metrics.mean_nearest_target_distance_pixels
                  << " mean_descriptor_positive_rank=" << metrics.mean_descriptor_positive_rank
                  << " descriptor_top1_accuracy=" << metrics.descriptor_top1_accuracy << '\n';
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "diagnostics failed: " << e.what() << '\n';
        return 1;
    }
}
