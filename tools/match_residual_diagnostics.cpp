#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "feature_io/feature_codec.h"
#include "feature_io/match_codec.h"
#include "infer/match_metrics.h"

namespace
{

struct Options
{
    std::string feature_a;
    std::string feature_b;
    std::string matches;
    std::string warp;
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
        else if (arg == "--matches")
        {
            options.matches = require_value("--matches");
        }
        else if (arg == "--warp-a-to-b")
        {
            options.warp = require_value("--warp-a-to-b");
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: pfm_match_residual_diagnostics --feature-a a.pt --feature-b b.pt "
                         "--matches matches.pt --warp-a-to-b pair.pt\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.feature_a.empty() || options.feature_b.empty() || options.matches.empty() || options.warp.empty())
    {
        throw std::invalid_argument("feature-a, feature-b, matches, and warp-a-to-b are required");
    }
    return options;
}

std::pair<double, double> map_feature_point_to_image_float(const torch::Tensor& point, int64_t map_width,
                                                           int64_t map_height, int64_t image_width,
                                                           int64_t image_height)
{
    const auto scale_x = static_cast<double>(image_width) / static_cast<double>(std::max<int64_t>(1, map_width));
    const auto scale_y = static_cast<double>(image_height) / static_cast<double>(std::max<int64_t>(1, map_height));
    const auto x = (static_cast<double>(point.index({0}).item<float>()) + 0.5) * scale_x - 0.5;
    const auto y = (static_cast<double>(point.index({1}).item<float>()) + 0.5) * scale_y - 0.5;
    return {std::min<double>(image_width - 1, std::max<double>(0.0, x)),
            std::min<double>(image_height - 1, std::max<double>(0.0, y))};
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        const auto features_a = pfm::load_feature_set(options.feature_a);
        const auto features_b = pfm::load_feature_set(options.feature_b);
        const auto matches = pfm::load_match_set(options.matches);
        const auto warp = pfm::load_warp_a_to_b_tensor(options.warp).to(torch::kCPU, torch::kFloat32).contiguous();
        auto sparse = matches.sparse_matches.to(torch::kCPU, torch::kLong).contiguous();
        auto scores = matches.sparse_scores.to(torch::kCPU, torch::kFloat32).contiguous();
        auto keypoints_a = features_a.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
        auto keypoints_b = features_b.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();

        struct Row
        {
            int64_t row = 0;
            int64_t ia = 0;
            int64_t ib = 0;
            double score = 0.0;
            double residual = 0.0;
            bool correct = false;
            double ax = 0.0;
            double ay = 0.0;
            double bx = 0.0;
            double by = 0.0;
        };
        std::vector<Row> rows;
        rows.reserve(static_cast<std::size_t>(sparse.size(0)));
        for (int64_t row = 0; row < sparse.size(0); ++row)
        {
            const auto ia = sparse.index({row, 0}).item<int64_t>();
            const auto ib = sparse.index({row, 1}).item<int64_t>();
            if (ia < 0 || ia >= keypoints_a.size(0) || ib < 0 || ib >= keypoints_b.size(0))
            {
                continue;
            }
            const auto [ax, ay] =
                map_feature_point_to_image_float(keypoints_a[ia], features_a.feature_map_width,
                                                 features_a.feature_map_height, warp.size(1), warp.size(0));
            const auto [bx, by] =
                map_feature_point_to_image_float(keypoints_b[ib], features_b.feature_map_width,
                                                 features_b.feature_map_height, warp.size(1), warp.size(0));
            const auto sx = std::min<int64_t>(warp.size(1) - 1, std::max<int64_t>(0, std::llround(ax)));
            const auto sy = std::min<int64_t>(warp.size(0) - 1, std::max<int64_t>(0, std::llround(ay)));
            const auto expected_x = warp.index({sy, sx, 0}).item<float>();
            const auto expected_y = warp.index({sy, sx, 1}).item<float>();
            const auto dx = bx - static_cast<double>(expected_x);
            const auto dy = by - static_cast<double>(expected_y);
            const auto residual = std::hypot(dx, dy);
            rows.push_back(
                Row{row, ia, ib, scores.index({row}).item<float>(), residual, residual <= 5.0, ax, ay, bx, by});
        }
        std::sort(rows.begin(), rows.end(),
                  [](const Row& lhs, const Row& rhs)
                  {
                      return lhs.residual < rhs.residual;
                  });
        for (const auto& row : rows)
        {
            std::cout << "row=" << row.row << " ia=" << row.ia << " ib=" << row.ib << " score=" << row.score
                      << " residual=" << row.residual << " correct=" << (row.correct ? 1 : 0) << " a=(" << row.ax << ","
                      << row.ay << ")"
                      << " b=(" << row.bx << "," << row.by << ")\n";
        }
        std::sort(rows.begin(), rows.end(),
                  [](const Row& lhs, const Row& rhs)
                  {
                      return lhs.score > rhs.score;
                  });
        int64_t prefix_correct = 0;
        for (std::size_t index = 0; index < rows.size(); ++index)
        {
            if (rows[index].correct)
            {
                ++prefix_correct;
            }
            const auto prefix_total = static_cast<int64_t>(index + 1);
            std::cerr << "score_prefix rank=" << prefix_total << " row=" << rows[index].row
                      << " score=" << rows[index].score << " correct=" << (rows[index].correct ? 1 : 0)
                      << " prefix_correct=" << prefix_correct << " prefix_wrong=" << (prefix_total - prefix_correct)
                      << " prefix_precision=" << static_cast<double>(prefix_correct) / static_cast<double>(prefix_total)
                      << '\n';
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "match residual diagnostics failed: " << e.what() << '\n';
        return 1;
    }
}
