#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "data/pair_archive_dataset.h"
#include "models/pfm_model_v21.h"

namespace
{

struct Args
{
    std::filesystem::path cache_dir;
    int64_t max_pairs = 8;
    int64_t resize = 512;
    int64_t samples_per_pair = 512;
    int64_t descriptor_dim = 256;
    double min_intensity = 0.01;
};

void printUsage(const char* program)
{
    std::cerr << "usage: " << program << " --cache-dir PATH [--max-pairs N] [--resize N]\n"
              << "       [--samples-per-pair N] [--descriptor-dim N] [--min-intensity V]\n";
}

int64_t parsePositiveInt64(const std::string& value, const char* name)
{
    const auto parsed = std::stoll(value);
    if (parsed <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return parsed;
}

double parseNonNegativeDouble(const std::string& value, const char* name)
{
    const auto parsed = std::stod(value);
    if (!std::isfinite(parsed) || parsed < 0.0)
    {
        throw std::invalid_argument(std::string(name) + " must be non-negative and finite");
    }
    return parsed;
}

Args parseArgs(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        const std::string key = argv[i];
        auto requireValue = [&](const char* name) -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[++i];
        };

        if (key == "--cache-dir")
        {
            args.cache_dir = requireValue("--cache-dir");
        }
        else if (key == "--max-pairs")
        {
            args.max_pairs = parsePositiveInt64(requireValue("--max-pairs"), "--max-pairs");
        }
        else if (key == "--resize")
        {
            args.resize = parsePositiveInt64(requireValue("--resize"), "--resize");
        }
        else if (key == "--samples-per-pair")
        {
            args.samples_per_pair = parsePositiveInt64(requireValue("--samples-per-pair"), "--samples-per-pair");
        }
        else if (key == "--descriptor-dim")
        {
            args.descriptor_dim = parsePositiveInt64(requireValue("--descriptor-dim"), "--descriptor-dim");
        }
        else if (key == "--min-intensity")
        {
            args.min_intensity = parseNonNegativeDouble(requireValue("--min-intensity"), "--min-intensity");
        }
        else if (key == "--help" || key == "-h")
        {
            printUsage(argv[0]);
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.cache_dir.empty())
    {
        throw std::invalid_argument("--cache-dir is required");
    }
    return args;
}

torch::Tensor resizeImage(const torch::Tensor& image, int64_t resize)
{
    const auto height = image.size(1);
    const auto width = image.size(2);
    const auto max_edge = std::max(height, width);
    if (max_edge <= resize)
    {
        return image.contiguous();
    }
    const double scale = static_cast<double>(resize) / static_cast<double>(max_edge);
    const auto target_height =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(height) * scale)));
    const auto target_width =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(width) * scale)));
    return torch::nn::functional::interpolate(image.unsqueeze(0),
                                              torch::nn::functional::InterpolateFuncOptions()
                                                  .size(std::vector<int64_t>{target_height, target_width})
                                                  .mode(torch::kBilinear)
                                                  .align_corners(false))
        .squeeze(0)
        .contiguous();
}

torch::Tensor resizeWarp(const torch::Tensor& warp, int64_t target_height, int64_t target_width)
{
    const auto source_height = warp.size(0);
    const auto source_width = warp.size(1);
    if (source_height == target_height && source_width == target_width)
    {
        return warp.contiguous();
    }
    auto resized = torch::nn::functional::interpolate(warp.permute({2, 0, 1}).unsqueeze(0),
                                                      torch::nn::functional::InterpolateFuncOptions()
                                                          .size(std::vector<int64_t>{target_height, target_width})
                                                          .mode(torch::kBilinear)
                                                          .align_corners(true))
                       .squeeze(0)
                       .permute({1, 2, 0})
                       .contiguous();
    resized.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0},
                       resized.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) *
                           (static_cast<double>(target_width - 1) /
                            static_cast<double>(std::max<int64_t>(1, source_width - 1))));
    resized.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 1},
                       resized.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) *
                           (static_cast<double>(target_height - 1) /
                            static_cast<double>(std::max<int64_t>(1, source_height - 1))));
    return resized;
}

torch::Tensor resizeMask(const torch::Tensor& valid_mask, int64_t target_height, int64_t target_width)
{
    if (valid_mask.size(0) == target_height && valid_mask.size(1) == target_width)
    {
        return valid_mask.to(torch::kBool).contiguous();
    }
    return torch::nn::functional::interpolate(valid_mask.to(torch::kFloat32).unsqueeze(0).unsqueeze(0),
                                              torch::nn::functional::InterpolateFuncOptions()
                                                  .size(std::vector<int64_t>{target_height, target_width})
                                                  .mode(torch::kArea))
        .squeeze(0)
        .squeeze(0)
        .gt(0.0)
        .contiguous();
}

torch::Tensor centerIntensityForPoints(const torch::Tensor& image, const torch::Tensor& points)
{
    if (points.numel() == 0)
    {
        return image.new_empty({0});
    }
    auto intensity = image.mean(0);
    const auto height = intensity.size(0);
    const auto width = intensity.size(1);
    auto rounded = points.round().to(torch::kLong);
    auto x = rounded.index({torch::indexing::Slice(), 0}).clamp(0, width - 1);
    auto y = rounded.index({torch::indexing::Slice(), 1}).clamp(0, height - 1);
    return intensity.index({y, x});
}

torch::Tensor scalePoints(const torch::Tensor& points, int64_t image_height, int64_t image_width, int64_t feature_height,
                          int64_t feature_width)
{
    auto x = points.index({torch::indexing::Slice(), 0}) *
             (static_cast<double>(std::max<int64_t>(1, feature_width - 1)) /
              static_cast<double>(std::max<int64_t>(1, image_width - 1)));
    auto y = points.index({torch::indexing::Slice(), 1}) *
             (static_cast<double>(std::max<int64_t>(1, feature_height - 1)) /
              static_cast<double>(std::max<int64_t>(1, image_height - 1)));
    return torch::stack({x, y}, 1).contiguous();
}

torch::Tensor sampleDescriptors(const torch::Tensor& descriptor_map, const torch::Tensor& points)
{
    const auto height = descriptor_map.size(2);
    const auto width = descriptor_map.size(3);
    auto x = points.index({torch::indexing::Slice(), 0});
    auto y = points.index({torch::indexing::Slice(), 1});
    auto grid_x = width > 1 ? x / static_cast<double>(width - 1) * 2.0 - 1.0 : torch::zeros_like(x);
    auto grid_y = height > 1 ? y / static_cast<double>(height - 1) * 2.0 - 1.0 : torch::zeros_like(y);
    auto grid = torch::stack({grid_x, grid_y}, 1).reshape({1, points.size(0), 1, 2}).contiguous();
    return torch::nn::functional::grid_sample(descriptor_map, grid,
                                              torch::nn::functional::GridSampleFuncOptions()
                                                  .mode(torch::kBilinear)
                                                  .padding_mode(torch::kZeros)
                                                  .align_corners(true))
        .squeeze(0)
        .squeeze(-1)
        .transpose(0, 1)
        .contiguous();
}

struct Metrics
{
    double top1 = 0.0;
    double positive = 0.0;
    double negative = 0.0;
    double rank = 0.0;
    int64_t count = 0;
};

Metrics descriptorMetrics(const torch::Tensor& desc_a, const torch::Tensor& desc_b)
{
    auto normalized_a = desc_a / desc_a.norm(2, 1, true).clamp_min(1.0e-3);
    auto normalized_b = desc_b / desc_b.norm(2, 1, true).clamp_min(1.0e-3);
    auto similarity = torch::matmul(normalized_a, normalized_b.transpose(0, 1));
    auto targets = torch::arange(desc_a.size(0), torch::TensorOptions().dtype(torch::kLong));
    auto top1 = similarity.argmax(1).eq(targets).to(torch::kFloat32).mean().item<double>();
    auto sorted = similarity.argsort(1, true);
    auto rank = (sorted.eq(targets.unsqueeze(1)).to(torch::kInt64).argmax(1).to(torch::kFloat32) + 1.0F)
                    .mean()
                    .item<double>();
    auto positive = similarity.diag().mean().item<double>();
    double negative = 0.0;
    if (desc_a.size(0) > 1)
    {
        auto off_diagonal = torch::eye(desc_a.size(0), torch::TensorOptions().dtype(torch::kBool)).logical_not();
        negative = similarity.index({off_diagonal}).mean().item<double>();
    }
    return Metrics{top1, positive, negative, rank, desc_a.size(0)};
}

Metrics evaluatePair(const pfm::PairArchiveSample& sample, const Args& args)
{
    auto view_a = resizeImage(sample.view_a, args.resize);
    auto view_b = resizeImage(sample.view_b, args.resize);
    auto warp = resizeWarp(sample.warp_a_to_b, view_a.size(1), view_a.size(2));
    auto valid_mask = resizeMask(sample.valid_mask, view_a.size(1), view_a.size(2));
    auto warp_x = warp.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    auto warp_y = warp.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
    auto valid = valid_mask.logical_and(warp_x.ge(0.0))
                     .logical_and(warp_x.le(static_cast<double>(view_b.size(2) - 1)))
                     .logical_and(warp_y.ge(0.0))
                     .logical_and(warp_y.le(static_cast<double>(view_b.size(1) - 1)));
    auto valid_indices = torch::nonzero(valid.reshape({view_a.size(1) * view_a.size(2)})).flatten();
    if (valid_indices.numel() == 0)
    {
        return Metrics{};
    }
    auto y = torch::floor_divide(valid_indices, view_a.size(2)).to(torch::kFloat32);
    auto x = valid_indices.remainder(view_a.size(2)).to(torch::kFloat32);
    auto points_a = torch::stack({x, y}, 1).contiguous();
    auto points_b = warp.reshape({view_a.size(1) * view_a.size(2), 2}).index_select(0, valid_indices).contiguous();
    auto textured =
        centerIntensityForPoints(view_a, points_a).gt(args.min_intensity).logical_and(
            centerIntensityForPoints(view_b, points_b).gt(args.min_intensity));
    points_a = points_a.index({textured});
    points_b = points_b.index({textured});
    if (points_a.numel() == 0)
    {
        return Metrics{};
    }
    const auto take = std::min<int64_t>(args.samples_per_pair, points_a.size(0));
    auto order = torch::randperm(points_a.size(0), torch::TensorOptions().dtype(torch::kLong)).narrow(0, 0, take);
    points_a = points_a.index_select(0, order);
    points_b = points_b.index_select(0, order);
    auto texture_a =
        pfm::v21::makeRotationInvariantTextureDescriptor(view_a.unsqueeze(0), view_a.size(1) / 4, view_a.size(2) / 4,
                                                         args.descriptor_dim);
    auto texture_b =
        pfm::v21::makeRotationInvariantTextureDescriptor(view_b.unsqueeze(0), view_b.size(1) / 4, view_b.size(2) / 4,
                                                         args.descriptor_dim);
    auto feature_a = scalePoints(points_a, view_a.size(1), view_a.size(2), texture_a.size(2), texture_a.size(3));
    auto feature_b = scalePoints(points_b, view_b.size(1), view_b.size(2), texture_b.size(2), texture_b.size(3));
    return descriptorMetrics(sampleDescriptors(texture_a, feature_a), sampleDescriptors(texture_b, feature_b));
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto args = parseArgs(argc, argv);
        auto paths = pfm::discoverPairArchivePaths(args.cache_dir);
        if (paths.empty())
        {
            throw std::invalid_argument("no pair_*.pt archives found");
        }
        if (static_cast<int64_t>(paths.size()) > args.max_pairs)
        {
            paths.resize(static_cast<std::size_t>(args.max_pairs));
        }
        double top1 = 0.0;
        double positive = 0.0;
        double negative = 0.0;
        double rank = 0.0;
        int64_t total_points = 0;
        int64_t valid_pairs = 0;
        for (const auto& path : paths)
        {
            auto metrics = evaluatePair(pfm::loadPairArchiveSample(path), args);
            if (metrics.count == 0)
            {
                continue;
            }
            top1 += metrics.top1;
            positive += metrics.positive;
            negative += metrics.negative;
            rank += metrics.rank;
            total_points += metrics.count;
            ++valid_pairs;
        }
        if (valid_pairs == 0)
        {
            throw std::runtime_error("no valid textured correspondences found");
        }
        std::cout << "pairs=" << valid_pairs << " points=" << total_points << " top1=" << top1 / valid_pairs
                  << " rank=" << rank / valid_pairs << " positive=" << positive / valid_pairs
                  << " negative=" << negative / valid_pairs << " margin=" << (positive - negative) / valid_pairs
                  << "\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
