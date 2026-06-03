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
    std::string device = "cpu";
    int64_t max_pairs = 8;
    int64_t resize = 512;
    int64_t samples_per_pair = 512;
    int64_t descriptor_dim = 256;
    double min_intensity = 0.01;
    std::string mask_mode = "raw";
};

void printUsage(const char* program)
{
    std::cerr << "usage: " << program << " --cache-dir PATH [--device cpu|cuda] [--max-pairs N] [--resize N]\n"
              << "       [--samples-per-pair N] [--descriptor-dim N] [--min-intensity V]\n"
              << "       [--mask-mode raw|training]\n";
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
        else if (key == "--device")
        {
            args.device = requireValue("--device");
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
        else if (key == "--mask-mode")
        {
            args.mask_mode = requireValue("--mask-mode");
            if (args.mask_mode != "raw" && args.mask_mode != "training")
            {
                throw std::invalid_argument("--mask-mode must be raw or training");
            }
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

torch::Device parseDevice(const std::string& value)
{
    if (value == "cpu")
    {
        return torch::Device(torch::kCPU);
    }
    if (value == "cuda")
    {
        if (!torch::cuda::is_available())
        {
            throw std::invalid_argument("CUDA requested but torch::cuda::is_available() is false");
        }
        return torch::Device(torch::kCUDA);
    }
    throw std::invalid_argument("--device must be cpu or cuda");
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

torch::Tensor makeIntensityMask(const torch::Tensor& image, double min_intensity)
{
    const auto intensity = image.to(torch::kFloat32).mean(0).contiguous();
    auto bright = intensity.ge(min_intensity);
    if (min_intensity <= 0.0 || intensity.size(0) < 7 || intensity.size(1) < 7)
    {
        return bright.to(torch::kBool).contiguous();
    }

    constexpr int64_t kernel = 7;
    auto local_support = torch::nn::functional::avg_pool2d(
        bright.to(torch::kFloat32).reshape({1, 1, intensity.size(0), intensity.size(1)}),
        torch::nn::functional::AvgPool2dFuncOptions({kernel, kernel})
            .stride(1)
            .padding(kernel / 2)
            .count_include_pad(false));
    auto local_mean = torch::nn::functional::avg_pool2d(
        intensity.reshape({1, 1, intensity.size(0), intensity.size(1)}),
        torch::nn::functional::AvgPool2dFuncOptions({kernel, kernel})
            .stride(1)
            .padding(kernel / 2)
            .count_include_pad(false));
    local_support = local_support.reshape({intensity.size(0), intensity.size(1)}).ge(0.25);
    local_mean = local_mean.reshape({intensity.size(0), intensity.size(1)}).ge(min_intensity);
    return bright.logical_and(local_support).logical_and(local_mean).to(torch::kBool).contiguous();
}

torch::Tensor warpMaskToViewB(const torch::Tensor& view_b_mask, const torch::Tensor& warp)
{
    using torch::indexing::Slice;

    auto grid = warp.to(torch::kFloat32).unsqueeze(0).contiguous();
    grid.index_put_({Slice(), Slice(), Slice(), 0},
                    grid.index({Slice(), Slice(), Slice(), 0}) /
                            static_cast<double>(std::max<int64_t>(1, view_b_mask.size(1) - 1)) * 2.0 -
                        1.0);
    grid.index_put_({Slice(), Slice(), Slice(), 1},
                    grid.index({Slice(), Slice(), Slice(), 1}) /
                            static_cast<double>(std::max<int64_t>(1, view_b_mask.size(0) - 1)) * 2.0 -
                        1.0);
    return torch::nn::functional::grid_sample(
               view_b_mask.unsqueeze(0).unsqueeze(0).to(torch::kFloat32), grid,
               torch::nn::functional::GridSampleFuncOptions()
                   .mode(torch::kNearest)
                   .padding_mode(torch::kZeros)
                   .align_corners(true))
        .squeeze(0)
        .squeeze(0)
        .gt(0.0)
        .contiguous();
}

torch::Tensor makeTrainingMask(const torch::Tensor& view_a, const torch::Tensor& view_b, const torch::Tensor& warp,
                               const torch::Tensor& valid_mask, double min_intensity)
{
    const auto mask_a = makeIntensityMask(view_a, min_intensity);
    const auto mask_b = makeIntensityMask(view_b, min_intensity);
    return valid_mask.to(torch::kBool).logical_and(mask_a).logical_and(warpMaskToViewB(mask_b, warp)).contiguous();
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
    auto targets = torch::arange(desc_a.size(0),
                                 torch::TensorOptions().dtype(torch::kLong).device(desc_a.device()));
    auto top1 = similarity.argmax(1).eq(targets).to(torch::kFloat32).mean().item<double>();
    auto sorted = similarity.argsort(1, true);
    auto rank = (sorted.eq(targets.unsqueeze(1)).to(torch::kInt64).argmax(1).to(torch::kFloat32) + 1.0F)
                    .mean()
                    .item<double>();
    auto positive = similarity.diag().mean().item<double>();
    double negative = 0.0;
    if (desc_a.size(0) > 1)
    {
        auto off_diagonal = torch::eye(desc_a.size(0),
                                       torch::TensorOptions().dtype(torch::kBool).device(desc_a.device()))
                                .logical_not();
        negative = similarity.index({off_diagonal}).mean().item<double>();
    }
    return Metrics{top1, positive, negative, rank, desc_a.size(0)};
}

struct ModeMetrics
{
    Metrics texture;
    Metrics learned;
    Metrics fused;
};

Metrics evaluateDescriptorMaps(const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b,
                               const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t image_height_a,
                               int64_t image_width_a, int64_t image_height_b, int64_t image_width_b)
{
    auto feature_a = scalePoints(points_a, image_height_a, image_width_a, descriptors_a.size(2), descriptors_a.size(3))
                         .to(descriptors_a.device());
    auto feature_b = scalePoints(points_b, image_height_b, image_width_b, descriptors_b.size(2), descriptors_b.size(3))
                         .to(descriptors_b.device());
    return descriptorMetrics(sampleDescriptors(descriptors_a, feature_a), sampleDescriptors(descriptors_b, feature_b));
}

ModeMetrics evaluatePair(const pfm::PairArchiveSample& sample, const Args& args, pfm::v21::PfmV21FeatureMatcher& model,
                         const torch::Device& device)
{
    auto view_a = resizeImage(sample.view_a, args.resize);
    auto view_b = resizeImage(sample.view_b, args.resize);
    auto warp = resizeWarp(sample.warp_a_to_b, view_a.size(1), view_a.size(2));
    auto valid_mask = resizeMask(sample.valid_mask, view_a.size(1), view_a.size(2));
    if (args.mask_mode == "training")
    {
        valid_mask = makeTrainingMask(view_a, view_b, warp, valid_mask, args.min_intensity);
    }
    auto warp_x = warp.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    auto warp_y = warp.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
    auto valid = valid_mask.logical_and(warp_x.ge(0.0))
                     .logical_and(warp_x.le(static_cast<double>(view_b.size(2) - 1)))
                     .logical_and(warp_y.ge(0.0))
                     .logical_and(warp_y.le(static_cast<double>(view_b.size(1) - 1)));
    auto valid_indices = torch::nonzero(valid.reshape({view_a.size(1) * view_a.size(2)})).flatten();
    if (valid_indices.numel() == 0)
    {
        return ModeMetrics{};
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
        return ModeMetrics{};
    }
    const auto take = std::min<int64_t>(args.samples_per_pair, points_a.size(0));
    auto order = torch::randperm(points_a.size(0), torch::TensorOptions().dtype(torch::kLong)).narrow(0, 0, take);
    points_a = points_a.index_select(0, order);
    points_b = points_b.index_select(0, order);

    torch::NoGradGuard no_grad;
    auto batch_a = view_a.unsqueeze(0).to(device);
    auto batch_b = view_b.unsqueeze(0).to(device);
    auto texture_a = model->textureDescriptorMapSingle(batch_a);
    auto texture_b = model->textureDescriptorMapSingle(batch_b);
    auto learned_a = model->learnedDescriptorMapSingle(batch_a);
    auto learned_b = model->learnedDescriptorMapSingle(batch_b);
    auto fused_a = model->fuseDescriptorMaps(learned_a, batch_a, 1.0);
    auto fused_b = model->fuseDescriptorMaps(learned_b, batch_b, 1.0);
    points_a = points_a.to(device);
    points_b = points_b.to(device);

    return ModeMetrics{
        evaluateDescriptorMaps(texture_a, texture_b, points_a, points_b, view_a.size(1), view_a.size(2), view_b.size(1),
                               view_b.size(2)),
        evaluateDescriptorMaps(learned_a, learned_b, points_a, points_b, view_a.size(1), view_a.size(2), view_b.size(1),
                               view_b.size(2)),
        evaluateDescriptorMaps(fused_a, fused_b, points_a, points_b, view_a.size(1), view_a.size(2), view_b.size(1),
                               view_b.size(2))};
}

void accumulate(Metrics& total, const Metrics& value)
{
    total.top1 += value.top1;
    total.positive += value.positive;
    total.negative += value.negative;
    total.rank += value.rank;
    total.count += value.count;
}

void printAveragedMetrics(const char* name, const Metrics& total, int64_t valid_pairs)
{
    std::cout << "mode=" << name << " pairs=" << valid_pairs << " points=" << total.count
              << " top1=" << total.top1 / static_cast<double>(valid_pairs)
              << " rank=" << total.rank / static_cast<double>(valid_pairs)
              << " positive=" << total.positive / static_cast<double>(valid_pairs)
              << " negative=" << total.negative / static_cast<double>(valid_pairs)
              << " margin=" << (total.positive - total.negative) / static_cast<double>(valid_pairs) << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto args = parseArgs(argc, argv);
        const auto device = parseDevice(args.device);
        auto paths = pfm::discoverPairArchivePaths(args.cache_dir);
        if (paths.empty())
        {
            throw std::invalid_argument("no pair_*.pt archives found");
        }
        if (static_cast<int64_t>(paths.size()) > args.max_pairs)
        {
            paths.resize(static_cast<std::size_t>(args.max_pairs));
        }
        pfm::v21::PfmV21Config config;
        config.descriptor_dim = args.descriptor_dim;
        auto model = pfm::v21::PfmV21FeatureMatcher(config);
        model->to(device);
        model->eval();
        Metrics texture;
        Metrics learned;
        Metrics fused;
        int64_t valid_pairs = 0;
        for (const auto& path : paths)
        {
            auto metrics = evaluatePair(pfm::loadPairArchiveSample(path), args, model, device);
            if (metrics.texture.count == 0)
            {
                continue;
            }
            accumulate(texture, metrics.texture);
            accumulate(learned, metrics.learned);
            accumulate(fused, metrics.fused);
            ++valid_pairs;
        }
        if (valid_pairs == 0)
        {
            throw std::runtime_error("no valid textured correspondences found");
        }
        printAveragedMetrics("texture", texture, valid_pairs);
        printAveragedMetrics("learned", learned, valid_pairs);
        printAveragedMetrics("fused", fused, valid_pairs);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
