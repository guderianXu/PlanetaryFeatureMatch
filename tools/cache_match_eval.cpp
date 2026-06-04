#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/serialize.h>
#include <torch/torch.h>

#include "data/pair_archive_dataset.h"
#include "infer/cache_match_eval.h"
#include "models/pfm_model_v21.h"

namespace
{

struct Args
{
    std::vector<std::filesystem::path> cache_dirs;
    std::filesystem::path checkpoint;
    std::filesystem::path output;
    std::string device = "cuda";
    int64_t limit_pairs = 0;
    int64_t max_keypoints = 4096;
    int64_t max_matches = 512;
    double min_intensity = 0.01;
    double threshold_px = 5.0;
    double texture_blend_weight = pfm::v21::INFERENCE_TEXTURE_BLEND_WEIGHT;
};

void printUsage(const char* program)
{
    std::cerr << "usage: " << program << " --cache-dir PATH [--cache-dir PATH ...] --checkpoint model.pt "
              << "--output metrics.csv [--device cpu|cuda] [--limit-pairs N] [--max-keypoints N] "
              << "[--max-matches N] [--min-intensity V] [--threshold-px V] [--texture-blend-weight V]\n";
}

std::string requireValue(int argc, char** argv, int& index)
{
    if (index + 1 >= argc)
    {
        throw std::invalid_argument(std::string(argv[index]) + " requires a value");
    }
    return argv[++index];
}

int64_t parseNonNegativeInt64(const std::string& value, const char* name)
{
    char* end = nullptr;
    const auto parsed = std::strtoll(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed < 0)
    {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    return parsed;
}

int64_t parsePositiveInt64(const std::string& value, const char* name)
{
    const auto parsed = parseNonNegativeInt64(value, name);
    if (parsed <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return parsed;
}

double parseFiniteDouble(const std::string& value, const char* name)
{
    char* end = nullptr;
    const auto parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0' || !std::isfinite(parsed))
    {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
    return parsed;
}

Args parseArgs(int argc, char** argv)
{
    Args args;
    for (int index = 1; index < argc; ++index)
    {
        const std::string option = argv[index];
        if (option == "--cache-dir")
        {
            args.cache_dirs.push_back(requireValue(argc, argv, index));
        }
        else if (option == "--checkpoint")
        {
            args.checkpoint = requireValue(argc, argv, index);
        }
        else if (option == "--output")
        {
            args.output = requireValue(argc, argv, index);
        }
        else if (option == "--device")
        {
            args.device = requireValue(argc, argv, index);
        }
        else if (option == "--limit-pairs")
        {
            args.limit_pairs = parseNonNegativeInt64(requireValue(argc, argv, index), "--limit-pairs");
        }
        else if (option == "--max-keypoints")
        {
            args.max_keypoints = parsePositiveInt64(requireValue(argc, argv, index), "--max-keypoints");
        }
        else if (option == "--max-matches")
        {
            args.max_matches = parsePositiveInt64(requireValue(argc, argv, index), "--max-matches");
        }
        else if (option == "--min-intensity")
        {
            args.min_intensity = parseFiniteDouble(requireValue(argc, argv, index), "--min-intensity");
        }
        else if (option == "--threshold-px")
        {
            args.threshold_px = parseFiniteDouble(requireValue(argc, argv, index), "--threshold-px");
        }
        else if (option == "--texture-blend-weight")
        {
            args.texture_blend_weight = parseFiniteDouble(requireValue(argc, argv, index), "--texture-blend-weight");
        }
        else if (option == "--help" || option == "-h")
        {
            printUsage(argv[0]);
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown argument: " + option);
        }
    }
    if (args.cache_dirs.empty())
    {
        throw std::invalid_argument("--cache-dir is required");
    }
    if (args.checkpoint.empty())
    {
        throw std::invalid_argument("--checkpoint is required");
    }
    if (args.output.empty())
    {
        throw std::invalid_argument("--output is required");
    }
    if (args.min_intensity < 0.0)
    {
        throw std::invalid_argument("--min-intensity must be non-negative");
    }
    if (args.threshold_px < 0.0)
    {
        throw std::invalid_argument("--threshold-px must be non-negative");
    }
    if (args.texture_blend_weight < 0.0)
    {
        throw std::invalid_argument("--texture-blend-weight must be non-negative");
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

int64_t readConfigValue(torch::serialize::InputArchive& archive, const char* name)
{
    torch::Tensor tensor;
    archive.read(name, tensor);
    if (!tensor.defined() || tensor.numel() != 1)
    {
        throw std::invalid_argument(std::string("checkpoint config missing ") + name);
    }
    return tensor.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
}

int64_t readOptionalConfigValue(torch::serialize::InputArchive& archive, const char* name, int64_t fallback)
{
    try
    {
        return readConfigValue(archive, name);
    }
    catch (const c10::Error&)
    {
        return fallback;
    }
}

pfm::v21::PfmV21Config readCheckpointConfig(const std::filesystem::path& checkpoint)
{
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint.string());
    torch::serialize::InputArchive config_archive;
    archive.read("config", config_archive);

    pfm::v21::PfmV21Config config;
    config.input_channels = readConfigValue(config_archive, "input_channels");
    config.base_channels = readConfigValue(config_archive, "base_channels");
    config.descriptor_dim = readConfigValue(config_archive, "descriptor_dim");
    config.graph_hidden_dim =
        readOptionalConfigValue(config_archive, "graph_hidden_dim", std::max<int64_t>(32, config.descriptor_dim));
    config.graph_attention_layers = readOptionalConfigValue(config_archive, "graph_attention_layers", 1);
    config.graph_keypoint_meta_dim = readOptionalConfigValue(config_archive, "graph_keypoint_meta_dim", 16);
    return config;
}

pfm::v21::PfmV21FeatureMatcher loadModel(const std::filesystem::path& checkpoint, torch::Device device)
{
    const auto config = readCheckpointConfig(checkpoint);
    auto model = pfm::v21::PfmV21FeatureMatcher(config);
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint.string());
    model->load(archive);
    model->to(device);
    model->eval();
    return model;
}

torch::Tensor adaptImageChannels(const torch::Tensor& image, int64_t input_channels)
{
    if (image.size(0) == input_channels)
    {
        return image.contiguous();
    }
    if (input_channels == 1)
    {
        return image.to(torch::kFloat32).mean(0, true).contiguous();
    }
    if (image.size(0) == 1)
    {
        return image.repeat({input_channels, 1, 1}).contiguous();
    }
    throw std::invalid_argument("image channel count cannot be adapted to checkpoint input_channels");
}

std::vector<std::filesystem::path> discoverPairs(const Args& args)
{
    std::vector<std::filesystem::path> paths;
    for (const auto& cache_dir : args.cache_dirs)
    {
        auto discovered = pfm::discoverPairArchivePaths(cache_dir);
        paths.insert(paths.end(), discovered.begin(), discovered.end());
    }
    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
    if (args.limit_pairs > 0 && static_cast<int64_t>(paths.size()) > args.limit_pairs)
    {
        paths.resize(static_cast<std::size_t>(args.limit_pairs));
    }
    if (paths.empty())
    {
        throw std::invalid_argument("no pair_*.pt archives found");
    }
    return paths;
}

void ensureOutputParent(const std::filesystem::path& output)
{
    const auto parent = output.parent_path();
    if (!parent.empty())
    {
        std::filesystem::create_directories(parent);
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto args = parseArgs(argc, argv);
        const auto device = parseDevice(args.device);
        const auto config = readCheckpointConfig(args.checkpoint);
        auto model = loadModel(args.checkpoint, device);
        const auto pairs = discoverPairs(args);
        ensureOutputParent(args.output);

        pfm::PythonDescriptorGridConfig keypoint_config;
        keypoint_config.max_keypoints = args.max_keypoints;
        keypoint_config.min_intensity = args.min_intensity;

        std::ofstream csv(args.output);
        if (!csv)
        {
            throw std::invalid_argument("failed to open output CSV: " + args.output.string());
        }
        csv << "pair_pt,matches,correct,wrong,precision\n";
        csv << std::fixed << std::setprecision(6);

        int64_t total_matches = 0;
        int64_t total_correct = 0;
        {
            torch::NoGradGuard no_grad;
            for (std::size_t index = 0; index < pairs.size(); ++index)
            {
                auto pair = pfm::loadPairArchiveSample(pairs[index], false);
                auto image_a = adaptImageChannels(pair.view_a, config.input_channels).unsqueeze(0).to(device);
                auto image_b = adaptImageChannels(pair.view_b, config.input_channels).unsqueeze(0).to(device);
                auto raw_a = model->forwardSingle(image_a, args.texture_blend_weight);
                auto raw_b = model->forwardSingle(image_b, args.texture_blend_weight);
                const auto result = pfm::evaluatePythonRawMutualDescriptorMaps(
                    pair, raw_a.descriptors.detach().to(torch::kCPU).contiguous(),
                    raw_b.descriptors.detach().to(torch::kCPU).contiguous(), keypoint_config, args.max_matches,
                    args.threshold_px);
                total_matches += result.matches;
                total_correct += result.correct;
                csv << pairs[index].string() << ',' << result.matches << ',' << result.correct << ',' << result.wrong
                    << ',' << result.precision << '\n';
                csv.flush();
                std::cout << '[' << (index + 1) << '/' << pairs.size() << "] " << pairs[index]
                          << " matches=" << result.matches << " correct=" << result.correct
                          << " precision=" << result.precision << '\n';
            }
        }

        const auto precision =
            total_matches == 0 ? 0.0 : static_cast<double>(total_correct) / static_cast<double>(total_matches);
        std::cout << "summary: pairs=" << pairs.size() << " total_matches=" << total_matches
                  << " correct=" << total_correct << " precision=" << precision << " output=" << args.output
                  << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "pfm_cache_match_eval failed: " << error.what() << '\n';
        return 1;
    }
}
