#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
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
    std::filesystem::path dataset_root;
    std::filesystem::path cache_dir;
    std::filesystem::path output_checkpoint;
    std::vector<std::filesystem::path> pairs;
    std::string device = "cpu";
    int64_t max_pairs = 2;
    int64_t max_image_size = 128;
    int64_t input_channels = 1;
    int64_t base_channels = 8;
    int64_t descriptor_dim = 16;
    int64_t graph_hidden_dim = 32;
    int64_t graph_attention_layers = 1;
    int64_t graph_keypoint_meta_dim = 16;
    double learning_rate = 1.0e-3;
};

void printUsage(const char* program)
{
    std::cerr << "usage: " << program << " [--dataset-root PATH | --cache-dir PATH | --pair PATH ...]\n"
              << "       [--max-pairs N] [--max-image-size N] [--device cpu|cuda] [--learning-rate LR]\n"
              << "       [--input-channels N] [--base-channels N] [--descriptor-dim N]\n"
              << "       [--graph-hidden-dim N] [--graph-attention-layers N] [--graph-keypoint-meta-dim N]\n"
              << "       [--full-v21]\n"
              << "       [--output-checkpoint PATH]\n";
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

double parsePositiveDouble(const std::string& value, const char* name)
{
    const auto parsed = std::stod(value);
    if (!(parsed > 0.0) || !std::isfinite(parsed))
    {
        throw std::invalid_argument(std::string(name) + " must be positive and finite");
    }
    return parsed;
}

Args parseArgs(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        const std::string key = argv[i];
        auto require_value = [&](const char* name) -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[++i];
        };

        if (key == "--dataset-root")
        {
            args.dataset_root = require_value("--dataset-root");
        }
        else if (key == "--cache-dir")
        {
            args.cache_dir = require_value("--cache-dir");
        }
        else if (key == "--pair")
        {
            args.pairs.push_back(require_value("--pair"));
        }
        else if (key == "--max-pairs")
        {
            args.max_pairs = parsePositiveInt64(require_value("--max-pairs"), "--max-pairs");
        }
        else if (key == "--max-image-size")
        {
            args.max_image_size = parsePositiveInt64(require_value("--max-image-size"), "--max-image-size");
        }
        else if (key == "--device")
        {
            args.device = require_value("--device");
        }
        else if (key == "--learning-rate")
        {
            args.learning_rate = parsePositiveDouble(require_value("--learning-rate"), "--learning-rate");
        }
        else if (key == "--input-channels")
        {
            args.input_channels = parsePositiveInt64(require_value("--input-channels"), "--input-channels");
        }
        else if (key == "--base-channels")
        {
            args.base_channels = parsePositiveInt64(require_value("--base-channels"), "--base-channels");
        }
        else if (key == "--descriptor-dim")
        {
            args.descriptor_dim = parsePositiveInt64(require_value("--descriptor-dim"), "--descriptor-dim");
        }
        else if (key == "--graph-hidden-dim")
        {
            args.graph_hidden_dim = parsePositiveInt64(require_value("--graph-hidden-dim"), "--graph-hidden-dim");
        }
        else if (key == "--graph-attention-layers")
        {
            args.graph_attention_layers =
                parsePositiveInt64(require_value("--graph-attention-layers"), "--graph-attention-layers");
        }
        else if (key == "--graph-keypoint-meta-dim")
        {
            args.graph_keypoint_meta_dim =
                parsePositiveInt64(require_value("--graph-keypoint-meta-dim"), "--graph-keypoint-meta-dim");
        }
        else if (key == "--full-v21")
        {
            args.input_channels = 1;
            args.base_channels = 64;
            args.descriptor_dim = 256;
            args.graph_hidden_dim = 512;
            args.graph_attention_layers = 8;
            args.graph_keypoint_meta_dim = 16;
        }
        else if (key == "--output-checkpoint")
        {
            args.output_checkpoint = require_value("--output-checkpoint");
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
    return args;
}

std::vector<std::filesystem::path> discoverPairs(const Args& args)
{
    std::vector<std::filesystem::path> pairs = args.pairs;
    std::filesystem::path root;
    if (!args.cache_dir.empty())
    {
        root = args.cache_dir;
    }
    else if (!args.dataset_root.empty())
    {
        root = args.dataset_root / "cache" / "train";
    }

    if (!root.empty())
    {
        const auto discovered = pfm::discoverPairArchivePaths(root);
        pairs.insert(pairs.end(), discovered.begin(), discovered.end());
    }

    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    if (pairs.empty())
    {
        throw std::invalid_argument("no pair_*.pt archives found");
    }
    if (static_cast<int64_t>(pairs.size()) > args.max_pairs)
    {
        pairs.resize(static_cast<std::size_t>(args.max_pairs));
    }
    return pairs;
}

torch::Tensor resizeImage(const torch::Tensor& image, int64_t max_image_size)
{
    const auto height = image.size(1);
    const auto width = image.size(2);
    const auto max_edge = std::max(height, width);
    if (max_edge <= max_image_size)
    {
        return image;
    }
    const double scale = static_cast<double>(max_image_size) / static_cast<double>(max_edge);
    const auto resized_height =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(height) * scale)));
    const auto resized_width =
        std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(width) * scale)));
    return torch::nn::functional::interpolate(image.unsqueeze(0),
                                              torch::nn::functional::InterpolateFuncOptions()
                                                  .size(std::vector<int64_t>{resized_height, resized_width})
                                                  .mode(torch::kBilinear)
                                                  .align_corners(false))
        .squeeze(0)
        .contiguous();
}

torch::Tensor resizeMaskForHeatmap(const torch::Tensor& mask, const torch::Device& device, int64_t height,
                                   int64_t width)
{
    return torch::nn::functional::interpolate(mask.to(torch::kFloat32).unsqueeze(0).unsqueeze(0),
                                              torch::nn::functional::InterpolateFuncOptions()
                                                  .size(std::vector<int64_t>{height, width})
                                                  .mode(torch::kNearest))
        .to(device)
        .contiguous();
}

double gradL2Norm(const std::vector<torch::Tensor>& parameters)
{
    double squared = 0.0;
    for (const auto& parameter : parameters)
    {
        if (parameter.grad().defined())
        {
            const auto grad = parameter.grad().detach();
            squared += grad.pow(2).sum().item<double>();
        }
    }
    return std::sqrt(squared);
}

torch::Device parseDevice(const std::string& value)
{
    if (value == "cuda")
    {
        if (!torch::cuda::is_available())
        {
            throw std::invalid_argument("CUDA requested but torch::cuda::is_available() is false");
        }
        return torch::Device(torch::kCUDA);
    }
    if (value == "cpu")
    {
        return torch::Device(torch::kCPU);
    }
    throw std::invalid_argument("unsupported device: " + value);
}

pfm::v21::PfmV21Config makeConfig(const Args& args)
{
    pfm::v21::PfmV21Config config;
    config.input_channels = args.input_channels;
    config.base_channels = args.base_channels;
    config.descriptor_dim = args.descriptor_dim;
    config.graph_hidden_dim = args.graph_hidden_dim;
    config.graph_attention_layers = args.graph_attention_layers;
    config.graph_keypoint_meta_dim = args.graph_keypoint_meta_dim;
    return config;
}

void writeInt64(torch::serialize::OutputArchive& archive, const char* name, int64_t value)
{
    archive.write(name, torch::tensor(value, torch::kInt64));
}

void writeConfigArchive(torch::serialize::OutputArchive& archive, const pfm::v21::PfmV21Config& config)
{
    torch::serialize::OutputArchive config_archive;
    writeInt64(config_archive, "input_channels", config.input_channels);
    writeInt64(config_archive, "base_channels", config.base_channels);
    writeInt64(config_archive, "descriptor_dim", config.descriptor_dim);
    writeInt64(config_archive, "graph_hidden_dim", config.graph_hidden_dim);
    writeInt64(config_archive, "graph_attention_layers", config.graph_attention_layers);
    writeInt64(config_archive, "graph_keypoint_meta_dim", config.graph_keypoint_meta_dim);
    archive.write("config", config_archive);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto args = parseArgs(argc, argv);
        const auto pair_paths = discoverPairs(args);
        const auto device = parseDevice(args.device);

        torch::manual_seed(20260602);

        const auto config = makeConfig(args);

        auto model = pfm::v21::PfmV21FeatureMatcher(config);
        model->to(device);
        model->train();
        torch::optim::AdamW optimizer(model->parameters(),
                                      torch::optim::AdamWOptions(args.learning_rate).weight_decay(1.0e-4));

        double first_loss = 0.0;
        double last_loss = 0.0;
        int64_t loaded_pairs = 0;
        int64_t channel_count = 0;

        for (const auto& path : pair_paths)
        {
            auto pair = pfm::loadPairArchiveSample(path);
            if (channel_count == 0)
            {
                channel_count = pair.view_a.size(0);
                if (channel_count != config.input_channels)
                {
                    throw std::invalid_argument("smoke model input_channels does not match pair image channels");
                }
            }

            auto view_a = resizeImage(pair.view_a, args.max_image_size).unsqueeze(0).to(device);
            auto view_b = resizeImage(pair.view_b, args.max_image_size).unsqueeze(0).to(device);

            optimizer.zero_grad();
            auto output_a = model->forwardSingle(view_a);
            auto output_b = model->forwardSingle(view_b);
            auto valid_target =
                resizeMaskForHeatmap(pair.valid_mask, device, output_a.heatmap.size(2), output_a.heatmap.size(3));

            auto heatmap_loss = torch::nn::functional::mse_loss(output_a.heatmap, valid_target) +
                                torch::nn::functional::mse_loss(output_b.heatmap, valid_target);
            auto quality_loss = torch::nn::functional::mse_loss(output_a.quality, valid_target) +
                                torch::nn::functional::mse_loss(output_b.quality, valid_target);
            auto descriptor_loss = (output_a.descriptors - output_b.descriptors).abs().mean();
            auto loss = heatmap_loss + quality_loss * 0.5 + descriptor_loss * 0.05;

            if (!torch::isfinite(loss).item<bool>())
            {
                throw std::runtime_error("non-finite loss for " + path.string());
            }
            loss.backward();
            const auto grad_l2 = gradL2Norm(model->parameters());
            if (!(grad_l2 > 0.0) || !std::isfinite(grad_l2))
            {
                throw std::runtime_error("invalid gradient norm for " + path.string());
            }
            optimizer.step();

            last_loss = loss.detach().cpu().item<double>();
            if (loaded_pairs == 0)
            {
                first_loss = last_loss;
            }
            ++loaded_pairs;

            std::cout << "step=" << loaded_pairs << " pair=" << path << " image_shape=" << view_a.size(1) << "x"
                      << view_a.size(2) << "x" << view_a.size(3)
                      << " valid_pixels=" << pair.valid_mask.sum().item<int64_t>() << " loss=" << last_loss
                      << " grad_l2=" << grad_l2 << "\n";
        }

        if (!args.output_checkpoint.empty())
        {
            std::filesystem::create_directories(args.output_checkpoint.parent_path());
            torch::serialize::OutputArchive archive;
            model->save(archive);
            writeConfigArchive(archive, config);
            archive.save_to(args.output_checkpoint.string());
        }

        std::cout << "summary pairs=" << loaded_pairs << " first_loss=" << first_loss << " last_loss=" << last_loss
                  << " device=" << args.device << " max_image_size=" << args.max_image_size
                  << " input_channels=" << config.input_channels << " base_channels=" << config.base_channels
                  << " descriptor_dim=" << config.descriptor_dim << " graph_hidden_dim=" << config.graph_hidden_dim
                  << " graph_attention_layers=" << config.graph_attention_layers
                  << " graph_keypoint_meta_dim=" << config.graph_keypoint_meta_dim;
        if (!args.output_checkpoint.empty())
        {
            std::cout << " checkpoint=" << args.output_checkpoint;
        }
        std::cout << "\n";
        return 0;
    }
    catch (const std::exception& exc)
    {
        std::cerr << "error: " << exc.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }
}
