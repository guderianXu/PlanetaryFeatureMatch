#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "models/pfm_model_v21.h"

namespace
{

constexpr int64_t INPUT_CHANNELS = 1;

struct Options
{
    std::string base_checkpoint;
    std::string candidate_checkpoint;
    std::string output_checkpoint;
    double alpha = 0.0;
};

struct CheckpointConfig
{
    int64_t base_channels = 0;
    int64_t descriptor_dim = 0;
    int64_t graph_hidden_dim = 0;
    int64_t graph_attention_layers = 0;
    int64_t graph_keypoint_meta_dim = 16;
};

struct Modules
{
    pfm::v21::PfmV21Backbone backbone{nullptr};
    pfm::v21::PfmV21DualFPNLite dual_fpn{nullptr};
    pfm::v21::PfmV21SparseHead sparse_head{nullptr};
    pfm::v21::PfmV21TextureDescriptorAdapter texture_adapter{nullptr};
    pfm::v21::PfmV21DescriptorFusionAdapter descriptor_fusion{nullptr};
    pfm::v21::PfmV21DenseHead dense_head{nullptr};
    pfm::v21::PfmV21QualityHead quality_head{nullptr};
    pfm::v21::PfmV21SemiDenseCandidateBranch semi_dense_branch{nullptr};
    pfm::v21::PfmV21GraphMatcher graph_matcher{nullptr};
};

std::string requireValue(int& index, int argc, char** argv, const char* option)
{
    if (index + 1 >= argc)
    {
        throw std::invalid_argument(std::string(option) + " requires a value");
    }
    return argv[++index];
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        if (arg == "--base")
        {
            options.base_checkpoint = requireValue(index, argc, argv, "--base");
        }
        else if (arg == "--candidate")
        {
            options.candidate_checkpoint = requireValue(index, argc, argv, "--candidate");
        }
        else if (arg == "--output")
        {
            options.output_checkpoint = requireValue(index, argc, argv, "--output");
        }
        else if (arg == "--alpha")
        {
            options.alpha = std::stod(requireValue(index, argc, argv, "--alpha"));
        }
        else if (arg == "-h" || arg == "--help")
        {
            std::cout << "Usage: pfm_interpolate_checkpoints --base stable.pt --candidate tuned.pt "
                         "--alpha 0.25 --output mixed.pt\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.base_checkpoint.empty() || options.candidate_checkpoint.empty() || options.output_checkpoint.empty())
    {
        throw std::invalid_argument("--base, --candidate, and --output are required");
    }
    if (!(options.alpha >= 0.0 && options.alpha <= 1.0))
    {
        throw std::invalid_argument("--alpha must be in [0, 1]");
    }
    return options;
}

int64_t readConfigInt64(torch::serialize::InputArchive& config_archive, const char* name)
{
    torch::Tensor tensor;
    config_archive.read(name, tensor);
    if (!tensor.defined() || tensor.numel() != 1)
    {
        throw std::invalid_argument(std::string("checkpoint config missing ") + name);
    }
    return tensor.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
}

CheckpointConfig readConfig(const std::string& checkpoint)
{
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint);
    torch::serialize::InputArchive config_archive;
    archive.read("config", config_archive);
    const auto input_channels = readConfigInt64(config_archive, "input_channels");
    if (input_channels != INPUT_CHANNELS)
    {
        throw std::invalid_argument("checkpoint input_channels mismatch");
    }
    return CheckpointConfig{
        readConfigInt64(config_archive, "base_channels"), readConfigInt64(config_archive, "descriptor_dim"),
        readConfigInt64(config_archive, "graph_hidden_dim"), readConfigInt64(config_archive, "graph_attention_layers"),
        readConfigInt64(config_archive, "graph_keypoint_meta_dim")};
}

void requireSameConfig(const CheckpointConfig& lhs, const CheckpointConfig& rhs)
{
    if (lhs.base_channels != rhs.base_channels || lhs.descriptor_dim != rhs.descriptor_dim ||
        lhs.graph_hidden_dim != rhs.graph_hidden_dim || lhs.graph_attention_layers != rhs.graph_attention_layers ||
        lhs.graph_keypoint_meta_dim != rhs.graph_keypoint_meta_dim)
    {
        throw std::invalid_argument("checkpoint architectures do not match");
    }
}

Modules makeModules(const CheckpointConfig& config)
{
    Modules modules;
    modules.backbone = pfm::v21::PfmV21Backbone(INPUT_CHANNELS, config.base_channels);
    modules.dual_fpn = pfm::v21::PfmV21DualFPNLite(config.base_channels);
    modules.sparse_head = pfm::v21::PfmV21SparseHead(config.base_channels * 2, config.descriptor_dim);
    modules.texture_adapter = pfm::v21::PfmV21TextureDescriptorAdapter(config.descriptor_dim);
    modules.descriptor_fusion = pfm::v21::PfmV21DescriptorFusionAdapter(config.descriptor_dim);
    modules.dense_head = pfm::v21::PfmV21DenseHead(config.base_channels);
    modules.quality_head = pfm::v21::PfmV21QualityHead(config.descriptor_dim);
    modules.semi_dense_branch = pfm::v21::PfmV21SemiDenseCandidateBranch(config.descriptor_dim);
    modules.graph_matcher = pfm::v21::PfmV21GraphMatcher(config.descriptor_dim, config.graph_hidden_dim,
                                                         config.graph_attention_layers, config.graph_keypoint_meta_dim);
    return modules;
}

void loadModules(const std::string& checkpoint, Modules& modules)
{
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint);
    torch::serialize::InputArchive backbone_archive;
    torch::serialize::InputArchive dual_fpn_archive;
    torch::serialize::InputArchive sparse_head_archive;
    torch::serialize::InputArchive texture_adapter_archive;
    torch::serialize::InputArchive descriptor_fusion_archive;
    torch::serialize::InputArchive dense_head_archive;
    torch::serialize::InputArchive quality_head_archive;
    torch::serialize::InputArchive semi_dense_branch_archive;
    torch::serialize::InputArchive graph_matcher_archive;
    archive.read("backbone", backbone_archive);
    archive.read("dual_fpn", dual_fpn_archive);
    archive.read("sparse_head", sparse_head_archive);
    archive.read("texture_adapter", texture_adapter_archive);
    archive.read("descriptor_fusion", descriptor_fusion_archive);
    archive.read("dense_head", dense_head_archive);
    archive.read("quality_head", quality_head_archive);
    archive.read("semi_dense_branch", semi_dense_branch_archive);
    archive.read("graph_matcher", graph_matcher_archive);
    modules.backbone->load(backbone_archive);
    modules.backbone->sanitizeNonfiniteState();
    modules.dual_fpn->load(dual_fpn_archive);
    modules.sparse_head->load(sparse_head_archive);
    modules.texture_adapter->load(texture_adapter_archive);
    modules.descriptor_fusion->load(descriptor_fusion_archive);
    modules.dense_head->load(dense_head_archive);
    modules.quality_head->load(quality_head_archive);
    modules.semi_dense_branch->load(semi_dense_branch_archive);
    modules.graph_matcher->load(graph_matcher_archive);
}

void saveModules(const std::string& checkpoint, const CheckpointConfig& config, Modules& modules)
{
    torch::serialize::OutputArchive archive;
    torch::serialize::OutputArchive config_archive;
    config_archive.write("checkpoint_version", torch::tensor({3}, torch::kInt64));
    config_archive.write("base_channels", torch::tensor({config.base_channels}, torch::kInt64));
    config_archive.write("descriptor_dim", torch::tensor({config.descriptor_dim}, torch::kInt64));
    config_archive.write("graph_hidden_dim", torch::tensor({config.graph_hidden_dim}, torch::kInt64));
    config_archive.write("graph_attention_layers", torch::tensor({config.graph_attention_layers}, torch::kInt64));
    config_archive.write("graph_keypoint_meta_dim", torch::tensor({config.graph_keypoint_meta_dim}, torch::kInt64));
    config_archive.write("input_channels", torch::tensor({INPUT_CHANNELS}, torch::kInt64));
    archive.write("config", config_archive);

    torch::serialize::OutputArchive backbone_archive;
    torch::serialize::OutputArchive dual_fpn_archive;
    torch::serialize::OutputArchive sparse_head_archive;
    torch::serialize::OutputArchive texture_adapter_archive;
    torch::serialize::OutputArchive descriptor_fusion_archive;
    torch::serialize::OutputArchive dense_head_archive;
    torch::serialize::OutputArchive quality_head_archive;
    torch::serialize::OutputArchive semi_dense_branch_archive;
    torch::serialize::OutputArchive graph_matcher_archive;
    modules.backbone->sanitizeNonfiniteState();
    modules.backbone->save(backbone_archive);
    modules.dual_fpn->save(dual_fpn_archive);
    modules.sparse_head->save(sparse_head_archive);
    modules.texture_adapter->save(texture_adapter_archive);
    modules.descriptor_fusion->save(descriptor_fusion_archive);
    modules.dense_head->save(dense_head_archive);
    modules.quality_head->save(quality_head_archive);
    modules.semi_dense_branch->save(semi_dense_branch_archive);
    modules.graph_matcher->save(graph_matcher_archive);
    archive.write("backbone", backbone_archive);
    archive.write("dual_fpn", dual_fpn_archive);
    archive.write("sparse_head", sparse_head_archive);
    archive.write("texture_adapter", texture_adapter_archive);
    archive.write("descriptor_fusion", descriptor_fusion_archive);
    archive.write("dense_head", dense_head_archive);
    archive.write("quality_head", quality_head_archive);
    archive.write("semi_dense_branch", semi_dense_branch_archive);
    archive.write("graph_matcher", graph_matcher_archive);
    archive.save_to(checkpoint);
}

void interpolateNamedTensors(const torch::OrderedDict<std::string, torch::Tensor>& base,
                             const torch::OrderedDict<std::string, torch::Tensor>& candidate, double alpha)
{
    for (const auto& item : base)
    {
        const auto& name = item.key();
        auto target = item.value();
        if (!candidate.contains(name))
        {
            throw std::invalid_argument("candidate missing tensor: " + name);
        }
        const auto source = candidate[name];
        if (target.sizes() != source.sizes())
        {
            throw std::invalid_argument("tensor shape mismatch: " + name);
        }
        if (!target.is_floating_point())
        {
            target.copy_(source);
            continue;
        }
        target.copy_(target * (1.0 - alpha) + source.to(target.device(), target.dtype()) * alpha);
    }
}

void interpolateModule(torch::nn::Module& base, torch::nn::Module& candidate, double alpha)
{
    interpolateNamedTensors(base.named_parameters(true), candidate.named_parameters(true), alpha);
    interpolateNamedTensors(base.named_buffers(true), candidate.named_buffers(true), alpha);
}

void interpolateModules(Modules& base, Modules& candidate, double alpha)
{
    torch::NoGradGuard no_grad;
    interpolateModule(*base.backbone, *candidate.backbone, alpha);
    interpolateModule(*base.dual_fpn, *candidate.dual_fpn, alpha);
    interpolateModule(*base.sparse_head, *candidate.sparse_head, alpha);
    interpolateModule(*base.texture_adapter, *candidate.texture_adapter, alpha);
    interpolateModule(*base.descriptor_fusion, *candidate.descriptor_fusion, alpha);
    interpolateModule(*base.dense_head, *candidate.dense_head, alpha);
    interpolateModule(*base.quality_head, *candidate.quality_head, alpha);
    interpolateModule(*base.semi_dense_branch, *candidate.semi_dense_branch, alpha);
    interpolateModule(*base.graph_matcher, *candidate.graph_matcher, alpha);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parseOptions(argc, argv);
        const auto base_config = readConfig(options.base_checkpoint);
        const auto candidate_config = readConfig(options.candidate_checkpoint);
        requireSameConfig(base_config, candidate_config);
        auto base_modules = makeModules(base_config);
        auto candidate_modules = makeModules(candidate_config);
        loadModules(options.base_checkpoint, base_modules);
        loadModules(options.candidate_checkpoint, candidate_modules);
        interpolateModules(base_modules, candidate_modules, options.alpha);
        saveModules(options.output_checkpoint, base_config, base_modules);
        std::cout << "interpolated checkpoint: output=" << options.output_checkpoint << " alpha=" << options.alpha
                  << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "interpolate checkpoints failed: " << error.what() << '\n';
        return 1;
    }
}
