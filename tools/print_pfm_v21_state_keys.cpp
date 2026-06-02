#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "models/pfm_model_v21.h"

namespace
{

std::string requireValue(int argc, char** argv, int& index)
{
    if (index + 1 >= argc)
    {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    ++index;
    return argv[index];
}

int64_t parseInt64(const std::string& value, const char* name)
{
    char* end = nullptr;
    const auto parsed = std::strtoll(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return parsed;
}

pfm::v21::PfmV21Config parseConfig(int argc, char** argv)
{
    pfm::v21::PfmV21Config config;
    for (int index = 1; index < argc; ++index)
    {
        const std::string option = argv[index];
        if (option == "--input-channels")
        {
            config.input_channels = parseInt64(requireValue(argc, argv, index), "input_channels");
        }
        else if (option == "--base-channels")
        {
            config.base_channels = parseInt64(requireValue(argc, argv, index), "base_channels");
        }
        else if (option == "--descriptor-dim")
        {
            config.descriptor_dim = parseInt64(requireValue(argc, argv, index), "descriptor_dim");
        }
        else if (option == "--graph-hidden-dim")
        {
            config.graph_hidden_dim = parseInt64(requireValue(argc, argv, index), "graph_hidden_dim");
        }
        else if (option == "--graph-attention-layers")
        {
            config.graph_attention_layers = parseInt64(requireValue(argc, argv, index), "graph_attention_layers");
        }
        else if (option == "--graph-keypoint-meta-dim")
        {
            config.graph_keypoint_meta_dim = parseInt64(requireValue(argc, argv, index), "graph_keypoint_meta_dim");
        }
        else if (option == "--help")
        {
            std::cout << "Usage: pfm_print_v21_state_keys [--input-channels N] [--base-channels N] "
                      << "[--descriptor-dim N] [--graph-hidden-dim N] [--graph-attention-layers N] "
                      << "[--graph-keypoint-meta-dim N]\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    return config;
}

std::string shapeString(const torch::Tensor& tensor)
{
    std::string result = "[";
    for (int64_t index = 0; index < tensor.dim(); ++index)
    {
        if (index != 0)
        {
            result += ",";
        }
        result += std::to_string(tensor.size(index));
    }
    result += "]";
    return result;
}

struct Row
{
    std::string kind;
    std::string key;
    std::string shape;
};

} // namespace

int main(int argc, char** argv)
{
    try
    {
        auto model = pfm::v21::PfmV21FeatureMatcher(parseConfig(argc, argv));
        std::vector<Row> rows;
        for (const auto& parameter : model->named_parameters(true))
        {
            rows.push_back(Row{"parameter", parameter.key(), shapeString(parameter.value())});
        }
        for (const auto& buffer : model->named_buffers(true))
        {
            rows.push_back(Row{"buffer", buffer.key(), shapeString(buffer.value())});
        }
        std::sort(rows.begin(), rows.end(),
                  [](const Row& left, const Row& right)
                  {
                      return left.key < right.key;
                  });
        for (const auto& row : rows)
        {
            std::cout << row.kind << '\t' << row.key << '\t' << row.shape << '\n';
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "pfm_print_v21_state_keys failed: " << error.what() << '\n';
        return 1;
    }
}
