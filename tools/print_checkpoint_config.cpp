#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

namespace {

int64_t readInt64(torch::serialize::InputArchive& archive, const char* name) {
    torch::Tensor tensor;
    archive.read(name, tensor);
    if (!tensor.defined() || tensor.numel() != 1) {
        throw std::invalid_argument(std::string("checkpoint config missing ") + name);
    }
    return tensor.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
}

int64_t readInt64Or(torch::serialize::InputArchive& archive, const char* name, int64_t fallback) {
    try {
        return readInt64(archive, name);
    } catch (const c10::Error&) {
        return fallback;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: pfm_print_checkpoint_config checkpoint.pt\n";
        return 2;
    }

    try {
        torch::serialize::InputArchive checkpoint;
        checkpoint.load_from(argv[1]);
        torch::serialize::InputArchive config;
        checkpoint.read("config", config);

        const auto input_channels = readInt64(config, "input_channels");
        const auto base_channels = readInt64(config, "base_channels");
        const auto descriptor_dim = readInt64(config, "descriptor_dim");
        const auto graph_hidden_dim = readInt64Or(config, "graph_hidden_dim", std::max<int64_t>(32, descriptor_dim));
        const auto graph_attention_layers = readInt64Or(config, "graph_attention_layers", 1);
        const auto checkpoint_version = readInt64Or(config, "checkpoint_version", 1);

        std::cout << "checkpoint_version=" << checkpoint_version << '\n'
                  << "input_channels=" << input_channels << '\n'
                  << "base_channels=" << base_channels << '\n'
                  << "descriptor_dim=" << descriptor_dim << '\n'
                  << "graph_hidden_dim=" << graph_hidden_dim << '\n'
                  << "graph_attention_layers=" << graph_attention_layers << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "print checkpoint config failed: " << error.what() << '\n';
        return 1;
    }
}
