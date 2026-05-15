#include <stdexcept>
#include <string>

#include <torch/serialize.h>
#include <torch/torch.h>

#include "infer/match_codec.h"

namespace pfm {
namespace {

void requireDefined(const torch::Tensor& tensor, const char* name) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
}

void writeTensor(torch::serialize::OutputArchive& archive, const torch::Tensor& tensor, const char* name) {
    requireDefined(tensor, name);
    archive.write(name, tensor);
}

torch::Tensor readTensor(torch::serialize::InputArchive& archive, const char* name) {
    torch::Tensor tensor;
    archive.read(name, tensor);
    requireDefined(tensor, name);
    return tensor;
}

}  // namespace

void save_match_set(const MatchSet& match_set, const std::string& path) {
    torch::serialize::OutputArchive archive;
    writeTensor(archive, match_set.sparse_matches, "sparse_matches");
    writeTensor(archive, match_set.sparse_scores, "sparse_scores");
    writeTensor(archive, match_set.points_a, "points_a");
    writeTensor(archive, match_set.points_b, "points_b");
    writeTensor(archive, match_set.confidence, "confidence");
    try {
        archive.save_to(path);
    } catch (const c10::Error& e) {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

MatchSet load_match_set(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        return MatchSet{
            readTensor(archive, "sparse_matches"),
            readTensor(archive, "sparse_scores"),
            readTensor(archive, "points_a"),
            readTensor(archive, "points_b"),
            readTensor(archive, "confidence")};
    } catch (const c10::Error& e) {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

}  // namespace pfm
