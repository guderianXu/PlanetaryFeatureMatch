#include "feature_io/match_codec.h"

#include <stdexcept>
#include <string>

#include <torch/serialize.h>
#include <torch/torch.h>

namespace pfm
{
namespace
{

void requireDefined(const torch::Tensor& tensor, const char* name)
{
    if (!tensor.defined())
    {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
}

void writeTensor(torch::serialize::OutputArchive& archive, const torch::Tensor& tensor, const char* name)
{
    requireDefined(tensor, name);
    archive.write(name, tensor);
}

void writeInt64(torch::serialize::OutputArchive& archive, int64_t value, const char* name)
{
    archive.write(name, torch::tensor(value, torch::TensorOptions().dtype(torch::kInt64)));
}

void writeDouble(torch::serialize::OutputArchive& archive, double value, const char* name)
{
    archive.write(name, torch::tensor(value, torch::TensorOptions().dtype(torch::kFloat64)));
}

torch::Tensor readTensor(torch::serialize::InputArchive& archive, const char* name)
{
    torch::Tensor tensor;
    archive.read(name, tensor);
    requireDefined(tensor, name);
    return tensor;
}

int64_t readOptionalInt64(torch::serialize::InputArchive& archive, const char* name, int64_t default_value)
{
    try
    {
        torch::Tensor tensor;
        archive.read(name, tensor);
        if (!tensor.defined() || tensor.numel() == 0)
        {
            return default_value;
        }
        return tensor.to(torch::kCPU).to(torch::kInt64).reshape({-1}).index({0}).item<int64_t>();
    }
    catch (const c10::Error&)
    {
        return default_value;
    }
}

double readOptionalDouble(torch::serialize::InputArchive& archive, const char* name, double default_value)
{
    try
    {
        torch::Tensor tensor;
        archive.read(name, tensor);
        if (!tensor.defined() || tensor.numel() == 0)
        {
            return default_value;
        }
        return tensor.to(torch::kCPU).to(torch::kFloat64).reshape({-1}).index({0}).item<double>();
    }
    catch (const c10::Error&)
    {
        return default_value;
    }
}

} // namespace

void save_match_set(const MatchSet& match_set, const std::string& path)
{
    torch::serialize::OutputArchive archive;
    writeTensor(archive, match_set.sparse_matches, "sparse_matches");
    writeTensor(archive, match_set.sparse_scores, "sparse_scores");
    writeTensor(archive, match_set.points_a, "points_a");
    writeTensor(archive, match_set.points_b, "points_b");
    writeTensor(archive, match_set.confidence, "confidence");
    writeInt64(archive, match_set.graph_executed_layers, "graph_executed_layers");
    writeInt64(archive, match_set.graph_input_keypoints_a, "graph_input_keypoints_a");
    writeInt64(archive, match_set.graph_input_keypoints_b, "graph_input_keypoints_b");
    writeInt64(archive, match_set.graph_kept_keypoints_a, "graph_kept_keypoints_a");
    writeInt64(archive, match_set.graph_kept_keypoints_b, "graph_kept_keypoints_b");
    writeInt64(archive, match_set.graph_pruned_keypoints_a, "graph_pruned_keypoints_a");
    writeInt64(archive, match_set.graph_pruned_keypoints_b, "graph_pruned_keypoints_b");
    writeInt64(archive, match_set.graph_attention_work_units, "graph_attention_work_units");
    writeInt64(archive, match_set.graph_full_attention_work_units, "graph_full_attention_work_units");
    writeDouble(archive, match_set.graph_attention_work_fraction, "graph_attention_work_fraction");
    try
    {
        archive.save_to(path);
    }
    catch (const c10::Error& e)
    {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

MatchSet load_match_set(const std::string& path)
{
    try
    {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        auto match_set = MatchSet{readTensor(archive, "sparse_matches"), readTensor(archive, "sparse_scores"),
                                  readTensor(archive, "points_a"), readTensor(archive, "points_b"),
                                  readTensor(archive, "confidence")};
        match_set.graph_executed_layers = readOptionalInt64(archive, "graph_executed_layers", 0);
        match_set.graph_input_keypoints_a = readOptionalInt64(archive, "graph_input_keypoints_a", 0);
        match_set.graph_input_keypoints_b = readOptionalInt64(archive, "graph_input_keypoints_b", 0);
        match_set.graph_kept_keypoints_a = readOptionalInt64(archive, "graph_kept_keypoints_a", 0);
        match_set.graph_kept_keypoints_b = readOptionalInt64(archive, "graph_kept_keypoints_b", 0);
        match_set.graph_pruned_keypoints_a = readOptionalInt64(archive, "graph_pruned_keypoints_a", 0);
        match_set.graph_pruned_keypoints_b = readOptionalInt64(archive, "graph_pruned_keypoints_b", 0);
        match_set.graph_attention_work_units = readOptionalInt64(archive, "graph_attention_work_units", 0);
        match_set.graph_full_attention_work_units = readOptionalInt64(archive, "graph_full_attention_work_units", 0);
        match_set.graph_attention_work_fraction = readOptionalDouble(archive, "graph_attention_work_fraction", 0.0);
        return match_set;
    }
    catch (const c10::Error& e)
    {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

} // namespace pfm
