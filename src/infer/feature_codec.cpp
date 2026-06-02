#include "infer/feature_codec.h"

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

torch::Tensor readTensor(torch::serialize::InputArchive& archive, const char* name)
{
    torch::Tensor tensor;
    archive.read(name, tensor);
    requireDefined(tensor, name);
    return tensor;
}

int64_t readOptionalInt64(torch::serialize::InputArchive& archive, const char* name)
{
    try
    {
        torch::Tensor tensor;
        archive.read(name, tensor);
        requireDefined(tensor, name);
        return tensor.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
    }
    catch (const c10::Error&)
    {
        return 0;
    }
}

} // namespace

void save_feature_set(const FeatureSet& feature_set, const std::string& path)
{
    torch::serialize::OutputArchive archive;
    writeTensor(archive, feature_set.keypoints, "keypoints");
    writeTensor(archive, feature_set.scores, "scores");
    writeTensor(archive, feature_set.descriptors, "descriptors");
    writeTensor(archive, feature_set.scale, "scale");
    writeTensor(archive, feature_set.orientation, "orientation");
    writeTensor(archive, feature_set.affine, "affine");
    writeTensor(archive, feature_set.dense_points, "dense_points");
    writeTensor(archive, feature_set.dense_confidence, "dense_confidence");
    archive.write("feature_map_width", torch::tensor({feature_set.feature_map_width}, torch::kInt64));
    archive.write("feature_map_height", torch::tensor({feature_set.feature_map_height}, torch::kInt64));
    try
    {
        archive.save_to(path);
    }
    catch (const c10::Error& e)
    {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

FeatureSet load_feature_set(const std::string& path)
{
    try
    {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        return FeatureSet{readTensor(archive, "keypoints"),
                          readTensor(archive, "scores"),
                          readTensor(archive, "descriptors"),
                          readTensor(archive, "scale"),
                          readTensor(archive, "orientation"),
                          readTensor(archive, "affine"),
                          readTensor(archive, "dense_points"),
                          readTensor(archive, "dense_confidence"),
                          readOptionalInt64(archive, "feature_map_width"),
                          readOptionalInt64(archive, "feature_map_height")};
    }
    catch (const c10::Error& e)
    {
        throw std::invalid_argument(e.what_without_backtrace());
    }
}

} // namespace pfm
