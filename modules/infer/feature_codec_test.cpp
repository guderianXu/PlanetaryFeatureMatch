#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include <unistd.h>

#include <torch/torch.h>

#include "infer/feature_codec.h"
#include "tests/test_harness.h"

namespace {

class TempPtFile {
public:
    explicit TempPtFile(const std::string& stem) {
        const auto suffix = std::to_string(static_cast<long long>(getpid())) + "_" +
                            std::to_string(std::random_device{}());
        _path = (std::filesystem::temp_directory_path() / (stem + "_" + suffix + ".pt")).string();
    }

    ~TempPtFile() {
        std::remove(_path.c_str());
    }

    const std::string& path() const {
        return _path;
    }

private:
    std::string _path;
};

void requireSameTensor(const torch::Tensor& lhs, const torch::Tensor& rhs) {
    PFM_REQUIRE(lhs.defined());
    PFM_REQUIRE(rhs.defined());
    PFM_REQUIRE(lhs.scalar_type() == rhs.scalar_type());
    PFM_REQUIRE(lhs.sizes() == rhs.sizes());
    PFM_REQUIRE(torch::equal(lhs.cpu(), rhs.cpu()));
}

pfm::FeatureSet makeFeatureSet() {
    return pfm::FeatureSet{
        torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}}, torch::kFloat32),
        torch::tensor({0.75F, 0.25F}, torch::kFloat32),
        torch::tensor({{0.1F, 0.2F, 0.3F}, {0.4F, 0.5F, 0.6F}}, torch::kFloat32),
        torch::tensor({1.5F, 2.5F}, torch::kFloat32),
        torch::tensor({0.0F, 1.57F}, torch::kFloat32),
        torch::eye(2, torch::kFloat32).repeat({2, 1, 1}),
        torch::tensor({{{0.0F, 0.0F}, {1.0F, 0.0F}}, {{0.0F, 1.0F}, {1.0F, 1.0F}}}, torch::kFloat32),
        torch::tensor({{0.9F, 0.8F}, {0.7F, 0.6F}}, torch::kFloat32)};
}

}  // namespace

static void feature_codec_round_trips_all_fields() {
    TempPtFile temp_file("pfm_feature_codec_round_trip");
    const auto expected = makeFeatureSet();

    pfm::save_feature_set(expected, temp_file.path());
    const auto actual = pfm::load_feature_set(temp_file.path());

    requireSameTensor(actual.keypoints, expected.keypoints);
    requireSameTensor(actual.scores, expected.scores);
    requireSameTensor(actual.descriptors, expected.descriptors);
    requireSameTensor(actual.scale, expected.scale);
    requireSameTensor(actual.orientation, expected.orientation);
    requireSameTensor(actual.affine, expected.affine);
    requireSameTensor(actual.dense_points, expected.dense_points);
    requireSameTensor(actual.dense_confidence, expected.dense_confidence);
}

static void feature_codec_rejects_missing_path() {
    TempPtFile temp_file("pfm_feature_codec_missing");
    PFM_REQUIRE(!std::filesystem::exists(temp_file.path()));

    PFM_REQUIRE_INVALID_ARG(pfm::load_feature_set(temp_file.path()));
}

static void feature_codec_rejects_undefined_required_tensor() {
    TempPtFile temp_file("pfm_feature_codec_undefined");
    auto feature_set = makeFeatureSet();
    feature_set.descriptors = torch::Tensor();

    PFM_REQUIRE_INVALID_ARG(pfm::save_feature_set(feature_set, temp_file.path()));
}

void register_feature_codec_tests() {
    register_test("feature_codec_round_trips_all_fields", feature_codec_round_trips_all_fields);
    register_test("feature_codec_rejects_missing_path", feature_codec_rejects_missing_path);
    register_test("feature_codec_rejects_undefined_required_tensor", feature_codec_rejects_undefined_required_tensor);
}
