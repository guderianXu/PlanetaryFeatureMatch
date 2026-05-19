#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include <unistd.h>

#include <torch/torch.h>

#include "infer/match_codec.h"
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

pfm::MatchSet makeMatchSet() {
    return pfm::MatchSet{
        torch::tensor({{0, 1}, {2, 3}}, torch::kInt64),
        torch::tensor({0.95F, 0.85F}, torch::kFloat32),
        torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}}, torch::kFloat32),
        torch::tensor({{5.0F, 6.0F}, {7.0F, 8.0F}}, torch::kFloat32),
        torch::tensor({{0.9F, 0.1F}, {0.2F, 0.8F}}, torch::kFloat32)};
}

}  // namespace

static void match_codec_round_trips_all_fields() {
    TempPtFile temp_file("pfm_match_codec_round_trip");
    const auto expected = makeMatchSet();

    pfm::save_match_set(expected, temp_file.path());
    const auto actual = pfm::load_match_set(temp_file.path());

    requireSameTensor(actual.sparse_matches, expected.sparse_matches);
    requireSameTensor(actual.sparse_scores, expected.sparse_scores);
    requireSameTensor(actual.points_a, expected.points_a);
    requireSameTensor(actual.points_b, expected.points_b);
    requireSameTensor(actual.confidence, expected.confidence);
}

static void match_codec_rejects_missing_path() {
    TempPtFile temp_file("pfm_match_codec_missing");
    PFM_REQUIRE(!std::filesystem::exists(temp_file.path()));

    PFM_REQUIRE_INVALID_ARG(pfm::load_match_set(temp_file.path()));
}

static void match_codec_rejects_undefined_required_tensor() {
    TempPtFile temp_file("pfm_match_codec_undefined");
    auto match_set = makeMatchSet();
    match_set.points_b = torch::Tensor();

    PFM_REQUIRE_INVALID_ARG(pfm::save_match_set(match_set, temp_file.path()));
}

void register_match_codec_tests() {
    register_test("match_codec_round_trips_all_fields", match_codec_round_trips_all_fields);
    register_test("match_codec_rejects_missing_path", match_codec_rejects_missing_path);
    register_test("match_codec_rejects_undefined_required_tensor", match_codec_rejects_undefined_required_tensor);
}
