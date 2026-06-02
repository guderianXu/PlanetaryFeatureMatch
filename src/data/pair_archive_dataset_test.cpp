#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>
#include <unistd.h>

#include "data/pair_archive_dataset.h"
#include "tests/test_harness.h"

namespace
{

class TempPairArchiveDirectory
{
  public:
    explicit TempPairArchiveDirectory(const std::string& name)
    {
        const auto root = std::filesystem::temp_directory_path();
        path = root / (name + "_" + std::to_string(::getpid()) + "_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
    }

    ~TempPairArchiveDirectory() = default;

    std::filesystem::path path;
};

struct TestPairTensors
{
    torch::Tensor view_a;
    torch::Tensor view_b;
    torch::Tensor warp_a_to_b;
    torch::Tensor valid_mask;
};

TestPairTensors makeValidPairTensors()
{
    return TestPairTensors{torch::ones({1, 4, 5}, torch::kFloat32), torch::ones({1, 4, 5}, torch::kFloat32) * 0.5,
                           torch::zeros({4, 5, 2}, torch::kFloat32), torch::ones({4, 5}, torch::kBool)};
}

void writePairArchive(const std::filesystem::path& path, const TestPairTensors& pair)
{
    std::filesystem::create_directories(path.parent_path());
    torch::jit::Module module("PairArchive");
    module.register_attribute("view_a", c10::TensorType::get(), pair.view_a);
    module.register_attribute("view_b", c10::TensorType::get(), pair.view_b);
    module.register_attribute("warp_a_to_b", c10::TensorType::get(), pair.warp_a_to_b);
    module.register_attribute("valid_mask", c10::TensorType::get(), pair.valid_mask);
    module.save(path.string());
}

void pair_archive_dataset_discovers_sorted_limited_paths()
{
    TempPairArchiveDirectory temp("pfm_pair_archive_discover");
    const auto cache = temp.path / "cache" / "train" / "source_00001";
    const auto pair = makeValidPairTensors();
    writePairArchive(cache / "pair_000002_b.pt", pair);
    writePairArchive(cache / "pair_000001_a.pt", pair);

    const auto paths = pfm::discoverPairArchivePaths(temp.path / "cache" / "train", 1);

    PFM_REQUIRE(paths.size() == 1);
    PFM_REQUIRE(paths.front().filename().string() == "pair_000001_a.pt");
}

void pair_archive_dataset_loads_valid_archive()
{
    TempPairArchiveDirectory temp("pfm_pair_archive_load");
    const auto path = temp.path / "cache" / "train" / "source_00001" / "pair_000001_a.pt";
    writePairArchive(path, makeValidPairTensors());

    pfm::PairArchiveDataset dataset({temp.path / "cache" / "train"});
    const auto sample = dataset.load(0);

    PFM_REQUIRE(dataset.size() == 1);
    PFM_REQUIRE(dataset.path(0) == path);
    PFM_REQUIRE(sample.path == path);
    PFM_REQUIRE(sample.view_a.sizes() == torch::IntArrayRef({1, 4, 5}));
    PFM_REQUIRE(sample.view_b.sizes() == torch::IntArrayRef({1, 4, 5}));
    PFM_REQUIRE(sample.warp_a_to_b.sizes() == torch::IntArrayRef({4, 5, 2}));
    PFM_REQUIRE(sample.valid_mask.sizes() == torch::IntArrayRef({4, 5}));
}

void pair_archive_dataset_rejects_empty_valid_mask_by_default()
{
    TempPairArchiveDirectory temp("pfm_pair_archive_empty_mask");
    const auto path = temp.path / "cache" / "train" / "source_00001" / "pair_000001_a.pt";
    auto pair = makeValidPairTensors();
    pair.valid_mask = torch::zeros({4, 5}, torch::kBool);
    writePairArchive(path, pair);

    PFM_REQUIRE_THROWS_AS((void)pfm::loadPairArchiveSample(path), std::invalid_argument);
    const auto sample = pfm::loadPairArchiveSample(path, false);
    PFM_REQUIRE(sample.valid_mask.sum().item<int64_t>() == 0);
}

void pair_archive_dataset_rejects_shape_mismatch()
{
    TempPairArchiveDirectory temp("pfm_pair_archive_shape");
    const auto path = temp.path / "cache" / "train" / "source_00001" / "pair_000001_a.pt";
    auto pair = makeValidPairTensors();
    pair.valid_mask = torch::ones({3, 5}, torch::kBool);
    writePairArchive(path, pair);

    PFM_REQUIRE_THROWS_AS((void)pfm::loadPairArchiveSample(path), std::invalid_argument);
}

void pair_archive_dataset_rejects_missing_cache_dir()
{
    TempPairArchiveDirectory temp("pfm_pair_archive_missing");
    PFM_REQUIRE_THROWS_AS((void)pfm::discoverPairArchivePaths(temp.path / "missing"), std::invalid_argument);
}

} // namespace

void register_pair_archive_dataset_tests()
{
    register_test("pair_archive_dataset_discovers_sorted_limited_paths",
                  pair_archive_dataset_discovers_sorted_limited_paths);
    register_test("pair_archive_dataset_loads_valid_archive", pair_archive_dataset_loads_valid_archive);
    register_test("pair_archive_dataset_rejects_empty_valid_mask_by_default",
                  pair_archive_dataset_rejects_empty_valid_mask_by_default);
    register_test("pair_archive_dataset_rejects_shape_mismatch", pair_archive_dataset_rejects_shape_mismatch);
    register_test("pair_archive_dataset_rejects_missing_cache_dir", pair_archive_dataset_rejects_missing_cache_dir);
}
