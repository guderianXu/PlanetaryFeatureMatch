#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>

#include "data/synthetic_pair_dataset.h"
#include "dataloader/async_dataloader.h"
#include "dataloader/collator.h"
#include "dataloader/sampler.h"
#include "tests/test_harness.h"

namespace
{

class RangeDataset : public pfm::TensorDataset
{
  public:
    explicit RangeDataset(size_t count) : _count(count)
    {
    }

    size_t size() const override
    {
        return _count;
    }

    pfm::TensorBatch get(size_t index) override
    {
        if (index >= _count)
        {
            throw std::out_of_range("range dataset index out of range");
        }
        pfm::TensorBatch sample;
        sample["value"] = torch::full({1, 1}, static_cast<int64_t>(index), torch::kInt64);
        return sample;
    }

  private:
    size_t _count;
};

class ThrowingDataset : public pfm::TensorDataset
{
  public:
    size_t size() const override
    {
        return 4;
    }

    pfm::TensorBatch get(size_t index) override
    {
        if (index == 2)
        {
            throw std::runtime_error("dataset failure at index 2");
        }
        pfm::TensorBatch sample;
        sample["value"] = torch::full({1, 1}, static_cast<int64_t>(index), torch::kInt64);
        return sample;
    }
};

class BlockingDataset : public pfm::TensorDataset
{
  public:
    explicit BlockingDataset(size_t count) : _count(count)
    {
    }

    size_t size() const override
    {
        return _count;
    }

    pfm::TensorBatch get(size_t index) override
    {
        ++_started_loads;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        pfm::TensorBatch sample;
        sample["value"] = torch::full({1, 1}, static_cast<int64_t>(index), torch::kInt64);
        return sample;
    }

    size_t startedLoads() const
    {
        return _started_loads.load();
    }

  private:
    size_t _count;
    std::atomic<size_t> _started_loads{0};
};

class SlowFirstDataset : public pfm::TensorDataset
{
  public:
    size_t size() const override
    {
        return 4;
    }

    pfm::TensorBatch get(size_t index) override
    {
        if (index == 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
        pfm::TensorBatch sample;
        sample["value"] = torch::full({1, 1}, static_cast<int64_t>(index), torch::kInt64);
        return sample;
    }
};

class ResetThrowingSampler : public pfm::Sampler
{
  public:
    explicit ResetThrowingSampler(size_t count) : _count(count)
    {
    }

    std::vector<size_t> indices() const override
    {
        if (_called.exchange(true))
        {
            throw std::runtime_error("reset sampler failure");
        }
        std::vector<size_t> values(_count);
        for (size_t index = 0; index < _count; ++index)
        {
            values[index] = index;
        }
        return values;
    }

  private:
    size_t _count;
    mutable std::atomic<bool> _called{false};
};

static pfm::TensorBatchCollator valueCollator()
{
    return pfm::TensorBatchCollator({{"value", pfm::TensorLayout::Hw}});
}

static std::vector<int64_t> collectValues(pfm::AsyncDataLoader& loader)
{
    std::vector<int64_t> values;
    while (auto batch = loader.next())
    {
        const auto tensor = batch->at("value").reshape({-1}).to(torch::kCPU);
        for (int64_t index = 0; index < tensor.size(0); ++index)
        {
            values.push_back(tensor.index({index}).item<int64_t>());
        }
    }
    return values;
}

static void sequentialSamplerReturnsOrderedIndices()
{
    pfm::SequentialSampler sampler(4);

    auto indices = sampler.indices();

    PFM_REQUIRE(indices == std::vector<size_t>({0, 1, 2, 3}));
}

static void shuffleSamplerIsDeterministicForSeed()
{
    pfm::ShuffleSampler first(8, 42);
    pfm::ShuffleSampler second(8, 42);

    auto first_indices = first.indices();
    auto second_indices = second.indices();

    PFM_REQUIRE(first_indices == second_indices);
    PFM_REQUIRE(first_indices != std::vector<size_t>({0, 1, 2, 3, 4, 5, 6, 7}));
}

static void shuffleSamplerAdvancesOrderAcrossEpochs()
{
    pfm::ShuffleSampler sampler(16, 42);

    auto first_epoch = sampler.indices();
    auto second_epoch = sampler.indices();
    auto first_sorted = first_epoch;
    auto second_sorted = second_epoch;
    std::sort(first_sorted.begin(), first_sorted.end());
    std::sort(second_sorted.begin(), second_sorted.end());

    PFM_REQUIRE(first_sorted == second_sorted);
    PFM_REQUIRE(first_epoch != second_epoch);
}

static void shuffleSamplerMatchesPythonRandomShuffle()
{
    pfm::ShuffleSampler sampler(8, 42);

    PFM_REQUIRE(sampler.indices() == std::vector<size_t>({3, 4, 6, 7, 2, 5, 0, 1}));
    PFM_REQUIRE(sampler.indices() == std::vector<size_t>({3, 7, 2, 0, 4, 6, 5, 1}));
    PFM_REQUIRE(sampler.indices() == std::vector<size_t>({3, 5, 2, 4, 1, 6, 7, 0}));
}

static void splitIndicesCoverDatasetOnce()
{
    auto split = pfm::make_train_validation_test_split(10, 0.6, 0.2, 0.2, 7, false);
    std::vector<size_t> all;
    all.insert(all.end(), split.train.begin(), split.train.end());
    all.insert(all.end(), split.validation.begin(), split.validation.end());
    all.insert(all.end(), split.test.begin(), split.test.end());
    std::sort(all.begin(), all.end());

    PFM_REQUIRE(split.train.size() == 6);
    PFM_REQUIRE(split.validation.size() == 2);
    PFM_REQUIRE(split.test.size() == 2);
    PFM_REQUIRE(all == std::vector<size_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

static void splitRejectsInvalidRatios()
{
    PFM_REQUIRE_THROWS_AS(pfm::make_train_validation_test_split(10, 0.5, 0.5, 0.5, 1, false), std::invalid_argument);
    PFM_REQUIRE_THROWS_AS(pfm::make_train_validation_test_split(10, -0.1, 0.6, 0.5, 1, false), std::invalid_argument);
}

static void splitRejectsNonFiniteRatios()
{
    const auto quiet_nan = std::numeric_limits<double>::quiet_NaN();
    const auto infinity = std::numeric_limits<double>::infinity();

    PFM_REQUIRE_THROWS_AS(pfm::make_train_validation_test_split(10, quiet_nan, 0.5, 0.5, 1, false),
                          std::invalid_argument);
    PFM_REQUIRE_THROWS_AS(pfm::make_train_validation_test_split(10, 0.5, quiet_nan, 0.5, 1, false),
                          std::invalid_argument);
    PFM_REQUIRE_THROWS_AS(pfm::make_train_validation_test_split(10, 0.5, 0.5, quiet_nan, 1, false),
                          std::invalid_argument);
    PFM_REQUIRE_THROWS_AS(pfm::make_train_validation_test_split(10, infinity, 0.0, 0.0, 1, false),
                          std::invalid_argument);
    PFM_REQUIRE_THROWS_AS(pfm::make_train_validation_test_split(10, 0.0, infinity, 0.0, 1, false),
                          std::invalid_argument);
    PFM_REQUIRE_THROWS_AS(pfm::make_train_validation_test_split(10, 0.0, 0.0, infinity, 1, false),
                          std::invalid_argument);
}

static void collatorPadsChwHwAndHwcTensors()
{
    pfm::TensorBatch first;
    first["view"] = torch::ones({1, 2, 3}, torch::kFloat32);
    first["mask"] = torch::ones({2, 3}, torch::kFloat32);
    first["warp"] = torch::ones({2, 3, 2}, torch::kFloat32);
    first["warp"].index_put_({0, 0, 0}, 10.0F);
    first["warp"].index_put_({0, 0, 1}, 20.0F);

    pfm::TensorBatch second;
    second["view"] = torch::ones({1, 3, 2}, torch::kFloat32) * 2.0F;
    second["mask"] = torch::ones({3, 2}, torch::kFloat32) * 2.0F;
    second["warp"] = torch::ones({3, 2, 2}, torch::kFloat32) * 2.0F;

    pfm::TensorBatchCollator collator({
        {"view", pfm::TensorLayout::Chw},
        {"mask", pfm::TensorLayout::Hw},
        {"warp", pfm::TensorLayout::Hwc},
    });

    auto batch = collator.collate({first, second});

    PFM_REQUIRE(batch.at("view").sizes().equals(torch::IntArrayRef({2, 1, 3, 3})));
    PFM_REQUIRE(batch.at("mask").sizes().equals(torch::IntArrayRef({2, 3, 3})));
    PFM_REQUIRE(batch.at("warp").sizes().equals(torch::IntArrayRef({2, 3, 3, 2})));
    PFM_REQUIRE_CLOSE(batch.at("view").index({0, 0, 2, 2}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(batch.at("mask").index({1, 2, 2}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(batch.at("warp").index({0, 2, 2, 0}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(batch.at("warp").index({0, 0, 0, 0}).item<float>(), 10.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(batch.at("warp").index({0, 0, 0, 1}).item<float>(), 20.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(batch.at("view").index({1, 0, 0, 0}).item<float>(), 2.0F, 1.0e-6F);
}

static void collatorRejectsEmptyLayouts()
{
    PFM_REQUIRE_THROWS_AS(pfm::TensorBatchCollator({}), std::invalid_argument);
}

static void collatorRejectsEmptySampleList()
{
    pfm::TensorBatchCollator collator({{"view", pfm::TensorLayout::Chw}});

    PFM_REQUIRE_THROWS_AS(collator.collate({}), std::invalid_argument);
}

static void collatorRejectsMissingRequiredKey()
{
    pfm::TensorBatch sample;
    sample["view"] = torch::ones({1, 2, 2}, torch::kFloat32);
    pfm::TensorBatchCollator collator({{"view", pfm::TensorLayout::Chw}, {"mask", pfm::TensorLayout::Hw}});

    PFM_REQUIRE_THROWS_AS(collator.collate({sample}), std::invalid_argument);
}

static void collatorRejectsInvalidRanks()
{
    pfm::TensorBatch hw_sample;
    hw_sample["x"] = torch::ones({1, 2, 2}, torch::kFloat32);
    pfm::TensorBatch chw_sample;
    chw_sample["x"] = torch::ones({2, 2}, torch::kFloat32);
    pfm::TensorBatch hwc_sample;
    hwc_sample["x"] = torch::ones({1, 2, 2, 1}, torch::kFloat32);

    PFM_REQUIRE_THROWS_AS(pfm::TensorBatchCollator({{"x", pfm::TensorLayout::Hw}}).collate({hw_sample}),
                          std::invalid_argument);
    PFM_REQUIRE_THROWS_AS(pfm::TensorBatchCollator({{"x", pfm::TensorLayout::Chw}}).collate({chw_sample}),
                          std::invalid_argument);
    PFM_REQUIRE_THROWS_AS(pfm::TensorBatchCollator({{"x", pfm::TensorLayout::Hwc}}).collate({hwc_sample}),
                          std::invalid_argument);
}

static void collatorRejectsNonSpatialDimensionMismatch()
{
    pfm::TensorBatch first;
    first["chw"] = torch::ones({1, 2, 2}, torch::kFloat32);
    first["hwc"] = torch::ones({2, 2, 1}, torch::kFloat32);
    pfm::TensorBatch second;
    second["chw"] = torch::ones({2, 2, 2}, torch::kFloat32);
    second["hwc"] = torch::ones({2, 2, 2}, torch::kFloat32);

    PFM_REQUIRE_THROWS_AS(pfm::TensorBatchCollator({{"chw", pfm::TensorLayout::Chw}}).collate({first, second}),
                          std::invalid_argument);
    PFM_REQUIRE_THROWS_AS(pfm::TensorBatchCollator({{"hwc", pfm::TensorLayout::Hwc}}).collate({first, second}),
                          std::invalid_argument);
}

static void moveBatchToDevicePreservesKeysDeviceAndDtype()
{
    pfm::TensorBatch batch;
    batch["x"] = torch::ones({2, 2}, torch::kFloat32);
    batch["y"] = torch::zeros({1}, torch::kInt64);

    auto moved = pfm::moveBatchToDevice(batch, torch::Device(torch::kCPU), false);

    PFM_REQUIRE(moved.count("x") == 1);
    PFM_REQUIRE(moved.count("y") == 1);
    PFM_REQUIRE(moved.at("x").device().is_cpu());
    PFM_REQUIRE(moved.at("y").device().is_cpu());
    PFM_REQUIRE(moved.at("x").dtype() == torch::kFloat32);
    PFM_REQUIRE(moved.at("y").dtype() == torch::kInt64);
}

static void pinTensorBatchMemoryPinsCpuWhenSupported()
{
    pfm::TensorBatch batch;
    batch["x"] = torch::ones({2, 2}, torch::kFloat32);

    try
    {
        const auto pinned = pfm::pinTensorBatchMemory(batch);
        PFM_REQUIRE(pinned.at("x").device().is_cpu());
        PFM_REQUIRE(pinned.at("x").is_pinned());
    }
    catch (const std::runtime_error& error)
    {
        const std::string message(error.what());
        PFM_REQUIRE(message.find("failed to pin tensor batch memory") != std::string::npos);
    }
}

static void pinTensorBatchMemoryKeepsCudaTensorOnCuda()
{
    if (!torch::cuda::is_available())
    {
        return;
    }

    pfm::TensorBatch batch;
    batch["x"] = torch::ones({2, 2}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    const auto pinned = pfm::pinTensorBatchMemory(batch);

    PFM_REQUIRE(pinned.at("x").device().is_cuda());
}

static void asyncDataLoaderSynchronousReturnsSamplerOrder()
{
    pfm::DataLoaderOptions options;
    options.batch_size = 2;
    options.worker_count = 0;

    pfm::AsyncDataLoader loader(std::make_shared<RangeDataset>(5),
                                std::make_unique<pfm::SubsetSampler>(std::vector<size_t>({3, 1, 4, 0, 2})),
                                valueCollator(), options);

    PFM_REQUIRE(collectValues(loader) == std::vector<int64_t>({3, 1, 4, 0, 2}));
}

static void asyncDataLoaderDropLastSkipsIncompleteFinalBatch()
{
    pfm::DataLoaderOptions options;
    options.batch_size = 2;
    options.worker_count = 0;
    options.drop_last = true;

    pfm::AsyncDataLoader loader(std::make_shared<RangeDataset>(5), std::make_unique<pfm::SequentialSampler>(5),
                                valueCollator(), options);

    PFM_REQUIRE(collectValues(loader) == std::vector<int64_t>({0, 1, 2, 3}));
}

static void asyncDataLoaderAsyncReturnsAllSamples()
{
    pfm::DataLoaderOptions options;
    options.batch_size = 2;
    options.worker_count = 2;
    options.prefetch_batches = 2;

    pfm::AsyncDataLoader loader(std::make_shared<RangeDataset>(6), std::make_unique<pfm::SequentialSampler>(6),
                                valueCollator(), options);

    const auto values = collectValues(loader);
    PFM_REQUIRE(std::set<int64_t>(values.begin(), values.end()) == std::set<int64_t>({0, 1, 2, 3, 4, 5}));
}

static void asyncDataLoaderAsyncPreservesSamplerOrder()
{
    pfm::DataLoaderOptions options;
    options.batch_size = 1;
    options.worker_count = 2;
    options.prefetch_batches = 2;

    pfm::AsyncDataLoader loader(std::make_shared<SlowFirstDataset>(), std::make_unique<pfm::SequentialSampler>(4),
                                valueCollator(), options);

    PFM_REQUIRE(collectValues(loader) == std::vector<int64_t>({0, 1, 2, 3}));
}

static void asyncDataLoaderDatasetExceptionsSurfaceFromNext()
{
    pfm::DataLoaderOptions options;
    options.batch_size = 2;
    options.worker_count = 2;
    options.prefetch_batches = 2;

    pfm::AsyncDataLoader loader(std::make_shared<ThrowingDataset>(), std::make_unique<pfm::SequentialSampler>(4),
                                valueCollator(), options);

    bool thrown = false;
    try
    {
        while (loader.next())
        {
        }
    }
    catch (const std::runtime_error& error)
    {
        thrown = std::string(error.what()).find("dataset failure at index 2") != std::string::npos;
    }
    PFM_REQUIRE(thrown);
}

static void asyncDataLoaderResetIteratesAgainFromBeginning()
{
    pfm::DataLoaderOptions options;
    options.batch_size = 2;
    options.worker_count = 0;

    pfm::AsyncDataLoader loader(std::make_shared<RangeDataset>(4), std::make_unique<pfm::SequentialSampler>(4),
                                valueCollator(), options);

    PFM_REQUIRE(collectValues(loader) == std::vector<int64_t>({0, 1, 2, 3}));
    loader.reset();
    PFM_REQUIRE(collectValues(loader) == std::vector<int64_t>({0, 1, 2, 3}));
}

static void asyncDataLoaderBoundedPrefetchResetStopsBlockedWorkers()
{
    pfm::DataLoaderOptions options;
    options.batch_size = 1;
    options.worker_count = 2;
    options.prefetch_batches = 1;

    auto dataset = std::make_shared<BlockingDataset>(8);
    pfm::AsyncDataLoader loader(dataset, std::make_unique<pfm::SequentialSampler>(8), valueCollator(), options);

    for (size_t attempt = 0; attempt < 50 && dataset->startedLoads() < 3; ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    PFM_REQUIRE(dataset->startedLoads() >= 3);

    loader.reset();
    const auto values = collectValues(loader);
    PFM_REQUIRE(std::set<int64_t>(values.begin(), values.end()) == std::set<int64_t>({0, 1, 2, 3, 4, 5, 6, 7}));
}

static void asyncDataLoaderBoundedPrefetchDestructionStopsBlockedWorkers()
{
    auto dataset = std::make_shared<BlockingDataset>(8);
    {
        pfm::DataLoaderOptions options;
        options.batch_size = 1;
        options.worker_count = 2;
        options.prefetch_batches = 1;

        pfm::AsyncDataLoader loader(dataset, std::make_unique<pfm::SequentialSampler>(8), valueCollator(), options);

        for (size_t attempt = 0; attempt < 50 && dataset->startedLoads() < 3; ++attempt)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        PFM_REQUIRE(dataset->startedLoads() >= 3);
    }
}

static void asyncDataLoaderFailedAsyncResetLeavesLoaderExhausted()
{
    pfm::DataLoaderOptions options;
    options.batch_size = 1;
    options.worker_count = 1;
    options.prefetch_batches = 1;

    pfm::AsyncDataLoader loader(std::make_shared<RangeDataset>(3), std::make_unique<ResetThrowingSampler>(3),
                                valueCollator(), options);

    PFM_REQUIRE_THROWS_AS(loader.reset(), std::runtime_error);
    PFM_REQUIRE(!loader.next().has_value());
}

static void syntheticPairTensorDatasetReturnsTrainingKeys()
{
    std::vector<torch::Tensor> images = {torch::ones({1, 8, 8}, torch::kFloat32)};
    pfm::ImagePairAugmentationConfig augment_config;
    augment_config.profile = pfm::AugmentationProfile::Mild;

    pfm::SyntheticPairTensorDataset dataset(images, 2, augment_config);

    PFM_REQUIRE(dataset.size() == 2);
    auto sample = dataset.get(1);
    PFM_REQUIRE(sample.count("view_a") == 1);
    PFM_REQUIRE(sample.count("view_b") == 1);
    PFM_REQUIRE(sample.count("warp_a_to_b") == 1);
    PFM_REQUIRE(sample.count("valid_mask") == 1);
}

static void asyncDataLoaderPinMemoryPinsCpuWhenSupported()
{
    pfm::DataLoaderOptions options;
    options.batch_size = 2;
    options.worker_count = 0;
    options.pin_memory = true;

    pfm::AsyncDataLoader loader(std::make_shared<RangeDataset>(2), std::make_unique<pfm::SequentialSampler>(2),
                                valueCollator(), options);

    try
    {
        const auto batch = loader.next();
        PFM_REQUIRE(batch.has_value());
        PFM_REQUIRE(batch->at("value").device().is_cpu());
        PFM_REQUIRE(batch->at("value").is_pinned());
    }
    catch (const std::runtime_error& error)
    {
        const std::string message(error.what());
        PFM_REQUIRE(message.find("failed to pin tensor batch memory") != std::string::npos);
    }
}

} // namespace

void register_dataloader_tests()
{
    register_test("sequential sampler returns ordered indices", sequentialSamplerReturnsOrderedIndices);
    register_test("shuffle sampler is deterministic for seed", shuffleSamplerIsDeterministicForSeed);
    register_test("shuffle sampler advances order across epochs", shuffleSamplerAdvancesOrderAcrossEpochs);
    register_test("shuffle sampler matches python random shuffle", shuffleSamplerMatchesPythonRandomShuffle);
    register_test("split indices cover dataset once", splitIndicesCoverDatasetOnce);
    register_test("split rejects invalid ratios", splitRejectsInvalidRatios);
    register_test("split rejects non-finite ratios", splitRejectsNonFiniteRatios);
    register_test("collator pads chw hw and hwc tensors", collatorPadsChwHwAndHwcTensors);
    register_test("collator rejects empty layouts", collatorRejectsEmptyLayouts);
    register_test("collator rejects empty sample list", collatorRejectsEmptySampleList);
    register_test("collator rejects missing required key", collatorRejectsMissingRequiredKey);
    register_test("collator rejects invalid ranks", collatorRejectsInvalidRanks);
    register_test("collator rejects non-spatial dimension mismatch", collatorRejectsNonSpatialDimensionMismatch);
    register_test("move batch to device preserves keys device and dtype", moveBatchToDevicePreservesKeysDeviceAndDtype);
    register_test("pin tensor batch memory pins cpu when supported", pinTensorBatchMemoryPinsCpuWhenSupported);
    register_test("pin tensor batch memory keeps cuda tensor on cuda", pinTensorBatchMemoryKeepsCudaTensorOnCuda);
    register_test("async data loader synchronous returns sampler order", asyncDataLoaderSynchronousReturnsSamplerOrder);
    register_test("async data loader drop last skips incomplete final batch",
                  asyncDataLoaderDropLastSkipsIncompleteFinalBatch);
    register_test("async data loader async returns all samples", asyncDataLoaderAsyncReturnsAllSamples);
    register_test("async data loader async preserves sampler order", asyncDataLoaderAsyncPreservesSamplerOrder);
    register_test("async data loader dataset exceptions surface from next",
                  asyncDataLoaderDatasetExceptionsSurfaceFromNext);
    register_test("async data loader reset iterates again from beginning",
                  asyncDataLoaderResetIteratesAgainFromBeginning);
    register_test("async data loader bounded prefetch reset stops blocked workers",
                  asyncDataLoaderBoundedPrefetchResetStopsBlockedWorkers);
    register_test("async data loader bounded prefetch destruction stops blocked workers",
                  asyncDataLoaderBoundedPrefetchDestructionStopsBlockedWorkers);
    register_test("async data loader failed async reset leaves loader exhausted",
                  asyncDataLoaderFailedAsyncResetLeavesLoaderExhausted);
    register_test("synthetic pair tensor dataset returns training keys", syntheticPairTensorDatasetReturnsTrainingKeys);
    register_test("async data loader pin memory pins cpu when supported", asyncDataLoaderPinMemoryPinsCpuWhenSupported);
}
