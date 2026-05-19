# Reusable Training Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reusable C++ deep-learning training infrastructure for asynchronous data loading, image-pair augmentation, structured logging, optional NVML GPU metrics, and phased trainer integration.

**Architecture:** Add focused submodules under the existing `modules/` tree: `runtime` for concurrency, `augment` for reusable image-pair transforms, `logging` for progress/CSV/GPU metrics, and `dataloader` for `TensorBatch` datasets, samplers, collation, async prefetch, and pinned memory. Keep existing synthetic pair and checkpoint formats compatible while progressively replacing trainer internals.

**Tech Stack:** C++17, LibTorch, OpenCV, CMake, optional NVIDIA NVML, current custom `pfm_tests` test harness.

---

## File Structure

### Create

- `modules/runtime/blocking_queue.h` — bounded blocking FIFO queue template with close semantics.
- `modules/runtime/thread_pool.h` — fixed worker pool API.
- `modules/runtime/thread_pool.cpp` — worker lifecycle, job execution, first-exception propagation.
- `modules/runtime/runtime_test.cpp` — queue and thread pool tests.
- `modules/dataloader/tensor_batch.h` — `TensorBatch`, layout enum, batch move/pin helpers declarations.
- `modules/dataloader/dataset.h` — `TensorDataset` interface.
- `modules/dataloader/sampler.h` — sampler interfaces and train/validation/test split API.
- `modules/dataloader/sampler.cpp` — sequential, shuffle, subset, split implementations.
- `modules/dataloader/collator.h` — `TensorBatchCollator` declarations.
- `modules/dataloader/collator.cpp` — layout-aware padding and stacking.
- `modules/dataloader/async_dataloader.h` — async loader API.
- `modules/dataloader/async_dataloader.cpp` — synchronous and async prefetch implementation.
- `modules/dataloader/pinned_memory.cpp` — pinned-memory and device transfer helpers.
- `modules/dataloader/dataloader_test.cpp` — sampler, split, collator, loader, pinned-memory tests.
- `modules/augment/augmentation_profile.h` — reusable profile enum and conversion helpers.
- `modules/augment/transform_sampler.h` — transform parameter structures and deterministic sampler.
- `modules/augment/transform_sampler.cpp` — deterministic profile-specific parameter sampling.
- `modules/augment/image_pair_augmentor.h` — `ImagePairAugmentationConfig`, `ImagePairSample`, augmentor API.
- `modules/augment/image_pair_augmentor.cpp` — migrated synthetic pair generation and photometric augmentation.
- `modules/augment/augment_test.cpp` — profile, transform, image-pair output tests.
- `modules/logging/training_metric.h` — metric record and GPU metric structs.
- `modules/logging/gpu_metric_provider.h` — provider interface, null provider, factory declarations.
- `modules/logging/gpu_metric_provider.cpp` — null provider and default factory.
- `modules/logging/nvml_gpu_metric_provider.cpp` — optional NVML implementation.
- `modules/logging/progress_logger.h` — console progress logger declarations.
- `modules/logging/progress_logger.cpp` — progress bar formatting and output.
- `modules/logging/csv_metric_logger.h` — CSV logger declarations.
- `modules/logging/csv_metric_logger.cpp` — stable header and row writer.
- `modules/logging/metric_logger_group.h` — fan-out logger declarations.
- `modules/logging/metric_logger_group.cpp` — fan-out logger implementation.
- `modules/logging/logging_test.cpp` — progress, CSV, null GPU provider tests.

### Modify

- `CMakeLists.txt` — add `pfm_runtime`, `pfm_dataloader`, `pfm_augment`, `pfm_logging`, optional NVML, and new tests.
- `tests/test_main.cpp` — register runtime, dataloader, augment, logging tests.
- `modules/data/synthetic_pair.h` — convert existing synthetic profile/config to new augment config.
- `modules/data/synthetic_pair.cpp` — replace implementation with compatibility wrapper around `ImagePairAugmentor`.
- `modules/data/synthetic_pair_test.cpp` — keep existing tests passing; add compatibility assertion if needed.
- `modules/train/trainer.h` — add high-level logging config fields only.
- `modules/train/trainer.cpp` — replace direct progress `std::cout` with structured loggers; later use DataLoader.
- `modules/train/trainer_test.cpp` — add CSV logger integration test and keep training smoke tests passing.
- `modules/cli/commands.h` — add high-level `log_csv` train option if required.
- `modules/cli/commands.cpp` — parse `--log-csv` into `TrainConfig` if required.
- `modules/cli/commands_test.cpp` — verify `--log-csv` parsing if added.
- `README.md` — document reusable modules, progress bar, CSV logs, optional GPU metrics.
- `docs/training.md` — document DataLoader/splits/samplers/logs and high matcher loss diagnosis.
- `docs/usage.md` — document any new high-level logging option.

---

## Task 1: Add Runtime BlockingQueue

**Files:**
- Create: `modules/runtime/blocking_queue.h`
- Create: `modules/runtime/runtime_test.cpp`
- Modify: `tests/test_main.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing queue tests**

Add this file:

```cpp
#include "runtime/blocking_queue.h"
#include "tests/test_harness.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace {

static void blocking_queue_preserves_fifo_order() {
    pfm::BlockingQueue<int> queue(2);

    queue.push(3);
    queue.push(7);

    auto first = queue.pop();
    auto second = queue.pop();

    PFM_REQUIRE(first.has_value());
    PFM_REQUIRE(second.has_value());
    PFM_REQUIRE(*first == 3);
    PFM_REQUIRE(*second == 7);
}

static void blocking_queue_close_wakes_waiting_consumer() {
    pfm::BlockingQueue<int> queue(1);
    std::atomic<bool> popped{false};

    std::thread consumer([&]() {
        auto value = queue.pop();
        PFM_REQUIRE(!value.has_value());
        popped.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    queue.close();
    consumer.join();

    PFM_REQUIRE(popped.load());
}

static void blocking_queue_rejects_zero_capacity() {
    PFM_REQUIRE_THROWS_AS(pfm::BlockingQueue<int>(0), std::invalid_argument);
}

}  // namespace

void register_runtime_tests() {
    register_test("blocking queue preserves fifo order", blocking_queue_preserves_fifo_order);
    register_test("blocking queue close wakes waiting consumer", blocking_queue_close_wakes_waiting_consumer);
    register_test("blocking queue rejects zero capacity", blocking_queue_rejects_zero_capacity);
}
```

Modify `tests/test_main.cpp`:

```cpp
void register_runtime_tests();
```

and call it before data/model tests:

```cpp
register_runtime_tests();
```

Modify `CMakeLists.txt` only enough to compile the test source:

```cmake
add_library(pfm_runtime STATIC
)

target_include_directories(pfm_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)

target_compile_options(pfm_runtime PRIVATE -Wall -Wextra -Wpedantic)
```

Add `modules/runtime/runtime_test.cpp` to `pfm_tests` sources and link `pfm_tests` to `pfm_runtime`.

- [ ] **Step 2: Run red check**

Run:

```bash
cmake --build . -j$(nproc)
```

Expected: compile fails because `runtime/blocking_queue.h` does not exist.

- [ ] **Step 3: Implement BlockingQueue**

Create `modules/runtime/blocking_queue.h`:

```cpp
#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>

namespace pfm {

template <typename T>
class BlockingQueue {
public:
    /// Creates a bounded blocking queue.
    /// \param capacity Maximum number of queued values.
    /// \throws std::invalid_argument if capacity is zero.
    explicit BlockingQueue(size_t capacity) : _capacity(capacity) {
        if (_capacity == 0) {
            throw std::invalid_argument("blocking queue capacity must be positive");
        }
    }

    /// Pushes one value, blocking while the queue is full.
    /// \param value Value to enqueue.
    /// \throws std::runtime_error if the queue is closed.
    void push(T value) {
        std::unique_lock<std::mutex> lock(_mutex);
        _not_full.wait(lock, [&]() { return _closed || _queue.size() < _capacity; });
        if (_closed) {
            throw std::runtime_error("cannot push to a closed blocking queue");
        }
        _queue.push(std::move(value));
        _not_empty.notify_one();
    }

    /// Pops one value, blocking while the queue is empty and open.
    /// \return Empty optional when the queue is closed and drained.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(_mutex);
        _not_empty.wait(lock, [&]() { return _closed || !_queue.empty(); });
        if (_queue.empty()) {
            return std::nullopt;
        }
        auto value = std::move(_queue.front());
        _queue.pop();
        _not_full.notify_one();
        return value;
    }

    /// Closes the queue and wakes all waiting producers and consumers.
    void close() {
        std::lock_guard<std::mutex> lock(_mutex);
        _closed = true;
        _not_empty.notify_all();
        _not_full.notify_all();
    }

    /// Returns the current queued value count.
    /// \return Queue size at the time the lock is held.
    size_t size() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }

private:
    size_t _capacity;
    mutable std::mutex _mutex;
    std::condition_variable _not_empty;
    std::condition_variable _not_full;
    std::queue<T> _queue;
    bool _closed = false;
};

}  // namespace pfm
```

- [ ] **Step 4: Run green check**

Run:

```bash
cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass and the output includes the three new blocking queue `PASS` lines.

- [ ] **Step 5: Commit if authorized**

If the user has explicitly authorized commits for this implementation session, run:

```bash
git add CMakeLists.txt tests/test_main.cpp modules/runtime/blocking_queue.h modules/runtime/runtime_test.cpp
git commit -m "Add reusable blocking queue"
```

If commits are not authorized, do not commit.

---

## Task 2: Add Runtime ThreadPool

**Files:**
- Create: `modules/runtime/thread_pool.h`
- Create: `modules/runtime/thread_pool.cpp`
- Modify: `modules/runtime/runtime_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing thread pool tests**

Append to `modules/runtime/runtime_test.cpp`:

```cpp
#include "runtime/thread_pool.h"

static void thread_pool_runs_all_jobs() {
    pfm::ThreadPool pool(3, 8);
    std::atomic<int> counter{0};

    for (int i = 0; i < 10; ++i) {
        pool.enqueue([&]() { counter.fetch_add(1); });
    }
    pool.close();
    pool.join();

    PFM_REQUIRE(counter.load() == 10);
}

static void thread_pool_rethrows_worker_exception() {
    pfm::ThreadPool pool(2, 4);
    pool.enqueue([]() { throw std::runtime_error("worker failed"); });
    pool.close();

    PFM_REQUIRE_THROWS_AS(pool.join(), std::runtime_error);
}

static void thread_pool_rejects_zero_workers() {
    PFM_REQUIRE_THROWS_AS(pfm::ThreadPool(0, 4), std::invalid_argument);
}
```

Register them:

```cpp
register_test("thread pool runs all jobs", thread_pool_runs_all_jobs);
register_test("thread pool rethrows worker exception", thread_pool_rethrows_worker_exception);
register_test("thread pool rejects zero workers", thread_pool_rejects_zero_workers);
```

Add `modules/runtime/thread_pool.cpp` to `pfm_runtime` in `CMakeLists.txt`.

- [ ] **Step 2: Run red check**

Run:

```bash
cmake --build . -j$(nproc)
```

Expected: compile fails because `runtime/thread_pool.h` does not exist.

- [ ] **Step 3: Implement ThreadPool**

Create `modules/runtime/thread_pool.h`:

```cpp
#pragma once

#include "runtime/blocking_queue.h"

#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace pfm {

class ThreadPool {
public:
    /// Creates a fixed-size worker pool.
    /// \param worker_count Number of worker threads.
    /// \param queue_capacity Maximum queued jobs.
    /// \throws std::invalid_argument if either argument is zero.
    ThreadPool(size_t worker_count, size_t queue_capacity);

    /// Closes the queue and joins workers without throwing.
    ~ThreadPool();

    /// Enqueues one job for worker execution.
    /// \param job Function to run on a worker.
    /// \throws std::runtime_error if the pool is closed.
    void enqueue(std::function<void()> job);

    /// Stops accepting new jobs after queued jobs drain.
    void close();

    /// Joins workers and rethrows the first worker exception, if any.
    void join();

private:
    void workerLoop();
    void captureException();

    BlockingQueue<std::function<void()>> _jobs;
    std::vector<std::thread> _workers;
    std::mutex _exception_mutex;
    std::exception_ptr _first_exception;
    bool _joined = false;
};

}  // namespace pfm
```

Create `modules/runtime/thread_pool.cpp`:

```cpp
#include "runtime/thread_pool.h"

#include <stdexcept>

namespace pfm {

ThreadPool::ThreadPool(size_t worker_count, size_t queue_capacity) : _jobs(queue_capacity) {
    if (worker_count == 0) {
        throw std::invalid_argument("thread pool worker count must be positive");
    }
    _workers.reserve(worker_count);
    for (size_t index = 0; index < worker_count; ++index) {
        _workers.emplace_back([this]() { workerLoop(); });
    }
}

ThreadPool::~ThreadPool() {
    close();
    try {
        join();
    } catch (...) {
    }
}

void ThreadPool::enqueue(std::function<void()> job) {
    _jobs.push(std::move(job));
}

void ThreadPool::close() {
    _jobs.close();
}

void ThreadPool::join() {
    if (!_joined) {
        for (auto& worker : _workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        _joined = true;
    }
    if (_first_exception) {
        std::rethrow_exception(_first_exception);
    }
}

void ThreadPool::workerLoop() {
    while (auto job = _jobs.pop()) {
        try {
            (*job)();
        } catch (...) {
            captureException();
        }
    }
}

void ThreadPool::captureException() {
    std::lock_guard<std::mutex> lock(_exception_mutex);
    if (!_first_exception) {
        _first_exception = std::current_exception();
    }
}

}  // namespace pfm
```

- [ ] **Step 4: Run green check**

Run:

```bash
cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass and runtime tests include thread pool PASS lines.

- [ ] **Step 5: Commit if authorized**

```bash
git add CMakeLists.txt modules/runtime/thread_pool.h modules/runtime/thread_pool.cpp modules/runtime/runtime_test.cpp
git commit -m "Add reusable thread pool"
```

Skip the commit if commits are not authorized.

---

## Task 3: Add TensorBatch, Samplers, and Dataset Splits

**Files:**
- Create: `modules/dataloader/tensor_batch.h`
- Create: `modules/dataloader/dataset.h`
- Create: `modules/dataloader/sampler.h`
- Create: `modules/dataloader/sampler.cpp`
- Create: `modules/dataloader/dataloader_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Write failing sampler and split tests**

Create `modules/dataloader/dataloader_test.cpp`:

```cpp
#include "dataloader/sampler.h"
#include "tests/test_harness.h"

#include <algorithm>
#include <set>
#include <vector>

namespace {

static void sequential_sampler_returns_ordered_indices() {
    pfm::SequentialSampler sampler(4);

    auto indices = sampler.indices();

    PFM_REQUIRE(indices == std::vector<size_t>({0, 1, 2, 3}));
}

static void shuffle_sampler_is_deterministic_for_seed() {
    pfm::ShuffleSampler first(8, 42);
    pfm::ShuffleSampler second(8, 42);

    auto first_indices = first.indices();
    auto second_indices = second.indices();

    PFM_REQUIRE(first_indices == second_indices);
    PFM_REQUIRE(first_indices != std::vector<size_t>({0, 1, 2, 3, 4, 5, 6, 7}));
}

static void split_indices_cover_dataset_once() {
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

static void split_rejects_invalid_ratios() {
    PFM_REQUIRE_THROWS_AS(
        pfm::make_train_validation_test_split(10, 0.5, 0.5, 0.5, 1, false),
        std::invalid_argument);
}

}  // namespace

void register_dataloader_tests() {
    register_test("sequential sampler returns ordered indices", sequential_sampler_returns_ordered_indices);
    register_test("shuffle sampler is deterministic for seed", shuffle_sampler_is_deterministic_for_seed);
    register_test("split indices cover dataset once", split_indices_cover_dataset_once);
    register_test("split rejects invalid ratios", split_rejects_invalid_ratios);
}
```

Modify `tests/test_main.cpp`:

```cpp
void register_dataloader_tests();
```

and call:

```cpp
register_dataloader_tests();
```

Modify `CMakeLists.txt` to add:

```cmake
add_library(pfm_dataloader STATIC
    modules/dataloader/sampler.cpp
)

target_include_directories(pfm_dataloader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)

target_link_libraries(pfm_dataloader PUBLIC pfm_runtime ${TORCH_LIBRARIES})
target_compile_options(pfm_dataloader PRIVATE -Wall -Wextra -Wpedantic)
```

Link `pfm` and `pfm_tests` with `pfm_dataloader`; add `modules/dataloader/dataloader_test.cpp` to `pfm_tests` sources.

- [ ] **Step 2: Run red check**

Run:

```bash
cmake --build . -j$(nproc)
```

Expected: compile fails because `dataloader/sampler.h` does not exist.

- [ ] **Step 3: Implement TensorBatch, Dataset, and Samplers**

Create `modules/dataloader/tensor_batch.h`:

```cpp
#pragma once

#include <string>
#include <unordered_map>

#include <torch/torch.h>

namespace pfm {

using TensorBatch = std::unordered_map<std::string, torch::Tensor>;

enum class TensorLayout {
    Hw,
    Chw,
    Hwc
};

}  // namespace pfm
```

Create `modules/dataloader/dataset.h`:

```cpp
#pragma once

#include "dataloader/tensor_batch.h"

#include <cstddef>

namespace pfm {

class TensorDataset {
public:
    /// Destroys the dataset.
    virtual ~TensorDataset() = default;

    /// Returns the number of samples.
    /// \return Dataset size.
    virtual size_t size() const = 0;

    /// Loads one sample by index.
    /// \param index Sample index.
    /// \return Tensor batch containing one unbatched sample.
    /// \throws std::out_of_range if index is invalid.
    virtual TensorBatch get(size_t index) = 0;
};

}  // namespace pfm
```

Create `modules/dataloader/sampler.h`:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pfm {

class Sampler {
public:
    /// Destroys the sampler.
    virtual ~Sampler() = default;

    /// Returns the index order for one epoch.
    /// \return Sample indices.
    virtual std::vector<size_t> indices() const = 0;
};

class SequentialSampler : public Sampler {
public:
    /// Creates an ordered sampler for count samples.
    /// \param count Number of samples.
    explicit SequentialSampler(size_t count);

    /// Returns indices 0..count-1.
    /// \return Ordered indices.
    std::vector<size_t> indices() const override;

private:
    size_t _count;
};

class ShuffleSampler : public Sampler {
public:
    /// Creates a deterministic shuffled sampler.
    /// \param count Number of samples.
    /// \param seed Random seed.
    ShuffleSampler(size_t count, uint64_t seed);

    /// Returns shuffled indices.
    /// \return Deterministically shuffled indices.
    std::vector<size_t> indices() const override;

private:
    size_t _count;
    uint64_t _seed;
};

class SubsetSampler : public Sampler {
public:
    /// Creates a sampler over an explicit index list.
    /// \param indices Indices to return.
    explicit SubsetSampler(std::vector<size_t> indices);

    /// Returns the configured subset.
    /// \return Subset indices.
    std::vector<size_t> indices() const override;

private:
    std::vector<size_t> _indices;
};

struct DatasetSplit {
    std::vector<size_t> train;
    std::vector<size_t> validation;
    std::vector<size_t> test;
};

/// Splits dataset indices into train, validation, and test subsets.
/// \param count Number of samples.
/// \param train_ratio Train split ratio.
/// \param validation_ratio Validation split ratio.
/// \param test_ratio Test split ratio.
/// \param seed Shuffle seed used when shuffle is true.
/// \param shuffle Whether to shuffle before splitting.
/// \return Three disjoint index vectors covering every sample exactly once.
/// \throws std::invalid_argument if ratios are negative or do not sum to one.
DatasetSplit make_train_validation_test_split(
    size_t count,
    double train_ratio,
    double validation_ratio,
    double test_ratio,
    uint64_t seed,
    bool shuffle);

}  // namespace pfm
```

Create `modules/dataloader/sampler.cpp`:

```cpp
#include "dataloader/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace pfm {
namespace {

std::vector<size_t> ordered_indices(size_t count) {
    std::vector<size_t> result(count);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

void require_valid_ratios(double train_ratio, double validation_ratio, double test_ratio) {
    if (train_ratio < 0.0 || validation_ratio < 0.0 || test_ratio < 0.0) {
        throw std::invalid_argument("dataset split ratios must be non-negative");
    }
    const auto total = train_ratio + validation_ratio + test_ratio;
    if (std::abs(total - 1.0) > 1.0e-9) {
        throw std::invalid_argument("dataset split ratios must sum to one");
    }
}

}  // namespace

SequentialSampler::SequentialSampler(size_t count) : _count(count) {}

std::vector<size_t> SequentialSampler::indices() const {
    return ordered_indices(_count);
}

ShuffleSampler::ShuffleSampler(size_t count, uint64_t seed) : _count(count), _seed(seed) {}

std::vector<size_t> ShuffleSampler::indices() const {
    auto result = ordered_indices(_count);
    std::mt19937_64 generator(_seed);
    std::shuffle(result.begin(), result.end(), generator);
    return result;
}

SubsetSampler::SubsetSampler(std::vector<size_t> indices) : _indices(std::move(indices)) {}

std::vector<size_t> SubsetSampler::indices() const {
    return _indices;
}

DatasetSplit make_train_validation_test_split(
    size_t count,
    double train_ratio,
    double validation_ratio,
    double test_ratio,
    uint64_t seed,
    bool shuffle
) {
    require_valid_ratios(train_ratio, validation_ratio, test_ratio);
    auto indices = ordered_indices(count);
    if (shuffle) {
        std::mt19937_64 generator(seed);
        std::shuffle(indices.begin(), indices.end(), generator);
    }

    const auto train_count = static_cast<size_t>(std::floor(static_cast<double>(count) * train_ratio));
    const auto validation_count = static_cast<size_t>(std::floor(static_cast<double>(count) * validation_ratio));
    const auto test_count = count - train_count - validation_count;

    DatasetSplit split;
    split.train.assign(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(train_count));
    split.validation.assign(
        indices.begin() + static_cast<std::ptrdiff_t>(train_count),
        indices.begin() + static_cast<std::ptrdiff_t>(train_count + validation_count));
    split.test.assign(indices.end() - static_cast<std::ptrdiff_t>(test_count), indices.end());
    return split;
}

}  // namespace pfm
```

- [ ] **Step 4: Run green check**

Run:

```bash
cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass and dataloader sampler/split tests pass.

- [ ] **Step 5: Commit if authorized**

```bash
git add CMakeLists.txt tests/test_main.cpp modules/dataloader/tensor_batch.h modules/dataloader/dataset.h modules/dataloader/sampler.h modules/dataloader/sampler.cpp modules/dataloader/dataloader_test.cpp
git commit -m "Add dataloader samplers and dataset splits"
```

Skip the commit if commits are not authorized.

---

## Task 4: Add TensorBatch Collator and Pinned/Device Helpers

**Files:**
- Create: `modules/dataloader/collator.h`
- Create: `modules/dataloader/collator.cpp`
- Create: `modules/dataloader/pinned_memory.cpp`
- Modify: `modules/dataloader/tensor_batch.h`
- Modify: `modules/dataloader/dataloader_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing collator and pinned helper tests**

Append to `modules/dataloader/dataloader_test.cpp`:

```cpp
#include "dataloader/collator.h"

static void collator_pads_chw_hw_and_hwc_tensors() {
    pfm::TensorBatch first;
    first["view"] = torch::ones({1, 2, 3}, torch::kFloat32);
    first["mask"] = torch::ones({2, 3}, torch::kFloat32);
    first["warp"] = torch::ones({2, 3, 2}, torch::kFloat32);

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
    PFM_REQUIRE_CLOSE(batch.at("view")[0][0][2][2].item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(batch.at("view")[1][0][0][0].item<float>(), 2.0F, 1.0e-6F);
}

static void collator_rejects_missing_required_key() {
    pfm::TensorBatch sample;
    sample["view"] = torch::ones({1, 2, 2}, torch::kFloat32);
    pfm::TensorBatchCollator collator({{"view", pfm::TensorLayout::Chw}, {"mask", pfm::TensorLayout::Hw}});

    PFM_REQUIRE_THROWS_AS(collator.collate({sample}), std::invalid_argument);
}

static void move_batch_to_device_preserves_keys() {
    pfm::TensorBatch batch;
    batch["x"] = torch::ones({2, 2}, torch::kFloat32);

    auto moved = pfm::move_batch_to_device(batch, torch::Device(torch::kCPU), false);

    PFM_REQUIRE(moved.count("x") == 1);
    PFM_REQUIRE(moved.at("x").device().is_cpu());
}
```

Register:

```cpp
register_test("collator pads chw hw and hwc tensors", collator_pads_chw_hw_and_hwc_tensors);
register_test("collator rejects missing required key", collator_rejects_missing_required_key);
register_test("move batch to device preserves keys", move_batch_to_device_preserves_keys);
```

Add `modules/dataloader/collator.cpp` and `modules/dataloader/pinned_memory.cpp` to `pfm_dataloader`.

- [ ] **Step 2: Run red check**

Run:

```bash
cmake --build . -j$(nproc)
```

Expected: compile fails because `dataloader/collator.h` does not exist.

- [ ] **Step 3: Implement collator and helpers**

Modify `modules/dataloader/tensor_batch.h`:

```cpp
#pragma once

#include <string>
#include <unordered_map>

#include <torch/torch.h>

namespace pfm {

using TensorBatch = std::unordered_map<std::string, torch::Tensor>;

enum class TensorLayout {
    Hw,
    Chw,
    Hwc
};

/// Copies every tensor in a batch to the requested device.
/// \param batch Source tensor batch.
/// \param device Target torch device.
/// \param non_blocking Whether to request non-blocking copies.
/// \return Batch with copied tensors.
TensorBatch move_batch_to_device(const TensorBatch& batch, const torch::Device& device, bool non_blocking);

/// Copies every CPU tensor in a batch to pinned memory.
/// \param batch Source tensor batch.
/// \return Batch whose tensors use pinned CPU memory when supported.
/// \throws std::runtime_error if pinned allocation fails.
TensorBatch pin_tensor_batch_memory(const TensorBatch& batch);

}  // namespace pfm
```

Create `modules/dataloader/collator.h`:

```cpp
#pragma once

#include "dataloader/tensor_batch.h"

#include <string>
#include <utility>
#include <vector>

namespace pfm {

class TensorBatchCollator {
public:
    /// Creates a collator with required keys and tensor layouts.
    /// \param layouts Required key-layout pairs.
    explicit TensorBatchCollator(std::vector<std::pair<std::string, TensorLayout>> layouts);

    /// Pads and stacks samples into one batch.
    /// \param samples Unbatched sample maps.
    /// \return Batched tensor map.
    /// \throws std::invalid_argument if samples are empty, keys are missing, or layouts are invalid.
    TensorBatch collate(const std::vector<TensorBatch>& samples) const;

private:
    std::vector<std::pair<std::string, TensorLayout>> _layouts;
};

}  // namespace pfm
```

Create `modules/dataloader/collator.cpp` with layout-aware padding:

```cpp
#include "dataloader/collator.h"

#include <algorithm>
#include <stdexcept>

#include <torch/torch.h>

namespace pfm {
namespace {

std::pair<int64_t, int64_t> spatial_size(const torch::Tensor& tensor, TensorLayout layout) {
    if (layout == TensorLayout::Hw) {
        if (tensor.dim() != 2) {
            throw std::invalid_argument("HW tensor must have rank 2");
        }
        return {tensor.size(0), tensor.size(1)};
    }
    if (layout == TensorLayout::Chw) {
        if (tensor.dim() != 3) {
            throw std::invalid_argument("CHW tensor must have rank 3");
        }
        return {tensor.size(1), tensor.size(2)};
    }
    if (layout == TensorLayout::Hwc) {
        if (tensor.dim() != 3) {
            throw std::invalid_argument("HWC tensor must have rank 3");
        }
        return {tensor.size(0), tensor.size(1)};
    }
    throw std::invalid_argument("unsupported tensor layout");
}

torch::Tensor pad_to_spatial_size(const torch::Tensor& tensor, TensorLayout layout, int64_t height, int64_t width) {
    const auto current = spatial_size(tensor, layout);
    const auto pad_h = height - current.first;
    const auto pad_w = width - current.second;
    if (pad_h < 0 || pad_w < 0) {
        throw std::invalid_argument("target spatial size cannot be smaller than tensor size");
    }
    if (layout == TensorLayout::Hw) {
        return torch::constant_pad_nd(tensor, {0, pad_w, 0, pad_h}, 0);
    }
    if (layout == TensorLayout::Chw) {
        return torch::constant_pad_nd(tensor, {0, pad_w, 0, pad_h}, 0);
    }
    return torch::constant_pad_nd(tensor, {0, 0, 0, pad_w, 0, pad_h}, 0);
}

}  // namespace

TensorBatchCollator::TensorBatchCollator(std::vector<std::pair<std::string, TensorLayout>> layouts)
    : _layouts(std::move(layouts)) {
    if (_layouts.empty()) {
        throw std::invalid_argument("tensor batch collator requires at least one layout");
    }
}

TensorBatch TensorBatchCollator::collate(const std::vector<TensorBatch>& samples) const {
    if (samples.empty()) {
        throw std::invalid_argument("cannot collate an empty sample list");
    }
    TensorBatch result;
    for (const auto& [key, layout] : _layouts) {
        int64_t height = 0;
        int64_t width = 0;
        for (const auto& sample : samples) {
            auto it = sample.find(key);
            if (it == sample.end()) {
                throw std::invalid_argument("tensor batch sample is missing required key: " + key);
            }
            const auto size = spatial_size(it->second, layout);
            height = std::max(height, size.first);
            width = std::max(width, size.second);
        }
        std::vector<torch::Tensor> padded;
        padded.reserve(samples.size());
        for (const auto& sample : samples) {
            padded.push_back(pad_to_spatial_size(sample.at(key), layout, height, width));
        }
        result[key] = torch::stack(padded, 0).contiguous();
    }
    return result;
}

}  // namespace pfm
```

Create `modules/dataloader/pinned_memory.cpp`:

```cpp
#include "dataloader/tensor_batch.h"

#include <stdexcept>

namespace pfm {

TensorBatch move_batch_to_device(const TensorBatch& batch, const torch::Device& device, bool non_blocking) {
    TensorBatch result;
    for (const auto& [key, tensor] : batch) {
        result[key] = tensor.to(device, tensor.dtype(), non_blocking, false);
    }
    return result;
}

TensorBatch pin_tensor_batch_memory(const TensorBatch& batch) {
    TensorBatch result;
    try {
        for (const auto& [key, tensor] : batch) {
            result[key] = tensor.device().is_cpu() ? tensor.pin_memory() : tensor;
        }
    } catch (const c10::Error& error) {
        throw std::runtime_error(std::string("failed to pin tensor batch memory: ") + error.what());
    }
    return result;
}

}  // namespace pfm
```

- [ ] **Step 4: Run green check**

Run:

```bash
cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass and collator/device helper tests pass.

- [ ] **Step 5: Commit if authorized**

```bash
git add CMakeLists.txt modules/dataloader/tensor_batch.h modules/dataloader/collator.h modules/dataloader/collator.cpp modules/dataloader/pinned_memory.cpp modules/dataloader/dataloader_test.cpp
git commit -m "Add tensor batch collation utilities"
```

Skip the commit if commits are not authorized.

---

## Task 5: Add AsyncDataLoader

**Files:**
- Create: `modules/dataloader/async_dataloader.h`
- Create: `modules/dataloader/async_dataloader.cpp`
- Modify: `modules/dataloader/dataloader_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing synchronous and async loader tests**

Append to `modules/dataloader/dataloader_test.cpp`:

```cpp
#include "dataloader/async_dataloader.h"
#include "dataloader/dataset.h"

class RangeTensorDataset : public pfm::TensorDataset {
public:
    explicit RangeTensorDataset(size_t count) : _count(count) {}

    size_t size() const override {
        return _count;
    }

    pfm::TensorBatch get(size_t index) override {
        if (index >= _count) {
            throw std::out_of_range("range dataset index out of range");
        }
        pfm::TensorBatch sample;
        sample["x"] = torch::tensor({static_cast<float>(index)}, torch::kFloat32);
        return sample;
    }

private:
    size_t _count;
};

class ThrowingTensorDataset : public pfm::TensorDataset {
public:
    size_t size() const override {
        return 2;
    }

    pfm::TensorBatch get(size_t index) override {
        if (index == 1) {
            throw std::runtime_error("dataset failed");
        }
        pfm::TensorBatch sample;
        sample["x"] = torch::tensor({0.0F}, torch::kFloat32);
        return sample;
    }
};

static void async_dataloader_synchronous_mode_returns_batches() {
    auto dataset = std::make_shared<RangeTensorDataset>(3);
    auto sampler = std::make_unique<pfm::SequentialSampler>(dataset->size());
    pfm::TensorBatchCollator collator({{"x", pfm::TensorLayout::Hw}});
    pfm::DataLoaderOptions options;
    options.batch_size = 2;
    options.worker_count = 0;

    pfm::AsyncDataLoader loader(dataset, std::move(sampler), collator, options);

    auto first = loader.next();
    auto second = loader.next();
    auto end = loader.next();

    PFM_REQUIRE(first.has_value());
    PFM_REQUIRE(second.has_value());
    PFM_REQUIRE(!end.has_value());
    PFM_REQUIRE(first->at("x").size(0) == 2);
    PFM_REQUIRE(second->at("x").size(0) == 1);
}

static void async_dataloader_worker_mode_returns_all_batches() {
    auto dataset = std::make_shared<RangeTensorDataset>(4);
    auto sampler = std::make_unique<pfm::SequentialSampler>(dataset->size());
    pfm::TensorBatchCollator collator({{"x", pfm::TensorLayout::Hw}});
    pfm::DataLoaderOptions options;
    options.batch_size = 2;
    options.worker_count = 2;
    options.prefetch_batches = 2;

    pfm::AsyncDataLoader loader(dataset, std::move(sampler), collator, options);

    int batches = 0;
    int rows = 0;
    while (auto batch = loader.next()) {
        ++batches;
        rows += static_cast<int>(batch->at("x").size(0));
    }

    PFM_REQUIRE(batches == 2);
    PFM_REQUIRE(rows == 4);
}

static void async_dataloader_propagates_dataset_errors() {
    auto dataset = std::make_shared<ThrowingTensorDataset>();
    auto sampler = std::make_unique<pfm::SequentialSampler>(dataset->size());
    pfm::TensorBatchCollator collator({{"x", pfm::TensorLayout::Hw}});
    pfm::DataLoaderOptions options;
    options.batch_size = 1;
    options.worker_count = 1;

    pfm::AsyncDataLoader loader(dataset, std::move(sampler), collator, options);

    auto first = loader.next();
    PFM_REQUIRE(first.has_value());
    PFM_REQUIRE_THROWS_AS(loader.next(), std::runtime_error);
}
```

Register:

```cpp
register_test("async dataloader synchronous mode returns batches", async_dataloader_synchronous_mode_returns_batches);
register_test("async dataloader worker mode returns all batches", async_dataloader_worker_mode_returns_all_batches);
register_test("async dataloader propagates dataset errors", async_dataloader_propagates_dataset_errors);
```

Add `modules/dataloader/async_dataloader.cpp` to `pfm_dataloader`.

- [ ] **Step 2: Run red check**

Run:

```bash
cmake --build . -j$(nproc)
```

Expected: compile fails because `dataloader/async_dataloader.h` does not exist.

- [ ] **Step 3: Implement AsyncDataLoader**

Create `modules/dataloader/async_dataloader.h`:

```cpp
#pragma once

#include "dataloader/collator.h"
#include "dataloader/dataset.h"
#include "dataloader/sampler.h"
#include "runtime/blocking_queue.h"

#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace pfm {

struct DataLoaderOptions {
    size_t batch_size = 1;
    size_t worker_count = 0;
    size_t prefetch_batches = 2;
    bool drop_last = false;
    bool pin_memory = false;
};

class AsyncDataLoader {
public:
    /// Creates a DataLoader over a dataset and sampler.
    /// \param dataset Dataset used to load samples.
    /// \param sampler Sampler that defines one epoch of indices.
    /// \param collator Batch collator.
    /// \param options Loader options.
    /// \throws std::invalid_argument if batch size or prefetch count is invalid.
    AsyncDataLoader(
        std::shared_ptr<TensorDataset> dataset,
        std::unique_ptr<Sampler> sampler,
        TensorBatchCollator collator,
        DataLoaderOptions options);

    /// Stops workers and drains resources.
    ~AsyncDataLoader();

    /// Restarts iteration from the beginning of the sampler order.
    void reset();

    /// Returns the next batch or empty optional at epoch end.
    /// \return Optional tensor batch.
    /// \throws std::runtime_error when worker loading failed.
    std::optional<TensorBatch> next();

private:
    struct BatchResult {
        std::optional<TensorBatch> batch;
        std::exception_ptr error;
    };

    std::optional<TensorBatch> nextSynchronous();
    void startWorkers();
    void stopWorkers();
    void workerLoop();
    BatchResult loadBatch(size_t start_index);

    std::shared_ptr<TensorDataset> _dataset;
    std::unique_ptr<Sampler> _sampler;
    TensorBatchCollator _collator;
    DataLoaderOptions _options;
    std::vector<size_t> _indices;
    size_t _cursor = 0;
    std::unique_ptr<BlockingQueue<size_t>> _jobs;
    std::unique_ptr<BlockingQueue<BatchResult>> _results;
    std::vector<std::thread> _workers;
};

}  // namespace pfm
```

Create `modules/dataloader/async_dataloader.cpp`:

```cpp
#include "dataloader/async_dataloader.h"

#include <algorithm>
#include <stdexcept>

namespace pfm {

AsyncDataLoader::AsyncDataLoader(
    std::shared_ptr<TensorDataset> dataset,
    std::unique_ptr<Sampler> sampler,
    TensorBatchCollator collator,
    DataLoaderOptions options
)
    : _dataset(std::move(dataset)),
      _sampler(std::move(sampler)),
      _collator(std::move(collator)),
      _options(options) {
    if (!_dataset || !_sampler) {
        throw std::invalid_argument("async dataloader requires dataset and sampler");
    }
    if (_options.batch_size == 0) {
        throw std::invalid_argument("async dataloader batch size must be positive");
    }
    if (_options.worker_count > 0 && _options.prefetch_batches == 0) {
        throw std::invalid_argument("async dataloader prefetch batches must be positive");
    }
    reset();
}

AsyncDataLoader::~AsyncDataLoader() {
    stopWorkers();
}

void AsyncDataLoader::reset() {
    stopWorkers();
    _indices = _sampler->indices();
    _cursor = 0;
    if (_options.worker_count > 0) {
        startWorkers();
    }
}

std::optional<TensorBatch> AsyncDataLoader::next() {
    if (_options.worker_count == 0) {
        return nextSynchronous();
    }
    auto result = _results->pop();
    if (!result.has_value()) {
        return std::nullopt;
    }
    if (result->error) {
        std::rethrow_exception(result->error);
    }
    return std::move(result->batch);
}

std::optional<TensorBatch> AsyncDataLoader::nextSynchronous() {
    if (_cursor >= _indices.size()) {
        return std::nullopt;
    }
    auto result = loadBatch(_cursor);
    _cursor += _options.batch_size;
    if (result.error) {
        std::rethrow_exception(result.error);
    }
    return std::move(result.batch);
}

void AsyncDataLoader::startWorkers() {
    _jobs = std::make_unique<BlockingQueue<size_t>>(_options.prefetch_batches);
    _results = std::make_unique<BlockingQueue<BatchResult>>(_options.prefetch_batches);
    for (size_t start = 0; start < _indices.size(); start += _options.batch_size) {
        const auto remaining = _indices.size() - start;
        if (_options.drop_last && remaining < _options.batch_size) {
            break;
        }
        _jobs->push(start);
    }
    _jobs->close();
    _workers.reserve(_options.worker_count);
    for (size_t index = 0; index < _options.worker_count; ++index) {
        _workers.emplace_back([this]() { workerLoop(); });
    }
}

void AsyncDataLoader::stopWorkers() {
    if (_jobs) {
        _jobs->close();
    }
    for (auto& worker : _workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    _workers.clear();
    if (_results) {
        _results->close();
    }
}

void AsyncDataLoader::workerLoop() {
    while (auto start = _jobs->pop()) {
        _results->push(loadBatch(*start));
    }
}

AsyncDataLoader::BatchResult AsyncDataLoader::loadBatch(size_t start_index) {
    try {
        const auto end = std::min(start_index + _options.batch_size, _indices.size());
        if (_options.drop_last && end - start_index < _options.batch_size) {
            return BatchResult{std::nullopt, nullptr};
        }
        std::vector<TensorBatch> samples;
        samples.reserve(end - start_index);
        for (size_t offset = start_index; offset < end; ++offset) {
            samples.push_back(_dataset->get(_indices[offset]));
        }
        auto batch = _collator.collate(samples);
        if (_options.pin_memory) {
            batch = pin_tensor_batch_memory(batch);
        }
        return BatchResult{std::move(batch), nullptr};
    } catch (...) {
        return BatchResult{std::nullopt, std::current_exception()};
    }
}

}  // namespace pfm
```

- [ ] **Step 4: Run green check**

Run:

```bash
cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass and async dataloader tests pass.

- [ ] **Step 5: Commit if authorized**

```bash
git add CMakeLists.txt modules/dataloader/async_dataloader.h modules/dataloader/async_dataloader.cpp modules/dataloader/dataloader_test.cpp
git commit -m "Add asynchronous tensor dataloader"
```

Skip the commit if commits are not authorized.

---

## Task 6: Add Logging Module with Console, CSV, and GPU Provider Interface

**Files:**
- Create: `modules/logging/training_metric.h`
- Create: `modules/logging/gpu_metric_provider.h`
- Create: `modules/logging/gpu_metric_provider.cpp`
- Create: `modules/logging/progress_logger.h`
- Create: `modules/logging/progress_logger.cpp`
- Create: `modules/logging/csv_metric_logger.h`
- Create: `modules/logging/csv_metric_logger.cpp`
- Create: `modules/logging/metric_logger_group.h`
- Create: `modules/logging/metric_logger_group.cpp`
- Create: `modules/logging/logging_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Write failing logging tests**

Create `modules/logging/logging_test.cpp`:

```cpp
#include "logging/csv_metric_logger.h"
#include "logging/gpu_metric_provider.h"
#include "logging/metric_logger_group.h"
#include "logging/progress_logger.h"
#include "tests/test_harness.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

pfm::TrainingMetric sample_metric() {
    pfm::TrainingMetric metric;
    metric.epoch = 2;
    metric.total_epochs = 10;
    metric.iteration = 3;
    metric.total_iterations = 5;
    metric.images_seen = 12;
    metric.total_images = 20;
    metric.learning_rate = 0.0003;
    metric.elapsed_seconds = 1.25;
    metric.values["loss_total"] = 4.5;
    metric.values["matcher_loss"] = 3.5;
    metric.values["dense_loss"] = 0.25;
    metric.values["offset_error_px"] = 12.0;
    return metric;
}

static void console_progress_logger_renders_core_fields() {
    std::ostringstream stream;
    pfm::ConsoleProgressLogger logger(stream, 20);

    logger.logIteration(sample_metric());
    logger.logEpochSummary(sample_metric());

    const auto text = stream.str();
    PFM_REQUIRE(text.find("epoch 2/10") != std::string::npos);
    PFM_REQUIRE(text.find("3/5") != std::string::npos);
    PFM_REQUIRE(text.find("loss=4.5") != std::string::npos);
    PFM_REQUIRE(text.find("matcher=3.5") != std::string::npos);
    PFM_REQUIRE(text.find("offset_px=12") != std::string::npos);
}

static void csv_metric_logger_writes_header_and_rows() {
    const auto path = std::filesystem::temp_directory_path() / "pfm_metric_logger_test.csv";
    std::filesystem::remove(path);

    pfm::CsvMetricLogger logger(path.string(), {"loss_total", "matcher_loss", "offset_error_px"});
    logger.logIteration(sample_metric());
    logger.flush();

    std::ifstream input(path);
    std::string header;
    std::string row;
    std::getline(input, header);
    std::getline(input, row);

    PFM_REQUIRE(header == "epoch,total_epochs,iteration,total_iterations,images_seen,total_images,learning_rate,elapsed_seconds,loss_total,matcher_loss,offset_error_px");
    PFM_REQUIRE(row.find("2,10,3,5,12,20") == 0);
    std::filesystem::remove(path);
}

static void null_gpu_metric_provider_returns_empty_values() {
    pfm::NullGpuMetricProvider provider;

    auto metrics = provider.sample();

    PFM_REQUIRE(!metrics.utilization_percent.has_value());
    PFM_REQUIRE(!metrics.power_watts.has_value());
}

}  // namespace

void register_logging_tests() {
    register_test("console progress logger renders core fields", console_progress_logger_renders_core_fields);
    register_test("csv metric logger writes header and rows", csv_metric_logger_writes_header_and_rows);
    register_test("null gpu metric provider returns empty values", null_gpu_metric_provider_returns_empty_values);
}
```

Modify `tests/test_main.cpp`:

```cpp
void register_logging_tests();
```

and call:

```cpp
register_logging_tests();
```

Modify `CMakeLists.txt` to add `pfm_logging` with the logging `.cpp` files and add `modules/logging/logging_test.cpp` to `pfm_tests`.

- [ ] **Step 2: Run red check**

Run:

```bash
cmake --build . -j$(nproc)
```

Expected: compile fails because logging headers do not exist.

- [ ] **Step 3: Implement logging interfaces and sinks**

Create `modules/logging/training_metric.h`:

```cpp
#pragma once

#include <optional>
#include <string>
#include <unordered_map>

namespace pfm {

struct GpuMetrics {
    std::optional<double> utilization_percent;
    std::optional<double> power_watts;
};

struct TrainingMetric {
    int epoch = 0;
    int total_epochs = 0;
    int iteration = 0;
    int total_iterations = 0;
    int images_seen = 0;
    int total_images = 0;
    double learning_rate = 0.0;
    double elapsed_seconds = 0.0;
    std::unordered_map<std::string, double> values;
};

class TrainingMetricLogger {
public:
    /// Destroys the logger.
    virtual ~TrainingMetricLogger() = default;

    /// Logs one training iteration.
    /// \param metric Metric record to log.
    virtual void logIteration(const TrainingMetric& metric) = 0;

    /// Logs one epoch summary.
    /// \param metric Last metric or summary metric for the epoch.
    virtual void logEpochSummary(const TrainingMetric& metric) = 0;

    /// Flushes buffered output.
    virtual void flush() = 0;
};

}  // namespace pfm
```

Create `modules/logging/gpu_metric_provider.h` and `.cpp`:

```cpp
#pragma once

#include "logging/training_metric.h"

#include <memory>

namespace pfm {

class GpuMetricProvider {
public:
    /// Destroys the provider.
    virtual ~GpuMetricProvider() = default;

    /// Samples current GPU metrics.
    /// \return Utilization and power values when available.
    virtual GpuMetrics sample() = 0;
};

class NullGpuMetricProvider : public GpuMetricProvider {
public:
    /// Returns empty GPU metric values.
    /// \return Empty metrics.
    GpuMetrics sample() override;
};

/// Creates the default GPU metric provider for this build.
/// \return NVML provider when available, otherwise null provider.
std::unique_ptr<GpuMetricProvider> make_default_gpu_metric_provider();

}  // namespace pfm
```

```cpp
#include "logging/gpu_metric_provider.h"

namespace pfm {

GpuMetrics NullGpuMetricProvider::sample() {
    return GpuMetrics{};
}

std::unique_ptr<GpuMetricProvider> make_default_gpu_metric_provider() {
    return std::make_unique<NullGpuMetricProvider>();
}

}  // namespace pfm
```

Create `modules/logging/progress_logger.h` and `.cpp`:

```cpp
#pragma once

#include "logging/training_metric.h"

#include <iosfwd>

namespace pfm {

class ConsoleProgressLogger : public TrainingMetricLogger {
public:
    /// Creates a console progress logger.
    /// \param stream Output stream.
    /// \param bar_width Progress bar width in characters.
    ConsoleProgressLogger(std::ostream& stream, int bar_width);

    /// Logs one iteration as a single progress line.
    /// \param metric Metric record.
    void logIteration(const TrainingMetric& metric) override;

    /// Logs one epoch summary line.
    /// \param metric Summary metric record.
    void logEpochSummary(const TrainingMetric& metric) override;

    /// Flushes the stream.
    void flush() override;

private:
    std::ostream& _stream;
    int _bar_width;
};

}  // namespace pfm
```

```cpp
#include "logging/progress_logger.h"

#include <algorithm>
#include <ostream>

namespace pfm {
namespace {

double metric_value(const TrainingMetric& metric, const std::string& key) {
    const auto it = metric.values.find(key);
    return it == metric.values.end() ? 0.0 : it->second;
}

}  // namespace

ConsoleProgressLogger::ConsoleProgressLogger(std::ostream& stream, int bar_width)
    : _stream(stream), _bar_width(std::max(1, bar_width)) {}

void ConsoleProgressLogger::logIteration(const TrainingMetric& metric) {
    const auto filled = metric.total_iterations > 0
                            ? std::min(_bar_width, metric.iteration * _bar_width / metric.total_iterations)
                            : 0;
    _stream << '\r' << "epoch " << metric.epoch << '/' << metric.total_epochs << " [";
    for (int index = 0; index < _bar_width; ++index) {
        _stream << (index < filled ? '=' : '-');
    }
    _stream << "] " << metric.iteration << '/' << metric.total_iterations
            << " loss=" << metric_value(metric, "loss_total")
            << " matcher=" << metric_value(metric, "matcher_loss")
            << " dense=" << metric_value(metric, "dense_loss")
            << " offset_px=" << metric_value(metric, "offset_error_px")
            << " lr=" << metric.learning_rate;
    const auto gpu_util = metric.values.find("gpu_utilization_percent");
    if (gpu_util != metric.values.end()) {
        _stream << " gpu=" << gpu_util->second << '%';
    }
    const auto power = metric.values.find("gpu_power_watts");
    if (power != metric.values.end()) {
        _stream << " power=" << power->second << 'W';
    }
    _stream.flush();
}

void ConsoleProgressLogger::logEpochSummary(const TrainingMetric& metric) {
    _stream << "\nepoch summary: epoch=" << metric.epoch << '/' << metric.total_epochs
            << " elapsed=" << metric.elapsed_seconds << "s\n";
}

void ConsoleProgressLogger::flush() {
    _stream.flush();
}

}  // namespace pfm
```

Create `modules/logging/csv_metric_logger.h/.cpp` and `metric_logger_group.h/.cpp` following the interfaces in the spec. Ensure `CsvMetricLogger` writes the exact header used in the test and `MetricLoggerGroup` forwards `logIteration`, `logEpochSummary`, and `flush` to owned loggers.

- [ ] **Step 4: Run green check**

Run:

```bash
cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass and logging tests pass.

- [ ] **Step 5: Commit if authorized**

```bash
git add CMakeLists.txt tests/test_main.cpp modules/logging modules/logging/logging_test.cpp
git commit -m "Add reusable training metric loggers"
```

Skip the commit if commits are not authorized.

---

## Task 7: Add Optional NVML CMake Hook

**Files:**
- Create: `modules/logging/nvml_gpu_metric_provider.cpp`
- Modify: `CMakeLists.txt`
- Modify: `modules/logging/gpu_metric_provider.cpp`
- Modify: `README.md`

- [ ] **Step 1: Write failing CMake/test expectation**

Add a small compile-time behavior test to `modules/logging/logging_test.cpp`:

```cpp
static void default_gpu_metric_provider_is_constructible() {
    auto provider = pfm::make_default_gpu_metric_provider();

    PFM_REQUIRE(provider != nullptr);
    auto metrics = provider->sample();
    PFM_REQUIRE(!metrics.utilization_percent.has_value() || metrics.utilization_percent.value() >= 0.0);
}
```

Register it:

```cpp
register_test("default gpu metric provider is constructible", default_gpu_metric_provider_is_constructible);
```

- [ ] **Step 2: Run red check if NVML source is referenced before creation**

Modify `CMakeLists.txt` to add:

```cmake
option(PFM_ENABLE_NVML "Enable NVML GPU metrics" ON)
```

and configure a `PFM_HAS_NVML` definition only when the library is found. If adding the NVML source before creating it, run:

```bash
cmake .. -DBUILD_TESTS=ON
```

Expected: configure or build fails because `modules/logging/nvml_gpu_metric_provider.cpp` does not exist.

- [ ] **Step 3: Implement optional NVML provider**

Create `modules/logging/nvml_gpu_metric_provider.cpp`:

```cpp
#include "logging/gpu_metric_provider.h"

#ifdef PFM_HAS_NVML
#include <nvml.h>
#endif

#include <iostream>
#include <memory>

namespace pfm {

#ifdef PFM_HAS_NVML
class NvmlGpuMetricProvider : public GpuMetricProvider {
public:
    NvmlGpuMetricProvider() {
        if (nvmlInit_v2() != NVML_SUCCESS) {
            _available = false;
        }
        if (_available && nvmlDeviceGetHandleByIndex_v2(0, &_device) != NVML_SUCCESS) {
            _available = false;
        }
    }

    ~NvmlGpuMetricProvider() override {
        if (_available) {
            nvmlShutdown();
        }
    }

    GpuMetrics sample() override {
        if (!_available) {
            return GpuMetrics{};
        }
        GpuMetrics result;
        nvmlUtilization_t utilization{};
        if (nvmlDeviceGetUtilizationRates(_device, &utilization) == NVML_SUCCESS) {
            result.utilization_percent = static_cast<double>(utilization.gpu);
        }
        unsigned int milliwatts = 0;
        if (nvmlDeviceGetPowerUsage(_device, &milliwatts) == NVML_SUCCESS) {
            result.power_watts = static_cast<double>(milliwatts) / 1000.0;
        }
        return result;
    }

private:
    bool _available = true;
    nvmlDevice_t _device{};
};
#endif

std::unique_ptr<GpuMetricProvider> make_nvml_gpu_metric_provider() {
#ifdef PFM_HAS_NVML
    return std::make_unique<NvmlGpuMetricProvider>();
#else
    return std::make_unique<NullGpuMetricProvider>();
#endif
}

}  // namespace pfm
```

Update `gpu_metric_provider.h` with:

```cpp
/// Creates an NVML GPU metric provider when compiled with NVML, otherwise a null provider.
/// \return GPU metric provider instance.
std::unique_ptr<GpuMetricProvider> make_nvml_gpu_metric_provider();
```

Update `gpu_metric_provider.cpp`:

```cpp
std::unique_ptr<GpuMetricProvider> make_default_gpu_metric_provider() {
    return make_nvml_gpu_metric_provider();
}
```

Update CMake to find NVML without failing:

```cmake
if(PFM_ENABLE_NVML)
    find_library(NVML_LIBRARY nvidia-ml)
    find_path(NVML_INCLUDE_DIR nvml.h)
endif()

if(NVML_LIBRARY AND NVML_INCLUDE_DIR)
    target_sources(pfm_logging PRIVATE modules/logging/nvml_gpu_metric_provider.cpp)
    target_include_directories(pfm_logging PRIVATE ${NVML_INCLUDE_DIR})
    target_link_libraries(pfm_logging PUBLIC ${NVML_LIBRARY})
    target_compile_definitions(pfm_logging PRIVATE PFM_HAS_NVML=1)
else()
    target_sources(pfm_logging PRIVATE modules/logging/nvml_gpu_metric_provider.cpp)
endif()
```

- [ ] **Step 4: Run green check with and without NVML option**

Run:

```bash
cmake .. -DBUILD_TESTS=ON -DPFM_ENABLE_NVML=ON && cmake --build . -j$(nproc) && ./pfm_tests
cmake .. -DBUILD_TESTS=ON -DPFM_ENABLE_NVML=OFF && cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: both configurations pass. If NVML is unavailable, tests still pass with null metrics.

- [ ] **Step 5: Restore default configure and commit if authorized**

Run:

```bash
cmake .. -DBUILD_TESTS=ON
```

Then, if commits are authorized:

```bash
git add CMakeLists.txt README.md modules/logging/gpu_metric_provider.h modules/logging/gpu_metric_provider.cpp modules/logging/nvml_gpu_metric_provider.cpp modules/logging/logging_test.cpp
git commit -m "Add optional NVML GPU metrics"
```

Skip the commit if commits are not authorized.

---

## Task 8: Add Augmentation Module and Synthetic Pair Compatibility Wrapper

**Files:**
- Create: `modules/augment/augmentation_profile.h`
- Create: `modules/augment/transform_sampler.h`
- Create: `modules/augment/transform_sampler.cpp`
- Create: `modules/augment/image_pair_augmentor.h`
- Create: `modules/augment/image_pair_augmentor.cpp`
- Create: `modules/augment/augment_test.cpp`
- Modify: `modules/data/synthetic_pair.cpp`
- Modify: `modules/data/synthetic_pair.h`
- Modify: `modules/data/synthetic_pair_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_main.cpp`

- [ ] **Step 1: Write failing augmentation module tests**

Create `modules/augment/augment_test.cpp`:

```cpp
#include "augment/image_pair_augmentor.h"
#include "augment/transform_sampler.h"
#include "tests/test_harness.h"

namespace {

static void transform_sampler_is_deterministic() {
    pfm::ImagePairAugmentationConfig config;
    config.profile = pfm::AugmentationProfile::Mixed;
    config.source_index = 3;
    config.variant_index = 4;
    config.seed = 11;

    auto first = pfm::sample_image_pair_transform(config);
    auto second = pfm::sample_image_pair_transform(config);

    PFM_REQUIRE_CLOSE(first.rotation_degrees, second.rotation_degrees, 1.0e-6);
    PFM_REQUIRE_CLOSE(first.scale, second.scale, 1.0e-6);
    PFM_REQUIRE_CLOSE(first.brightness, second.brightness, 1.0e-6);
}

static void image_pair_augmentor_returns_current_training_keys() {
    auto image = torch::linspace(0.0, 1.0, 64, torch::kFloat32).reshape({1, 8, 8});
    pfm::ImagePairAugmentationConfig config;
    config.profile = pfm::AugmentationProfile::Mild;
    config.source_index = 1;
    config.variant_index = 2;

    pfm::ImagePairAugmentor augmentor(config);
    auto sample = augmentor.augment(image);

    PFM_REQUIRE(sample.view_a.sizes().equals(torch::IntArrayRef({1, 8, 8})));
    PFM_REQUIRE(sample.view_b.sizes().equals(torch::IntArrayRef({1, 8, 8})));
    PFM_REQUIRE(sample.warp_a_to_b.sizes().equals(torch::IntArrayRef({8, 8, 2})));
    PFM_REQUIRE(sample.valid_mask.sizes().equals(torch::IntArrayRef({8, 8})));
    PFM_REQUIRE(sample.view_a.dtype() == torch::kFloat32);
}

}  // namespace

void register_augment_tests() {
    register_test("transform sampler is deterministic", transform_sampler_is_deterministic);
    register_test("image pair augmentor returns current training keys", image_pair_augmentor_returns_current_training_keys);
}
```

Modify `tests/test_main.cpp` to declare and call `register_augment_tests()`. Add `pfm_augment` target and `modules/augment/augment_test.cpp` to CMake.

- [ ] **Step 2: Run red check**

Run:

```bash
cmake --build . -j$(nproc)
```

Expected: compile fails because `augment/image_pair_augmentor.h` does not exist.

- [ ] **Step 3: Implement augmentation module by moving current logic**

Create `modules/augment/augmentation_profile.h`:

```cpp
#pragma once

namespace pfm {

enum class AugmentationProfile {
    Mixed,
    Mild,
    Medium,
    Hard,
    Extreme
};

}  // namespace pfm
```

Create `modules/augment/image_pair_augmentor.h`:

```cpp
#pragma once

#include "augment/augmentation_profile.h"

#include <cstdint>

#include <torch/torch.h>

namespace pfm {

struct ImagePairAugmentationConfig {
    AugmentationProfile profile = AugmentationProfile::Mixed;
    double max_rotation_degrees = 20.0;
    double max_translation_fraction = 0.12;
    double min_scale = 0.9;
    double max_scale = 1.1;
    double perspective_strength = 0.0;
    double brightness = 0.0;
    double contrast = 1.0;
    double gamma = 1.0;
    double shadow_strength = 0.0;
    double noise_stddev = 0.0;
    int source_index = 0;
    int variant_index = 0;
    uint64_t seed = 0;
    double extreme_pair_ratio = 0.2;
};

struct ImagePairSample {
    torch::Tensor view_a;
    torch::Tensor view_b;
    torch::Tensor warp_a_to_b;
    torch::Tensor valid_mask;
};

class ImagePairAugmentor {
public:
    /// Creates an image-pair augmentor.
    /// \param config Augmentation configuration.
    explicit ImagePairAugmentor(ImagePairAugmentationConfig config);

    /// Generates a synthetic image pair and dense correspondence supervision.
    /// \param image Source CHW float image.
    /// \return Augmented image-pair sample.
    /// \throws std::invalid_argument if image shape or dtype is unsupported.
    ImagePairSample augment(const torch::Tensor& image) const;

private:
    ImagePairAugmentationConfig _config;
};

}  // namespace pfm
```

Create `modules/augment/transform_sampler.h/.cpp` with `ImagePairTransformParameters` containing `rotation_degrees`, `translation_x`, `translation_y`, `scale`, `brightness`, `contrast`, `gamma`, `shadow_strength`, and `noise_stddev`. Move the deterministic profile parameter calculation from `modules/data/synthetic_pair.cpp` into `sample_image_pair_transform()`.

Create `modules/augment/image_pair_augmentor.cpp` by moving the current implementation body from `make_synthetic_pair()` into `ImagePairAugmentor::augment()`. Keep the same affine warp, valid mask, gamma, shadow, and noise behavior so existing synthetic pair tests remain valid.

Modify `modules/data/synthetic_pair.cpp` so `make_synthetic_pair()` converts `SyntheticPairConfig` to `ImagePairAugmentationConfig`, calls `ImagePairAugmentor`, and converts `ImagePairSample` back to `SyntheticPair`.

- [ ] **Step 4: Run green check**

Run:

```bash
cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass, including existing synthetic pair/cache tests and new augment tests.

- [ ] **Step 5: Commit if authorized**

```bash
git add CMakeLists.txt tests/test_main.cpp modules/augment modules/data/synthetic_pair.h modules/data/synthetic_pair.cpp modules/data/synthetic_pair_test.cpp
git commit -m "Extract reusable image pair augmentation module"
```

Skip the commit if commits are not authorized.

---

## Task 9: Integrate Logging into Trainer and CLI

**Files:**
- Modify: `modules/train/trainer.h`
- Modify: `modules/train/trainer.cpp`
- Modify: `modules/train/trainer_test.cpp`
- Modify: `modules/cli/commands.h`
- Modify: `modules/cli/commands.cpp`
- Modify: `modules/cli/commands_test.cpp`
- Modify: `docs/usage.md`

- [ ] **Step 1: Write failing CLI and trainer CSV tests**

Add to `modules/cli/commands_test.cpp`:

```cpp
static void parse_train_log_csv_option() {
    auto command = pfm::parse_command({
        "pfm", "train",
        "--image-dir", "images",
        "--checkpoint", "model.pt",
        "--log-csv", "train_log.csv"
    });

    PFM_REQUIRE(command.train.log_csv == "train_log.csv");
}
```

Register it near other train parse tests:

```cpp
register_test("parse train log csv option", parse_train_log_csv_option);
```

Add to `modules/train/trainer_test.cpp` using existing temp training helpers:

```cpp
static void trainer_writes_csv_log_when_requested() {
    TempTrainingDirectory temp_dir("pfm_trainer_csv_log");
    require_image_written(temp_dir.file("image_a.png"), 0);
    auto config = tiny_config(temp_dir);
    config.log_csv = temp_dir.file("train_log.csv");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    std::ifstream input(config.log_csv);
    std::string header;
    std::getline(input, header);
    PFM_REQUIRE(header.find("loss_total") != std::string::npos);
    PFM_REQUIRE(header.find("graph_matching_loss") != std::string::npos);
    PFM_REQUIRE(header.find("offset_error_px") != std::string::npos);
}
```

Register it:

```cpp
register_test("trainer writes csv log when requested", trainer_writes_csv_log_when_requested);
```

- [ ] **Step 2: Run red check**

Run:

```bash
cmake --build . -j$(nproc)
```

Expected: compile fails because `TrainConfig` and parsed train command do not have `log_csv`.

- [ ] **Step 3: Add high-level log option and structured trainer logging**

Modify `TrainConfig` in `modules/train/trainer.h`:

```cpp
std::string log_csv;
```

Modify the train command struct in `modules/cli/commands.h` with:

```cpp
std::string log_csv;
```

Modify train CLI parse in `modules/cli/commands.cpp`:

```cpp
train->add_option("--log-csv", result.train.log_csv, "Write per-iteration training metrics to a CSV file");
```

When building `TrainConfig` from CLI, set:

```cpp
config.log_csv = command.train.log_csv;
```

In `modules/train/trainer.cpp`, create a `MetricLoggerGroup` at training start. Always include `ConsoleProgressLogger`. If `config.log_csv` is not empty, add `CsvMetricLogger` with columns:

```cpp
{
    "loss_total", "feature_loss", "repeatability_loss", "descriptor_loss",
    "matcher_loss", "graph_matching_loss", "dense_loss", "offset_loss",
    "confidence_loss", "descriptor_accuracy", "descriptor_diversity",
    "offset_error_px", "gpu_utilization_percent", "gpu_power_watts"
}
```

Replace the current `std::cout << "train progress: ..."` block with construction of `TrainingMetric` and `logger.logIteration(metric)`. Preserve epoch summary through `logger.logEpochSummary(metric)`.

Sample metric construction:

```cpp
TrainingMetric metric;
metric.epoch = epoch;
metric.total_epochs = config.epochs;
metric.iteration = batch_index + 1;
metric.total_iterations = batches_per_epoch;
metric.images_seen = processed_images;
metric.total_images = total_samples;
metric.learning_rate = config.learning_rate;
metric.elapsed_seconds = epoch_timer.elapsedSeconds();
metric.values["loss_total"] = total_loss_value;
metric.values["feature_loss"] = feature_loss_value;
metric.values["repeatability_loss"] = repeatability_loss_value;
metric.values["descriptor_loss"] = descriptor_loss_value;
metric.values["matcher_loss"] = matcher_loss_value;
metric.values["graph_matching_loss"] = graph_matching_loss_value;
metric.values["dense_loss"] = dense_loss_value;
metric.values["offset_loss"] = offset_loss_value;
metric.values["confidence_loss"] = confidence_loss_value;
metric.values["descriptor_accuracy"] = descriptor_accuracy_value;
metric.values["descriptor_diversity"] = descriptor_diversity_value;
metric.values["offset_error_px"] = offset_error_value;
```

Sample GPU metric merge:

```cpp
auto gpu_metrics = gpu_provider->sample();
if (gpu_metrics.utilization_percent) {
    metric.values["gpu_utilization_percent"] = *gpu_metrics.utilization_percent;
}
if (gpu_metrics.power_watts) {
    metric.values["gpu_power_watts"] = *gpu_metrics.power_watts;
}
```

- [ ] **Step 4: Run green check**

Run:

```bash
cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass; CLI parse test and trainer CSV test pass.

- [ ] **Step 5: Update usage docs**

Modify `docs/usage.md` train command example to include optional logging:

```bash
  --log-csv build/train_log.csv
```

Add one sentence:

```text
`--log-csv` 会按 iteration 写入 loss、matcher、dense、offset、学习率和 GPU 指标字段，便于后续绘图分析。
```

- [ ] **Step 6: Commit if authorized**

```bash
git add modules/train/trainer.h modules/train/trainer.cpp modules/train/trainer_test.cpp modules/cli/commands.h modules/cli/commands.cpp modules/cli/commands_test.cpp docs/usage.md
git commit -m "Add structured training CSV logging"
```

Skip the commit if commits are not authorized.

---

## Task 10: Add Synthetic Pair TensorDataset Adapters

**Files:**
- Create: `modules/dataloader/synthetic_pair_dataset.h`
- Create: `modules/dataloader/synthetic_pair_dataset.cpp`
- Modify: `modules/dataloader/dataloader_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing dataset adapter tests**

Append to `modules/dataloader/dataloader_test.cpp`:

```cpp
#include "dataloader/synthetic_pair_dataset.h"

static void synthetic_pair_tensor_dataset_returns_training_keys() {
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
```

Register it:

```cpp
register_test("synthetic pair tensor dataset returns training keys", synthetic_pair_tensor_dataset_returns_training_keys);
```

Add source to CMake.

- [ ] **Step 2: Run red check**

Run:

```bash
cmake --build . -j$(nproc)
```

Expected: compile fails because `dataloader/synthetic_pair_dataset.h` does not exist.

- [ ] **Step 3: Implement dataset adapter**

Create `modules/dataloader/synthetic_pair_dataset.h`:

```cpp
#pragma once

#include "augment/image_pair_augmentor.h"
#include "dataloader/dataset.h"

#include <vector>

namespace pfm {

class SyntheticPairTensorDataset : public TensorDataset {
public:
    /// Creates an online synthetic pair dataset.
    /// \param images Source images as CHW float tensors.
    /// \param pairs_per_image Synthetic pairs generated from each image.
    /// \param config Base augmentation config.
    /// \throws std::invalid_argument if images are empty or pairs_per_image is zero.
    SyntheticPairTensorDataset(
        std::vector<torch::Tensor> images,
        size_t pairs_per_image,
        ImagePairAugmentationConfig config);

    /// Returns total synthetic pair count.
    /// \return images.size() * pairs_per_image.
    size_t size() const override;

    /// Generates one synthetic pair sample.
    /// \param index Dataset index.
    /// \return Tensor batch with view_a, view_b, warp_a_to_b, valid_mask.
    /// \throws std::out_of_range if index is invalid.
    TensorBatch get(size_t index) override;

private:
    std::vector<torch::Tensor> _images;
    size_t _pairs_per_image;
    ImagePairAugmentationConfig _config;
};

}  // namespace pfm
```

Create `modules/dataloader/synthetic_pair_dataset.cpp`:

```cpp
#include "dataloader/synthetic_pair_dataset.h"

#include <stdexcept>

namespace pfm {

SyntheticPairTensorDataset::SyntheticPairTensorDataset(
    std::vector<torch::Tensor> images,
    size_t pairs_per_image,
    ImagePairAugmentationConfig config
)
    : _images(std::move(images)), _pairs_per_image(pairs_per_image), _config(config) {
    if (_images.empty()) {
        throw std::invalid_argument("synthetic pair tensor dataset requires at least one image");
    }
    if (_pairs_per_image == 0) {
        throw std::invalid_argument("pairs per image must be positive");
    }
}

size_t SyntheticPairTensorDataset::size() const {
    return _images.size() * _pairs_per_image;
}

TensorBatch SyntheticPairTensorDataset::get(size_t index) {
    if (index >= size()) {
        throw std::out_of_range("synthetic pair tensor dataset index out of range");
    }
    const auto source_index = index / _pairs_per_image;
    const auto variant_index = index % _pairs_per_image;
    auto config = _config;
    config.source_index = static_cast<int>(source_index);
    config.variant_index = static_cast<int>(variant_index);
    auto sample = ImagePairAugmentor(config).augment(_images[source_index]);

    TensorBatch batch;
    batch["view_a"] = sample.view_a;
    batch["view_b"] = sample.view_b;
    batch["warp_a_to_b"] = sample.warp_a_to_b;
    batch["valid_mask"] = sample.valid_mask;
    return batch;
}

}  // namespace pfm
```

- [ ] **Step 4: Run green check**

Run:

```bash
cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass and dataset adapter test passes.

- [ ] **Step 5: Commit if authorized**

```bash
git add CMakeLists.txt modules/dataloader/synthetic_pair_dataset.h modules/dataloader/synthetic_pair_dataset.cpp modules/dataloader/dataloader_test.cpp
git commit -m "Add synthetic pair tensor dataset adapter"
```

Skip the commit if commits are not authorized.

---

## Task 11: Integrate AsyncDataLoader into Trainer Loop

**Files:**
- Modify: `modules/train/trainer.cpp`
- Modify: `modules/train/trainer_test.cpp`
- Modify: `modules/dataloader/synthetic_pair_dataset.*`
- Modify: `docs/training.md`

- [ ] **Step 1: Write failing trainer DataLoader smoke test**

Add to `modules/train/trainer_test.cpp`:

```cpp
static void trainer_uses_async_dataloader_when_workers_requested() {
    TempTrainingDirectory temp_dir("pfm_trainer_async_dataloader");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.dataloader_workers = 2;
    config.prefetch_batches = 2;
    config.pin_memory = false;

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
}
```

Register it:

```cpp
register_test("trainer uses async dataloader when workers requested", trainer_uses_async_dataloader_when_workers_requested);
```

- [ ] **Step 2: Run red check**

Run:

```bash
cmake --build . -j$(nproc)
```

Expected: compile fails because `TrainConfig` does not have `dataloader_workers`, `prefetch_batches`, or `pin_memory`.

- [ ] **Step 3: Add high-level DataLoader config fields**

Modify `TrainConfig`:

```cpp
int dataloader_workers = 0;
int prefetch_batches = 2;
bool pin_memory = false;
```

Validate:

```cpp
if (config.dataloader_workers < 0) {
    throw std::invalid_argument("dataloader_workers must be non-negative");
}
if (config.prefetch_batches <= 0) {
    throw std::invalid_argument("prefetch_batches must be positive");
}
```

Do not add CLI flags in this task unless the user explicitly asks; tests can configure these fields directly.

- [ ] **Step 4: Use AsyncDataLoader for online synthetic pairs**

In `train_model()`, for the non-cache path and when `config.dataloader_workers > 0`, build:

```cpp
auto tensor_dataset = std::make_shared<SyntheticPairTensorDataset>(source_images, config.pairs_per_image, augment_config);
auto sampler = std::make_unique<SequentialSampler>(tensor_dataset->size());
TensorBatchCollator collator({
    {"view_a", TensorLayout::Chw},
    {"view_b", TensorLayout::Chw},
    {"warp_a_to_b", TensorLayout::Hwc},
    {"valid_mask", TensorLayout::Hw},
});
DataLoaderOptions options;
options.batch_size = static_cast<size_t>(config.batch_size);
options.worker_count = static_cast<size_t>(config.dataloader_workers);
options.prefetch_batches = static_cast<size_t>(config.prefetch_batches);
options.pin_memory = config.pin_memory;
AsyncDataLoader loader(tensor_dataset, std::move(sampler), collator, options);
```

Convert `TensorBatch` to the existing `SyntheticPairBatch` structure or add a helper:

```cpp
SyntheticPairBatch batch_from_tensor_batch(const TensorBatch& batch) {
    return SyntheticPairBatch{
        batch.at("view_a"),
        batch.at("view_b"),
        batch.at("warp_a_to_b"),
        batch.at("valid_mask")};
}
```

Keep existing manual path when `dataloader_workers == 0` so the first integration is low-risk.

- [ ] **Step 5: Run green check**

Run:

```bash
cmake --build . -j$(nproc) && ./pfm_tests
```

Expected: all tests pass, including async trainer smoke.

- [ ] **Step 6: Commit if authorized**

```bash
git add modules/train/trainer.h modules/train/trainer.cpp modules/train/trainer_test.cpp modules/dataloader/synthetic_pair_dataset.h modules/dataloader/synthetic_pair_dataset.cpp docs/training.md
git commit -m "Integrate async dataloader into trainer"
```

Skip the commit if commits are not authorized.

---

## Task 12: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/training.md`
- Modify: `docs/usage.md`

- [ ] **Step 1: Update README module overview**

Add bullets under implemented capabilities:

```markdown
- 可复用训练基础设施：`runtime` 提供线程池/阻塞队列，`dataloader` 提供异步 TensorBatch DataLoader，`augment` 提供图像对增强，`logging` 提供进度条、CSV 和可选 GPU 指标。
- 训练日志：支持类似 PyTorch/tqdm 的控制台进度条，并可写 CSV 供后续绘图分析。
```

Add module tree entries:

```text
  runtime/    通用线程池和阻塞队列
  dataloader/ TensorBatch Dataset、Sampler、异步预取和 pinned memory
  augment/    图像对几何/光照增强与 dense correspondence 生成
  logging/    训练进度条、CSV 日志和 GPU 指标采集
```

- [ ] **Step 2: Update training docs**

Add a section explaining:

```markdown
## 训练数据加载与日志

训练基础设施拆分为通用模块：`runtime`、`dataloader`、`augment` 和 `logging`。DataLoader 可以在后台线程读取/增强图像对并预取 batch；CUDA 训练时可以启用 pinned memory，让主线程更快把 batch 送入 GPU。日志系统会在控制台显示 epoch/iter/loss/lr/GPU 利用率/功耗，并可写 CSV。

如果训练后期 `feature_loss` 接近 0、`descriptor_accuracy` 接近 1，但 `graph_matching_loss` 仍在 6-9，说明特征描述子监督已经较容易，瓶颈主要在图匹配分类目标、正负样本/dustbin 比例或 hard pair 数据策略上。CSV 日志可以帮助按 profile、epoch 和 split 分析该问题。
```

- [ ] **Step 3: Update usage docs**

Document `--log-csv` if Task 9 added it:

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --log-csv build/train_log.csv
```

- [ ] **Step 4: Run final project verification**

Run the project-required verification from the build directory:

```bash
cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) && ./pfm_tests && ctest --output-on-failure
```

Expected:

- configure completes;
- build completes;
- `./pfm_tests` reports all tests passed;
- CTest reports `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 5: Review git diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only planned module, trainer, CLI, CMake, test, README, and docs files changed.

- [ ] **Step 6: Commit if authorized**

```bash
git add README.md docs/training.md docs/usage.md
git commit -m "Document reusable training infrastructure"
```

Skip the commit if commits are not authorized.

---

## Self-Review Checklist

- Spec coverage:
  - `runtime` queue/thread pool: Tasks 1-2.
  - `dataloader` TensorBatch/samplers/splits/collator/async/pinned memory: Tasks 3-5 and 10-11.
  - `augment` reusable image-pair augmentation and compatibility wrapper: Task 8.
  - `logging` console/CSV/null GPU/NVML provider: Tasks 6-7 and 9.
  - CMake module targets and optional NVML: Tasks 1, 3, 6, 7, 8.
  - Trainer phased integration: Tasks 9 and 11.
  - Docs: Task 12.
- Placeholder scan: this plan contains no unresolved placeholder instructions; where code must preserve existing implementation, it explicitly says to move the current implementation and keep existing tests passing.
- Type consistency: `TensorBatch`, `TensorLayout`, `TensorDataset`, `Sampler`, `TensorBatchCollator`, `AsyncDataLoader`, `TrainingMetric`, and `GpuMetricProvider` names are consistent across tasks.
