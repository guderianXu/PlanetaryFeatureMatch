#pragma once

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "dataloader/collator.h"
#include "dataloader/dataset.h"
#include "dataloader/sampler.h"
#include "dataloader/tensor_batch.h"
#include "runtime/blocking_queue.h"

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
    /// Creates a data loader for one dataset and sampler.
    /// \param dataset Dataset used to load samples.
    /// \param sampler Sampler that provides one epoch of indices.
    /// \param collator Collator used to pad and stack samples.
    /// \param options Batch size, worker, prefetch, drop-last, and pinned-memory options.
    /// \throws std::invalid_argument if dataset or sampler is null, batch_size is zero, or async prefetch is zero.
    AsyncDataLoader(
        std::shared_ptr<TensorDataset> dataset,
        std::unique_ptr<Sampler> sampler,
        TensorBatchCollator collator,
        DataLoaderOptions options);

    /// Stops workers and releases queued batches.
    ~AsyncDataLoader();

    AsyncDataLoader(const AsyncDataLoader&) = delete;
    AsyncDataLoader& operator=(const AsyncDataLoader&) = delete;

    /// Starts a fresh epoch using the sampler order.
    void reset();

    /// Returns the next batch, or std::nullopt when the epoch is exhausted.
    /// \return Optional tensor batch.
    /// \throws std::exception rethrows dataset, collation, or pinned-memory failures.
    std::optional<TensorBatch> next();

private:
    struct QueueItem;

    TensorBatch makeBatch(const std::vector<size_t>& batch_indices);
    std::optional<std::vector<size_t>> nextBatchIndices();
    void startAsyncEpoch();
    void stopAsyncEpoch();
    void workerLoop();
    void captureException();
    void throwIfWorkerFailed();

    std::shared_ptr<TensorDataset> _dataset;
    std::unique_ptr<Sampler> _sampler;
    TensorBatchCollator _collator;
    DataLoaderOptions _options;
    std::vector<size_t> _indices;
    size_t _cursor = 0;
    std::mutex _cursor_mutex;
    std::mutex _exception_mutex;
    std::exception_ptr _first_exception;
    std::unique_ptr<BlockingQueue<QueueItem>> _queue;
    std::vector<std::thread> _workers;
    size_t _finished_workers = 0;
    bool _exhausted = false;
};

}  // namespace pfm
