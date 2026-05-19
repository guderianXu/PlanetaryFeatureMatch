#include <algorithm>
#include <stdexcept>
#include <utility>

#include "dataloader/async_dataloader.h"

namespace pfm {

struct AsyncDataLoader::QueueItem {
    std::optional<TensorBatch> batch;
    bool worker_finished = false;
};

AsyncDataLoader::AsyncDataLoader(
    std::shared_ptr<TensorDataset> dataset,
    std::unique_ptr<Sampler> sampler,
    TensorBatchCollator collator,
    DataLoaderOptions options)
    : _dataset(std::move(dataset)),
      _sampler(std::move(sampler)),
      _collator(std::move(collator)),
      _options(options) {
    if (!_dataset) {
        throw std::invalid_argument("async data loader requires a dataset");
    }
    if (!_sampler) {
        throw std::invalid_argument("async data loader requires a sampler");
    }
    if (_options.batch_size == 0) {
        throw std::invalid_argument("async data loader batch size must be positive");
    }
    if (_options.worker_count > 0 && _options.prefetch_batches == 0) {
        throw std::invalid_argument("async data loader prefetch batches must be positive when workers are used");
    }
    reset();
}

AsyncDataLoader::~AsyncDataLoader() {
    stopAsyncEpoch();
}

void AsyncDataLoader::reset() {
    stopAsyncEpoch();
    _indices.clear();
    _cursor = 0;
    _finished_workers = 0;
    _exhausted = true;
    {
        std::lock_guard<std::mutex> lock(_exception_mutex);
        _first_exception = nullptr;
    }

    _indices = _sampler->indices();
    if (_options.worker_count > 0) {
        startAsyncEpoch();
    }
    _exhausted = false;
}

std::optional<TensorBatch> AsyncDataLoader::next() {
    if (_options.worker_count == 0) {
        if (_exhausted) {
            return std::nullopt;
        }
        auto batch_indices = nextBatchIndices();
        if (!batch_indices) {
            _exhausted = true;
            return std::nullopt;
        }
        return makeBatch(*batch_indices);
    }

    if (_exhausted) {
        throwIfWorkerFailed();
        return std::nullopt;
    }

    while (auto item = _queue->pop()) {
        if (item->worker_finished) {
            ++_finished_workers;
            if (_finished_workers == _options.worker_count) {
                _exhausted = true;
                throwIfWorkerFailed();
                return std::nullopt;
            }
            continue;
        }
        throwIfWorkerFailed();
        return std::move(item->batch);
    }

    _exhausted = true;
    throwIfWorkerFailed();
    return std::nullopt;
}

TensorBatch AsyncDataLoader::makeBatch(const std::vector<size_t>& batch_indices) {
    std::vector<TensorBatch> samples;
    samples.reserve(batch_indices.size());
    for (const auto index : batch_indices) {
        samples.push_back(_dataset->get(index));
    }
    auto batch = _collator.collate(samples);
    if (_options.pin_memory) {
        batch = pinTensorBatchMemory(batch);
    }
    return batch;
}

std::optional<std::vector<size_t>> AsyncDataLoader::nextBatchIndices() {
    std::lock_guard<std::mutex> lock(_cursor_mutex);
    if (_cursor >= _indices.size()) {
        return std::nullopt;
    }

    const auto remaining = _indices.size() - _cursor;
    if (remaining < _options.batch_size && _options.drop_last) {
        _cursor = _indices.size();
        return std::nullopt;
    }

    const auto current_size = std::min(_options.batch_size, remaining);
    std::vector<size_t> batch_indices(
        _indices.begin() + static_cast<std::ptrdiff_t>(_cursor),
        _indices.begin() + static_cast<std::ptrdiff_t>(_cursor + current_size));
    _cursor += current_size;
    return batch_indices;
}

void AsyncDataLoader::startAsyncEpoch() {
    _queue = std::make_unique<BlockingQueue<QueueItem>>(_options.prefetch_batches);
    _workers.reserve(_options.worker_count);
    try {
        for (size_t index = 0; index < _options.worker_count; ++index) {
            _workers.emplace_back([this]() {
                workerLoop();
            });
        }
    } catch (...) {
        stopAsyncEpoch();
        throw;
    }
}

void AsyncDataLoader::stopAsyncEpoch() {
    if (_queue) {
        _queue->close();
    }
    for (auto& worker : _workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    _workers.clear();
    _queue.reset();
}

void AsyncDataLoader::workerLoop() {
    try {
        while (auto batch_indices = nextBatchIndices()) {
            QueueItem item;
            item.batch = makeBatch(*batch_indices);
            _queue->push(std::move(item));
        }
    } catch (...) {
        captureException();
    }

    try {
        QueueItem finished;
        finished.worker_finished = true;
        _queue->push(std::move(finished));
    } catch (...) {
    }
}

void AsyncDataLoader::captureException() {
    std::lock_guard<std::mutex> lock(_exception_mutex);
    if (!_first_exception) {
        _first_exception = std::current_exception();
    }
}

void AsyncDataLoader::throwIfWorkerFailed() {
    std::exception_ptr exception;
    {
        std::lock_guard<std::mutex> lock(_exception_mutex);
        exception = _first_exception;
    }
    if (exception) {
        std::rethrow_exception(exception);
    }
}

}  // namespace pfm
