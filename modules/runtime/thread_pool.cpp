#include "runtime/thread_pool.h"

#include <stdexcept>
#include <utility>

namespace pfm {

ThreadPool::ThreadPool(std::size_t worker_count, std::size_t queue_capacity) : _jobs(queue_capacity) {
    if (worker_count == 0) {
        throw std::invalid_argument("thread pool worker count must be positive");
    }

    _workers.reserve(worker_count);
    try {
        for (std::size_t index = 0; index < worker_count; ++index) {
            _workers.emplace_back([this]() {
                workerLoop();
            });
        }
    } catch (...) {
        close();
        for (auto& worker : _workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        throw;
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
    {
        std::lock_guard<std::mutex> lock(_join_mutex);
        if (!_joined) {
            for (auto& worker : _workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            _joined = true;
        }
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
