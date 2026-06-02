#pragma once

#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "runtime/blocking_queue.h"

namespace pfm
{

class ThreadPool
{
  public:
    /// Creates a fixed-size worker pool.
    ///
    /// @param worker_count Number of worker threads to start.
    /// @param queue_capacity Maximum number of queued jobs.
    /// @throws std::invalid_argument when worker_count is zero, or when queue_capacity is invalid.
    ThreadPool(std::size_t worker_count, std::size_t queue_capacity);

    /// Closes the job queue and joins worker threads without throwing.
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Enqueues one job for worker execution.
    ///
    /// @param job Function to run on a worker thread.
    /// @throws std::runtime_error when the pool queue is closed.
    void enqueue(std::function<void()> job);

    /// Stops accepting new jobs after queued jobs drain.
    void close();

    /// Joins all worker threads and rethrows the first worker exception, if any.
    void join();

  private:
    void workerLoop();
    void captureException();

    BlockingQueue<std::function<void()>> _jobs;
    std::vector<std::thread> _workers;
    std::mutex _join_mutex;
    std::mutex _exception_mutex;
    std::exception_ptr _first_exception;
    bool _joined = false;
};

} // namespace pfm
