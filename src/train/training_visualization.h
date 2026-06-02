#pragma once

#include <condition_variable>
#include <exception>
#include <filesystem>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>

namespace pfm
{

/// Writes a single training visualization PNG immediately.
/// @param output_path PNG path to write.
/// @param image Tensor with shape {3,H,W}, {1,H,W}, or {H,W}, normalized to [0,1].
/// @param overlay Text drawn in the upper-left corner.
/// @throws std::invalid_argument if the tensor shape is invalid or PNG write fails.
void writeVisualizationImage(const std::filesystem::path& output_path, const torch::Tensor& image,
                             const std::string& overlay);

/// Asynchronously writes training diagnostic visualization PNG files.
class AsyncVisualizationWriter
{
  public:
    /// Creates a writer with background worker threads and a bounded queue.
    /// @param capacity Maximum queued image jobs before enqueue waits.
    /// @param worker_count Number of background writer threads.
    /// @throws std::invalid_argument if capacity or worker_count is zero.
    explicit AsyncVisualizationWriter(std::size_t capacity, std::size_t worker_count = 1);

    /// Flushes queued work, joins the writer thread, and rethrows writer errors.
    ~AsyncVisualizationWriter() noexcept(false);

    AsyncVisualizationWriter(const AsyncVisualizationWriter&) = delete;
    AsyncVisualizationWriter& operator=(const AsyncVisualizationWriter&) = delete;
    AsyncVisualizationWriter(AsyncVisualizationWriter&&) = delete;
    AsyncVisualizationWriter& operator=(AsyncVisualizationWriter&&) = delete;

    /// Queues a single tensor PNG with a text overlay.
    /// @param output_path PNG path to write.
    /// @param image Tensor with shape {3,H,W}, {1,H,W}, or {H,W}, normalized to [0,1].
    /// @param overlay Text drawn in the upper-left corner.
    /// @throws std::invalid_argument if the tensor shape is invalid or writer has already joined.
    void enqueueImage(const std::filesystem::path& output_path, const torch::Tensor& image, const std::string& overlay);

    /// Queues a background visualization job to run on a worker thread.
    /// @param job Function that performs rendering or writing work.
    /// @throws std::invalid_argument if the job is empty or writer has already joined.
    void enqueueJob(std::function<void()> job);

    /// Flushes queued jobs, joins the background thread, and rethrows writer errors.
    void join();

  private:
    using VisualizationJob = std::function<void()>;

    void run();

    std::size_t _capacity;
    std::mutex _mutex;
    std::condition_variable _not_empty;
    std::condition_variable _not_full;
    std::queue<VisualizationJob> _jobs;
    bool _stopping = false;
    bool _joined = false;
    std::exception_ptr _error;
    std::vector<std::thread> _threads;
};

} // namespace pfm
