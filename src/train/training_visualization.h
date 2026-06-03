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

/// 立即写出一张训练可视化 PNG。
/// @param output_path 目标 PNG 路径。
/// @param image 形状为 {3,H,W}、{1,H,W} 或 {H,W} 的张量，值域归一化到 [0,1]。
/// @param overlay 绘制在左上角的文本。
/// @throws std::invalid_argument 当张量形状非法或 PNG 写出失败时抛出。
void writeVisualizationImage(const std::filesystem::path& output_path, const torch::Tensor& image,
                             const std::string& overlay);

/// 异步写出训练诊断可视化 PNG 文件。
class AsyncVisualizationWriter
{
  public:
    /// 创建带后台 worker 和有界队列的写出器。
    /// @param capacity enqueue 阻塞前允许排队的最大任务数量。
    /// @param worker_count 后台写出线程数量。
    /// @throws std::invalid_argument 当 capacity 或 worker_count 为 0 时抛出。
    explicit AsyncVisualizationWriter(std::size_t capacity, std::size_t worker_count = 1);

    /// 刷新队列、等待写出线程结束，并重新抛出写出错误。
    ~AsyncVisualizationWriter() noexcept(false);

    AsyncVisualizationWriter(const AsyncVisualizationWriter&) = delete;
    AsyncVisualizationWriter& operator=(const AsyncVisualizationWriter&) = delete;
    AsyncVisualizationWriter(AsyncVisualizationWriter&&) = delete;
    AsyncVisualizationWriter& operator=(AsyncVisualizationWriter&&) = delete;

    /// 将一张带文字叠加的张量 PNG 加入队列。
    /// @param output_path 目标 PNG 路径。
    /// @param image 形状为 {3,H,W}、{1,H,W} 或 {H,W} 的张量，值域归一化到 [0,1]。
    /// @param overlay 绘制在左上角的文本。
    /// @throws std::invalid_argument 当张量形状非法或写出器已经 join 时抛出。
    void enqueueImage(const std::filesystem::path& output_path, const torch::Tensor& image, const std::string& overlay);

    /// 将后台可视化任务加入队列，由 worker 线程执行。
    /// @param job 执行渲染或写出工作的函数。
    /// @throws std::invalid_argument 当 job 为空或写出器已经 join 时抛出。
    void enqueueJob(std::function<void()> job);

    /// 刷新队列、等待后台线程结束，并重新抛出写出错误。
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
