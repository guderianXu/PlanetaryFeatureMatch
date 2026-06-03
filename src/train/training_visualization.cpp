#include "train/training_visualization.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pfm
{
namespace
{

cv::Mat tensor_to_bgr_u8(const torch::Tensor& image)
{
    // 训练内部使用 RGB/灰度 float 张量，OpenCV 写 PNG 需要转换成 BGR uint8。
    if (!image.defined())
    {
        throw std::invalid_argument("visualization image tensor must be defined");
    }
    auto tensor = image.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    if (tensor.dim() == 3 && tensor.size(0) == 1)
    {
        tensor = tensor.squeeze(0).contiguous();
    }
    tensor = tensor.clamp(0.0, 1.0).mul(255.0).to(torch::kU8).contiguous();
    if (tensor.dim() == 2)
    {
        cv::Mat gray(static_cast<int>(tensor.size(0)), static_cast<int>(tensor.size(1)), CV_8UC1,
                     tensor.data_ptr<uint8_t>());
        cv::Mat color;
        cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
        return color;
    }
    if (tensor.dim() != 3 || tensor.size(0) != 3)
    {
        throw std::invalid_argument("visualization image tensor must have shape {3,H,W}, {1,H,W}, or {H,W}");
    }
    tensor = tensor.permute({1, 2, 0}).contiguous();
    cv::Mat rgb(static_cast<int>(tensor.size(0)), static_cast<int>(tensor.size(1)), CV_8UC3,
                tensor.data_ptr<uint8_t>());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr.clone();
}

void write_image_job(const std::filesystem::path& output_path, const torch::Tensor& image, const std::string& overlay)
{
    auto color = tensor_to_bgr_u8(image);
    if (!overlay.empty())
    {
        cv::putText(color, overlay, cv::Point(4, 12), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
    }
    std::filesystem::create_directories(output_path.parent_path());
    if (!cv::imwrite(output_path.string(), color))
    {
        throw std::invalid_argument("failed to write training visualization png: " + output_path.string());
    }
}

} // namespace

void writeVisualizationImage(const std::filesystem::path& output_path, const torch::Tensor& image,
                             const std::string& overlay)
{
    write_image_job(output_path, image, overlay);
}

AsyncVisualizationWriter::AsyncVisualizationWriter(std::size_t capacity, std::size_t worker_count) : _capacity(capacity)
{
    if (_capacity == 0)
    {
        throw std::invalid_argument("visualization writer capacity must be positive");
    }
    if (worker_count == 0)
    {
        throw std::invalid_argument("visualization writer worker count must be positive");
    }
    _threads.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index)
    {
        _threads.emplace_back(
            [this]()
            {
                run();
            });
    }
}

AsyncVisualizationWriter::~AsyncVisualizationWriter() noexcept(false)
{
    if (!_joined)
    {
        join();
    }
}

void AsyncVisualizationWriter::enqueueImage(const std::filesystem::path& output_path, const torch::Tensor& image,
                                            const std::string& overlay)
{
    if (!image.defined())
    {
        throw std::invalid_argument("visualization image tensor must be defined");
    }
    // 入队前复制到 CPU，避免后台线程访问训练步骤中可能被释放或复用的 GPU/临时张量。
    auto cpu_image = image.detach().to(torch::kCPU).contiguous();
    enqueueJob(
        [output_path, cpu_image, overlay]()
        {
            write_image_job(output_path, cpu_image, overlay);
        });
}

void AsyncVisualizationWriter::enqueueJob(std::function<void()> job)
{
    if (!job)
    {
        throw std::invalid_argument("visualization job must be callable");
    }
    std::unique_lock<std::mutex> lock(_mutex);
    if (_joined || _stopping)
    {
        throw std::invalid_argument("visualization writer already joined");
    }
    _not_full.wait(lock,
                   [this]()
                   {
                       return _jobs.size() < _capacity || _error != nullptr;
                   });
    if (_error)
    {
        std::rethrow_exception(_error);
    }
    _jobs.push(std::move(job));
    _not_empty.notify_one();
}

void AsyncVisualizationWriter::join()
{
    // join 可以重复调用；第一次负责停止 worker，后续只重新抛出已记录的错误。
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_joined)
        {
            if (_error)
            {
                std::rethrow_exception(_error);
            }
            return;
        }
        _stopping = true;
    }
    _not_empty.notify_all();
    for (auto& thread : _threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    _joined = true;
    if (_error)
    {
        std::rethrow_exception(_error);
    }
}

void AsyncVisualizationWriter::run()
{
    try
    {
        while (true)
        {
            VisualizationJob job;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                // worker 在队列为空时休眠；析构或显式 join 时即使无任务也会被唤醒退出。
                _not_empty.wait(lock,
                                [this]()
                                {
                                    return _stopping || !_jobs.empty();
                                });
                if (_jobs.empty())
                {
                    return;
                }
                job = std::move(_jobs.front());
                _jobs.pop();
                _not_full.notify_one();
            }
            job();
        }
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _error = std::current_exception();
        _not_full.notify_all();
    }
}

} // namespace pfm
