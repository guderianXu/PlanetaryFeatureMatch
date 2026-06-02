#include <atomic>
#include <chrono>
#include <filesystem>
#include <random>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <unistd.h>

#include "tests/test_harness.h"
#include "train/training_visualization.h"

namespace
{

class TempTrainingVisualizationDirectory
{
  public:
    explicit TempTrainingVisualizationDirectory(const std::string& stem)
    {
        const auto suffix =
            std::to_string(static_cast<long long>(getpid())) + "_" + std::to_string(std::random_device{}());
        _path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
        std::filesystem::create_directory(_path);
    }

    ~TempTrainingVisualizationDirectory()
    {
        std::error_code ignored;
        for (const auto& entry : std::filesystem::directory_iterator(_path))
        {
            std::filesystem::remove(entry.path(), ignored);
        }
        std::filesystem::remove(_path, ignored);
    }

    const std::filesystem::path& path() const
    {
        return _path;
    }

  private:
    std::filesystem::path _path;
};

bool has_text_pixels(const cv::Mat& image)
{
    for (int y = 0; y < std::min(14, image.rows); ++y)
    {
        for (int x = 0; x < std::min(96, image.cols); ++x)
        {
            const auto pixel = image.at<cv::Vec3b>(y, x);
            if (pixel[0] > 180 && pixel[1] > 180 && pixel[2] > 180)
            {
                return true;
            }
        }
    }
    return false;
}

} // namespace

static void async_visualization_writer_flushes_all_queued_pngs()
{
    TempTrainingVisualizationDirectory temp_dir("pfm_async_train_vis");
    pfm::AsyncVisualizationWriter writer(256);

    for (int index = 0; index < 3; ++index)
    {
        const auto path = temp_dir.path() / ("image_" + std::to_string(index) + ".png");
        writer.enqueueImage(path, torch::full({1, 16, 16}, static_cast<float>(index) / 3.0F), "features=1");
    }
    writer.join();

    PFM_REQUIRE(std::filesystem::exists(temp_dir.path() / "image_0.png"));
    PFM_REQUIRE(std::filesystem::exists(temp_dir.path() / "image_1.png"));
    PFM_REQUIRE(std::filesystem::exists(temp_dir.path() / "image_2.png"));
}

static void async_visualization_writer_draws_count_overlay()
{
    TempTrainingVisualizationDirectory temp_dir("pfm_async_train_vis_overlay");
    const auto path = temp_dir.path() / "overlay.png";
    pfm::AsyncVisualizationWriter writer(256);

    writer.enqueueImage(path, torch::zeros({1, 32, 128}, torch::kFloat32), "features=12");
    writer.join();
    const auto image = cv::imread(path.string(), cv::IMREAD_COLOR);

    PFM_REQUIRE(!image.empty());
    PFM_REQUIRE(has_text_pixels(image));
}

static void async_visualization_writer_preserves_rgb_tensor_colors()
{
    TempTrainingVisualizationDirectory temp_dir("pfm_async_train_vis_color");
    const auto path = temp_dir.path() / "color.png";
    pfm::AsyncVisualizationWriter writer(256);
    auto image = torch::zeros({3, 8, 8}, torch::kFloat32);
    image.index_put_({0, 3, 4}, 1.0F);

    writer.enqueueImage(path, image, "");
    writer.join();
    const auto written = cv::imread(path.string(), cv::IMREAD_COLOR);

    PFM_REQUIRE(!written.empty());
    const auto pixel = written.at<cv::Vec3b>(3, 4);
    PFM_REQUIRE(pixel[2] > 200);
    PFM_REQUIRE(pixel[1] < 20);
    PFM_REQUIRE(pixel[0] < 20);
}

static void async_visualization_writer_runs_background_jobs()
{
    pfm::AsyncVisualizationWriter writer(256, 2);
    std::atomic<int> rendered{0};

    writer.enqueueJob(
        [&rendered]()
        {
            rendered.fetch_add(1);
        });
    writer.enqueueJob(
        [&rendered]()
        {
            rendered.fetch_add(1);
        });
    writer.join();

    PFM_REQUIRE(rendered.load() == 2);
}

static void async_visualization_writer_fast_jobs_do_not_wait_for_slow_job()
{
    pfm::AsyncVisualizationWriter writer(256, 2);
    std::atomic<bool> slow_done{false};
    std::atomic<bool> fast_done{false};

    writer.enqueueJob(
        [&slow_done]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            slow_done.store(true);
        });
    writer.enqueueJob(
        [&fast_done]()
        {
            fast_done.store(true);
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    PFM_REQUIRE(fast_done.load());
    PFM_REQUIRE(!slow_done.load());
    writer.join();
}

void register_training_visualization_tests()
{
    register_test("async_visualization_writer_flushes_all_queued_pngs",
                  async_visualization_writer_flushes_all_queued_pngs);
    register_test("async_visualization_writer_draws_count_overlay", async_visualization_writer_draws_count_overlay);
    register_test("async_visualization_writer_preserves_rgb_tensor_colors",
                  async_visualization_writer_preserves_rgb_tensor_colors);
    register_test("async_visualization_writer_runs_background_jobs", async_visualization_writer_runs_background_jobs);
    register_test("async_visualization_writer_fast_jobs_do_not_wait_for_slow_job",
                  async_visualization_writer_fast_jobs_do_not_wait_for_slow_job);
}
