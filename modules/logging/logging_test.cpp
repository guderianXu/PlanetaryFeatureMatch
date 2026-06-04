#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "logging/csv_metric_logger.h"
#include "logging/gpu_metric_provider.h"
#include "logging/metric_logger_group.h"
#include "logging/progress_logger.h"
#include "tests/test_harness.h"

namespace
{

pfm::TrainingMetric sampleMetric()
{
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

class CountingMetricLogger : public pfm::TrainingMetricLogger
{
  public:
    void logIteration(const pfm::TrainingMetric&) override
    {
        ++iteration_count;
    }

    void logEpochSummary(const pfm::TrainingMetric&) override
    {
        ++summary_count;
    }

    void flush() override
    {
        ++flush_count;
    }

    int iteration_count = 0;
    int summary_count = 0;
    int flush_count = 0;
};

static void consoleProgressLoggerRendersCoreFields()
{
    std::ostringstream stream;
    pfm::ConsoleProgressLogger logger(stream, 20);

    logger.logIteration(sampleMetric());
    logger.logEpochSummary(sampleMetric());

    const auto text = stream.str();
    PFM_REQUIRE(text.find("epoch 2/10") != std::string::npos);
    PFM_REQUIRE(text.find("3/5") != std::string::npos);
    PFM_REQUIRE(text.find("loss=4.5") != std::string::npos);
    PFM_REQUIRE(text.find("match=3.5") != std::string::npos);
    PFM_REQUIRE(text.find("off=12") != std::string::npos);
}

static void csvMetricLoggerWritesHeaderAndRows()
{
    const auto path = std::filesystem::temp_directory_path() / "pfm_metric_logger_test.csv";
    std::filesystem::remove(path);

    pfm::CsvMetricLogger logger(path.string(), {"loss_total", "matcher_loss", "offset_error_px"});
    logger.logIteration(sampleMetric());
    logger.flush();

    std::ifstream input(path);
    std::string header;
    std::string row;
    std::getline(input, header);
    std::getline(input, row);

    PFM_REQUIRE(header == "epoch,total_epochs,iteration,total_iterations,images_seen,total_images,learning_rate,"
                          "elapsed_seconds,loss_total,matcher_loss,offset_error_px");
    PFM_REQUIRE(row.find("2,10,3,5,12,20") == 0);
    PFM_REQUIRE(row.find(",4.5,3.5,12") != std::string::npos);
    std::filesystem::remove(path);
}

static void nullGpuMetricProviderReturnsEmptyValues()
{
    pfm::NullGpuMetricProvider provider;

    auto metrics = provider.sample();

    PFM_REQUIRE(!metrics.utilization_percent.has_value());
    PFM_REQUIRE(!metrics.power_watts.has_value());
    PFM_REQUIRE(!metrics.memory_used_mb.has_value());
    PFM_REQUIRE(!metrics.memory_total_mb.has_value());
    PFM_REQUIRE(!metrics.memory_free_mb.has_value());
}

static void defaultGpuMetricProviderIsConstructible()
{
    auto provider = pfm::makeDefaultGpuMetricProvider();

    PFM_REQUIRE(provider != nullptr);
    auto metrics = provider->sample();
    PFM_REQUIRE(!metrics.utilization_percent.has_value() || metrics.utilization_percent.value() >= 0.0);
    PFM_REQUIRE(!metrics.power_watts.has_value() || metrics.power_watts.value() >= 0.0);
    PFM_REQUIRE(!metrics.memory_used_mb.has_value() || metrics.memory_used_mb.value() >= 0.0);
    PFM_REQUIRE(!metrics.memory_total_mb.has_value() || metrics.memory_total_mb.value() >= 0.0);
    PFM_REQUIRE(!metrics.memory_free_mb.has_value() || metrics.memory_free_mb.value() >= 0.0);
}

static void metricLoggerGroupForwardsToOwnedLoggers()
{
    auto first = std::make_unique<CountingMetricLogger>();
    auto second = std::make_unique<CountingMetricLogger>();
    auto* first_ptr = first.get();
    auto* second_ptr = second.get();
    pfm::MetricLoggerGroup group;
    group.addLogger(std::move(first));
    group.addLogger(std::move(second));

    group.logIteration(sampleMetric());
    group.logEpochSummary(sampleMetric());
    group.flush();

    PFM_REQUIRE(first_ptr->iteration_count == 1);
    PFM_REQUIRE(first_ptr->summary_count == 1);
    PFM_REQUIRE(first_ptr->flush_count == 1);
    PFM_REQUIRE(second_ptr->iteration_count == 1);
    PFM_REQUIRE(second_ptr->summary_count == 1);
    PFM_REQUIRE(second_ptr->flush_count == 1);
}

} // namespace

void register_logging_tests()
{
    register_test("console progress logger renders core fields", consoleProgressLoggerRendersCoreFields);
    register_test("csv metric logger writes header and rows", csvMetricLoggerWritesHeaderAndRows);
    register_test("null gpu metric provider returns empty values", nullGpuMetricProviderReturnsEmptyValues);
    register_test("default gpu metric provider is constructible", defaultGpuMetricProviderIsConstructible);
    register_test("metric logger group forwards to owned loggers", metricLoggerGroupForwardsToOwnedLoggers);
}
