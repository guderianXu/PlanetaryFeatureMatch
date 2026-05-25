#include "tests/test_harness.h"

#include "training_schedule/training_schedule.h"

namespace {

using pfm::training_schedule::CurriculumStage;
using pfm::training_schedule::DatasetKind;
using pfm::training_schedule::HardFineTuneConfig;
using pfm::training_schedule::SampleMetrics;
using pfm::training_schedule::TrainingStageConfig;
using pfm::training_schedule::evaluateSample;
using pfm::training_schedule::hardFineTuneLearningRate;

void highQualityRotationSampleKeepsNormalWeight() {
    TrainingStageConfig config;
    config.stage = CurriculumStage::RotationClean;
    config.base_weight = 1.0;

    SampleMetrics sample;
    sample.precision = 0.995;
    sample.match_count = 180;
    sample.hardness = 0.02;
    sample.dataset_kind = DatasetKind::RotationClean;

    const auto decision = evaluateSample(sample, config);

    PFM_REQUIRE(decision.enabled);
    PFM_REQUIRE(decision.stage_name == "rotation_clean");
    PFM_REQUIRE_CLOSE(decision.weight, 1.0, 1.0e-12);
}

void lowPrecisionHardPairIsWeightedButCapped() {
    TrainingStageConfig config;
    config.stage = CurriculumStage::HardPairs;
    config.base_weight = 1.0;
    config.hard_pair_weight_cap = 2.5;
    config.hardness_weight = 3.0;
    config.low_precision_weight = 4.0;

    SampleMetrics sample;
    sample.precision = 0.10;
    sample.match_count = 18;
    sample.hardness = 0.95;
    sample.dataset_kind = DatasetKind::HardPair;

    const auto decision = evaluateSample(sample, config);

    PFM_REQUIRE(decision.enabled);
    PFM_REQUIRE(decision.weight > 1.0);
    PFM_REQUIRE_CLOSE(decision.weight, 2.5, 1.0e-12);
}

void curriculumFiltersExtremeSamplesEarlyAndReleasesLater() {
    TrainingStageConfig config;
    config.stage = CurriculumStage::RotationClean;

    SampleMetrics hard_sample;
    hard_sample.precision = 0.20;
    hard_sample.match_count = 12;
    hard_sample.hardness = 0.90;
    hard_sample.dataset_kind = DatasetKind::HardPair;

    PFM_REQUIRE(!evaluateSample(hard_sample, config).enabled);

    config.stage = CurriculumStage::MixedView;
    PFM_REQUIRE(!evaluateSample(hard_sample, config).enabled);

    config.stage = CurriculumStage::HardPairs;
    PFM_REQUIRE(evaluateSample(hard_sample, config).enabled);
}

void hardFineTuneLearningRateUsesScaleAndWarmupIsMonotonic() {
    HardFineTuneConfig config;
    config.base_lr = 1.0e-4;
    config.hard_lr_scale = 0.10;
    config.warmup_steps = 10;

    const auto lr0 = hardFineTuneLearningRate(config, 0);
    const auto lr5 = hardFineTuneLearningRate(config, 5);
    const auto lr10 = hardFineTuneLearningRate(config, 10);
    const auto lr20 = hardFineTuneLearningRate(config, 20);

    PFM_REQUIRE_CLOSE(lr0, 0.0, 1.0e-12);
    PFM_REQUIRE(lr0 < lr5);
    PFM_REQUIRE(lr5 < lr10);
    PFM_REQUIRE_CLOSE(lr10, 1.0e-5, 1.0e-12);
    PFM_REQUIRE_CLOSE(lr20, 1.0e-5, 1.0e-12);
}

}  // namespace

void register_training_schedule_tests() {
    register_test("training schedule high quality rotation sample keeps normal weight",
                  highQualityRotationSampleKeepsNormalWeight);
    register_test("training schedule low precision hard pair is weighted but capped",
                  lowPrecisionHardPairIsWeightedButCapped);
    register_test("training schedule curriculum filters extreme samples early and releases later",
                  curriculumFiltersExtremeSamplesEarlyAndReleasesLater);
    register_test("training schedule hard fine tune learning rate uses scale and warmup is monotonic",
                  hardFineTuneLearningRateUsesScaleAndWarmupIsMonotonic);
}
