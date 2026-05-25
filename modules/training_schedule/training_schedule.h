#pragma once

#include <cstdint>
#include <string>

namespace pfm::training_schedule {

enum class DatasetKind {
    RotationClean,
    MixedView,
    HardPair,
};

enum class CurriculumStage {
    RotationClean,
    MixedView,
    HardPairs,
};

struct SampleMetrics {
    double precision = 1.0;
    int match_count = 0;
    double hardness = 0.0;
    DatasetKind dataset_kind = DatasetKind::RotationClean;
};

struct TrainingStageConfig {
    CurriculumStage stage = CurriculumStage::RotationClean;
    double base_weight = 1.0;
    double hard_pair_weight_cap = 3.0;
    double hardness_weight = 1.0;
    double low_precision_weight = 1.0;
    double low_match_weight = 0.25;
    int target_match_count = 64;
};

struct SampleDecision {
    bool enabled = false;
    double weight = 0.0;
    std::string stage_name;
};

struct HardFineTuneConfig {
    double base_lr = 1.0e-4;
    double hard_lr_scale = 0.1;
    int warmup_steps = 0;
};

std::string datasetKindName(DatasetKind kind);
std::string stageName(CurriculumStage stage);
bool isDatasetEnabled(DatasetKind kind, CurriculumStage stage);
SampleDecision evaluateSample(const SampleMetrics& sample, const TrainingStageConfig& config);
double hardFineTuneTargetLearningRate(const HardFineTuneConfig& config);
double hardFineTuneLearningRate(const HardFineTuneConfig& config, int64_t step);

}  // namespace pfm::training_schedule
