#include "training_schedule/training_schedule.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pfm::training_schedule {
namespace {

void requireFiniteNonNegative(const double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
    }
}

double clamp01(const double value) {
    return std::clamp(value, 0.0, 1.0);
}

void validateSample(const SampleMetrics& sample) {
    if (!std::isfinite(sample.precision) || sample.precision < 0.0 || sample.precision > 1.0) {
        throw std::invalid_argument("sample precision must be finite and in [0, 1]");
    }
    if (sample.match_count < 0) {
        throw std::invalid_argument("sample match_count must be non-negative");
    }
    requireFiniteNonNegative(sample.hardness, "sample hardness");
}

void validateTrainingConfig(const TrainingStageConfig& config) {
    requireFiniteNonNegative(config.base_weight, "base_weight");
    requireFiniteNonNegative(config.hard_pair_weight_cap, "hard_pair_weight_cap");
    requireFiniteNonNegative(config.hardness_weight, "hardness_weight");
    requireFiniteNonNegative(config.low_precision_weight, "low_precision_weight");
    requireFiniteNonNegative(config.low_match_weight, "low_match_weight");
    if (config.target_match_count < 0) {
        throw std::invalid_argument("target_match_count must be non-negative");
    }
}

void validateHardFineTuneConfig(const HardFineTuneConfig& config) {
    requireFiniteNonNegative(config.base_lr, "base_lr");
    requireFiniteNonNegative(config.hard_lr_scale, "hard_lr_scale");
    if (config.warmup_steps < 0) {
        throw std::invalid_argument("warmup_steps must be non-negative");
    }
}

double lowMatchFraction(const SampleMetrics& sample, const TrainingStageConfig& config) {
    if (config.target_match_count <= 0 || sample.match_count >= config.target_match_count) {
        return 0.0;
    }
    return static_cast<double>(config.target_match_count - sample.match_count) /
           static_cast<double>(config.target_match_count);
}

}  // namespace

std::string datasetKindName(const DatasetKind kind) {
    switch (kind) {
        case DatasetKind::RotationClean:
            return "rotation_clean";
        case DatasetKind::MixedView:
            return "mixed_view";
        case DatasetKind::HardPair:
            return "hard_pairs";
    }
    throw std::invalid_argument("unknown dataset kind");
}

std::string stageName(const CurriculumStage stage) {
    switch (stage) {
        case CurriculumStage::RotationClean:
            return "rotation_clean";
        case CurriculumStage::MixedView:
            return "mixed_view";
        case CurriculumStage::HardPairs:
            return "hard_pairs";
    }
    throw std::invalid_argument("unknown curriculum stage");
}

bool isDatasetEnabled(const DatasetKind kind, const CurriculumStage stage) {
    switch (stage) {
        case CurriculumStage::RotationClean:
            return kind == DatasetKind::RotationClean;
        case CurriculumStage::MixedView:
            return kind == DatasetKind::RotationClean || kind == DatasetKind::MixedView;
        case CurriculumStage::HardPairs:
            return true;
    }
    throw std::invalid_argument("unknown curriculum stage");
}

SampleDecision evaluateSample(const SampleMetrics& sample, const TrainingStageConfig& config) {
    validateSample(sample);
    validateTrainingConfig(config);

    SampleDecision decision;
    decision.stage_name = stageName(config.stage);
    decision.enabled = isDatasetEnabled(sample.dataset_kind, config.stage);
    if (!decision.enabled) {
        return decision;
    }

    double weight = config.base_weight;
    if (sample.dataset_kind == DatasetKind::HardPair) {
        const double hardness = clamp01(sample.hardness);
        const double precision_gap = 1.0 - sample.precision;
        weight += config.hardness_weight * hardness;
        weight += config.low_precision_weight * precision_gap;
        weight += config.low_match_weight * lowMatchFraction(sample, config);
        weight = std::min(weight, config.hard_pair_weight_cap);
    }

    decision.weight = weight;
    return decision;
}

double hardFineTuneTargetLearningRate(const HardFineTuneConfig& config) {
    validateHardFineTuneConfig(config);
    return config.base_lr * config.hard_lr_scale;
}

double hardFineTuneLearningRate(const HardFineTuneConfig& config, const int64_t step) {
    const double target_lr = hardFineTuneTargetLearningRate(config);
    if (step <= 0) {
        return config.warmup_steps == 0 ? target_lr : 0.0;
    }
    if (config.warmup_steps <= 0 || step >= config.warmup_steps) {
        return target_lr;
    }
    return target_lr * static_cast<double>(step) / static_cast<double>(config.warmup_steps);
}

}  // namespace pfm::training_schedule
