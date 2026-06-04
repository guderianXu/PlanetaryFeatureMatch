#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using TestFn = void (*)();

struct TestCase {
    std::string name;
    TestFn fn;
};

std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

void register_test(const std::string& name, TestFn fn) {
    registry().push_back({name, fn});
}

void register_device_tests();
void register_tensor_utils_tests();
void register_timer_tests();
void register_runtime_tests();
void register_dataloader_tests();
void register_logging_tests();
void register_optim_tests();
void register_cache_eval_tests();
void register_spatial_match_quality_tests();
void register_training_schedule_tests();
void register_adaptive_match_policy_tests();
void register_image_dataset_tests();
void register_image_io_tests();
void register_intensity_mask_tests();
void register_normalization_tests();
void register_pair_archive_dataset_tests();
void register_lazy_pose_pair_dataset_tests();
void register_warp_tests();
void register_synthetic_pair_tests();
void register_synthetic_pair_cache_tests();
void register_augment_tests();
void register_pfm_model_v21_tests();
void register_planetary_graph_matcher_tests();
void register_feature_codec_tests();
void register_feature_extractor_tests();
void register_match_codec_tests();
void register_match_metrics_tests();
void register_cache_match_eval_tests();
void register_checkpoint_gate_tests();
void register_matching_pipeline_tests();
void register_eval_pipeline_tests();
void register_pipeline_tests();
void register_visualization_tests();
void register_loss_tests();
void register_metric_tests();
void register_cli_tests();
void register_trainer_tests();
void register_training_visualization_tests();

int main() {
    setenv("OMP_NUM_THREADS", "1", 0);
    setenv("MKL_NUM_THREADS", "1", 0);

    register_device_tests();
    register_tensor_utils_tests();
    register_timer_tests();
    register_runtime_tests();
    register_dataloader_tests();
    register_logging_tests();
    register_optim_tests();
    register_cache_eval_tests();
    register_spatial_match_quality_tests();
    register_training_schedule_tests();
    register_adaptive_match_policy_tests();
    register_image_dataset_tests();
    register_image_io_tests();
    register_intensity_mask_tests();
    register_normalization_tests();
    register_pair_archive_dataset_tests();
    register_lazy_pose_pair_dataset_tests();
    register_warp_tests();
    register_synthetic_pair_tests();
    register_synthetic_pair_cache_tests();
    register_augment_tests();
    register_pfm_model_v21_tests();
    register_planetary_graph_matcher_tests();
    register_feature_codec_tests();
    register_feature_extractor_tests();
    register_match_codec_tests();
    register_match_metrics_tests();
    register_cache_match_eval_tests();
    register_checkpoint_gate_tests();
    register_matching_pipeline_tests();
    register_eval_pipeline_tests();
    register_pipeline_tests();
    register_visualization_tests();
    register_loss_tests();
    register_metric_tests();
    register_cli_tests();
    register_trainer_tests();
    register_training_visualization_tests();

    int failures = 0;
    for (const auto& test : registry()) {
        try {
            test.fn();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& e) {
            ++failures;
            std::cerr << "FAIL " << test.name << ": " << e.what() << '\n';
        }
    }
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << registry().size() << " test(s) passed\n";
    return 0;
}
