cmake_minimum_required(VERSION 3.18)
project(planetary_feature_matching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

option(PFM_ENABLE_NVML "Enable NVML GPU metrics" ON)
if(PFM_ENABLE_NVML)
    find_library(NVML_LIBRARY nvidia-ml)
    find_path(NVML_INCLUDE_DIR nvml.h)
endif()

enable_testing()

add_library(pfm_runtime STATIC
    modules/runtime/thread_pool.cpp
)
target_include_directories(pfm_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_runtime PUBLIC Threads::Threads)
target_compile_options(pfm_runtime PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_dataloader STATIC
    modules/dataloader/async_dataloader.cpp
    modules/dataloader/collator.cpp
    modules/dataloader/pinned_memory.cpp
    modules/dataloader/sampler.cpp
)
target_include_directories(pfm_dataloader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_dataloader PUBLIC pfm_runtime ${TORCH_LIBRARIES})
target_compile_options(pfm_dataloader PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_logging STATIC
    modules/logging/csv_metric_logger.cpp
    modules/logging/gpu_metric_provider.cpp
    modules/logging/metric_logger_group.cpp
    modules/logging/nvml_gpu_metric_provider.cpp
    modules/logging/progress_logger.cpp
)
target_include_directories(pfm_logging PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
if(NVML_LIBRARY AND NVML_INCLUDE_DIR)
    target_include_directories(pfm_logging PRIVATE ${NVML_INCLUDE_DIR})
    target_link_libraries(pfm_logging PUBLIC ${NVML_LIBRARY})
    target_compile_definitions(pfm_logging PRIVATE PFM_HAS_NVML=1)
endif()
target_compile_options(pfm_logging PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_optim STATIC
    modules/optim/amp.cpp
    modules/optim/descriptor_similarity.cpp
)
target_include_directories(pfm_optim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_optim PUBLIC ${TORCH_LIBRARIES})
target_compile_options(pfm_optim PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_losses STATIC
    modules/losses/basic_losses.cpp
)
target_include_directories(pfm_losses PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_losses PUBLIC ${TORCH_LIBRARIES})
target_compile_options(pfm_losses PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_cache_eval STATIC
    modules/cache_eval/cache_eval_manifest.cpp
    modules/cache_eval/cache_eval_metrics.cpp
    modules/cache_eval/cache_eval_quality.cpp
    modules/cache_eval/cache_eval_training.cpp
)
target_include_directories(pfm_cache_eval PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_compile_options(pfm_cache_eval PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_match_quality STATIC
    modules/match_quality/spatial_match_quality.cpp
)
target_include_directories(pfm_match_quality PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_compile_options(pfm_match_quality PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_training_schedule STATIC
    modules/training_schedule/training_schedule.cpp
)
target_include_directories(pfm_training_schedule PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_compile_options(pfm_training_schedule PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_adaptive_match_policy STATIC
    modules/adaptive_match_policy/adaptive_match_policy.cpp
)
target_include_directories(pfm_adaptive_match_policy PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_compile_options(pfm_adaptive_match_policy PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_core STATIC
    modules/core/device.cpp
    modules/core/tensor_utils.cpp
    modules/core/timer.cpp
)
target_include_directories(pfm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_core PUBLIC ${TORCH_LIBRARIES})
target_compile_options(pfm_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_image STATIC
    modules/image/image_io.cpp
    modules/image/normalization.cpp
    modules/image/intensity_mask.cpp
)
target_include_directories(pfm_image PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_image PUBLIC pfm_core ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_compile_options(pfm_image PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_geometry STATIC
    modules/geometry/warp.cpp
)
target_include_directories(pfm_geometry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_geometry PUBLIC ${TORCH_LIBRARIES})
target_compile_options(pfm_geometry PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_augment STATIC
    src/augment/image_pair_augmentor.cpp
    src/augment/transform_sampler.cpp
)
target_include_directories(pfm_augment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_augment PUBLIC pfm_core pfm_image pfm_geometry ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_compile_options(pfm_augment PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_eval STATIC
    modules/eval/metrics.cpp
)
target_include_directories(pfm_eval PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_eval PUBLIC ${TORCH_LIBRARIES})
target_compile_options(pfm_eval PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_checkpoint_gate_core STATIC
    modules/checkpoint_gate/checkpoint_gate.cpp
)
target_include_directories(pfm_checkpoint_gate_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_compile_options(pfm_checkpoint_gate_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_feature_io STATIC
    modules/feature_io/feature_codec.cpp
    modules/feature_io/match_codec.cpp
)
target_include_directories(pfm_feature_io PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_feature_io PUBLIC ${TORCH_LIBRARIES})
target_compile_options(pfm_feature_io PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm STATIC
    src/cli/commands.cpp
    src/data/image_dataset.cpp
    src/data/pair_archive_dataset.cpp
    src/data/synthetic_pair.cpp
    src/data/synthetic_pair_cache.cpp
    src/data/synthetic_pair_dataset.cpp
    src/infer/eval_pipeline.cpp
    src/infer/feature_extractor.cpp
    src/infer/match_metrics.cpp
    src/infer/matching_pipeline.cpp
    src/infer/pipeline.cpp
    src/infer/visualization.cpp
    src/losses/losses.cpp
    src/models/pfm_model_v21.cpp
    src/models/planetary_graph_matcher.cpp
    src/train/trainer.cpp
    src/train/training_visualization.cpp
)

target_include_directories(pfm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)

target_link_libraries(pfm PUBLIC pfm_core pfm_image pfm_geometry pfm_augment pfm_eval pfm_checkpoint_gate_core pfm_feature_io pfm_dataloader pfm_logging pfm_optim pfm_losses ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_compile_options(pfm PRIVATE -Wall -Wextra -Wpedantic)

add_executable(pfm_cli src/main.cpp)
target_link_libraries(pfm_cli PRIVATE pfm)

add_executable(pfm_rotation_sweep_eval tools/rotation_sweep_eval.cpp)
target_link_libraries(pfm_rotation_sweep_eval PRIVATE pfm ${OpenCV_LIBS})

add_executable(pfm_feature_pair_diagnostics tools/feature_pair_diagnostics.cpp)
target_link_libraries(pfm_feature_pair_diagnostics PRIVATE pfm)

add_executable(pfm_warp_image_diagnostics tools/warp_image_diagnostics.cpp)
target_link_libraries(pfm_warp_image_diagnostics PRIVATE pfm)

add_executable(pfm_match_residual_diagnostics tools/match_residual_diagnostics.cpp)
target_link_libraries(pfm_match_residual_diagnostics PRIVATE pfm)

add_executable(pfm_cache_eval_report tools/cache_eval_report.cpp)
target_link_libraries(pfm_cache_eval_report PRIVATE pfm pfm_cache_eval)

add_executable(pfm_checkpoint_gate tools/checkpoint_gate.cpp)
target_link_libraries(pfm_checkpoint_gate PRIVATE pfm)

add_executable(pfm_interpolate_checkpoints tools/interpolate_checkpoints.cpp)
target_link_libraries(pfm_interpolate_checkpoints PRIVATE pfm)

add_executable(pfm_print_checkpoint_config tools/print_checkpoint_config.cpp)
target_link_libraries(pfm_print_checkpoint_config PRIVATE pfm)

add_executable(pfm_print_v21_state_keys tools/print_pfm_v21_state_keys.cpp)
target_link_libraries(pfm_print_v21_state_keys PRIVATE pfm)

add_executable(pfm_pair_cache_training_smoke tools/pair_cache_training_smoke.cpp)
target_link_libraries(pfm_pair_cache_training_smoke PRIVATE pfm)

add_executable(pfm_texture_descriptor_diagnostics tools/texture_descriptor_diagnostics.cpp)
target_link_libraries(pfm_texture_descriptor_diagnostics PRIVATE pfm)

add_executable(pfm_tests
    tests/test_main.cpp
    src/cli/commands_test.cpp
    modules/core/device_test.cpp
    modules/core/tensor_utils_test.cpp
    modules/core/timer_test.cpp
    modules/runtime/runtime_test.cpp
    modules/dataloader/dataloader_test.cpp
    modules/logging/logging_test.cpp
    modules/optim/optim_test.cpp
    modules/cache_eval/cache_eval_metrics_test.cpp
    modules/match_quality/spatial_match_quality_test.cpp
    modules/training_schedule/training_schedule_test.cpp
    modules/adaptive_match_policy/adaptive_match_policy.test.cpp
    modules/checkpoint_gate/checkpoint_gate_test.cpp
    modules/feature_io/feature_codec_test.cpp
    modules/feature_io/match_codec_test.cpp
    src/infer/eval_pipeline_test.cpp
    src/infer/feature_extractor_test.cpp
    src/infer/match_metrics_test.cpp
    src/infer/matching_pipeline_test.cpp
    src/infer/pipeline_test.cpp
    src/infer/visualization_test.cpp
    src/data/image_dataset_test.cpp
    modules/image/image_io_test.cpp
    modules/image/intensity_mask_test.cpp
    modules/image/normalization_test.cpp
    src/data/pair_archive_dataset_test.cpp
    src/data/synthetic_pair_test.cpp
    src/data/synthetic_pair_cache_test.cpp
    src/augment/augment_test.cpp
    modules/eval/metrics_test.cpp
    modules/geometry/warp_test.cpp
    src/losses/losses_test.cpp
    src/models/pfm_model_v21_test.cpp
    src/models/planetary_graph_matcher_test.cpp
    src/train/trainer_test.cpp
    src/train/training_visualization_test.cpp
)
target_include_directories(pfm_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(pfm_tests PRIVATE pfm pfm_core pfm_image pfm_geometry pfm_augment pfm_eval pfm_runtime pfm_dataloader pfm_logging pfm_optim pfm_losses pfm_cache_eval pfm_match_quality pfm_training_schedule pfm_adaptive_match_policy)
add_test(NAME pfm_tests COMMAND pfm_tests)
