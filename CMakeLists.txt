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

add_library(pfm_core STATIC
    src/core/device.cpp
    src/core/tensor_utils.cpp
    src/core/timer.cpp
)
target_include_directories(pfm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_core PUBLIC ${TORCH_LIBRARIES})
target_compile_options(pfm_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_image STATIC
    src/data/image_io.cpp
    src/data/normalization.cpp
    src/data/intensity_mask.cpp
)
target_include_directories(pfm_image PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_image PUBLIC pfm_core ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_compile_options(pfm_image PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm_geometry STATIC
    src/geometry/warp.cpp
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
    src/eval/metrics.cpp
)
target_include_directories(pfm_eval PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)
target_link_libraries(pfm_eval PUBLIC ${TORCH_LIBRARIES})
target_compile_options(pfm_eval PRIVATE -Wall -Wextra -Wpedantic)

add_library(pfm STATIC
    src/cli/commands.cpp
    src/data/image_dataset.cpp
    src/data/synthetic_pair.cpp
    src/data/synthetic_pair_cache.cpp
    src/data/synthetic_pair_dataset.cpp
    src/infer/eval_pipeline.cpp
    src/infer/feature_codec.cpp
    src/infer/feature_extractor.cpp
    src/infer/match_codec.cpp
    src/infer/matching_pipeline.cpp
    src/infer/pipeline.cpp
    src/infer/visualization.cpp
    src/losses/losses.cpp
    src/models/backbone.cpp
    src/models/dense_head.cpp
    src/models/matcher.cpp
    src/models/planetary_graph_matcher.cpp
    src/models/sparse_head.cpp
    src/train/trainer.cpp
    src/train/training_visualization.cpp
)

target_include_directories(pfm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)

target_link_libraries(pfm PUBLIC pfm_core pfm_image pfm_geometry pfm_augment pfm_eval pfm_dataloader pfm_logging ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_compile_options(pfm PRIVATE -Wall -Wextra -Wpedantic)

add_executable(pfm_cli src/main.cpp)
target_link_libraries(pfm_cli PRIVATE pfm)

add_executable(pfm_rotation_sweep_eval tools/rotation_sweep_eval.cpp)
target_link_libraries(pfm_rotation_sweep_eval PRIVATE pfm ${OpenCV_LIBS})

add_executable(pfm_tests
    tests/test_main.cpp
    src/cli/commands_test.cpp
    src/core/device_test.cpp
    src/core/tensor_utils_test.cpp
    src/core/timer_test.cpp
    modules/runtime/runtime_test.cpp
    modules/dataloader/dataloader_test.cpp
    modules/logging/logging_test.cpp
    src/infer/eval_pipeline_test.cpp
    src/infer/feature_codec_test.cpp
    src/infer/feature_extractor_test.cpp
    src/infer/match_codec_test.cpp
    src/infer/matching_pipeline_test.cpp
    src/infer/pipeline_test.cpp
    src/infer/visualization_test.cpp
    src/data/image_dataset_test.cpp
    src/data/image_io_test.cpp
    src/data/intensity_mask_test.cpp
    src/data/normalization_test.cpp
    src/data/synthetic_pair_test.cpp
    src/data/synthetic_pair_cache_test.cpp
    src/augment/augment_test.cpp
    src/eval/metrics_test.cpp
    src/geometry/warp_test.cpp
    src/losses/losses_test.cpp
    src/models/backbone_test.cpp
    src/models/dense_head_test.cpp
    src/models/matcher_test.cpp
    src/models/planetary_graph_matcher_test.cpp
    src/models/sparse_head_test.cpp
    src/train/trainer_test.cpp
    src/train/training_visualization_test.cpp
)
target_include_directories(pfm_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(pfm_tests PRIVATE pfm pfm_core pfm_image pfm_geometry pfm_augment pfm_eval pfm_runtime pfm_dataloader pfm_logging)
add_test(NAME pfm_tests COMMAND pfm_tests)
