cmake_minimum_required(VERSION 3.18)
project(planetary_feature_matching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

enable_testing()

add_library(pfm STATIC
    modules/cli/commands.cpp
    modules/core/device.cpp
    modules/core/tensor_utils.cpp
    modules/infer/eval_pipeline.cpp
    modules/infer/feature_codec.cpp
    modules/infer/feature_extractor.cpp
    modules/infer/match_codec.cpp
    modules/infer/matching_pipeline.cpp
    modules/infer/pipeline.cpp
    modules/data/image_dataset.cpp
    modules/data/image_io.cpp
    modules/data/normalization.cpp
    modules/data/synthetic_pair.cpp
    modules/eval/metrics.cpp
    modules/geometry/warp.cpp
    modules/losses/losses.cpp
    modules/models/backbone.cpp
    modules/models/dense_head.cpp
    modules/models/matcher.cpp
    modules/models/sparse_head.cpp
    modules/train/trainer.cpp
)

target_include_directories(pfm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)

target_link_libraries(pfm PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_compile_options(pfm PRIVATE -Wall -Wextra -Wpedantic)

add_executable(pfm_cli src/main.cpp)
target_link_libraries(pfm_cli PRIVATE pfm)

add_executable(pfm_tests
    tests/test_main.cpp
    modules/cli/commands_test.cpp
    modules/core/device_test.cpp
    modules/core/tensor_utils_test.cpp
    modules/infer/eval_pipeline_test.cpp
    modules/infer/feature_codec_test.cpp
    modules/infer/feature_extractor_test.cpp
    modules/infer/match_codec_test.cpp
    modules/infer/matching_pipeline_test.cpp
    modules/infer/pipeline_test.cpp
    modules/data/image_dataset_test.cpp
    modules/data/image_io_test.cpp
    modules/data/normalization_test.cpp
    modules/data/synthetic_pair_test.cpp
    modules/eval/metrics_test.cpp
    modules/geometry/warp_test.cpp
    modules/losses/losses_test.cpp
    modules/models/backbone_test.cpp
    modules/models/dense_head_test.cpp
    modules/models/matcher_test.cpp
    modules/models/sparse_head_test.cpp
    modules/train/trainer_test.cpp
)
target_include_directories(pfm_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(pfm_tests PRIVATE pfm)
add_test(NAME pfm_tests COMMAND pfm_tests)
