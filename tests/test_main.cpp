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

void register_tensor_utils_tests();
void register_image_io_tests();
void register_normalization_tests();
void register_warp_tests();
void register_synthetic_pair_tests();
void register_backbone_tests();
void register_sparse_head_tests();
void register_dense_head_tests();
void register_matcher_tests();
void register_loss_tests();
void register_metric_tests();
void register_cli_tests();

int main() {
    register_tensor_utils_tests();
    register_image_io_tests();
    register_normalization_tests();
    register_warp_tests();
    register_synthetic_pair_tests();
    register_backbone_tests();
    register_sparse_head_tests();
    register_dense_head_tests();
    register_matcher_tests();
    register_loss_tests();
    register_metric_tests();
    register_cli_tests();

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

