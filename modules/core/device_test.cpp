#include <torch/torch.h>

#include "core/device.h"
#include "tests/test_harness.h"

namespace
{

static void device_resolves_cpu()
{
    auto device = pfm::resolve_compute_device("cpu");

    PFM_REQUIRE(device.is_cpu());
}

static void device_rejects_invalid_names()
{
    PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device(""));
    PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device("gpu"));
    PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device("CPU"));
    PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device("cpu:0"));
}

static void device_rejects_invalid_cuda_indices()
{
    PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device("cuda:"));
    PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device("cuda:-1"));
    PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device("cuda:abc"));
    PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device("cuda:0:1"));
}

static void device_handles_cuda_availability()
{
    if (!torch::cuda::is_available())
    {
        PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device("cuda"));
        PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device("cuda:0"));
        return;
    }

    auto implicit_device = pfm::resolve_compute_device("cuda");
    auto explicit_device = pfm::resolve_compute_device("cuda:0");

    PFM_REQUIRE(implicit_device.is_cuda());
    PFM_REQUIRE(explicit_device.is_cuda());
    PFM_REQUIRE(implicit_device.index() == 0);
    PFM_REQUIRE(explicit_device.index() == 0);
}

static void device_rejects_cuda_index_out_of_range()
{
    if (!torch::cuda::is_available())
    {
        return;
    }

    const auto invalid_device = "cuda:" + std::to_string(torch::cuda::device_count());

    PFM_REQUIRE_INVALID_ARG(pfm::resolve_compute_device(invalid_device));
}

} // namespace

void register_device_tests()
{
    register_test("device resolves cpu", device_resolves_cpu);
    register_test("device rejects invalid names", device_rejects_invalid_names);
    register_test("device rejects invalid cuda indices", device_rejects_invalid_cuda_indices);
    register_test("device handles cuda availability", device_handles_cuda_availability);
    register_test("device rejects cuda index out of range", device_rejects_cuda_index_out_of_range);
}
