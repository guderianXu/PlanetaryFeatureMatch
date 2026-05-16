#include <string>

#include "core/timer.h"
#include "tests/test_harness.h"

static void timer_formats_seconds_with_three_decimals() {
    PFM_REQUIRE(pfm::formatSeconds(1.23456) == "1.235");
    PFM_REQUIRE(pfm::formatSeconds(0.0) == "0.000");
}

static void timer_elapsed_seconds_is_non_negative() {
    pfm::Timer timer;

    PFM_REQUIRE(timer.elapsedSeconds() >= 0.0);
}

void register_timer_tests() {
    register_test("timer_formats_seconds_with_three_decimals", timer_formats_seconds_with_three_decimals);
    register_test("timer_elapsed_seconds_is_non_negative", timer_elapsed_seconds_is_non_negative);
}
