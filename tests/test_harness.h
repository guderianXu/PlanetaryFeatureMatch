#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#define PFM_REQUIRE(cond) do { if (!(cond)) throw std::runtime_error(std::string("require failed: ") + #cond); } while (0)
#define PFM_REQUIRE_CLOSE(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error("close check failed"); } while (0)
#define PFM_REQUIRE_THROWS_AS(expr, exception_type) \
    do { \
        bool thrown = false; \
        try { \
            (void)(expr); \
        } catch (const exception_type&) { \
            thrown = true; \
        } \
        PFM_REQUIRE(thrown); \
    } while (0)
#define PFM_REQUIRE_INVALID_ARG(expr) PFM_REQUIRE_THROWS_AS(expr, std::invalid_argument)

void register_test(const std::string& name, void (*fn)());
