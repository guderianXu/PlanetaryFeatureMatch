#pragma once

#include <chrono>
#include <string>

namespace pfm {

class Timer {
public:
    /// Creates a timer starting at construction time.
    Timer();

    /// Resets the timer start time to now.
    void reset();

    /// Returns elapsed wall-clock seconds since construction or reset.
    /// @return Non-negative elapsed seconds.
    double elapsedSeconds() const;

private:
    std::chrono::steady_clock::time_point _start;
};

/// Formats seconds with three decimal places for CLI output.
/// @param seconds Duration in seconds.
/// @return Fixed-point string without unit suffix.
std::string formatSeconds(double seconds);

}  // namespace pfm
