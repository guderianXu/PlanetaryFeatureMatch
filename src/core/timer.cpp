#include <iomanip>
#include <sstream>

#include "core/timer.h"

namespace pfm {

Timer::Timer() : _start(std::chrono::steady_clock::now()) {}

void Timer::reset() {
    _start = std::chrono::steady_clock::now();
}

double Timer::elapsedSeconds() const {
    const auto elapsed = std::chrono::steady_clock::now() - _start;
    return std::chrono::duration<double>(elapsed).count();
}

std::string formatSeconds(double seconds) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << seconds;
    return stream.str();
}

}  // namespace pfm
