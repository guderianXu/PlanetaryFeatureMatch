#pragma once

#include <chrono>
#include <string>

namespace pfm
{

class Timer
{
  public:
    /// 创建计时器，起点为构造时刻。
    Timer();

    /// 将计时起点重置为当前时刻。
    void reset();

    /// 返回从构造或最近一次 reset 起经过的墙钟秒数。
    /// @return 非负秒数。
    double elapsedSeconds() const;

  private:
    std::chrono::steady_clock::time_point _start;
};

/// 将秒数格式化为三位小数，供 CLI 和训练日志输出。
/// @param seconds 持续时间，单位为秒。
/// @return 不带单位后缀的定点小数字符串。
std::string formatSeconds(double seconds);

} // namespace pfm
