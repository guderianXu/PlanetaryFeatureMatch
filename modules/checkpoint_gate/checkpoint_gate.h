#pragma once

#include <cstdint>
#include <string>

namespace pfm
{

struct CheckpointGateMetrics
{
    /// 满足质量门控判定的正确匹配数。
    int64_t correct_matches = 0;
    /// 被判定为错误的匹配数。
    int64_t wrong_matches = 0;
    /// 匹配精度，通常为 correct_matches / total_matches。
    double precision = 0.0;
    /// 图匹配注意力实际工作量相对满量注意力的比例，缺失时按 0 处理以兼容旧输出。
    double graph_attention_work_fraction = 0.0;

    /// @return 正确匹配与错误匹配的总数。
    int64_t total_matches() const;
};

struct CheckpointGateThreshold
{
    /// 通过质量门控所需的最小正确匹配数。
    int64_t min_correct_matches = 0;
    /// 通过质量门控所需的最小匹配精度。
    double min_precision = 0.0;
    /// 允许通过质量门控的最大图匹配注意力工作量比例。
    double max_graph_attention_work_fraction = 1.0;
};

struct CheckpointGateDecision
{
    /// 检查点是否通过质量门控。
    bool passed = false;
    /// 通过或失败原因，面向日志和报告。
    std::string reason;
};

/// 从 match 命令输出文本中解析检查点质量门控需要的匹配指标。
/// @param match_output match 命令输出，需包含 correct_matches、wrong_matches 和 match_precision 字段。
/// @return 解析后的匹配指标。
/// @throws std::invalid_argument 当必需字段缺失或格式非法时抛出。
CheckpointGateMetrics parse_checkpoint_gate_metrics(const std::string& match_output);

/// 根据阈值判断检查点是否通过质量门控。
/// @param metrics 已解析的匹配指标。
/// @param threshold 最小正确匹配数和最小精度阈值。
/// @return 通过状态和面向日志的原因说明。
CheckpointGateDecision evaluate_checkpoint_gate_metrics(const CheckpointGateMetrics& metrics,
                                                        const CheckpointGateThreshold& threshold);

} // namespace pfm
