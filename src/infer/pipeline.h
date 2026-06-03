#pragma once

#include "cli/commands.h"

namespace pfm
{

/// 执行训练命令。
/// @param options 已解析的 CLI 选项，包含 image_dir、检查点、设备、训练轮数和批大小等字段。
/// @return 成功写出可加载检查点时返回 0；校验或训练失败时返回非 0。
int run_train_command(const CliOptions& options);

/// 执行特征提取命令。
/// @param options 已解析的 CLI 选项，包含影像、检查点、输出、设备和解码配置。
/// @return 成功写出可加载特征文件时返回 0；校验或提取失败时返回非 0。
int run_extract_command(const CliOptions& options);

/// 执行影像匹配命令。
/// @param options 已解析的 CLI 选项，包含两幅影像、检查点、输出、设备和解码配置。
/// @return 成功写出可加载匹配文件时返回 0；校验、特征提取或匹配失败时返回非 0。
int run_match_command(const CliOptions& options);

/// 执行影像对评估命令。
/// @param options 已解析的 CLI 选项，包含影像对文件、检查点、输出、设备和解码配置。
/// @return 成功写出 LibTorch report archive 时返回 0；校验、特征提取或匹配失败时返回非 0。
int run_eval_command(const CliOptions& options);

/// 执行模型导出命令。
/// @param options 已解析的 CLI 选项，包含检查点和输出。
/// @return 成功写出可加载检查点副本时返回 0；校验或导出失败时返回非 0。
int run_export_command(const CliOptions& options);

} // namespace pfm
