#pragma once

#include <filesystem>
#include <string>

#include <torch/torch.h>

#include "feature_io/feature_codec.h"
#include "feature_io/match_codec.h"

namespace pfm
{

/// 保存显示提取特征点的 PNG 叠加图。
/// @param image_path 作为可视化背景的源影像路径。
/// @param feature_set 已解码特征张量，关键点使用影像坐标顺序 {x, y}。
/// @param visualization_dir PNG 输出目录；不存在时会创建。
/// @return 已写出的 PNG 文件路径。
/// @throws std::invalid_argument 当影像无法读取或 PNG 无法写出时抛出。
std::filesystem::path save_feature_visualization(const std::string& image_path, const FeatureSet& feature_set,
                                                 const std::string& visualization_dir);

/// 将 feature-map 坐标缩放到源影像像素后保存 PNG 叠加图。
/// @param image_path 作为可视化背景的源影像路径。
/// @param feature_set 已解码特征张量，关键点使用特征图坐标。
/// @param visualization_dir PNG 输出目录；不存在时会创建。
/// @param feature_map_width 产生关键点的特征图宽度。
/// @param feature_map_height 产生关键点的特征图高度。
/// @return 已写出的 PNG 文件路径。
/// @throws std::invalid_argument 当尺寸非法、影像无法读取或 PNG 无法写出时抛出。
std::filesystem::path save_feature_visualization(const std::string& image_path, const FeatureSet& feature_set,
                                                 const std::string& visualization_dir, int64_t feature_map_width,
                                                 int64_t feature_map_height);

/// 保存两幅影像之间匹配点的左右拼接 PNG 叠加图。
/// @param image_a_path 第一幅源影像路径。
/// @param image_b_path 第二幅源影像路径。
/// @param match_set 匹配张量，points 使用影像坐标顺序 {x, y}，并包含置信度分数。
/// @param visualization_dir PNG 输出目录；不存在时会创建。
/// @return 已写出的 PNG 文件路径。
/// @throws std::invalid_argument 当任一影像无法读取或 PNG 无法写出时抛出。
std::filesystem::path save_match_visualization(const std::string& image_a_path, const std::string& image_b_path,
                                               const MatchSet& match_set, const std::string& visualization_dir);

/// 保存左右拼接 PNG 叠加图，并按稠密变形场正确性给匹配着色。
/// 正确匹配为绿色，错误匹配为红色。
std::filesystem::path save_match_visualization(const std::string& image_a_path, const std::string& image_b_path,
                                               const MatchSet& match_set, const std::string& visualization_dir,
                                               const torch::Tensor& warp_a_to_b, double correct_threshold_pixels);

/// 将 feature-map 点坐标缩放到源影像像素后保存 PNG 匹配叠加图。
/// @param image_a_path 第一幅源影像路径。
/// @param image_b_path 第二幅源影像路径。
/// @param match_set 匹配张量，points 使用 feature-map 坐标。
/// @param visualization_dir PNG 输出目录；不存在时会创建。
/// @param feature_map_a_width 第一幅特征图宽度。
/// @param feature_map_a_height 第一幅特征图高度。
/// @param feature_map_b_width 第二幅特征图宽度。
/// @param feature_map_b_height 第二幅特征图高度。
/// @return 已写出的 PNG 文件路径。
/// @throws std::invalid_argument 当尺寸非法、任一影像无法读取或 PNG 无法写出时抛出。
std::filesystem::path save_match_visualization(const std::string& image_a_path, const std::string& image_b_path,
                                               const MatchSet& match_set, const std::string& visualization_dir,
                                               int64_t feature_map_a_width, int64_t feature_map_a_height,
                                               int64_t feature_map_b_width, int64_t feature_map_b_height);

/// 保存已缩放坐标的 PNG 匹配叠加图，并按稠密变形场正确性给匹配着色。
std::filesystem::path save_match_visualization(const std::string& image_a_path, const std::string& image_b_path,
                                               const MatchSet& match_set, const std::string& visualization_dir,
                                               int64_t feature_map_a_width, int64_t feature_map_a_height,
                                               int64_t feature_map_b_width, int64_t feature_map_b_height,
                                               const torch::Tensor& warp_a_to_b, double correct_threshold_pixels);

/// 使用稀疏匹配索引和特征图关键点保存 PNG 匹配叠加图。
/// @param image_a_path 第一幅源影像路径。
/// @param image_b_path 第二幅源影像路径。
/// @param features_a 第一幅影像的已解码特征集合。
/// @param features_b 第二幅影像的已解码特征集合。
/// @param match_set 包含稀疏索引和/或稠密点的匹配张量。
/// @param visualization_dir PNG 输出目录；不存在时会创建。
/// @param feature_map_a_width 第一幅特征图宽度。
/// @param feature_map_a_height 第一幅特征图高度。
/// @param feature_map_b_width 第二幅特征图宽度。
/// @param feature_map_b_height 第二幅特征图高度。
/// @return 已写出的 PNG 文件路径。
/// @throws std::invalid_argument 当张量或尺寸非法、任一影像无法读取或写出失败时抛出。
std::filesystem::path save_match_visualization(const std::string& image_a_path, const std::string& image_b_path,
                                               const FeatureSet& features_a, const FeatureSet& features_b,
                                               const MatchSet& match_set, const std::string& visualization_dir,
                                               int64_t feature_map_a_width, int64_t feature_map_a_height,
                                               int64_t feature_map_b_width, int64_t feature_map_b_height);

/// 保存稀疏/稠密 PNG 匹配叠加图，并按稠密变形场正确性给匹配着色。
std::filesystem::path save_match_visualization(const std::string& image_a_path, const std::string& image_b_path,
                                               const FeatureSet& features_a, const FeatureSet& features_b,
                                               const MatchSet& match_set, const std::string& visualization_dir,
                                               int64_t feature_map_a_width, int64_t feature_map_a_height,
                                               int64_t feature_map_b_width, int64_t feature_map_b_height,
                                               const torch::Tensor& warp_a_to_b, double correct_threshold_pixels);

} // namespace pfm
