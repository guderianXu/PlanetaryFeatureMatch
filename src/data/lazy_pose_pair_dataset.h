#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "data/synthetic_pair.h"
#include "dataloader/collator.h"
#include "dataloader/dataset.h"

namespace pfm
{

struct PoseCamera
{
    /// 水平焦距，单位为像素。
    double fu = 0.0;
    /// 垂直焦距，单位为像素。
    double fv = 0.0;
    /// 主点 u 坐标，单位为像素。
    double cu = 0.0;
    /// 主点 v 坐标，单位为像素。
    double cv = 0.0;
    /// 相机中心世界坐标。
    std::array<double, 3> center{};
    /// world-to-camera 旋转矩阵，row-major。
    std::array<double, 9> rotation_world_to_camera{};
    /// 解析来源 TSAI 路径。
    std::filesystem::path path;
};

struct PoseRenderRecord
{
    /// 渲染姿态唯一 ID。
    std::string pose_id;
    /// 同一位置下的 base ID。
    std::string base_id;
    /// 变体名称，例如 nadir、small_01、mid_01。
    std::string variant;
    /// 数据集 split。
    std::string split;
    /// 经度，单位度。
    double lon_deg = 0.0;
    /// 纬度，单位度。
    double lat_deg = 0.0;
    /// TSAI 相机路径。
    std::filesystem::path tsai_path;
    /// 原始 render 图像路径。
    std::filesystem::path render_image_path;
    /// 训练侧实际读取图像路径，优先来自 uint8 manifest。
    std::filesystem::path selected_image_path;
    /// 深度图路径。
    std::filesystem::path depth_path;
    /// 渲染 chunk 序号。
    int64_t chunk_index = 0;
};

struct LazyPosePairBuildOptions
{
    /// reference 变体，默认 nadir。
    std::string reference_variant = "nadir";
    /// target 变体；为空时使用 small_01..extreme_03 默认列表。
    std::vector<std::string> target_variants;
    /// 可选 split 过滤；为空时保留全部 split。
    std::string split_filter;
    /// 是否同时构建 target->reference 反向 pair。
    bool bidirectional = false;
    /// 是否跳过缺失 image/depth/TSAI 的记录。
    bool require_files = true;
    /// pair 上限；0 表示不限制。
    int64_t limit_pairs = 0;
};

struct LazyPosePairDatasetConfig : LazyPosePairBuildOptions
{
    /// pose render manifest CSV。
    std::filesystem::path render_manifest;
    /// uint8 manifest CSV；为空或不存在时读取 render image。
    std::filesystem::path uint8_manifest;
    /// 按需输出 crop 尺寸；0 表示输出完整 2048 等原始大小。
    int64_t crop_size = 0;
    /// 深度一致性绝对阈值，单位米。
    double absolute_depth_tolerance_m = 100.0;
    /// 深度一致性相对阈值。
    double relative_depth_tolerance = 0.005;
};

struct LazyPosePairSpec
{
    /// pair 在构建列表中的稳定序号。
    std::size_t pair_index = 0;
    /// pair 所属 split。
    std::string split;
    /// 同一位置下的 base ID。
    std::string base_id;
    /// reference pose ID。
    std::string reference_pose_id;
    /// target pose ID。
    std::string target_pose_id;
    /// reference 变体。
    std::string reference_variant;
    /// target 变体。
    std::string target_variant;
    /// reference 训练图像路径。
    std::filesystem::path reference_image_path;
    /// target 训练图像路径。
    std::filesystem::path target_image_path;
    /// reference depth 路径。
    std::filesystem::path reference_depth_path;
    /// target depth 路径。
    std::filesystem::path target_depth_path;
    /// reference TSAI 路径。
    std::filesystem::path reference_tsai_path;
    /// target TSAI 路径。
    std::filesystem::path target_tsai_path;
};

struct PoseWarpResult
{
    /// A 到 B 的 HxWx2 像素坐标 warp。
    torch::Tensor warp_a_to_b;
    /// A 像素投影到 B 后深度一致的有效 mask。
    torch::Tensor valid_mask;
    /// A depth 中有效像素占比。
    double valid_a_fraction = 0.0;
    /// A 有效像素投影落入 B 图像范围内的占比。
    double target_inside_fraction = 0.0;
    /// A 有效像素最终可监督占比。
    double valid_pair_fraction = 0.0;
    /// 最终有效像素数量。
    int64_t valid_pixels = 0;
};

/// 读取 uint8 manifest，返回 render image 路径到 uint8 image 路径的映射。
/// @param path `source_path,uint8_path` CSV；路径为空或不存在时返回空映射。
/// @return 以原始 source_path 字符串为 key 的路径映射。
std::unordered_map<std::string, std::filesystem::path> readPoseUint8Manifest(const std::filesystem::path& path);

/// 读取 pose render manifest，并按 uint8 manifest 替换训练图像路径。
/// @param path render_manifest.csv。
/// @param uint8_paths source_path 到 uint8_path 的映射。
/// @return manifest 中的渲染记录。
/// @throws std::invalid_argument 当 CSV 缺失必要列或数值非法时抛出。
std::vector<PoseRenderRecord>
readPoseRenderManifest(const std::filesystem::path& path,
                       const std::unordered_map<std::string, std::filesystem::path>& uint8_paths);

/// 从 render records 构建 nadir->small/mid/extreme 的 lazy pair 规格列表。
/// @param records render manifest 记录。
/// @param options reference/target/split/文件存在性过滤选项。
/// @return 稳定排序的 pair spec。
std::vector<LazyPosePairSpec> buildLazyPosePairSpecs(const std::vector<PoseRenderRecord>& records,
                                                     const LazyPosePairBuildOptions& options);

/// 解析 ASP TSAI 相机文件。
/// @param path TSAI 文件路径。
/// @return C++ 投影使用的相机参数。
/// @throws std::invalid_argument 当 TSAI 缺少 fu/fv/cu/cv/C/R 时抛出。
PoseCamera parsePoseTsaiCamera(const std::filesystem::path& path);

/// CPU 侧将 A depth 投影到 B 相机并生成 HxWx2 warp 和 valid mask。
/// @param depth_a A 图深度，HxW float CPU tensor。
/// @param depth_b B 图深度，HxW float CPU tensor。
/// @param camera_a A 相机参数。
/// @param camera_b B 相机参数。
/// @param absolute_depth_tolerance_m 深度一致性绝对阈值。
/// @param relative_depth_tolerance 深度一致性相对阈值。
/// @return warp、valid mask 和覆盖统计。
PoseWarpResult projectPoseDepthWarp(const torch::Tensor& depth_a, const torch::Tensor& depth_b,
                                    const PoseCamera& camera_a, const PoseCamera& camera_b,
                                    double absolute_depth_tolerance_m, double relative_depth_tolerance);

/// 返回 lazy pose pair 使用的通用 collator。
/// @return view_a/view_b/warp_a_to_b/valid_mask 的 TensorBatchCollator。
TensorBatchCollator makeLazyPosePairCollator();

class LazyPosePairDataset : public TensorDataset
{
  public:
    /// 从 pose render manifest 创建按需 pair 数据集。
    /// @param config manifest、pair 构建、crop 和深度一致性配置。
    /// @throws std::invalid_argument 当 manifest 无效或没有可用 pair 时抛出。
    explicit LazyPosePairDataset(const LazyPosePairDatasetConfig& config);

    /// 返回可生成 pair 数量。
    /// @return pair spec 数量。
    size_t size() const override;

    /// 返回指定 pair spec。
    /// @param index pair 索引。
    /// @throws std::out_of_range 当 index 非法时抛出。
    const LazyPosePairSpec& spec(size_t index) const;

    /// 按需读取 image/depth/TSAI 并生成训练 pair。
    /// @param index pair 索引。
    /// @return view_a、view_b、warp_a_to_b、valid_mask。
    /// @throws std::out_of_range 当 index 非法时抛出。
    SyntheticPair load(size_t index) const;

    /// 按 TensorDataset 接口返回未组 batch 的样本。
    /// @param index pair 索引。
    /// @return 包含 view_a、view_b、warp_a_to_b、valid_mask 的 TensorBatch。
    TensorBatch get(size_t index) override;

  private:
    std::vector<LazyPosePairSpec> _specs;
    int64_t _crop_size = 0;
    double _absolute_depth_tolerance_m = 100.0;
    double _relative_depth_tolerance = 0.005;
};

} // namespace pfm
