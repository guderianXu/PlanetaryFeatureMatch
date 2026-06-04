#include "data/lazy_pose_pair_dataset.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include "dataloader/tensor_batch.h"
#include "image/image_io.h"

namespace pfm
{
namespace
{

using torch::indexing::Slice;

constexpr std::array<const char*, 9> DEFAULT_TARGET_VARIANTS = {
    "small_01", "small_02", "small_03", "mid_01", "mid_02", "mid_03", "extreme_01", "extreme_02", "extreme_03",
};

std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;
    for (std::size_t index = 0; index < line.size(); ++index)
    {
        const char ch = line[index];
        if (ch == '"')
        {
            if (in_quotes && index + 1 < line.size() && line[index + 1] == '"')
            {
                current.push_back('"');
                ++index;
            }
            else
            {
                in_quotes = !in_quotes;
            }
            continue;
        }
        if (ch == ',' && !in_quotes)
        {
            fields.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    fields.push_back(current);
    return fields;
}

std::map<std::string, std::size_t> headerIndex(const std::vector<std::string>& header)
{
    std::map<std::string, std::size_t> result;
    for (std::size_t index = 0; index < header.size(); ++index)
    {
        result.emplace(header[index], index);
    }
    return result;
}

std::string csvField(const std::vector<std::string>& row, const std::map<std::string, std::size_t>& header,
                     const std::string& name, const std::filesystem::path& path)
{
    const auto found = header.find(name);
    if (found == header.end())
    {
        throw std::invalid_argument(path.string() + " missing CSV column: " + name);
    }
    if (found->second >= row.size())
    {
        return {};
    }
    return row[found->second];
}

bool isDuplicateHeaderRow(const std::vector<std::string>& row)
{
    return !row.empty() && row[0] == "source_path";
}

std::vector<std::string> defaultTargetVariants()
{
    std::vector<std::string> variants;
    variants.reserve(DEFAULT_TARGET_VARIANTS.size());
    for (const auto* variant : DEFAULT_TARGET_VARIANTS)
    {
        variants.emplace_back(variant);
    }
    return variants;
}

bool fileExists(const std::filesystem::path& path)
{
    return !path.empty() && std::filesystem::exists(path);
}

bool recordFilesExist(const PoseRenderRecord& record)
{
    return fileExists(record.selected_image_path) && fileExists(record.depth_path) && fileExists(record.tsai_path);
}

double parseDoubleField(const std::string& value, const std::string& name, const std::filesystem::path& path)
{
    try
    {
        return std::stod(value);
    }
    catch (const std::exception& exc)
    {
        throw std::invalid_argument(path.string() + " invalid " + name + ": " + exc.what());
    }
}

int64_t parseInt64Field(const std::string& value, const std::string& name, const std::filesystem::path& path)
{
    try
    {
        return std::stoll(value);
    }
    catch (const std::exception& exc)
    {
        throw std::invalid_argument(path.string() + " invalid " + name + ": " + exc.what());
    }
}

std::vector<double> parseFloatList(const std::string& value)
{
    std::istringstream stream(value);
    std::vector<double> result;
    double number = 0.0;
    while (stream >> number)
    {
        result.push_back(number);
    }
    return result;
}

std::array<double, 3> matVec(const std::array<double, 9>& matrix, double x, double y, double z)
{
    return {matrix[0] * x + matrix[1] * y + matrix[2] * z,
            matrix[3] * x + matrix[4] * y + matrix[5] * z,
            matrix[6] * x + matrix[7] * y + matrix[8] * z};
}

std::array<double, 3> transposeMatVec(const std::array<double, 9>& matrix, double x, double y, double z)
{
    return {matrix[0] * x + matrix[3] * y + matrix[6] * z,
            matrix[1] * x + matrix[4] * y + matrix[7] * z,
            matrix[2] * x + matrix[5] * y + matrix[8] * z};
}

torch::Tensor readDepthTensor(const std::filesystem::path& path)
{
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        throw std::invalid_argument("depth image could not be loaded: " + path.string());
    }
    if (image.channels() > 1)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    cv::Mat float_image;
    image.convertTo(float_image, CV_32F);
    if (!float_image.isContinuous())
    {
        float_image = float_image.clone();
    }
    auto tensor = torch::from_blob(float_image.data, {float_image.rows, float_image.cols},
                                   torch::TensorOptions().dtype(torch::kFloat32))
                      .clone();
    return torch::nan_to_num(tensor, 0.0, 0.0, 0.0).contiguous();
}

torch::Tensor ensureSingleChannel(const torch::Tensor& image)
{
    if (image.dim() != 3)
    {
        throw std::invalid_argument("lazy pose image must have shape CxHxW");
    }
    if (image.size(0) == 1)
    {
        return image.contiguous();
    }
    return image.mean(0, true).contiguous();
}

torch::Tensor loadViewTensor(const std::filesystem::path& path)
{
    return ensureSingleChannel(load_image_tensor(path.string())).clamp(0.0, 1.0).contiguous();
}

int64_t clampCropStart(double center, int64_t crop_size, int64_t full_size)
{
    auto start = static_cast<int64_t>(std::llround(center - static_cast<double>(crop_size) * 0.5));
    start = std::max<int64_t>(0, start);
    return std::min<int64_t>(start, full_size - crop_size);
}

SyntheticPair cropPairDeterministic(const SyntheticPair& pair, int64_t crop_size)
{
    if (crop_size <= 0)
    {
        return pair;
    }
    const auto height = pair.valid_mask.size(0);
    const auto width = pair.valid_mask.size(1);
    if (crop_size > height || crop_size > width)
    {
        throw std::invalid_argument("lazy pose pair crop_size exceeds pair spatial size");
    }

    const auto ax0 = (width - crop_size) / 2;
    const auto ay0 = (height - crop_size) / 2;
    auto view_a = pair.view_a.index({Slice(), Slice(ay0, ay0 + crop_size), Slice(ax0, ax0 + crop_size)}).contiguous();
    auto warp = pair.warp_a_to_b.index({Slice(ay0, ay0 + crop_size), Slice(ax0, ax0 + crop_size), Slice()})
                    .contiguous();
    auto valid = pair.valid_mask.index({Slice(ay0, ay0 + crop_size), Slice(ax0, ax0 + crop_size)}).contiguous();

    auto valid_u8 = valid.to(torch::kUInt8).contiguous();
    auto valid_acc = valid_u8.accessor<uint8_t, 2>();
    auto warp_acc = warp.accessor<float, 3>();
    double sum_x = 0.0;
    double sum_y = 0.0;
    int64_t count = 0;
    for (int64_t y = 0; y < crop_size; ++y)
    {
        for (int64_t x = 0; x < crop_size; ++x)
        {
            if (valid_acc[y][x] != 0U)
            {
                sum_x += static_cast<double>(warp_acc[y][x][0]);
                sum_y += static_cast<double>(warp_acc[y][x][1]);
                ++count;
            }
        }
    }
    const double fallback_center_x = static_cast<double>(width) * 0.5;
    const double fallback_center_y = static_cast<double>(height) * 0.5;
    const double target_center_x = count > 0 ? sum_x / static_cast<double>(count) : fallback_center_x;
    const double target_center_y = count > 0 ? sum_y / static_cast<double>(count) : fallback_center_y;
    const auto bx0 = clampCropStart(target_center_x, crop_size, width);
    const auto by0 = clampCropStart(target_center_y, crop_size, height);

    auto view_b = pair.view_b.index({Slice(), Slice(by0, by0 + crop_size), Slice(bx0, bx0 + crop_size)}).contiguous();
    auto x_channel = warp.index({Slice(), Slice(), 0});
    auto y_channel = warp.index({Slice(), Slice(), 1});
    auto inside = (x_channel >= static_cast<double>(bx0)) & (x_channel < static_cast<double>(bx0 + crop_size)) &
                  (y_channel >= static_cast<double>(by0)) & (y_channel < static_cast<double>(by0 + crop_size));
    warp.index_put_({Slice(), Slice(), 0}, x_channel - static_cast<double>(bx0));
    warp.index_put_({Slice(), Slice(), 1}, y_channel - static_cast<double>(by0));
    valid = (valid & inside).contiguous();
    return SyntheticPair{view_a, view_b, warp.contiguous(), valid};
}

TensorBatch pairToTensorBatch(const SyntheticPair& pair)
{
    TensorBatch batch;
    batch["view_a"] = pair.view_a;
    batch["view_b"] = pair.view_b;
    batch["warp_a_to_b"] = pair.warp_a_to_b;
    batch["valid_mask"] = pair.valid_mask;
    return batch;
}

void validateDepthTensor(const torch::Tensor& depth, const char* name)
{
    if (depth.dim() != 2)
    {
        throw std::invalid_argument(std::string(name) + " must have shape HxW");
    }
    if (depth.scalar_type() != torch::kFloat32)
    {
        throw std::invalid_argument(std::string(name) + " must be float32");
    }
    if (!depth.device().is_cpu())
    {
        throw std::invalid_argument(std::string(name) + " must be a CPU tensor");
    }
}

} // namespace

std::unordered_map<std::string, std::filesystem::path> readPoseUint8Manifest(const std::filesystem::path& path)
{
    std::unordered_map<std::string, std::filesystem::path> mapping;
    if (path.empty() || !std::filesystem::exists(path))
    {
        return mapping;
    }

    std::ifstream stream(path);
    if (!stream)
    {
        throw std::invalid_argument("failed to open uint8 manifest: " + path.string());
    }
    std::string line;
    if (!std::getline(stream, line))
    {
        return mapping;
    }
    const auto header = headerIndex(splitCsvLine(line));
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto row = splitCsvLine(line);
        if (isDuplicateHeaderRow(row))
        {
            continue;
        }
        const auto source = csvField(row, header, "source_path", path);
        const auto target = csvField(row, header, "uint8_path", path);
        if (!source.empty() && !target.empty())
        {
            mapping[source] = std::filesystem::path(target);
        }
    }
    return mapping;
}

std::vector<PoseRenderRecord>
readPoseRenderManifest(const std::filesystem::path& path,
                       const std::unordered_map<std::string, std::filesystem::path>& uint8_paths)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::invalid_argument("failed to open render manifest: " + path.string());
    }
    std::string line;
    if (!std::getline(stream, line))
    {
        throw std::invalid_argument("render manifest is empty: " + path.string());
    }
    const auto header = headerIndex(splitCsvLine(line));
    std::vector<PoseRenderRecord> records;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto row = splitCsvLine(line);
        if (row.empty() || row[0] == "pose_id")
        {
            continue;
        }

        PoseRenderRecord record;
        record.pose_id = csvField(row, header, "pose_id", path);
        record.base_id = csvField(row, header, "base_id", path);
        record.variant = csvField(row, header, "variant", path);
        record.split = csvField(row, header, "split", path);
        record.lon_deg = parseDoubleField(csvField(row, header, "lon_deg", path), "lon_deg", path);
        record.lat_deg = parseDoubleField(csvField(row, header, "lat_deg", path), "lat_deg", path);
        record.tsai_path = csvField(row, header, "tsai_path", path);
        record.render_image_path = csvField(row, header, "image_path", path);
        record.depth_path = csvField(row, header, "depth_path", path);
        record.chunk_index = parseInt64Field(csvField(row, header, "chunk_index", path), "chunk_index", path);
        const auto uint8_path = uint8_paths.find(record.render_image_path.string());
        record.selected_image_path = uint8_path != uint8_paths.end() ? uint8_path->second : record.render_image_path;
        records.push_back(std::move(record));
    }
    return records;
}

std::vector<LazyPosePairSpec> buildLazyPosePairSpecs(const std::vector<PoseRenderRecord>& records,
                                                     const LazyPosePairBuildOptions& options)
{
    if (options.limit_pairs < 0)
    {
        throw std::invalid_argument("lazy pose pair limit_pairs must be nonnegative");
    }
    const auto target_variants = options.target_variants.empty() ? defaultTargetVariants() : options.target_variants;
    std::map<std::string, std::map<std::string, PoseRenderRecord>> by_base;
    for (const auto& record : records)
    {
        if (options.require_files && !recordFilesExist(record))
        {
            continue;
        }
        by_base[record.base_id][record.variant] = record;
    }

    std::vector<LazyPosePairSpec> specs;
    for (const auto& [base_id, variants] : by_base)
    {
        const auto reference_found = variants.find(options.reference_variant);
        if (reference_found == variants.end())
        {
            continue;
        }
        const auto& reference = reference_found->second;
        for (const auto& variant : target_variants)
        {
            const auto target_found = variants.find(variant);
            if (target_found == variants.end())
            {
                continue;
            }
            const auto& target = target_found->second;
            for (const auto& endpoints : {std::pair<const PoseRenderRecord*, const PoseRenderRecord*>(&reference,
                                                                                                        &target),
                                          std::pair<const PoseRenderRecord*, const PoseRenderRecord*>(&target,
                                                                                                        &reference)})
            {
                if (!options.bidirectional && endpoints.first != &reference)
                {
                    continue;
                }
                if (!options.split_filter.empty() && endpoints.first->split != options.split_filter)
                {
                    continue;
                }
                LazyPosePairSpec spec;
                spec.pair_index = specs.size();
                spec.split = endpoints.first->split;
                spec.base_id = base_id;
                spec.reference_pose_id = endpoints.first->pose_id;
                spec.target_pose_id = endpoints.second->pose_id;
                spec.reference_variant = endpoints.first->variant;
                spec.target_variant = endpoints.second->variant;
                spec.reference_image_path = endpoints.first->selected_image_path;
                spec.target_image_path = endpoints.second->selected_image_path;
                spec.reference_depth_path = endpoints.first->depth_path;
                spec.target_depth_path = endpoints.second->depth_path;
                spec.reference_tsai_path = endpoints.first->tsai_path;
                spec.target_tsai_path = endpoints.second->tsai_path;
                specs.push_back(std::move(spec));
                if (options.limit_pairs > 0 && static_cast<int64_t>(specs.size()) >= options.limit_pairs)
                {
                    return specs;
                }
            }
        }
    }
    return specs;
}

PoseCamera parsePoseTsaiCamera(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::invalid_argument("failed to open TSAI file: " + path.string());
    }

    std::map<std::string, std::vector<double>> values;
    std::string line;
    while (std::getline(stream, line))
    {
        const auto separator = line.find('=');
        if (separator == std::string::npos)
        {
            continue;
        }
        const auto key = line.substr(0, separator);
        const auto value = line.substr(separator + 1);
        std::string trimmed_key;
        std::copy_if(key.begin(), key.end(), std::back_inserter(trimmed_key),
                     [](unsigned char ch)
                     {
                         return !std::isspace(ch);
                     });
        const auto parsed = parseFloatList(value);
        if (!parsed.empty())
        {
            values[trimmed_key] = parsed;
        }
    }

    for (const auto& required : {"fu", "fv", "cu", "cv", "C", "R"})
    {
        if (values.find(required) == values.end())
        {
            throw std::invalid_argument(path.string() + " missing TSAI field: " + required);
        }
    }
    if (values["C"].size() != 3 || values["R"].size() != 9)
    {
        throw std::invalid_argument(path.string() + " has invalid C/R field length");
    }

    const auto& center = values["C"];
    const auto& r = values["R"];
    PoseCamera camera;
    camera.fu = values["fu"].front();
    camera.fv = values["fv"].front();
    camera.cu = values["cu"].front();
    camera.cv = values["cv"].front();
    camera.center = {center[0], center[1], center[2]};
    // ASP TSAI 写出的 R 为 column-major；这里转成 row-major world-to-camera，与 Python 生成器一致。
    camera.rotation_world_to_camera = {r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]};
    camera.path = path;
    return camera;
}

PoseWarpResult projectPoseDepthWarp(const torch::Tensor& depth_a, const torch::Tensor& depth_b,
                                    const PoseCamera& camera_a, const PoseCamera& camera_b,
                                    double absolute_depth_tolerance_m, double relative_depth_tolerance)
{
    validateDepthTensor(depth_a, "depth_a");
    validateDepthTensor(depth_b, "depth_b");
    if (depth_a.sizes() != depth_b.sizes())
    {
        throw std::invalid_argument("pose depth shape mismatch");
    }
    if (absolute_depth_tolerance_m < 0.0 || relative_depth_tolerance < 0.0)
    {
        throw std::invalid_argument("pose depth tolerances must be nonnegative");
    }

    const auto depth_a_cpu = depth_a.contiguous();
    const auto depth_b_cpu = depth_b.contiguous();
    const int height = static_cast<int>(depth_a_cpu.size(0));
    const int width = static_cast<int>(depth_a_cpu.size(1));
    const auto depth_a_acc = depth_a_cpu.accessor<float, 2>();
    cv::Mat depth_b_mat(height, width, CV_32FC1, const_cast<float*>(depth_b_cpu.data_ptr<float>()));
    cv::Mat map_x(height, width, CV_32FC1);
    cv::Mat map_y(height, width, CV_32FC1);
    std::vector<float> warp_values(static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 2U, 0.0F);
    std::vector<double> projected_depth_values(static_cast<std::size_t>(height) * static_cast<std::size_t>(width),
                                               0.0);
    std::vector<uint8_t> valid_a_values(static_cast<std::size_t>(height) * static_cast<std::size_t>(width), 0U);
    std::vector<uint8_t> inside_values(static_cast<std::size_t>(height) * static_cast<std::size_t>(width), 0U);

    int64_t valid_a_count = 0;
    int64_t inside_count = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const auto flat =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
            const double z = static_cast<double>(depth_a_acc[y][x]);
            const bool valid_a = std::isfinite(z) && z > 0.0;
            valid_a_values[flat] = valid_a ? 1U : 0U;
            valid_a_count += valid_a ? 1 : 0;

            const double x_cam = (static_cast<double>(x) + 0.5 - camera_a.cu) / camera_a.fu * z;
            const double y_cam = (static_cast<double>(y) + 0.5 - camera_a.cv) / camera_a.fv * z;
            const auto world_offset = transposeMatVec(camera_a.rotation_world_to_camera, x_cam, y_cam, z);
            const double world_x = camera_a.center[0] + world_offset[0];
            const double world_y = camera_a.center[1] + world_offset[1];
            const double world_z = camera_a.center[2] + world_offset[2];
            const auto projected_b =
                matVec(camera_b.rotation_world_to_camera, world_x - camera_b.center[0], world_y - camera_b.center[1],
                       world_z - camera_b.center[2]);
            const double pb_z = projected_b[2];
            double u_b = 0.0;
            double v_b = 0.0;
            bool inside = false;
            if (std::isfinite(pb_z) && pb_z > 0.0)
            {
                u_b = camera_b.fu * (projected_b[0] / pb_z) + camera_b.cu - 0.5;
                v_b = camera_b.fv * (projected_b[1] / pb_z) + camera_b.cv - 0.5;
                inside = valid_a && u_b >= 0.0 && u_b <= static_cast<double>(width - 1) && v_b >= 0.0 &&
                         v_b <= static_cast<double>(height - 1);
            }
            if (!std::isfinite(u_b) || !std::isfinite(v_b))
            {
                u_b = 0.0;
                v_b = 0.0;
            }
            inside_values[flat] = inside ? 1U : 0U;
            inside_count += inside ? 1 : 0;
            projected_depth_values[flat] = pb_z;
            warp_values[flat * 2U] = static_cast<float>(u_b);
            warp_values[flat * 2U + 1U] = static_cast<float>(v_b);
            map_x.at<float>(y, x) = static_cast<float>(u_b);
            map_y.at<float>(y, x) = static_cast<float>(v_b);
        }
    }

    cv::Mat sampled_depth_b;
    cv::remap(depth_b_mat, sampled_depth_b, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, -1.0);

    std::vector<uint8_t> valid_values(static_cast<std::size_t>(height) * static_cast<std::size_t>(width), 0U);
    int64_t valid_count = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const auto flat =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
            if (valid_a_values[flat] == 0U || inside_values[flat] == 0U)
            {
                continue;
            }
            const double projected_z = projected_depth_values[flat];
            const double sampled_z = static_cast<double>(sampled_depth_b.at<float>(y, x));
            const double tolerance =
                std::max(absolute_depth_tolerance_m, relative_depth_tolerance * std::abs(projected_z));
            const bool valid =
                std::isfinite(sampled_z) && sampled_z > 0.0 && std::abs(sampled_z - projected_z) <= tolerance;
            valid_values[flat] = valid ? 1U : 0U;
            valid_count += valid ? 1 : 0;
        }
    }

    auto warp = torch::from_blob(warp_values.data(), {height, width, 2}, torch::kFloat32).clone().contiguous();
    auto valid_mask = torch::from_blob(valid_values.data(), {height, width}, torch::kUInt8).clone().to(torch::kBool);
    PoseWarpResult result;
    result.warp_a_to_b = warp;
    result.valid_mask = valid_mask.contiguous();
    result.valid_a_fraction =
        static_cast<double>(valid_a_count) / static_cast<double>(std::max<int64_t>(1, height * width));
    result.target_inside_fraction =
        static_cast<double>(inside_count) / static_cast<double>(std::max<int64_t>(1, valid_a_count));
    result.valid_pair_fraction =
        static_cast<double>(valid_count) / static_cast<double>(std::max<int64_t>(1, valid_a_count));
    result.valid_pixels = valid_count;
    return result;
}

TensorBatchCollator makeLazyPosePairCollator()
{
    return TensorBatchCollator({
        {"view_a", TensorLayout::Chw},
        {"view_b", TensorLayout::Chw},
        {"warp_a_to_b", TensorLayout::Hwc},
        {"valid_mask", TensorLayout::Hw},
    });
}

LazyPosePairDataset::LazyPosePairDataset(const LazyPosePairDatasetConfig& config)
    : _crop_size(config.crop_size),
      _absolute_depth_tolerance_m(config.absolute_depth_tolerance_m),
      _relative_depth_tolerance(config.relative_depth_tolerance)
{
    if (_crop_size < 0)
    {
        throw std::invalid_argument("lazy pose pair crop_size must be nonnegative");
    }
    const auto uint8_paths = readPoseUint8Manifest(config.uint8_manifest);
    const auto records = readPoseRenderManifest(config.render_manifest, uint8_paths);
    _specs = buildLazyPosePairSpecs(records, config);
    if (_specs.empty())
    {
        throw std::invalid_argument("lazy pose pair dataset has no usable pairs");
    }
}

size_t LazyPosePairDataset::size() const
{
    return _specs.size();
}

const LazyPosePairSpec& LazyPosePairDataset::spec(size_t index) const
{
    if (index >= _specs.size())
    {
        throw std::out_of_range("lazy pose pair dataset index out of range");
    }
    return _specs[index];
}

SyntheticPair LazyPosePairDataset::load(size_t index) const
{
    const auto& item = spec(index);
    auto view_a = loadViewTensor(item.reference_image_path);
    auto view_b = loadViewTensor(item.target_image_path);
    auto depth_a = readDepthTensor(item.reference_depth_path);
    auto depth_b = readDepthTensor(item.target_depth_path);
    const auto camera_a = parsePoseTsaiCamera(item.reference_tsai_path);
    const auto camera_b = parsePoseTsaiCamera(item.target_tsai_path);
    const auto warp = projectPoseDepthWarp(depth_a, depth_b, camera_a, camera_b, _absolute_depth_tolerance_m,
                                           _relative_depth_tolerance);
    if (view_a.size(1) != warp.valid_mask.size(0) || view_a.size(2) != warp.valid_mask.size(1) ||
        view_b.size(1) != warp.valid_mask.size(0) || view_b.size(2) != warp.valid_mask.size(1))
    {
        throw std::invalid_argument("lazy pose pair image/depth spatial size mismatch for pair " +
                                    std::to_string(item.pair_index));
    }
    return cropPairDeterministic(SyntheticPair{view_a, view_b, warp.warp_a_to_b, warp.valid_mask}, _crop_size);
}

TensorBatch LazyPosePairDataset::get(size_t index)
{
    return pairToTensorBatch(load(index));
}

} // namespace pfm
