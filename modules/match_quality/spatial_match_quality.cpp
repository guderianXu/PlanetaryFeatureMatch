#include "match_quality/spatial_match_quality.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

namespace pfm::match_quality
{
namespace
{

bool validExtent(ImageExtent extent)
{
    return std::isfinite(extent.width) && std::isfinite(extent.height) && extent.width > 0.0 && extent.height > 0.0;
}

double clampUnit(double value)
{
    if (!std::isfinite(value))
    {
        return 0.0;
    }
    return std::max(0.0, std::min(1.0, value));
}

double normalizedBoundingBoxArea(const std::vector<MatchPoint>& matches, ImageExtent extent, bool use_a)
{
    if (matches.empty() || !validExtent(extent))
    {
        return 0.0;
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const auto& match : matches)
    {
        const double x = use_a ? match.ax : match.bx;
        const double y = use_a ? match.ay : match.by;
        if (!std::isfinite(x) || !std::isfinite(y))
        {
            continue;
        }
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
    }

    if (!std::isfinite(min_x) || max_x <= min_x || max_y <= min_y)
    {
        return 0.0;
    }
    return clampUnit(((max_x - min_x) * (max_y - min_y)) / (extent.width * extent.height));
}

struct GridStats
{
    double coverage = 0.0;
    double max_cell_fraction = 0.0;
};

GridStats gridStats(const std::vector<MatchPoint>& matches, ImageExtent extent, std::size_t rows, std::size_t cols,
                    bool use_a)
{
    if (matches.empty() || !validExtent(extent) || rows == 0 || cols == 0)
    {
        return {};
    }

    std::vector<std::size_t> counts(rows * cols, 0);
    std::size_t counted = 0;
    for (const auto& match : matches)
    {
        const double x = use_a ? match.ax : match.bx;
        const double y = use_a ? match.ay : match.by;
        if (!std::isfinite(x) || !std::isfinite(y))
        {
            continue;
        }
        const double clamped_x = std::max(0.0, std::min(std::nextafter(extent.width, 0.0), x));
        const double clamped_y = std::max(0.0, std::min(std::nextafter(extent.height, 0.0), y));
        const std::size_t col = std::min(cols - 1, static_cast<std::size_t>((clamped_x / extent.width) * cols));
        const std::size_t row = std::min(rows - 1, static_cast<std::size_t>((clamped_y / extent.height) * rows));
        ++counts[row * cols + col];
        ++counted;
    }

    if (counted == 0)
    {
        return {};
    }

    std::size_t occupied = 0;
    std::size_t max_count = 0;
    for (const auto count : counts)
    {
        if (count > 0)
        {
            ++occupied;
        }
        max_count = std::max(max_count, count);
    }

    GridStats stats;
    stats.coverage = static_cast<double>(occupied) / static_cast<double>(rows * cols);
    stats.max_cell_fraction = static_cast<double>(max_count) / static_cast<double>(counted);
    return stats;
}

double displacementConcentration(const std::vector<MatchPoint>& matches, ImageExtent image_a, ImageExtent image_b,
                                 double bin_fraction)
{
    if (matches.empty() || !validExtent(image_a) || !validExtent(image_b))
    {
        return 0.0;
    }

    const double span = std::max({image_a.width, image_a.height, image_b.width, image_b.height});
    const double bin_size = std::max(1.0, span * std::max(0.0, bin_fraction));
    struct BinKey
    {
        long long x = 0;
        long long y = 0;

        bool operator==(const BinKey& other) const
        {
            return x == other.x && y == other.y;
        }
    };
    struct BinKeyHash
    {
        std::size_t operator()(const BinKey& key) const
        {
            const auto hx = std::hash<long long>{}(key.x);
            const auto hy = std::hash<long long>{}(key.y);
            return hx ^ (hy + 0x9e3779b97f4a7c15ULL + (hx << 6U) + (hx >> 2U));
        }
    };

    std::unordered_map<BinKey, std::size_t, BinKeyHash> bins;
    std::size_t counted = 0;

    for (const auto& match : matches)
    {
        const double dx = match.bx - match.ax;
        const double dy = match.by - match.ay;
        if (!std::isfinite(dx) || !std::isfinite(dy))
        {
            continue;
        }
        const BinKey key{static_cast<long long>(std::floor(dx / bin_size)),
                         static_cast<long long>(std::floor(dy / bin_size))};
        ++bins[key];
        ++counted;
    }

    if (counted == 0)
    {
        return 0.0;
    }

    std::size_t max_count = 0;
    for (const auto& entry : bins)
    {
        max_count = std::max(max_count, entry.second);
    }
    return static_cast<double>(max_count) / static_cast<double>(counted);
}

} // namespace

SpatialMatchQuality evaluateSpatialMatchQuality(const std::vector<MatchPoint>& matches, ImageExtent image_a,
                                                ImageExtent image_b, const SpatialQualityOptions& options)
{
    SpatialMatchQuality quality;
    quality.match_count = matches.size();

    quality.normalized_spatial_coverage_a = normalizedBoundingBoxArea(matches, image_a, true);
    quality.normalized_spatial_coverage_b = normalizedBoundingBoxArea(matches, image_b, false);

    const auto grid_a = gridStats(matches, image_a, options.grid_rows, options.grid_cols, true);
    const auto grid_b = gridStats(matches, image_b, options.grid_rows, options.grid_cols, false);
    quality.grid_coverage_a = grid_a.coverage;
    quality.grid_coverage_b = grid_b.coverage;
    quality.local_cluster_concentration_a = grid_a.max_cell_fraction;
    quality.local_cluster_concentration_b = grid_b.max_cell_fraction;
    quality.displacement_concentration =
        displacementConcentration(matches, image_a, image_b, options.displacement_bin_fraction);

    const bool can_assess = validExtent(image_a) && validExtent(image_b) && !matches.empty();
    const bool low_count = quality.match_count <= options.low_count_threshold;
    const bool narrow_bbox = quality.normalized_spatial_coverage_a <= options.narrow_bbox_area_threshold &&
                             quality.normalized_spatial_coverage_b <= options.narrow_bbox_area_threshold;
    const bool narrow_grid = quality.grid_coverage_a <= options.narrow_grid_coverage_threshold &&
                             quality.grid_coverage_b <= options.narrow_grid_coverage_threshold;
    const bool locally_clustered =
        quality.local_cluster_concentration_a >= options.local_cluster_concentration_threshold ||
        quality.local_cluster_concentration_b >= options.local_cluster_concentration_threshold;
    const bool coherent_displacement =
        quality.displacement_concentration >= options.displacement_concentration_threshold;

    quality.low_count_narrow_cluster =
        can_assess && low_count && narrow_bbox && narrow_grid && (locally_clustered || coherent_displacement);
    return quality;
}

} // namespace pfm::match_quality
