#pragma once

#include <cstddef>
#include <vector>

namespace pfm::match_quality {

struct MatchPoint {
    double ax = 0.0;
    double ay = 0.0;
    double bx = 0.0;
    double by = 0.0;
};

struct ImageExtent {
    double width = 0.0;
    double height = 0.0;
};

struct SpatialQualityOptions {
    std::size_t low_count_threshold = 35;
    std::size_t grid_rows = 4;
    std::size_t grid_cols = 4;
    double narrow_bbox_area_threshold = 0.06;
    double narrow_grid_coverage_threshold = 0.25;
    double local_cluster_concentration_threshold = 0.55;
    double displacement_concentration_threshold = 0.70;
    double displacement_bin_fraction = 0.03;
};

struct SpatialMatchQuality {
    std::size_t match_count = 0;
    double normalized_spatial_coverage_a = 0.0;
    double normalized_spatial_coverage_b = 0.0;
    double grid_coverage_a = 0.0;
    double grid_coverage_b = 0.0;
    double local_cluster_concentration_a = 0.0;
    double local_cluster_concentration_b = 0.0;
    double displacement_concentration = 0.0;
    bool low_count_narrow_cluster = false;
};

SpatialMatchQuality evaluateSpatialMatchQuality(
    const std::vector<MatchPoint>& matches,
    ImageExtent image_a,
    ImageExtent image_b,
    const SpatialQualityOptions& options = {}
);

}  // namespace pfm::match_quality
