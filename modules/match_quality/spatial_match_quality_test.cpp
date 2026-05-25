#include "tests/test_harness.h"

#include <vector>

#include "match_quality/spatial_match_quality.h"

namespace {

std::vector<pfm::match_quality::MatchPoint> makeWideMatches() {
    std::vector<pfm::match_quality::MatchPoint> matches;
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 6; ++col) {
            const double ax = 80.0 + col * 160.0;
            const double ay = 90.0 + row * 170.0;
            matches.push_back({ax, ay, ax + 18.0, ay - 12.0});
        }
    }
    return matches;
}

std::vector<pfm::match_quality::MatchPoint> makeNarrowCluster() {
    std::vector<pfm::match_quality::MatchPoint> matches;
    for (int i = 0; i < 24; ++i) {
        const double ax = 420.0 + static_cast<double>(i % 6) * 3.0;
        const double ay = 510.0 + static_cast<double>(i / 6) * 3.0;
        matches.push_back({ax, ay, ax + 27.0, ay - 11.0});
    }
    return matches;
}

void emptySetReportsZerosAndNotSuspicious() {
    const auto quality = pfm::match_quality::evaluateSpatialMatchQuality({}, {1000.0, 900.0}, {1000.0, 900.0});

    PFM_REQUIRE(quality.match_count == 0);
    PFM_REQUIRE_CLOSE(quality.normalized_spatial_coverage_a, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.normalized_spatial_coverage_b, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.grid_coverage_a, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.grid_coverage_b, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.displacement_concentration, 0.0, 1.0e-12);
    PFM_REQUIRE(!quality.low_count_narrow_cluster);
}

void wideCoverageMatchesAreNotNarrowClusters() {
    pfm::match_quality::SpatialQualityOptions options;
    options.low_count_threshold = 35;

    const auto quality = pfm::match_quality::evaluateSpatialMatchQuality(
        makeWideMatches(),
        {1000.0, 1000.0},
        {1000.0, 1000.0},
        options
    );

    PFM_REQUIRE(quality.match_count == 30);
    PFM_REQUIRE(quality.normalized_spatial_coverage_a > 0.45);
    PFM_REQUIRE(quality.grid_coverage_a > 0.50);
    PFM_REQUIRE(quality.grid_coverage_b > 0.50);
    PFM_REQUIRE(!quality.low_count_narrow_cluster);
}

void lowCountConcentratedClusterIsSuspicious() {
    const auto quality = pfm::match_quality::evaluateSpatialMatchQuality(
        makeNarrowCluster(),
        {1000.0, 1000.0},
        {1000.0, 1000.0}
    );

    PFM_REQUIRE(quality.match_count == 24);
    PFM_REQUIRE(quality.normalized_spatial_coverage_a < 0.001);
    PFM_REQUIRE(quality.normalized_spatial_coverage_b < 0.001);
    PFM_REQUIRE(quality.grid_coverage_a <= 0.25);
    PFM_REQUIRE(quality.local_cluster_concentration_a >= 0.50);
    PFM_REQUIRE(quality.displacement_concentration >= 0.70);
    PFM_REQUIRE(quality.low_count_narrow_cluster);
}

void repeatedPointsRemainStableAndClustered() {
    const std::vector<pfm::match_quality::MatchPoint> matches(12, {120.0, 140.0, 130.0, 150.0});

    const auto quality = pfm::match_quality::evaluateSpatialMatchQuality(
        matches,
        {800.0, 600.0},
        {800.0, 600.0}
    );

    PFM_REQUIRE(quality.match_count == 12);
    PFM_REQUIRE_CLOSE(quality.normalized_spatial_coverage_a, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.grid_coverage_a, 1.0 / 16.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.local_cluster_concentration_a, 1.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.displacement_concentration, 1.0, 1.0e-12);
    PFM_REQUIRE(quality.low_count_narrow_cluster);
}

void invalidImageSizesAreStableAndNotClassified() {
    const auto quality = pfm::match_quality::evaluateSpatialMatchQuality(
        makeNarrowCluster(),
        {0.0, 1000.0},
        {1000.0, -1.0}
    );

    PFM_REQUIRE(quality.match_count == 24);
    PFM_REQUIRE_CLOSE(quality.normalized_spatial_coverage_a, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.normalized_spatial_coverage_b, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.grid_coverage_a, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.grid_coverage_b, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(quality.displacement_concentration, 0.0, 1.0e-12);
    PFM_REQUIRE(!quality.low_count_narrow_cluster);
}

}  // namespace

void register_spatial_match_quality_tests() {
    register_test("spatial match quality empty set reports zeros and not suspicious", emptySetReportsZerosAndNotSuspicious);
    register_test("spatial match quality wide coverage matches are not narrow clusters",
                  wideCoverageMatchesAreNotNarrowClusters);
    register_test("spatial match quality low count concentrated cluster is suspicious",
                  lowCountConcentratedClusterIsSuspicious);
    register_test("spatial match quality repeated points remain stable and clustered", repeatedPointsRemainStableAndClustered);
    register_test("spatial match quality invalid image sizes are stable and not classified",
                  invalidImageSizesAreStableAndNotClassified);
}
