#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include "infer/feature_codec.h"
#include "infer/match_codec.h"

namespace {

struct Options {
    std::string image;
    std::string checkpoint;
    std::string output_dir = "rotation_sweep";
    std::string pfm_cli = "./build-pfm-cf/pfm_cli";
    std::string device = "cuda";
    int angle_step = 30;
    int max_keypoints = 1024;
    int min_keypoints = 1024;
    double min_keypoint_intensity = 0.05;
    double threshold_px = 3.0;
};

std::string shellQuote(const std::string& value) {
    std::string quoted = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

void requireValue(int index, int argc, const char* option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string(option) + " requires a value");
    }
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--image") {
            requireValue(index, argc, "--image");
            options.image = argv[++index];
        } else if (arg == "--checkpoint") {
            requireValue(index, argc, "--checkpoint");
            options.checkpoint = argv[++index];
        } else if (arg == "--output-dir") {
            requireValue(index, argc, "--output-dir");
            options.output_dir = argv[++index];
        } else if (arg == "--pfm-cli") {
            requireValue(index, argc, "--pfm-cli");
            options.pfm_cli = argv[++index];
        } else if (arg == "--device") {
            requireValue(index, argc, "--device");
            options.device = argv[++index];
        } else if (arg == "--angle-step") {
            requireValue(index, argc, "--angle-step");
            options.angle_step = std::stoi(argv[++index]);
        } else if (arg == "--max-keypoints") {
            requireValue(index, argc, "--max-keypoints");
            options.max_keypoints = std::stoi(argv[++index]);
        } else if (arg == "--min-keypoints") {
            requireValue(index, argc, "--min-keypoints");
            options.min_keypoints = std::stoi(argv[++index]);
        } else if (arg == "--min-keypoint-intensity") {
            requireValue(index, argc, "--min-keypoint-intensity");
            options.min_keypoint_intensity = std::stod(argv[++index]);
        } else if (arg == "--threshold-px") {
            requireValue(index, argc, "--threshold-px");
            options.threshold_px = std::stod(argv[++index]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: pfm_rotation_sweep_eval --image img/100.tif --checkpoint model.pt [options]\n"
                << "Options: --output-dir DIR --pfm-cli PATH --device cpu|cuda --angle-step N\n"
                << "         --max-keypoints N --min-keypoints N --min-keypoint-intensity F --threshold-px F\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.image.empty() || options.checkpoint.empty()) {
        throw std::invalid_argument("--image and --checkpoint are required");
    }
    if (options.angle_step <= 0 || 360 % options.angle_step != 0) {
        throw std::invalid_argument("--angle-step must be a positive divisor of 360");
    }
    return options;
}

void runCommand(const std::string& command) {
    const int code = std::system(command.c_str());
    if (code != 0) {
        throw std::runtime_error("command failed: " + command);
    }
}

std::filesystem::path anglePath(const std::filesystem::path& dir, int angle, const std::string& suffix) {
    std::ostringstream name;
    name << "rot_" << std::setw(3) << std::setfill('0') << angle << suffix;
    return dir / name.str();
}

void writeRotatedImage(const cv::Mat& image, const std::filesystem::path& path, int angle) {
    const cv::Point2f center((image.cols - 1) * 0.5F, (image.rows - 1) * 0.5F);
    const auto matrix = cv::getRotationMatrix2D(center, static_cast<double>(angle), 1.0);
    cv::Mat rotated;
    cv::warpAffine(image, rotated, matrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    if (!cv::imwrite(path.string(), rotated)) {
        throw std::runtime_error("failed to write rotated image: " + path.string());
    }
}

std::array<double, 2> rotatePoint(double x, double y, double width, double height, int angle) {
    constexpr double pi = 3.14159265358979323846;
    const double radians = static_cast<double>(angle) * pi / 180.0;
    const double alpha = std::cos(radians);
    const double beta = std::sin(radians);
    const double cx = (width - 1.0) * 0.5;
    const double cy = (height - 1.0) * 0.5;
    const double shifted_x = x - cx;
    const double shifted_y = y - cy;
    return {alpha * shifted_x + beta * shifted_y + cx, -beta * shifted_x + alpha * shifted_y + cy};
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double index = q * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(index));
    const auto hi = static_cast<size_t>(std::ceil(index));
    if (lo == hi) {
        return values[lo];
    }
    const double t = index - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

struct AngleResult {
    int angle = 0;
    int64_t matches = 0;
    int64_t in_bounds = 0;
    double pass_rate = 0.0;
    double mean_error = 0.0;
    double median_error = 0.0;
    double p90_error = 0.0;
};

struct DescriptorResult {
    int64_t mutual_matches = 0;
    double pass_rate = 0.0;
    double mean_error = 0.0;
    double mean_score = 0.0;
    int64_t finite_a = 0;
    int64_t finite_b = 0;
};

struct RepeatabilityResult {
    int64_t repeatable_keypoints = 0;
    double repeatability_rate = 0.0;
    double mean_descriptor_score = 0.0;
};

int64_t countFiniteRows(const torch::Tensor& descriptors) {
    if (!descriptors.defined() || descriptors.dim() != 2 || descriptors.size(0) == 0) {
        return 0;
    }
    return torch::isfinite(descriptors.to(torch::kCPU, torch::kFloat32)).all(1).sum().item<int64_t>();
}

double descriptorScore(const torch::Tensor& descriptors_a, int64_t index_a, const torch::Tensor& descriptors_b, int64_t index_b) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (int64_t channel = 0; channel < descriptors_a.size(1); ++channel) {
        const double value_a = descriptors_a.index({index_a, channel}).item<float>();
        const double value_b = descriptors_b.index({index_b, channel}).item<float>();
        if (!std::isfinite(value_a) || !std::isfinite(value_b)) {
            return 0.0;
        }
        dot += value_a * value_b;
        norm_a += value_a * value_a;
        norm_b += value_b * value_b;
    }
    if (norm_a <= 0.0 || norm_b <= 0.0) {
        return 0.0;
    }
    return dot / std::sqrt(norm_a * norm_b);
}

AngleResult evaluateAngle(
    int angle,
    const pfm::FeatureSet& source_features,
    const pfm::FeatureSet& rotated_features,
    const pfm::MatchSet& matches,
    double threshold_px
) {
    AngleResult result;
    result.angle = angle;
    result.matches = matches.sparse_matches.size(0);
    if (result.matches == 0) {
        return result;
    }

    const auto match_indices = matches.sparse_matches.to(torch::kCPU, torch::kInt64).contiguous();
    const auto keypoints_a = source_features.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto keypoints_b = rotated_features.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    std::vector<double> errors;
    errors.reserve(static_cast<size_t>(result.matches));

    for (int64_t row = 0; row < result.matches; ++row) {
        const auto index_a = match_indices.index({row, 0}).item<int64_t>();
        const auto index_b = match_indices.index({row, 1}).item<int64_t>();
        if (index_a < 0 || index_b < 0 || index_a >= keypoints_a.size(0) || index_b >= keypoints_b.size(0)) {
            continue;
        }
        const double ax = keypoints_a.index({index_a, 0}).item<float>();
        const double ay = keypoints_a.index({index_a, 1}).item<float>();
        const double bx = keypoints_b.index({index_b, 0}).item<float>();
        const double by = keypoints_b.index({index_b, 1}).item<float>();
        const auto expected = rotatePoint(
            ax,
            ay,
            static_cast<double>(rotated_features.feature_map_width),
            static_cast<double>(rotated_features.feature_map_height),
            angle);
        if (expected[0] >= 0.0 && expected[0] <= static_cast<double>(rotated_features.feature_map_width - 1) &&
            expected[1] >= 0.0 && expected[1] <= static_cast<double>(rotated_features.feature_map_height - 1)) {
            ++result.in_bounds;
        }
        const double dx = bx - expected[0];
        const double dy = by - expected[1];
        errors.push_back(std::sqrt(dx * dx + dy * dy));
    }

    if (!errors.empty()) {
        const auto passing = std::count_if(errors.begin(), errors.end(), [&](double error) {
            return error <= threshold_px;
        });
        result.pass_rate = static_cast<double>(passing) / static_cast<double>(errors.size());
        result.mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
        result.median_error = percentile(errors, 0.5);
        result.p90_error = percentile(errors, 0.9);
    }
    return result;
}

DescriptorResult evaluateDescriptorMutualAngle(
    int angle,
    const pfm::FeatureSet& source_features,
    const pfm::FeatureSet& rotated_features,
    double threshold_px
) {
    DescriptorResult result;
    result.finite_a = countFiniteRows(source_features.descriptors);
    result.finite_b = countFiniteRows(rotated_features.descriptors);
    if (!source_features.descriptors.defined() || !rotated_features.descriptors.defined() ||
        source_features.descriptors.dim() != 2 || rotated_features.descriptors.dim() != 2 ||
        source_features.descriptors.size(0) == 0 || rotated_features.descriptors.size(0) == 0) {
        return result;
    }

    const auto desc_a = torch::nn::functional::normalize(
        source_features.descriptors.to(torch::kCPU, torch::kFloat32),
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    const auto desc_b = torch::nn::functional::normalize(
        rotated_features.descriptors.to(torch::kCPU, torch::kFloat32),
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    if (!torch::isfinite(desc_a).all().item<bool>() || !torch::isfinite(desc_b).all().item<bool>()) {
        return result;
    }

    const auto scores = torch::matmul(desc_a, desc_b.transpose(0, 1));
    const auto best_ab = scores.max(1);
    const auto best_scores = std::get<0>(best_ab).contiguous();
    const auto target_indices = std::get<1>(best_ab).to(torch::kCPU, torch::kInt64).contiguous();
    const auto best_ba = std::get<1>(scores.max(0)).to(torch::kCPU, torch::kInt64).contiguous();
    const auto keypoints_a = source_features.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto keypoints_b = rotated_features.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();

    std::vector<double> errors;
    errors.reserve(static_cast<size_t>(source_features.descriptors.size(0)));
    double score_sum = 0.0;
    for (int64_t index_a = 0; index_a < target_indices.size(0); ++index_a) {
        const auto index_b = target_indices.index({index_a}).item<int64_t>();
        if (index_b < 0 || index_b >= best_ba.size(0) || best_ba.index({index_b}).item<int64_t>() != index_a) {
            continue;
        }
        const double ax = keypoints_a.index({index_a, 0}).item<float>();
        const double ay = keypoints_a.index({index_a, 1}).item<float>();
        const double bx = keypoints_b.index({index_b, 0}).item<float>();
        const double by = keypoints_b.index({index_b, 1}).item<float>();
        const auto expected = rotatePoint(
            ax,
            ay,
            static_cast<double>(rotated_features.feature_map_width),
            static_cast<double>(rotated_features.feature_map_height),
            angle);
        const double dx = bx - expected[0];
        const double dy = by - expected[1];
        errors.push_back(std::sqrt(dx * dx + dy * dy));
        score_sum += best_scores.index({index_a}).item<float>();
    }

    result.mutual_matches = static_cast<int64_t>(errors.size());
    if (!errors.empty()) {
        const auto passing = std::count_if(errors.begin(), errors.end(), [&](double error) {
            return error <= threshold_px;
        });
        result.pass_rate = static_cast<double>(passing) / static_cast<double>(errors.size());
        result.mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
        result.mean_score = score_sum / static_cast<double>(errors.size());
    }
    return result;
}

RepeatabilityResult evaluateRepeatability(
    int angle,
    const pfm::FeatureSet& source_features,
    const pfm::FeatureSet& rotated_features,
    double threshold_px
) {
    RepeatabilityResult result;
    if (!source_features.keypoints.defined() || !rotated_features.keypoints.defined() ||
        source_features.keypoints.size(0) == 0 || rotated_features.keypoints.size(0) == 0) {
        return result;
    }

    const auto keypoints_a = source_features.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto keypoints_b = rotated_features.keypoints.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto desc_a = source_features.descriptors.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto desc_b = rotated_features.descriptors.to(torch::kCPU, torch::kFloat32).contiguous();
    double descriptor_score_sum = 0.0;

    for (int64_t index_a = 0; index_a < keypoints_a.size(0); ++index_a) {
        const double ax = keypoints_a.index({index_a, 0}).item<float>();
        const double ay = keypoints_a.index({index_a, 1}).item<float>();
        const auto expected = rotatePoint(
            ax,
            ay,
            static_cast<double>(rotated_features.feature_map_width),
            static_cast<double>(rotated_features.feature_map_height),
            angle);
        double best_error = std::numeric_limits<double>::infinity();
        int64_t best_index_b = -1;
        for (int64_t index_b = 0; index_b < keypoints_b.size(0); ++index_b) {
            const double bx = keypoints_b.index({index_b, 0}).item<float>();
            const double by = keypoints_b.index({index_b, 1}).item<float>();
            const double dx = bx - expected[0];
            const double dy = by - expected[1];
            const double error = std::sqrt(dx * dx + dy * dy);
            if (error < best_error) {
                best_error = error;
                best_index_b = index_b;
            }
        }
        if (best_index_b >= 0 && best_error <= threshold_px) {
            ++result.repeatable_keypoints;
            descriptor_score_sum += descriptorScore(desc_a, index_a, desc_b, best_index_b);
        }
    }

    result.repeatability_rate =
        static_cast<double>(result.repeatable_keypoints) / static_cast<double>(keypoints_a.size(0));
    if (result.repeatable_keypoints > 0) {
        result.mean_descriptor_score = descriptor_score_sum / static_cast<double>(result.repeatable_keypoints);
    }
    return result;
}

std::string commonDecodeArgs(const Options& options) {
    std::ostringstream args;
    args << " --device " << shellQuote(options.device)
         << " --max-keypoints " << options.max_keypoints
         << " --min-keypoints " << options.min_keypoints
         << " --min-keypoint-intensity " << options.min_keypoint_intensity;
    return args.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parseOptions(argc, argv);
        const auto output_dir = std::filesystem::absolute(options.output_dir);
        std::filesystem::create_directories(output_dir);

        const auto source_image = std::filesystem::absolute(options.image);
        const auto checkpoint = std::filesystem::absolute(options.checkpoint);
        const auto source_features = output_dir / "source_features.pt";
        const auto summary_path = output_dir / "summary.csv";

        const auto image = cv::imread(source_image.string(), cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            throw std::runtime_error("failed to read image: " + source_image.string());
        }

        runCommand(
            shellQuote(options.pfm_cli) + " extract --image " + shellQuote(source_image.string()) + " --checkpoint " +
            shellQuote(checkpoint.string()) + " --output " + shellQuote(source_features.string()) + commonDecodeArgs(options));
        const auto source = pfm::load_feature_set(source_features.string());

        std::ofstream summary(summary_path);
        summary << "angle,matches,in_bounds,pass_rate,mean_error_px,median_error_px,p90_error_px,"
                << "desc_mutual_matches,desc_mutual_pass_rate,desc_mutual_mean_error_px,desc_mutual_mean_score,"
                << "desc_finite_a,desc_finite_b,repeatable_keypoints,repeatability_rate,repeatable_mean_desc_score\n";
        std::cout << "angle,matches,in_bounds,pass_rate,mean_error_px,median_error_px,p90_error_px,"
                  << "desc_mutual_matches,desc_mutual_pass_rate,desc_mutual_mean_error_px,desc_mutual_mean_score,"
                  << "desc_finite_a,desc_finite_b,repeatable_keypoints,repeatability_rate,repeatable_mean_desc_score\n";

        for (int angle = 0; angle < 360; angle += options.angle_step) {
            const auto rotated_image = anglePath(output_dir, angle, ".tif");
            const auto rotated_features = anglePath(output_dir, angle, "_features.pt");
            const auto rotated_matches = anglePath(output_dir, angle, "_matches.pt");
            const auto vis_dir = anglePath(output_dir, angle, "_vis");
            writeRotatedImage(image, rotated_image, angle);

            runCommand(
                shellQuote(options.pfm_cli) + " extract --image " + shellQuote(rotated_image.string()) +
                " --checkpoint " + shellQuote(checkpoint.string()) + " --output " +
                shellQuote(rotated_features.string()) + commonDecodeArgs(options));
            runCommand(
                shellQuote(options.pfm_cli) + " match --feature-a " + shellQuote(source_features.string()) +
                " --feature-b " + shellQuote(rotated_features.string()) + " --image-a " +
                shellQuote(source_image.string()) + " --image-b " + shellQuote(rotated_image.string()) +
                " --checkpoint " + shellQuote(checkpoint.string()) + " --output " +
                shellQuote(rotated_matches.string()) + " --visualization-dir " + shellQuote(vis_dir.string()) +
                commonDecodeArgs(options));

            const auto rotated = pfm::load_feature_set(rotated_features.string());
            const auto matches = pfm::load_match_set(rotated_matches.string());
            const auto result = evaluateAngle(angle, source, rotated, matches, options.threshold_px);
            const auto descriptor_result = evaluateDescriptorMutualAngle(angle, source, rotated, options.threshold_px);
            const auto repeatability_result = evaluateRepeatability(angle, source, rotated, options.threshold_px);
            summary << result.angle << ',' << result.matches << ',' << result.in_bounds << ',' << result.pass_rate
                    << ',' << result.mean_error << ',' << result.median_error << ',' << result.p90_error << ','
                    << descriptor_result.mutual_matches << ',' << descriptor_result.pass_rate << ','
                    << descriptor_result.mean_error << ',' << descriptor_result.mean_score << ','
                    << descriptor_result.finite_a << ',' << descriptor_result.finite_b << ','
                    << repeatability_result.repeatable_keypoints << ',' << repeatability_result.repeatability_rate << ','
                    << repeatability_result.mean_descriptor_score << '\n';
            std::cout << result.angle << ',' << result.matches << ',' << result.in_bounds << ',' << result.pass_rate
                      << ',' << result.mean_error << ',' << result.median_error << ',' << result.p90_error << ','
                      << descriptor_result.mutual_matches << ',' << descriptor_result.pass_rate << ','
                      << descriptor_result.mean_error << ',' << descriptor_result.mean_score << ','
                      << descriptor_result.finite_a << ',' << descriptor_result.finite_b << ','
                      << repeatability_result.repeatable_keypoints << ',' << repeatability_result.repeatability_rate << ','
                      << repeatability_result.mean_descriptor_score << '\n';
        }

        std::cout << "summary=" << summary_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rotation sweep failed: " << error.what() << '\n';
        return 1;
    }
}
