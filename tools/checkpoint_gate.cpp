#include "checkpoint_gate/checkpoint_gate.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct GateCase
{
    std::string name;
    std::string image_a;
    std::string image_b;
    std::string warp;
    pfm::CheckpointGateThreshold threshold;
};

struct Options
{
    std::string checkpoint;
    std::string pfm_cli = "./pfm_cli";
    std::string output_dir = "checkpoint_gate";
    std::string device = "cuda";
    int max_keypoints = 2048;
    int min_keypoints = 0;
    int keypoint_grid_rows = 12;
    int keypoint_grid_cols = 12;
    int nms_radius = 2;
    int descriptor_pool_radius = 0;
    bool disable_descriptor_orientation_canonicalization = false;
    double min_keypoint_intensity = 0.08;
    double match_threshold_pixels = 5.0;
    std::vector<GateCase> cases;
};

std::string shellQuote(const std::string& value)
{
    std::string quoted = "'";
    for (const char ch : value)
    {
        if (ch == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string requireValue(int& index, int argc, char** argv, const char* option)
{
    if (index + 1 >= argc)
    {
        throw std::invalid_argument(std::string(option) + " requires a value");
    }
    return argv[++index];
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        if (arg == "--checkpoint")
        {
            options.checkpoint = requireValue(index, argc, argv, "--checkpoint");
        }
        else if (arg == "--pfm-cli")
        {
            options.pfm_cli = requireValue(index, argc, argv, "--pfm-cli");
        }
        else if (arg == "--output-dir")
        {
            options.output_dir = requireValue(index, argc, argv, "--output-dir");
        }
        else if (arg == "--device")
        {
            options.device = requireValue(index, argc, argv, "--device");
        }
        else if (arg == "--max-keypoints")
        {
            options.max_keypoints = std::stoi(requireValue(index, argc, argv, "--max-keypoints"));
        }
        else if (arg == "--min-keypoints")
        {
            options.min_keypoints = std::stoi(requireValue(index, argc, argv, "--min-keypoints"));
        }
        else if (arg == "--keypoint-grid-rows")
        {
            options.keypoint_grid_rows = std::stoi(requireValue(index, argc, argv, "--keypoint-grid-rows"));
        }
        else if (arg == "--keypoint-grid-cols")
        {
            options.keypoint_grid_cols = std::stoi(requireValue(index, argc, argv, "--keypoint-grid-cols"));
        }
        else if (arg == "--nms-radius")
        {
            options.nms_radius = std::stoi(requireValue(index, argc, argv, "--nms-radius"));
        }
        else if (arg == "--descriptor-pool-radius")
        {
            options.descriptor_pool_radius = std::stoi(requireValue(index, argc, argv, "--descriptor-pool-radius"));
        }
        else if (arg == "--disable-descriptor-orientation-canonicalization")
        {
            options.disable_descriptor_orientation_canonicalization = true;
        }
        else if (arg == "--min-keypoint-intensity")
        {
            options.min_keypoint_intensity = std::stod(requireValue(index, argc, argv, "--min-keypoint-intensity"));
        }
        else if (arg == "--match-threshold-pixels")
        {
            options.match_threshold_pixels = std::stod(requireValue(index, argc, argv, "--match-threshold-pixels"));
        }
        else if (arg == "--case")
        {
            if (index + 6 >= argc)
            {
                throw std::invalid_argument("--case requires: name image_a image_b warp min_correct min_precision");
            }
            GateCase test_case;
            test_case.name = argv[++index];
            test_case.image_a = argv[++index];
            test_case.image_b = argv[++index];
            test_case.warp = argv[++index];
            test_case.threshold.min_correct_matches = std::stoll(argv[++index]);
            test_case.threshold.min_precision = std::stod(argv[++index]);
            options.cases.push_back(std::move(test_case));
        }
        else if (arg == "-h" || arg == "--help")
        {
            std::cout << "Usage: pfm_checkpoint_gate --checkpoint model.pt --pfm-cli ./pfm_cli --case "
                         "name image_a image_b warp min_correct min_precision [--case ...]\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.checkpoint.empty())
    {
        throw std::invalid_argument("--checkpoint is required");
    }
    if (options.cases.empty())
    {
        throw std::invalid_argument("at least one --case is required");
    }
    return options;
}

std::string runCommandCapture(const std::string& command)
{
    std::array<char, 4096> buffer{};
    std::string output;
    FILE* pipe = popen((command + " 2>&1").c_str(), "r");
    if (pipe == nullptr)
    {
        throw std::runtime_error("failed to start command");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
    {
        output += buffer.data();
    }
    const int code = pclose(pipe);
    if (code != 0)
    {
        throw std::runtime_error("command failed: " + command + "\n" + output);
    }
    return output;
}

std::string buildMatchCommand(const Options& options, const GateCase& test_case)
{
    const auto case_dir = std::filesystem::path(options.output_dir) / test_case.name;
    std::filesystem::create_directories(case_dir);
    const auto output_path = case_dir / "matches.pt";
    std::ostringstream command;
    command << shellQuote(options.pfm_cli) << " match --image-a " << shellQuote(test_case.image_a) << " --image-b "
            << shellQuote(test_case.image_b) << " --checkpoint " << shellQuote(options.checkpoint) << " --device "
            << shellQuote(options.device) << " --output " << shellQuote(output_path.string()) << " --visualization-dir "
            << shellQuote(case_dir.string()) << " --warp-a-to-b " << shellQuote(test_case.warp)
            << " --match-correct-threshold-pixels " << options.match_threshold_pixels << " --match-mode sparse"
            << " --min-keypoint-intensity " << options.min_keypoint_intensity << " --max-keypoints "
            << options.max_keypoints << " --min-keypoints " << options.min_keypoints << " --keypoint-grid-rows "
            << options.keypoint_grid_rows << " --keypoint-grid-cols " << options.keypoint_grid_cols << " --nms-radius "
            << options.nms_radius << " --descriptor-pool-radius " << options.descriptor_pool_radius;
    if (options.disable_descriptor_orientation_canonicalization)
    {
        command << " --disable-descriptor-orientation-canonicalization";
    }
    return command.str();
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parseOptions(argc, argv);
        int failures = 0;
        for (const auto& test_case : options.cases)
        {
            const auto output = runCommandCapture(buildMatchCommand(options, test_case));
            const auto metrics = pfm::parse_checkpoint_gate_metrics(output);
            const auto decision = pfm::evaluate_checkpoint_gate_metrics(metrics, test_case.threshold);
            std::cout << test_case.name << " correct_matches=" << metrics.correct_matches
                      << " wrong_matches=" << metrics.wrong_matches << " match_precision=" << metrics.precision
                      << " required_correct=" << test_case.threshold.min_correct_matches
                      << " required_precision=" << test_case.threshold.min_precision
                      << " status=" << (decision.passed ? "PASS" : "FAIL");
            if (!decision.passed)
            {
                std::cout << " reason=\"" << decision.reason << '"';
                ++failures;
            }
            std::cout << '\n';
        }
        if (failures != 0)
        {
            std::cerr << failures << " checkpoint gate case(s) failed\n";
            return 1;
        }
        std::cout << "checkpoint gate passed: cases=" << options.cases.size() << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "checkpoint gate failed: " << error.what() << '\n';
        return 1;
    }
}
