#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <unistd.h>

#include "data/synthetic_pair_cache.h"
#include "dataloader/sampler.h"
#include "feature_io/feature_codec.h"
#include "feature_io/match_codec.h"
#include "models/head_outputs.h"
#include "models/planetary_graph_matcher.h"
#include "models/pfm_model_v21.h"
#include "tests/test_harness.h"
#include "train/trainer.h"

namespace pfm::testing
{

torch::Tensor resize_offsets_for_dense_head_for_test(const torch::Tensor& warp, const torch::Tensor& offsets);
torch::Tensor make_sparse_descriptor_loss_for_test(const torch::Tensor& descriptors_a,
                                                   const torch::Tensor& descriptors_b, const torch::Tensor& warp,
                                                   const torch::Tensor& valid_mask);
torch::Tensor make_dense_descriptor_hard_negative_loss_for_test(const torch::Tensor& descriptors_a,
                                                                const torch::Tensor& descriptors_b,
                                                                const torch::Tensor& warp,
                                                                const torch::Tensor& valid_mask);
torch::Tensor make_bidirectional_dense_descriptor_hard_negative_loss_for_test(const torch::Tensor& descriptors_a,
                                                                              const torch::Tensor& descriptors_b,
                                                                              const torch::Tensor& warp,
                                                                              const torch::Tensor& valid_mask);
torch::Tensor make_warp_descriptor_contrastive_loss_for_test(const torch::Tensor& descriptors_a,
                                                             const torch::Tensor& descriptors_b,
                                                             const torch::Tensor& warp,
                                                             const torch::Tensor& valid_mask);
torch::Tensor make_direct_full_map_descriptor_loss_for_test(const torch::Tensor& descriptors_a,
                                                            const torch::Tensor& descriptors_b,
                                                            const torch::Tensor& warp, const torch::Tensor& valid_mask);
torch::Tensor make_descriptor_map_regularization_loss_for_test(const torch::Tensor& descriptors);
torch::Tensor make_descriptor_target_coordinates_for_test(const torch::Tensor& warp,
                                                          const torch::Tensor& sample_indices,
                                                          int64_t descriptor_height, int64_t descriptor_width);
torch::Tensor sample_warped_descriptors_for_test(const torch::Tensor& descriptors,
                                                 const torch::Tensor& target_coordinates);
torch::Tensor make_graph_matching_loss_for_test(PlanetaryGraphMatcherImpl& graph_matcher,
                                                const torch::Tensor& descriptors_a, const torch::Tensor& descriptors_b,
                                                const torch::Tensor& warp, const torch::Tensor& valid_mask);
torch::Tensor assign_graph_matching_targets_for_test(const torch::Tensor& keypoints_a, const torch::Tensor& keypoints_b,
                                                     const torch::Tensor& warp, const torch::Tensor& valid_mask,
                                                     double positive_radius_pixels);
torch::Tensor make_graph_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t keypoint_count_b,
                                                    int64_t max_candidates);
torch::Tensor make_graph_training_query_indices_for_test(const torch::Tensor& target_indices, int64_t keypoint_count_b,
                                                         int64_t max_queries);
torch::Tensor make_keypoint_graph_matching_loss_for_test(PlanetaryGraphMatcherImpl& graph_matcher,
                                                         const FeatureSet& features_a, const FeatureSet& features_b,
                                                         const torch::Tensor& warp, const torch::Tensor& valid_mask);
torch::Tensor make_keypoint_descriptor_loss_for_test(const FeatureSet& features_a, const FeatureSet& features_b,
                                                     const torch::Tensor& warp, const torch::Tensor& valid_mask);
torch::Tensor make_keypoint_descriptor_metric_tensor_for_test(const FeatureSet& features_a,
                                                              const FeatureSet& features_b, const torch::Tensor& warp,
                                                              const torch::Tensor& valid_mask);
torch::Tensor make_keypoint_dense_descriptor_loss_for_test(const FeatureSet& features_a,
                                                           const torch::Tensor& descriptors_b,
                                                           const torch::Tensor& warp, const torch::Tensor& valid_mask);
torch::Tensor make_keypoint_patch_descriptor_alignment_loss_for_test(const FeatureSet& features_a,
                                                                     const torch::Tensor& descriptors_b,
                                                                     const torch::Tensor& warp,
                                                                     const torch::Tensor& valid_mask);
torch::Tensor make_warped_keypoint_descriptor_contrastive_loss_for_test(const FeatureSet& features_a,
                                                                        const torch::Tensor& descriptors_b,
                                                                        const torch::Tensor& warp,
                                                                        const torch::Tensor& valid_mask);
torch::Tensor make_decoded_keypoint_repeatability_loss_for_test(const FeatureSet& features_a,
                                                                const torch::Tensor& heatmap_b,
                                                                const torch::Tensor& warp,
                                                                const torch::Tensor& valid_mask);
std::pair<FeatureSet, FeatureSet> make_warp_completed_keypoint_feature_pair_for_test(const FeatureSet& features_a,
                                                                                     const torch::Tensor& descriptors_b,
                                                                                     const torch::Tensor& warp,
                                                                                     const torch::Tensor& valid_mask);
torch::Tensor scale_feature_keypoints_to_image_for_test(const torch::Tensor& keypoints, int64_t feature_width,
                                                        int64_t feature_height, int64_t image_width,
                                                        int64_t image_height);
torch::Tensor make_orientation_supervision_loss_for_test(const SparseHeadOutput& sparse_a,
                                                         const SparseHeadOutput& sparse_b, const torch::Tensor& view_a,
                                                         const torch::Tensor& view_b, const torch::Tensor& warp,
                                                         double min_keypoint_intensity);
torch::Tensor make_descriptor_sample_indices_for_test(const torch::Tensor& descriptors);
torch::Tensor make_descriptor_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t spatial_count);
torch::Tensor make_descriptor_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t spatial_count,
                                                         int64_t broad_far_negative_count);
torch::Tensor make_descriptor_candidate_indices_for_test(const torch::Tensor& target_indices, int64_t spatial_count,
                                                         const torch::Tensor& candidate_valid_mask);
torch::Tensor make_supervised_descriptor_ranking_loss_for_test(const torch::Tensor& sampled_a,
                                                               const torch::Tensor& candidate_b);
torch::Tensor make_sampled_descriptor_decorrelation_loss_for_test(const torch::Tensor& sampled_descriptors,
                                                                  const torch::Tensor& sample_indices,
                                                                  int64_t descriptor_width);
torch::Tensor make_positive_descriptor_alignment_loss_for_test(const torch::Tensor& sampled_a,
                                                               const torch::Tensor& positive_b);
torch::Tensor make_patch_descriptor_alignment_loss_for_test(const torch::Tensor& descriptors_a,
                                                            const torch::Tensor& descriptors_b,
                                                            const torch::Tensor& warp, const torch::Tensor& valid_mask);
torch::Tensor descriptor_candidate_similarity_scores_for_test(const torch::Tensor& descriptors_a,
                                                              const torch::Tensor& candidate_descriptors);
torch::Tensor make_strict_descriptor_cross_entropy_loss_for_test(const torch::Tensor& descriptors_a,
                                                                 const torch::Tensor& descriptors_b,
                                                                 const torch::Tensor& target_indices);
torch::Tensor blend_rotation_invariant_texture_descriptor_for_test(const torch::Tensor& descriptors,
                                                                   const torch::Tensor& image);
torch::Tensor canonicalize_descriptor_map_by_orientation_for_test(const torch::Tensor& descriptors,
                                                                  const torch::Tensor& orientation);
torch::Tensor make_descriptor_finetune_anchor_loss_for_test(const torch::Tensor& current_a,
                                                            const torch::Tensor& current_b,
                                                            const torch::Tensor& anchor_a,
                                                            const torch::Tensor& anchor_b,
                                                            const torch::Tensor& valid_mask);
double descriptor_texture_teacher_weight_for_test();
double descriptor_texture_target_weight_for_test();
double descriptor_texture_blend_weight_for_test();
double descriptor_finetune_anchor_weight_for_test();
int64_t descriptor_negative_sample_count_for_test();
double descriptor_global_ce_weight_for_test();
int64_t supervised_descriptor_topk_negatives_for_test();
double supervised_descriptor_soft_rank_weight_for_test();
double supervised_descriptor_tail_rank_weight_for_test();
double learned_keypoint_graph_loss_weight_for_test();
double warp_completed_keypoint_graph_loss_weight_for_test();
double supervised_keypoint_graph_loss_weight_for_test();
int64_t descriptor_broad_far_negative_count_for_progress_for_test(double progress);
int64_t training_variant_index_for_pair_for_test(std::size_t pair_index, std::size_t train_image_count, int epoch,
                                                 int pairs_per_image);
torch::Tensor limit_training_image_size_for_test(const torch::Tensor& image, int64_t max_edge);
SyntheticPair limit_training_pair_size_for_test(const SyntheticPair& pair, int64_t max_edge);
SyntheticPair crop_training_pair_with_seed_for_test(const SyntheticPair& pair, int64_t crop_size, uint64_t seed);
torch::Tensor stack_chw_batch_for_test(const std::vector<torch::Tensor>& tensors);
torch::Tensor stack_hw_batch_for_test(const std::vector<torch::Tensor>& tensors);
torch::Tensor stack_hwc_batch_for_test(const std::vector<torch::Tensor>& tensors);
torch::Tensor make_cache_training_sample_indices_for_test(std::size_t count, const TrainConfig& config);
std::vector<std::string> make_training_cache_dirs_for_test(const TrainConfig& config);
std::vector<std::string> make_training_cache_entries_for_test(const TrainConfig& config);
torch::Tensor weighted_total_training_loss_for_test(const torch::Tensor& repeatability, const torch::Tensor& descriptor,
                                                    const torch::Tensor& offset, const torch::Tensor& confidence,
                                                    const torch::Tensor& descriptor_diversity = torch::tensor(0.0F));
torch::Tensor warp_heatmap_for_repeatability_for_test(const torch::Tensor& heatmap, const torch::Tensor& warp);
torch::Tensor make_heatmap_correspondence_target_loss_for_test(const torch::Tensor& heatmap_a,
                                                               const torch::Tensor& heatmap_b_at_a,
                                                               const torch::Tensor& target, const torch::Tensor& mask);
torch::Tensor make_heatmap_positive_target_loss_for_test(const torch::Tensor& heatmap, const torch::Tensor& target,
                                                         const torch::Tensor& mask);
torch::Tensor make_training_valid_mask_for_test(const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                const torch::Tensor& warp, const torch::Tensor& valid_mask,
                                                double min_keypoint_intensity);
torch::Tensor make_pair_loss_valid_mask_for_test(const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                 const torch::Tensor& warp, const torch::Tensor& valid_mask,
                                                 double min_keypoint_intensity,
                                                 const std::string& training_profile);
torch::Tensor make_warp_aligned_keypoint_targets_for_test(const torch::Tensor& view_a, const torch::Tensor& view_b,
                                                          const torch::Tensor& warp, const torch::Tensor& mask,
                                                          int64_t target_height, int64_t target_width);
FeatureSet decode_training_features_fast_for_test(const torch::Tensor& view, const SparseHeadOutput& sparse,
                                                  const TrainConfig& config);
torch::Tensor training_warp_overlay_image_for_test(const SyntheticPair& pair);
torch::Tensor training_feature_overlay_image_for_test(const torch::Tensor& image, const FeatureSet& features,
                                                      double min_keypoint_intensity);
torch::Tensor training_match_overlay_image_for_test(const torch::Tensor& image_a, const torch::Tensor& image_b,
                                                    const FeatureSet& features_a, const FeatureSet& features_b,
                                                    const MatchSet& matches);
torch::Tensor training_match_overlay_image_for_test(const torch::Tensor& image_a, const torch::Tensor& image_b,
                                                    const FeatureSet& features_a, const FeatureSet& features_b,
                                                    const MatchSet& matches, const torch::Tensor& warp_a_to_b,
                                                    double correct_threshold_pixels);
std::string training_model_match_overlay_text_for_test(const FeatureSet& features_a, const FeatureSet& features_b,
                                                       const MatchSet& matches, const torch::Tensor& warp_a_to_b,
                                                       double correct_threshold_pixels);
bool should_enqueue_training_visualization_for_test(std::size_t enqueued_count, std::size_t visualization_limit);
bool should_use_online_dataloader_for_test(const pfm::TrainConfig& config);
std::string effective_augmentation_profile_for_epoch_for_test(const pfm::TrainConfig& config, int epoch);
std::vector<std::size_t> make_training_image_indices_for_test(std::size_t total_images, const pfm::TrainConfig& config);
std::vector<std::size_t> make_validation_image_indices_for_test(std::size_t total_images,
                                                                const pfm::TrainConfig& config);
double training_learning_rate_for_step_for_test(const pfm::TrainConfig& config, int64_t step, int64_t total_steps);
bool training_profile_uses_dense_quality_forward_for_test(const std::string& training_profile);
std::vector<std::string> trainable_parameter_names_for_config_for_test(const pfm::TrainConfig& config);
torch::Tensor make_python_compare_graph_loss_for_test(pfm::v21::PfmV21GraphMatcherImpl& graph_matcher,
                                                      const torch::Tensor& desc_a, const torch::Tensor& desc_b,
                                                      const torch::Tensor& points_a, const torch::Tensor& points_b,
                                                      int64_t meta_dim);
torch::Tensor make_python_compare_graph_loss_for_test(pfm::v21::PfmV21GraphMatcherImpl& graph_matcher,
                                                      const torch::Tensor& desc_a, const torch::Tensor& desc_b,
                                                      const torch::Tensor& points_a, const torch::Tensor& points_b,
                                                      int64_t meta_dim, double accept_weight);
torch::Tensor make_python_compare_graph_loss_for_test(pfm::v21::PfmV21GraphMatcherImpl& graph_matcher,
                                                      const torch::Tensor& desc_a, const torch::Tensor& desc_b,
                                                      const torch::Tensor& points_a, const torch::Tensor& points_b,
                                                      int64_t meta_dim, double accept_weight,
                                                      double prune_ranking_weight, double prune_ranking_margin);
torch::Tensor make_python_compare_graph_loss_for_test(pfm::v21::PfmV21GraphMatcherImpl& graph_matcher,
                                                      const torch::Tensor& desc_a, const torch::Tensor& desc_b,
                                                      const torch::Tensor& points_a, const torch::Tensor& points_b,
                                                      int64_t meta_dim, double accept_weight,
                                                      double prune_ranking_weight, double prune_ranking_margin,
                                                      double stop_confidence_weight);
torch::Tensor make_python_compare_graph_loss_with_raw_preservation_for_test(
    pfm::v21::PfmV21GraphMatcherImpl& graph_matcher, const torch::Tensor& desc_a, const torch::Tensor& desc_b,
    const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t meta_dim, double raw_preservation_weight,
    double raw_preservation_margin, double raw_preservation_raw_margin);
torch::Tensor make_python_compare_graph_loss_with_hard_negative_dustbin_for_test(
    pfm::v21::PfmV21GraphMatcherImpl& graph_matcher, const torch::Tensor& desc_a, const torch::Tensor& desc_b,
    const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t meta_dim,
    double hard_negative_dustbin_weight, int64_t hard_negative_dustbin_topk,
    double hard_negative_dustbin_margin, double hard_negative_dustbin_spatial_min_distance);
torch::Tensor make_python_compare_graph_metadata_for_test(const torch::Tensor& points, int64_t meta_dim,
                                                          const std::string& metadata_mode);
torch::Tensor make_python_compare_graph_loss_with_attention_budget_for_test(
    pfm::v21::PfmV21GraphMatcherImpl& graph_matcher, const torch::Tensor& desc_a, const torch::Tensor& desc_b,
    const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t meta_dim, int64_t max_attention_layers);
torch::Tensor make_python_compare_graph_loss_with_random_attention_budget_for_test(
    pfm::v21::PfmV21GraphMatcherImpl& graph_matcher, const torch::Tensor& desc_a, const torch::Tensor& desc_b,
    const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t meta_dim, int64_t max_attention_layers,
    uint64_t seed);
torch::Tensor make_python_compare_graph_loss_with_attention_work_fraction_for_test(
    pfm::v21::PfmV21GraphMatcherImpl& graph_matcher, const torch::Tensor& desc_a, const torch::Tensor& desc_b,
    const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t meta_dim,
    double max_attention_work_fraction);
std::pair<torch::Tensor, int64_t> make_python_compare_graph_loss_with_width_keep_ratio_for_test(
    pfm::v21::PfmV21GraphMatcherImpl& graph_matcher, const torch::Tensor& desc_a, const torch::Tensor& desc_b,
    const torch::Tensor& points_a, const torch::Tensor& points_b, int64_t meta_dim, double width_keep_ratio,
    uint64_t seed);

} // namespace pfm::testing

namespace
{

struct CoutCapture
{
    std::ostringstream stream;
    std::streambuf* old = nullptr;

    CoutCapture() : old(std::cout.rdbuf(stream.rdbuf()))
    {
    }
    CoutCapture(const CoutCapture&) = delete;
    CoutCapture& operator=(const CoutCapture&) = delete;
    CoutCapture(CoutCapture&&) = delete;
    CoutCapture& operator=(CoutCapture&&) = delete;

    ~CoutCapture() noexcept
    {
        try
        {
            if (old != nullptr)
            {
                std::cout.rdbuf(old);
            }
        }
        catch (...)
        {
        }
    }

    std::string str() const
    {
        return stream.str();
    }
};

class TempTrainingDirectory
{
  public:
    explicit TempTrainingDirectory(const std::string& stem)
    {
        const auto suffix =
            std::to_string(static_cast<long long>(getpid())) + "_" + std::to_string(std::random_device{}());
        _path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
        std::filesystem::create_directory(_path);
    }

    ~TempTrainingDirectory()
    {
        for (const auto& file_path : _files)
        {
            std::remove(file_path.string().c_str());
        }
        std::error_code ignored;
        const auto cache_dir = _path / "pair_cache";
        if (std::filesystem::exists(cache_dir, ignored))
        {
            for (const auto& entry : std::filesystem::directory_iterator(cache_dir))
            {
                std::filesystem::remove(entry.path(), ignored);
            }
            std::filesystem::remove(cache_dir, ignored);
        }
        std::filesystem::remove(_path, ignored);
    }

    const std::filesystem::path& path() const
    {
        return _path;
    }

    std::filesystem::path file(const std::string& name)
    {
        auto file_path = _path / name;
        _files.push_back(file_path);
        return file_path;
    }

  private:
    std::filesystem::path _path;
    std::vector<std::filesystem::path> _files;
};

void require_image_written(const std::filesystem::path& path, int offset)
{
    cv::Mat image(32, 32, CV_8UC1);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            image.at<uint8_t>(y, x) = static_cast<uint8_t>((x * 7 + y * 11 + offset) % 256);
        }
    }
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

void require_sized_image_written(const std::filesystem::path& path, int height, int width, int offset)
{
    cv::Mat image(height, width, CV_8UC1);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            image.at<uint8_t>(y, x) = static_cast<uint8_t>((x * 7 + y * 11 + offset) % 256);
        }
    }
    PFM_REQUIRE(cv::imwrite(path.string(), image));
}

int64_t read_checkpoint_config_value(const std::string& checkpoint, const char* name)
{
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint);
    torch::serialize::InputArchive config_archive;
    archive.read("config", config_archive);
    torch::Tensor value;
    config_archive.read(name, value);
    PFM_REQUIRE(value.defined());
    return value.to(torch::kCPU, torch::kInt64).reshape({1}).item<int64_t>();
}

torch::Tensor read_nested_checkpoint_tensor(const std::string& checkpoint, const std::vector<std::string>& archive_path,
                                            const char* tensor_name)
{
    torch::serialize::InputArchive current;
    current.load_from(checkpoint);
    std::vector<torch::serialize::InputArchive> archives;
    archives.reserve(archive_path.size());
    for (const auto& name : archive_path)
    {
        archives.emplace_back();
        current.read(name, archives.back());
        current = std::move(archives.back());
    }
    torch::Tensor tensor;
    current.read(tensor_name, tensor);
    PFM_REQUIRE(tensor.defined());
    return tensor.detach().clone();
}

pfm::TrainConfig tiny_config(const TempTrainingDirectory& temp_dir)
{
    pfm::TrainConfig config;
    config.image_dir = temp_dir.path().string();
    config.checkpoint = (temp_dir.path() / "checkpoint.pt").string();
    config.epochs = 1;
    config.batch_size = 1;
    config.base_channels = 2;
    config.descriptor_dim = 4;
    config.graph_hidden_dim = 16;
    config.graph_attention_layers = 1;
    config.graph_keypoint_meta_dim = 16;
    config.learning_rate = 1.0e-3;
    config.min_keypoint_intensity = 0.0;
    config.train_ratio = 1.0;
    config.val_ratio = 0.0;
    return config;
}

bool has_trainable_parameter_prefix(const std::vector<std::string>& names, const std::string& prefix)
{
    for (const auto& name : names)
    {
        if (name.rfind(prefix, 0) == 0)
        {
            return true;
        }
    }
    return false;
}

} // namespace

static void trainer_one_epoch_saves_loadable_checkpoint()
{
    TempTrainingDirectory temp_dir("pfm_trainer_checkpoint");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.initial_loss > 0.0);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(config.checkpoint));
}

static void trainer_default_config_uses_larger_model_settings()
{
    pfm::TrainConfig config;

    PFM_REQUIRE(config.base_channels == 32);
    PFM_REQUIRE(config.descriptor_dim == 128);
    PFM_REQUIRE(config.graph_hidden_dim == 256);
    PFM_REQUIRE(config.graph_attention_layers == 6);
    PFM_REQUIRE(config.graph_keypoint_meta_dim == 16);
    PFM_REQUIRE(config.training_profile == "full");
    PFM_REQUIRE(config.graph_matcher_metadata_mode == "full");
    PFM_REQUIRE_CLOSE(config.graph_matcher_accept_weight, 0.2, 1.0e-12);
    PFM_REQUIRE(config.graph_matcher_no_match_points == 0);
    PFM_REQUIRE_CLOSE(config.graph_matcher_no_match_min_distance, 4.0, 1.0e-12);
    PFM_REQUIRE(config.graph_matcher_train_max_attention_layers == 0);
    PFM_REQUIRE(!config.graph_matcher_train_random_attention_layers);
    PFM_REQUIRE_CLOSE(config.graph_matcher_train_max_attention_work_fraction, 1.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.graph_matcher_train_width_keep_ratio, 1.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.graph_matcher_prune_ranking_weight, 0.1, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.graph_matcher_prune_ranking_margin, 0.25, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.graph_matcher_stop_confidence_weight, 0.05, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.graph_matcher_stop_confidence_margin, 0.5, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.graph_matcher_raw_preservation_weight, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.graph_matcher_raw_preservation_margin, 1.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.graph_matcher_raw_preservation_raw_margin, 0.05, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.graph_matcher_hard_negative_dustbin_weight, 0.0, 1.0e-12);
    PFM_REQUIRE(config.graph_matcher_hard_negative_dustbin_topk == 8);
    PFM_REQUIRE_CLOSE(config.graph_matcher_hard_negative_dustbin_margin, 0.25, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.graph_matcher_hard_negative_dustbin_spatial_min_distance, 0.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.learning_rate, 3.0e-4, 1.0e-9);
    PFM_REQUIRE(config.lr_warmup_steps == 0);
    PFM_REQUIRE_CLOSE(config.min_learning_rate_ratio, 0.01, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.weight_decay, 5.0e-4, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.gradient_clip_norm, 1.0, 1.0e-12);
    PFM_REQUIRE(config.dataloader_workers == 0);
    PFM_REQUIRE(config.prefetch_batches == 2);
    PFM_REQUIRE(!config.pin_memory);
    PFM_REQUIRE(!config.train_backbone);
    PFM_REQUIRE(!config.train_dual_fpn);
    PFM_REQUIRE(!config.freeze_descriptor_head);
    PFM_REQUIRE(!config.train_sparse_context);
    PFM_REQUIRE(!config.train_keypoint_head);
    PFM_REQUIRE(!config.train_geometry_head);
    PFM_REQUIRE(!config.train_blended_descriptors);
    PFM_REQUIRE(!config.train_texture_adapter);
    PFM_REQUIRE(!config.train_descriptor_fusion);
    PFM_REQUIRE(!config.train_quality_head);
    PFM_REQUIRE(!config.train_graph_matcher);
    PFM_REQUIRE_CLOSE(config.training_texture_blend_weight, 1.0, 1.0e-12);
    PFM_REQUIRE(!config.descriptor_only_finetune);
    PFM_REQUIRE(!config.graph_only_finetune);
    PFM_REQUIRE_CLOSE(config.min_keypoint_intensity, 0.08, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.train_ratio, 1.0, 1.0e-12);
    PFM_REQUIRE_CLOSE(config.val_ratio, 0.0, 1.0e-12);
}

static void trainer_learning_rate_schedule_warms_up_then_decays_to_floor()
{
    pfm::TrainConfig config;
    config.learning_rate = 1.0e-4;
    config.lr_warmup_steps = 4;
    config.min_learning_rate_ratio = 0.1;

    const auto first = pfm::testing::training_learning_rate_for_step_for_test(config, 0, 12);
    const auto second = pfm::testing::training_learning_rate_for_step_for_test(config, 1, 12);
    const auto warm = pfm::testing::training_learning_rate_for_step_for_test(config, 3, 12);
    const auto decay_start = pfm::testing::training_learning_rate_for_step_for_test(config, 4, 12);
    const auto final = pfm::testing::training_learning_rate_for_step_for_test(config, 11, 12);

    PFM_REQUIRE_CLOSE(first, 2.5e-5, 1.0e-12);
    PFM_REQUIRE(second > first);
    PFM_REQUIRE_CLOSE(warm, config.learning_rate, 1.0e-12);
    PFM_REQUIRE_CLOSE(decay_start, config.learning_rate, 1.0e-12);
    PFM_REQUIRE_CLOSE(final, 1.0e-5, 1.0e-12);
}

static void trainer_checkpoint_saves_graph_matcher_architecture_config()
{
    TempTrainingDirectory temp_dir("pfm_trainer_checkpoint_architecture");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.graph_hidden_dim = 16;
    config.graph_attention_layers = 3;
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(read_checkpoint_config_value(config.checkpoint, "checkpoint_version") == 3);
    PFM_REQUIRE(read_checkpoint_config_value(config.checkpoint, "graph_hidden_dim") == 16);
    PFM_REQUIRE(read_checkpoint_config_value(config.checkpoint, "graph_attention_layers") == 3);
    PFM_REQUIRE(read_checkpoint_config_value(config.checkpoint, "graph_keypoint_meta_dim") == 16);
}

static void trainer_descriptor_only_finetune_freezes_backbone_but_updates_descriptor_head()
{
    TempTrainingDirectory temp_dir("pfm_trainer_descriptor_only_finetune");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);

    auto base_config = tiny_config(temp_dir);
    base_config.max_train_batches = 1;
    base_config.checkpoint = temp_dir.file("base.pt").string();
    auto base_result = pfm::train_model(base_config);
    PFM_REQUIRE(base_result.epochs_completed == 1);

    auto finetune_config = tiny_config(temp_dir);
    finetune_config.max_train_batches = 1;
    finetune_config.init_checkpoint = base_config.checkpoint;
    finetune_config.checkpoint = temp_dir.file("finetuned.pt").string();
    finetune_config.descriptor_only_finetune = true;
    auto finetune_result = pfm::train_model(finetune_config);
    PFM_REQUIRE(finetune_result.epochs_completed == 1);

    const auto base_backbone =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"backbone", "stage1", "0"}, "weight");
    const auto finetuned_backbone =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"backbone", "stage1", "0"}, "weight");
    PFM_REQUIRE(torch::allclose(base_backbone, finetuned_backbone, 0.0, 0.0));

    const auto base_backbone_running_mean =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"backbone", "stage1", "1"}, "running_mean");
    const auto finetuned_backbone_running_mean =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"backbone", "stage1", "1"}, "running_mean");
    PFM_REQUIRE(torch::allclose(base_backbone_running_mean, finetuned_backbone_running_mean, 0.0, 0.0));

    const auto base_graph_matcher =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"graph_matcher", "descriptor_projection"}, "weight");
    const auto finetuned_graph_matcher =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"graph_matcher", "descriptor_projection"}, "weight");
    PFM_REQUIRE(torch::allclose(base_graph_matcher, finetuned_graph_matcher, 0.0, 0.0));

    const auto base_descriptor =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"sparse_head", "descriptors", "6"}, "weight");
    const auto finetuned_descriptor =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"sparse_head", "descriptors", "6"}, "weight");
    PFM_REQUIRE(!torch::allclose(base_descriptor, finetuned_descriptor, 1.0e-7, 1.0e-7));
}

static void trainer_graph_only_finetune_freezes_feature_extractor_but_updates_graph_matcher()
{
    TempTrainingDirectory temp_dir("pfm_trainer_graph_only_finetune");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);

    auto base_config = tiny_config(temp_dir);
    base_config.max_train_batches = 1;
    base_config.checkpoint = temp_dir.file("base.pt").string();
    auto base_result = pfm::train_model(base_config);
    PFM_REQUIRE(base_result.epochs_completed == 1);

    auto finetune_config = tiny_config(temp_dir);
    finetune_config.max_train_batches = 1;
    finetune_config.init_checkpoint = base_config.checkpoint;
    finetune_config.checkpoint = temp_dir.file("graph_finetuned.pt").string();
    finetune_config.graph_only_finetune = true;
    auto finetune_result = pfm::train_model(finetune_config);
    PFM_REQUIRE(finetune_result.epochs_completed == 1);

    const auto base_backbone =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"backbone", "stage1", "0"}, "weight");
    const auto finetuned_backbone =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"backbone", "stage1", "0"}, "weight");
    PFM_REQUIRE(torch::allclose(base_backbone, finetuned_backbone, 0.0, 0.0));

    const auto base_descriptor =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"sparse_head", "descriptors", "6"}, "weight");
    const auto finetuned_descriptor =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"sparse_head", "descriptors", "6"}, "weight");
    PFM_REQUIRE(torch::allclose(base_descriptor, finetuned_descriptor, 0.0, 0.0));

    const auto base_dense =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"dense_head", "correlation_projection"}, "weight");
    const auto finetuned_dense =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"dense_head", "correlation_projection"}, "weight");
    PFM_REQUIRE(torch::allclose(base_dense, finetuned_dense, 0.0, 0.0));

    const auto base_graph_matcher =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"graph_matcher", "descriptor_projection"}, "weight");
    const auto finetuned_graph_matcher =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"graph_matcher", "descriptor_projection"}, "weight");
    PFM_REQUIRE(!torch::allclose(base_graph_matcher, finetuned_graph_matcher, 1.0e-7, 1.0e-7));
}

static void trainer_viewpoint_head_only_finetune_updates_only_viewpoint_descriptor_branch()
{
    TempTrainingDirectory temp_dir("pfm_trainer_viewpoint_head_only_finetune");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);

    auto base_config = tiny_config(temp_dir);
    base_config.max_train_batches = 1;
    base_config.checkpoint = temp_dir.file("base.pt").string();
    auto base_result = pfm::train_model(base_config);
    PFM_REQUIRE(base_result.epochs_completed == 1);

    auto finetune_config = tiny_config(temp_dir);
    finetune_config.max_train_batches = 1;
    finetune_config.init_checkpoint = base_config.checkpoint;
    finetune_config.checkpoint = temp_dir.file("viewpoint_finetuned.pt").string();
    finetune_config.viewpoint_head_only_finetune = true;
    auto finetune_result = pfm::train_model(finetune_config);
    PFM_REQUIRE(finetune_result.epochs_completed == 1);

    const auto base_backbone =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"backbone", "stage1", "0"}, "weight");
    const auto finetuned_backbone =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"backbone", "stage1", "0"}, "weight");
    PFM_REQUIRE(torch::allclose(base_backbone, finetuned_backbone, 0.0, 0.0));

    const auto base_descriptor =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"sparse_head", "descriptors", "6"}, "weight");
    const auto finetuned_descriptor =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"sparse_head", "descriptors", "6"}, "weight");
    PFM_REQUIRE(torch::allclose(base_descriptor, finetuned_descriptor, 0.0, 0.0));

    const auto base_graph_matcher =
        read_nested_checkpoint_tensor(base_config.checkpoint, {"graph_matcher", "descriptor_projection"}, "weight");
    const auto finetuned_graph_matcher =
        read_nested_checkpoint_tensor(finetune_config.checkpoint, {"graph_matcher", "descriptor_projection"}, "weight");
    PFM_REQUIRE(torch::allclose(base_graph_matcher, finetuned_graph_matcher, 0.0, 0.0));

    const auto base_viewpoint = read_nested_checkpoint_tensor(
        base_config.checkpoint, {"sparse_head", "descriptor_viewpoint_context"}, "weight");
    const auto finetuned_viewpoint = read_nested_checkpoint_tensor(
        finetune_config.checkpoint, {"sparse_head", "descriptor_viewpoint_context"}, "weight");
    PFM_REQUIRE(!torch::allclose(base_viewpoint, finetuned_viewpoint, 1.0e-7, 1.0e-7));
}

static void trainer_missing_image_dir_throws_invalid_argument()
{
    TempTrainingDirectory temp_dir("pfm_trainer_missing_dir");
    auto config = tiny_config(temp_dir);
    config.image_dir = (temp_dir.path() / "missing").string();
    temp_dir.file("checkpoint.pt");

    PFM_REQUIRE_INVALID_ARG(pfm::train_model(config));
}

static void trainer_invalid_numeric_parameters_throw_invalid_argument()
{
    TempTrainingDirectory temp_dir("pfm_trainer_invalid_numeric");
    require_image_written(temp_dir.file("image.png"), 0);
    auto config = tiny_config(temp_dir);
    temp_dir.file("checkpoint.pt");

    auto invalid_epochs = config;
    invalid_epochs.epochs = 0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_epochs));

    auto invalid_batch = config;
    invalid_batch.batch_size = 0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_batch));

    auto invalid_base_channels = config;
    invalid_base_channels.base_channels = 0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_base_channels));

    auto invalid_descriptor_dim = config;
    invalid_descriptor_dim.descriptor_dim = 0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_descriptor_dim));

    auto invalid_learning_rate = config;
    invalid_learning_rate.learning_rate = 0.0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_learning_rate));

    auto invalid_warmup = config;
    invalid_warmup.lr_warmup_steps = -1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_warmup));

    auto invalid_min_lr_ratio = config;
    invalid_min_lr_ratio.min_learning_rate_ratio = 1.1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_min_lr_ratio));

    auto invalid_resize = config;
    invalid_resize.resize = -1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_resize));

    auto invalid_pairs_per_image = config;
    invalid_pairs_per_image.pairs_per_image = 0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_pairs_per_image));

    auto invalid_no_match_points = config;
    invalid_no_match_points.graph_matcher_no_match_points = -1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_no_match_points));

    auto invalid_no_match_distance = config;
    invalid_no_match_distance.graph_matcher_no_match_min_distance = -1.0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_no_match_distance));

    auto invalid_attention_budget = config;
    invalid_attention_budget.graph_matcher_train_max_attention_layers = -1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_attention_budget));

    auto invalid_metadata_mode = config;
    invalid_metadata_mode.graph_matcher_metadata_mode = "bad_mode";
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_metadata_mode));

    auto invalid_attention_work_budget = config;
    invalid_attention_work_budget.graph_matcher_train_max_attention_work_fraction = 1.5;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_attention_work_budget));

    auto invalid_width_keep = config;
    invalid_width_keep.graph_matcher_train_width_keep_ratio = 0.0;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_width_keep));

    auto invalid_raw_preservation_weight = config;
    invalid_raw_preservation_weight.graph_matcher_raw_preservation_weight = -0.1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_raw_preservation_weight));

    auto invalid_raw_preservation_margin = config;
    invalid_raw_preservation_margin.graph_matcher_raw_preservation_margin = -0.1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_raw_preservation_margin));

    auto invalid_raw_preservation_raw_margin = config;
    invalid_raw_preservation_raw_margin.graph_matcher_raw_preservation_raw_margin = -0.1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_raw_preservation_raw_margin));

    auto invalid_hard_negative_dustbin_weight = config;
    invalid_hard_negative_dustbin_weight.graph_matcher_hard_negative_dustbin_weight = -0.1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_hard_negative_dustbin_weight));

    auto invalid_hard_negative_dustbin_topk = config;
    invalid_hard_negative_dustbin_topk.graph_matcher_hard_negative_dustbin_topk = -1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_hard_negative_dustbin_topk));

    auto invalid_hard_negative_dustbin_margin = config;
    invalid_hard_negative_dustbin_margin.graph_matcher_hard_negative_dustbin_margin = -0.1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_hard_negative_dustbin_margin));

    auto invalid_hard_negative_dustbin_distance = config;
    invalid_hard_negative_dustbin_distance.graph_matcher_hard_negative_dustbin_spatial_min_distance = -0.1;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_hard_negative_dustbin_distance));

    auto invalid_training_profile = config;
    invalid_training_profile.training_profile = "wide-open";
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_training_profile));

    auto invalid_finetune_mode = config;
    invalid_finetune_mode.descriptor_only_finetune = true;
    invalid_finetune_mode.graph_only_finetune = true;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_finetune_mode));

    auto invalid_viewpoint_finetune_mode = config;
    invalid_viewpoint_finetune_mode.descriptor_only_finetune = true;
    invalid_viewpoint_finetune_mode.viewpoint_head_only_finetune = true;
    PFM_REQUIRE_INVALID_ARG(pfm::train_model(invalid_viewpoint_finetune_mode));
}

static void trainer_invalid_device_throws_invalid_argument()
{
    TempTrainingDirectory temp_dir("pfm_trainer_invalid_device");
    require_image_written(temp_dir.file("image.png"), 0);
    auto config = tiny_config(temp_dir);
    config.device = "cuda:abc";
    temp_dir.file("checkpoint.pt");

    PFM_REQUIRE_INVALID_ARG(pfm::train_model(config));
}

static void trainer_cuda_device_is_strictly_validated()
{
    if (torch::cuda::is_available())
    {
        return;
    }

    TempTrainingDirectory temp_dir("pfm_trainer_cuda_unavailable");
    require_image_written(temp_dir.file("image.png"), 0);
    auto config = tiny_config(temp_dir);
    config.device = "cuda";
    temp_dir.file("checkpoint.pt");

    PFM_REQUIRE_INVALID_ARG(pfm::train_model(config));
}

static void trainer_cuda_one_epoch_saves_cpu_loadable_checkpoint_when_available()
{
    if (!torch::cuda::is_available())
    {
        return;
    }

    TempTrainingDirectory temp_dir("pfm_trainer_cuda_checkpoint");
    require_image_written(temp_dir.file("image.png"), 0);
    auto config = tiny_config(temp_dir);
    config.device = "cuda";
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(config.checkpoint));
}

static void trainer_resizes_dense_warp_as_normalized_local_offsets()
{
    auto xy = torch::meshgrid({torch::arange(32, torch::kFloat32), torch::arange(32, torch::kFloat32)}, "ij");
    auto warp = torch::stack({xy[1] + 16.0F, xy[0]}, 2).unsqueeze(0);
    auto offsets = torch::zeros({1, 2, 16, 16}, torch::kFloat32);

    auto target = pfm::testing::resize_offsets_for_dense_head_for_test(warp, offsets);
    auto expected_x = torch::ones({16, 16});
    auto expected_y = torch::zeros({16, 16});

    PFM_REQUIRE(target.sizes() == offsets.sizes());
    PFM_REQUIRE(torch::allclose(target.index({0, 0}), expected_x, 1.0e-6, 1.0e-6));
    PFM_REQUIRE(torch::allclose(target.index({0, 1}), expected_y, 1.0e-6, 1.0e-6));
}

static void trainer_repeatability_uses_warped_heatmap_correspondence()
{
    auto heatmap_b = torch::tensor({{{{0.0F, 1.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto warp_a_to_b = torch::tensor({{{{1.0F, 0.0F}, {1.0F, 0.0F}}, {{1.0F, 1.0F}, {1.0F, 1.0F}}}}, torch::kFloat32);

    auto heatmap_b_at_a = pfm::testing::warp_heatmap_for_repeatability_for_test(heatmap_b, warp_a_to_b);

    PFM_REQUIRE(torch::allclose(heatmap_b_at_a.index({0, 0, 0, 0}), torch::tensor(1.0F), 1.0e-6, 1.0e-6));
}

static void trainer_detector_target_loss_prefers_warp_consistent_peaks()
{
    auto target = torch::tensor({{{{0.0F, 1.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto mask = torch::ones({1, 1, 2, 2}, torch::kFloat32);
    auto good_a = torch::tensor({{{{0.01F, 0.95F}, {0.01F, 0.01F}}}}, torch::kFloat32);
    auto good_b_at_a = torch::tensor({{{{0.01F, 0.90F}, {0.01F, 0.01F}}}}, torch::kFloat32);
    auto bad_a = torch::tensor({{{{0.95F, 0.01F}, {0.01F, 0.01F}}}}, torch::kFloat32);
    auto bad_b_at_a = torch::tensor({{{{0.01F, 0.01F}, {0.95F, 0.01F}}}}, torch::kFloat32);

    auto good_loss = pfm::testing::make_heatmap_correspondence_target_loss_for_test(good_a, good_b_at_a, target, mask);
    auto bad_loss = pfm::testing::make_heatmap_correspondence_target_loss_for_test(bad_a, bad_b_at_a, target, mask);

    PFM_REQUIRE(good_loss.item<float>() < bad_loss.item<float>() * 0.25F);
}

static void trainer_detector_target_loss_weights_missing_positive_peaks()
{
    auto target = torch::zeros({1, 1, 8, 8}, torch::kFloat32);
    target.index_put_({0, 0, 3, 3}, 1.0F);
    auto mask = torch::ones({1, 1, 8, 8}, torch::kFloat32);
    auto missed_positive = torch::full({1, 1, 8, 8}, 0.01F, torch::kFloat32);
    auto false_positive = torch::full({1, 1, 8, 8}, 0.01F, torch::kFloat32);
    false_positive.index_put_({0, 0, 0, 0}, 0.95F);
    false_positive.index_put_({0, 0, 3, 3}, 0.95F);

    auto missed_loss =
        pfm::testing::make_heatmap_correspondence_target_loss_for_test(missed_positive, missed_positive, target, mask);
    auto false_positive_loss =
        pfm::testing::make_heatmap_correspondence_target_loss_for_test(false_positive, false_positive, target, mask);

    PFM_REQUIRE(missed_loss.item<float>() > false_positive_loss.item<float>() * 2.0F);
}

static void trainer_positive_target_loss_directly_raises_target_peaks()
{
    auto heatmap = torch::full({1, 1, 8, 8}, 0.01F, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    auto target = torch::zeros({1, 1, 8, 8}, torch::kFloat32);
    target.index_put_({0, 0, 3, 3}, 1.0F);
    auto mask = torch::ones({1, 1, 8, 8}, torch::kFloat32);

    auto before = heatmap.index({0, 0, 3, 3}).item<float>();
    auto loss = pfm::testing::make_heatmap_positive_target_loss_for_test(heatmap, target, mask);
    loss.backward();
    {
        torch::NoGradGuard guard;
        heatmap -= heatmap.grad() * 0.01F;
        heatmap.clamp_(0.0F, 1.0F);
    }

    auto after = heatmap.index({0, 0, 3, 3}).item<float>();
    PFM_REQUIRE(after > before + 0.1F);
}

static void trainer_descriptor_loss_uses_warped_correspondence()
{
    auto descriptors_a = torch::tensor({{{{1.0F, 0.0F}}, {{0.0F, 1.0F}}}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{{{0.0F, 1.0F}}, {{1.0F, 0.0F}}}}, torch::kFloat32);
    auto warp = torch::tensor({{{{1.0F, 0.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto valid_mask = torch::ones({1, 1, 2}, torch::kBool);

    auto loss = pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.item<float>() < 0.4F);
}

static void trainer_descriptor_loss_ignores_invalid_warp_targets()
{
    auto descriptors_a = torch::tensor({{{{1.0F, 0.0F}}, {{0.0F, 1.0F}}}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{{{0.0F, 1.0F}}, {{1.0F, 0.0F}}}}, torch::kFloat32);
    auto warp = torch::tensor({{{{0.0F, 0.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto valid_mask = torch::tensor({{{false, true}}}, torch::kBool);

    auto loss = pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.item<float>() < 0.4F);
}

static void trainer_descriptor_loss_penalizes_globally_collapsed_descriptors()
{
    auto descriptors_a = torch::eye(16, torch::kFloat32).narrow(0, 0, 4).transpose(0, 1).reshape({1, 16, 1, 4});
    auto descriptors_b = descriptors_a.clone();
    auto collapsed_a = torch::ones({1, 16, 1, 4}, torch::kFloat32);
    auto collapsed_b = torch::ones({1, 16, 1, 4}, torch::kFloat32);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(4, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto distinctive_loss =
        pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);
    auto collapsed_loss =
        pfm::testing::make_sparse_descriptor_loss_for_test(collapsed_a, collapsed_b, warp, valid_mask);

    PFM_REQUIRE(collapsed_loss.item<float>() > distinctive_loss.item<float>() + 1.0F);
}

static void trainer_dense_descriptor_hard_negative_loss_scans_full_map()
{
    auto descriptors = torch::eye(81, torch::kFloat32).narrow(0, 0, 80);
    auto clean_a = descriptors.transpose(0, 1).reshape({1, 81, 1, 80});
    auto clean_b = clean_a.clone();
    auto hard_rows = descriptors.clone();
    hard_rows.index_put_({torch::indexing::Slice(40, 80)}, descriptors.index({torch::indexing::Slice(0, 40)}));
    auto hard_b = hard_rows.transpose(0, 1).reshape({1, 81, 1, 80});
    auto warp = torch::zeros({1, 1, 80, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(80, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 80}, torch::kBool);

    auto clean_loss =
        pfm::testing::make_dense_descriptor_hard_negative_loss_for_test(clean_a, clean_b, warp, valid_mask);
    auto hard_loss = pfm::testing::make_dense_descriptor_hard_negative_loss_for_test(clean_a, hard_b, warp, valid_mask);

    PFM_REQUIRE(clean_loss.item<float>() < 1.0e-4F);
    PFM_REQUIRE(hard_loss.item<float>() > clean_loss.item<float>() + 0.1F);
}

static void trainer_dense_descriptor_hard_negative_loss_weights_multiple_hard_negatives()
{
    constexpr int64_t width = 20;
    auto descriptors = torch::eye(width, torch::kFloat32).transpose(0, 1).reshape({1, width, 1, width});
    auto one_hard_b = descriptors.clone();
    auto many_hard_b = descriptors.clone();
    for (int64_t x = 10; x < 18; ++x)
    {
        many_hard_b.index_put_({0, torch::indexing::Slice(), 0, x},
                               descriptors.index({0, torch::indexing::Slice(), 0, 0}));
    }
    one_hard_b.index_put_({0, torch::indexing::Slice(), 0, 10}, descriptors.index({0, torch::indexing::Slice(), 0, 0}));
    auto warp = torch::zeros({1, 1, width, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(width, torch::kFloat32));
    auto valid_mask = torch::zeros({1, 1, width}, torch::kBool);
    valid_mask.index_put_({0, 0, 0}, true);

    auto one_hard_loss =
        pfm::testing::make_dense_descriptor_hard_negative_loss_for_test(descriptors, one_hard_b, warp, valid_mask);
    auto many_hard_loss =
        pfm::testing::make_dense_descriptor_hard_negative_loss_for_test(descriptors, many_hard_b, warp, valid_mask);

    PFM_REQUIRE(many_hard_loss.item<float>() > one_hard_loss.item<float>() + 0.05F);
}

static void trainer_bidirectional_dense_descriptor_hard_negative_loss_catches_reverse_duplicates()
{
    constexpr int64_t width = 24;
    auto descriptors = torch::eye(width, torch::kFloat32).transpose(0, 1).reshape({1, width, 1, width});
    auto duplicate_a = descriptors.clone();
    duplicate_a.index_put_({0, torch::indexing::Slice(), 0, 20},
                           descriptors.index({0, torch::indexing::Slice(), 0, 0}));
    auto warp = torch::zeros({1, 1, width, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(width, torch::kFloat32));
    auto valid_mask = torch::zeros({1, 1, width}, torch::kBool);
    valid_mask.index_put_({0, 0, 0}, true);

    auto forward_only =
        pfm::testing::make_dense_descriptor_hard_negative_loss_for_test(duplicate_a, descriptors, warp, valid_mask);
    auto bidirectional = pfm::testing::make_bidirectional_dense_descriptor_hard_negative_loss_for_test(
        duplicate_a, descriptors, warp, valid_mask);

    PFM_REQUIRE(forward_only.item<float>() < 1.0e-4F);
    PFM_REQUIRE(bidirectional.item<float>() > forward_only.item<float>() + 0.02F);
    PFM_REQUIRE(bidirectional.item<float>() < forward_only.item<float>() + 0.08F);
}

static torch::Tensor make_cyclic_safe_descriptor_row(int64_t width, int64_t group_shift = 0)
{
    auto descriptors = torch::zeros({1, 16, 1, width}, torch::kFloat32);
    const auto group = ((group_shift % 4) + 4) % 4;
    for (int64_t x = 0; x < width; ++x)
    {
        descriptors.index_put_({0, group * 4 + x, 0, x}, 1.0F);
    }
    return descriptors;
}

static torch::Tensor make_cyclic_safe_flipped_descriptor_row(int64_t width, int64_t group_shift = 0)
{
    auto descriptors = torch::zeros({1, 16, 1, width}, torch::kFloat32);
    const auto group = ((group_shift % 4) + 4) % 4;
    for (int64_t x = 0; x < width; ++x)
    {
        descriptors.index_put_({0, group * 4 + x, 0, width - 1 - x}, 1.0F);
    }
    return descriptors;
}

static torch::Tensor make_cyclic_safe_descriptor_grid(int64_t height, int64_t width, int64_t group_shift = 0,
                                                      bool half_turn_spatial = false)
{
    auto descriptors = torch::zeros({1, 16, height, width}, torch::kFloat32);
    const auto group = ((group_shift % 4) + 4) % 4;
    for (int64_t y = 0; y < height; ++y)
    {
        for (int64_t x = 0; x < width; ++x)
        {
            const auto identity = y * width + x;
            const auto target_y = half_turn_spatial ? height - 1 - y : y;
            const auto target_x = half_turn_spatial ? width - 1 - x : x;
            descriptors.index_put_({0, group * 4 + identity, target_y, target_x}, 1.0F);
        }
    }
    return descriptors;
}

static void trainer_warp_descriptor_contrastive_loss_uses_half_turn_correspondence()
{
    auto descriptors_a = make_cyclic_safe_descriptor_grid(2, 2);
    auto correct_b = make_cyclic_safe_descriptor_grid(2, 2, 0, true);
    auto same_position_b = make_cyclic_safe_descriptor_grid(2, 2, 0, false);
    auto warp = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    warp.index_put_({0, 0, 0, 0}, 1.0F);
    warp.index_put_({0, 0, 0, 1}, 1.0F);
    warp.index_put_({0, 0, 1, 0}, 0.0F);
    warp.index_put_({0, 0, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 0, 0}, 1.0F);
    warp.index_put_({0, 1, 0, 1}, 0.0F);
    warp.index_put_({0, 1, 1, 0}, 0.0F);
    warp.index_put_({0, 1, 1, 1}, 0.0F);
    auto valid_mask = torch::ones({1, 2, 2}, torch::kBool);

    auto correct_loss =
        pfm::testing::make_warp_descriptor_contrastive_loss_for_test(descriptors_a, correct_b, warp, valid_mask);
    auto same_position_loss =
        pfm::testing::make_warp_descriptor_contrastive_loss_for_test(descriptors_a, same_position_b, warp, valid_mask);

    PFM_REQUIRE(same_position_loss.item<float>() > correct_loss.item<float>() + 5.0F);
}

static void trainer_warp_descriptor_contrastive_loss_rejects_untrained_cyclic_descriptor_shift()
{
    auto descriptors_a = make_cyclic_safe_descriptor_row(4);
    auto descriptors_b = make_cyclic_safe_descriptor_row(4, 1);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(4, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto loss =
        pfm::testing::make_warp_descriptor_contrastive_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.item<float>() > 1.0F);
}

static void trainer_direct_full_map_descriptor_loss_penalizes_global_distractor()
{
    auto descriptors_a = make_cyclic_safe_descriptor_row(4);
    auto clean_b = make_cyclic_safe_flipped_descriptor_row(4, 0);
    auto hard_b = clean_b.clone();
    hard_b.index_put_({0, torch::indexing::Slice(), 0, 0}, clean_b.index({0, torch::indexing::Slice(), 0, 3}));

    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, 0, 0}, 3.0F);
    warp.index_put_({0, 0, 1, 0}, 2.0F);
    warp.index_put_({0, 0, 2, 0}, 1.0F);
    warp.index_put_({0, 0, 3, 0}, 0.0F);
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto clean_loss =
        pfm::testing::make_direct_full_map_descriptor_loss_for_test(descriptors_a, clean_b, warp, valid_mask);
    auto hard_loss =
        pfm::testing::make_direct_full_map_descriptor_loss_for_test(descriptors_a, hard_b, warp, valid_mask);

    PFM_REQUIRE(clean_loss.item<float>() < 0.1F);
    PFM_REQUIRE(hard_loss.item<float>() > clean_loss.item<float>() + 0.1F);
}

static void trainer_direct_full_map_descriptor_loss_rejects_untrained_cyclic_descriptor_shift()
{
    auto descriptors_a = make_cyclic_safe_descriptor_row(4);
    auto descriptors_b = make_cyclic_safe_descriptor_row(4, 1);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(4, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto loss =
        pfm::testing::make_direct_full_map_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.item<float>() > 1.0F);
}

static void trainer_descriptor_targets_use_cell_centers_for_warp_coordinates()
{
    auto warp = torch::zeros({1, 4, 4, 2}, torch::kFloat32);
    warp.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(), 0},
                    torch::arange(4, torch::kFloat32).reshape({1, 4}).expand({4, 4}));
    warp.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(), 1},
                    torch::arange(4, torch::kFloat32).reshape({4, 1}).expand({4, 4}));
    auto sample_indices = torch::tensor({0, 3}, torch::kLong);

    auto coordinates = pfm::testing::make_descriptor_target_coordinates_for_test(warp, sample_indices, 1, 4);

    PFM_REQUIRE_CLOSE(coordinates.index({0, 0, 0}).item<float>(), 0.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(coordinates.index({0, 1, 0}).item<float>(), 3.0F, 1.0e-5F);
}

static void trainer_warped_descriptor_sampling_preserves_subpixel_correspondence()
{
    auto descriptors = torch::zeros({1, 2, 1, 4}, torch::kFloat32);
    descriptors.index_put_({0, 0, 0, 1}, 1.0F);
    descriptors.index_put_({0, 1, 0, 2}, 1.0F);
    auto target_coordinates = torch::tensor({{{1.5F, 0.0F}}}, torch::kFloat32);

    auto sampled = pfm::testing::sample_warped_descriptors_for_test(descriptors, target_coordinates);

    PFM_REQUIRE_CLOSE(sampled.index({0, 0, 0}).item<float>(), 0.5F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(sampled.index({0, 0, 1}).item<float>(), 0.5F, 1.0e-5F);
}

static void trainer_descriptor_map_regularization_penalizes_spatial_collapse()
{
    auto collapsed = torch::ones({1, 8, 4, 4}, torch::kFloat32);
    auto diverse = torch::zeros({1, 8, 4, 4}, torch::kFloat32);
    for (int64_t y = 0; y < 4; ++y)
    {
        for (int64_t x = 0; x < 4; ++x)
        {
            diverse.index_put_({0, (y * 4 + x) % 8, y, x}, 1.0F);
        }
    }

    auto collapsed_loss = pfm::testing::make_descriptor_map_regularization_loss_for_test(collapsed);
    auto diverse_loss = pfm::testing::make_descriptor_map_regularization_loss_for_test(diverse);

    PFM_REQUIRE(collapsed_loss.item<float>() > diverse_loss.item<float>() + 0.2F);
}

static pfm::FeatureSet make_keypoint_descriptor_feature_set(const torch::Tensor& keypoints,
                                                            const torch::Tensor& descriptors)
{
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    return pfm::FeatureSet{keypoints.clone().to(torch::kFloat32),
                           torch::ones({keypoints.size(0)}, float_options),
                           descriptors.clone().to(torch::kFloat32),
                           torch::ones({keypoints.size(0)}, float_options),
                           torch::zeros({keypoints.size(0), 2}, float_options),
                           torch::zeros({keypoints.size(0), 2, 2}, float_options),
                           torch::empty({0, 2}, float_options),
                           torch::empty({0}, float_options),
                           4,
                           1};
}

static void trainer_keypoint_descriptor_loss_uses_sparse_keypoint_hard_negatives()
{
    auto keypoints = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}, torch::kFloat32);
    auto descriptors = torch::eye(16, torch::kFloat32).narrow(0, 0, 4);
    auto collapsed = torch::ones({4, 16}, torch::kFloat32);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(4, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto distinctive_loss = pfm::testing::make_keypoint_descriptor_loss_for_test(
        make_keypoint_descriptor_feature_set(keypoints, descriptors),
        make_keypoint_descriptor_feature_set(keypoints, descriptors), warp, valid_mask);
    auto collapsed_loss = pfm::testing::make_keypoint_descriptor_loss_for_test(
        make_keypoint_descriptor_feature_set(keypoints, collapsed),
        make_keypoint_descriptor_feature_set(keypoints, collapsed), warp, valid_mask);

    PFM_REQUIRE(collapsed_loss.item<float>() > distinctive_loss.item<float>() + 1.0F);
}

static void trainer_keypoint_descriptor_loss_penalizes_hardest_negative_margin()
{
    auto keypoints = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}}, torch::kFloat32);
    auto query_descriptors = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {-1.0F, 0.0F}}, torch::kFloat32);
    auto easy_b = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {-1.0F, 0.0F}}, torch::kFloat32);
    auto hard_b = torch::tensor({{1.0F, 0.0F}, {0.99F, 0.01F}, {-1.0F, 0.0F}}, torch::kFloat32);
    auto warp = torch::zeros({1, 1, 3, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(3, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 3}, torch::kBool);

    auto easy_loss = pfm::testing::make_keypoint_descriptor_loss_for_test(
        make_keypoint_descriptor_feature_set(keypoints, query_descriptors),
        make_keypoint_descriptor_feature_set(keypoints, easy_b), warp, valid_mask);
    auto hard_loss = pfm::testing::make_keypoint_descriptor_loss_for_test(
        make_keypoint_descriptor_feature_set(keypoints, query_descriptors),
        make_keypoint_descriptor_feature_set(keypoints, hard_b), warp, valid_mask);

    PFM_REQUIRE(hard_loss.item<float>() > easy_loss.item<float>() + 0.1F);
}

static void trainer_keypoint_descriptor_metrics_report_sparse_match_quality()
{
    auto keypoints = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}}, torch::kFloat32);
    auto descriptors = torch::eye(6, torch::kFloat32).narrow(0, 0, 3);
    auto wrong_b = descriptors.index_select(0, torch::tensor({1, 0, 2}, torch::kLong));
    auto warp = torch::zeros({1, 1, 3, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(3, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 3}, torch::kBool);
    auto features_a = make_keypoint_descriptor_feature_set(keypoints, descriptors);
    auto good_features_b = make_keypoint_descriptor_feature_set(keypoints, descriptors);
    auto bad_features_b = make_keypoint_descriptor_feature_set(keypoints, wrong_b);
    features_a.feature_map_width = 3;
    good_features_b.feature_map_width = 3;
    bad_features_b.feature_map_width = 3;

    auto good_metrics =
        pfm::testing::make_keypoint_descriptor_metric_tensor_for_test(features_a, good_features_b, warp, valid_mask);
    auto bad_metrics =
        pfm::testing::make_keypoint_descriptor_metric_tensor_for_test(features_a, bad_features_b, warp, valid_mask);

    PFM_REQUIRE_CLOSE(good_metrics.index({1}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE(good_metrics.index({4}).item<float>() > 0.5F);
    PFM_REQUIRE_CLOSE(good_metrics.index({5}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE(bad_metrics.index({1}).item<float>() < good_metrics.index({1}).item<float>());
    PFM_REQUIRE(bad_metrics.index({4}).item<float>() < good_metrics.index({4}).item<float>());
}

static void trainer_keypoint_dense_descriptor_loss_uses_warp_target_in_full_map()
{
    auto features_a = make_keypoint_descriptor_feature_set(torch::tensor({{0.0F, 0.0F}}, torch::kFloat32),
                                                           torch::tensor({{1.0F, 0.0F}}, torch::kFloat32));
    features_a.feature_map_width = 4;
    features_a.feature_map_height = 1;
    auto clean_b = torch::zeros({1, 2, 1, 4}, torch::kFloat32);
    clean_b.index_put_({0, 0, 0, 0}, 1.0F);
    clean_b.index_put_({0, 1, 0, torch::indexing::Slice(1, 4)}, 1.0F);
    auto hard_b = clean_b.clone();
    hard_b.index_put_({0, 0, 0, 3}, 1.0F);
    hard_b.index_put_({0, 1, 0, 3}, 0.0F);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(4, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto clean_loss = pfm::testing::make_keypoint_dense_descriptor_loss_for_test(features_a, clean_b, warp, valid_mask);
    auto hard_loss = pfm::testing::make_keypoint_dense_descriptor_loss_for_test(features_a, hard_b, warp, valid_mask);

    PFM_REQUIRE(clean_loss.item<float>() < 0.1F);
    PFM_REQUIRE(hard_loss.item<float>() > clean_loss.item<float>() + 0.5F);
}

static void trainer_keypoint_descriptor_losses_ignore_out_of_bounds_warp_targets()
{
    auto features_a = make_keypoint_descriptor_feature_set(torch::tensor({{0.0F, 0.0F}}, torch::kFloat32),
                                                           torch::tensor({{1.0F, 0.0F}}, torch::kFloat32));
    features_a.feature_map_width = 4;
    features_a.feature_map_height = 1;
    auto descriptors_b = torch::zeros({1, 2, 1, 4}, torch::kFloat32);
    descriptors_b.index_put_({0, 1, 0, 0}, 1.0F);
    descriptors_b.index_put_({0, 0, 0, torch::indexing::Slice(1, 4)}, 1.0F);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, -10.0F);
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto dense_loss =
        pfm::testing::make_keypoint_dense_descriptor_loss_for_test(features_a, descriptors_b, warp, valid_mask);
    auto warped_loss = pfm::testing::make_warped_keypoint_descriptor_contrastive_loss_for_test(
        features_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE_CLOSE(dense_loss.item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(warped_loss.item<float>(), 0.0F, 1.0e-6F);
}

static void trainer_keypoint_patch_descriptor_alignment_uses_warp_neighborhood()
{
    auto features_a = make_keypoint_descriptor_feature_set(torch::tensor({{1.0F, 1.0F}}, torch::kFloat32),
                                                           torch::tensor({{1.0F, 0.0F}}, torch::kFloat32));
    features_a.feature_map_width = 4;
    features_a.feature_map_height = 4;
    auto clean_b = torch::zeros({1, 2, 4, 4}, torch::kFloat32);
    clean_b.index_put_({0, 0, torch::indexing::Slice(1, 3), torch::indexing::Slice(1, 3)}, 1.0F);
    auto bad_b = torch::zeros({1, 2, 4, 4}, torch::kFloat32);
    bad_b.index_put_({0, 1, torch::indexing::Slice(1, 3), torch::indexing::Slice(1, 3)}, 1.0F);
    auto warp = torch::zeros({1, 4, 4, 2}, torch::kFloat32);
    auto xy = torch::meshgrid({torch::arange(4, torch::kFloat32), torch::arange(4, torch::kFloat32)}, "ij");
    warp.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(), 0}, xy[1]);
    warp.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(), 1}, xy[0]);
    auto valid_mask = torch::ones({1, 4, 4}, torch::kBool);

    auto clean_loss =
        pfm::testing::make_keypoint_patch_descriptor_alignment_loss_for_test(features_a, clean_b, warp, valid_mask);
    auto bad_loss =
        pfm::testing::make_keypoint_patch_descriptor_alignment_loss_for_test(features_a, bad_b, warp, valid_mask);

    PFM_REQUIRE(clean_loss.item<float>() < bad_loss.item<float>() - 0.5F);
}

static void trainer_warped_keypoint_descriptor_contrastive_loss_uses_true_warp_targets()
{
    auto features_a =
        make_keypoint_descriptor_feature_set(torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}}, torch::kFloat32),
                                             torch::eye(6, torch::kFloat32).narrow(0, 0, 3));
    features_a.feature_map_width = 3;
    features_a.feature_map_height = 1;
    auto clean_b = torch::zeros({1, 6, 1, 3}, torch::kFloat32);
    clean_b.index_put_({0, 0, 0, 0}, 1.0F);
    clean_b.index_put_({0, 1, 0, 1}, 1.0F);
    clean_b.index_put_({0, 2, 0, 2}, 1.0F);
    auto shuffled_b = torch::zeros({1, 6, 1, 3}, torch::kFloat32);
    shuffled_b.index_put_({0, 1, 0, 0}, 1.0F);
    shuffled_b.index_put_({0, 0, 0, 1}, 1.0F);
    shuffled_b.index_put_({0, 2, 0, 2}, 1.0F);
    auto warp = torch::zeros({1, 1, 3, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(3, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, 3}, torch::kBool);

    auto clean_loss =
        pfm::testing::make_warped_keypoint_descriptor_contrastive_loss_for_test(features_a, clean_b, warp, valid_mask);
    auto shuffled_loss = pfm::testing::make_warped_keypoint_descriptor_contrastive_loss_for_test(features_a, shuffled_b,
                                                                                                 warp, valid_mask);

    PFM_REQUIRE(clean_loss.item<float>() < shuffled_loss.item<float>() - 0.5F);
}

static void trainer_decoded_keypoint_repeatability_raises_warped_b_heatmap_targets()
{
    auto features_a = make_keypoint_descriptor_feature_set(torch::tensor({{1.0F, 0.0F}}, torch::kFloat32),
                                                           torch::tensor({{1.0F, 0.0F}}, torch::kFloat32));
    features_a.feature_map_width = 4;
    features_a.feature_map_height = 1;
    auto good_heatmap_b = torch::full({1, 1, 1, 4}, 0.05F, torch::kFloat32);
    good_heatmap_b.index_put_({0, 0, 0, 3}, 0.95F);
    auto bad_heatmap_b = torch::full({1, 1, 1, 4}, 0.05F, torch::kFloat32);
    bad_heatmap_b.index_put_({0, 0, 0, 3}, 0.10F);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, 0, 0}, 0.0F);
    warp.index_put_({0, 0, 1, 0}, 3.0F);
    warp.index_put_({0, 0, 2, 0}, 2.0F);
    warp.index_put_({0, 0, 3, 0}, 1.0F);
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);

    auto good_loss =
        pfm::testing::make_decoded_keypoint_repeatability_loss_for_test(features_a, good_heatmap_b, warp, valid_mask);
    auto bad_loss =
        pfm::testing::make_decoded_keypoint_repeatability_loss_for_test(features_a, bad_heatmap_b, warp, valid_mask);

    PFM_REQUIRE(good_loss.item<float>() < 0.1F);
    PFM_REQUIRE(bad_loss.item<float>() > good_loss.item<float>() + 1.0F);
}

static void trainer_warp_completed_keypoint_pair_uses_true_warped_b_descriptors()
{
    auto features_a = make_keypoint_descriptor_feature_set(
        torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}}, torch::kFloat32),
        torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {-1.0F, 0.0F}}, torch::kFloat32));
    features_a.feature_map_width = 4;
    features_a.feature_map_height = 1;
    auto descriptors_b = torch::zeros({1, 2, 1, 4}, torch::kFloat32);
    descriptors_b.index_put_({0, 0, 0, 3}, 1.0F);
    descriptors_b.index_put_({0, 1, 0, 1}, 1.0F);
    descriptors_b.index_put_({0, 0, 0, 2}, -1.0F);
    auto warp = torch::zeros({1, 1, 4, 2}, torch::kFloat32);
    warp.index_put_({0, 0, 0, 0}, 3.0F);
    warp.index_put_({0, 0, 1, 0}, 1.0F);
    warp.index_put_({0, 0, 2, 0}, 2.0F);
    auto valid_mask = torch::ones({1, 1, 4}, torch::kBool);
    valid_mask.index_put_({0, 0, 2}, false);

    auto completed =
        pfm::testing::make_warp_completed_keypoint_feature_pair_for_test(features_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(completed.first.keypoints.size(0) == 2);
    PFM_REQUIRE(completed.second.keypoints.size(0) == 2);
    PFM_REQUIRE_CLOSE(completed.second.keypoints.index({0, 0}).item<float>(), 3.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(completed.second.keypoints.index({1, 0}).item<float>(), 1.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(completed.second.descriptors.index({0, 0}).item<float>(), 1.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(completed.second.descriptors.index({1, 1}).item<float>(), 1.0F, 1.0e-5F);
}

static void trainer_scales_feature_keypoints_to_image_pixel_centers()
{
    auto keypoints = torch::tensor({{0.0F, 0.0F}, {255.0F, 255.0F}}, torch::kFloat32);

    auto image_keypoints = pfm::testing::scale_feature_keypoints_to_image_for_test(keypoints, 256, 256, 1024, 1024);

    PFM_REQUIRE_CLOSE(image_keypoints.index({0, 0}).item<float>(), 1.5F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(image_keypoints.index({0, 1}).item<float>(), 1.5F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(image_keypoints.index({1, 0}).item<float>(), 1021.5F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(image_keypoints.index({1, 1}).item<float>(), 1021.5F, 1.0e-5F);
}

static pfm::SparseHeadOutput make_sparse_orientation_output(const torch::Tensor& orientation)
{
    return pfm::SparseHeadOutput{torch::empty({0}, torch::kFloat32), torch::empty({0}, torch::kFloat32),
                                 torch::empty({0}, torch::kFloat32), orientation.clone().to(torch::kFloat32),
                                 torch::empty({0}, torch::kFloat32)};
}

static void trainer_orientation_supervision_uses_warp_rotation()
{
    auto orientation_a = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    orientation_a.index_put_({0, 0, torch::indexing::Slice(), torch::indexing::Slice()}, 1.0F);
    auto correct_b = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    correct_b.index_put_({0, 1, torch::indexing::Slice(), torch::indexing::Slice()}, 1.0F);
    auto wrong_b = torch::zeros({1, 2, 2, 2}, torch::kFloat32);
    wrong_b.index_put_({0, 0, torch::indexing::Slice(), torch::indexing::Slice()}, -1.0F);

    auto view = torch::ones({1, 1, 4, 4}, torch::kFloat32);
    auto warp = torch::zeros({1, 4, 4, 2}, torch::kFloat32);
    for (int64_t x = 0; x < 4; ++x)
    {
        warp.index_put_({0, torch::indexing::Slice(), x, 1}, static_cast<float>(x));
    }

    auto correct_loss = pfm::testing::make_orientation_supervision_loss_for_test(
        make_sparse_orientation_output(orientation_a), make_sparse_orientation_output(correct_b), view, view, warp,
        0.05);
    auto wrong_loss = pfm::testing::make_orientation_supervision_loss_for_test(
        make_sparse_orientation_output(orientation_a), make_sparse_orientation_output(wrong_b), view, view, warp, 0.05);

    PFM_REQUIRE(correct_loss.item<float>() < 1.0e-4F);
    PFM_REQUIRE(wrong_loss.item<float>() > correct_loss.item<float>() + 0.25F);
}

static void trainer_keypoint_descriptor_loss_covers_more_than_graph_query_limit()
{
    const int64_t count = 300;
    auto keypoints = torch::stack({torch::arange(count, torch::kFloat32), torch::zeros({count}, torch::kFloat32)}, 1);
    auto descriptors_a = torch::zeros({count, 2}, torch::kFloat32);
    auto descriptors_b = torch::zeros({count, 2}, torch::kFloat32);
    descriptors_a.index_put_({torch::indexing::Slice(), 0}, 1.0F);
    descriptors_b.index_put_({torch::indexing::Slice(), 0}, 1.0F);
    descriptors_a.index_put_({count - 1, 0}, 0.0F);
    descriptors_a.index_put_({count - 1, 1}, 1.0F);
    descriptors_b.index_put_({count - 1, 0}, 0.0F);
    descriptors_b.index_put_({count - 1, 1}, 1.0F);

    auto features_a = make_keypoint_descriptor_feature_set(keypoints, descriptors_a);
    auto features_b = make_keypoint_descriptor_feature_set(keypoints, descriptors_b);
    features_a.feature_map_width = count;
    features_b.feature_map_width = count;
    features_b.descriptors.set_requires_grad(true);
    auto warp = torch::zeros({1, 1, count, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(count, torch::kFloat32));
    auto valid_mask = torch::ones({1, 1, count}, torch::kBool);

    auto loss = pfm::testing::make_keypoint_descriptor_loss_for_test(features_a, features_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(features_b.descriptors.grad().defined());
    PFM_REQUIRE(features_b.descriptors.grad().index({count - 1}).abs().sum().item<float>() > 0.0F);
}

static void trainer_graph_matching_loss_trains_graph_matcher_parameters()
{
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    auto descriptors_a = torch::tensor({{{{1.0F, 0.0F}}, {{0.0F, 1.0F}}}}, torch::kFloat32);
    auto descriptors_b = torch::tensor({{{{0.0F, 1.0F}}, {{1.0F, 0.0F}}}}, torch::kFloat32);
    auto warp = torch::tensor({{{{1.0F, 0.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto valid_mask = torch::ones({1, 1, 2}, torch::kBool);

    auto loss =
        pfm::testing::make_graph_matching_loss_for_test(*matcher, descriptors_a, descriptors_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(matcher->parameters().front().grad().defined());
    PFM_REQUIRE(matcher->parameters().front().grad().abs().sum().item<float>() > 0.0F);
}

static void trainer_graph_matching_loss_is_finite_with_many_descriptors()
{
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    auto descriptors_a = torch::ones({1, 2, 1, 1026}, torch::kFloat32);
    auto descriptors_b = torch::ones({1, 2, 1, 1026}, torch::kFloat32);
    auto warp = torch::zeros({1, 1, 1026, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, 1025.0F);
    auto valid_mask = torch::ones({1, 1, 1026}, torch::kBool);

    auto loss =
        pfm::testing::make_graph_matching_loss_for_test(*matcher, descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
}

static void trainer_keypoint_graph_targets_use_warped_nearest_b_keypoint()
{
    auto keypoints_a = torch::tensor({{1.0F, 1.0F}, {3.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{5.0F, 1.0F}, {7.0F, 1.0F}, {1.0F, 6.0F}}, torch::kFloat32);
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 3, 0}, 7.0F);
    warp.index_put_({0, 1, 3, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);

    auto targets =
        pfm::testing::assign_graph_matching_targets_for_test(keypoints_a, keypoints_b, warp, valid_mask, 2.0);

    PFM_REQUIRE(targets.sizes() == std::vector<int64_t>({2}));
    PFM_REQUIRE(targets[0].item<int64_t>() == 0);
    PFM_REQUIRE(targets[1].item<int64_t>() == 1);
}

static void trainer_keypoint_graph_targets_use_dustbin_for_unmatched_keypoints()
{
    auto keypoints_a = torch::tensor({{1.0F, 1.0F}, {3.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{6.0F, 6.0F}}, torch::kFloat32);
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 3, 0}, 7.0F);
    warp.index_put_({0, 1, 3, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);

    auto targets =
        pfm::testing::assign_graph_matching_targets_for_test(keypoints_a, keypoints_b, warp, valid_mask, 1.0);

    PFM_REQUIRE(targets[0].item<int64_t>() == 1);
    PFM_REQUIRE(targets[1].item<int64_t>() == 1);
}

static void trainer_keypoint_graph_targets_use_dustbin_for_invalid_source_pixels()
{
    auto keypoints_a = torch::tensor({{1.0F, 1.0F}}, torch::kFloat32);
    auto keypoints_b = torch::tensor({{5.0F, 1.0F}}, torch::kFloat32);
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);
    valid_mask.index_put_({0, 1, 1}, false);

    auto targets =
        pfm::testing::assign_graph_matching_targets_for_test(keypoints_a, keypoints_b, warp, valid_mask, 2.0);

    PFM_REQUIRE(targets[0].item<int64_t>() == 1);
}

static void trainer_graph_candidates_include_positives_once_and_dustbin_last()
{
    auto target_indices = torch::tensor({0, 2, 2, 5}, torch::kLong);

    auto candidates = pfm::testing::make_graph_candidate_indices_for_test(target_indices, 5, 6);

    PFM_REQUIRE(candidates.size(0) == 6);
    PFM_REQUIRE(candidates[-1].item<int64_t>() == 5);
    PFM_REQUIRE((candidates == 0).sum().item<int64_t>() == 1);
    PFM_REQUIRE((candidates == 2).sum().item<int64_t>() == 1);
    PFM_REQUIRE((candidates == 5).sum().item<int64_t>() == 1);
}

static void trainer_graph_query_sampler_prioritizes_late_positive_targets()
{
    auto target_indices = torch::full({520}, 10, torch::kLong);
    target_indices.index_put_({519}, 3);

    auto query_indices = pfm::testing::make_graph_training_query_indices_for_test(target_indices, 10, 512);

    PFM_REQUIRE(query_indices.size(0) == 512);
    PFM_REQUIRE(query_indices[0].item<int64_t>() == 519);
    PFM_REQUIRE((query_indices == 519).sum().item<int64_t>() == 1);
    PFM_REQUIRE(query_indices.max().item<int64_t>() == 519);
}

static void trainer_graph_query_sampler_keeps_background_when_positives_are_abundant()
{
    auto target_indices = torch::full({800}, 7, torch::kLong);
    target_indices.narrow(0, 0, 600).fill_(3);

    auto query_indices = pfm::testing::make_graph_training_query_indices_for_test(target_indices, 7, 512);
    auto selected_targets = target_indices.index_select(0, query_indices);

    PFM_REQUIRE(query_indices.size(0) == 512);
    PFM_REQUIRE(selected_targets.lt(7).sum().item<int64_t>() == 384);
    PFM_REQUIRE(selected_targets.eq(7).sum().item<int64_t>() == 128);
}

static void trainer_keypoint_graph_matching_loss_trains_graph_matcher_parameters()
{
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    pfm::FeatureSet features_a;
    features_a.keypoints = torch::tensor({{1.0F, 1.0F}, {3.0F, 1.0F}}, torch::kFloat32);
    features_a.scores = torch::tensor({1.0F, 0.9F}, torch::kFloat32);
    features_a.descriptors = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}}, torch::kFloat32);
    features_a.feature_map_width = 8;
    features_a.feature_map_height = 8;
    pfm::FeatureSet features_b;
    features_b.keypoints = torch::tensor({{5.0F, 1.0F}, {7.0F, 1.0F}, {1.0F, 6.0F}}, torch::kFloat32);
    features_b.scores = torch::tensor({1.0F, 0.9F, 0.1F}, torch::kFloat32);
    features_b.descriptors = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {0.5F, 0.5F}}, torch::kFloat32);
    features_b.feature_map_width = 8;
    features_b.feature_map_height = 8;
    auto warp = torch::zeros({1, 8, 8, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    warp.index_put_({0, 1, 3, 0}, 7.0F);
    warp.index_put_({0, 1, 3, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 8}, torch::kBool);

    auto loss =
        pfm::testing::make_keypoint_graph_matching_loss_for_test(*matcher, features_a, features_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
    PFM_REQUIRE(matcher->parameters().front().grad().defined());
    PFM_REQUIRE(matcher->parameters().front().grad().abs().sum().item<float>() > 0.0F);
}

static void trainer_keypoint_graph_matching_loss_uses_full_b_candidate_set()
{
    pfm::PlanetaryGraphMatcher matcher(2, 8, 1);
    pfm::FeatureSet features_a;
    features_a.keypoints = torch::tensor({{1.0F, 1.0F}}, torch::kFloat32);
    features_a.scores = torch::tensor({1.0F}, torch::kFloat32);
    features_a.descriptors = torch::tensor({{1.0F, 0.0F}}, torch::kFloat32);
    features_a.feature_map_width = 80;
    features_a.feature_map_height = 8;

    std::vector<float> keypoint_values;
    std::vector<float> descriptor_values;
    keypoint_values.reserve(70 * 2);
    descriptor_values.reserve(70 * 2);
    for (int index = 0; index < 70; ++index)
    {
        keypoint_values.push_back(static_cast<float>(index + 5));
        keypoint_values.push_back(1.0F);
        descriptor_values.push_back(index == 0 ? 1.0F : 0.0F);
        descriptor_values.push_back(index == 0 ? 0.0F : 1.0F);
    }

    pfm::FeatureSet features_b;
    features_b.keypoints = torch::from_blob(keypoint_values.data(), {70, 2}, torch::kFloat32).clone();
    features_b.scores = torch::ones({70}, torch::kFloat32);
    features_b.descriptors = torch::from_blob(descriptor_values.data(), {70, 2}, torch::kFloat32).clone();
    features_b.descriptors.set_requires_grad(true);
    features_b.feature_map_width = 80;
    features_b.feature_map_height = 8;

    auto warp = torch::zeros({1, 8, 80, 2}, torch::kFloat32);
    warp.index_put_({0, 1, 1, 0}, 5.0F);
    warp.index_put_({0, 1, 1, 1}, 1.0F);
    auto valid_mask = torch::ones({1, 8, 80}, torch::kBool);

    auto loss =
        pfm::testing::make_keypoint_graph_matching_loss_for_test(*matcher, features_a, features_b, warp, valid_mask);
    loss.backward();

    PFM_REQUIRE(features_b.descriptors.grad().defined());
    PFM_REQUIRE(features_b.descriptors.grad().index({69}).abs().sum().item<float>() > 0.0F);
}

static void trainer_graph_loss_prioritizes_decoded_keypoint_pairs()
{
    PFM_REQUIRE(pfm::testing::learned_keypoint_graph_loss_weight_for_test() >
                pfm::testing::warp_completed_keypoint_graph_loss_weight_for_test());
    PFM_REQUIRE(pfm::testing::warp_completed_keypoint_graph_loss_weight_for_test() >
                pfm::testing::supervised_keypoint_graph_loss_weight_for_test());
}

static void trainer_stacks_variable_spatial_training_tensors_with_padding()
{
    auto chw = pfm::testing::stack_chw_batch_for_test(
        {torch::ones({1, 2, 3}, torch::kFloat32), torch::ones({1, 3, 2}, torch::kFloat32) * 2.0F});
    PFM_REQUIRE(chw.sizes() == std::vector<int64_t>({2, 1, 3, 3}));
    PFM_REQUIRE_CLOSE(chw.index({0, 0, 1, 2}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(chw.index({0, 0, 2, 2}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(chw.index({1, 0, 2, 1}).item<float>(), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(chw.index({1, 0, 2, 2}).item<float>(), 0.0F, 1.0e-6F);

    auto hw =
        pfm::testing::stack_hw_batch_for_test({torch::ones({2, 3}, torch::kBool), torch::zeros({3, 2}, torch::kBool)});
    PFM_REQUIRE(hw.sizes() == std::vector<int64_t>({2, 3, 3}));
    PFM_REQUIRE(hw.index({0, 1, 2}).item<bool>());
    PFM_REQUIRE(!hw.index({0, 2, 2}).item<bool>());

    auto warp_a = torch::ones({2, 3, 2}, torch::kFloat32);
    auto warp_b = torch::ones({3, 2, 2}, torch::kFloat32) * 2.0F;
    auto hwc = pfm::testing::stack_hwc_batch_for_test({warp_a, warp_b});
    PFM_REQUIRE(hwc.sizes() == std::vector<int64_t>({2, 3, 3, 2}));
    PFM_REQUIRE_CLOSE(hwc.index({0, 1, 2, 1}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(hwc.index({0, 2, 2, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(hwc.index({1, 2, 1, 1}).item<float>(), 2.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(hwc.index({1, 2, 2, 1}).item<float>(), 0.0F, 1.0e-6F);
}

static void trainer_training_valid_mask_requires_bright_source_and_target_pixels()
{
    auto view_a = torch::tensor({{{{0.0F, 1.0F}, {1.0F, 1.0F}}}}, torch::kFloat32);
    auto view_b = torch::tensor({{{{1.0F, 0.0F}, {1.0F, 1.0F}}}}, torch::kFloat32);
    auto warp = torch::tensor({{{{0.0F, 0.0F}, {1.0F, 0.0F}}, {{0.0F, 1.0F}, {1.0F, 1.0F}}}}, torch::kFloat32);
    auto valid_mask = torch::ones({1, 2, 2}, torch::kBool);

    auto masked = pfm::testing::make_training_valid_mask_for_test(view_a, view_b, warp, valid_mask, 0.5);
    auto expected = torch::tensor({{{false, false}, {true, true}}}, torch::kBool);

    PFM_REQUIRE(torch::equal(masked, expected));
}

static void trainer_python_compare_pair_loss_mask_keeps_python_center_intensity_samples()
{
    auto view_a = torch::zeros({1, 1, 9, 9}, torch::kFloat32);
    auto view_b = torch::zeros({1, 1, 9, 9}, torch::kFloat32);
    view_a.index_put_({0, 0, 4, 4}, 1.0F);
    view_b.index_put_({0, 0, 4, 4}, 1.0F);

    auto warp = torch::zeros({1, 9, 9, 2}, torch::kFloat32);
    for (int64_t y = 0; y < 9; ++y)
    {
        for (int64_t x = 0; x < 9; ++x)
        {
            warp.index_put_({0, y, x, 0}, static_cast<float>(x));
            warp.index_put_({0, y, x, 1}, static_cast<float>(y));
        }
    }
    auto valid_mask = torch::zeros({1, 9, 9}, torch::kBool);
    valid_mask.index_put_({0, 4, 4}, true);

    const auto python_mask =
        pfm::testing::make_pair_loss_valid_mask_for_test(view_a, view_b, warp, valid_mask, 0.1, "python-compare");
    const auto descriptor_mask =
        pfm::testing::make_pair_loss_valid_mask_for_test(view_a, view_b, warp, valid_mask, 0.1, "descriptor");

    PFM_REQUIRE(python_mask.index({0, 4, 4}).item<bool>());
    PFM_REQUIRE(!descriptor_mask.index({0, 4, 4}).item<bool>());
}

static void trainer_python_compare_profile_skips_dense_quality_forward()
{
    PFM_REQUIRE(!pfm::testing::training_profile_uses_dense_quality_forward_for_test("python-compare"));
    PFM_REQUIRE(!pfm::testing::training_profile_uses_dense_quality_forward_for_test("full"));
    PFM_REQUIRE(pfm::testing::training_profile_uses_dense_quality_forward_for_test("descriptor"));
    PFM_REQUIRE(pfm::testing::training_profile_uses_dense_quality_forward_for_test("graph"));
}

static void trainer_python_compare_trainable_parameters_match_python_defaults()
{
    pfm::TrainConfig config;
    config.training_profile = "python-compare";
    config.graph_matcher_loss_weight = 1.0;

    const auto names = pfm::testing::trainable_parameter_names_for_config_for_test(config);

    PFM_REQUIRE(has_trainable_parameter_prefix(names, "sparse_head.descriptors."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "graph_matcher."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "backbone."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "dual_fpn."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "texture_adapter."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "descriptor_fusion."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "dense_head."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "quality_head."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "semi_dense_branch."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "sparse_head.heatmap."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "sparse_head.scale."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "sparse_head.orientation."));
    PFM_REQUIRE(!has_trainable_parameter_prefix(names, "sparse_head.affine."));
}

static void trainer_python_compare_trainable_parameters_train_graph_when_requested()
{
    pfm::TrainConfig config;
    config.training_profile = "python-compare";
    config.train_graph_matcher = true;

    const auto names = pfm::testing::trainable_parameter_names_for_config_for_test(config);

    PFM_REQUIRE(has_trainable_parameter_prefix(names, "sparse_head.descriptors."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "graph_matcher."));
}

static void trainer_python_compare_trainable_parameters_follow_python_full_flags()
{
    pfm::TrainConfig config;
    config.training_profile = "python-compare";
    config.train_backbone = true;
    config.train_dual_fpn = true;
    config.train_sparse_context = true;
    config.train_geometry_head = true;
    config.train_blended_descriptors = true;
    config.train_texture_adapter = true;
    config.train_descriptor_fusion = true;
    config.train_quality_head = true;
    config.train_graph_matcher = true;

    const auto names = pfm::testing::trainable_parameter_names_for_config_for_test(config);

    PFM_REQUIRE(has_trainable_parameter_prefix(names, "backbone."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "dual_fpn."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "sparse_head.descriptors."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "sparse_head.descriptor_context."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "sparse_head.scale."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "sparse_head.orientation."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "sparse_head.affine."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "texture_adapter."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "descriptor_fusion."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "quality_head."));
    PFM_REQUIRE(has_trainable_parameter_prefix(names, "graph_matcher."));
}

static void trainer_python_compare_graph_loss_disables_candidate_mask_for_supervision()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16, 1);
    matcher->eval();

    auto descriptors_a = torch::zeros({3, 8}, torch::kFloat32);
    auto descriptors_b = torch::zeros({3, 8}, torch::kFloat32);
    descriptors_a.index_put_({0, 0}, 1.0F);
    descriptors_a.index_put_({1, 0}, -1.0F);
    descriptors_a.index_put_({2, 0}, -1.0F);
    descriptors_b.index_put_({0, 0}, -1.0F);
    descriptors_b.index_put_({1, 0}, 1.0F);
    descriptors_b.index_put_({2, 0}, 1.0F);
    auto points = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}}, torch::kFloat32);

    const auto loss = pfm::testing::make_python_compare_graph_loss_for_test(*matcher, descriptors_a, descriptors_b,
                                                                            points, points, 16);

    PFM_REQUIRE(loss.item<float>() < 100.0F);
}

static void trainer_python_compare_graph_loss_can_train_accept_head()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16, 1);
    matcher->train();

    auto descriptors_a = torch::eye(3, 8, torch::kFloat32);
    auto descriptors_b = descriptors_a.clone();
    auto points = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}}, torch::kFloat32);

    const auto loss = pfm::testing::make_python_compare_graph_loss_for_test(*matcher, descriptors_a, descriptors_b,
                                                                            points, points, 16, 0.5);
    loss.backward();

    bool saw_accept_grad = false;
    for (const auto& parameter : matcher->named_parameters())
    {
        if (parameter.key() == "accept_head.2.weight")
        {
            saw_accept_grad = true;
            PFM_REQUIRE(parameter.value().grad().defined());
            PFM_REQUIRE(parameter.value().grad().abs().sum().item<float>() > 0.0F);
        }
    }
    PFM_REQUIRE(saw_accept_grad);
}

static void trainer_python_compare_graph_loss_penalizes_unmatched_accept_logits()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16, 0);
    matcher->eval();
    torch::NoGradGuard no_grad;

    for (auto& parameter : matcher->named_parameters(true))
    {
        if (parameter.key().rfind("accept_head.", 0) == 0)
        {
            parameter.value().zero_();
        }
        if (parameter.key() == "accept_head.2.bias")
        {
            parameter.value().fill_(4.0F);
        }
    }

    auto descriptors_balanced = torch::eye(1, 8, torch::kFloat32);
    auto descriptors_extra = torch::eye(2, 8, torch::kFloat32);
    auto points_balanced = torch::tensor({{0.0F, 0.0F}}, torch::kFloat32);
    auto points_extra = torch::tensor({{0.0F, 0.0F}, {2.0F, 0.0F}}, torch::kFloat32);

    const auto balanced_loss = pfm::testing::make_python_compare_graph_loss_for_test(
        *matcher, descriptors_balanced, descriptors_balanced, points_balanced, points_balanced, 16, 1.0);
    const auto extra_loss = pfm::testing::make_python_compare_graph_loss_for_test(
        *matcher, descriptors_extra, descriptors_balanced, points_extra, points_balanced, 16, 1.0);

    PFM_REQUIRE(extra_loss.item<float>() > balanced_loss.item<float>() + 1.0F);
}

static void trainer_python_compare_graph_loss_respects_attention_layer_budget()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 3, 16, 1);
    matcher->train();

    auto descriptors_a = torch::eye(4, 8, torch::kFloat32);
    auto descriptors_b = descriptors_a.clone();
    auto points = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}, torch::kFloat32);

    const auto loss = pfm::testing::make_python_compare_graph_loss_with_attention_budget_for_test(
        *matcher, descriptors_a, descriptors_b, points, points, 16, 1);

    PFM_REQUIRE(torch::isfinite(loss).all().item<bool>());
    PFM_REQUIRE(matcher->lastExecutedAttentionLayers() == 1);
}

static void trainer_python_compare_graph_loss_can_randomize_attention_layer_budget()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 3, 16, 1);
    matcher->train();

    auto descriptors_a = torch::eye(4, 8, torch::kFloat32);
    auto descriptors_b = descriptors_a.clone();
    auto points = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}, torch::kFloat32);

    const auto loss = pfm::testing::make_python_compare_graph_loss_with_random_attention_budget_for_test(
        *matcher, descriptors_a, descriptors_b, points, points, 16, 3, 2);

    PFM_REQUIRE(torch::isfinite(loss).all().item<bool>());
    PFM_REQUIRE(matcher->lastExecutedAttentionLayers() == 1);
}

static void trainer_python_compare_graph_loss_respects_attention_work_budget()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 2, 16, 1);
    matcher->train();

    auto descriptors_a = torch::eye(3, 8, torch::kFloat32);
    auto descriptors_b = descriptors_a.clone();
    auto points = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}}, torch::kFloat32);

    const auto loss = pfm::testing::make_python_compare_graph_loss_with_attention_work_fraction_for_test(
        *matcher, descriptors_a, descriptors_b, points, points, 16, 0.5);

    PFM_REQUIRE(torch::isfinite(loss).all().item<bool>());
    PFM_REQUIRE(matcher->lastExecutedAttentionLayers() == 1);
}

static void trainer_python_compare_graph_metadata_mode_matches_python_masks()
{
    auto points = torch::tensor({{0.0F, 0.0F}, {2.0F, 1.0F}, {4.0F, 3.0F}}, torch::kFloat32);

    const auto full = pfm::testing::make_python_compare_graph_metadata_for_test(points, 16, "full");
    const auto descriptor_only =
        pfm::testing::make_python_compare_graph_metadata_for_test(points, 16, "descriptor_only");
    const auto no_xy = pfm::testing::make_python_compare_graph_metadata_for_test(points, 16, "no_xy");
    const auto no_geometry = pfm::testing::make_python_compare_graph_metadata_for_test(points, 16, "no_geometry");
    const auto no_quality = pfm::testing::make_python_compare_graph_metadata_for_test(points, 16, "no_quality");

    PFM_REQUIRE(torch::allclose(descriptor_only, torch::zeros_like(descriptor_only)));
    PFM_REQUIRE(torch::allclose(no_xy.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)}),
                                torch::zeros({points.size(0), 4}, no_xy.options())));
    PFM_REQUIRE(torch::allclose(no_xy.index({torch::indexing::Slice(), torch::indexing::Slice(4, 16)}),
                                full.index({torch::indexing::Slice(), torch::indexing::Slice(4, 16)})));
    PFM_REQUIRE(torch::allclose(no_geometry.index({torch::indexing::Slice(), torch::indexing::Slice(0, 5)}),
                                full.index({torch::indexing::Slice(), torch::indexing::Slice(0, 5)})));
    PFM_REQUIRE(torch::allclose(no_geometry.index({torch::indexing::Slice(), torch::indexing::Slice(5, 12)}),
                                torch::zeros({points.size(0), 7}, no_geometry.options())));
    PFM_REQUIRE(torch::allclose(no_geometry.index({torch::indexing::Slice(), torch::indexing::Slice(12, 16)}),
                                full.index({torch::indexing::Slice(), torch::indexing::Slice(12, 16)})));
    PFM_REQUIRE(torch::allclose(no_quality.index({torch::indexing::Slice(), torch::indexing::Slice(0, 12)}),
                                full.index({torch::indexing::Slice(), torch::indexing::Slice(0, 12)})));
    PFM_REQUIRE(torch::allclose(no_quality.index({torch::indexing::Slice(), torch::indexing::Slice(12, 16)}),
                                torch::zeros({points.size(0), 4}, no_quality.options())));
}

static void trainer_python_compare_graph_loss_can_train_with_width_dropout()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16, 1);
    matcher->train();

    auto descriptors_a = torch::eye(6, 8, torch::kFloat32);
    auto descriptors_b = descriptors_a.clone();
    auto points =
        torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 1.0F}},
                      torch::kFloat32);

    const auto result = pfm::testing::make_python_compare_graph_loss_with_width_keep_ratio_for_test(
        *matcher, descriptors_a, descriptors_b, points, points, 16, 0.5, 20260606);

    PFM_REQUIRE(torch::isfinite(result.first).all().item<bool>());
    PFM_REQUIRE(result.second == 3);
}

static void trainer_python_compare_graph_loss_can_train_prune_ranking_accept_head()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16, 1);
    matcher->train();

    auto descriptors_a = torch::eye(4, 8, torch::kFloat32);
    auto descriptors_b = descriptors_a.clone();
    auto points = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}, torch::kFloat32);

    const auto loss = pfm::testing::make_python_compare_graph_loss_for_test(*matcher, descriptors_a, descriptors_b,
                                                                            points, points, 16, 0.0, 0.5, 0.25);
    loss.backward();

    bool saw_accept_grad = false;
    for (const auto& parameter : matcher->named_parameters())
    {
        if (parameter.key() == "accept_head.2.weight")
        {
            saw_accept_grad = true;
            PFM_REQUIRE(parameter.value().grad().defined());
            PFM_REQUIRE(parameter.value().grad().abs().sum().item<float>() > 0.0F);
        }
    }
    PFM_REQUIRE(saw_accept_grad);
}

static void trainer_python_compare_graph_loss_can_train_stop_confidence_score_path()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16, 1);
    matcher->train();

    auto descriptors_a = torch::eye(4, 8, torch::kFloat32);
    auto descriptors_b = descriptors_a.clone();
    auto points = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}, torch::kFloat32);

    const auto loss = pfm::testing::make_python_compare_graph_loss_for_test(*matcher, descriptors_a, descriptors_b,
                                                                            points, points, 16, 0.0, 0.0, 0.25,
                                                                            0.5);
    loss.backward();

    bool saw_score_grad = false;
    for (const auto& parameter : matcher->named_parameters())
    {
        if (parameter.key() == "score_projection.weight")
        {
            saw_score_grad = true;
            PFM_REQUIRE(parameter.value().grad().defined());
            PFM_REQUIRE(parameter.value().grad().abs().sum().item<float>() > 0.0F);
        }
    }
    PFM_REQUIRE(saw_score_grad);
}

static void trainer_python_compare_graph_loss_adds_raw_preservation_margin()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16, 1);
    matcher->eval();
    torch::manual_seed(20260606);

    auto descriptors_a = torch::eye(4, 8, torch::kFloat32);
    auto descriptors_b = descriptors_a.clone();
    auto points = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}, torch::kFloat32);

    const auto base_loss = pfm::testing::make_python_compare_graph_loss_for_test(*matcher, descriptors_a,
                                                                                 descriptors_b, points, points, 16);
    const auto preserved_loss = pfm::testing::make_python_compare_graph_loss_with_raw_preservation_for_test(
        *matcher, descriptors_a, descriptors_b, points, points, 16, 1.0, 100.0, 0.05);

    PFM_REQUIRE(preserved_loss.item<float>() > base_loss.item<float>() + 1.0F);
}

static void trainer_python_compare_graph_loss_adds_hard_negative_dustbin_margin()
{
    auto matcher = pfm::v21::PfmV21GraphMatcher(8, 16, 1, 16, 1);
    matcher->eval();
    torch::manual_seed(20260606);

    auto descriptors_a = torch::eye(4, 8, torch::kFloat32);
    auto descriptors_b = descriptors_a.clone();
    auto points = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}, torch::kFloat32);

    const auto base_loss = pfm::testing::make_python_compare_graph_loss_for_test(*matcher, descriptors_a,
                                                                                 descriptors_b, points, points, 16);
    const auto dustbin_loss = pfm::testing::make_python_compare_graph_loss_with_hard_negative_dustbin_for_test(
        *matcher, descriptors_a, descriptors_b, points, points, 16, 1.0, 2, 8.0, 0.0);

    PFM_REQUIRE(dustbin_loss.item<float>() > base_loss.item<float>() + 1.0F);
}

static void trainer_descriptor_candidates_do_not_repeat_positive_target()
{
    auto target_indices = torch::tensor({{0, 4}}, torch::kLong);

    auto candidates = pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, 5);

    PFM_REQUIRE(candidates.size(2) == 5);
    PFM_REQUIRE(candidates.index({0, 0, 0}).item<int64_t>() == 0);
    PFM_REQUIRE(candidates.index({0, 1, 0}).item<int64_t>() == 4);
    PFM_REQUIRE(candidates.index({0, 0, 1}).item<int64_t>() != 0);
    PFM_REQUIRE(candidates.index({0, 0, 2}).item<int64_t>() != 0);
    PFM_REQUIRE(candidates.index({0, 0, 3}).item<int64_t>() != 0);
    PFM_REQUIRE(candidates.index({0, 0, 4}).item<int64_t>() != 0);
    PFM_REQUIRE(candidates.index({0, 1, 1}).item<int64_t>() != 4);
    PFM_REQUIRE(candidates.index({0, 1, 2}).item<int64_t>() != 4);
    PFM_REQUIRE(candidates.index({0, 1, 3}).item<int64_t>() != 4);
    PFM_REQUIRE(candidates.index({0, 1, 4}).item<int64_t>() != 4);
}

static void trainer_descriptor_candidates_exclude_spatial_near_positives()
{
    constexpr int64_t width = 8;
    auto target_indices = torch::tensor({{27}}, torch::kLong);

    auto candidates = pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, width * width);

    const auto positive_x = target_indices.item<int64_t>() % width;
    const auto positive_y = target_indices.item<int64_t>() / width;
    for (int64_t candidate = 1; candidate < candidates.size(2); ++candidate)
    {
        const auto index = candidates.index({0, 0, candidate}).item<int64_t>();
        const auto x = index % width;
        const auto y = index / width;
        const auto dx = x - positive_x;
        const auto dy = y - positive_y;
        PFM_REQUIRE(dx * dx + dy * dy > 4);
    }
}

static void trainer_descriptor_candidates_prioritize_near_ring_hard_negatives()
{
    constexpr int64_t width = 16;
    auto target_indices = torch::tensor({{8 * width + 8}}, torch::kLong);

    auto candidates = pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, width * width);

    const auto target = target_indices.item<int64_t>();
    const auto positive_x = target % width;
    const auto positive_y = target / width;
    const auto first_negative = candidates.index({0, 0, 1}).item<int64_t>();
    const auto first_x = first_negative % width;
    const auto first_y = first_negative / width;
    const auto first_dx = first_x - positive_x;
    const auto first_dy = first_y - positive_y;
    const auto first_distance_sq = first_dx * first_dx + first_dy * first_dy;

    PFM_REQUIRE(first_distance_sq > 4);
    PFM_REQUIRE(first_distance_sq <= 36);
}

static void trainer_descriptor_candidates_cover_broad_far_negative_regions()
{
    constexpr int64_t width = 32;
    auto target_indices = torch::tensor({{16 * width + 16}}, torch::kLong);

    auto candidates = pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, width * width);

    bool has_bottom_right_far_negative = false;
    for (int64_t candidate = 1; candidate < candidates.size(2); ++candidate)
    {
        const auto index = candidates.index({0, 0, candidate}).item<int64_t>();
        const auto x = index % width;
        const auto y = index / width;
        if (x >= 24 && y >= 24)
        {
            has_bottom_right_far_negative = true;
            break;
        }
    }
    PFM_REQUIRE(has_bottom_right_far_negative);
}

static void trainer_descriptor_candidates_can_disable_broad_far_negatives_for_curriculum()
{
    constexpr int64_t width = 32;
    auto target_indices = torch::tensor({{16 * width + 16}}, torch::kLong);

    auto early_candidates = pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, width * width, 0);
    auto late_candidates = pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, width * width, 16);

    auto has_bottom_right = [&](const torch::Tensor& candidates)
    {
        for (int64_t candidate = 1; candidate < candidates.size(2); ++candidate)
        {
            const auto index = candidates.index({0, 0, candidate}).item<int64_t>();
            const auto x = index % width;
            const auto y = index / width;
            if (x >= 24 && y >= 24)
            {
                return true;
            }
        }
        return false;
    };

    PFM_REQUIRE(!has_bottom_right(early_candidates));
    PFM_REQUIRE(has_bottom_right(late_candidates));
}

static void trainer_descriptor_candidates_curriculum_limits_early_pool_to_near_ring()
{
    constexpr int64_t width = 32;
    auto target_indices = torch::tensor({{16 * width + 16}}, torch::kLong);

    auto early_candidates = pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, width * width, 0);
    auto late_candidates = pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, width * width, 16);

    const auto target = target_indices.item<int64_t>();
    const auto target_x = target % width;
    const auto target_y = target / width;
    for (int64_t candidate = 1; candidate < early_candidates.size(2); ++candidate)
    {
        const auto index = early_candidates.index({0, 0, candidate}).item<int64_t>();
        const auto x = index % width;
        const auto y = index / width;
        const auto dx = x - target_x;
        const auto dy = y - target_y;
        PFM_REQUIRE(dx * dx + dy * dy <= 36);
    }
    PFM_REQUIRE(early_candidates.size(2) < late_candidates.size(2));
}

static void trainer_descriptor_broad_far_negative_curriculum_ramps_with_progress()
{
    const auto early = pfm::testing::descriptor_broad_far_negative_count_for_progress_for_test(0.0);
    const auto middle = pfm::testing::descriptor_broad_far_negative_count_for_progress_for_test(0.5);
    const auto late = pfm::testing::descriptor_broad_far_negative_count_for_progress_for_test(1.0);

    PFM_REQUIRE(early == 0);
    PFM_REQUIRE(middle > early);
    PFM_REQUIRE(late > middle);
}

static void trainer_descriptor_broad_far_negative_curriculum_caps_extreme_ce_pool()
{
    const auto late = pfm::testing::descriptor_broad_far_negative_count_for_progress_for_test(1.0);

    PFM_REQUIRE(late <= 8);
}

static void trainer_descriptor_ranking_loss_scans_late_hard_negatives()
{
    auto sampled_a = torch::tensor({{{1.0F, 0.0F}}}, torch::kFloat32);
    auto candidates = torch::zeros({1, 1, 40, 2}, torch::kFloat32);
    candidates.index_put_({0, 0, 0}, torch::tensor({1.0F, 0.0F}));
    for (int64_t candidate = 1; candidate < 39; ++candidate)
    {
        candidates.index_put_({0, 0, candidate}, torch::tensor({0.0F, 1.0F}));
    }
    candidates.index_put_({0, 0, 39}, torch::tensor({1.0F, 0.0F}));

    auto loss = pfm::testing::make_supervised_descriptor_ranking_loss_for_test(sampled_a, candidates);

    PFM_REQUIRE(loss.item<float>() > 0.1F);
}

static void trainer_descriptor_ranking_loss_weights_multiple_hard_negatives()
{
    auto sampled_a = torch::tensor({{{1.0F, 0.0F}}}, torch::kFloat32);
    auto one_hard = torch::zeros({1, 1, 9, 2}, torch::kFloat32);
    auto many_hard = torch::zeros({1, 1, 9, 2}, torch::kFloat32);
    one_hard.index_put_({0, 0, 0}, torch::tensor({1.0F, 0.0F}));
    many_hard.index_put_({0, 0, 0}, torch::tensor({1.0F, 0.0F}));
    for (int64_t candidate = 1; candidate < 9; ++candidate)
    {
        one_hard.index_put_({0, 0, candidate}, torch::tensor({0.0F, 1.0F}));
        many_hard.index_put_({0, 0, candidate}, torch::tensor({0.0F, 1.0F}));
    }
    one_hard.index_put_({0, 0, 8}, torch::tensor({0.9F, 0.1F}));
    for (int64_t candidate = 5; candidate < 9; ++candidate)
    {
        many_hard.index_put_({0, 0, candidate}, torch::tensor({0.9F, 0.1F}));
    }

    auto one_hard_loss = pfm::testing::make_supervised_descriptor_ranking_loss_for_test(sampled_a, one_hard);
    auto many_hard_loss = pfm::testing::make_supervised_descriptor_ranking_loss_for_test(sampled_a, many_hard);

    PFM_REQUIRE(many_hard_loss.item<float>() > one_hard_loss.item<float>() + 0.05F);
}

static void trainer_descriptor_ranking_loss_penalizes_many_near_top_distractors()
{
    auto sampled_a = torch::tensor({{{1.0F, 0.0F}}}, torch::kFloat32);
    auto one_near = torch::zeros({1, 1, 33, 2}, torch::kFloat32);
    auto many_near = torch::zeros({1, 1, 33, 2}, torch::kFloat32);
    one_near.index_put_({0, 0, 0}, torch::tensor({1.0F, 0.0F}));
    many_near.index_put_({0, 0, 0}, torch::tensor({1.0F, 0.0F}));
    for (int64_t candidate = 1; candidate < 33; ++candidate)
    {
        one_near.index_put_({0, 0, candidate}, torch::tensor({0.0F, 1.0F}));
        many_near.index_put_({0, 0, candidate}, torch::tensor({0.0F, 1.0F}));
    }
    one_near.index_put_({0, 0, 1}, torch::tensor({0.9F, 0.1F}));
    for (int64_t candidate = 1; candidate < 17; ++candidate)
    {
        many_near.index_put_({0, 0, candidate}, torch::tensor({0.9F, 0.1F}));
    }

    auto one_near_loss = pfm::testing::make_supervised_descriptor_ranking_loss_for_test(sampled_a, one_near);
    auto many_near_loss = pfm::testing::make_supervised_descriptor_ranking_loss_for_test(sampled_a, many_near);

    PFM_REQUIRE(many_near_loss.item<float>() > one_near_loss.item<float>() + 0.5F);
    PFM_REQUIRE(pfm::testing::supervised_descriptor_soft_rank_weight_for_test() > 0.0);
}

static void trainer_descriptor_ranking_loss_has_tail_rank_penalty()
{
    PFM_REQUIRE(pfm::testing::supervised_descriptor_tail_rank_weight_for_test() > 0.0);
}

static void trainer_descriptor_ranking_loss_scans_broad_cross_view_hard_negative_set()
{
    PFM_REQUIRE(pfm::testing::supervised_descriptor_topk_negatives_for_test() >= 64);
}

static void trainer_descriptor_candidates_scan_wide_cross_view_negative_pool()
{
    PFM_REQUIRE(pfm::testing::descriptor_negative_sample_count_for_test() >= 127);
}

static void trainer_descriptor_global_ce_has_cross_view_weight()
{
    PFM_REQUIRE(pfm::testing::descriptor_global_ce_weight_for_test() >= 0.25);
}

static void trainer_descriptor_ranking_step_reduces_hard_negative_score()
{
    auto sampled_a = torch::tensor({{{0.2F, 0.98F}}}, torch::kFloat32).set_requires_grad(true);
    auto candidates = torch::zeros({1, 1, 3, 2}, torch::kFloat32);
    candidates.index_put_({0, 0, 0}, torch::tensor({1.0F, 0.0F}));
    candidates.index_put_({0, 0, 1}, torch::tensor({0.0F, 1.0F}));
    candidates.index_put_({0, 0, 2}, torch::tensor({0.1F, 0.99F}));
    auto before = pfm::testing::descriptor_candidate_similarity_scores_for_test(sampled_a.detach(), candidates);

    auto loss = pfm::testing::make_supervised_descriptor_ranking_loss_for_test(sampled_a, candidates);
    loss.backward();
    auto updated = sampled_a - sampled_a.grad() * 0.2F;
    auto after = pfm::testing::descriptor_candidate_similarity_scores_for_test(updated.detach(), candidates);

    PFM_REQUIRE(after.index({0, 0, 0}).item<float>() > before.index({0, 0, 0}).item<float>());
    PFM_REQUIRE(after.index({0, 0, 2}).item<float>() < before.index({0, 0, 2}).item<float>());
}

static void trainer_sampled_descriptor_decorrelation_penalizes_far_duplicate_descriptors()
{
    auto duplicate = torch::zeros({1, 4, 2}, torch::kFloat32);
    duplicate.index_put_({0, torch::indexing::Slice(), 0}, 1.0F);
    auto orthogonal = torch::eye(4, torch::kFloat32).unsqueeze(0);
    auto sample_indices = torch::tensor({0, 6, 12, 18}, torch::kLong);

    auto duplicate_loss =
        pfm::testing::make_sampled_descriptor_decorrelation_loss_for_test(duplicate, sample_indices, 5);
    auto orthogonal_loss =
        pfm::testing::make_sampled_descriptor_decorrelation_loss_for_test(orthogonal, sample_indices, 5);

    PFM_REQUIRE(duplicate_loss.item<float>() > orthogonal_loss.item<float>() + 0.5F);
    PFM_REQUIRE(orthogonal_loss.item<float>() < 1.0e-5F);
}

static void trainer_sampled_descriptor_decorrelation_ignores_nearby_descriptors()
{
    auto duplicate = torch::zeros({1, 2, 2}, torch::kFloat32);
    duplicate.index_put_({0, torch::indexing::Slice(), 0}, 1.0F);
    auto nearby_indices = torch::tensor({0, 1}, torch::kLong);

    auto loss = pfm::testing::make_sampled_descriptor_decorrelation_loss_for_test(duplicate, nearby_indices, 5);

    PFM_REQUIRE(loss.item<float>() < 1.0e-5F);
}

static void trainer_positive_descriptor_alignment_step_increases_positive_cosine()
{
    auto sampled_a = torch::tensor({{{1.0F, 0.0F}}}, torch::kFloat32).set_requires_grad(true);
    auto positive_b = torch::tensor({{{0.0F, 1.0F}}}, torch::kFloat32);
    auto before = (sampled_a.detach() / sampled_a.detach().norm(2, 2, true).clamp_min(1.0e-12) * positive_b /
                   positive_b.norm(2, 2, true).clamp_min(1.0e-12))
                      .sum(2)
                      .item<float>();

    auto loss = pfm::testing::make_positive_descriptor_alignment_loss_for_test(sampled_a, positive_b);
    loss.backward();
    auto updated = sampled_a - sampled_a.grad() * 0.25F;
    auto after = (updated.detach() / updated.detach().norm(2, 2, true).clamp_min(1.0e-12) * positive_b /
                  positive_b.norm(2, 2, true).clamp_min(1.0e-12))
                     .sum(2)
                     .item<float>();

    PFM_REQUIRE(after > before + 0.1F);
}

static void trainer_patch_descriptor_alignment_uses_local_neighborhood()
{
    auto descriptors_a = torch::zeros({1, 2, 3, 3}, torch::kFloat32);
    auto clean_b = torch::zeros({1, 2, 3, 3}, torch::kFloat32);
    auto bad_b = torch::zeros({1, 2, 3, 3}, torch::kFloat32);
    descriptors_a.index_put_({0, 0, 1, 1}, 1.0F);
    clean_b.index_put_({0, 0, 0, 0}, 1.0F);
    bad_b.index_put_({0, 1, 0, 0}, 1.0F);
    auto warp = torch::zeros({1, 3, 3, 2}, torch::kFloat32);
    auto xy = torch::meshgrid({torch::arange(3, torch::kFloat32), torch::arange(3, torch::kFloat32)}, "ij");
    warp.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(), 0}, xy[1]);
    warp.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(), 1}, xy[0]);
    auto valid_mask = torch::zeros({1, 3, 3}, torch::kBool);
    valid_mask.index_put_({0, 1, 1}, true);

    auto clean_loss =
        pfm::testing::make_patch_descriptor_alignment_loss_for_test(descriptors_a, clean_b, warp, valid_mask);
    auto bad_loss = pfm::testing::make_patch_descriptor_alignment_loss_for_test(descriptors_a, bad_b, warp, valid_mask);

    PFM_REQUIRE(clean_loss.item<float>() < bad_loss.item<float>() - 0.5F);
}

static void trainer_patch_descriptor_alignment_uses_wider_viewpoint_neighborhood()
{
    auto descriptors_a = torch::zeros({1, 2, 5, 5}, torch::kFloat32);
    auto clean_b = torch::zeros({1, 2, 5, 5}, torch::kFloat32);
    auto bad_b = torch::zeros({1, 2, 5, 5}, torch::kFloat32);
    descriptors_a.index_put_({0, 0, 2, 2}, 1.0F);
    clean_b.index_put_({0, 0, 0, 0}, 1.0F);
    bad_b.index_put_({0, 1, 0, 0}, 1.0F);
    auto warp = torch::zeros({1, 5, 5, 2}, torch::kFloat32);
    auto xy = torch::meshgrid({torch::arange(5, torch::kFloat32), torch::arange(5, torch::kFloat32)}, "ij");
    warp.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(), 0}, xy[1]);
    warp.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(), 1}, xy[0]);
    auto valid_mask = torch::zeros({1, 5, 5}, torch::kBool);
    valid_mask.index_put_({0, 2, 2}, true);

    auto clean_loss =
        pfm::testing::make_patch_descriptor_alignment_loss_for_test(descriptors_a, clean_b, warp, valid_mask);
    auto bad_loss = pfm::testing::make_patch_descriptor_alignment_loss_for_test(descriptors_a, bad_b, warp, valid_mask);

    PFM_REQUIRE(clean_loss.item<float>() < bad_loss.item<float>() - 0.25F);
}

static void trainer_descriptor_training_similarity_rejects_channel_shifted_negative()
{
    auto sampled_a = torch::tensor({{{1.0F, 0.0F, 0.0F, 0.0F}}}, torch::kFloat32);
    auto candidates = torch::tensor({{{{1.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F, 0.0F}}}}, torch::kFloat32);

    auto scores = pfm::testing::descriptor_candidate_similarity_scores_for_test(sampled_a, candidates);

    PFM_REQUIRE(scores.index({0, 0, 0}).item<float>() > scores.index({0, 0, 1}).item<float>() + 0.5F);
}

static void trainer_strict_descriptor_ce_rejects_channel_shifted_negative()
{
    auto query = torch::tensor({{{1.0F, 0.0F, 0.0F, 0.0F}}}, torch::kFloat32);
    auto candidates = torch::tensor({{{1.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F, 0.0F}}}, torch::kFloat32);
    auto labels = torch::zeros({1, 1}, torch::kLong);

    auto loss = pfm::testing::make_strict_descriptor_cross_entropy_loss_for_test(query, candidates, labels);

    PFM_REQUIRE(loss.item<float>() < 0.1F);
}

static void trainer_descriptor_candidates_skip_invalid_target_regions()
{
    constexpr int64_t width = 8;
    auto target_indices = torch::tensor({{27}}, torch::kLong);
    auto candidate_valid = torch::ones({1, width * width}, torch::kBool);
    candidate_valid.index_put_({0, 0}, false);

    auto candidates =
        pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, width * width, candidate_valid);

    for (int64_t candidate = 1; candidate < candidates.size(2); ++candidate)
    {
        PFM_REQUIRE(candidates.index({0, 0, candidate}).item<int64_t>() != 0);
    }
}

static void trainer_descriptor_candidates_repeat_valid_regions_instead_of_invalid_fill()
{
    constexpr int64_t width = 8;
    auto target_indices = torch::tensor({{27}}, torch::kLong);
    auto candidate_valid = torch::zeros({1, width * width}, torch::kBool);
    candidate_valid.index_put_({0, 27}, true);
    candidate_valid.index_put_({0, 63}, true);

    auto candidates =
        pfm::testing::make_descriptor_candidate_indices_for_test(target_indices, width * width, candidate_valid);

    PFM_REQUIRE(candidates.size(2) == 2);
    PFM_REQUIRE(candidates.index({0, 0, 1}).item<int64_t>() == 63);
}

static void trainer_texture_blend_preserves_learned_positive_margin()
{
    auto descriptors_a = torch::zeros({1, 4, 2, 2}, torch::kFloat32);
    auto descriptors_b = torch::zeros({1, 4, 2, 2}, torch::kFloat32);
    descriptors_a.index_put_({0, 0, 0, 0}, 1.0F);
    descriptors_b.index_put_({0, 0, 0, 0}, 1.0F);
    descriptors_b.index_put_({0, 1, 1, 1}, 1.0F);

    auto image_a = torch::tensor({{{{1.0F, 0.0F}, {0.0F, 0.0F}}}}, torch::kFloat32);
    auto image_b = torch::tensor({{{{0.0F, 0.0F}, {0.0F, 1.0F}}}}, torch::kFloat32);

    auto blended_a = pfm::testing::blend_rotation_invariant_texture_descriptor_for_test(descriptors_a, image_a);
    auto blended_b = pfm::testing::blend_rotation_invariant_texture_descriptor_for_test(descriptors_b, image_b);

    auto positive =
        (blended_a.index({0, torch::indexing::Slice(), 0, 0}) * blended_b.index({0, torch::indexing::Slice(), 0, 0}))
            .sum();
    auto negative =
        (blended_a.index({0, torch::indexing::Slice(), 0, 0}) * blended_b.index({0, torch::indexing::Slice(), 1, 1}))
            .sum();

    PFM_REQUIRE(positive.item<float>() > negative.item<float>() + 0.1F);
}

static void trainer_texture_blend_does_not_overwrite_learned_descriptor()
{
    auto descriptors = torch::zeros({1, 4, 2, 2}, torch::kFloat32);
    descriptors.index_put_({0, 0}, 1.0F);
    auto image = torch::tensor({{{{1.0F, 0.0F}, {0.5F, 0.25F}}}}, torch::kFloat32);

    auto blended = pfm::testing::blend_rotation_invariant_texture_descriptor_for_test(descriptors, image);
    auto cosine = (descriptors * blended).sum(1);

    PFM_REQUIRE(cosine.gt(0.85F).all().item<bool>());
}

static void trainer_descriptor_training_enables_texture_target_without_pairwise_teacher()
{
    PFM_REQUIRE_CLOSE(pfm::testing::descriptor_texture_teacher_weight_for_test(), 0.0, 1.0e-12);
    PFM_REQUIRE(pfm::testing::descriptor_texture_target_weight_for_test() > 0.0);
}

static void trainer_descriptor_orientation_canonicalization_rolls_channel_groups()
{
    auto descriptors = torch::zeros({1, 4, 1, 4}, torch::kFloat32);
    descriptors.index_put_({0, 0, 0, 0}, 1.0F);
    descriptors.index_put_({0, 1, 0, 1}, 1.0F);
    descriptors.index_put_({0, 2, 0, 2}, 1.0F);
    descriptors.index_put_({0, 3, 0, 3}, 1.0F);
    auto orientation = torch::zeros({1, 2, 1, 4}, torch::kFloat32);
    orientation.index_put_({0, 0, 0, 0}, 1.0F);
    orientation.index_put_({0, 1, 0, 1}, 1.0F);
    orientation.index_put_({0, 0, 0, 2}, -1.0F);
    orientation.index_put_({0, 1, 0, 3}, -1.0F);

    auto canonical = pfm::testing::canonicalize_descriptor_map_by_orientation_for_test(descriptors, orientation);

    auto expected = torch::zeros_like(descriptors);
    expected.index_put_({0, 0, 0, torch::indexing::Slice()}, 1.0F);
    PFM_REQUIRE(torch::allclose(canonical, expected, 1.0e-6, 1.0e-6));
}

static void trainer_descriptor_finetune_anchor_penalizes_teacher_drift()
{
    const auto current_a =
        torch::tensor({{{{1.0F, 0.0F}, {0.0F, 1.0F}}, {{0.0F, 1.0F}, {1.0F, 0.0F}}}}, torch::kFloat32);
    const auto current_b = current_a.clone();
    const auto anchor_a = current_a.clone();
    const auto anchor_b = current_b.clone();
    const auto valid_mask = torch::ones({1, 4, 4}, torch::kFloat32);

    const auto matching_loss = pfm::testing::make_descriptor_finetune_anchor_loss_for_test(
        current_a, current_b, anchor_a, anchor_b, valid_mask);

    auto drifted_a = current_a.clone();
    drifted_a.index_put_({0, 0, 0, 0}, 0.0F);
    drifted_a.index_put_({0, 1, 0, 0}, 1.0F);
    const auto drifted_loss = pfm::testing::make_descriptor_finetune_anchor_loss_for_test(
        drifted_a, current_b, anchor_a, anchor_b, valid_mask);

    PFM_REQUIRE(matching_loss.item<float>() < 1.0e-6F);
    PFM_REQUIRE(drifted_loss.item<float>() > 0.05F);
}

static void trainer_descriptor_finetune_anchor_weight_preserves_rotation_baseline()
{
    PFM_REQUIRE(pfm::testing::descriptor_finetune_anchor_weight_for_test() >= 800.0);
}

static void trainer_texture_blend_weight_matches_inference_regularizer()
{
    PFM_REQUIRE(pfm::testing::descriptor_texture_blend_weight_for_test() <= 0.5);
}

static void trainer_bounds_descriptor_loss_spatial_samples()
{
    const int64_t height = 80;
    const int64_t width = 80;
    auto grid = torch::arange(height * width, torch::kFloat32).reshape({1, 1, height, width});
    auto descriptors_a = torch::cat({grid, grid + 1.0F, grid + 2.0F, grid + 3.0F}, 1);
    auto descriptors_b = descriptors_a.clone();
    auto xy = torch::meshgrid({torch::arange(height, torch::kFloat32), torch::arange(width, torch::kFloat32)}, "ij");
    auto warp = torch::stack({xy[1], xy[0]}, 2).unsqueeze(0);
    auto valid_mask = torch::ones({1, height, width}, torch::kBool);

    auto sample_indices = pfm::testing::make_descriptor_sample_indices_for_test(descriptors_a);
    auto loss = pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(sample_indices.size(0) == 1024);
    PFM_REQUIRE(sample_indices.min().item<int64_t>() >= 0);
    PFM_REQUIRE(sample_indices.max().item<int64_t>() < height * width);
    auto sorted_indices = std::get<0>(sample_indices.sort());
    PFM_REQUIRE(sorted_indices.slice(0, 1).ne(sorted_indices.slice(0, 0, -1)).all().item<bool>());
    PFM_REQUIRE(loss.defined());
    PFM_REQUIRE(loss.dim() == 0);
    PFM_REQUIRE(std::isfinite(loss.item<float>()));
}

static void trainer_descriptor_loss_samples_sparse_valid_regions_on_large_maps()
{
    constexpr int64_t height = 300;
    constexpr int64_t width = 300;
    auto descriptors_a = torch::zeros({1, 4, height, width}, torch::kFloat32);
    auto descriptors_b = torch::zeros({1, 4, height, width}, torch::kFloat32);
    descriptors_a.index_put_({0, 0}, 1.0F);
    descriptors_b.index_put_({0, 0}, 1.0F);
    auto xy = torch::meshgrid({torch::arange(height, torch::kFloat32), torch::arange(width, torch::kFloat32)}, "ij");
    auto warp = torch::stack({xy[1], xy[0]}, 2).unsqueeze(0);
    auto valid_mask = torch::zeros({1, height, width}, torch::kBool);
    valid_mask.index_put_({0, height - 1, width - 1}, true);

    torch::manual_seed(0);
    auto loss = pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.item<float>() > 0.0F);
}

static void trainer_descriptor_loss_handles_disjoint_valid_regions_across_batch()
{
    constexpr int64_t height = 64;
    constexpr int64_t width = 64;
    auto descriptors_a = torch::zeros({2, 4, height, width}, torch::kFloat32);
    auto descriptors_b = torch::zeros({2, 4, height, width}, torch::kFloat32);
    descriptors_a.index_put_({torch::indexing::Slice(), 0}, 1.0F);
    descriptors_b.index_put_({torch::indexing::Slice(), 0}, 1.0F);
    auto xy = torch::meshgrid({torch::arange(height, torch::kFloat32), torch::arange(width, torch::kFloat32)}, "ij");
    auto warp_one = torch::stack({xy[1], xy[0]}, 2);
    auto warp = torch::stack({warp_one, warp_one}, 0);
    auto valid_mask = torch::zeros({2, height, width}, torch::kBool);
    valid_mask.index_put_({0, 0, 0}, true);
    valid_mask.index_put_({1, height - 1, width - 1}, true);

    auto loss = pfm::testing::make_sparse_descriptor_loss_for_test(descriptors_a, descriptors_b, warp, valid_mask);

    PFM_REQUIRE(loss.item<float>() > 0.0F);
}

static void trainer_resizes_large_training_image()
{
    auto image = torch::zeros({1, 900, 600}, torch::kFloat32);

    auto resized = pfm::testing::limit_training_image_size_for_test(image, 64);

    PFM_REQUIRE(resized.sizes() == torch::IntArrayRef({1, 64, 43}));
    PFM_REQUIRE(resized.is_contiguous());
}

static void trainer_uses_configured_resize()
{
    auto image = torch::zeros({1, 900, 600}, torch::kFloat32);

    auto resized = pfm::testing::limit_training_image_size_for_test(image, 300);

    PFM_REQUIRE(resized.sizes() == torch::IntArrayRef({1, 300, 200}));
    PFM_REQUIRE(resized.is_contiguous());
}

static void trainer_resizes_pair_warp_coordinates_in_view_b_space()
{
    auto view_a = torch::zeros({1, 4, 8}, torch::kFloat32);
    auto view_b = torch::zeros({1, 10, 20}, torch::kFloat32);
    auto warp = torch::zeros({4, 8, 2}, torch::kFloat32);
    warp.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0}, 19.0F);
    warp.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 1}, 9.0F);
    auto valid_mask = torch::ones({4, 8}, torch::kBool);

    const auto resized =
        pfm::testing::limit_training_pair_size_for_test(pfm::SyntheticPair{view_a, view_b, warp, valid_mask}, 4);
    const auto bottom_right_x = resized.warp_a_to_b.index({1, 3, 0}).item<float>();
    const auto bottom_right_y = resized.warp_a_to_b.index({1, 3, 1}).item<float>();

    PFM_REQUIRE(resized.view_a.sizes() == torch::IntArrayRef({1, 2, 4}));
    PFM_REQUIRE(resized.view_b.sizes() == torch::IntArrayRef({1, 2, 4}));
    PFM_REQUIRE(resized.warp_a_to_b.sizes() == torch::IntArrayRef({2, 4, 2}));
    PFM_REQUIRE_CLOSE(bottom_right_x, 3.0F, 1.0e-5F);
    PFM_REQUIRE_CLOSE(bottom_right_y, 1.0F, 1.0e-5F);
}

static void trainer_crop_uses_dedicated_training_generator()
{
    auto view_a = torch::arange(64, torch::kFloat32).reshape({1, 8, 8});
    auto view_b = view_a.clone();
    auto xy = torch::meshgrid({torch::arange(8, torch::kFloat32), torch::arange(8, torch::kFloat32)}, "ij");
    auto warp = torch::stack({xy[1], xy[0]}, 2);
    auto valid_mask = torch::ones({8, 8}, torch::kBool);
    auto pair = pfm::SyntheticPair{view_a, view_b, warp, valid_mask};

    const auto first = pfm::testing::crop_training_pair_with_seed_for_test(pair, 4, 20260603);
    (void)torch::randn({1024}, torch::kFloat32);
    const auto second = pfm::testing::crop_training_pair_with_seed_for_test(pair, 4, 20260603);

    PFM_REQUIRE(torch::equal(first.view_a, second.view_a));
    PFM_REQUIRE(torch::equal(first.view_b, second.view_b));
    PFM_REQUIRE(torch::equal(first.warp_a_to_b, second.warp_a_to_b));
    PFM_REQUIRE(torch::equal(first.valid_mask, second.valid_mask));
}

static void trainer_crop_origin_uses_python_half_even_rounding()
{
    auto view_a = torch::arange(64, torch::kFloat32).reshape({1, 8, 8});
    auto view_b = view_a.clone();
    auto xy = torch::meshgrid({torch::arange(8, torch::kFloat32), torch::arange(8, torch::kFloat32)}, "ij");
    auto warp = torch::stack({xy[1], xy[0]}, 2);
    auto valid_mask = torch::zeros({8, 8}, torch::kBool);
    valid_mask.index_put_({4, 4}, true);
    auto pair = pfm::SyntheticPair{view_a, view_b, warp, valid_mask};

    const auto cropped = pfm::testing::crop_training_pair_with_seed_for_test(pair, 4, 20260603);

    PFM_REQUIRE(cropped.view_a.index({0, 0, 0}).item<float>() == 18.0F);
    PFM_REQUIRE(cropped.warp_a_to_b.index({2, 2, 0}).item<float>() == 2.0F);
    PFM_REQUIRE(cropped.warp_a_to_b.index({2, 2, 1}).item<float>() == 2.0F);
    PFM_REQUIRE(cropped.valid_mask.index({2, 2}).item<bool>());
}

static void trainer_training_and_validation_indices_use_dataloader_split()
{
    pfm::TrainConfig config;
    config.train_ratio = 0.6;
    config.val_ratio = 0.2;
    config.split_seed = 7;

    const auto train = pfm::testing::make_training_image_indices_for_test(10, config);
    const auto validation = pfm::testing::make_validation_image_indices_for_test(10, config);
    const auto split = pfm::make_train_validation_test_split(10, 0.6, 0.2, 0.2, 7, true);

    PFM_REQUIRE(train == split.train);
    PFM_REQUIRE(validation == split.validation);
}

static void trainer_variant_indices_advance_across_epochs()
{
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(0, 1, 0, 8) == 0);
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(7, 1, 0, 8) == 7);
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(0, 1, 1, 8) == 8);
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(7, 1, 1, 8) == 15);
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(0, 2, 1, 8) == 8);
    PFM_REQUIRE(pfm::testing::training_variant_index_for_pair_for_test(2, 2, 1, 8) == 9);
}

static void trainer_augmentation_curriculum_stages_profile_by_epoch()
{
    pfm::TrainConfig config;
    config.epochs = 6;
    config.augmentation_profile = "compound-viewpoint";
    config.augmentation_curriculum = true;

    PFM_REQUIRE(pfm::testing::effective_augmentation_profile_for_epoch_for_test(config, 0) == "mixed");
    PFM_REQUIRE(pfm::testing::effective_augmentation_profile_for_epoch_for_test(config, 1) == "mixed");
    PFM_REQUIRE(pfm::testing::effective_augmentation_profile_for_epoch_for_test(config, 2) == "viewpoint");
    PFM_REQUIRE(pfm::testing::effective_augmentation_profile_for_epoch_for_test(config, 3) == "viewpoint");
    PFM_REQUIRE(pfm::testing::effective_augmentation_profile_for_epoch_for_test(config, 4) == "compound-viewpoint");
    PFM_REQUIRE(pfm::testing::effective_augmentation_profile_for_epoch_for_test(config, 5) == "compound-viewpoint");
}

static void trainer_augmentation_curriculum_preserves_requested_profile_when_disabled()
{
    pfm::TrainConfig config;
    config.epochs = 6;
    config.augmentation_profile = "extreme";
    config.augmentation_curriculum = false;

    PFM_REQUIRE(pfm::testing::effective_augmentation_profile_for_epoch_for_test(config, 0) == "extreme");
    PFM_REQUIRE(pfm::testing::effective_augmentation_profile_for_epoch_for_test(config, 3) == "extreme");
}

static void trainer_curriculum_disables_fixed_online_loader()
{
    pfm::TrainConfig config;
    config.dataloader_workers = 2;
    PFM_REQUIRE(pfm::testing::should_use_online_dataloader_for_test(config));

    config.augmentation_curriculum = true;
    PFM_REQUIRE(!pfm::testing::should_use_online_dataloader_for_test(config));
}

static void trainer_cache_training_sampler_shuffles_cached_pairs_deterministically()
{
    pfm::TrainConfig config;
    config.seed = 123;

    auto first = pfm::testing::make_cache_training_sample_indices_for_test(16, config);
    auto second = pfm::testing::make_cache_training_sample_indices_for_test(16, config);
    auto sequential = torch::arange(16, torch::kLong);

    PFM_REQUIRE(torch::equal(first, second));
    PFM_REQUIRE(!torch::equal(first, sequential));
    PFM_REQUIRE(torch::equal(std::get<0>(first.sort()), sequential));
}

static void trainer_cache_training_sampler_uses_training_seed_not_split_seed()
{
    pfm::TrainConfig base;
    base.seed = 123;
    base.split_seed = 7;

    auto same_seed_different_split = base;
    same_seed_different_split.split_seed = 99;
    auto different_seed_same_split = base;
    different_seed_same_split.seed = 456;

    const auto base_indices = pfm::testing::make_cache_training_sample_indices_for_test(32, base);
    const auto same_seed_indices =
        pfm::testing::make_cache_training_sample_indices_for_test(32, same_seed_different_split);
    const auto different_seed_indices =
        pfm::testing::make_cache_training_sample_indices_for_test(32, different_seed_same_split);

    PFM_REQUIRE(torch::equal(base_indices, same_seed_indices));
    PFM_REQUIRE(!torch::equal(base_indices, different_seed_indices));
}

static void trainer_hard_cache_dirs_are_repeated_after_base_and_extra_caches()
{
    pfm::TrainConfig config;
    config.synthetic_pair_cache_dir = "base_cache";
    config.extra_synthetic_pair_cache_dirs = {"rotate_cache", "viewpoint_cache"};
    config.hard_synthetic_pair_cache_dirs = {"compound_hard_cache", "extreme_hard_cache"};
    config.hard_synthetic_pair_cache_repeats = 3;

    const auto cache_dirs = pfm::testing::make_training_cache_dirs_for_test(config);

    PFM_REQUIRE(cache_dirs ==
                std::vector<std::string>({"base_cache", "rotate_cache", "viewpoint_cache", "compound_hard_cache",
                                          "compound_hard_cache", "compound_hard_cache", "extreme_hard_cache",
                                          "extreme_hard_cache", "extreme_hard_cache"}));
}

static void trainer_hard_cache_indices_repeat_only_selected_pairs()
{
    pfm::TrainConfig config;
    config.extra_synthetic_pair_cache_dirs = {"rotate_cache"};
    config.hard_synthetic_pair_cache_dirs = {"compound_cache"};
    config.hard_synthetic_pair_cache_repeats = 2;
    config.hard_synthetic_pair_cache_indices = {3, 7, 8};

    const auto entries = pfm::testing::make_training_cache_entries_for_test(config);

    PFM_REQUIRE(entries ==
                std::vector<std::string>({"rotate_cache:*", "compound_cache:3", "compound_cache:7", "compound_cache:8",
                                          "compound_cache:3", "compound_cache:7", "compound_cache:8"}));
}

static void trainer_total_loss_downweights_dense_offset_pixels()
{
    auto repeatability = torch::tensor(1.0F);
    auto descriptor = torch::tensor(2.0F);
    auto offset = torch::tensor(30.0F);
    auto confidence = torch::tensor(4.0F);

    auto loss = pfm::testing::weighted_total_training_loss_for_test(repeatability, descriptor, offset, confidence);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 102.0F, 1.0e-6F);
}

static void trainer_total_loss_penalizes_descriptor_spatial_collapse()
{
    auto repeatability = torch::tensor(1.0F);
    auto descriptor = torch::tensor(2.0F);
    auto offset = torch::tensor(3.0F);
    auto confidence = torch::tensor(4.0F);
    auto descriptor_diversity = torch::tensor(1.0F);

    auto loss = pfm::testing::weighted_total_training_loss_for_test(repeatability, descriptor, offset, confidence,
                                                                    descriptor_diversity);

    PFM_REQUIRE_CLOSE(loss.item<float>(), 107.0F, 1.0e-6F);
}

static void trainer_progress_reports_loss_components()
{
    TempTrainingDirectory temp_dir("pfm_trainer_loss_components");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    CoutCapture capture;
    auto result = pfm::train_model(config);
    const auto output = capture.str();

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(output.find("loss=") != std::string::npos);
    PFM_REQUIRE(output.find("match=") != std::string::npos);
    PFM_REQUIRE(output.find("feat=") != std::string::npos);
    PFM_REQUIRE(output.find("dense=") != std::string::npos);
    PFM_REQUIRE(output.find("off=") != std::string::npos);
    PFM_REQUIRE(output.find("epoch summary") != std::string::npos);
}

static void trainer_reports_epoch_and_batch_timing()
{
    TempTrainingDirectory temp_dir("pfm_trainer_timing");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    CoutCapture capture;
    const auto result = pfm::train_model(config);
    const auto output = capture.str();

    PFM_REQUIRE(result.total_time_seconds > 0.0);
    PFM_REQUIRE(result.avg_batch_time_seconds > 0.0);
    PFM_REQUIRE(output.find("elapsed=") != std::string::npos);
}

static void trainer_writes_csv_metric_log()
{
    TempTrainingDirectory temp_dir("pfm_trainer_csv_metrics");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    config.log_csv = (temp_dir.path() / "metrics.csv").string();
    temp_dir.file("checkpoint.pt");

    const auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    std::ifstream input(config.log_csv);
    std::string header;
    std::string row;
    std::string second_row;
    std::getline(input, header);
    std::getline(input, row);
    std::getline(input, second_row);
    PFM_REQUIRE(header.find("loss_total") != std::string::npos);
    PFM_REQUIRE(header.find("graph_matching_loss") != std::string::npos);
    PFM_REQUIRE(header.find("offset_error_px") != std::string::npos);
    PFM_REQUIRE(row.find("1,1,1,2") == 0);
    PFM_REQUIRE(second_row.find("1,1,2,2") == 0);
}

static void trainer_uses_online_dataloader_when_workers_requested()
{
    pfm::TrainConfig config;
    config.dataloader_workers = 2;
    config.synthetic_pair_cache_dir.clear();

    PFM_REQUIRE(pfm::testing::should_use_online_dataloader_for_test(config));
}

static void trainer_trains_with_online_dataloader_workers()
{
    TempTrainingDirectory temp_dir("pfm_trainer_online_dataloader");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    config.dataloader_workers = 2;
    config.prefetch_batches = 2;
    temp_dir.file("checkpoint.pt");

    const auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
}

static void trainer_trains_full_dataset()
{
    TempTrainingDirectory temp_dir("pfm_trainer_full_dataset");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    require_image_written(temp_dir.file("image_c.png"), 73);
    auto config = tiny_config(temp_dir);
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(config.checkpoint));
}

static void trainer_with_synthetic_pair_cache_writes_cache_and_checkpoint()
{
    TempTrainingDirectory temp_dir("pfm_trainer_pair_cache");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
    PFM_REQUIRE(pfm::checkpoint_can_load(config.checkpoint));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.synthetic_pair_cache_dir) / "manifest.pt"));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.synthetic_pair_cache_dir) /
                                        "source_000000_image_a" / "pair_000000.pt"));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.synthetic_pair_cache_dir) /
                                        "source_000000_image_a" / "source_000000_view_a.png"));
}

static void trainer_pairs_per_image_expands_cached_training_pairs()
{
    TempTrainingDirectory temp_dir("pfm_trainer_pairs_per_image_cache");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    config.pairs_per_image = 2;
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(result.final_loss > 0.0);
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.synthetic_pair_cache_dir) /
                                        "source_000001_image_b" / "pair_000003.pt"));
    PFM_REQUIRE(std::filesystem::exists(std::filesystem::path(config.synthetic_pair_cache_dir) /
                                        "source_000001_image_b" / "source_000001_view_a.png"));
}

static void trainer_reuses_existing_synthetic_pair_cache()
{
    TempTrainingDirectory temp_dir("pfm_trainer_pair_cache_reuse");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    config.resize = 32;
    temp_dir.file("checkpoint.pt");
    (void)pfm::train_model(config);
    const auto pair_path =
        std::filesystem::path(config.synthetic_pair_cache_dir) / "source_000000_image_a" / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    config.checkpoint = (temp_dir.path() / "checkpoint_2.pt").string();
    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) == first_write_time);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
}

static void trainer_rebuilds_synthetic_pair_cache_when_requested()
{
    TempTrainingDirectory temp_dir("pfm_trainer_pair_cache_rebuild");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.synthetic_pair_cache_dir = (temp_dir.path() / "pair_cache").string();
    config.resize = 32;
    temp_dir.file("checkpoint.pt");
    (void)pfm::train_model(config);
    const auto pair_path =
        std::filesystem::path(config.synthetic_pair_cache_dir) / "source_000000_image_a" / "pair_000000.pt";
    const auto first_write_time = std::filesystem::last_write_time(pair_path);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    config.checkpoint = (temp_dir.path() / "checkpoint_2.pt").string();
    config.synthetic_pair_cache_rebuild = true;
    auto result = pfm::train_model(config);

    PFM_REQUIRE(result.epochs_completed == 1);
    PFM_REQUIRE(std::filesystem::last_write_time(pair_path) != first_write_time);
    PFM_REQUIRE(std::filesystem::exists(config.checkpoint));
}

static float cached_pair_mean_displacement(const std::string& cache_dir, std::size_t index)
{
    pfm::SyntheticPairCacheDataset cache_dataset(cache_dir);
    const auto pair = cache_dataset.load(index);
    const auto xy = torch::meshgrid({torch::arange(pair.warp_a_to_b.size(0), torch::kFloat32),
                                     torch::arange(pair.warp_a_to_b.size(1), torch::kFloat32)},
                                    "ij");
    const auto grid = torch::stack({xy[1], xy[0]}, 2);
    return (pair.warp_a_to_b - grid).norm(2, 2).mean().item<float>();
}

static void trainer_forwards_augmentation_profile_to_cached_pairs()
{
    TempTrainingDirectory temp_dir("pfm_trainer_profile_forwarding");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto mild_config = tiny_config(temp_dir);
    mild_config.synthetic_pair_cache_dir = (temp_dir.path() / "mild_cache").string();
    mild_config.augmentation_profile = "mild";
    mild_config.pairs_per_image = 2;
    mild_config.resize = 32;
    temp_dir.file("mild_checkpoint.pt");
    (void)pfm::train_model(mild_config);

    auto extreme_config = tiny_config(temp_dir);
    extreme_config.checkpoint = (temp_dir.path() / "extreme_checkpoint.pt").string();
    extreme_config.synthetic_pair_cache_dir = (temp_dir.path() / "extreme_cache").string();
    extreme_config.augmentation_profile = "extreme";
    extreme_config.pairs_per_image = 2;
    extreme_config.resize = 32;
    temp_dir.file("extreme_checkpoint.pt");
    (void)pfm::train_model(extreme_config);

    PFM_REQUIRE(cached_pair_mean_displacement(extreme_config.synthetic_pair_cache_dir, 2) >
                cached_pair_mean_displacement(mild_config.synthetic_pair_cache_dir, 2) * 2.0F);
}

static void trainer_visualization_writes_expected_pngs_for_sampled_pair()
{
    TempTrainingDirectory temp_dir("pfm_trainer_visualization_sampled");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.visualization_dir = (temp_dir.path() / "train_vis").string();
    config.visualization_samples = 1;
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    (void)pfm::train_model(config);
    const auto vis_dir = std::filesystem::path(config.visualization_dir);

    const auto static_dir = vis_dir / "static";
    const auto epoch_dir = vis_dir / "epoch_000001";
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000000_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000000_view_b.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000000_valid_mask.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000000_warp_matches.png"));
    PFM_REQUIRE(!std::filesystem::exists(epoch_dir / "pair_000000_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(epoch_dir / "pair_000000_features_a.png"));
    PFM_REQUIRE(std::filesystem::exists(epoch_dir / "pair_000000_features_b.png"));
    PFM_REQUIRE(std::filesystem::exists(epoch_dir / "pair_000000_model_matches.png"));
    PFM_REQUIRE(!std::filesystem::exists(static_dir / "pair_000001_view_a.png"));
}

static void trainer_visualization_all_writes_every_pair()
{
    TempTrainingDirectory temp_dir("pfm_trainer_visualization_all");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.visualization_dir = (temp_dir.path() / "train_vis").string();
    config.visualization_samples_all = true;
    config.pairs_per_image = 2;
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    (void)pfm::train_model(config);
    const auto vis_dir = std::filesystem::path(config.visualization_dir);

    const auto static_dir = vis_dir / "static";
    const auto epoch_dir = vis_dir / "epoch_000001";
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000000_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000001_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000002_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(static_dir / "pair_000003_view_a.png"));
    PFM_REQUIRE(std::filesystem::exists(epoch_dir / "pair_000000_model_matches.png"));
    PFM_REQUIRE(std::filesystem::exists(epoch_dir / "pair_000003_model_matches.png"));
}

static void trainer_visualization_zero_samples_writes_no_pngs()
{
    TempTrainingDirectory temp_dir("pfm_trainer_visualization_zero");
    require_image_written(temp_dir.file("image_a.png"), 0);
    auto config = tiny_config(temp_dir);
    config.visualization_dir = (temp_dir.path() / "train_vis").string();
    config.visualization_samples = 0;
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    (void)pfm::train_model(config);

    PFM_REQUIRE(!std::filesystem::exists(std::filesystem::path(config.visualization_dir) / "epoch_000001"));
}

static void trainer_visualization_writes_sampled_pair_for_each_epoch()
{
    TempTrainingDirectory temp_dir("pfm_trainer_visualization_each_epoch");
    require_image_written(temp_dir.file("image_a.png"), 0);
    require_image_written(temp_dir.file("image_b.png"), 37);
    auto config = tiny_config(temp_dir);
    config.epochs = 2;
    config.visualization_dir = (temp_dir.path() / "train_vis").string();
    config.visualization_samples = 1;
    config.resize = 32;
    temp_dir.file("checkpoint.pt");

    (void)pfm::train_model(config);
    const auto vis_dir = std::filesystem::path(config.visualization_dir);

    PFM_REQUIRE(std::filesystem::exists(vis_dir / "epoch_000001" / "pair_000000_model_matches.png"));
    PFM_REQUIRE(std::filesystem::exists(vis_dir / "epoch_000002" / "pair_000000_model_matches.png"));
    PFM_REQUIRE(!std::filesystem::exists(vis_dir / "epoch_000002" / "pair_000001_view_a.png"));
}

static void trainer_visualization_warp_overlay_does_not_mutate_source_pair()
{
    pfm::SyntheticPair pair;
    pair.view_a = torch::zeros({1, 8, 8}, torch::kFloat32);
    pair.view_b = torch::zeros({1, 8, 8}, torch::kFloat32);
    pair.warp_a_to_b = torch::zeros({8, 8, 2}, torch::kFloat32);
    pair.valid_mask = torch::ones({8, 8}, torch::kFloat32);

    (void)pfm::testing::training_warp_overlay_image_for_test(pair);

    PFM_REQUIRE_CLOSE(pair.view_a.max().item<float>(), 0.0F, 1.0e-6F);
}

static void trainer_fast_decode_uses_configured_keypoint_grid()
{
    auto heatmap = torch::zeros({1, 1, 4, 4}, torch::kFloat32);
    heatmap.index_put_({0, 0, 0, 0}, 10.0F);
    heatmap.index_put_({0, 0, 0, 1}, 9.0F);
    heatmap.index_put_({0, 0, 1, 0}, 8.0F);
    heatmap.index_put_({0, 0, 1, 1}, 7.0F);
    heatmap.index_put_({0, 0, 0, 3}, 1.0F);
    heatmap.index_put_({0, 0, 3, 0}, 1.0F);
    heatmap.index_put_({0, 0, 3, 3}, 1.0F);
    pfm::SparseHeadOutput sparse{heatmap, torch::ones({1, 2, 4, 4}, torch::kFloat32),
                                 torch::empty({0}, torch::kFloat32), torch::empty({0}, torch::kFloat32),
                                 torch::empty({0}, torch::kFloat32)};
    auto config = pfm::TrainConfig{};
    config.max_keypoints = 4;
    config.keypoint_grid_rows = 2;
    config.keypoint_grid_cols = 2;
    config.keypoints_per_cell = 1;
    config.nms_radius = 0;
    config.min_keypoint_intensity = 0.0;

    auto features =
        pfm::testing::decode_training_features_fast_for_test(torch::ones({1, 4, 4}, torch::kFloat32), sparse, config);

    auto points = features.keypoints.to(torch::kCPU, torch::kFloat32);
    auto occupied = torch::zeros({2, 2}, torch::kInt64);
    for (int64_t index = 0; index < points.size(0); ++index)
    {
        const auto col = std::min<int64_t>(1, static_cast<int64_t>(points.index({index, 0}).item<float>()) / 2);
        const auto row = std::min<int64_t>(1, static_cast<int64_t>(points.index({index, 1}).item<float>()) / 2);
        occupied.index_put_({row, col}, occupied.index({row, col}).item<int64_t>() + 1);
    }
    PFM_REQUIRE(features.keypoints.size(0) == 4);
    PFM_REQUIRE(torch::all(occupied.eq(1)).item<bool>());
}

static void trainer_fast_decode_suppresses_neighbors_with_nms_radius()
{
    auto heatmap = torch::zeros({1, 1, 4, 4}, torch::kFloat32);
    heatmap.index_put_({0, 0, 1, 1}, 10.0F);
    heatmap.index_put_({0, 0, 1, 2}, 9.0F);
    heatmap.index_put_({0, 0, 3, 3}, 8.0F);
    pfm::SparseHeadOutput sparse{heatmap, torch::ones({1, 2, 4, 4}, torch::kFloat32),
                                 torch::empty({0}, torch::kFloat32), torch::empty({0}, torch::kFloat32),
                                 torch::empty({0}, torch::kFloat32)};
    auto config = pfm::TrainConfig{};
    config.max_keypoints = 2;
    config.keypoint_grid_rows = 1;
    config.keypoint_grid_cols = 1;
    config.keypoints_per_cell = 2;
    config.nms_radius = 1;
    config.min_keypoint_intensity = 0.0;

    auto features =
        pfm::testing::decode_training_features_fast_for_test(torch::ones({1, 4, 4}, torch::kFloat32), sparse, config);

    PFM_REQUIRE(features.keypoints.sizes() == torch::IntArrayRef({2, 2}));
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 0}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({0, 1}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 0}).item<float>(), 3.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(features.keypoints.index({1, 1}).item<float>(), 3.0F, 1.0e-6F);
}

static void trainer_fast_decode_uses_image_saliency_when_heatmap_is_flat()
{
    auto heatmap = torch::full({1, 1, 8, 8}, 0.5F, torch::kFloat32);
    pfm::SparseHeadOutput sparse{heatmap, torch::ones({1, 2, 8, 8}, torch::kFloat32),
                                 torch::empty({0}, torch::kFloat32), torch::empty({0}, torch::kFloat32),
                                 torch::empty({0}, torch::kFloat32)};
    auto image = torch::zeros({1, 8, 8}, torch::kFloat32);
    image.index_put_({0, torch::indexing::Slice(3, 5), torch::indexing::Slice(3, 5)}, 1.0F);
    auto config = pfm::TrainConfig{};
    config.max_keypoints = 1;
    config.keypoint_grid_rows = 1;
    config.keypoint_grid_cols = 1;
    config.nms_radius = 0;
    config.min_keypoint_intensity = 0.0;

    auto features = pfm::testing::decode_training_features_fast_for_test(image, sparse, config);

    PFM_REQUIRE(features.keypoints.size(0) == 1);
    const auto x = features.keypoints.index({0, 0}).item<float>();
    const auto y = features.keypoints.index({0, 1}).item<float>();
    PFM_REQUIRE(x >= 2.0F && x <= 5.0F);
    PFM_REQUIRE(y >= 2.0F && y <= 5.0F);
}

static void trainer_warp_aligned_keypoint_targets_mark_independent_b_coordinates()
{
    auto view_a = torch::zeros({1, 1, 1, 5}, torch::kFloat32);
    auto view_b = torch::zeros({1, 1, 1, 5}, torch::kFloat32);
    view_a.index_put_({0, 0, 0, 1}, 1.0F);
    view_b.index_put_({0, 0, 0, 3}, 1.0F);
    auto warp = torch::zeros({1, 1, 5, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(5, torch::kFloat32) + 2.0F);
    auto mask = torch::ones({1, 1, 5}, torch::kBool);

    auto targets = pfm::testing::make_warp_aligned_keypoint_targets_for_test(view_a, view_b, warp, mask, 1, 5);

    auto target_a = targets.index({0, 0});
    auto target_b = targets.index({1, 0});
    PFM_REQUIRE(target_a.index({0, 1}).item<float>() > 0.5F);
    PFM_REQUIRE(target_b.index({0, 3}).item<float>() > 0.5F);
    PFM_REQUIRE(target_b.index({0, 1}).item<float>() < 0.5F);
}

static void trainer_warp_aligned_keypoint_targets_keep_single_cell_centers()
{
    auto view_a = torch::zeros({1, 1, 1, 5}, torch::kFloat32);
    auto view_b = torch::zeros({1, 1, 1, 5}, torch::kFloat32);
    view_a.index_put_({0, 0, 0, 1}, 1.0F);
    view_b.index_put_({0, 0, 0, 3}, 1.0F);
    auto warp = torch::zeros({1, 1, 5, 2}, torch::kFloat32);
    warp.index_put_({0, 0, torch::indexing::Slice(), 0}, torch::arange(5, torch::kFloat32) + 2.0F);
    auto mask = torch::ones({1, 1, 5}, torch::kBool);

    auto targets = pfm::testing::make_warp_aligned_keypoint_targets_for_test(view_a, view_b, warp, mask, 1, 5);

    auto target_a = targets.index({0, 0});
    auto target_b = targets.index({1, 0});
    PFM_REQUIRE_CLOSE(target_a.sum().item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(target_b.sum().item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE(target_a.index({0, 1}).item<float>() > 0.5F);
    PFM_REQUIRE(target_b.index({0, 3}).item<float>() > 0.5F);
}

static void trainer_visualization_feature_overlay_suppresses_dark_pixels()
{
    auto image = torch::zeros({1, 8, 8}, torch::kFloat32);
    image.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(4, 8)}, 0.5F);
    pfm::FeatureSet features;
    features.keypoints = torch::tensor({{1.0F, 3.0F}, {5.0F, 3.0F}}, torch::kFloat32);
    features.feature_map_width = 8;
    features.feature_map_height = 8;

    const auto overlay = pfm::testing::training_feature_overlay_image_for_test(image, features, 0.05);

    PFM_REQUIRE(overlay.size(0) == 3);
    PFM_REQUIRE_CLOSE(overlay.index({0, 3, 1}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({0, 3, 5}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({1, 3, 5}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({2, 3, 5}).item<float>(), 0.0F, 1.0e-6F);
}

static void trainer_visualization_model_matches_uses_side_by_side_canvas()
{
    auto image_a = torch::zeros({1, 16, 16}, torch::kFloat32);
    auto image_b = torch::zeros({1, 16, 16}, torch::kFloat32);
    pfm::FeatureSet features_a;
    features_a.keypoints = torch::tensor({{1.0F, 1.0F}}, torch::kFloat32);
    features_a.feature_map_width = 8;
    features_a.feature_map_height = 8;
    pfm::FeatureSet features_b;
    features_b.keypoints = torch::tensor({{6.0F, 6.0F}}, torch::kFloat32);
    features_b.feature_map_width = 8;
    features_b.feature_map_height = 8;
    pfm::MatchSet matches;
    matches.sparse_matches = torch::tensor({{0, 0}}, torch::kLong);
    matches.points_a = torch::tensor({{1.0F, 6.0F}}, torch::kFloat32);
    matches.points_b = torch::tensor({{6.0F, 6.0F}}, torch::kFloat32);
    matches.confidence = torch::ones({1}, torch::kFloat32);

    const auto overlay =
        pfm::testing::training_match_overlay_image_for_test(image_a, image_b, features_a, features_b, matches);

    PFM_REQUIRE(overlay.size(1) == 16);
    PFM_REQUIRE(overlay.size(2) == 32);
    PFM_REQUIRE_CLOSE(overlay.index({0, 2, 2}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({0, 12, 28}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({0, 12, 2}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({0, 12, 28}).item<float>(), 1.0F, 1.0e-6F);
}

static pfm::FeatureSet make_match_color_features(const torch::Tensor& keypoints)
{
    pfm::FeatureSet features;
    features.keypoints = keypoints;
    features.feature_map_width = 8;
    features.feature_map_height = 8;
    return features;
}

static torch::Tensor make_match_color_warp()
{
    auto warp = torch::zeros({8, 8, 2}, torch::kFloat32);
    warp.index_put_({1, 1, 0}, 6.0F);
    warp.index_put_({1, 1, 1}, 1.0F);
    warp.index_put_({6, 1, 0}, 1.0F);
    warp.index_put_({6, 1, 1}, 6.0F);
    return warp;
}

static void trainer_visualization_model_matches_colors_sparse_correct_and_wrong_lines()
{
    auto image_a = torch::zeros({1, 8, 8}, torch::kFloat32);
    auto image_b = torch::zeros({1, 8, 8}, torch::kFloat32);
    const auto features_a = make_match_color_features(torch::tensor({{1.0F, 1.0F}, {1.0F, 6.0F}}, torch::kFloat32));
    const auto features_b = make_match_color_features(torch::tensor({{6.0F, 1.0F}, {6.0F, 6.0F}}, torch::kFloat32));
    pfm::MatchSet matches;
    matches.sparse_matches = torch::tensor({{0, 0}, {1, 1}}, torch::kLong);

    const auto overlay = pfm::testing::training_match_overlay_image_for_test(image_a, image_b, features_a, features_b,
                                                                             matches, make_match_color_warp(), 1.0);

    PFM_REQUIRE_CLOSE(overlay.index({0, 1, 10}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({1, 1, 10}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({2, 1, 10}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({0, 6, 10}).item<float>(), 1.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({1, 6, 10}).item<float>(), 0.0F, 1.0e-6F);
    PFM_REQUIRE_CLOSE(overlay.index({2, 6, 10}).item<float>(), 0.0F, 1.0e-6F);
}

static void trainer_visualization_model_matches_text_includes_correct_and_wrong_counts()
{
    const auto features_a = make_match_color_features(torch::tensor({{1.0F, 1.0F}, {1.0F, 6.0F}}, torch::kFloat32));
    const auto features_b = make_match_color_features(torch::tensor({{6.0F, 1.0F}, {6.0F, 6.0F}}, torch::kFloat32));
    pfm::MatchSet matches;
    matches.sparse_matches = torch::tensor({{0, 0}, {1, 1}}, torch::kLong);
    matches.points_a = torch::tensor({{1.0F, 1.0F}}, torch::kFloat32);
    matches.points_b = torch::tensor({{6.0F, 1.0F}}, torch::kFloat32);

    const auto text = pfm::testing::training_model_match_overlay_text_for_test(features_a, features_b, matches,
                                                                               make_match_color_warp(), 1.0);

    PFM_REQUIRE(text.find("correct_matches=2") != std::string::npos);
    PFM_REQUIRE(text.find("wrong_matches=1") != std::string::npos);
}

void register_trainer_tests()
{
    register_test("trainer_one_epoch_saves_loadable_checkpoint", trainer_one_epoch_saves_loadable_checkpoint);
    register_test("trainer_default_config_uses_larger_model_settings",
                  trainer_default_config_uses_larger_model_settings);
    register_test("trainer_learning_rate_schedule_warms_up_then_decays_to_floor",
                  trainer_learning_rate_schedule_warms_up_then_decays_to_floor);
    register_test("trainer_checkpoint_saves_graph_matcher_architecture_config",
                  trainer_checkpoint_saves_graph_matcher_architecture_config);
    register_test("trainer_descriptor_only_finetune_freezes_backbone_but_updates_descriptor_head",
                  trainer_descriptor_only_finetune_freezes_backbone_but_updates_descriptor_head);
    register_test("trainer_viewpoint_head_only_finetune_updates_only_viewpoint_descriptor_branch",
                  trainer_viewpoint_head_only_finetune_updates_only_viewpoint_descriptor_branch);
    register_test("trainer_graph_only_finetune_freezes_feature_extractor_but_updates_graph_matcher",
                  trainer_graph_only_finetune_freezes_feature_extractor_but_updates_graph_matcher);
    register_test("trainer_missing_image_dir_throws_invalid_argument",
                  trainer_missing_image_dir_throws_invalid_argument);
    register_test("trainer_invalid_numeric_parameters_throw_invalid_argument",
                  trainer_invalid_numeric_parameters_throw_invalid_argument);
    register_test("trainer_invalid_device_throws_invalid_argument", trainer_invalid_device_throws_invalid_argument);
    register_test("trainer_cuda_device_is_strictly_validated", trainer_cuda_device_is_strictly_validated);
    register_test("trainer_cuda_one_epoch_saves_cpu_loadable_checkpoint_when_available",
                  trainer_cuda_one_epoch_saves_cpu_loadable_checkpoint_when_available);
    register_test("trainer_resizes_dense_warp_as_normalized_local_offsets",
                  trainer_resizes_dense_warp_as_normalized_local_offsets);
    register_test("trainer_repeatability_uses_warped_heatmap_correspondence",
                  trainer_repeatability_uses_warped_heatmap_correspondence);
    register_test("trainer_detector_target_loss_prefers_warp_consistent_peaks",
                  trainer_detector_target_loss_prefers_warp_consistent_peaks);
    register_test("trainer_detector_target_loss_weights_missing_positive_peaks",
                  trainer_detector_target_loss_weights_missing_positive_peaks);
    register_test("trainer_positive_target_loss_directly_raises_target_peaks",
                  trainer_positive_target_loss_directly_raises_target_peaks);
    register_test("trainer_descriptor_loss_uses_warped_correspondence",
                  trainer_descriptor_loss_uses_warped_correspondence);
    register_test("trainer_descriptor_loss_ignores_invalid_warp_targets",
                  trainer_descriptor_loss_ignores_invalid_warp_targets);
    register_test("trainer_descriptor_loss_penalizes_globally_collapsed_descriptors",
                  trainer_descriptor_loss_penalizes_globally_collapsed_descriptors);
    register_test("trainer_dense_descriptor_hard_negative_loss_scans_full_map",
                  trainer_dense_descriptor_hard_negative_loss_scans_full_map);
    register_test("trainer_dense_descriptor_hard_negative_loss_weights_multiple_hard_negatives",
                  trainer_dense_descriptor_hard_negative_loss_weights_multiple_hard_negatives);
    register_test("trainer_bidirectional_dense_descriptor_hard_negative_loss_catches_reverse_duplicates",
                  trainer_bidirectional_dense_descriptor_hard_negative_loss_catches_reverse_duplicates);
    register_test("trainer_warp_descriptor_contrastive_loss_uses_half_turn_correspondence",
                  trainer_warp_descriptor_contrastive_loss_uses_half_turn_correspondence);
    register_test("trainer_warp_descriptor_contrastive_loss_rejects_untrained_cyclic_descriptor_shift",
                  trainer_warp_descriptor_contrastive_loss_rejects_untrained_cyclic_descriptor_shift);
    register_test("trainer_direct_full_map_descriptor_loss_penalizes_global_distractor",
                  trainer_direct_full_map_descriptor_loss_penalizes_global_distractor);
    register_test("trainer_direct_full_map_descriptor_loss_rejects_untrained_cyclic_descriptor_shift",
                  trainer_direct_full_map_descriptor_loss_rejects_untrained_cyclic_descriptor_shift);
    register_test("trainer_descriptor_targets_use_cell_centers_for_warp_coordinates",
                  trainer_descriptor_targets_use_cell_centers_for_warp_coordinates);
    register_test("trainer_warped_descriptor_sampling_preserves_subpixel_correspondence",
                  trainer_warped_descriptor_sampling_preserves_subpixel_correspondence);
    register_test("trainer_descriptor_map_regularization_penalizes_spatial_collapse",
                  trainer_descriptor_map_regularization_penalizes_spatial_collapse);
    register_test("trainer_keypoint_descriptor_loss_uses_sparse_keypoint_hard_negatives",
                  trainer_keypoint_descriptor_loss_uses_sparse_keypoint_hard_negatives);
    register_test("trainer_keypoint_descriptor_loss_penalizes_hardest_negative_margin",
                  trainer_keypoint_descriptor_loss_penalizes_hardest_negative_margin);
    register_test("trainer_keypoint_descriptor_metrics_report_sparse_match_quality",
                  trainer_keypoint_descriptor_metrics_report_sparse_match_quality);
    register_test("trainer_keypoint_dense_descriptor_loss_uses_warp_target_in_full_map",
                  trainer_keypoint_dense_descriptor_loss_uses_warp_target_in_full_map);
    register_test("trainer_keypoint_descriptor_losses_ignore_out_of_bounds_warp_targets",
                  trainer_keypoint_descriptor_losses_ignore_out_of_bounds_warp_targets);
    register_test("trainer_keypoint_patch_descriptor_alignment_uses_warp_neighborhood",
                  trainer_keypoint_patch_descriptor_alignment_uses_warp_neighborhood);
    register_test("trainer_warped_keypoint_descriptor_contrastive_loss_uses_true_warp_targets",
                  trainer_warped_keypoint_descriptor_contrastive_loss_uses_true_warp_targets);
    register_test("trainer_decoded_keypoint_repeatability_raises_warped_b_heatmap_targets",
                  trainer_decoded_keypoint_repeatability_raises_warped_b_heatmap_targets);
    register_test("trainer_scales_feature_keypoints_to_image_pixel_centers",
                  trainer_scales_feature_keypoints_to_image_pixel_centers);
    register_test("trainer_orientation_supervision_uses_warp_rotation",
                  trainer_orientation_supervision_uses_warp_rotation);
    register_test("trainer_keypoint_descriptor_loss_covers_more_than_graph_query_limit",
                  trainer_keypoint_descriptor_loss_covers_more_than_graph_query_limit);
    register_test("trainer_graph_matching_loss_trains_graph_matcher_parameters",
                  trainer_graph_matching_loss_trains_graph_matcher_parameters);
    register_test("trainer_graph_matching_loss_is_finite_with_many_descriptors",
                  trainer_graph_matching_loss_is_finite_with_many_descriptors);
    register_test("trainer keypoint graph targets use warped nearest b keypoint",
                  trainer_keypoint_graph_targets_use_warped_nearest_b_keypoint);
    register_test("trainer keypoint graph targets use dustbin for unmatched keypoints",
                  trainer_keypoint_graph_targets_use_dustbin_for_unmatched_keypoints);
    register_test("trainer keypoint graph targets use dustbin for invalid source pixels",
                  trainer_keypoint_graph_targets_use_dustbin_for_invalid_source_pixels);
    register_test("trainer graph candidates include positives once and dustbin last",
                  trainer_graph_candidates_include_positives_once_and_dustbin_last);
    register_test("trainer graph query sampler prioritizes late positive targets",
                  trainer_graph_query_sampler_prioritizes_late_positive_targets);
    register_test("trainer graph query sampler keeps background when positives are abundant",
                  trainer_graph_query_sampler_keeps_background_when_positives_are_abundant);
    register_test("trainer keypoint graph matching loss trains graph matcher parameters",
                  trainer_keypoint_graph_matching_loss_trains_graph_matcher_parameters);
    register_test("trainer keypoint graph matching loss uses full b candidate set",
                  trainer_keypoint_graph_matching_loss_uses_full_b_candidate_set);
    register_test("trainer graph loss prioritizes decoded keypoint pairs",
                  trainer_graph_loss_prioritizes_decoded_keypoint_pairs);
    register_test("trainer_stacks_variable_spatial_training_tensors_with_padding",
                  trainer_stacks_variable_spatial_training_tensors_with_padding);
    register_test("trainer_training_valid_mask_requires_bright_source_and_target_pixels",
                  trainer_training_valid_mask_requires_bright_source_and_target_pixels);
    register_test("trainer_python_compare_pair_loss_mask_keeps_python_center_intensity_samples",
                  trainer_python_compare_pair_loss_mask_keeps_python_center_intensity_samples);
    register_test("trainer_python_compare_profile_skips_dense_quality_forward",
                  trainer_python_compare_profile_skips_dense_quality_forward);
    register_test("trainer_python_compare_trainable_parameters_match_python_defaults",
                  trainer_python_compare_trainable_parameters_match_python_defaults);
    register_test("trainer_python_compare_trainable_parameters_train_graph_when_requested",
                  trainer_python_compare_trainable_parameters_train_graph_when_requested);
    register_test("trainer_python_compare_trainable_parameters_follow_python_full_flags",
                  trainer_python_compare_trainable_parameters_follow_python_full_flags);
    register_test("trainer_python_compare_graph_loss_disables_candidate_mask_for_supervision",
                  trainer_python_compare_graph_loss_disables_candidate_mask_for_supervision);
    register_test("trainer python compare graph loss can train accept head",
                  trainer_python_compare_graph_loss_can_train_accept_head);
    register_test("trainer python compare graph loss penalizes unmatched accept logits",
                  trainer_python_compare_graph_loss_penalizes_unmatched_accept_logits);
    register_test("trainer python compare graph loss respects attention layer budget",
                  trainer_python_compare_graph_loss_respects_attention_layer_budget);
    register_test("trainer python compare graph loss can randomize attention layer budget",
                  trainer_python_compare_graph_loss_can_randomize_attention_layer_budget);
    register_test("trainer python compare graph loss respects attention work budget",
                  trainer_python_compare_graph_loss_respects_attention_work_budget);
    register_test("trainer python compare graph metadata mode matches python masks",
                  trainer_python_compare_graph_metadata_mode_matches_python_masks);
    register_test("trainer python compare graph loss can train with width dropout",
                  trainer_python_compare_graph_loss_can_train_with_width_dropout);
    register_test("trainer python compare graph loss can train prune ranking accept head",
                  trainer_python_compare_graph_loss_can_train_prune_ranking_accept_head);
    register_test("trainer python compare graph loss can train stop confidence score path",
                  trainer_python_compare_graph_loss_can_train_stop_confidence_score_path);
    register_test("trainer python compare graph loss adds raw preservation margin",
                  trainer_python_compare_graph_loss_adds_raw_preservation_margin);
    register_test("trainer python compare graph loss adds hard negative dustbin margin",
                  trainer_python_compare_graph_loss_adds_hard_negative_dustbin_margin);
    register_test("trainer_descriptor_candidates_do_not_repeat_positive_target",
                  trainer_descriptor_candidates_do_not_repeat_positive_target);
    register_test("trainer_descriptor_candidates_exclude_spatial_near_positives",
                  trainer_descriptor_candidates_exclude_spatial_near_positives);
    register_test("trainer_descriptor_candidates_prioritize_near_ring_hard_negatives",
                  trainer_descriptor_candidates_prioritize_near_ring_hard_negatives);
    register_test("trainer_descriptor_candidates_cover_broad_far_negative_regions",
                  trainer_descriptor_candidates_cover_broad_far_negative_regions);
    register_test("trainer_descriptor_candidates_can_disable_broad_far_negatives_for_curriculum",
                  trainer_descriptor_candidates_can_disable_broad_far_negatives_for_curriculum);
    register_test("trainer_descriptor_candidates_curriculum_limits_early_pool_to_near_ring",
                  trainer_descriptor_candidates_curriculum_limits_early_pool_to_near_ring);
    register_test("trainer_descriptor_broad_far_negative_curriculum_ramps_with_progress",
                  trainer_descriptor_broad_far_negative_curriculum_ramps_with_progress);
    register_test("trainer_descriptor_broad_far_negative_curriculum_caps_extreme_ce_pool",
                  trainer_descriptor_broad_far_negative_curriculum_caps_extreme_ce_pool);
    register_test("trainer_descriptor_ranking_loss_scans_late_hard_negatives",
                  trainer_descriptor_ranking_loss_scans_late_hard_negatives);
    register_test("trainer_descriptor_ranking_loss_weights_multiple_hard_negatives",
                  trainer_descriptor_ranking_loss_weights_multiple_hard_negatives);
    register_test("trainer_descriptor_ranking_loss_penalizes_many_near_top_distractors",
                  trainer_descriptor_ranking_loss_penalizes_many_near_top_distractors);
    register_test("trainer_descriptor_ranking_loss_has_tail_rank_penalty",
                  trainer_descriptor_ranking_loss_has_tail_rank_penalty);
    register_test("trainer_descriptor_ranking_loss_scans_broad_cross_view_hard_negative_set",
                  trainer_descriptor_ranking_loss_scans_broad_cross_view_hard_negative_set);
    register_test("trainer_descriptor_candidates_scan_wide_cross_view_negative_pool",
                  trainer_descriptor_candidates_scan_wide_cross_view_negative_pool);
    register_test("trainer_descriptor_global_ce_has_cross_view_weight",
                  trainer_descriptor_global_ce_has_cross_view_weight);
    register_test("trainer_descriptor_ranking_step_reduces_hard_negative_score",
                  trainer_descriptor_ranking_step_reduces_hard_negative_score);
    register_test("trainer_sampled_descriptor_decorrelation_penalizes_far_duplicate_descriptors",
                  trainer_sampled_descriptor_decorrelation_penalizes_far_duplicate_descriptors);
    register_test("trainer_sampled_descriptor_decorrelation_ignores_nearby_descriptors",
                  trainer_sampled_descriptor_decorrelation_ignores_nearby_descriptors);
    register_test("trainer_positive_descriptor_alignment_step_increases_positive_cosine",
                  trainer_positive_descriptor_alignment_step_increases_positive_cosine);
    register_test("trainer_patch_descriptor_alignment_uses_local_neighborhood",
                  trainer_patch_descriptor_alignment_uses_local_neighborhood);
    register_test("trainer_patch_descriptor_alignment_uses_wider_viewpoint_neighborhood",
                  trainer_patch_descriptor_alignment_uses_wider_viewpoint_neighborhood);
    register_test("trainer_descriptor_training_similarity_rejects_channel_shifted_negative",
                  trainer_descriptor_training_similarity_rejects_channel_shifted_negative);
    register_test("trainer_strict_descriptor_ce_rejects_channel_shifted_negative",
                  trainer_strict_descriptor_ce_rejects_channel_shifted_negative);
    register_test("trainer_descriptor_candidates_skip_invalid_target_regions",
                  trainer_descriptor_candidates_skip_invalid_target_regions);
    register_test("trainer_descriptor_candidates_repeat_valid_regions_instead_of_invalid_fill",
                  trainer_descriptor_candidates_repeat_valid_regions_instead_of_invalid_fill);
    register_test("trainer_texture_blend_preserves_learned_positive_margin",
                  trainer_texture_blend_preserves_learned_positive_margin);
    register_test("trainer_texture_blend_does_not_overwrite_learned_descriptor",
                  trainer_texture_blend_does_not_overwrite_learned_descriptor);
    register_test("trainer_descriptor_training_enables_texture_target_without_pairwise_teacher",
                  trainer_descriptor_training_enables_texture_target_without_pairwise_teacher);
    register_test("trainer_descriptor_orientation_canonicalization_rolls_channel_groups",
                  trainer_descriptor_orientation_canonicalization_rolls_channel_groups);
    register_test("trainer_descriptor_finetune_anchor_penalizes_teacher_drift",
                  trainer_descriptor_finetune_anchor_penalizes_teacher_drift);
    register_test("trainer_descriptor_finetune_anchor_weight_preserves_rotation_baseline",
                  trainer_descriptor_finetune_anchor_weight_preserves_rotation_baseline);
    register_test("trainer_texture_blend_weight_matches_inference_regularizer",
                  trainer_texture_blend_weight_matches_inference_regularizer);
    register_test("trainer_bounds_descriptor_loss_spatial_samples", trainer_bounds_descriptor_loss_spatial_samples);
    register_test("trainer_descriptor_loss_samples_sparse_valid_regions_on_large_maps",
                  trainer_descriptor_loss_samples_sparse_valid_regions_on_large_maps);
    register_test("trainer_descriptor_loss_handles_disjoint_valid_regions_across_batch",
                  trainer_descriptor_loss_handles_disjoint_valid_regions_across_batch);
    register_test("trainer_total_loss_downweights_dense_offset_pixels",
                  trainer_total_loss_downweights_dense_offset_pixels);
    register_test("trainer_total_loss_penalizes_descriptor_spatial_collapse",
                  trainer_total_loss_penalizes_descriptor_spatial_collapse);
    register_test("trainer_progress_reports_loss_components", trainer_progress_reports_loss_components);
    register_test("trainer_reports_epoch_and_batch_timing", trainer_reports_epoch_and_batch_timing);
    register_test("trainer_writes_csv_metric_log", trainer_writes_csv_metric_log);
    register_test("trainer_uses_online_dataloader_when_workers_requested",
                  trainer_uses_online_dataloader_when_workers_requested);
    register_test("trainer_trains_with_online_dataloader_workers", trainer_trains_with_online_dataloader_workers);
    register_test("trainer_resizes_large_training_image", trainer_resizes_large_training_image);
    register_test("trainer_uses_configured_resize", trainer_uses_configured_resize);
    register_test("trainer_resizes_pair_warp_coordinates_in_view_b_space",
                  trainer_resizes_pair_warp_coordinates_in_view_b_space);
    register_test("trainer_crop_uses_dedicated_training_generator",
                  trainer_crop_uses_dedicated_training_generator);
    register_test("trainer_crop_origin_uses_python_half_even_rounding",
                  trainer_crop_origin_uses_python_half_even_rounding);
    register_test("trainer_training_and_validation_indices_use_dataloader_split",
                  trainer_training_and_validation_indices_use_dataloader_split);
    register_test("trainer_variant_indices_advance_across_epochs", trainer_variant_indices_advance_across_epochs);
    register_test("trainer_augmentation_curriculum_stages_profile_by_epoch",
                  trainer_augmentation_curriculum_stages_profile_by_epoch);
    register_test("trainer_augmentation_curriculum_preserves_requested_profile_when_disabled",
                  trainer_augmentation_curriculum_preserves_requested_profile_when_disabled);
    register_test("trainer_curriculum_disables_fixed_online_loader", trainer_curriculum_disables_fixed_online_loader);
    register_test("trainer_cache_training_sampler_shuffles_cached_pairs_deterministically",
                  trainer_cache_training_sampler_shuffles_cached_pairs_deterministically);
    register_test("trainer_cache_training_sampler_uses_training_seed_not_split_seed",
                  trainer_cache_training_sampler_uses_training_seed_not_split_seed);
    register_test("trainer_hard_cache_dirs_are_repeated_after_base_and_extra_caches",
                  trainer_hard_cache_dirs_are_repeated_after_base_and_extra_caches);
    register_test("trainer_hard_cache_indices_repeat_only_selected_pairs",
                  trainer_hard_cache_indices_repeat_only_selected_pairs);
    register_test("trainer_trains_full_dataset", trainer_trains_full_dataset);
    register_test("trainer_with_synthetic_pair_cache_writes_cache_and_checkpoint",
                  trainer_with_synthetic_pair_cache_writes_cache_and_checkpoint);
    register_test("trainer_pairs_per_image_expands_cached_training_pairs",
                  trainer_pairs_per_image_expands_cached_training_pairs);
    register_test("trainer_reuses_existing_synthetic_pair_cache", trainer_reuses_existing_synthetic_pair_cache);
    register_test("trainer_rebuilds_synthetic_pair_cache_when_requested",
                  trainer_rebuilds_synthetic_pair_cache_when_requested);
    register_test("trainer_forwards_augmentation_profile_to_cached_pairs",
                  trainer_forwards_augmentation_profile_to_cached_pairs);
    register_test("trainer_visualization_writes_expected_pngs_for_sampled_pair",
                  trainer_visualization_writes_expected_pngs_for_sampled_pair);
    register_test("trainer_visualization_all_writes_every_pair", trainer_visualization_all_writes_every_pair);
    register_test("trainer_visualization_zero_samples_writes_no_pngs",
                  trainer_visualization_zero_samples_writes_no_pngs);
    register_test("trainer_visualization_writes_sampled_pair_for_each_epoch",
                  trainer_visualization_writes_sampled_pair_for_each_epoch);
    register_test("trainer_visualization_warp_overlay_does_not_mutate_source_pair",
                  trainer_visualization_warp_overlay_does_not_mutate_source_pair);
    register_test("trainer_fast_decode_uses_configured_keypoint_grid",
                  trainer_fast_decode_uses_configured_keypoint_grid);
    register_test("trainer_fast_decode_suppresses_neighbors_with_nms_radius",
                  trainer_fast_decode_suppresses_neighbors_with_nms_radius);
    register_test("trainer_fast_decode_uses_image_saliency_when_heatmap_is_flat",
                  trainer_fast_decode_uses_image_saliency_when_heatmap_is_flat);
    register_test("trainer_warp_completed_keypoint_pair_uses_true_warped_b_descriptors",
                  trainer_warp_completed_keypoint_pair_uses_true_warped_b_descriptors);
    register_test("trainer_warp_aligned_keypoint_targets_mark_independent_b_coordinates",
                  trainer_warp_aligned_keypoint_targets_mark_independent_b_coordinates);
    register_test("trainer_warp_aligned_keypoint_targets_keep_single_cell_centers",
                  trainer_warp_aligned_keypoint_targets_keep_single_cell_centers);
    register_test("trainer_visualization_feature_overlay_suppresses_dark_pixels",
                  trainer_visualization_feature_overlay_suppresses_dark_pixels);
    register_test("trainer_visualization_model_matches_uses_side_by_side_canvas",
                  trainer_visualization_model_matches_uses_side_by_side_canvas);
    register_test("trainer_visualization_model_matches_colors_sparse_correct_and_wrong_lines",
                  trainer_visualization_model_matches_colors_sparse_correct_and_wrong_lines);
    register_test("trainer_visualization_model_matches_text_includes_correct_and_wrong_counts",
                  trainer_visualization_model_matches_text_includes_correct_and_wrong_counts);
}
