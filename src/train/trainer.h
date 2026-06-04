#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace pfm
{

struct TrainConfig
{
    /// 原始训练图像目录；使用 pair cache 训练时可为空。
    std::string image_dir;
    /// 训练完成后写出的检查点路径。
    std::string checkpoint;
    /// 可选初始化检查点，用于继续训练或微调。
    std::string init_checkpoint;
    /// 计算设备名称，例如 cpu 或 cuda:0。
    std::string device = "cpu";
    /// 训练轮数。
    int epochs = 1;
    /// batch size。
    int batch_size = 1;
    /// v2.1 backbone 的基础通道数。
    int base_channels = 32;
    /// 稀疏/稠密描述子通道数。
    int descriptor_dim = 128;
    /// 图匹配器隐藏层维度。
    int graph_hidden_dim = 256;
    /// 图匹配器 self/cross attention 层数。
    int graph_attention_layers = 6;
    /// 图匹配器关键点几何元数据维度。
    int graph_keypoint_meta_dim = 16;
    /// 训练损失组合档位，例如 full 或 smoke。
    std::string training_profile = "full";
    /// Python 对齐训练中每个 pair 采样的对应点数量。
    int samples_per_pair = 512;
    /// Python 对齐训练中的 descriptor synthetic loss 权重。
    double synthetic_loss_weight = 0.1;
    /// Python 对齐训练中的 graph matcher loss 权重。
    double graph_matcher_loss_weight = 1.0;
    /// Python 对齐训练中的 descriptor softmax temperature。
    double temperature = 0.07;
    /// 训练图像缩放上限；0 或负值表示不缩放。
    int resize = 512;
    /// pair archive 训练时先裁剪的局部窗口大小；0 表示不裁剪。
    int training_crop_size = 0;
    /// 每张图像在线生成的影像对数量。
    int pairs_per_image = 1;
    /// 每个 epoch 最多训练 batch 数；0 表示不限制。
    int max_train_batches = 0;
    /// 在线合成增强档位名称。
    std::string augmentation_profile = "mixed";
    /// 是否随训练进度逐步放开更强增强。
    bool augmentation_curriculum = false;
    /// mixed 档位中 extreme 样本占比。
    double extreme_pair_ratio = 0.2;
    /// 离散旋转增强步长，单位为度。
    double rotation_step_degrees = 15.0;
    /// 需要生成或复用的主合成 pair cache 目录。
    std::string synthetic_pair_cache_dir;
    /// 额外普通合成 pair cache 目录。
    std::vector<std::string> extra_synthetic_pair_cache_dirs;
    /// hard case 合成 pair cache 目录。
    std::vector<std::string> hard_synthetic_pair_cache_dirs;
    /// 直接作为训练输入的 pair archive cache 目录列表。
    std::vector<std::string> pair_cache_dirs;
    /// 每个 pair archive cache 最多读取的样本数；0 表示不限制。
    int64_t pair_cache_limit = 0;
    /// pair archive CPU 内存缓存样本数；0 表示不缓存。
    int64_t pair_memory_cache_size = 0;
    /// hard cache 相对普通 cache 的重复采样次数。
    int hard_synthetic_pair_cache_repeats = 3;
    /// hard cache 中只重复采样的 pair 索引；为空表示全部使用。
    std::vector<int64_t> hard_synthetic_pair_cache_indices;
    /// 只生成 cache，不执行训练。
    bool cache_only = false;
    /// 可选 CSV 训练日志路径。
    std::string log_csv;
    /// 是否强制重建合成 pair cache。
    bool synthetic_pair_cache_rebuild = false;
    /// 训练过程可视化输出目录。
    std::string visualization_dir;
    /// 每轮最多可视化样本数。
    int visualization_samples = 4;
    /// 是否可视化所有样本。
    bool visualization_samples_all = false;
    /// 解码时最多保留的稀疏关键点数量。
    int max_keypoints = 1024;
    /// 解码时至少尝试保留的稀疏关键点数量。
    int min_keypoints = 0;
    /// 稀疏关键点网格采样行数。
    int keypoint_grid_rows = 8;
    /// 稀疏关键点网格采样列数。
    int keypoint_grid_cols = 8;
    /// 每个网格单元最多保留的关键点数；0 表示自动推导。
    int keypoints_per_cell = 0;
    /// 稀疏关键点 NMS 半径，单位为特征图像素。
    int nms_radius = 4;
    /// 低于该强度的图像位置不参与关键点训练和可视化。
    double min_keypoint_intensity = 0.08;
    /// 初始学习率。
    double learning_rate = 3.0e-4;
    /// 学习率 warmup 步数。
    int lr_warmup_steps = 0;
    /// 余弦或阶段调度中的最小学习率比例。
    double min_learning_rate_ratio = 0.01;
    /// 优化器权重衰减。
    double weight_decay = 5.0e-4;
    /// 梯度裁剪范数上限。
    double gradient_clip_norm = 1.0;
    /// 训练 split 比例。
    double train_ratio = 1.0;
    /// 验证 split 比例。
    double val_ratio = 0.0;
    /// 训练随机种子，用于模型初始化、训练采样和 pair cache shuffle。
    int seed = 1234;
    /// train/val split 随机种子。
    int split_seed = 42;
    /// 数据加载器后台 worker 数。
    int dataloader_workers = 0;
    /// 异步 DataLoader 预取 batch 数。
    int prefetch_batches = 2;
    /// 是否为 CPU batch 启用 pinned memory。
    bool pin_memory = false;
    /// 仅微调描述子相关分支。
    bool descriptor_only_finetune = false;
    /// 仅微调视角/方向相关分支。
    bool viewpoint_head_only_finetune = false;
    /// 仅微调图匹配器。
    bool graph_only_finetune = false;
    /// 是否启用描述子方向归一化。
    bool descriptor_orientation_canonicalization = true;
};

struct TrainResult
{
    /// 实际完成的 epoch 数。
    int epochs_completed = 0;
    /// 首个观测到的训练 loss。
    double initial_loss = 0.0;
    /// 最后一个观测到的训练 loss。
    double final_loss = 0.0;
    /// 最优验证 loss；没有验证集时保持最大值。
    double best_val_loss = std::numeric_limits<double>::max();
    /// 训练总耗时，单位为秒。
    double total_time_seconds = 0.0;
    /// 平均 batch 耗时，单位为秒。
    double avg_batch_time_seconds = 0.0;
};

/// 按配置训练真实影像 MVP 模型，并保存检查点。
/// @param config 训练图像目录、检查点路径、计算设备、数据限制、缓存设置和优化器设置。
/// @return 已完成 epoch 数，以及首次和最终观测到的训练损失。
/// @throws std::invalid_argument 当路径、数值参数或请求的设备非法时抛出。
TrainResult train_model(const TrainConfig& config);

/// 检查训练检查点能否作为 LibTorch archive 加载。
/// @param checkpoint 由 train_model 写出的检查点文件路径。
/// @return archive 可加载且包含必需配置张量时返回 true，否则返回 false。
bool checkpoint_can_load(const std::string& checkpoint);

} // namespace pfm
