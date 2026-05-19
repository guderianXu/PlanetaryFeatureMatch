# Reusable Training Infrastructure Design

## Goal

Add reusable deep-learning training infrastructure under the existing `modules/` tree and integrate it into PlanetaryFeatureMatch in phases. The first outcome is better data throughput, reproducible image-pair augmentation, clearer training logs, CSV metrics, optional NVML GPU telemetry, and a path to replace the current hand-written training batch loop with an asynchronous DataLoader.

## Context

The current trainer owns too many responsibilities: image loading, synthetic pair generation, cache access, batching, loss computation, console logging, checkpointing, and visualization dispatch. Recent training logs show that `feature_loss` and descriptor accuracy can converge while `graph_matching_loss` remains high, so the next changes need better data strategy and observability before more matcher-specific loss work. The new modules should be useful in future deep-learning projects, not only in this repository.

## Scope

This design covers four reusable modules and phased integration:

1. `modules/runtime/`: thread pool, blocking queue, cancellation, and exception propagation.
2. `modules/dataloader/`: `TensorBatch`, dataset abstraction, samplers, train/validation/test split, async prefetch, and pinned-memory batches.
3. `modules/augment/`: reusable image-pair augmentation and synthetic correspondence generation.
4. `modules/logging/`: console progress logger, CSV logger, optional NVML GPU metrics, and null fallback.

The first implementation should not change checkpoint formats or output `.pt` formats. It should keep existing CLI behavior unless adding a small number of high-level options is necessary for log output paths or split ratios.

## Non-goals

- Do not redesign the neural network architecture in this phase.
- Do not add low-level model structure CLI flags.
- Do not remove the existing synthetic pair cache format.
- Do not make NVML mandatory.
- Do not replace all trainer logic in one large change.

## Architecture

The new modules sit below `modules/train/` and are used by trainer as reusable infrastructure:

```text
modules/
  runtime/
    blocking_queue.*
    thread_pool.*
  dataloader/
    tensor_batch.*
    dataset.*
    sampler.*
    async_dataloader.*
    pinned_memory.*
  augment/
    image_pair_augmentor.*
    transform_sampler.*
    augmentation_profile.*
  logging/
    training_metric.*
    progress_logger.*
    csv_metric_logger.*
    gpu_metric_provider.*
    nvml_gpu_metric_provider.*
  data/
    synthetic_pair.*        # compatibility wrapper around augment
    synthetic_pair_cache.*  # continues using same cache format
  train/
    trainer.*              # phased integration point
```

`runtime` has no project-specific image or model dependencies. `dataloader` depends on `runtime` and LibTorch. `augment` depends on LibTorch and the existing geometry/image utilities. `logging` depends on standard C++ and optionally NVML; it should not depend on trainer internals.

## Runtime Module

`modules/runtime/` provides the concurrency primitives shared by DataLoader and future async jobs.

### BlockingQueue

Responsibilities:

- Bounded queue with blocking `push()` and `pop()`.
- `close()` wakes all waiting threads.
- `pop()` returns an empty optional after close and drain.
- No busy-waiting.
- Propagates no exceptions itself; producers place values or close the queue.

Expected interface:

```cpp
template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity);
    void push(T value);
    std::optional<T> pop();
    void close();
    size_t size() const;
};
```

### ThreadPool

Responsibilities:

- Start fixed number of worker threads.
- Accept jobs while running.
- Stop cleanly in destructor.
- Capture first job exception and rethrow from `join()`.
- Support cancellation by closing the internal queue.

Expected interface:

```cpp
class ThreadPool {
public:
    explicit ThreadPool(size_t worker_count, size_t queue_capacity);
    ~ThreadPool();

    void enqueue(std::function<void()> job);
    void close();
    void join();
};
```

The existing `AsyncVisualizationWriter` can later be simplified to use these primitives, but the first integration should avoid touching visualization unless needed.

## DataLoader Module

`modules/dataloader/` uses a generic `TensorBatch` rather than C++ templates as the first reusable API:

```cpp
using TensorBatch = std::unordered_map<std::string, torch::Tensor>;
```

This supports current training keys:

- `view_a`
- `view_b`
- `warp_a_to_b`
- `valid_mask`

and future tasks can add keys such as `label`, `image`, `mask`, `metadata_index`, or `sample_weight`.

### Dataset

Responsibilities:

- Provide deterministic sample count.
- Load one sample by index.
- Throw clear exceptions for invalid indices and malformed tensors.

Expected interface:

```cpp
class TensorDataset {
public:
    virtual ~TensorDataset() = default;
    virtual size_t size() const = 0;
    virtual TensorBatch get(size_t index) = 0;
};
```

PlanetaryFeatureMatch-specific adapters:

- `SyntheticPairTensorDataset`: wraps online synthetic pair generation.
- `CachedSyntheticPairTensorDataset`: wraps `SyntheticPairCacheDataset`.

### Samplers and Splits

Sampler responsibilities:

- Generate index order for one epoch.
- Support deterministic seeding.
- Support sequential and shuffled sampling.
- Support subset ranges for train/validation/test splits.

Split responsibilities:

- Input: dataset size, train ratio, validation ratio, test ratio, seed, shuffle flag.
- Output: three index vectors.
- Ratios must sum to 1.0 within tolerance.
- Every sample appears in exactly one split.

Initial sampler types:

- `SequentialSampler`
- `ShuffleSampler`
- `SubsetSampler`

### Collation and Padding

`TensorBatchCollator` stacks per-sample `TensorBatch` values into one batch. It handles variable-sized image pairs with layout-aware padding, preserving the current trainer behavior:

- CHW tensors such as `view_a` and `view_b` pad H/W on the bottom/right.
- HW tensors such as `valid_mask` pad H/W.
- HWC tensors such as `warp_a_to_b` pad H/W.

The collator should reject unsupported layouts and missing keys with clear exceptions.

### AsyncDataLoader

Responsibilities:

- Spawn worker threads.
- Pull indices from sampler.
- Load and preprocess samples in workers.
- Collate into batches.
- Prefetch a bounded number of batches.
- Surface worker exceptions to the main thread.
- Stop cleanly at epoch end or destruction.

Expected interface:

```cpp
struct DataLoaderOptions {
    size_t batch_size = 1;
    size_t worker_count = 0;
    size_t prefetch_batches = 2;
    bool drop_last = false;
    bool pin_memory = false;
};

class AsyncDataLoader {
public:
    AsyncDataLoader(
        std::shared_ptr<TensorDataset> dataset,
        std::unique_ptr<Sampler> sampler,
        TensorBatchCollator collator,
        DataLoaderOptions options);

    void reset();
    std::optional<TensorBatch> next();
};
```

If `worker_count == 0`, the loader runs synchronously. This keeps tests deterministic and gives a CPU-only fallback.

### Pinned Memory

Pinned-memory behavior:

- Enabled only when `pin_memory=true`.
- Only CPU tensors are copied to pinned memory.
- Non-tensor metadata can be ignored in the first version because `TensorBatch` only stores tensors.
- CUDA availability is not required to allocate pinned tensors if LibTorch supports it; if pinned allocation fails, the loader should throw a clear exception rather than silently changing behavior.

Trainer-side transfer:

```cpp
auto batch_on_device = move_batch_to_device(batch, device, /*non_blocking=*/config.device.is_cuda());
```

## Augmentation Module

`modules/augment/` owns reusable image-pair augmentation and correspondence generation.

### Configuration

`ImagePairAugmentationConfig` includes:

- profile: `Mild`, `Medium`, `Hard`, `Extreme`, `Mixed`
- max rotation degrees
- max translation fraction
- min/max scale
- perspective strength
- brightness range
- contrast range
- gamma range
- shadow strength
- noise standard deviation
- source index
- variant index
- seed
- extreme pair ratio

The first implementation can migrate current affine, rotation, scale, brightness, contrast, gamma, shadow, and noise behavior. Perspective should be represented in the config and added if current geometry utilities support it cleanly; otherwise it remains disabled by default but the API should not block adding it later.

### TransformSampler

`TransformSampler` converts `source_index`, `variant_index`, `seed`, and profile into deterministic transform parameters. Determinism is required so cache validation and tests remain stable.

### ImagePairAugmentor

Responsibilities:

- Accept one source image tensor and generate two related views.
- Apply geometric transform to produce `view_b`.
- Produce `warp_a_to_b` dense correspondence field.
- Produce `valid_mask`.
- Apply photometric augmentation without breaking tensor shape or dtype.
- Return a `TensorBatch` or a typed `ImagePairSample` that can be converted to `TensorBatch`.

Expected output keys match current training and cache format:

```cpp
struct ImagePairSample {
    torch::Tensor view_a;
    torch::Tensor view_b;
    torch::Tensor warp_a_to_b;
    torch::Tensor valid_mask;
};
```

### Compatibility Wrapper

Existing `make_synthetic_pair()` remains as a compatibility wrapper:

```cpp
SyntheticPair make_synthetic_pair(const torch::Tensor& image, const SyntheticPairConfig& config) {
    return toSyntheticPair(ImagePairAugmentor(toAugmentConfig(config)).augment(image));
}
```

This keeps existing trainer, cache, and tests working while the new module becomes the canonical implementation.

## Logging Module

`modules/logging/` replaces direct trainer `std::cout` logging with structured metric sinks.

### TrainingMetric

Use a flexible metric record:

```cpp
struct TrainingMetric {
    int epoch = 0;
    int total_epochs = 0;
    int iteration = 0;
    int total_iterations = 0;
    int images_seen = 0;
    int total_images = 0;
    double learning_rate = 0.0;
    double elapsed_seconds = 0.0;
    std::unordered_map<std::string, double> values;
};
```

`values` stores losses and diagnostics:

- `loss_total`
- `feature_loss`
- `repeatability_loss`
- `descriptor_loss`
- `matcher_loss`
- `graph_matching_loss`
- `dense_loss`
- `offset_loss`
- `confidence_loss`
- `descriptor_accuracy`
- `descriptor_diversity`
- `offset_error_px`
- `gpu_utilization_percent`
- `gpu_power_watts`

### ConsoleProgressLogger

Responsibilities:

- Render a single-line progress bar during an epoch.
- Show core fields: epoch, iteration, loss, matcher loss, dense loss, offset error, learning rate, GPU utilization, GPU power.
- Print epoch summary on a new line.
- Avoid flooding the terminal with one full line per batch unless configured.

Example:

```text
epoch 100/100 [====================>---] 255/289 loss=9.438 matcher=9.310 dense=0.128 offset_px=31.95 lr=3e-4 gpu=82% power=210W
```

### CsvMetricLogger

Responsibilities:

- Open a CSV file path from config.
- Write header once.
- Write one row per iteration.
- Flush periodically or at epoch end.
- Include all metric fields, including GPU values if available.

Default file name when enabled:

```text
train_log.csv
```

The first CLI integration can use a high-level option such as `--log-csv train_log.csv` if the project wants explicit control. If avoiding new CLI options is preferred, default to writing CSV only when a training output/log directory is configured.

### GPU Metrics

`GpuMetricProvider` interface:

```cpp
struct GpuMetrics {
    std::optional<double> utilization_percent;
    std::optional<double> power_watts;
};

class GpuMetricProvider {
public:
    virtual ~GpuMetricProvider() = default;
    virtual GpuMetrics sample() = 0;
};
```

Implementations:

- `NullGpuMetricProvider`: returns empty values and is always available.
- `NvmlGpuMetricProvider`: compiled only when NVML is found and `PFM_ENABLE_NVML=ON`.

If NVML initialization fails at runtime, logging should fall back to null metrics with one warning, not fail training.

## Trainer Integration Phases

### Phase 1: Augmentation and Logging

- Move synthetic pair augmentation implementation into `modules/augment/`.
- Keep `make_synthetic_pair()` public API working.
- Add structured logging module.
- Replace direct trainer progress `std::cout` with `MetricLoggerGroup`.
- Preserve current loss names and values in CSV.
- Add optional NVML metrics.

This phase gives better logs and a cleaner augmentation boundary without changing the training data loading model.

### Phase 2: DataLoader Infrastructure

- Add `runtime` and `dataloader` modules.
- Add synchronous DataLoader mode and tests.
- Add async worker mode and tests.
- Add pinned-memory prefetch tests where supported.
- Keep trainer on existing loop until DataLoader is independently stable.

### Phase 3: Trainer Uses AsyncDataLoader

- Wrap online synthetic pairs and cached synthetic pairs as `TensorDataset` implementations.
- Replace manual per-batch pair loading with `AsyncDataLoader`.
- Main thread receives prefetched `TensorBatch`, moves it to the target device, runs forward/backward.
- Keep existing visualization behavior and checkpoint behavior unchanged.

### Phase 4: Matcher-Specific Data Strategy

After the infrastructure exists, use its sampler hooks to address high `graph_matching_loss`:

- hard/easy profile scheduling
- balanced positive/dustbin target sampling
- validation split metrics for matcher loss
- profile-wise CSV metrics to see whether extreme pairs dominate failures

This phase is intentionally separate from the infrastructure work.

## CMake Design

Add internal library targets:

```cmake
add_library(pfm_runtime STATIC ...)
add_library(pfm_dataloader STATIC ...)
add_library(pfm_augment STATIC ...)
add_library(pfm_logging STATIC ...)
```

Dependencies:

- `pfm_runtime`: standard C++ threads only.
- `pfm_dataloader`: `pfm_runtime`, Torch.
- `pfm_augment`: Torch, OpenCV if needed, existing geometry/data utilities.
- `pfm_logging`: standard C++; optional NVML.
- `pfm`: links the new module targets and existing source files.

NVML option:

```cmake
option(PFM_ENABLE_NVML "Enable NVML GPU metrics" ON)
```

If NVML is not found, configure continues and `pfm_logging` builds with `NullGpuMetricProvider` only.

Tests remain in `pfm_tests` following the current project pattern, with new registrations in `tests/test_main.cpp`.

## Error Handling

- Invalid queue capacity, worker count, split ratios, batch size, or missing batch keys throw `std::invalid_argument`.
- Worker exceptions are captured and rethrown on `AsyncDataLoader::next()` or `ThreadPool::join()`.
- NVML failure does not fail training; it returns null GPU metrics after one warning.
- Pinned-memory allocation failure throws with a clear message when `pin_memory=true`.
- CSV file open failure throws before training starts.

## Testing Strategy

Follow TDD for every module.

Runtime tests:

- Blocking queue preserves FIFO order.
- `pop()` blocks until data or close.
- `close()` wakes consumers.
- ThreadPool runs all jobs.
- ThreadPool rethrows worker exceptions.

DataLoader tests:

- Sequential sampler returns deterministic order.
- Shuffle sampler is deterministic with seed.
- train/val/test split covers all indices exactly once.
- Collator pads CHW/HW/HWC tensors correctly.
- Synchronous loader returns expected batches.
- Async loader returns all samples and propagates worker errors.
- Pinned-memory option produces pinned tensors when supported.

Augmentation tests:

- Profile sampling is deterministic.
- Output shapes and dtypes match current synthetic pair behavior.
- Valid mask and warp remain aligned after transform.
- Existing synthetic pair tests pass through compatibility wrapper.

Logging tests:

- Console logger renders progress fields and handles missing GPU metrics.
- CSV logger writes stable header and rows.
- Null GPU provider returns empty metrics.
- NVML provider is behind CMake detection and can be excluded from tests when unavailable.

Trainer integration tests:

- Existing trainer tests still pass.
- CSV logging writes loss columns.
- Augmentation wrapper preserves cache compatibility.
- Later DataLoader integration keeps training smoke passing.

## Documentation Updates

Update:

- `README.md`: new reusable training infrastructure modules and high-level training log behavior.
- `docs/training.md`: DataLoader, CSV logs, GPU metrics, progress bar, train/validation/test split and sampler behavior.
- `docs/usage.md`: command examples for enabling CSV logs and any high-level split options if added.

Document that high `graph_matching_loss` after feature convergence means matcher supervision/data strategy still needs dedicated optimization, and that the new CSV logs make that diagnosis easier.

## Open Decisions Resolved

- Rollout: phased landing, not one large rewrite.
- GPU metrics: optional NVML dependency with null fallback.
- Directory layout: new submodules under existing `modules/`.
- DataLoader API: `TensorBatch` first, not template-heavy Dataset first.
