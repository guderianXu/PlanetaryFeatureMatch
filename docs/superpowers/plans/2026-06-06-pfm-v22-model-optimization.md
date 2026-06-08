# PFM v22 Model Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-C4 v22 model path with strict descriptors, continuous geometry, matchability/uncertainty/no-match heads, richer GraphMatcher metadata, and hard-negative/rotation-ready training hooks.

**Architecture:** Keep the existing backbone and v21 module boundaries, but extend sparse-head outputs and feature metadata. Descriptor matching remains strict cosine; rotation robustness comes from continuous geometry and consistency losses. Training uses scheduleable loss terms so the code lands once while difficult objectives can be enabled progressively.

**Tech Stack:** Python/PyTorch training and evaluation, C++17/LibTorch/OpenCV inference/training parity, existing `pfm_tests` harness and Python `unittest`.

---

### Task 1: Python Sparse Head Reliability Outputs

**Files:**
- Modify: `python/pfm_model.py`
- Test: `python/test_pfm_model.py`

- [ ] **Step 1: Write the failing test**

Add a test that checks `SparseHeadOutput` exposes matchability, descriptor uncertainty, and no-match prior with stable shapes and bounded values:

```python
def test_sparse_head_outputs_reliability_maps(self):
    model = pfm_model.PlanetaryFeatureMatcher(input_channels=1, base_channels=8, descriptor_dim=8)
    image = torch.rand(1, 1, 16, 16)

    sparse, _ = model(image)

    self.assertEqual(sparse.matchability.shape, sparse.heatmap.shape)
    self.assertEqual(sparse.descriptor_uncertainty.shape, sparse.heatmap.shape)
    self.assertEqual(sparse.no_match_prior.shape, sparse.heatmap.shape)
    for tensor in (sparse.matchability, sparse.descriptor_uncertainty, sparse.no_match_prior):
        self.assertTrue(bool(torch.all(tensor >= 0.0)))
        self.assertTrue(bool(torch.all(tensor <= 1.0)))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.test_pfm_model.PfmModelTest.test_sparse_head_outputs_reliability_maps
```

Expected: FAIL because `SparseHeadOutput` does not yet have the three fields.

- [ ] **Step 3: Write minimal implementation**

Update `SparseHeadOutput` and `SparseHead`:

```python
@dataclass
class SparseHeadOutput:
    heatmap: torch.Tensor
    descriptors: torch.Tensor
    scale: torch.Tensor
    orientation: torch.Tensor
    affine: torch.Tensor
    keypoint_offsets: torch.Tensor
    matchability: torch.Tensor
    descriptor_uncertainty: torch.Tensor
    no_match_prior: torch.Tensor
```

Add three 1x1 heads initialized to neutral logits:

```python
self.matchability = nn.Conv2d(input_channels, 1, 1)
self.descriptor_uncertainty = nn.Conv2d(input_channels, 1, 1)
self.no_match_prior = nn.Conv2d(input_channels, 1, 1)
_zero_module(self.matchability)
_zero_module(self.descriptor_uncertainty)
_zero_module(self.no_match_prior)
```

In `forward()`:

```python
matchability = torch.sigmoid(self.matchability(geometry_context))
descriptor_uncertainty = torch.sigmoid(self.descriptor_uncertainty(geometry_context))
no_match_prior = torch.sigmoid(self.no_match_prior(geometry_context))
return SparseHeadOutput(
    heatmap,
    descriptors,
    scale,
    orientation,
    affine,
    keypoint_offsets,
    matchability,
    descriptor_uncertainty,
    no_match_prior,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run the same command. Expected: PASS.

- [ ] **Step 5: Run focused Python model tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python/test_pfm_model.py
```

Expected: OK.

### Task 2: Python Metadata Uses Reliability Fields

**Files:**
- Modify: `python/pfm_model.py`
- Modify: `python/pytorch_cache_match_eval.py`
- Modify: `python/pfm_pytorch_training.py`
- Test: `python/test_pfm_model.py`
- Test: `python/test_pfm_pytorch_training.py`

- [ ] **Step 1: Write failing metadata test**

Add a test that sampled graph metadata includes matchability, uncertainty, and no-match prior when provided:

```python
def test_graph_metadata_uses_sparse_reliability_fields(self):
    keypoints = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    matchability = torch.tensor([[0.8], [0.2]])
    uncertainty = torch.tensor([[0.1], [0.7]])
    no_match = torch.tensor([[0.05], [0.9]])

    metadata = pfm_model.graph_keypoint_metadata(
        keypoints,
        meta_dim=16,
        matchability=matchability,
        descriptor_uncertainty=uncertainty,
        no_match_prior=no_match,
    )

    self.assertTrue(torch.allclose(metadata[:, 12:13], matchability))
    self.assertTrue(torch.allclose(metadata[:, 14:15], uncertainty))
    self.assertTrue(torch.allclose(metadata[:, 15:16], no_match))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.test_pfm_model.PfmModelTest.test_graph_metadata_uses_sparse_reliability_fields
```

Expected: FAIL because the metadata helper does not accept these keyword arguments.

- [ ] **Step 3: Implement metadata extension**

Extend `graph_keypoint_metadata()` to accept optional tensors. Keep existing 16-column layout compatible:

```text
0:2 xy
2:4 normalized xy proxy
4:6 orientation
6:10 affine
10 quality
11 contrast
12 matchability
13 raw descriptor margin placeholder
14 descriptor uncertainty
15 no_match_prior
```

Default missing reliability fields to existing neutral values:

```python
matchability_column = quality_column if matchability is None else matchability.to(...)
margin_column = keypoints.new_zeros((count, 1))
uncertainty_column = (1.0 - quality_column).clamp(0.0, 1.0) if descriptor_uncertainty is None else descriptor_uncertainty.to(...)
no_match_column = keypoints.new_zeros((count, 1)) if no_match_prior is None else no_match_prior.to(...)
```

- [ ] **Step 4: Thread sampled maps through evaluation/training**

When keypoints are selected, sample the three sparse reliability maps with `sample_map_rows_at_keypoints()` and pass them to graph metadata builders. For training code paths that only have points, keep neutral defaults.

- [ ] **Step 5: Run focused tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python/test_pfm_model.py python/test_pfm_pytorch_training.py
```

Expected: OK.

### Task 3: C++ Sparse Head Reliability Parity

**Files:**
- Modify: `src/models/pfm_model_v21.h`
- Modify: `src/models/pfm_model_v21.cpp`
- Modify: `src/models/head_outputs.h`
- Modify: `src/train/trainer.cpp`
- Test: `src/models/pfm_model_v21_test.cpp`
- Test: `src/train/trainer_test.cpp`

- [ ] **Step 1: Write failing C++ model test**

Add a test:

```cpp
static void pfm_v21_sparse_head_outputs_reliability_maps()
{
    auto head = pfm::v21::PfmV21SparseHead(8, 8);
    auto input = torch::rand({1, 8, 4, 4}, torch::kFloat32);
    const auto output = head->forward(input);

    PFM_REQUIRE(output.matchability.sizes() == output.heatmap.sizes());
    PFM_REQUIRE(output.descriptor_uncertainty.sizes() == output.heatmap.sizes());
    PFM_REQUIRE(output.no_match_prior.sizes() == output.heatmap.sizes());
    PFM_REQUIRE(output.matchability.min().item<float>() >= 0.0F);
    PFM_REQUIRE(output.matchability.max().item<float>() <= 1.0F);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
LIBRARY_PATH=/home/w24/anaconda3/envs/cppTorch/lib:$LIBRARY_PATH LD_LIBRARY_PATH=/home/w24/anaconda3/envs/cppTorch/lib:$LD_LIBRARY_PATH cmake --build build -j$(nproc)
```

Expected: compile failure because `PfmV21SparseHeadOutput` lacks reliability fields.

- [ ] **Step 3: Implement C++ output fields**

Add tensors to both sparse output structs:

```cpp
torch::Tensor matchability;
torch::Tensor descriptor_uncertainty;
torch::Tensor no_match_prior;
```

Add registered modules to `PfmV21SparseHeadImpl`:

```cpp
torch::nn::Conv2d _matchability{nullptr};
torch::nn::Conv2d _descriptor_uncertainty{nullptr};
torch::nn::Conv2d _no_match_prior{nullptr};
```

Initialize and return:

```cpp
_matchability = register_module("matchability", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 1, 1)));
_descriptor_uncertainty =
    register_module("descriptor_uncertainty", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 1, 1)));
_no_match_prior = register_module("no_match_prior", torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_channels, 1, 1)));
zeroConv(_matchability);
zeroConv(_descriptor_uncertainty);
zeroConv(_no_match_prior);
```

```cpp
auto matchability = torch::sigmoid(_matchability->forward(geometry_context));
auto descriptor_uncertainty = torch::sigmoid(_descriptor_uncertainty->forward(geometry_context));
auto no_match_prior = torch::sigmoid(_no_match_prior->forward(geometry_context));
return PfmV21SparseHeadOutput{heatmap, descriptors, scale, orientation, affine, keypoint_offsets,
                              matchability, descriptor_uncertainty, no_match_prior};
```

- [ ] **Step 4: Adapt trainer sparse output conversions**

Update every `SparseHeadOutput{...}` initializer in `src/train/trainer.cpp` and tests to include reliability maps. For missing maps, use:

```cpp
auto neutral = sparse.heatmap.new_full(sparse.heatmap.sizes(), 0.5);
```

- [ ] **Step 5: Run C++ tests**

Run:

```bash
LIBRARY_PATH=/home/w24/anaconda3/envs/cppTorch/lib:$LIBRARY_PATH LD_LIBRARY_PATH=/home/w24/anaconda3/envs/cppTorch/lib:$LD_LIBRARY_PATH cmake --build build -j$(nproc)
LD_LIBRARY_PATH=/home/w24/anaconda3/envs/cppTorch/lib:$LD_LIBRARY_PATH ./build/pfm_tests
```

Expected: 547+ tests pass.

### Task 4: Matchability-Aware Feature Selection

**Files:**
- Modify: `python/pytorch_cache_match_eval.py`
- Modify: `src/infer/feature_extractor.cpp`
- Modify: `src/infer/pipeline.cpp`
- Test: `python/test_pytorch_cache_match_eval.py`
- Test: `src/infer/feature_extractor_test.cpp`

- [ ] **Step 1: Write failing selection tests**

Python: create two equal-heatmap points with different matchability and assert the higher matchability point is selected first.

C++: add a decode test that sets heatmap equal and matchability different, then asserts selected score follows `heatmap * matchability * (1 - uncertainty)`.

- [ ] **Step 2: Implement scoring**

Compute selection score:

```text
effective_score = heatmap * matchability * (1 - descriptor_uncertainty)
```

Keep `no_match_prior` out of keypoint selection initially; it should influence GraphMatcher dustbin, not suppress all candidates.

- [ ] **Step 3: Preserve compatibility**

If reliability maps are undefined, use:

```text
matchability = 1
descriptor_uncertainty = 0
```

- [ ] **Step 4: Run focused tests**

Run Python cache eval tests and C++ `pfm_tests`.

### Task 5: Reliability and No-Match Losses

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Modify: `src/train/trainer.cpp`
- Test: `python/test_pfm_pytorch_training.py`
- Test: `src/train/trainer_test.cpp`

- [ ] **Step 1: Write failing loss tests**

Add tests for:

```python
def test_matchability_loss_rewards_valid_positive_points(self): ...
def test_no_match_prior_loss_rewards_unmatched_points(self): ...
def test_descriptor_uncertainty_loss_increases_on_high_score_wrong_matches(self): ...
```

- [ ] **Step 2: Implement Python losses**

Add:

```python
def matchability_supervision_loss(matchability, positive_points_xy, negative_points_xy): ...
def no_match_prior_supervision_loss(no_match_prior, no_match_points_xy, positive_points_xy): ...
def descriptor_uncertainty_supervision_loss(uncertainty, false_match_points_xy, positive_points_xy): ...
```

Use sampled map values and binary cross entropy.

- [ ] **Step 3: Add CLI weights**

Add args:

```text
--matchability-weight
--descriptor-uncertainty-weight
--no-match-prior-weight
```

Default all to `0.0` for compatibility.

- [ ] **Step 4: Mirror C++ train config**

Add config fields and validations in `src/train/trainer.h` and `src/train/trainer.cpp`. Default to zero.

- [ ] **Step 5: Run focused tests**

Run Python training tests and C++ `pfm_tests`.

### Task 6: Rotation and Affine Consistency Hooks

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Modify: `src/train/trainer.cpp`
- Test: `python/test_pfm_pytorch_training.py`
- Test: `src/train/trainer_test.cpp`

- [ ] **Step 1: Write failing tests**

Add tests that verify:

```text
orientation consistency is near zero when predicted orientation delta matches a known rotation
scale consistency is near zero when predicted scale follows synthetic resize
affine consistency is near zero for identity affine under identity warp
```

- [ ] **Step 2: Implement loss functions**

Add scheduleable functions:

```python
orientation_consistency_loss(...)
scale_consistency_loss(...)
affine_consistency_loss(...)
rotation_descriptor_consistency_loss(...)
```

Use existing warp sampling utilities. All weights default to `0.0`.

- [ ] **Step 3: Add args/config**

Add:

```text
--orientation-consistency-weight
--scale-consistency-weight
--affine-consistency-weight
--rotation-descriptor-consistency-weight
```

- [ ] **Step 4: Run tests**

Run focused Python/C++ training tests.

### Task 7: Hard Negative and Mining Interfaces

**Files:**
- Modify: `python/pfm_pytorch_training.py`
- Modify: `python/pytorch_cache_match_eval.py`
- Modify: `src/infer/cache_match_eval.cpp`
- Modify: `src/infer/cache_match_eval.h`
- Test: `python/test_pfm_pytorch_training.py`
- Test: `python/test_pytorch_cache_match_eval.py`
- Test: `src/infer/cache_match_eval_test.cpp`

- [ ] **Step 1: Write failing CSV schema tests**

Assert exported false-match mining rows contain:

```text
pair_path, source_x, source_y, target_x, target_y, raw_score, graph_score,
warp_error, texture_score, angle_bucket, error_type
```

- [ ] **Step 2: Implement error bucket classification**

Classify into:

```text
near_wrong, far_wrong, same_texture_wrong, weak_texture_wrong, forced_no_match_wrong, rotation_wrong
```

- [ ] **Step 3: Training loader accepts mined negatives**

Extend existing false-match CSV support to consume the richer schema while preserving old schema compatibility.

- [ ] **Step 4: Run tests**

Run Python cache eval/training tests and C++ `pfm_tests`.

### Task 8: Full Verification

**Files:**
- No new production files.

- [ ] **Step 1: Python verification**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python/test_pfm_model.py python/test_pytorch_cache_match_eval.py python/test_pfm_pytorch_training.py
```

Expected: OK.

- [ ] **Step 2: C++ verification**

Run:

```bash
cmake -S . -B build -DBUILD_TESTS=ON -DTorch_DIR=/home/w24/anaconda3/envs/cppTorch/lib/python3.12/site-packages/torch/share/cmake/Torch -DOpenCV_DIR=/home/w24/anaconda3/envs/cppTorch/lib/cmake/opencv4
LIBRARY_PATH=/home/w24/anaconda3/envs/cppTorch/lib:$LIBRARY_PATH LD_LIBRARY_PATH=/home/w24/anaconda3/envs/cppTorch/lib:$LD_LIBRARY_PATH cmake --build build -j$(nproc)
LD_LIBRARY_PATH=/home/w24/anaconda3/envs/cppTorch/lib:$LD_LIBRARY_PATH ./build/pfm_tests
LD_LIBRARY_PATH=/home/w24/anaconda3/envs/cppTorch/lib:$LD_LIBRARY_PATH ctest --test-dir build --output-on-failure
```

Expected: build succeeds, `pfm_tests` passes, `ctest` passes.

---

## Self-Review

Spec coverage: The plan covers model reliability heads, strict descriptor compatibility, GraphMatcher metadata, matchability-aware selection, new no-match/uncertainty losses, continuous rotation/affine hooks, mined hard-negative interfaces, and full verification.

Placeholder scan: No task contains TBD/TODO placeholders. Later tasks reference functions introduced in earlier tasks or existing project files.

Type consistency: Python names use `matchability`, `descriptor_uncertainty`, and `no_match_prior`. C++ uses the same snake_case field names in output structs, matching existing project style for tensor members.
