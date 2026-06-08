# Cross-Domain Lazy Pair Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add controlled lazy training pairs across same-position viewpoint changes, different camera positions, and different field-of-view manifests.

**Architecture:** Extend `RenderRecord` and `LazyPairSpec` with dataset and pair-type metadata, then split pair construction into small builders for same-position, cross-camera, and cross-FOV families. The trainer keeps using the existing lazy projection path, while pair-family selection is controlled by new CLI arguments and reported in JSON/CSV/HTML artifacts.

**Tech Stack:** Python 3, PyTorch, OpenCV, standard-library `argparse`, existing `python.tests.test_benchmark_lazy_pose_pairs` unittest suite.

---

### Task 1: Pair Metadata And CLI Parsers

**Files:**
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Write failing parser tests**

Add tests covering `--pair-mode`, offset parsing, and type-weight parsing:

```python
def test_parse_args_accepts_cross_domain_pair_options(self) -> None:
    argv = [
        "benchmark_lazy_pose_pairs.py",
        "--render-manifest",
        "fov090/render_manifest.csv",
        "fov110/render_manifest.csv",
        "--output-dir",
        "run",
        "--pair-mode",
        "mixed",
        "--cross-camera-offsets",
        "1,3,7",
        "--cross-fov-offsets",
        "0,2",
        "--cross-pair-variant",
        "nadir",
        "--cross-pair-variant",
        "extreme_01",
        "--pair-type-weights",
        "same_position_view=0.2,cross_camera=0.5,cross_fov=0.3",
    ]

    with mock.patch.object(sys, "argv", argv):
        args = lazy_bench.parse_args()

    self.assertEqual(args.pair_mode, "mixed")
    self.assertEqual(args.cross_camera_offsets, [1, 3, 7])
    self.assertEqual(args.cross_fov_offsets, [0, 2])
    self.assertEqual(args.cross_pair_variant, ["nadir", "extreme_01"])
    self.assertEqual(
        args.pair_type_weights,
        {"same_position_view": 0.2, "cross_camera": 0.5, "cross_fov": 0.3},
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs.BenchmarkLazyPosePairsTest.test_parse_args_accepts_cross_domain_pair_options
```

Expected: FAIL because the parser does not know these arguments.

- [ ] **Step 3: Implement parsers and dataclass fields**

Add constants, parsers, and fields:

```python
PAIR_TYPE_SAME_POSITION_VIEW = "same_position_view"
PAIR_TYPE_CROSS_CAMERA = "cross_camera"
PAIR_TYPE_CROSS_FOV = "cross_fov"
PAIR_TYPES = (PAIR_TYPE_SAME_POSITION_VIEW, PAIR_TYPE_CROSS_CAMERA, PAIR_TYPE_CROSS_FOV)
DEFAULT_PAIR_TYPE_WEIGHTS = {
    PAIR_TYPE_SAME_POSITION_VIEW: 0.40,
    PAIR_TYPE_CROSS_CAMERA: 0.35,
    PAIR_TYPE_CROSS_FOV: 0.25,
}

def parse_int_list(value: str) -> list[int]:
    ...

def parse_pair_type_weights(value: str) -> dict[str, float]:
    ...
```

Extend `RenderRecord`:

```python
dataset_id: str
raw_base_id: str
```

Extend `LazyPairSpec`:

```python
pair_type: str = PAIR_TYPE_SAME_POSITION_VIEW
```

Add parser arguments:

```python
parser.add_argument("--pair-mode", choices=["same-position", "cross-camera", "cross-fov", "mixed"], default="same-position")
parser.add_argument("--cross-camera-offsets", type=parse_int_list, default=parse_int_list("1,2,4,8"))
parser.add_argument("--cross-fov-offsets", type=parse_int_list, default=parse_int_list("0,1,2,4"))
parser.add_argument("--cross-pair-variant", action="append", default=[])
parser.add_argument("--pair-type-weights", type=parse_pair_type_weights, default=DEFAULT_PAIR_TYPE_WEIGHTS.copy())
```

- [ ] **Step 4: Run parser tests**

Run the same unittest command. Expected: PASS.

### Task 2: Cross-Domain Spec Builders

**Files:**
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Write failing builder tests**

Add tests that create small `RenderRecord` objects with existing temp files and verify:

```python
def test_build_cross_camera_pair_specs_uses_different_base_same_dataset(self) -> None:
    records = self.make_render_records(dataset_id="fov090", base_ids=("b001", "b002"), variants=("nadir", "mid_01"))

    specs = lazy_bench.build_cross_camera_pair_specs(
        records,
        split="train",
        cross_variants=("nadir", "mid_01"),
        offsets=(1,),
        image_source="uint8",
        start_index=0,
    )

    self.assertTrue(specs)
    self.assertTrue(all(spec.pair_type == lazy_bench.PAIR_TYPE_CROSS_CAMERA for spec in specs))
    self.assertTrue(all(spec.reference.raw_base_id != spec.target.raw_base_id for spec in specs))
    self.assertTrue(all(spec.reference.dataset_id == spec.target.dataset_id for spec in specs))

def test_build_cross_fov_pair_specs_uses_different_datasets(self) -> None:
    records = (
        self.make_render_records(dataset_id="fov090", base_ids=("b001",), variants=("nadir",))
        + self.make_render_records(dataset_id="fov110", base_ids=("b001",), variants=("nadir",))
    )

    specs = lazy_bench.build_cross_fov_pair_specs(
        records,
        split="train",
        cross_variants=("nadir",),
        offsets=(0,),
        image_source="uint8",
        start_index=0,
    )

    self.assertEqual(len(specs), 1)
    self.assertEqual(specs[0].pair_type, lazy_bench.PAIR_TYPE_CROSS_FOV)
    self.assertNotEqual(specs[0].reference.dataset_id, specs[0].target.dataset_id)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs.BenchmarkLazyPosePairsTest.test_build_cross_camera_pair_specs_uses_different_base_same_dataset python.tests.test_benchmark_lazy_pose_pairs.BenchmarkLazyPosePairsTest.test_build_cross_fov_pair_specs_uses_different_datasets
```

Expected: FAIL because builder functions are missing.

- [ ] **Step 3: Implement focused builders**

Keep `build_pair_specs()` for same-position compatibility, but set `pair_type=PAIR_TYPE_SAME_POSITION_VIEW`. Add:

```python
def build_cross_camera_pair_specs(...):
    by_dataset_variant = defaultdict(list)
    ...

def build_cross_fov_pair_specs(...):
    by_dataset_variant = defaultdict(list)
    ...

def _eligible_record(record, split, image_source):
    ...
```

Both builders sort records by natural `raw_base_id`, use positive offsets, and skip self-pairs. Cross-FOV pairs iterate dataset-id combinations and only pair different datasets.

- [ ] **Step 4: Run builder tests**

Run the same unittest command. Expected: PASS.

### Task 3: Mixed Pair Mode And Reporting

**Files:**
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Write failing mixed-mode tests**

Add tests for high-level spec construction:

```python
def test_build_lazy_pair_specs_mixed_includes_all_requested_pair_types(self) -> None:
    records = (
        self.make_render_records(dataset_id="fov090", base_ids=("b001", "b002"), variants=("nadir", "mid_01"))
        + self.make_render_records(dataset_id="fov110", base_ids=("b001", "b002"), variants=("nadir", "mid_01"))
    )

    specs, counts = lazy_bench.build_lazy_pair_specs(
        records,
        split="train",
        pair_mode="mixed",
        reference_variant="nadir",
        target_variants=("mid_01",),
        cross_variants=("nadir", "mid_01"),
        cross_camera_offsets=(1,),
        cross_fov_offsets=(0,),
        image_source="uint8",
        limit_pairs=0,
        seed=123,
        shuffle=False,
    )

    self.assertGreater(counts[lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW], 0)
    self.assertGreater(counts[lazy_bench.PAIR_TYPE_CROSS_CAMERA], 0)
    self.assertGreater(counts[lazy_bench.PAIR_TYPE_CROSS_FOV], 0)
    self.assertEqual(sum(counts.values()), len(specs))
```

- [ ] **Step 2: Run test to verify it fails**

Run the specific unittest. Expected: FAIL because `build_lazy_pair_specs()` is missing.

- [ ] **Step 3: Implement high-level builder**

Add `build_lazy_pair_specs()` to select families by `pair_mode`, assign continuous `pair_index` values, optionally shuffle, and apply `limit_pairs`. Return `(specs, pair_type_counts)`.

Add validation:

```python
if pair_mode in {"cross-fov", "mixed"} and fewer than two dataset ids exist:
    raise ValueError("cross-fov pair mode requires at least two render manifests/datasets")
```

Update `main()` to call this function and write pair metadata into `input_summary.json`.

- [ ] **Step 4: Run mixed-mode tests**

Run the same unittest. Expected: PASS.

### Task 4: Training Metrics Include Pair Type Counts

**Files:**
- Modify: `scripts/benchmark_lazy_pose_pairs.py`
- Test: `python/tests/test_benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Write failing helper test**

Add a helper test:

```python
def test_count_pair_types_counts_lazy_results(self) -> None:
    specs = [
        lazy_bench.LazyPairSpec(0, "train", self.record("a", "nadir"), self.record("a", "mid_01"), lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW),
        lazy_bench.LazyPairSpec(1, "train", self.record("b", "nadir"), self.record("c", "nadir"), lazy_bench.PAIR_TYPE_CROSS_CAMERA),
    ]

    counts = lazy_bench.count_pair_types(specs)

    self.assertEqual(counts[lazy_bench.PAIR_TYPE_SAME_POSITION_VIEW], 1)
    self.assertEqual(counts[lazy_bench.PAIR_TYPE_CROSS_CAMERA], 1)
    self.assertEqual(counts[lazy_bench.PAIR_TYPE_CROSS_FOV], 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run the specific unittest. Expected: FAIL because `count_pair_types()` is missing.

- [ ] **Step 3: Implement metrics wiring**

Add `count_pair_types()`. Add train CSV columns:

```text
pair_type_same_position_view
pair_type_cross_camera
pair_type_cross_fov
```

At each step, count `result.spec.pair_type` for consumed results and write the three counts. Also include pair type in preprocess CSV rows.

- [ ] **Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs
```

Expected: OK.

### Task 5: Smoke Verification

**Files:**
- Modify only if tests expose a defect: `scripts/benchmark_lazy_pose_pairs.py`

- [ ] **Step 1: Run unit tests**

Run:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python -m unittest python.tests.test_benchmark_lazy_pose_pairs
```

Expected: OK.

- [ ] **Step 2: Run a one-pair preprocess smoke**

Run against the current copied manifests:

```bash
PYTHONPATH=python:scripts /home/w24/anaconda3/envs/cppTorch/bin/python scripts/benchmark_lazy_pose_pairs.py \
  --render-manifest \
    /media/w24/D/xjw深度学习训练数据/pfm_runs/from_scratch_2048_amp_checkpoint_reliability_20260608_004701_prefetch/manifests/fov090_render_manifest.csv \
    /media/w24/D/xjw深度学习训练数据/pfm_runs/from_scratch_2048_amp_checkpoint_reliability_20260608_004701_prefetch/manifests/fov110_render_manifest.csv \
  --output-dir /tmp/pfm_cross_domain_lazy_pair_smoke \
  --mode preprocess \
  --pair-mode mixed \
  --cross-camera-offsets 1 \
  --cross-fov-offsets 0 \
  --cross-pair-variant nadir \
  --target-variant mid_01 \
  --pairs 2 \
  --workers 1 \
  --prefetch-batches 1 \
  --crop-size 512 \
  --image-source uint8 \
  --min-valid-fraction 0.0 \
  --skip-bad-pairs
```

Expected: command exits 0 and `input_summary.json` contains nonzero `same_position_view`, `cross_camera`, and `cross_fov` spec counts.

- [ ] **Step 3: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.
