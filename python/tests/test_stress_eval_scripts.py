import unittest

import torch

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from patch_descriptor_training import SyntheticPair
from continuous_rotation_stress_eval import rotate_pair_from_view
from illumination_stress_eval import make_illumination_variants
from benchmark_lazy_pose_pairs import (
    CropWindow,
    LazyPairSpec,
    LazyPairResult,
    PAIR_TYPE_CROSS_CAMERA,
    RenderRecord,
    write_pair_spec_manifest,
)
import visualize_lazy_pose_matches as visual_mod
import training_visual_report as training_report_mod
import run_graph_depth_ablation as depth_ablation_mod
import run_graph_filter_sweep as filter_sweep_mod
from visualize_lazy_pose_matches import (
    LazyMatchVisual,
    filter_visual_matches,
    make_illumination_stress_lazy_results,
    selected_draw_indices,
)


class StressEvalScriptsTest(unittest.TestCase):
    def test_graph_depth_ablation_parses_unique_positive_depths(self) -> None:
        self.assertEqual(depth_ablation_mod.parse_depths("1,2,2,4"), [1, 2, 4])
        with self.assertRaises(ValueError):
            depth_ablation_mod.parse_depths("0,2")

    def test_graph_depth_ablation_builds_visual_command_with_depth_control(self) -> None:
        args = SimpleNamespace(
            render_manifest=Path("render.csv"),
            uint8_manifest=Path("uint8.csv"),
            pytorch_state=Path("model.pt"),
            run_dir=None,
            metrics_csv=None,
            split="train",
            reference_variant="nadir",
            pair_mode="spatial-index",
            image_source="uint8",
            candidate_pairs=12,
            select_count=4,
            seed=7,
            crop_size=2048,
            max_image_size=768,
            device="cuda",
            descriptor_mode="learned",
            keypoint_score_mode="learned",
            max_keypoints=512,
            max_matches=0,
            draw_matches=0,
            threshold_px=5.0,
            graph_max_attention_work_fraction=0.5,
            graph_width_prune_keep_ratio=0.75,
            graph_width_prune_min_score=-1.0,
            graph_early_stop_min_confidence=-1.0,
            graph_dustbin_delta=0.1,
            graph_acceptance_margin=0.2,
            graph_min_raw_score=0.3,
            graph_min_raw_margin=0.04,
            graph_min_accept_probability=0.7,
            filtered_geometry_filter="local",
            filtered_min_margin=0.02,
            filtered_min_score=-1.0,
            filtered_max_matches=0,
            filtered_draw_matches=0,
            pair_spec_manifest=Path("pairs.csv"),
            target_variant=["mid_01"],
            cross_pair_variant=["mid_01"],
            cross_camera_offsets="1,3",
            cross_fov_offsets="0,2",
            pair_type_weights="same_position_view=0,cross_camera=1,cross_fov=0",
            spatial_index_height_km="100",
            spatial_index_planet_radius_m=3396190.0,
            spatial_index_footprint_samples=5,
            spatial_index_margin_m=2000.0,
            shuffle=True,
            filtered_report=True,
            filtered_mutual=True,
            illumination_stress=False,
            input_local_contrast=False,
            input_local_contrast_strength=0.0,
            input_local_contrast_kernel=31,
        )

        command = depth_ablation_mod.build_visual_command(args, depth=2, report_dir=Path("out/layers_2"))

        self.assertIn("--graph-max-attention-layers", command)
        self.assertEqual(command[command.index("--graph-max-attention-layers") + 1], "2")
        self.assertIn("--graph-max-attention-work-fraction", command)
        self.assertEqual(command[command.index("--graph-max-attention-work-fraction") + 1], "0.5")
        self.assertIn("--graph-width-prune-keep-ratio", command)
        self.assertEqual(command[command.index("--graph-width-prune-keep-ratio") + 1], "0.75")
        self.assertIn("--graph-dustbin-delta", command)
        self.assertEqual(command[command.index("--graph-dustbin-delta") + 1], "0.1")
        self.assertIn("--graph-acceptance-margin", command)
        self.assertEqual(command[command.index("--graph-acceptance-margin") + 1], "0.2")
        self.assertIn("--graph-min-raw-score", command)
        self.assertEqual(command[command.index("--graph-min-raw-score") + 1], "0.3")
        self.assertIn("--graph-min-raw-margin", command)
        self.assertEqual(command[command.index("--graph-min-raw-margin") + 1], "0.04")
        self.assertIn("--graph-min-accept-probability", command)
        self.assertEqual(command[command.index("--graph-min-accept-probability") + 1], "0.7")

    def test_training_visual_report_parse_args_accepts_lightglue_graph_options(self) -> None:
        argv = [
            "training_visual_report.py",
            "--run-dir",
            "run",
            "--validation-cache-dir",
            "val",
            "--graph-width-prune-min-score",
            "0.25",
            "--graph-early-stop-min-confidence",
            "0.85",
            "--graph-max-attention-layers",
            "2",
            "--graph-max-attention-work-fraction",
            "0.5",
            "--graph-width-prune-keep-ratio",
            "0.75",
            "--graph-inference-preset",
            "fast",
            "--graph-min-accept-probability",
            "0.7",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = training_report_mod.parse_args()

        self.assertEqual(args.graph_width_prune_min_score, 0.25)
        self.assertEqual(args.graph_early_stop_min_confidence, 0.85)
        self.assertEqual(args.graph_max_attention_layers, 2)
        self.assertEqual(args.graph_max_attention_work_fraction, 0.5)
        self.assertEqual(args.graph_width_prune_keep_ratio, 0.75)
        self.assertEqual(args.graph_inference_preset, "fast")
        self.assertEqual(args.graph_min_accept_probability, 0.7)

    def test_lazy_visual_parse_args_accepts_graph_depth_controls(self) -> None:
        argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "report",
            "--matcher-mode",
            "graph_matcher",
            "--graph-max-attention-layers",
            "2",
            "--graph-max-attention-work-fraction",
            "0.5",
            "--graph-width-prune-keep-ratio",
            "0.75",
            "--graph-dustbin-delta",
            "0.1",
            "--graph-acceptance-margin",
            "0.2",
            "--graph-min-raw-score",
            "0.3",
            "--graph-min-raw-margin",
            "0.04",
            "--graph-min-accept-probability",
            "0.7",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = visual_mod.parse_args()

        self.assertEqual(args.graph_max_attention_layers, 2)
        self.assertEqual(args.graph_max_attention_work_fraction, 0.5)
        self.assertEqual(args.graph_width_prune_keep_ratio, 0.75)
        self.assertEqual(args.graph_dustbin_delta, 0.1)
        self.assertEqual(args.graph_acceptance_margin, 0.2)
        self.assertEqual(args.graph_min_raw_score, 0.3)
        self.assertEqual(args.graph_min_raw_margin, 0.04)
        self.assertEqual(args.graph_min_accept_probability, 0.7)

    def test_graph_filter_sweep_builds_visual_command(self) -> None:
        config = filter_sweep_mod.GraphFilterConfig(
            min_score=0.05,
            dustbin_delta=0.1,
            acceptance_margin=0.2,
            min_raw_score=0.3,
            min_raw_margin=0.04,
            min_accept_probability=0.7,
        )
        args = SimpleNamespace(
            render_manifest=Path("render.csv"),
            uint8_manifest=Path("uint8.csv"),
            pytorch_state=Path("model.pt"),
            run_dir=None,
            metrics_csv=None,
            split="train",
            reference_variant="nadir",
            pair_mode="spatial-index",
            image_source="uint8",
            candidate_pairs=12,
            select_count=4,
            seed=7,
            crop_size=2048,
            max_image_size=768,
            device="cuda",
            descriptor_mode="learned",
            keypoint_score_mode="learned",
            max_keypoints=512,
            max_matches=0,
            draw_matches=0,
            threshold_px=5.0,
            graph_max_attention_layers=2,
            graph_max_attention_work_fraction=1.0,
            graph_width_prune_keep_ratio=1.0,
            graph_width_prune_min_score=-1.0,
            graph_early_stop_min_confidence=-1.0,
            filtered_geometry_filter="local",
            filtered_min_margin=0.02,
            filtered_min_score=-1.0,
            filtered_max_matches=0,
            filtered_draw_matches=0,
            pair_spec_manifest=Path("pairs.csv"),
            target_variant=["mid_01"],
            cross_pair_variant=["mid_01"],
            cross_camera_offsets="1,3",
            cross_fov_offsets="0,2",
            pair_type_weights="same_position_view=0,cross_camera=1,cross_fov=0",
            spatial_index_height_km="100",
            spatial_index_planet_radius_m=3396190.0,
            spatial_index_footprint_samples=5,
            spatial_index_margin_m=2000.0,
            shuffle=True,
            filtered_report=True,
            filtered_mutual=True,
            illumination_stress=False,
            input_local_contrast=False,
            input_local_contrast_strength=0.0,
            input_local_contrast_kernel=31,
        )

        command = filter_sweep_mod.build_visual_command(args, config=config, report_dir=Path("out/cfg"))

        self.assertIn("--min-score", command)
        self.assertEqual(command[command.index("--min-score") + 1], "0.05")
        self.assertIn("--graph-dustbin-delta", command)
        self.assertEqual(command[command.index("--graph-dustbin-delta") + 1], "0.1")
        self.assertIn("--graph-acceptance-margin", command)
        self.assertEqual(command[command.index("--graph-acceptance-margin") + 1], "0.2")
        self.assertIn("--graph-min-raw-score", command)
        self.assertEqual(command[command.index("--graph-min-raw-score") + 1], "0.3")
        self.assertIn("--graph-min-raw-margin", command)
        self.assertEqual(command[command.index("--graph-min-raw-margin") + 1], "0.04")
        self.assertIn("--graph-min-accept-probability", command)
        self.assertEqual(command[command.index("--graph-min-accept-probability") + 1], "0.7")

    def test_graph_filter_sweep_parses_float_lists_and_slugs_config(self) -> None:
        self.assertEqual(filter_sweep_mod.parse_float_list("-1,0.05,0.5"), [-1.0, 0.05, 0.5])
        config = filter_sweep_mod.GraphFilterConfig(
            min_score=0.05,
            dustbin_delta=-0.1,
            acceptance_margin=0.2,
            min_raw_score=-1.0,
            min_raw_margin=0.04,
            min_accept_probability=0.7,
        )

        self.assertEqual(
            filter_sweep_mod.slug_for_config(config),
            "score0p05_dustneg0p1_accept0p2_rawneg1_margin0p04_prob0p7",
        )

    def test_graph_filter_sweep_summarizes_raw_and_filtered_reports(self) -> None:
        config = filter_sweep_mod.GraphFilterConfig(
            min_score=0.05,
            dustbin_delta=0.1,
            acceptance_margin=0.2,
            min_raw_score=0.3,
            min_raw_margin=0.04,
            min_accept_probability=0.7,
        )
        with tempfile.TemporaryDirectory() as temp:
            report_dir = Path(temp)
            (report_dir / "summary.csv").write_text(
                "label,matches,correct,wrong,precision,median_error_px\n"
                "a,10,7,3,0.700000,2.5\n"
                "b,5,2,3,0.400000,9.0\n",
                encoding="utf-8",
            )
            (report_dir / "filtered_summary.csv").write_text(
                "label,matches,correct,wrong,precision,median_error_px\n"
                "a,4,4,0,1.000000,1.5\n",
                encoding="utf-8",
            )

            summary = filter_sweep_mod.summarize_report(report_dir, config=config)

        self.assertEqual(summary.raw_rows, 2)
        self.assertEqual(summary.raw_matches, 15)
        self.assertEqual(summary.raw_correct, 9)
        self.assertAlmostEqual(summary.raw_precision, 0.6)
        self.assertEqual(summary.filtered_rows, 1)
        self.assertEqual(summary.filtered_matches, 4)
        self.assertEqual(summary.filtered_correct, 4)
        self.assertAlmostEqual(summary.filtered_precision, 1.0)

    def test_lazy_visual_parse_args_defaults_to_filtered_all_match_report(self) -> None:
        argv = [
            "visualize_lazy_pose_matches.py",
            "--render-manifest",
            "render.csv",
            "--uint8-manifest",
            "uint8.csv",
            "--pytorch-state",
            "model.pt",
            "--output-dir",
            "report",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = visual_mod.parse_args()

        self.assertTrue(args.filtered_report)
        self.assertTrue(args.filtered_mutual)
        self.assertEqual(args.filtered_geometry_filter, "local")
        self.assertEqual(args.max_matches, 0)
        self.assertEqual(args.draw_matches, 0)
        self.assertEqual(args.filtered_max_matches, 0)
        self.assertEqual(args.filtered_draw_matches, 0)
        self.assertGreater(args.filtered_min_margin, 0.0)

    def test_lazy_visual_selects_pair_specs_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = RenderRecord(
                pose_id="fov090_a_mid",
                base_id="fov090:a",
                variant="mid_01",
                split="train",
                tsai_path=root / "a.tsai",
                image_path=root / "a.tif",
                uint8_path=root / "a.png",
                depth_path=root / "a_depth.tif",
                dataset_id="fov090",
                raw_base_id="a",
            )
            target = RenderRecord(
                pose_id="fov090_b_extreme",
                base_id="fov090:b",
                variant="extreme_03",
                split="train",
                tsai_path=root / "b.tsai",
                image_path=root / "b.tif",
                uint8_path=root / "b.png",
                depth_path=root / "b_depth.tif",
                dataset_id="fov090",
                raw_base_id="b",
            )
            manifest = root / "overlap_edges.csv"
            pair = SyntheticPair(
                view_a=torch.zeros(1, 4, 4),
                view_b=torch.zeros(1, 4, 4),
                warp_a_to_b=torch.zeros(4, 4, 2),
                valid_mask=torch.ones(4, 4, dtype=torch.bool),
            )
            write_pair_spec_manifest(
                manifest,
                [
                    LazyPairResult(
                        spec=LazyPairSpec(
                            pair_index=0,
                            split="train",
                            reference=reference,
                            target=target,
                            pair_type=PAIR_TYPE_CROSS_CAMERA,
                        ),
                        pair=pair,
                        valid_fraction=0.5,
                        valid_pixels=16,
                        attempt_count=1,
                        elapsed_ms=1.0,
                        crop_a=CropWindow(1, 2, 5, 6),
                        crop_b=CropWindow(7, 8, 11, 12),
                    )
                ],
            )
            args = SimpleNamespace(
                pair_spec_manifest=manifest,
                split="train",
                reference_variant="nadir",
                target_variant=[],
                pair_mode="same-position",
                cross_pair_variant=[],
                cross_camera_offsets=(1,),
                cross_fov_offsets=(0,),
                pair_type_weights={},
                spatial_index_planet_radius_m=3396190.0,
                spatial_index_footprint_samples=5,
                spatial_index_margin_m=2000.0,
                spatial_index_height_km=[],
                image_source="uint8",
                limit_pairs=0,
                seed=123,
                shuffle=False,
            )

            with mock.patch.object(visual_mod, "build_lazy_pair_specs", side_effect=AssertionError("should not build")):
                specs, pair_source, pair_type_counts = visual_mod.select_visual_pair_specs(args, [reference, target])

        self.assertEqual(pair_source, "pair_spec_manifest")
        self.assertEqual(pair_type_counts[PAIR_TYPE_CROSS_CAMERA], 1)
        self.assertEqual(specs[0].target.pose_id, target.pose_id)
        self.assertEqual(specs[0].fixed_crop_a, CropWindow(1, 2, 5, 6))
        self.assertEqual(specs[0].fixed_crop_b, CropWindow(7, 8, 11, 12))

    def test_lazy_visual_reads_manifest_dataset_id_compatible_with_pair_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            render_manifest = manifest_dir / "h100km_fov090_render_manifest.csv"
            uint8_manifest = manifest_dir / "h100km_fov090_uint8_manifest.csv"
            render_manifest.write_text(
                "pose_id,base_id,variant,split,lon_deg,lat_deg,tsai_path,image_path,depth_path,chunk_index\n"
                f"pose_a,base_a,mid_01,train,0,0,{root / 'a.tsai'},{root / 'a.tif'},{root / 'a_depth.tif'},0\n"
                f"pose_b,base_b,extreme_03,train,0,0,{root / 'b.tsai'},{root / 'b.tif'},{root / 'b_depth.tif'},0\n",
                encoding="utf-8",
            )
            uint8_manifest.write_text("source_path,uint8_path\n", encoding="utf-8")
            records = [
                RenderRecord(
                    pose_id="pose_a",
                    base_id="base_a",
                    variant="mid_01",
                    split="train",
                    tsai_path=root / "a.tsai",
                    image_path=root / "a.tif",
                    uint8_path=root / "a.tif",
                    depth_path=root / "a_depth.tif",
                    dataset_id="h100km_fov090",
                    raw_base_id="base_a",
                ),
                RenderRecord(
                    pose_id="pose_b",
                    base_id="base_b",
                    variant="extreme_03",
                    split="train",
                    tsai_path=root / "b.tsai",
                    image_path=root / "b.tif",
                    uint8_path=root / "b.tif",
                    depth_path=root / "b_depth.tif",
                    dataset_id="h100km_fov090",
                    raw_base_id="base_b",
                ),
            ]
            manifest = root / "overlap_edges.csv"
            pair = SyntheticPair(
                view_a=torch.zeros(1, 4, 4),
                view_b=torch.zeros(1, 4, 4),
                warp_a_to_b=torch.zeros(4, 4, 2),
                valid_mask=torch.ones(4, 4, dtype=torch.bool),
            )
            write_pair_spec_manifest(
                manifest,
                [
                    LazyPairResult(
                        spec=LazyPairSpec(
                            pair_index=0,
                            split="train",
                            reference=records[0],
                            target=records[1],
                            pair_type=PAIR_TYPE_CROSS_CAMERA,
                        ),
                        pair=pair,
                        valid_fraction=0.5,
                        valid_pixels=16,
                        attempt_count=1,
                        elapsed_ms=1.0,
                        crop_a=CropWindow(0, 0, 4, 4),
                        crop_b=CropWindow(0, 0, 4, 4),
                    )
                ],
            )
            args = SimpleNamespace(
                pair_spec_manifest=manifest,
                split="train",
                reference_variant="nadir",
                target_variant=[],
                pair_mode="same-position",
                cross_pair_variant=[],
                cross_camera_offsets=(1,),
                cross_fov_offsets=(0,),
                pair_type_weights={},
                spatial_index_planet_radius_m=3396190.0,
                spatial_index_footprint_samples=5,
                spatial_index_margin_m=2000.0,
                spatial_index_height_km=[],
                image_source="uint8",
                limit_pairs=0,
                seed=123,
                shuffle=False,
            )

            visual_records = visual_mod.read_visual_records(render_manifest, uint8_manifest)
            specs, pair_source, pair_type_counts = visual_mod.select_visual_pair_specs(args, visual_records)

        self.assertEqual(pair_source, "pair_spec_manifest")
        self.assertEqual(pair_type_counts[PAIR_TYPE_CROSS_CAMERA], 1)
        self.assertEqual(specs[0].reference.dataset_id, "h100km_fov090")
        self.assertEqual(specs[0].target.pose_id, "pose_b")

    def test_smooth_series_keeps_short_series_length(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0]).numpy()

        smoothed = visual_mod.smooth_series(values, window=5)

        self.assertEqual(smoothed.shape, values.shape)

    def test_illumination_variants_preserve_shape(self) -> None:
        image = torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(1, 4, 4)
        variants = make_illumination_variants(image)

        names = [name for name, _ in variants]
        self.assertIn("original", names)
        self.assertIn("gamma_dark", names)
        self.assertIn("shadow_band", names)
        self.assertTrue(all(variant.shape == image.shape for _, variant in variants))
        self.assertFalse(torch.allclose(dict(variants)["gamma_dark"], image))

    def test_zero_degree_rotation_keeps_identity_warp(self) -> None:
        image = torch.arange(16, dtype=torch.float32).view(1, 4, 4) / 15.0
        pair = rotate_pair_from_view(image, angle_deg=0.0)

        self.assertIsInstance(pair, SyntheticPair)
        self.assertTrue(torch.allclose(pair.view_a, image))
        self.assertTrue(torch.allclose(pair.view_b, image, atol=1e-5))
        self.assertTrue(pair.valid_mask.all())
        self.assertTrue(torch.allclose(pair.warp_a_to_b[0, 0], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.allclose(pair.warp_a_to_b[-1, -1], torch.tensor([3.0, 3.0])))

    def test_lazy_visual_illumination_stress_keeps_geometry(self) -> None:
        record = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        spec = LazyPairSpec(
            pair_index=7,
            split="train",
            reference=record,
            target=RenderRecord(
                pose_id="pose_b",
                base_id="base_001",
                variant="extreme_03",
                split="train",
                tsai_path=Path("b.tsai"),
                image_path=Path("b.tif"),
                uint8_path=Path("b.png"),
                depth_path=Path("b_depth.tif"),
            ),
        )
        pair = SyntheticPair(
            view_a=torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(1, 4, 4),
            view_b=torch.linspace(1.0, 0.0, 16, dtype=torch.float32).view(1, 4, 4),
            warp_a_to_b=torch.zeros(4, 4, 2, dtype=torch.float32),
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        visual = LazyMatchVisual(
            label="困难/失败",
            spec=spec,
            pair=pair,
            valid_fraction=0.5,
            points_a=torch.empty(0, 2).numpy(),
            points_b=torch.empty(0, 2).numpy(),
            scores=torch.empty(0).numpy(),
            errors=torch.empty(0).numpy(),
            correct=torch.empty(0, dtype=torch.bool).numpy(),
        )

        variants = make_illumination_stress_lazy_results([visual])

        self.assertGreater(len(variants), 1)
        self.assertTrue(any(item.label.endswith("gamma_dark") for item in variants))
        for item in variants:
            self.assertIsInstance(item.result, LazyPairResult)
            self.assertTrue(torch.allclose(item.result.pair.view_a, pair.view_a))
            self.assertTrue(torch.allclose(item.result.pair.warp_a_to_b, pair.warp_a_to_b))
            self.assertTrue(torch.equal(item.result.pair.valid_mask, pair.valid_mask))

    def test_lazy_visual_draw_zero_selects_all_matches(self) -> None:
        record = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        spec = LazyPairSpec(
            pair_index=1,
            split="train",
            reference=record,
            target=record,
        )
        pair = SyntheticPair(
            view_a=torch.zeros(1, 4, 4),
            view_b=torch.zeros(1, 4, 4),
            warp_a_to_b=torch.zeros(4, 4, 2),
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        visual = LazyMatchVisual(
            label="测试",
            spec=spec,
            pair=pair,
            valid_fraction=1.0,
            points_a=torch.zeros(5, 2).numpy(),
            points_b=torch.zeros(5, 2).numpy(),
            scores=torch.linspace(0.1, 0.5, 5).numpy(),
            errors=torch.zeros(5).numpy(),
            correct=torch.tensor([True, False, True, False, True]).numpy(),
        )

        self.assertEqual(selected_draw_indices(visual, 0).tolist(), [0, 1, 2, 3, 4])

    def test_lazy_visual_html_report_shows_source_image_paths(self) -> None:
        reference = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="val",
            tsai_path=Path("a.tsai"),
            image_path=Path("/raw/a.tif"),
            uint8_path=Path("/uint8/a.png"),
            depth_path=Path("a_depth.tif"),
        )
        target = RenderRecord(
            pose_id="pose_b",
            base_id="base_001",
            variant="extreme_03",
            split="val",
            tsai_path=Path("b.tsai"),
            image_path=Path("/raw/b.tif"),
            uint8_path=Path("/uint8/b.png"),
            depth_path=Path("b_depth.tif"),
        )
        spec = LazyPairSpec(pair_index=1, split="val", reference=reference, target=target)
        pair = SyntheticPair(
            view_a=torch.zeros(1, 4, 4),
            view_b=torch.zeros(1, 4, 4),
            warp_a_to_b=torch.zeros(4, 4, 2),
            valid_mask=torch.ones(4, 4, dtype=torch.bool),
        )
        visual = LazyMatchVisual(
            label="测试",
            spec=spec,
            pair=pair,
            valid_fraction=1.0,
            points_a=torch.zeros(1, 2).numpy(),
            points_b=torch.zeros(1, 2).numpy(),
            scores=torch.ones(1).numpy(),
            errors=torch.zeros(1).numpy(),
            correct=torch.ones(1, dtype=torch.bool).numpy(),
            image_name="pair.png",
            crop_a=CropWindow(x0=10, y0=20, x1=778, y1=788),
            crop_b=CropWindow(x0=30, y0=40, x1=798, y1=808),
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "pair.png"
            image_path.write_bytes(b"fake-png")
            html_path = tmp_path / "index.html"

            visual_mod.write_html_report(
                html_path,
                args=SimpleNamespace(pytorch_state=Path("state.pt")),
                all_results=[visual],
                selected=[visual],
                image_paths={"pair.png": image_path},
                artifact_paths={},
                elapsed_s=1.0,
            )

            html_text = html_path.read_text(encoding="utf-8")

        self.assertIn("A图文件", html_text)
        self.assertIn("B图文件", html_text)
        self.assertIn("A图 crop", html_text)
        self.assertIn("B图 crop", html_text)
        self.assertIn("/uint8/a.png", html_text)
        self.assertIn("/uint8/b.png", html_text)
        self.assertIn("x=10, y=20, w=768, h=768", html_text)
        self.assertIn("x=30, y=40, w=768, h=768", html_text)

    def test_lazy_visual_geometry_filter_removes_outlier_matches(self) -> None:
        record = RenderRecord(
            pose_id="pose_a",
            base_id="base_001",
            variant="nadir",
            split="train",
            tsai_path=Path("a.tsai"),
            image_path=Path("a.tif"),
            uint8_path=Path("a.png"),
            depth_path=Path("a_depth.tif"),
        )
        spec = LazyPairSpec(
            pair_index=1,
            split="train",
            reference=record,
            target=record,
        )
        yy, xx = torch.meshgrid(torch.arange(16, dtype=torch.float32), torch.arange(16, dtype=torch.float32), indexing="ij")
        warp = torch.stack([xx, yy], dim=-1)
        pair = SyntheticPair(
            view_a=torch.ones(1, 16, 16),
            view_b=torch.ones(1, 16, 16),
            warp_a_to_b=warp,
            valid_mask=torch.ones(16, 16, dtype=torch.bool),
        )
        visual = LazyMatchVisual(
            label="raw",
            spec=spec,
            pair=pair,
            valid_fraction=1.0,
            points_a=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0], [14.0, 2.0]]).numpy(),
            points_b=torch.tensor([[1.0, 1.0], [4.0, 4.0], [8.0, 8.0], [12.0, 12.0], [0.0, 15.0]]).numpy(),
            scores=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.95]).numpy(),
            errors=torch.tensor([0.0, 0.0, 0.0, 0.0, 20.0]).numpy(),
            correct=torch.tensor([True, True, True, True, False]).numpy(),
        )

        filtered = filter_visual_matches(visual, geometry_filter="local", threshold_px=1.0)

        self.assertEqual(filtered.label, "raw / filtered")
        self.assertEqual(filtered.matches, 4)
        self.assertEqual(filtered.correct_count, 4)


if __name__ == "__main__":
    unittest.main()
