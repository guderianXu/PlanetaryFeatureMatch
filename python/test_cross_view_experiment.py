import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

import cross_view_experiment as exp


class CrossViewExperimentTest(unittest.TestCase):
    def test_build_training_command_uses_existing_training_script_and_all_train_caches(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/numeric/rotate"), Path("/runs/splits/train/timestamp/viewpoint")],
            validation_cache_dirs=[Path("/runs/splits/val/numeric/rotate")],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=Path("/runs/base.pt"),
            init_pytorch_state=None,
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
        )

        self.assertEqual(command[:3], ["/env/bin/python", "/repo/python/pfm_pytorch_training.py", "--checkpoint"])
        self.assertIn("/runs/base.pt", command)
        self.assertEqual(command.count("--cache-dir"), 2)
        self.assertEqual(command.count("--validation-cache-dir"), 1)
        self.assertIn("/runs/splits/train/numeric/rotate", command)
        self.assertIn("/runs/splits/train/timestamp/viewpoint", command)
        self.assertIn("/runs/splits/val/numeric/rotate", command)
        self.assertIn("--train-blended-descriptors", command)
        self.assertIn("--skip-nonfinite-steps", command)
        self.assertIn("--exclude-self-pairs", command)

    def test_build_training_command_can_request_random_initialization(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/numeric/rotate")],
            validation_cache_dirs=[],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=None,
            init_random=True,
            device="cuda",
            steps=1,
            batch_pairs=1,
            samples_per_pair=32,
            learning_rate=1.0e-4,
        )

        self.assertIn("--init-random", command)
        self.assertNotIn("--checkpoint", command)

    def test_build_training_command_passes_gradient_accumulation_steps(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/numeric/rotate")],
            validation_cache_dirs=[],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=Path("/runs/base_state.pt"),
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
            gradient_accumulation_steps=2,
        )

        self.assertIn("--gradient-accumulation-steps", command)
        self.assertIn("2", command)

    def test_build_training_command_can_request_balanced_cache_sampling(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[
                Path("/runs/splits/train/numeric/rotate"),
                Path("/runs/splits/train/numeric/viewpoint"),
                Path("/runs/splits/train/timestamp/compound"),
            ],
            validation_cache_dirs=[],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=Path("/runs/base_state.pt"),
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
            balanced_cache_sampling=True,
        )

        self.assertIn("--balanced-cache-sampling", command)

    def test_build_training_command_passes_training_texture_blend_weight(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/numeric/rotate")],
            validation_cache_dirs=[],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=Path("/runs/base_state.pt"),
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
            training_texture_blend_weight=0.25,
        )

        self.assertIn("--training-texture-blend-weight", command)
        self.assertIn("0.25", command)

    def test_build_training_command_can_limit_training_descriptor_eval_pairs(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/numeric/rotate")],
            validation_cache_dirs=[Path("/runs/splits/val/numeric/rotate")],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=Path("/runs/base_state.pt"),
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
            training_eval_pairs=128,
        )

        self.assertIn("--eval-pairs", command)
        self.assertIn("128", command)

    def test_build_training_command_passes_warp_hard_negative_options(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/numeric/viewpoint")],
            validation_cache_dirs=[],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=Path("/runs/base_state.pt"),
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
            warp_hard_negative_weight=0.25,
            warp_hard_negative_radius=3.0,
            warp_hard_negative_margin=0.4,
            warp_hard_negative_candidates=2048,
        )

        self.assertIn("--warp-hard-negative-weight", command)
        self.assertIn("0.25", command)
        self.assertIn("--warp-hard-negative-radius", command)
        self.assertIn("3", command)
        self.assertIn("--warp-hard-negative-margin", command)
        self.assertIn("0.4", command)
        self.assertIn("--warp-hard-negative-candidates", command)
        self.assertIn("2048", command)

    def test_build_training_command_passes_abstention_options(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/numeric/viewpoint")],
            validation_cache_dirs=[],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=Path("/runs/base_state.pt"),
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
            abstention_weight=0.35,
            abstention_negative_radius=3.0,
            abstention_max_false_score=0.4,
            abstention_topk=6,
            abstention_candidates=2048,
        )

        self.assertIn("--abstention-weight", command)
        self.assertIn("0.35", command)
        self.assertIn("--abstention-negative-radius", command)
        self.assertIn("3", command)
        self.assertIn("--abstention-max-false-score", command)
        self.assertIn("0.4", command)
        self.assertIn("--abstention-topk", command)
        self.assertIn("6", command)
        self.assertIn("--abstention-candidates", command)
        self.assertIn("2048", command)

    def test_build_training_command_passes_false_match_options(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/numeric/viewpoint")],
            validation_cache_dirs=[],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=Path("/runs/base_state.pt"),
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
            false_match_csvs=[Path("/runs/mined/false_matches.csv")],
            false_match_weight=0.6,
            false_match_max_points=24,
            false_match_max_score=0.25,
            false_match_curriculum_max_probability=0.75,
            false_match_curriculum_warmup_steps=20,
        )

        self.assertIn("--false-match-csv", command)
        self.assertIn("/runs/mined/false_matches.csv", command)
        self.assertIn("--false-match-weight", command)
        self.assertIn("0.6", command)
        self.assertIn("--false-match-max-points", command)
        self.assertIn("24", command)
        self.assertIn("--false-match-max-score", command)
        self.assertIn("0.25", command)
        self.assertIn("--false-match-curriculum-max-probability", command)
        self.assertIn("0.75", command)
        self.assertIn("--false-match-curriculum-warmup-steps", command)
        self.assertIn("20", command)

    def test_build_training_command_passes_hard_pair_curriculum_options(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/timestamp/viewpoint")],
            validation_cache_dirs=[],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=Path("/runs/base_state.pt"),
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
            hard_summaries=[Path("/runs/hard/timestamp/viewpoint/summary.csv")],
            hard_limit=32,
            hard_min_matches=12,
            hard_max_precision=0.35,
            hard_repeat=4,
            hard_curriculum_max_probability=0.5,
            hard_curriculum_warmup_steps=40,
        )

        self.assertIn("--hard-summary", command)
        self.assertIn("/runs/hard/timestamp/viewpoint/summary.csv", command)
        self.assertIn("--hard-limit", command)
        self.assertIn("32", command)
        self.assertIn("--hard-min-matches", command)
        self.assertIn("12", command)
        self.assertIn("--hard-max-precision", command)
        self.assertIn("0.35", command)
        self.assertIn("--hard-repeat", command)
        self.assertIn("4", command)
        self.assertIn("--hard-curriculum-max-probability", command)
        self.assertIn("0.5", command)
        self.assertIn("--hard-curriculum-warmup-steps", command)
        self.assertIn("40", command)

    def test_build_training_command_passes_pseudo_label_options(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/timestamp/viewpoint")],
            validation_cache_dirs=[],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=Path("/runs/base_state.pt"),
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
            pseudo_label_csvs=[Path("/runs/pseudo/pseudo_labels.csv")],
            pseudo_label_weight=0.5,
            pseudo_label_max_points=96,
            pseudo_label_curriculum_max_probability=0.75,
            pseudo_label_curriculum_warmup_steps=10,
            synthetic_loss_weight=0.25,
        )

        self.assertIn("--pseudo-label-csv", command)
        self.assertIn("/runs/pseudo/pseudo_labels.csv", command)
        self.assertIn("--pseudo-label-weight", command)
        self.assertIn("0.5", command)
        self.assertIn("--pseudo-label-max-points", command)
        self.assertIn("96", command)
        self.assertIn("--pseudo-label-curriculum-max-probability", command)
        self.assertIn("0.75", command)
        self.assertIn("--pseudo-label-curriculum-warmup-steps", command)
        self.assertIn("10", command)
        self.assertIn("--synthetic-loss-weight", command)
        self.assertIn("0.25", command)

    def test_build_training_command_passes_pseudo_keypoint_options(self):
        command = exp.build_training_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            train_cache_dirs=[Path("/runs/splits/train/timestamp/viewpoint")],
            validation_cache_dirs=[],
            output_dir=Path("/runs/experiment/training"),
            checkpoint=None,
            init_pytorch_state=Path("/runs/base_state.pt"),
            init_random=False,
            device="cuda",
            steps=25,
            batch_pairs=3,
            samples_per_pair=128,
            learning_rate=1.0e-4,
            pseudo_label_csvs=[Path("/runs/pseudo/pseudo_labels.csv")],
            pseudo_label_weight=0.0,
            pseudo_keypoint_weight=0.75,
            pseudo_keypoint_negative_weight=0.03,
        )

        self.assertIn("--pseudo-keypoint-weight", command)
        self.assertIn("0.75", command)
        self.assertIn("--pseudo-keypoint-negative-weight", command)
        self.assertIn("0.03", command)

    def test_evaluation_groups_returns_two_styles_times_three_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_root = Path(tmp)
            for style in ("numeric", "timestamp"):
                for gate in ("rotate", "viewpoint", "compound"):
                    group = split_root / "test" / style / gate
                    group.mkdir(parents=True)
                    (group / "source_000001_1").mkdir()

            groups = exp.evaluation_groups(split_root, split="test")

            self.assertEqual(len(groups), 6)
            self.assertEqual(
                [(group.style, group.gate) for group in groups],
                [
                    ("numeric", "rotate"),
                    ("numeric", "viewpoint"),
                    ("numeric", "compound"),
                    ("timestamp", "rotate"),
                    ("timestamp", "viewpoint"),
                    ("timestamp", "compound"),
                ],
            )

    def test_select_visualization_pairs_is_deterministic_and_limited(self):
        pairs = [Path(f"pair_{index:06d}.pt") for index in range(10)]

        first = exp.select_visualization_pairs(pairs, count=2, seed=11)
        second = exp.select_visualization_pairs(reversed(pairs), count=2, seed=11)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_draw_match_visualization_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "matches.png"
            view_a = torch.linspace(0, 1, steps=16, dtype=torch.float32).reshape(1, 4, 4)
            view_b = torch.flip(view_a, dims=[2])
            points_a = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
            points_b = torch.tensor([[3.0, 0.0], [0.0, 3.0]])

            exp.draw_match_visualization(view_a, view_b, points_a, points_b, output)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_render_sample_visualizations_writes_two_deterministic_pair_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = [root / f"pair_{index:06d}.pt" for index in range(4)]
            for pair in pairs:
                pair.write_text("pair", encoding="utf-8")

            def load_matches(_pair_path):
                view_a = torch.ones(1, 4, 4)
                view_b = torch.ones(1, 4, 4)
                points_a = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
                points_b = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
                return view_a, view_b, points_a, points_b

            written = exp.render_sample_visualizations(
                pairs,
                root / "visualizations",
                count=2,
                seed=5,
                load_matches=load_matches,
            )

            self.assertEqual(len(written), 2)
            self.assertTrue(all(path.exists() for path in written))
            self.assertEqual([path.suffix for path in written], [".png", ".png"])

    def test_build_eval_command_passes_seeded_pair_sampling_to_evaluator(self):
        command = exp.build_eval_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            group=exp.EvalGroup("numeric", "rotate", Path("/runs/splits/test/numeric/rotate")),
            output_csv=Path("/runs/eval/numeric/rotate/summary.csv"),
            pytorch_state=Path("/runs/training/pytorch_pfm_state.pt"),
            device="cuda",
            limit_pairs=64,
            sample_seed=1234,
            max_keypoints=2048,
            descriptor_topk=32,
        )

        self.assertIn("--limit-pairs", command)
        self.assertIn("64", command)
        self.assertIn("--sample-seed", command)
        self.assertIn("1234", command)
        self.assertIn("--exclude-self-pairs", command)

    def test_build_eval_command_passes_texture_blend_weight(self):
        command = exp.build_eval_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            group=exp.EvalGroup("numeric", "rotate", Path("/runs/splits/test/numeric/rotate")),
            output_csv=Path("/runs/eval/numeric/rotate/summary.csv"),
            pytorch_state=Path("/runs/training/pytorch_pfm_state.pt"),
            device="cuda",
            limit_pairs=64,
            sample_seed=1234,
            max_keypoints=2048,
            descriptor_topk=32,
            texture_blend_weight=0.25,
        )

        self.assertIn("--texture-blend-weight", command)
        self.assertIn("0.25", command)

    def test_build_eval_command_passes_keypoint_spatial_bins(self):
        command = exp.build_eval_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            group=exp.EvalGroup("numeric", "rotate", Path("/runs/splits/test/numeric/rotate")),
            output_csv=Path("/runs/eval/numeric/rotate/summary.csv"),
            pytorch_state=Path("/runs/training/pytorch_pfm_state.pt"),
            device="cuda",
            limit_pairs=64,
            sample_seed=1234,
            max_keypoints=2048,
            descriptor_topk=32,
            keypoint_spatial_bins=16,
        )

        self.assertIn("--keypoint-spatial-bins", command)
        self.assertIn("16", command)

    def test_build_eval_command_passes_keypoint_score_mode(self):
        command = exp.build_eval_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            group=exp.EvalGroup("timestamp", "viewpoint", Path("/runs/splits/test/timestamp/viewpoint")),
            output_csv=Path("/runs/eval/timestamp/viewpoint/summary.csv"),
            pytorch_state=Path("/runs/training/pytorch_pfm_state.pt"),
            device="cuda",
            limit_pairs=64,
            sample_seed=1234,
            max_keypoints=2048,
            descriptor_topk=32,
            keypoint_score_mode="learned",
        )

        self.assertIn("--keypoint-score-mode", command)
        self.assertIn("learned", command)

    def test_build_eval_command_passes_match_margin(self):
        command = exp.build_eval_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            group=exp.EvalGroup("numeric", "compound", Path("/runs/splits/test/numeric/compound")),
            output_csv=Path("/runs/eval/numeric/compound/summary.csv"),
            pytorch_state=Path("/runs/training/pytorch_pfm_state.pt"),
            device="cuda",
            limit_pairs=64,
            sample_seed=1234,
            max_keypoints=2048,
            descriptor_topk=32,
            min_margin=0.01,
        )

        self.assertIn("--min-margin", command)
        self.assertIn("0.01", command)

    def test_build_eval_command_passes_min_target_gradient(self):
        command = exp.build_eval_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            group=exp.EvalGroup("timestamp", "compound", Path("/runs/splits/test/timestamp/compound")),
            output_csv=Path("/runs/eval/timestamp/compound/summary.csv"),
            pytorch_state=Path("/runs/training/pytorch_pfm_state.pt"),
            device="cuda",
            limit_pairs=64,
            sample_seed=1234,
            max_keypoints=2048,
            descriptor_topk=32,
            min_target_gradient=20.25,
        )

        self.assertIn("--min-target-gradient", command)
        self.assertIn("20.25", command)

    def test_build_eval_command_passes_min_target_local_contrast(self):
        command = exp.build_eval_command(
            python_exe=Path("/env/bin/python"),
            project_root=Path("/repo"),
            group=exp.EvalGroup("timestamp", "compound", Path("/runs/splits/test/timestamp/compound")),
            output_csv=Path("/runs/eval/timestamp/compound/summary.csv"),
            pytorch_state=Path("/runs/training/pytorch_pfm_state.pt"),
            device="cuda",
            limit_pairs=64,
            sample_seed=1234,
            max_keypoints=2048,
            descriptor_topk=32,
            min_target_local_contrast=5.32,
        )

        self.assertIn("--min-target-local-contrast", command)
        self.assertIn("5.32", command)

    def test_parse_blend_weight_candidates_preserves_order_and_deduplicates(self):
        self.assertEqual(
            exp.parse_blend_weight_candidates("1,0.25, 1,4"),
            [1.0, 0.25, 4.0],
        )

    def test_parse_match_margin_candidates_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            exp.parse_match_margin_candidates("0, -0.1")

        self.assertEqual(exp.parse_match_margin_candidates("0,0.01,0.01,0.02"), [0.0, 0.01, 0.02])

    def test_parse_target_gradient_candidates_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            exp.parse_target_gradient_candidates("0, -1")

        self.assertEqual(exp.parse_target_gradient_candidates("0,20.25,20.25,22.18"), [0.0, 20.25, 22.18])

    def test_parse_target_local_contrast_candidates_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            exp.parse_target_local_contrast_candidates("0, -1")

        self.assertEqual(exp.parse_target_local_contrast_candidates("0,5.32,5.32,6.0"), [0.0, 5.32, 6.0])

    def test_parse_sample_seeds_preserves_order_and_deduplicates(self):
        self.assertEqual(exp.parse_sample_seeds("1234, 2234,1234"), [1234, 2234])

    def test_parse_group_keys_accepts_style_gate_pairs(self):
        self.assertEqual(
            exp.parse_group_keys("numeric/viewpoint, timestamp/compound"),
            {("numeric", "viewpoint"), ("timestamp", "compound")},
        )

    def test_select_created_cache_dirs_can_filter_training_groups(self):
        created = {
            ("train", "numeric", "rotate"): Path("/runs/splits/train/numeric/rotate"),
            ("train", "timestamp", "viewpoint"): Path("/runs/splits/train/timestamp/viewpoint"),
            ("train", "timestamp", "compound"): Path("/runs/splits/train/timestamp/compound"),
            ("val", "timestamp", "viewpoint"): Path("/runs/splits/val/timestamp/viewpoint"),
        }

        train_dirs = exp.select_created_cache_dirs(
            created,
            split="train",
            groups={("timestamp", "viewpoint")},
        )
        val_dirs = exp.select_created_cache_dirs(
            created,
            split="val",
            groups={("timestamp", "viewpoint")},
        )

        self.assertEqual(train_dirs, [Path("/runs/splits/train/timestamp/viewpoint")])
        self.assertEqual(val_dirs, [Path("/runs/splits/val/timestamp/viewpoint")])

    def test_parse_args_accepts_training_group_filter(self):
        argv = [
            "cross_view_experiment.py",
            "--cache-dir",
            "cache",
            "--output-dir",
            "runs/experiment",
            "--init-pytorch-state",
            "state.pt",
            "--training-groups",
            "timestamp/viewpoint,numeric/compound",
        ]

        with mock.patch("sys.argv", argv):
            args = exp.parse_args()

        self.assertEqual(args.training_groups, "timestamp/viewpoint,numeric/compound")

    def test_parse_args_accepts_hard_pair_training_mining_options(self):
        argv = [
            "cross_view_experiment.py",
            "--cache-dir",
            "cache",
            "--output-dir",
            "runs/experiment",
            "--init-pytorch-state",
            "state.pt",
            "--mine-hard-training-pairs",
            "--hard-mine-limit-pairs",
            "96",
            "--hard-summary",
            "runs/existing_hard.csv",
            "--hard-limit",
            "48",
            "--hard-min-matches",
            "10",
            "--hard-max-precision",
            "0.4",
            "--hard-repeat",
            "2",
            "--hard-curriculum-max-probability",
            "0.5",
            "--hard-curriculum-warmup-steps",
            "30",
        ]

        with mock.patch("sys.argv", argv):
            args = exp.parse_args()

        self.assertTrue(args.mine_hard_training_pairs)
        self.assertEqual(args.hard_mine_limit_pairs, 96)
        self.assertEqual(args.hard_summary, [Path("runs/existing_hard.csv")])
        self.assertEqual(args.hard_limit, 48)
        self.assertEqual(args.hard_min_matches, 10)
        self.assertEqual(args.hard_max_precision, 0.4)
        self.assertEqual(args.hard_repeat, 2)
        self.assertEqual(args.hard_curriculum_max_probability, 0.5)
        self.assertEqual(args.hard_curriculum_warmup_steps, 30)

    def test_parse_calibration_pytorch_state_entries_preserves_labels(self):
        entries = exp.parse_calibration_pytorch_state_entries(
            ["timestamp=runs/timestamp.pt", "balanced=runs/base.pt"]
        )

        self.assertEqual(
            entries,
            [
                ("timestamp", Path("runs/timestamp.pt")),
                ("balanced", Path("runs/base.pt")),
            ],
        )

    def test_parse_args_accepts_extra_calibration_pytorch_states(self):
        argv = [
            "cross_view_experiment.py",
            "--cache-dir",
            "cache",
            "--output-dir",
            "runs/experiment",
            "--init-pytorch-state",
            "state.pt",
            "--calibration-pytorch-state",
            "timestamp=runs/timestamp.pt",
            "--calibration-pytorch-state",
            "balanced=runs/base.pt",
        ]

        with mock.patch("sys.argv", argv):
            args = exp.parse_args()

        self.assertEqual(args.calibration_pytorch_state, ["timestamp=runs/timestamp.pt", "balanced=runs/base.pt"])

    def test_parse_args_accepts_checkpoint_switch_safeguards(self):
        argv = [
            "cross_view_experiment.py",
            "--cache-dir",
            "cache",
            "--output-dir",
            "runs/experiment",
            "--init-pytorch-state",
            "state.pt",
            "--calibration-state-switch-reference-label",
            "base",
            "--calibration-state-switch-min-precision-gain",
            "0.03",
            "--calibration-state-switch-min-match-ratio",
            "0.25",
        ]

        with mock.patch("sys.argv", argv):
            args = exp.parse_args()

        self.assertEqual(args.calibration_state_switch_reference_label, "base")
        self.assertEqual(args.calibration_state_switch_min_precision_gain, 0.03)
        self.assertEqual(args.calibration_state_switch_min_match_ratio, 0.25)

    def test_parse_args_accepts_match_margin_calibration_options(self):
        argv = [
            "cross_view_experiment.py",
            "--cache-dir",
            "cache",
            "--output-dir",
            "runs/experiment",
            "--init-pytorch-state",
            "state.pt",
            "--calibrate-match-min-margins",
            "--match-min-margin-candidates",
            "0,0.01",
            "--match-min-target-gradient",
            "20.25",
            "--calibrate-target-gradients",
            "--target-gradient-candidates",
            "0,20.25",
            "--match-min-target-local-contrast",
            "5.32",
            "--calibrate-target-local-contrasts",
            "--target-local-contrast-candidates",
            "0,5.32",
            "--geometry-filter",
            "local",
            "--keypoint-score-mode",
            "learned",
            "--calibrate-keypoint-score-modes",
            "--keypoint-score-mode-candidates",
            "texture,learned",
            "--calibration-min-matches",
            "128",
            "--calibration-min-match-fraction",
            "0.25",
        ]

        with mock.patch("sys.argv", argv):
            args = exp.parse_args()

        self.assertTrue(args.calibrate_match_min_margins)
        self.assertEqual(args.match_min_margin_candidates, "0,0.01")
        self.assertEqual(args.match_min_target_gradient, 20.25)
        self.assertTrue(args.calibrate_target_gradients)
        self.assertEqual(args.target_gradient_candidates, "0,20.25")
        self.assertEqual(args.match_min_target_local_contrast, 5.32)
        self.assertTrue(args.calibrate_target_local_contrasts)
        self.assertEqual(args.target_local_contrast_candidates, "0,5.32")
        self.assertEqual(args.geometry_filter, "local")
        self.assertEqual(args.keypoint_score_mode, "learned")
        self.assertTrue(args.calibrate_keypoint_score_modes)
        self.assertEqual(args.keypoint_score_mode_candidates, "texture,learned")
        self.assertEqual(args.calibration_min_matches, 128)
        self.assertEqual(args.calibration_min_match_fraction, 0.25)

    def test_select_best_blend_weight_summaries_prefers_precision_then_correct_then_matches(self):
        rows = [
            exp.BlendWeightSummary("numeric", "rotate", 0.25, matches=100, correct=75, precision=0.75, summary_csv=Path("a.csv")),
            exp.BlendWeightSummary("numeric", "rotate", 0.50, matches=120, correct=90, precision=0.75, summary_csv=Path("b.csv")),
            exp.BlendWeightSummary("numeric", "viewpoint", 1.00, matches=80, correct=20, precision=0.25, summary_csv=Path("c.csv")),
            exp.BlendWeightSummary("numeric", "viewpoint", 2.00, matches=60, correct=18, precision=0.30, summary_csv=Path("d.csv")),
        ]

        selected = exp.select_best_blend_weight_summaries(rows)

        self.assertEqual(selected[("numeric", "rotate")].texture_blend_weight, 0.50)
        self.assertEqual(selected[("numeric", "viewpoint")].texture_blend_weight, 2.00)

    def test_select_best_blend_weight_summaries_tracks_keypoint_score_mode(self):
        rows = [
            exp.BlendWeightSummary(
                "timestamp",
                "viewpoint",
                1.0,
                matches=120,
                correct=12,
                precision=0.10,
                summary_csv=Path("texture.csv"),
                keypoint_score_mode="texture",
            ),
            exp.BlendWeightSummary(
                "timestamp",
                "viewpoint",
                2.0,
                matches=100,
                correct=20,
                precision=0.20,
                summary_csv=Path("learned.csv"),
                keypoint_score_mode="learned",
            ),
        ]

        selected = exp.select_best_blend_weight_summaries(rows)

        self.assertEqual(selected[("timestamp", "viewpoint")].keypoint_score_mode, "learned")
        self.assertEqual(selected[("timestamp", "viewpoint")].texture_blend_weight, 2.0)

    def test_select_best_blend_weight_summaries_can_require_match_support(self):
        rows = [
            exp.BlendWeightSummary("timestamp", "compound", 0.0, matches=200, correct=8, precision=0.04, summary_csv=Path("a.csv")),
            exp.BlendWeightSummary("timestamp", "compound", 1.0, matches=40, correct=4, precision=0.10, summary_csv=Path("b.csv"), min_margin=0.01),
            exp.BlendWeightSummary("timestamp", "compound", 2.0, matches=140, correct=7, precision=0.05, summary_csv=Path("c.csv"), min_margin=0.01),
        ]

        selected = exp.select_best_blend_weight_summaries(rows, min_matches=100)

        self.assertEqual(selected[("timestamp", "compound")].texture_blend_weight, 2.0)
        self.assertEqual(selected[("timestamp", "compound")].min_margin, 0.01)

    def test_select_best_blend_weight_summaries_can_guard_checkpoint_switches(self):
        rows = [
            exp.BlendWeightSummary(
                "timestamp",
                "compound",
                2.0,
                matches=182,
                correct=10,
                precision=10 / 182,
                summary_csv=Path("trained.csv"),
                min_margin=0.01,
                pytorch_state_label="trained",
                pytorch_state=Path("/runs/trained.pt"),
            ),
            exp.BlendWeightSummary(
                "timestamp",
                "compound",
                1.0,
                matches=28,
                correct=3,
                precision=3 / 28,
                summary_csv=Path("specialist_low_support.csv"),
                min_margin=0.01,
                pytorch_state_label="blend02540",
                pytorch_state=Path("/runs/blend02540.pt"),
            ),
            exp.BlendWeightSummary(
                "timestamp",
                "compound",
                2.0,
                matches=59,
                correct=4,
                precision=4 / 59,
                summary_csv=Path("specialist_weak_gain.csv"),
                min_margin=0.01,
                pytorch_state_label="blend02540",
                pytorch_state=Path("/runs/blend02540.pt"),
            ),
            exp.BlendWeightSummary(
                "numeric",
                "rotate",
                4.0,
                matches=200,
                correct=120,
                precision=0.60,
                summary_csv=Path("numeric_trained.csv"),
                min_margin=0.01,
                pytorch_state_label="trained",
                pytorch_state=Path("/runs/trained.pt"),
            ),
            exp.BlendWeightSummary(
                "numeric",
                "rotate",
                1.0,
                matches=80,
                correct=64,
                precision=0.80,
                summary_csv=Path("numeric_specialist.csv"),
                min_margin=0.01,
                pytorch_state_label="blend02540",
                pytorch_state=Path("/runs/blend02540.pt"),
            ),
        ]

        selected = exp.select_best_blend_weight_summaries(
            rows,
            state_switch_min_precision_gain=0.03,
            state_switch_min_match_ratio=0.25,
        )

        self.assertEqual(selected[("timestamp", "compound")].pytorch_state_label, "trained")
        self.assertEqual(selected[("timestamp", "compound")].texture_blend_weight, 2.0)
        self.assertEqual(selected[("numeric", "rotate")].pytorch_state_label, "blend02540")
        self.assertEqual(selected[("numeric", "rotate")].texture_blend_weight, 1.0)

    def test_calibrate_texture_blend_weights_runs_validation_evals_and_writes_selected_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "experiment"
            group = exp.EvalGroup("numeric", "rotate", root / "splits" / "val" / "numeric" / "rotate")
            group.cache_dir.mkdir(parents=True)

            def fake_run(command, *, cwd, quiet=False):
                output_csv = Path(command[command.index("--output") + 1])
                weight = float(command[command.index("--texture-blend-weight") + 1])
                correct = 12 if weight == 0.25 else 8
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                output_csv.write_text(
                    "pair_pt,matches,correct,wrong,precision\n"
                    f"pair.pt,20,{correct},{20 - correct},{correct / 20:.6f}\n",
                    encoding="utf-8",
                )

            with mock.patch("cross_view_experiment.run_command", side_effect=fake_run):
                selected = exp.calibrate_texture_blend_weights(
                    python_exe=Path("/env/bin/python"),
                    project_root=Path("/repo"),
                    validation_groups=[group],
                    output_dir=output_dir,
                    pytorch_state=Path("/runs/state.pt"),
                    device="cuda",
                    candidates=[0.25, 1.0],
                    limit_pairs=64,
                    sample_seed=1234,
                    max_keypoints=2048,
                    descriptor_topk=32,
                )

            self.assertEqual(selected[("numeric", "rotate")].texture_blend_weight, 0.25)
            selected_csv = output_dir / "calibration" / "selected_weights.csv"
            self.assertTrue(selected_csv.exists())
            self.assertIn("numeric,rotate,0.25", selected_csv.read_text(encoding="utf-8"))

    def test_calibrate_texture_blend_weights_can_aggregate_multiple_sample_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "experiment"
            group = exp.EvalGroup("numeric", "viewpoint", root / "splits" / "val" / "numeric" / "viewpoint")
            group.cache_dir.mkdir(parents=True)

            def fake_run(command, *, cwd, quiet=False):
                output_csv = Path(command[command.index("--output") + 1])
                weight = float(command[command.index("--texture-blend-weight") + 1])
                seed = int(command[command.index("--sample-seed") + 1])
                correct = 6 if (weight, seed) == (1.0, 11) else 1
                if (weight, seed) == (0.0, 22):
                    correct = 6
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                output_csv.write_text(
                    "pair_pt,matches,correct,wrong,precision\n"
                    f"pair.pt,10,{correct},{10 - correct},{correct / 10:.6f}\n",
                    encoding="utf-8",
                )

            with mock.patch("cross_view_experiment.run_command", side_effect=fake_run):
                selected = exp.calibrate_texture_blend_weights(
                    python_exe=Path("/env/bin/python"),
                    project_root=Path("/repo"),
                    validation_groups=[group],
                    output_dir=output_dir,
                    pytorch_state=Path("/runs/state.pt"),
                    device="cuda",
                    candidates=[0.0, 1.0],
                    limit_pairs=64,
                    sample_seed=11,
                    calibration_sample_seeds=[11, 22],
                    max_keypoints=2048,
                    descriptor_topk=32,
                )

            self.assertEqual(selected[("numeric", "viewpoint")].texture_blend_weight, 0.0)
            self.assertEqual(selected[("numeric", "viewpoint")].matches, 20)
            self.assertEqual(selected[("numeric", "viewpoint")].correct, 7)

    def test_calibrate_texture_blend_weights_can_select_match_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "experiment"
            group = exp.EvalGroup("timestamp", "viewpoint", root / "splits" / "val" / "timestamp" / "viewpoint")
            group.cache_dir.mkdir(parents=True)

            def fake_run(command, *, cwd, quiet=False):
                output_csv = Path(command[command.index("--output") + 1])
                margin = float(command[command.index("--min-margin") + 1]) if "--min-margin" in command else 0.0
                correct = 12 if margin == 0.01 else 5
                matches = 20 if margin == 0.01 else 80
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                output_csv.write_text(
                    "pair_pt,matches,correct,wrong,precision\n"
                    f"pair.pt,{matches},{correct},{matches - correct},{correct / matches:.6f}\n",
                    encoding="utf-8",
                )

            with mock.patch("cross_view_experiment.run_command", side_effect=fake_run):
                selected = exp.calibrate_texture_blend_weights(
                    python_exe=Path("/env/bin/python"),
                    project_root=Path("/repo"),
                    validation_groups=[group],
                    output_dir=output_dir,
                    pytorch_state=Path("/runs/state.pt"),
                    device="cuda",
                    candidates=[1.0],
                    match_margin_candidates=[0.0, 0.01],
                    limit_pairs=64,
                    sample_seed=1234,
                    max_keypoints=2048,
                    descriptor_topk=32,
                )

            self.assertEqual(selected[("timestamp", "viewpoint")].texture_blend_weight, 1.0)
            self.assertEqual(selected[("timestamp", "viewpoint")].min_margin, 0.01)
            selected_csv = output_dir / "calibration" / "selected_weights.csv"
            self.assertIn("min_margin", selected_csv.read_text(encoding="utf-8"))

    def test_calibrate_texture_blend_weights_can_select_target_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "experiment"
            group = exp.EvalGroup("timestamp", "compound", root / "splits" / "val" / "timestamp" / "compound")
            group.cache_dir.mkdir(parents=True)

            def fake_run(command, *, cwd, quiet=False):
                output_csv = Path(command[command.index("--output") + 1])
                target_gradient = (
                    float(command[command.index("--min-target-gradient") + 1])
                    if "--min-target-gradient" in command
                    else 0.0
                )
                matches = 105 if target_gradient == 20.25 else 208
                correct = 10 if target_gradient == 20.25 else 12
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                output_csv.write_text(
                    "pair_pt,matches,correct,wrong,precision\n"
                    f"pair.pt,{matches},{correct},{matches - correct},{correct / matches:.6f}\n",
                    encoding="utf-8",
                )

            with mock.patch("cross_view_experiment.run_command", side_effect=fake_run):
                selected = exp.calibrate_texture_blend_weights(
                    python_exe=Path("/env/bin/python"),
                    project_root=Path("/repo"),
                    validation_groups=[group],
                    output_dir=output_dir,
                    pytorch_state=Path("/runs/state.pt"),
                    device="cuda",
                    candidates=[1.0],
                    target_gradient_candidates=[0.0, 20.25],
                    limit_pairs=64,
                    sample_seed=1234,
                    max_keypoints=2048,
                    descriptor_topk=32,
                )

            self.assertEqual(selected[("timestamp", "compound")].min_target_gradient, 20.25)
            selected_csv = output_dir / "calibration" / "selected_weights.csv"
            self.assertIn("min_target_gradient", selected_csv.read_text(encoding="utf-8"))

    def test_calibrate_texture_blend_weights_can_select_target_local_contrast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "experiment"
            group = exp.EvalGroup("timestamp", "compound", root / "splits" / "val" / "timestamp" / "compound")
            group.cache_dir.mkdir(parents=True)

            def fake_run(command, *, cwd, quiet=False):
                output_csv = Path(command[command.index("--output") + 1])
                target_contrast = (
                    float(command[command.index("--min-target-local-contrast") + 1])
                    if "--min-target-local-contrast" in command
                    else 0.0
                )
                matches = 91 if target_contrast == 5.32 else 208
                correct = 10 if target_contrast == 5.32 else 12
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                output_csv.write_text(
                    "pair_pt,matches,correct,wrong,precision\n"
                    f"pair.pt,{matches},{correct},{matches - correct},{correct / matches:.6f}\n",
                    encoding="utf-8",
                )

            with mock.patch("cross_view_experiment.run_command", side_effect=fake_run):
                selected = exp.calibrate_texture_blend_weights(
                    python_exe=Path("/env/bin/python"),
                    project_root=Path("/repo"),
                    validation_groups=[group],
                    output_dir=output_dir,
                    pytorch_state=Path("/runs/state.pt"),
                    device="cuda",
                    candidates=[1.0],
                    target_local_contrast_candidates=[0.0, 5.32],
                    limit_pairs=64,
                    sample_seed=1234,
                    max_keypoints=2048,
                    descriptor_topk=32,
                )

            self.assertEqual(selected[("timestamp", "compound")].min_target_local_contrast, 5.32)
            selected_csv = output_dir / "calibration" / "selected_weights.csv"
            self.assertIn("min_target_local_contrast", selected_csv.read_text(encoding="utf-8"))

    def test_calibrate_texture_blend_weights_can_select_pytorch_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "experiment"
            group = exp.EvalGroup("timestamp", "viewpoint", root / "splits" / "val" / "timestamp" / "viewpoint")
            group.cache_dir.mkdir(parents=True)

            def fake_run(command, *, cwd, quiet=False):
                output_csv = Path(command[command.index("--output") + 1])
                state_path = Path(command[command.index("--pytorch-state") + 1])
                correct = 12 if state_path.name == "timestamp.pt" else 5
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                output_csv.write_text(
                    "pair_pt,matches,correct,wrong,precision\n"
                    f"pair.pt,20,{correct},{20 - correct},{correct / 20:.6f}\n",
                    encoding="utf-8",
                )

            with mock.patch("cross_view_experiment.run_command", side_effect=fake_run):
                selected = exp.calibrate_texture_blend_weights(
                    python_exe=Path("/env/bin/python"),
                    project_root=Path("/repo"),
                    validation_groups=[group],
                    output_dir=output_dir,
                    pytorch_state=Path("/runs/base.pt"),
                    calibration_pytorch_states=[("timestamp", Path("/runs/timestamp.pt"))],
                    device="cuda",
                    candidates=[1.0],
                    limit_pairs=64,
                    sample_seed=1234,
                    max_keypoints=2048,
                    descriptor_topk=32,
                )

            self.assertEqual(selected[("timestamp", "viewpoint")].pytorch_state_label, "timestamp")
            self.assertEqual(selected[("timestamp", "viewpoint")].pytorch_state, Path("/runs/timestamp.pt"))
            selected_csv = output_dir / "calibration" / "selected_weights.csv"
            self.assertIn("pytorch_state_label", selected_csv.read_text(encoding="utf-8"))

    def test_calibrate_texture_blend_weights_can_select_keypoint_score_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "experiment"
            group = exp.EvalGroup("timestamp", "viewpoint", root / "splits" / "val" / "timestamp" / "viewpoint")
            group.cache_dir.mkdir(parents=True)

            def fake_run(command, *, cwd, quiet=False):
                output_csv = Path(command[command.index("--output") + 1])
                score_mode = command[command.index("--keypoint-score-mode") + 1]
                correct = 18 if score_mode == "learned" else 8
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                output_csv.write_text(
                    "pair_pt,matches,correct,wrong,precision\n"
                    f"pair.pt,100,{correct},{100 - correct},{correct / 100:.6f}\n",
                    encoding="utf-8",
                )

            with mock.patch("cross_view_experiment.run_command", side_effect=fake_run):
                selected = exp.calibrate_texture_blend_weights(
                    python_exe=Path("/env/bin/python"),
                    project_root=Path("/repo"),
                    validation_groups=[group],
                    output_dir=output_dir,
                    pytorch_state=Path("/runs/state.pt"),
                    device="cuda",
                    candidates=[1.0],
                    keypoint_score_mode_candidates=["texture", "learned"],
                    limit_pairs=64,
                    sample_seed=1234,
                    max_keypoints=2048,
                    descriptor_topk=32,
                )

            self.assertEqual(selected[("timestamp", "viewpoint")].keypoint_score_mode, "learned")
            selected_csv = output_dir / "calibration" / "selected_weights.csv"
            selected_text = selected_csv.read_text(encoding="utf-8")
            self.assertIn("keypoint_score_mode", selected_text)
            self.assertIn(",learned,", selected_text)

    def test_calibrate_texture_blend_weights_can_use_non_trained_switch_reference_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "experiment"
            group = exp.EvalGroup("numeric", "viewpoint", root / "splits" / "val" / "numeric" / "viewpoint")
            group.cache_dir.mkdir(parents=True)

            def fake_run(command, *, cwd, quiet=False):
                output_csv = Path(command[command.index("--output") + 1])
                state = Path(command[command.index("--pytorch-state") + 1]).name
                if state == "hard.pt":
                    matches, correct = 100, 30
                elif state == "base.pt":
                    matches, correct = 100, 28
                else:
                    matches, correct = 100, 32
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                output_csv.write_text(
                    "pair_pt,matches,correct,wrong,precision\n"
                    f"pair.pt,{matches},{correct},{matches - correct},{correct / matches:.6f}\n",
                    encoding="utf-8",
                )

            with mock.patch("cross_view_experiment.run_command", side_effect=fake_run):
                selected = exp.calibrate_texture_blend_weights(
                    python_exe=Path("/env/bin/python"),
                    project_root=Path("/repo"),
                    validation_groups=[group],
                    output_dir=output_dir,
                    pytorch_state=Path("/runs/hard.pt"),
                    calibration_pytorch_states=[
                        ("base", Path("/runs/base.pt")),
                        ("specialist", Path("/runs/specialist.pt")),
                    ],
                    device="cuda",
                    candidates=[1.0],
                    limit_pairs=64,
                    sample_seed=1234,
                    max_keypoints=2048,
                    descriptor_topk=32,
                    calibration_state_switch_reference_label="base",
                    calibration_state_switch_min_precision_gain=0.05,
                )

            self.assertEqual(selected[("numeric", "viewpoint")].pytorch_state_label, "base")

    def test_calibrate_texture_blend_weights_can_require_margin_match_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "experiment"
            group = exp.EvalGroup("timestamp", "compound", root / "splits" / "val" / "timestamp" / "compound")
            group.cache_dir.mkdir(parents=True)

            def fake_run(command, *, cwd, quiet=False):
                output_csv = Path(command[command.index("--output") + 1])
                margin = float(command[command.index("--min-margin") + 1]) if "--min-margin" in command else 0.0
                matches = 40 if margin == 0.01 else 200
                correct = 8 if margin == 0.01 else 10
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                output_csv.write_text(
                    "pair_pt,matches,correct,wrong,precision\n"
                    f"pair.pt,{matches},{correct},{matches - correct},{correct / matches:.6f}\n",
                    encoding="utf-8",
                )

            with mock.patch("cross_view_experiment.run_command", side_effect=fake_run):
                selected = exp.calibrate_texture_blend_weights(
                    python_exe=Path("/env/bin/python"),
                    project_root=Path("/repo"),
                    validation_groups=[group],
                    output_dir=output_dir,
                    pytorch_state=Path("/runs/state.pt"),
                    device="cuda",
                    candidates=[1.0],
                    match_margin_candidates=[0.0, 0.01],
                    limit_pairs=64,
                    sample_seed=1234,
                    max_keypoints=2048,
                    descriptor_topk=32,
                    calibration_min_matches=100,
                )

            self.assertEqual(selected[("timestamp", "compound")].min_margin, 0.0)
            self.assertEqual(selected[("timestamp", "compound")].matches, 200)

    def test_calibrate_texture_blend_weights_can_limit_multiseed_to_selected_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "experiment"
            rotate = exp.EvalGroup("numeric", "rotate", root / "splits" / "val" / "numeric" / "rotate")
            viewpoint = exp.EvalGroup("numeric", "viewpoint", root / "splits" / "val" / "numeric" / "viewpoint")
            rotate.cache_dir.mkdir(parents=True)
            viewpoint.cache_dir.mkdir(parents=True)
            seen: list[tuple[str, int]] = []

            def fake_run(command, *, cwd, quiet=False):
                output_csv = Path(command[command.index("--output") + 1])
                cache_dir = command[command.index("--cache-dir") + 1]
                group_name = "viewpoint" if "viewpoint" in cache_dir else "rotate"
                seed = int(command[command.index("--sample-seed") + 1])
                seen.append((group_name, seed))
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                output_csv.write_text(
                    "pair_pt,matches,correct,wrong,precision\n"
                    "pair.pt,10,5,5,0.500000\n",
                    encoding="utf-8",
                )

            with mock.patch("cross_view_experiment.run_command", side_effect=fake_run):
                exp.calibrate_texture_blend_weights(
                    python_exe=Path("/env/bin/python"),
                    project_root=Path("/repo"),
                    validation_groups=[rotate, viewpoint],
                    output_dir=output_dir,
                    pytorch_state=Path("/runs/state.pt"),
                    device="cuda",
                    candidates=[1.0],
                    limit_pairs=64,
                    sample_seed=11,
                    calibration_sample_seeds=[11, 22],
                    multiseed_groups={("numeric", "viewpoint")},
                    max_keypoints=2048,
                    descriptor_topk=32,
                )

            self.assertEqual(seen.count(("rotate", 11)), 1)
            self.assertNotIn(("rotate", 22), seen)
            self.assertIn(("viewpoint", 11), seen)
            self.assertIn(("viewpoint", 22), seen)

    def test_run_command_forces_gnu_mkl_threading_layer_for_child_processes(self):
        with mock.patch("cross_view_experiment.subprocess.run") as run:
            exp.run_command(["/bin/true"], cwd=Path("/repo"))

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["env"]["MKL_THREADING_LAYER"], "GNU")
        self.assertTrue(kwargs["check"])

    def test_run_command_can_suppress_child_output(self):
        with mock.patch("cross_view_experiment.subprocess.run") as run:
            exp.run_command(["/bin/true"], cwd=Path("/repo"), quiet=True)

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["stdout"], exp.subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], exp.subprocess.STDOUT)


if __name__ == "__main__":
    unittest.main()
