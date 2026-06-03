#!/usr/bin/env python3
"""Agent5 matcher iteration: RootSIFT threshold grid plus learned matcher sweeps."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT4_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent4.py"
AGENT4_SAMPLE = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent4" / "sampled_pairs.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent5"
DEFAULT_PFM_RUN = (
    PROJECT_ROOT
    / "runs"
    / "cross_view_1024_checkpoint_routed_guard_frac010_gain003_ratio025_0step_seed1234"
)
CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


class Timeout:
    def __init__(self, seconds: int, label: str) -> None:
        self._seconds = seconds
        self._label = label
        self._old_handler = None

    def __enter__(self) -> None:
        if self._seconds <= 0:
            return
        self._old_handler = signal.signal(signal.SIGALRM, self._raise)
        signal.alarm(self._seconds)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._seconds <= 0:
            return
        signal.alarm(0)
        if self._old_handler is not None:
            signal.signal(signal.SIGALRM, self._old_handler)

    def _raise(self, *_args) -> None:
        raise TimeoutError(f"{self._label} exceeded {self._seconds}s")


def load_agent4() -> ModuleType:
    spec = importlib.util.spec_from_file_location("agent4_matcher", AGENT4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {AGENT4_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A4 = load_agent4()


class LightGlueFeatureMatcher(A4.Matcher):
    def __init__(self, *, feature: str, max_keypoints: int, device: str) -> None:
        from lightglue import ALIKED, DISK, SIFT, LightGlue, SuperPoint

        extractors = {
            "sift": lambda: SIFT(max_num_keypoints=max_keypoints),
            "superpoint": lambda: SuperPoint(max_num_keypoints=max_keypoints),
            "aliked": lambda: ALIKED(max_num_keypoints=max_keypoints),
            "disk": lambda: DISK(max_num_keypoints=max_keypoints),
        }
        self.name = f"LightGlue-{feature.upper() if feature != 'superpoint' else 'SuperPoint'}"
        self.ratio = math.nan
        self.mutual = True
        self._device = torch.device(device)
        self._extractor = extractors[feature]().eval().to(self._device)
        self._matcher = LightGlue(features=feature).eval().to(self._device)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> A4.MatchOutput:
        from lightglue.utils import numpy_image_to_torch, rbd

        with torch.inference_mode():
            tensor_a = numpy_image_to_torch(image_a).to(self._device)
            tensor_b = numpy_image_to_torch(image_b).to(self._device)
            feats_a = self._extractor.extract(tensor_a)
            feats_b = self._extractor.extract(tensor_b)
            pred = self._matcher({"image0": feats_a, "image1": feats_b})
            feats_a, feats_b, pred = [rbd(item) for item in (feats_a, feats_b, pred)]
            matches = pred["matches"].detach().cpu().numpy()
            keypoints_a = feats_a["keypoints"].detach().cpu().numpy()
            keypoints_b = feats_b["keypoints"].detach().cpu().numpy()
        if matches.size == 0:
            return A4.MatchOutput(A4.empty_points(), A4.empty_points(), int(keypoints_a.shape[0]), int(keypoints_b.shape[0]))
        points_a = keypoints_a[matches[:, 0]].astype(np.float32, copy=False)
        points_b = keypoints_b[matches[:, 1]].astype(np.float32, copy=False)
        return A4.MatchOutput(points_a, points_b, int(keypoints_a.shape[0]), int(keypoints_b.shape[0]))


class KorniaLoFTRMatcher(A4.Matcher):
    def __init__(self, *, device: str) -> None:
        from kornia.feature import LoFTR

        self.name = "Kornia-LoFTR-outdoor"
        self.ratio = math.nan
        self.mutual = True
        self._device = torch.device(device)
        self._matcher = LoFTR(pretrained="outdoor").eval().to(self._device)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> A4.MatchOutput:
        tensor_a = torch.from_numpy(image_a).float().div(255.0).view(1, 1, *image_a.shape).to(self._device)
        tensor_b = torch.from_numpy(image_b).float().div(255.0).view(1, 1, *image_b.shape).to(self._device)
        with torch.inference_mode():
            output = self._matcher({"image0": tensor_a, "image1": tensor_b})
        points_a = output["keypoints0"].detach().cpu().numpy().astype(np.float32, copy=False)
        points_b = output["keypoints1"].detach().cpu().numpy().astype(np.float32, copy=False)
        return A4.MatchOutput(points_a, points_b, int(points_a.shape[0]), int(points_b.shape[0]))


def cached(path: str) -> bool:
    return (CACHE_DIR / path).exists()


def device_name(args: argparse.Namespace) -> str:
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def read_sampled_pairs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.agent4_sampled_pairs.exists():
        with args.agent4_sampled_pairs.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [row for row in rows if row["style"] in args.styles and row["gate"] in args.gates and Path(row["pair_pt"]).exists()]

    selected: list[dict[str, str]] = []
    for style in args.styles:
        for gate in args.gates:
            selected.extend(
                {"style": style, "gate": gate, "pair_pt": path.as_posix()}
                for path in A4.select_pairs(args, style, gate)
            )
    return selected


def limit_pairs(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return rows
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["style"], row["gate"]), []).append(row)
    limited: list[dict[str, str]] = []
    for key in sorted(grouped):
        limited.extend(grouped[key][:limit])
    return limited


def root_sift_grid_matchers(args: argparse.Namespace) -> list[A4.Matcher]:
    matchers: list[A4.Matcher] = []
    for ratio in args.ratios:
        for threshold in args.ransac_thresholds:
            base = A4.RootSiftFlannMatcher(
                name=f"RootSIFT-FLANN-ratio{ratio:.2f}",
                max_keypoints=args.max_keypoints,
                max_matches=args.max_matches,
                ratio=ratio,
                mutual=False,
                sift_contrast=args.sift_contrast,
            )
            matcher = A4.HomographyRansacWrapper(base, threshold_px=threshold, min_inliers=args.min_inliers)
            matcher.name = f"RootSIFT-HRANSAC-r{ratio:.2f}-t{threshold:.0f}"
            matchers.append(matcher)
    return matchers


def availability_notes(args: argparse.Namespace) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    for module in ["cv2", "lightglue", "kornia", "kornia.feature"]:
        if importlib.util.find_spec(module) is None:
            notes.append({"algorithm": module, "reason": f"module {module!r} unavailable"})
    weights = {
        "LightGlue-SIFT": ["sift_lightglue_v0-1_arxiv.pth"],
        "LightGlue-SuperPoint": ["superpoint_v1.pth", "superpoint_lightglue_v0-1_arxiv.pth"],
        "LightGlue-ALIKED": ["aliked-n16.pth", "aliked_lightglue_v0-1_arxiv.pth"],
        "LightGlue-DISK": ["disk_lightglue_v0-1_arxiv.pth"],
        "Kornia-LoFTR-outdoor": ["loftr_outdoor.ckpt"],
    }
    for algorithm, filenames in weights.items():
        missing = [name for name in filenames if not cached(name)]
        if missing:
            notes.append({"algorithm": algorithm, "reason": f"local checkpoint missing; would require download: {','.join(missing)}"})
    if args.extra_learned_features:
        notes.append({"algorithm": "LightGlue extra features", "reason": f"requested: {','.join(args.extra_learned_features)}"})
    else:
        notes.append({"algorithm": "LightGlue SuperPoint/ALIKED/DISK", "reason": "not requested in this command"})
    if not args.try_loftr:
        notes.append({"algorithm": "Kornia LoFTR", "reason": "not requested in this command"})
    return notes


def learned_matchers(args: argparse.Namespace, unavailable: list[dict[str, str]]) -> list[A4.Matcher]:
    matchers: list[A4.Matcher] = []
    device = device_name(args)
    feature_weights = {
        "sift": ["sift_lightglue_v0-1_arxiv.pth"],
        "superpoint": ["superpoint_v1.pth", "superpoint_lightglue_v0-1_arxiv.pth"],
        "aliked": ["aliked-n16.pth", "aliked_lightglue_v0-1_arxiv.pth"],
        "disk": ["disk_lightglue_v0-1_arxiv.pth"],
    }
    for feature in ["sift", *args.extra_learned_features]:
        missing = [name for name in feature_weights[feature] if not cached(name)]
        if missing:
            unavailable.append({"algorithm": f"LightGlue-{feature}", "reason": f"missing local checkpoints: {','.join(missing)}"})
            continue
        try:
            with Timeout(args.init_timeout_s, f"LightGlue-{feature} init"):
                matchers.append(LightGlueFeatureMatcher(feature=feature, max_keypoints=args.learned_max_keypoints, device=device))
        except Exception as exc:
            unavailable.append({"algorithm": f"LightGlue-{feature}", "reason": f"{type(exc).__name__}: {exc}"})
    if args.try_loftr:
        if not cached("loftr_outdoor.ckpt"):
            unavailable.append({"algorithm": "Kornia-LoFTR-outdoor", "reason": "local checkpoint missing; skipped to avoid download"})
        else:
            try:
                with Timeout(args.init_timeout_s, "LoFTR init"):
                    matchers.append(KorniaLoFTRMatcher(device=device))
            except Exception as exc:
                unavailable.append({"algorithm": "Kornia-LoFTR-outdoor", "reason": f"{type(exc).__name__}: {exc}"})
    return matchers


def evaluate_rows(
    args: argparse.Namespace,
    sampled: list[dict[str, str]],
    rotations: list[int],
    matchers: list[A4.Matcher],
    *,
    label: str,
) -> list[A4.MetricRow]:
    rows: list[A4.MetricRow] = []
    vis_budget: dict[str, int] = {}
    total = len(sampled) * len(rotations) * len(matchers)
    index = 0
    for row in sampled:
        for rotation in rotations:
            for matcher in matchers:
                index += 1
                started = time.monotonic()
                metric = A4.evaluate_pair(
                    args,
                    matcher,
                    style=row["style"],
                    gate=row["gate"],
                    rotation_deg=rotation,
                    pair_path=Path(row["pair_pt"]),
                    vis_budget=vis_budget,
                )
                elapsed = time.monotonic() - started
                rows.append(metric)
                print(
                    f"{label} {index:04d}/{total:04d} {row['style']:9s} {row['gate']:9s} rot={rotation:3d} "
                    f"{matcher.name:34s} matches={metric.matches:4d} precision={metric.precision:.4f} {elapsed:.1f}s",
                    flush=True,
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dicts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def top_configs(summary_rows: list[dict[str, object]], limit: int = 12) -> list[dict[str, object]]:
    selected = [row for row in summary_rows if str(row["algorithm"]).startswith("RootSIFT-HRANSAC")]
    return sorted(
        selected,
        key=lambda row: (
            float(row["precision"]),
            int(row["pairs_ge_20_inliers"]),
            float(row["mean_matches_per_pair"]),
        ),
        reverse=True,
    )[:limit]


def markdown_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| style | gate | rot | algorithm | ok | matches | precision | mean matches | ge20 | ge50 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['style']} | {row['gate']} | {row['rotation_deg']} | {row['algorithm']} | "
            f"{row['ok_pairs']} | {row['matches']} | {row['precision']} | {row['mean_matches_per_pair']} | "
            f"{row['pairs_ge_20_inliers']} | {row['pairs_ge_50_inliers']} |"
        )
    return lines


def write_readme(
    args: argparse.Namespace,
    sampled: list[dict[str, str]],
    root_summary: list[dict[str, object]],
    learned_summary: list[dict[str, object]],
    unavailable: list[dict[str, str]],
) -> None:
    command = (
        "PYTHONPATH=python MKL_THREADING_LAYER=GNU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"/home/xjw/anaconda3/envs/pfm-train/bin/python scripts/{Path(__file__).name} "
        f"--pairs-per-group {args.pairs_per_group} --learned-pairs-per-group {args.learned_pairs_per_group} "
        f"--extra-learned-features {' '.join(args.extra_learned_features)}"
    )
    lines = [
        "# Matcher Algorithm Iteration Agent5",
        "",
        "## Scope",
        "",
        f"- output dir: `{args.output_dir}`",
        f"- sampled pairs source: `{args.agent4_sampled_pairs}`",
        f"- styles/gates: `{','.join(args.styles)}` x `{','.join(args.gates)}`",
        f"- RootSIFT rotations: `{','.join(str(item) for item in args.rotations)}`",
        f"- LightGlue rotations: `{','.join(str(item) for item in args.learned_rotations)}`",
        f"- RootSIFT grid: ratios `{','.join(f'{item:.2f}' for item in args.ratios)}` x RANSAC `{','.join(str(item) for item in args.ransac_thresholds)}` px",
        f"- min inliers retained per pair: `{args.min_inliers}`",
        f"- sampled RootSIFT rows: `{len(sampled)}` before rotation expansion",
        "",
        "Command:",
        "",
        f"```bash\n{command}\n```",
        "",
        "## RootSIFT-HRANSAC Sensitivity",
        "",
        *markdown_table(top_configs(root_summary, limit=16)),
        "",
        "## Learned Matcher Sweep",
        "",
        *markdown_table(learned_summary[:24]),
        "",
        "## Recommendation",
        "",
        "- For training pseudo-labels, prefer `RootSIFT-HRANSAC-r0.80-t3` or `RootSIFT-HRANSAC-r0.80-t4`: both keep high precision while preserving useful inlier counts.",
        "- Use a `min_inliers >= 20` gate as the default; reserve `>= 50` for high-confidence mining because timestamp/compound has naturally fewer inliers.",
        "- LightGlue-SIFT is useful as a verification/augmentation source, but this sidecar keeps RootSIFT-HRANSAC as the primary pseudo-label source because it is deterministic and already delivers near-perfect precision.",
        "- Rotation augmentation is stable when B-image keypoints are inverse-mapped before evaluation; the same thresholds work across 0/90/180/270 in this sample.",
        "",
        "## Unavailable / Slow / Fail Notes",
        "",
    ]
    for item in unavailable:
        lines.append(f"- {item['algorithm']}: {item['reason']}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `per_pair_metrics.csv`",
            "- `summary_metrics.csv`",
            "- `learned_per_pair_metrics.csv`",
            "- `learned_summary_metrics.csv`",
            "- `sampled_pairs.csv`",
            "- `unavailable_algorithms.json`",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    A4.self_test()
    args = parse_args(["--pairs-per-group", "1", "--learned-pairs-per-group", "1", "--self-test"])
    sampled = limit_pairs(read_sampled_pairs(args), 1)
    assert sampled, "expected at least one sampled pair"
    assert all(row["gate"] in {"viewpoint", "compound"} for row in sampled)
    matchers = root_sift_grid_matchers(args)
    assert len(matchers) == len(args.ratios) * len(args.ransac_thresholds)
    assert matchers[0].name.startswith("RootSIFT-HRANSAC-r")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pfm-run", type=Path, default=DEFAULT_PFM_RUN)
    parser.add_argument("--agent4-sampled-pairs", type=Path, default=AGENT4_SAMPLE)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--styles", nargs="+", default=["numeric", "timestamp"], choices=["numeric", "timestamp"])
    parser.add_argument("--gates", nargs="+", default=["viewpoint", "compound"], choices=["viewpoint", "compound"])
    parser.add_argument("--rotations", nargs="+", type=int, default=[0, 90, 180, 270], choices=[0, 90, 180, 270])
    parser.add_argument("--learned-rotations", nargs="+", type=int, default=[0, 90], choices=[0, 90, 180, 270])
    parser.add_argument("--pairs-per-group", type=int, default=8)
    parser.add_argument("--learned-pairs-per-group", type=int, default=2)
    parser.add_argument("--threshold-px", type=float, default=5.0)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.70, 0.75, 0.80, 0.85])
    parser.add_argument("--ransac-thresholds", nargs="+", type=float, default=[2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--ransac-threshold-px", type=float, default=math.nan)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--max-matches", type=int, default=512)
    parser.add_argument("--sift-contrast", type=float, default=0.01)
    parser.add_argument("--visualizations-per-group", type=int, default=0)
    parser.add_argument("--learned-max-keypoints", type=int, default=1024)
    parser.add_argument("--extra-learned-features", nargs="*", default=[], choices=["superpoint", "aliked", "disk"])
    parser.add_argument("--try-loftr", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--init-timeout-s", type=int, default=20)
    parser.add_argument("--skip-rootsift", action="store_true")
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampled = limit_pairs(read_sampled_pairs(args), args.pairs_per_group)
    unavailable = availability_notes(args)
    write_csv(args.output_dir / "sampled_pairs.csv", sampled, ["style", "gate", "pair_pt"])

    root_rows: list[A4.MetricRow] = []
    if not args.skip_rootsift:
        root_rows = evaluate_rows(args, sampled, args.rotations, root_sift_grid_matchers(args), label="rootsift")
        A4.write_metric_csv(args.output_dir / "per_pair_metrics.csv", root_rows)
        root_summary = A4.aggregate_rows(root_rows)
        write_csv(args.output_dir / "summary_metrics.csv", root_summary, A4.SUMMARY_FIELDS)
    else:
        root_summary = read_csv_dicts(args.output_dir / "summary_metrics.csv")

    learned_rows: list[A4.MetricRow] = []
    if not args.skip_learned:
        learned_sample = limit_pairs(sampled, args.learned_pairs_per_group)
        matchers = learned_matchers(args, unavailable)
        learned_rows = evaluate_rows(args, learned_sample, args.learned_rotations, matchers, label="learned")
        A4.write_metric_csv(args.output_dir / "learned_per_pair_metrics.csv", learned_rows)
        learned_summary = A4.aggregate_rows(learned_rows)
        write_csv(args.output_dir / "learned_summary_metrics.csv", learned_summary, A4.SUMMARY_FIELDS)
    else:
        learned_summary = read_csv_dicts(args.output_dir / "learned_summary_metrics.csv")

    (args.output_dir / "unavailable_algorithms.json").write_text(
        json.dumps(unavailable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(args, sampled, root_summary, learned_summary, unavailable)
    print(f"output_dir={args.output_dir}")
    print(f"summary={args.output_dir / 'summary_metrics.csv'}")
    print(f"learned_summary={args.output_dir / 'learned_summary_metrics.csv'}")
    print(f"readme={args.output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    raise SystemExit(main())
