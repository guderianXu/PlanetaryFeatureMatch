#!/usr/bin/env python3
"""Agent13 stage5 hard-tail mining with different matcher paradigms."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import pseudo_label_generation as plg


AGENT13_SCRIPT = PROJECT_ROOT / "scripts" / "matcher_algorithm_iteration_agent13.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage5"
STAGE4_HARDTAIL = PROJECT_ROOT / "runs" / "matcher_algorithm_iteration_agent13_stage4" / "hardtail_pairs.csv"


PAIR_FIELDS = [
    "source_name",
    "pair_name",
    "pair_rel",
    "profile",
    "algorithm",
    "teacher_use",
    "status",
    "keypoints_a",
    "keypoints_b",
    "raw_matches",
    "homography_inliers",
    "truth_labels",
    "wrong_inliers",
    "truth_precision",
    "kept_pair",
    "diagnostic_pair",
    "candidate_labels",
    "homography_pass",
    "min_labels",
    "min_truth_precision",
    "mean_error_px",
    "median_error_px",
    "runtime_ms",
    "failure_reason",
    "message",
]

SUMMARY_FIELDS = [
    "profile",
    "algorithm",
    "teacher_use",
    "hardtail_pairs",
    "ok_pairs",
    "homography_pass_pairs",
    "homography_pass_rate",
    "kept_pairs",
    "kept_pair_rate",
    "unique_kept_pairs",
    "candidate_labels",
    "kept_truth_labels",
    "kept_homography_inliers",
    "kept_truth_precision",
    "truth_labels",
    "homography_inliers",
    "wrong_inliers",
    "truth_precision",
    "median_inliers",
    "median_truth_labels",
    "kept_sources",
    "failure_counts",
]

LABEL_FIELDS = ["profile", "algorithm", "teacher_use", "source_name", "pair_name", "pair_rel", "ax", "ay", "bx", "by", "error_px"]
SKIPPED_FIELDS = ["profile", "algorithm", "reason"]


@dataclass(frozen=True)
class MatchOutput:
    points_a: np.ndarray
    points_b: np.ndarray
    keypoints_a: int
    keypoints_b: int
    raw_matches: int


@dataclass(frozen=True)
class TeacherConfig:
    profile: str
    algorithm: str
    kind: str
    teacher_use: str
    ransac_threshold_px: float = 3.0
    min_inliers: int = 4
    max_keypoints: int = 0
    loftr_pretrained: str = "outdoor"
    loftr_max_side: int = 640
    flow_step: int = 24


@dataclass(frozen=True)
class PairRow:
    source_name: str
    pair_name: str
    pair_rel: str
    profile: str
    algorithm: str
    teacher_use: str
    status: str
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    homography_inliers: int
    truth_labels: int
    wrong_inliers: int
    truth_precision: float
    kept_pair: int
    diagnostic_pair: int
    candidate_labels: int
    homography_pass: int
    min_labels: int
    min_truth_precision: float
    mean_error_px: float
    median_error_px: float
    runtime_ms: float
    failure_reason: str
    message: str = ""


@dataclass(frozen=True)
class LabelRow:
    profile: str
    algorithm: str
    teacher_use: str
    source_name: str
    pair_name: str
    pair_rel: str
    ax: float
    ay: float
    bx: float
    by: float
    error_px: float


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A13 = load_module(AGENT13_SCRIPT, "agent13_matcher_for_stage5")
A4 = A13.A4


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def format_value(value: object) -> object:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_hardtail(path: Path) -> list[dict[str, object]]:
    out = []
    for row in read_csv(path):
        pair_path = repo_path(row["pair_rel"])
        out.append(
            {
                "path": pair_path,
                "source_name": row["source_name"],
                "pair_name": row["pair_name"],
                "pair_rel": rel(pair_path),
            }
        )
    return out


class LightGlueSiftMatcher:
    def __init__(self, max_keypoints: int, device: str) -> None:
        self.impl = A13.LightGlueMatcher(max_keypoints=max_keypoints, device=device)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        output = self.impl.match(image_a, image_b)
        return MatchOutput(output.points_a, output.points_b, output.keypoints_a, output.keypoints_b, output.raw_matches)


class LoFTRMatcher:
    def __init__(self, pretrained: str, max_side: int, device: str) -> None:
        import torch
        import kornia.feature as KF

        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.max_side = max_side
        self.matcher = KF.LoFTR(pretrained=pretrained).eval().to(self.device)

    def _resize(self, image: np.ndarray):
        import cv2

        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / float(max(h, w)))
        new_h = max(8, int(round(h * scale / 8.0)) * 8)
        new_w = max(8, int(round(w * scale / 8.0)) * 8)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        sx = w / float(new_w)
        sy = h / float(new_h)
        return resized, sx, sy

    def _tensor(self, image: np.ndarray):
        tensor = self.torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]
        return tensor.to(self.device)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        a, sx_a, sy_a = self._resize(image_a)
        b, sx_b, sy_b = self._resize(image_b)
        with self.torch.inference_mode():
            pred = self.matcher({"image0": self._tensor(a), "image1": self._tensor(b)})
        if "keypoints0" not in pred or pred["keypoints0"].numel() == 0:
            return MatchOutput(plg.empty_points(), plg.empty_points(), 0, 0, 0)
        p0 = pred["keypoints0"].detach().cpu().numpy().astype(np.float32)
        p1 = pred["keypoints1"].detach().cpu().numpy().astype(np.float32)
        if "confidence" in pred:
            conf = pred["confidence"].detach().cpu().numpy()
            order = np.argsort(-conf)
            p0 = p0[order]
            p1 = p1[order]
        p0[:, 0] *= sx_a
        p0[:, 1] *= sy_a
        p1[:, 0] *= sx_b
        p1[:, 1] *= sy_b
        return MatchOutput(p0, p1, int(p0.shape[0]), int(p1.shape[0]), int(p0.shape[0]))


class DenseFlowMatcher:
    def __init__(self, method: str, step: int) -> None:
        self.method = method
        self.step = step

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchOutput:
        import cv2

        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(image_a, image_b, None, 0.5, 5, 31, 5, 7, 1.5, 0)
        elif self.method == "dis":
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow = dis.calc(image_a, image_b, None)
        else:
            raise ValueError(self.method)
        h, w = image_a.shape[:2]
        ys = np.arange(self.step // 2, h, self.step, dtype=np.float32)
        xs = np.arange(self.step // 2, w, self.step, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        points_a = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
        ix = np.clip(np.rint(points_a[:, 0]).astype(np.int64), 0, w - 1)
        iy = np.clip(np.rint(points_a[:, 1]).astype(np.int64), 0, h - 1)
        disp = flow[iy, ix].astype(np.float32)
        points_b = points_a + disp
        inside = (points_b[:, 0] >= 0) & (points_b[:, 0] < w) & (points_b[:, 1] >= 0) & (points_b[:, 1] < h)
        points_a = points_a[inside]
        points_b = points_b[inside]
        return MatchOutput(points_a, points_b, int(points_a.shape[0]), int(points_b.shape[0]), int(points_a.shape[0]))


def teacher_configs() -> list[TeacherConfig]:
    return [
        TeacherConfig("loftr_outdoor_640", "Kornia-LoFTR-outdoor-max640", "loftr", "train_candidate", loftr_pretrained="outdoor", loftr_max_side=640),
        TeacherConfig("loftr_indoor_640", "Kornia-LoFTR-indoor-max640", "loftr", "diagnostic", loftr_pretrained="indoor", loftr_max_side=640),
        TeacherConfig("lightglue_sift_k2048", "LightGlue-SIFT-k2048-Ht3", "lightglue", "train_candidate", max_keypoints=2048),
        TeacherConfig("lightglue_sift_k4096", "LightGlue-SIFT-k4096-Ht3", "lightglue", "train_candidate", max_keypoints=4096),
        TeacherConfig("farneback_grid24", "Farneback-grid24-Ht3", "dense_flow_farneback", "diagnostic", flow_step=24),
        TeacherConfig("disflow_grid24", "DISFlow-grid24-Ht3", "dense_flow_dis", "diagnostic", flow_step=24),
    ]


class MatcherFactory:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cache: dict[str, object] = {}
        self.skipped: list[dict[str, str]] = []

    def matcher(self, config: TeacherConfig) -> object | None:
        if config.profile in self.cache:
            return self.cache[config.profile]
        try:
            if config.kind == "loftr":
                if importlib.util.find_spec("kornia.feature") is None:
                    self.skipped.append({"profile": config.profile, "algorithm": config.algorithm, "reason": "kornia.feature unavailable"})
                    return None
                matcher = LoFTRMatcher(config.loftr_pretrained, config.loftr_max_side, self.args.device)
            elif config.kind == "lightglue":
                if importlib.util.find_spec("lightglue") is None:
                    self.skipped.append({"profile": config.profile, "algorithm": config.algorithm, "reason": "lightglue unavailable"})
                    return None
                matcher = LightGlueSiftMatcher(config.max_keypoints, self.args.device)
            elif config.kind == "dense_flow_farneback":
                matcher = DenseFlowMatcher("farneback", config.flow_step)
            elif config.kind == "dense_flow_dis":
                matcher = DenseFlowMatcher("dis", config.flow_step)
            else:
                raise ValueError(config.kind)
            self.cache[config.profile] = matcher
            return matcher
        except Exception as exc:
            self.skipped.append({"profile": config.profile, "algorithm": config.algorithm, "reason": f"{type(exc).__name__}: {exc}"})
            return None


def evaluate_pair(args: argparse.Namespace, config: TeacherConfig, matcher: object, item: dict[str, object]) -> tuple[PairRow, list[LabelRow]]:
    start = time.perf_counter()
    try:
        image_a, image_b, warp_a_to_b, valid_mask = plg.load_pair(Path(item["path"]))
        raw = matcher.match(image_a, image_b)
        inlier_a, inlier_b = A4.ransac_inliers(raw.points_a, raw.points_b, threshold_px=config.ransac_threshold_px)
        homography_pass = int(inlier_a.shape[0] >= config.min_inliers)
        if not homography_pass:
            inlier_a, inlier_b = plg.empty_points(), plg.empty_points()
        truth_a, truth_b, errors = plg.filter_matches_by_warp_truth(
            inlier_a,
            inlier_b,
            warp_a_to_b,
            valid_mask,
            threshold_px=args.truth_threshold_px,
        )
        capped_a, capped_b, capped_errors = plg.cap_matches(
            truth_a,
            truth_b,
            errors,
            max_matches=args.max_labels_per_pair,
            seed=args.seed + hash((config.profile, item["pair_rel"])) % 100000,
        )
        truth_labels = int(truth_a.shape[0])
        inliers = int(inlier_a.shape[0])
        wrong = max(0, inliers - truth_labels)
        precision = 0.0 if inliers == 0 else truth_labels / inliers
        kept = int(homography_pass and truth_labels >= args.min_labels and precision >= args.min_truth_precision)
        diagnostic = int(homography_pass and truth_labels >= args.min_labels and not kept)
        if raw.raw_matches < 4:
            reason = "too_few_raw_matches"
        elif not homography_pass:
            reason = "homography_fail"
        elif truth_labels < args.min_labels:
            reason = "too_few_truth_labels"
        elif precision < args.min_truth_precision:
            reason = "low_truth_precision"
        else:
            reason = "kept"
        labels: list[LabelRow] = []
        if kept and config.teacher_use != "diagnostic":
            labels = [
                LabelRow(
                    profile=config.profile,
                    algorithm=config.algorithm,
                    teacher_use=config.teacher_use,
                    source_name=str(item["source_name"]),
                    pair_name=str(item["pair_name"]),
                    pair_rel=str(item["pair_rel"]),
                    ax=float(a[0]),
                    ay=float(a[1]),
                    bx=float(b[0]),
                    by=float(b[1]),
                    error_px=float(error),
                )
                for a, b, error in zip(capped_a, capped_b, capped_errors)
            ]
        return (
            PairRow(
                source_name=str(item["source_name"]),
                pair_name=str(item["pair_name"]),
                pair_rel=str(item["pair_rel"]),
                profile=config.profile,
                algorithm=config.algorithm,
                teacher_use=config.teacher_use,
                status="ok",
                keypoints_a=raw.keypoints_a,
                keypoints_b=raw.keypoints_b,
                raw_matches=raw.raw_matches,
                homography_inliers=inliers,
                truth_labels=truth_labels,
                wrong_inliers=wrong,
                truth_precision=precision,
                kept_pair=kept,
                diagnostic_pair=diagnostic,
                candidate_labels=len(labels),
                homography_pass=homography_pass,
                min_labels=args.min_labels,
                min_truth_precision=args.min_truth_precision,
                mean_error_px=float(errors.mean()) if errors.size else math.nan,
                median_error_px=float(np.median(errors)) if errors.size else math.nan,
                runtime_ms=(time.perf_counter() - start) * 1000.0,
                failure_reason=reason,
            ),
            labels,
        )
    except Exception as exc:
        return (
            PairRow(
                source_name=str(item["source_name"]),
                pair_name=str(item["pair_name"]),
                pair_rel=str(item["pair_rel"]),
                profile=config.profile,
                algorithm=config.algorithm,
                teacher_use=config.teacher_use,
                status="error",
                keypoints_a=0,
                keypoints_b=0,
                raw_matches=0,
                homography_inliers=0,
                truth_labels=0,
                wrong_inliers=0,
                truth_precision=0.0,
                kept_pair=0,
                diagnostic_pair=0,
                candidate_labels=0,
                homography_pass=0,
                min_labels=args.min_labels,
                min_truth_precision=args.min_truth_precision,
                mean_error_px=math.nan,
                median_error_px=math.nan,
                runtime_ms=(time.perf_counter() - start) * 1000.0,
                failure_reason="error",
                message=f"{type(exc).__name__}: {exc}",
            ),
            [],
        )


def summarize(rows: list[PairRow], configs: list[TeacherConfig]) -> list[dict[str, object]]:
    by_config = {config.profile: config for config in configs}
    grouped: dict[str, list[PairRow]] = {}
    for row in rows:
        grouped.setdefault(row.profile, []).append(row)
    out: list[dict[str, object]] = []
    for profile, items in sorted(grouped.items()):
        ok = [row for row in items if row.status == "ok"]
        kept = [row for row in ok if row.kept_pair]
        failures: dict[str, int] = {}
        for row in items:
            failures[row.failure_reason] = failures.get(row.failure_reason, 0) + 1
        total_inliers = sum(row.homography_inliers for row in ok)
        total_truth = sum(row.truth_labels for row in ok)
        kept_inliers = sum(row.homography_inliers for row in kept)
        kept_truth = sum(row.truth_labels for row in kept)
        config = by_config[profile]
        out.append(
            {
                "profile": profile,
                "algorithm": config.algorithm,
                "teacher_use": config.teacher_use,
                "hardtail_pairs": len(items),
                "ok_pairs": len(ok),
                "homography_pass_pairs": sum(row.homography_pass for row in ok),
                "homography_pass_rate": 0.0 if not ok else sum(row.homography_pass for row in ok) / len(ok),
                "kept_pairs": len(kept),
                "kept_pair_rate": 0.0 if not ok else len(kept) / len(ok),
                "unique_kept_pairs": len({row.pair_rel for row in kept}),
                "candidate_labels": sum(row.candidate_labels for row in ok),
                "kept_truth_labels": kept_truth,
                "kept_homography_inliers": kept_inliers,
                "kept_truth_precision": 0.0 if kept_inliers == 0 else kept_truth / kept_inliers,
                "truth_labels": total_truth,
                "homography_inliers": total_inliers,
                "wrong_inliers": sum(row.wrong_inliers for row in ok),
                "truth_precision": 0.0 if total_inliers == 0 else total_truth / total_inliers,
                "median_inliers": float(np.median([row.homography_inliers for row in ok])) if ok else math.nan,
                "median_truth_labels": float(np.median([row.truth_labels for row in ok])) if ok else math.nan,
                "kept_sources": len({row.source_name for row in kept}),
                "failure_counts": ";".join(f"{key}:{value}" for key, value in sorted(failures.items())),
            }
        )
    return out


def ranking(row: dict[str, object]) -> tuple[int, int, float, str]:
    return (-int(row["unique_kept_pairs"]), -int(row["candidate_labels"]), -float(row["kept_truth_precision"]), str(row["profile"]))


def write_summary(
    args: argparse.Namespace,
    summary_rows: list[dict[str, object]],
    pair_rows: list[PairRow],
    label_rows: list[LabelRow],
    skipped: list[dict[str, str]],
    hardtail_count: int,
) -> None:
    ordered = sorted(summary_rows, key=ranking)
    train_kept_pairs = {row.pair_rel for row in pair_rows if row.status == "ok" and row.kept_pair and row.teacher_use != "diagnostic"}
    all_kept_pairs = {row.pair_rel for row in pair_rows if row.status == "ok" and row.kept_pair}
    candidate_pairs = {row.pair_rel for row in label_rows}
    lines = [
        "# Agent13 Stage5 Hard-Tail Matcher Paradigm Test",
        "",
        "## Scope",
        "",
        f"- Hard-tail pairs: `{args.hardtail_csv}`; recovered `{hardtail_count}` pairs.",
        f"- Kept gate: truth threshold `{args.truth_threshold_px}` px, min labels `{args.min_labels}`, min pair precision `{args.min_truth_precision}`.",
        "- Tested LoFTR, larger LightGlue-SIFT keypoint budgets, and dense optical flow diagnostics.",
        "- No PFM training was run.",
        "",
        "## Summary Metrics",
        "",
        "| profile | use | kept pairs | candidate labels | kept precision | all-pair precision | median inliers/truth | failures |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['profile']} | {row['teacher_use']} | {row['unique_kept_pairs']} | {row['candidate_labels']} | "
            f"{float(row['kept_truth_precision']):.4f} | {float(row['truth_precision']):.4f} | "
            f"{float(row['median_inliers']):.1f}/{float(row['median_truth_labels']):.1f} | {row['failure_counts']} |"
        )
    lines.extend(["", "## Findings", ""])
    lines.append(
        f"- Stage5 unique non-diagnostic kept pairs: {len(train_kept_pairs)}/{hardtail_count}; candidate-label pairs: {len(candidate_pairs)}/{hardtail_count}; all kept including diagnostics: {len(all_kept_pairs)}/{hardtail_count}. Stage4 baseline was 1/24."
    )
    if ordered:
        best = ordered[0]
        lines.append(
            f"- Best Stage5 profile by unique kept pairs: `{best['profile']}` with {best['unique_kept_pairs']}/{hardtail_count} pairs and {best['candidate_labels']} candidate labels."
        )
    train = [row for row in ordered if row["teacher_use"] != "diagnostic" and int(row["candidate_labels"]) > 0]
    if train:
        labels = sum(int(row["candidate_labels"]) for row in train)
        pairs = max(int(row["unique_kept_pairs"]) for row in train)
        lines.append(f"- Non-diagnostic Stage5 candidate labels total {labels}; best unique kept coverage is {pairs}/{hardtail_count}.")
    else:
        lines.append("- No non-diagnostic Stage5 teacher produced training candidate labels.")
    lines.extend(
        [
            "- Dense optical-flow rows are diagnostic only; even when they produce many raw correspondences, warp-truth filtering decides whether they are usable.",
            "",
            "## Next Recommendation",
            "",
            "- If Stage5 does not substantially exceed Stage4's 1/24 unique kept pairs, stop trying to force sparse/dense matcher pseudo-labels for this hard-tail and move to data/pair selection or larger-context cache generation.",
            "- Use `candidate_labels.csv` only if it adds unique pairs beyond Stage4; otherwise treat it as diagnostic evidence.",
            "",
            "## Skipped / Unavailable",
            "",
        ]
    )
    if skipped:
        for item in skipped:
            lines.append(f"- {item['profile']} `{item['algorithm']}`: {item['reason']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `summary_metrics.csv`: per matcher paradigm hard-tail coverage.",
            "- `pair_metrics.csv`: per hard-tail pair diagnostics.",
            "- `candidate_labels.csv`: high-precision non-diagnostic labels with repo-relative `pair_rel`.",
            "- `skipped_teachers.csv`: unavailable matcher setup.",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    assert STAGE4_HARDTAIL.exists()
    configs = teacher_configs()
    assert any(config.kind == "loftr" for config in configs)
    assert any(config.kind == "dense_flow_farneback" for config in configs)
    assert load_hardtail(STAGE4_HARDTAIL)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hardtail-csv", type=Path, default=STAGE4_HARDTAIL)
    parser.add_argument("--truth-threshold-px", type=float, default=5.0)
    parser.add_argument("--min-truth-precision", type=float, default=0.95)
    parser.add_argument("--min-labels", type=int, default=8)
    parser.add_argument("--max-labels-per-pair", type=int, default=128)
    parser.add_argument("--seed", type=int, default=5234)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("self-test ok")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hardtail = load_hardtail(args.hardtail_csv)
    configs = teacher_configs()
    factory = MatcherFactory(args)
    pair_rows: list[PairRow] = []
    label_rows: list[LabelRow] = []
    for config in configs:
        matcher = factory.matcher(config)
        if matcher is None:
            continue
        for item in hardtail:
            row, labels = evaluate_pair(args, config, matcher, item)
            pair_rows.append(row)
            label_rows.extend(labels)
        print(f"{config.profile} done", flush=True)
    summary_rows = summarize(pair_rows, configs)
    write_csv(args.output_dir / "pair_metrics.csv", [asdict(row) for row in pair_rows], PAIR_FIELDS)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.output_dir / "candidate_labels.csv", [asdict(row) for row in label_rows], LABEL_FIELDS)
    write_csv(args.output_dir / "skipped_teachers.csv", factory.skipped, SKIPPED_FIELDS)
    write_summary(args, summary_rows, pair_rows, label_rows, factory.skipped, len(hardtail))
    print(f"output_dir={args.output_dir}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
