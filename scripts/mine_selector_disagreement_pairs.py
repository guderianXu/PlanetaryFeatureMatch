#!/usr/bin/env python3
"""Mine reusable pair manifests from two selector outputs that disagree."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mine_hard_failure_pairs import PAIR_MANIFEST_FIELDS


EXTRA_FIELDS = [
    "hard_reasons",
    "hard_score",
    "match_delta",
    "correct_delta",
    "wrong_delta",
    "precision_delta",
    "source_name",
    "source_row_index",
    "source_active_label",
    "source_candidate_label",
    "source_active_selected_model",
    "source_candidate_selected_model",
    "source_active_selector_reason",
    "source_candidate_selector_reason",
    "source_active_matches",
    "source_active_correct",
    "source_active_wrong",
    "source_active_precision",
    "source_active_score_mean",
    "source_active_homography_p90_px",
    "source_candidate_matches",
    "source_candidate_correct",
    "source_candidate_wrong",
    "source_candidate_precision",
    "source_candidate_score_mean",
    "source_candidate_homography_p90_px",
    "match_delta_active_minus_candidate",
    "correct_delta_active_minus_candidate",
    "wrong_delta_candidate_minus_active",
    "precision_delta_active_minus_candidate",
    "match_delta_candidate_minus_active",
    "correct_delta_candidate_minus_active",
    "wrong_delta_active_minus_candidate",
    "precision_delta_candidate_minus_active",
]


@dataclass(frozen=True)
class PairManifestSource:
    source_name: str
    split: str
    path: Path


@dataclass(frozen=True)
class DisagreementConfig:
    target_variants: tuple[str, ...] = ("extreme_02", "extreme_03")
    include_non_target_regressions: bool = False
    min_precision_drop: float = 0.001
    mine_mode: str = "active_regressions"
    min_candidate_correct_gain: int = 1
    max_candidate_wrong_increase: int = 0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _summary_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (
        row.get("source", ""),
        row.get("split", ""),
        _int_value(row, "row_index", -1),
    )


def _manifest_key(source_name: str, split: str, row_index: int) -> tuple[str, str, int]:
    return source_name, split, row_index


def _build_pair_lookup(sources: list[PairManifestSource]) -> dict[tuple[str, str, int], dict[str, str]]:
    lookup: dict[tuple[str, str, int], dict[str, str]] = {}
    for source in sources:
        rows = _read_csv_rows(source.path)
        for ordinal, row in enumerate(rows):
            lookup[_manifest_key(source.source_name, source.split, ordinal)] = row
    return lookup


def classify_disagreement(
    active: dict[str, str],
    candidate: dict[str, str],
    *,
    config: DisagreementConfig,
) -> tuple[list[str], float]:
    target_variant = active.get("target_variant", "") or candidate.get("target_variant", "")
    target_set = set(config.target_variants)
    is_target = target_variant in target_set

    active_matches = _int_value(active, "matches")
    active_correct = _int_value(active, "correct")
    active_wrong = _int_value(active, "wrong")
    active_precision = _float_value(active, "precision")
    candidate_matches = _int_value(candidate, "matches")
    candidate_correct = _int_value(candidate, "correct")
    candidate_wrong = _int_value(candidate, "wrong")
    candidate_precision = _float_value(candidate, "precision")

    active_correct_gain = active_correct - candidate_correct
    candidate_correct_gain = candidate_correct - active_correct
    candidate_wrong_increase = candidate_wrong - active_wrong
    active_wrong_increase = active_wrong - candidate_wrong
    active_match_gain = active_matches - candidate_matches
    candidate_match_gain = candidate_matches - active_matches
    active_precision_gain = active_precision - candidate_precision
    candidate_precision_gain = candidate_precision - active_precision

    reasons: list[str] = []
    if config.mine_mode == "candidate_gains":
        if candidate_correct_gain < config.min_candidate_correct_gain:
            return [], 0.0
        if candidate_wrong_increase > config.max_candidate_wrong_increase:
            return [], 0.0
        reasons.append("candidate_correct_gain")
        if candidate_match_gain > 0:
            reasons.append("candidate_match_gain")
        if active_wrong_increase > 0:
            reasons.append("candidate_wrong_reduction")
        if candidate_precision_gain > config.min_precision_drop:
            reasons.append("candidate_precision_gain")
        if active.get("selected_model", "") != candidate.get("selected_model", ""):
            reasons.append("selector_choice_disagreement")
    else:
        if active_correct_gain > 0:
            reasons.append("candidate_missed_active_correct")
        if candidate_wrong_increase > 0:
            reasons.append("candidate_wrong_increase")
        if active_match_gain > 0:
            reasons.append("candidate_match_drop")
        if active_precision_gain > config.min_precision_drop:
            reasons.append("candidate_precision_drop")
        if active.get("selected_model", "") != candidate.get("selected_model", ""):
            reasons.append("selector_choice_disagreement")

    if not reasons:
        return [], 0.0
    if not is_target and not config.include_non_target_regressions:
        return [], 0.0
    if is_target:
        reasons.append("extreme_view")

    if config.mine_mode == "candidate_gains":
        hard_score = (
            float(max(0, candidate_correct_gain)) * 10.0
            + float(max(0, active_wrong_increase)) * 5.0
            + float(max(0, candidate_match_gain))
            + max(0.0, candidate_precision_gain) * 100.0
        )
    else:
        hard_score = (
            float(max(0, active_correct_gain)) * 10.0
            + float(max(0, candidate_wrong_increase)) * 5.0
            + float(max(0, active_match_gain))
            + max(0.0, active_precision_gain) * 100.0
        )
    if active.get("selected_model", "") != candidate.get("selected_model", ""):
        hard_score += 2.0
    if is_target:
        hard_score += 5.0
    return list(dict.fromkeys(reasons)), hard_score


def _extra_values(
    active: dict[str, str],
    candidate: dict[str, str],
    *,
    active_label: str,
    candidate_label: str,
    reasons: list[str],
    hard_score: float,
    mine_mode: str,
) -> dict[str, str]:
    active_matches = _int_value(active, "matches")
    active_correct = _int_value(active, "correct")
    active_wrong = _int_value(active, "wrong")
    active_precision = _float_value(active, "precision")
    candidate_matches = _int_value(candidate, "matches")
    candidate_correct = _int_value(candidate, "correct")
    candidate_wrong = _int_value(candidate, "wrong")
    candidate_precision = _float_value(candidate, "precision")
    active_minus_candidate = {
        "matches": active_matches - candidate_matches,
        "correct": active_correct - candidate_correct,
        "wrong": active_wrong - candidate_wrong,
        "precision": active_precision - candidate_precision,
    }
    candidate_minus_active = {
        "matches": candidate_matches - active_matches,
        "correct": candidate_correct - active_correct,
        "wrong": candidate_wrong - active_wrong,
        "precision": candidate_precision - active_precision,
    }
    standard_delta = candidate_minus_active
    return {
        "hard_reasons": "|".join(reasons),
        "hard_score": f"{hard_score:.6f}",
        "source_name": active.get("source", ""),
        "source_row_index": str(_int_value(active, "row_index")),
        "source_active_label": active_label,
        "source_candidate_label": candidate_label,
        "source_active_selected_model": active.get("selected_model", ""),
        "source_candidate_selected_model": candidate.get("selected_model", ""),
        "source_active_selector_reason": active.get("selector_reason", ""),
        "source_candidate_selector_reason": candidate.get("selector_reason", ""),
        "source_active_matches": str(active_matches),
        "source_active_correct": str(active_correct),
        "source_active_wrong": str(active_wrong),
        "source_active_precision": f"{active_precision:.6f}",
        "source_active_score_mean": active.get("score_mean", ""),
        "source_active_homography_p90_px": active.get("homography_residual_p90_px", ""),
        "source_candidate_matches": str(candidate_matches),
        "source_candidate_correct": str(candidate_correct),
        "source_candidate_wrong": str(candidate_wrong),
        "source_candidate_precision": f"{candidate_precision:.6f}",
        "source_candidate_score_mean": candidate.get("score_mean", ""),
        "source_candidate_homography_p90_px": candidate.get("homography_residual_p90_px", ""),
        "match_delta": str(standard_delta["matches"]),
        "correct_delta": str(standard_delta["correct"]),
        "wrong_delta": str(standard_delta["wrong"]),
        "precision_delta": f"{standard_delta['precision']:.6f}",
        "match_delta_active_minus_candidate": str(active_minus_candidate["matches"]),
        "correct_delta_active_minus_candidate": str(active_minus_candidate["correct"]),
        "wrong_delta_candidate_minus_active": str(candidate_minus_active["wrong"]),
        "precision_delta_active_minus_candidate": f"{active_minus_candidate['precision']:.6f}",
        "match_delta_candidate_minus_active": str(candidate_minus_active["matches"]),
        "correct_delta_candidate_minus_active": str(candidate_minus_active["correct"]),
        "wrong_delta_active_minus_candidate": str(active_minus_candidate["wrong"]),
        "precision_delta_candidate_minus_active": f"{candidate_minus_active['precision']:.6f}",
    }


def mine_selector_disagreement_rows(
    *,
    active_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    pair_sources: list[PairManifestSource],
    active_label: str,
    candidate_label: str,
    config: DisagreementConfig,
) -> list[dict[str, str]]:
    pair_lookup = _build_pair_lookup(pair_sources)
    candidate_by_key = {_summary_key(row): row for row in candidate_rows}
    mined: list[dict[str, str]] = []
    for active in active_rows:
        key = _summary_key(active)
        candidate = candidate_by_key.get(key)
        if candidate is None:
            continue
        reasons, hard_score = classify_disagreement(active, candidate, config=config)
        if not reasons:
            continue
        pair = pair_lookup.get(key)
        if pair is None:
            continue
        row = {field: pair.get(field, "") for field in PAIR_MANIFEST_FIELDS}
        row.update(
            _extra_values(
                active,
                candidate,
                active_label=active_label,
                candidate_label=candidate_label,
                reasons=reasons,
                hard_score=hard_score,
                mine_mode=config.mine_mode,
            )
        )
        mined.append(row)

    mined.sort(
        key=lambda row: (
            -float(row.get("hard_score") or 0.0),
            row.get("split", ""),
            row.get("reference_base_id", ""),
            row.get("target_variant", ""),
        )
    )
    for index, row in enumerate(mined):
        row["pair_index"] = str(index)
    return mined


def summarize_rows(
    rows: list[dict[str, str]],
    *,
    active_label: str,
    candidate_label: str,
    mine_mode: str,
) -> dict[str, object]:
    by_reason: Counter[str] = Counter()
    by_variant: Counter[str] = Counter()
    by_source_split: Counter[str] = Counter()
    totals = {
        "match_delta_active_minus_candidate": 0,
        "correct_delta_active_minus_candidate": 0,
        "wrong_delta_candidate_minus_active": 0,
        "match_delta_candidate_minus_active": 0,
        "correct_delta_candidate_minus_active": 0,
        "wrong_delta_active_minus_candidate": 0,
        "match_delta": 0,
        "correct_delta": 0,
        "wrong_delta": 0,
    }
    for row in rows:
        for reason in row.get("hard_reasons", "").split("|"):
            if reason:
                by_reason[reason] += 1
        by_variant[row.get("target_variant", "")] += 1
        by_source_split[f"{row.get('source_name', '')}/{row.get('split', '')}"] += 1
        totals["match_delta_active_minus_candidate"] += _int_value(row, "match_delta_active_minus_candidate")
        totals["correct_delta_active_minus_candidate"] += _int_value(row, "correct_delta_active_minus_candidate")
        totals["wrong_delta_candidate_minus_active"] += _int_value(row, "wrong_delta_candidate_minus_active")
        totals["match_delta_candidate_minus_active"] += _int_value(row, "match_delta_candidate_minus_active")
        totals["correct_delta_candidate_minus_active"] += _int_value(row, "correct_delta_candidate_minus_active")
        totals["wrong_delta_active_minus_candidate"] += _int_value(row, "wrong_delta_active_minus_candidate")
        totals["match_delta"] += _int_value(row, "match_delta")
        totals["correct_delta"] += _int_value(row, "correct_delta")
        totals["wrong_delta"] += _int_value(row, "wrong_delta")
    return {
        "active_label": active_label,
        "candidate_label": candidate_label,
        "mine_mode": mine_mode,
        "rows": len(rows),
        "by_reason": dict(sorted(by_reason.items())),
        "by_variant": dict(sorted(by_variant.items())),
        "by_source_split": dict(sorted(by_source_split.items())),
        "totals": totals,
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*PAIR_MANIFEST_FIELDS, *EXTRA_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report_html(
    path: Path,
    *,
    output_manifest: Path,
    summary: dict[str, object],
    pair_sources: list[PairManifestSource],
) -> None:
    payload = {
        "output_manifest": str(output_manifest),
        "summary": summary,
        "pair_sources": [
            {"source": source.source_name, "split": source.split, "path": str(source.path)}
            for source in pair_sources
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-CN">',
                '<head><meta charset="utf-8"><title>Selector disagreement hard pairs</title></head>',
                "<body>",
                "<h1>Selector disagreement hard pairs</h1>",
                "<p>该报告用于挖 phase5d/phase5e 等 selector 分歧样本；correct/wrong 只用于离线 hard mining，不用于推理 selector 决策。</p>",
                f"<pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )


def parse_pair_manifest_source(value: str) -> PairManifestSource:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--pair-manifest-source must be source,split,path")
    source_name, split, path = parts
    return PairManifestSource(source_name=source_name, split=split, path=Path(path))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--pair-manifest-source", type=parse_pair_manifest_source, action="append", required=True)
    parser.add_argument("--active-label", default="active_selector")
    parser.add_argument("--candidate-label", default="candidate_selector")
    parser.add_argument("--target-variants", default="extreme_02,extreme_03")
    parser.add_argument("--include-non-target-regressions", action="store_true")
    parser.add_argument("--min-precision-drop", type=float, default=0.001)
    parser.add_argument(
        "--mine-mode",
        choices=("active_regressions", "candidate_gains"),
        default="active_regressions",
        help=(
            "active_regressions mines candidate failures against the active selector; "
            "candidate_gains mines clean candidate improvements against the active selector"
        ),
    )
    parser.add_argument("--min-candidate-correct-gain", type=int, default=1)
    parser.add_argument("--max-candidate-wrong-increase", type=int, default=0)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target_variants = tuple(item.strip() for item in args.target_variants.split(",") if item.strip())
    config = DisagreementConfig(
        target_variants=target_variants,
        include_non_target_regressions=bool(args.include_non_target_regressions),
        min_precision_drop=float(args.min_precision_drop),
        mine_mode=str(args.mine_mode),
        min_candidate_correct_gain=int(args.min_candidate_correct_gain),
        max_candidate_wrong_increase=int(args.max_candidate_wrong_increase),
    )
    rows = mine_selector_disagreement_rows(
        active_rows=_read_csv_rows(args.active_summary),
        candidate_rows=_read_csv_rows(args.candidate_summary),
        pair_sources=args.pair_manifest_source,
        active_label=str(args.active_label),
        candidate_label=str(args.candidate_label),
        config=config,
    )
    summary = summarize_rows(
        rows,
        active_label=str(args.active_label),
        candidate_label=str(args.candidate_label),
        mine_mode=str(args.mine_mode),
    )
    write_manifest(args.output_manifest, rows)
    write_summary_json(args.summary_json, summary)
    write_report_html(
        args.output_html,
        output_manifest=args.output_manifest,
        summary=summary,
        pair_sources=args.pair_manifest_source,
    )
    print(f"selector_disagreement_rows={len(rows)} output={args.output_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
