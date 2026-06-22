#!/usr/bin/env python3
"""Build reusable lazy false-match labels from raw visual match details."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


OUTPUT_FIELDS = [
    "pair_pt",
    "lazy_pair_key",
    "reference_dataset_id",
    "reference_pose_id",
    "reference_raw_base_id",
    "reference_base_id",
    "reference_variant",
    "target_dataset_id",
    "target_pose_id",
    "target_raw_base_id",
    "target_base_id",
    "target_variant",
    "pair_type",
    "crop_a_x0",
    "crop_a_y0",
    "crop_a_x1",
    "crop_a_y1",
    "crop_b_x0",
    "crop_b_y0",
    "crop_b_x1",
    "crop_b_y1",
    "ax",
    "ay",
    "bx",
    "by",
    "error_px",
    "score",
    "raw_similarity",
    "raw_margin",
    "accept_probability",
    "matcher",
    "geometry_rejected",
    "mine_source",
    "source_split",
    "source_pair_index",
    "source_match_index",
    "source_label",
    "source_correct",
    "source_valid_fraction",
]


@dataclass(frozen=True)
class PairMetadata:
    pair_index: str
    split: str
    pair_type: str
    reference_dataset_id: str
    reference_pose_id: str
    reference_raw_base_id: str
    reference_base_id: str
    reference_variant: str
    target_dataset_id: str
    target_pose_id: str
    target_raw_base_id: str
    target_base_id: str
    target_variant: str
    crop_a_x0: str
    crop_a_y0: str
    crop_a_x1: str
    crop_a_y1: str
    crop_b_x0: str
    crop_b_y0: str
    crop_b_x1: str
    crop_b_y1: str


@dataclass(frozen=True)
class FalseMatchCandidate:
    pair_key: tuple[str, str]
    score: float
    error_px: float
    row: dict[str, str]


def _field(row: dict[str, str], name: str, default: str = "") -> str:
    return (row.get(name) or default).strip()


def _float_or_none(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _false_match_key_part(value: object) -> str:
    return str(value).replace("|", "%7C").replace("\n", " ").strip()


def _crop_token(meta: PairMetadata, prefix: str) -> str:
    values = [
        getattr(meta, f"{prefix}_x0"),
        getattr(meta, f"{prefix}_y0"),
        getattr(meta, f"{prefix}_x1"),
        getattr(meta, f"{prefix}_y1"),
    ]
    if any(value == "" for value in values):
        return "none"
    return ",".join(values)


def lazy_pair_key(meta: PairMetadata) -> str:
    parts = [
        "lazy_pair_false_v1",
        meta.pair_type,
        meta.reference_dataset_id,
        meta.reference_pose_id,
        meta.reference_raw_base_id,
        meta.reference_variant,
        meta.target_dataset_id,
        meta.target_pose_id,
        meta.target_raw_base_id,
        meta.target_variant,
        _crop_token(meta, "crop_a"),
        _crop_token(meta, "crop_b"),
    ]
    return "|".join(_false_match_key_part(part) for part in parts)


def _pair_metadata_from_row(row: dict[str, str]) -> PairMetadata:
    return PairMetadata(
        pair_index=_field(row, "pair_index"),
        split=_field(row, "split", "train"),
        pair_type=_field(row, "pair_type", "same_position_view"),
        reference_dataset_id=_field(row, "reference_dataset_id"),
        reference_pose_id=_field(row, "reference_pose_id"),
        reference_raw_base_id=_field(row, "reference_raw_base_id", _field(row, "reference_base_id")),
        reference_base_id=_field(row, "reference_base_id"),
        reference_variant=_field(row, "reference_variant"),
        target_dataset_id=_field(row, "target_dataset_id"),
        target_pose_id=_field(row, "target_pose_id"),
        target_raw_base_id=_field(row, "target_raw_base_id", _field(row, "target_base_id")),
        target_base_id=_field(row, "target_base_id"),
        target_variant=_field(row, "target_variant"),
        crop_a_x0=_field(row, "crop_a_x0"),
        crop_a_y0=_field(row, "crop_a_y0"),
        crop_a_x1=_field(row, "crop_a_x1"),
        crop_a_y1=_field(row, "crop_a_y1"),
        crop_b_x0=_field(row, "crop_b_x0"),
        crop_b_y0=_field(row, "crop_b_y0"),
        crop_b_x1=_field(row, "crop_b_x1"),
        crop_b_y1=_field(row, "crop_b_y1"),
    )


def read_pair_metadata(paths: Iterable[Path]) -> dict[tuple[str, str], PairMetadata]:
    metadata: dict[tuple[str, str], PairMetadata] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                meta = _pair_metadata_from_row(row)
                key = (meta.split, meta.pair_index)
                if key in metadata:
                    raise ValueError(f"duplicate pair metadata for split={meta.split!r} pair_index={meta.pair_index!r}")
                metadata[key] = meta
    return metadata


def _passes_optional_threshold(value: str, threshold: float | None) -> bool:
    if threshold is None:
        return True
    parsed = _float_or_none(value)
    return parsed is not None and parsed >= threshold


def _is_wrong_match(row: dict[str, str]) -> bool:
    correct = _float_or_none(_field(row, "correct"))
    return correct is not None and correct <= 0.0


def _candidate_from_match_row(
    row: dict[str, str],
    meta: PairMetadata,
    *,
    min_error_px: float,
    max_error_px: float | None,
    target_variants: set[str] | None,
    min_score: float,
    min_raw_similarity: float | None,
    min_accept_probability: float | None,
    matcher: str,
    mine_source: str,
) -> FalseMatchCandidate | None:
    if not _is_wrong_match(row):
        return None
    if target_variants is not None and meta.target_variant not in target_variants:
        return None

    coordinates = [
        _float_or_none(_field(row, "point_a_x_px")),
        _float_or_none(_field(row, "point_a_y_px")),
        _float_or_none(_field(row, "point_b_x_px")),
        _float_or_none(_field(row, "point_b_y_px")),
    ]
    if any(value is None for value in coordinates):
        return None
    error_px = _float_or_none(_field(row, "error_px"))
    score = _float_or_none(_field(row, "score"))
    if error_px is None or score is None:
        return None
    if error_px < min_error_px or score < min_score:
        return None
    if max_error_px is not None and error_px > max_error_px:
        return None
    if not _passes_optional_threshold(_field(row, "raw_similarity"), min_raw_similarity):
        return None
    if not _passes_optional_threshold(_field(row, "accept_probability"), min_accept_probability):
        return None

    ax, ay, bx, by = [float(value) for value in coordinates if value is not None]
    output_row = {
        "pair_pt": "",
        "lazy_pair_key": lazy_pair_key(meta),
        "reference_dataset_id": meta.reference_dataset_id,
        "reference_pose_id": meta.reference_pose_id,
        "reference_raw_base_id": meta.reference_raw_base_id,
        "reference_base_id": meta.reference_base_id,
        "reference_variant": meta.reference_variant,
        "target_dataset_id": meta.target_dataset_id,
        "target_pose_id": meta.target_pose_id,
        "target_raw_base_id": meta.target_raw_base_id,
        "target_base_id": meta.target_base_id,
        "target_variant": meta.target_variant,
        "pair_type": meta.pair_type,
        "crop_a_x0": meta.crop_a_x0,
        "crop_a_y0": meta.crop_a_y0,
        "crop_a_x1": meta.crop_a_x1,
        "crop_a_y1": meta.crop_a_y1,
        "crop_b_x0": meta.crop_b_x0,
        "crop_b_y0": meta.crop_b_y0,
        "crop_b_x1": meta.crop_b_x1,
        "crop_b_y1": meta.crop_b_y1,
        "ax": _format_float(ax),
        "ay": _format_float(ay),
        "bx": _format_float(bx),
        "by": _format_float(by),
        "error_px": _format_float(error_px),
        "score": _format_float(score, digits=6),
        "raw_similarity": _field(row, "raw_similarity"),
        "raw_margin": _field(row, "raw_margin"),
        "accept_probability": _field(row, "accept_probability"),
        "matcher": matcher,
        "geometry_rejected": "0",
        "mine_source": mine_source,
        "source_split": _field(row, "split"),
        "source_pair_index": _field(row, "pair_index"),
        "source_match_index": _field(row, "match_index"),
        "source_label": _field(row, "label"),
        "source_correct": _field(row, "correct"),
        "source_valid_fraction": _field(row, "valid_fraction"),
    }
    return FalseMatchCandidate(pair_key=(meta.split, meta.pair_index), score=score, error_px=error_px, row=output_row)


def _insert_capped(
    grouped: dict[tuple[str, str], list[FalseMatchCandidate]],
    candidate: FalseMatchCandidate,
    *,
    max_per_pair: int,
) -> None:
    bucket = grouped.setdefault(candidate.pair_key, [])
    bucket.append(candidate)
    bucket.sort(key=lambda item: (-item.score, -item.error_px))
    if max_per_pair > 0:
        del bucket[max_per_pair:]


def _write_summary_html(path: Path, summary: dict[str, object]) -> None:
    variant_rows = []
    by_target_variant = summary.get("by_target_variant", {})
    if isinstance(by_target_variant, dict):
        for variant, count in sorted(by_target_variant.items()):
            variant_rows.append(
                f"<tr><td>{html.escape(str(variant))}</td><td><code>{html.escape(str(count))}</code></td></tr>"
            )
    variants = "\n".join(variant_rows)
    body = f"""<!doctype html><meta charset="utf-8">
<title>Lazy False Match CSV Summary</title>
<h1>Lazy False Match CSV Summary</h1>
<table>
<tr><th>metric</th><th>value</th></tr>
"""
    for key, value in summary.items():
        if key == "by_target_variant":
            continue
        body += f"<tr><td>{html.escape(str(key))}</td><td><code>{html.escape(str(value))}</code></td></tr>\n"
    body += "</table>\n<h2>By target variant</h2>\n<table><tr><th>variant</th><th>exported_rows</th></tr>\n"
    body += variants
    body += "\n</table>\n"
    path.write_text(body, encoding="utf-8")


def build_lazy_false_match_csv(
    *,
    pair_manifest_paths: list[Path],
    match_detail_paths: list[Path],
    output_csv: Path,
    min_error_px: float,
    max_error_px: float | None = None,
    target_variants: set[str] | None = None,
    min_score: float,
    min_raw_similarity: float | None,
    min_accept_probability: float | None,
    max_per_pair: int,
    matcher: str,
    mine_source: str,
    summary_json: Path | None = None,
    report_html: Path | None = None,
) -> dict[str, object]:
    if min_error_px < 0.0:
        raise ValueError("--min-error-px must be non-negative")
    if max_error_px is not None and max_error_px < min_error_px:
        raise ValueError("--max-error-px must be greater than or equal to --min-error-px")
    if max_per_pair < 0:
        raise ValueError("--max-per-pair must be non-negative; use 0 to keep all candidates")

    metadata = read_pair_metadata(pair_manifest_paths)
    grouped: dict[tuple[str, str], list[FalseMatchCandidate]] = {}
    detail_rows = 0
    wrong_rows = 0
    candidate_wrong_rows = 0
    skipped_missing_pair = 0
    skipped_invalid_or_threshold = 0

    for path in match_detail_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                detail_rows += 1
                if _is_wrong_match(row):
                    wrong_rows += 1
                key = (_field(row, "split"), _field(row, "pair_index"))
                meta = metadata.get(key)
                if meta is None:
                    skipped_missing_pair += 1
                    continue
                candidate = _candidate_from_match_row(
                    row,
                    meta,
                    min_error_px=min_error_px,
                    max_error_px=max_error_px,
                    target_variants=target_variants,
                    min_score=min_score,
                    min_raw_similarity=min_raw_similarity,
                    min_accept_probability=min_accept_probability,
                    matcher=matcher,
                    mine_source=mine_source,
                )
                if candidate is None:
                    if _is_wrong_match(row):
                        skipped_invalid_or_threshold += 1
                    continue
                candidate_wrong_rows += 1
                _insert_capped(grouped, candidate, max_per_pair=max_per_pair)

    output_rows = [
        candidate.row
        for pair_key in sorted(grouped)
        for candidate in sorted(grouped[pair_key], key=lambda item: (-item.score, -item.error_px))
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    by_target_variant: dict[str, int] = {}
    for row in output_rows:
        variant = row["target_variant"]
        by_target_variant[variant] = by_target_variant.get(variant, 0) + 1

    summary: dict[str, object] = {
        "pair_manifest_count": len(pair_manifest_paths),
        "match_detail_count": len(match_detail_paths),
        "manifest_pairs": len(metadata),
        "detail_rows": detail_rows,
        "wrong_rows": wrong_rows,
        "candidate_wrong_rows": candidate_wrong_rows,
        "exported_rows": len(output_rows),
        "exported_pairs": len(grouped),
        "skipped_missing_pair": skipped_missing_pair,
        "skipped_invalid_or_threshold": skipped_invalid_or_threshold,
        "min_error_px": min_error_px,
        "max_error_px": max_error_px,
        "target_variants": sorted(target_variants) if target_variants is not None else [],
        "min_score": min_score,
        "min_raw_similarity": min_raw_similarity,
        "min_accept_probability": min_accept_probability,
        "max_per_pair": max_per_pair,
        "matcher": matcher,
        "mine_source": mine_source,
        "output_csv": output_csv.as_posix(),
        "by_target_variant": by_target_variant,
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if report_html is not None:
        report_html.parent.mkdir(parents=True, exist_ok=True)
        _write_summary_html(report_html, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lazy false-match CSV labels from raw PFM match details")
    parser.add_argument("--pair-manifest", action="append", required=True, type=Path)
    parser.add_argument("--match-details", action="append", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--report-html", type=Path, default=None)
    parser.add_argument("--min-error-px", type=float, default=5.0)
    parser.add_argument("--max-error-px", type=float, default=None)
    parser.add_argument("--target-variant", action="append", default=[])
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-raw-similarity", type=float, default=None)
    parser.add_argument("--min-accept-probability", type=float, default=None)
    parser.add_argument("--max-per-pair", type=int, default=128)
    parser.add_argument("--matcher", default="graph_matcher")
    parser.add_argument("--mine-source", default="raw_true_geometry_wrong")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_variants = {item for item in args.target_variant if item}
    summary = build_lazy_false_match_csv(
        pair_manifest_paths=args.pair_manifest,
        match_detail_paths=args.match_details,
        output_csv=args.output_csv,
        min_error_px=args.min_error_px,
        max_error_px=args.max_error_px,
        target_variants=target_variants or None,
        min_score=args.min_score,
        min_raw_similarity=args.min_raw_similarity,
        min_accept_probability=args.min_accept_probability,
        max_per_pair=args.max_per_pair,
        matcher=args.matcher,
        mine_source=args.mine_source,
        summary_json=args.summary_json,
        report_html=args.report_html,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
