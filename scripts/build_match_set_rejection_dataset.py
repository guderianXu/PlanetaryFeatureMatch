#!/usr/bin/env python3
"""Build pair-level match-set rejection calibration datasets."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


OUTPUT_FIELDS = [
    "source_name",
    "split",
    "pair_index",
    "pair_type",
    "base_id",
    "reference_variant",
    "target_variant",
    "pfm_matches",
    "pfm_correct",
    "pfm_wrong",
    "pfm_precision",
    "lightglue_matches",
    "lightglue_correct",
    "lightglue_wrong",
    "lightglue_precision",
    "teacher_match_delta",
    "teacher_correct_delta",
    "teacher_wrong_delta",
    "teacher_precision_delta",
    "reject_label",
    "reject_reasons",
    "keep_label",
    "feature_valid_fraction",
    "feature_matches",
    "feature_score_min",
    "feature_score_mean",
    "feature_score_median",
    "feature_score_max",
    "feature_bbox_area_min_px2",
    "feature_bbox_area_ratio",
    "feature_displacement_median_px",
    "feature_displacement_mad_px",
    "feature_homography_residual_valid",
    "feature_homography_residual_median_px",
    "feature_homography_residual_p90_px",
    "feature_target_is_extreme",
    "feature_target_is_extreme_01",
    "feature_target_is_extreme_02",
    "feature_target_is_extreme_03",
    "feature_detail_count",
    "feature_detail_score_min",
    "feature_detail_score_mean",
    "feature_detail_score_median",
    "feature_detail_score_max",
    "feature_detail_raw_margin_min",
    "feature_detail_raw_margin_mean",
    "feature_detail_raw_margin_median",
    "feature_detail_accept_probability_min",
    "feature_detail_accept_probability_mean",
    "feature_detail_accept_probability_median",
    "feature_detail_low_accept_fraction",
    "feature_detail_low_raw_margin_fraction",
    "feature_detail_positive_vs_dustbin_margin_min",
    "feature_detail_positive_vs_dustbin_margin_mean",
    "feature_detail_positive_vs_dustbin_margin_median",
    "feature_detail_pair_logit_mean",
    "feature_detail_accept_logit_mean",
    "feature_detail_displacement_dx_median_px",
    "feature_detail_displacement_dy_median_px",
    "feature_detail_displacement_dx_mad_px",
    "feature_detail_displacement_dy_mad_px",
]


@dataclass(frozen=True)
class RejectionLabelConfig:
    reject_wrong_threshold: int = 3
    reject_precision_threshold: float = 0.995
    teacher_wrong_excess_threshold: int = 2
    teacher_precision_advantage_threshold: float = 0.005
    keep_max_wrong: int = 1
    keep_min_precision: float = 0.995


@dataclass(frozen=True)
class RejectionDatasetSource:
    split: str
    pair_manifest: Path
    pfm_summary: Path
    lightglue_metrics: Path | None = None
    match_details: Path | None = None


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


def _has_kept_metrics(row: dict[str, str]) -> bool:
    return all(row.get(key, "") != "" for key in ("kept_matches", "kept_correct", "kept_wrong"))


def _pfm_metric(row: dict[str, str], key: str) -> int:
    if _has_kept_metrics(row):
        return _int_value(row, f"kept_{key}")
    return _int_value(row, key)


def _pfm_precision(row: dict[str, str], matches: int, correct: int) -> float:
    if _has_kept_metrics(row) and row.get("kept_precision", "") != "":
        return _float_value(row, "kept_precision", correct / matches if matches else 0.0)
    return _float_value(row, "precision", correct / matches if matches else 0.0)


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _fmt_int(value: int) -> str:
    return str(int(value))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _median(values)
    return _median([abs(value - center) for value in values])


def _pair_index(row: dict[str, str], ordinal: int) -> str:
    value = row.get("pair_index", "")
    return str(ordinal) if value == "" else value


def _base_id(pair: dict[str, str], summary: dict[str, str]) -> str:
    return (
        pair.get("reference_base_id")
        or pair.get("target_base_id")
        or summary.get("base_id")
        or ""
    )


def _base_id_candidates(pair: dict[str, str], summary: dict[str, str]) -> set[str]:
    return {
        value
        for value in (
            summary.get("base_id", ""),
            pair.get("reference_base_id", ""),
            pair.get("target_base_id", ""),
        )
        if value
    }


def _target_variant(pair: dict[str, str], summary: dict[str, str]) -> str:
    return pair.get("target_variant") or summary.get("target_variant") or ""


def _reference_variant(pair: dict[str, str]) -> str:
    return pair.get("reference_variant") or ""


def _teacher_key(row: dict[str, str]) -> str:
    return row.get("manifest_pair_index") or row.get("pair_index") or ""


def _teacher_rows_by_pair_index(
    lightglue_rows: Sequence[dict[str, str]],
    *,
    teacher_label: str,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in lightglue_rows:
        if teacher_label and row.get("label", "") != teacher_label:
            continue
        key = _teacher_key(row)
        if key:
            indexed[key] = row
    return indexed


def _teacher_matches_pair(
    teacher: dict[str, str] | None,
    pair: dict[str, str],
    pfm: dict[str, str],
) -> bool:
    if teacher is None:
        return False
    expected_base_id = _base_id(pair, pfm)
    expected_base_ids = _base_id_candidates(pair, pfm)
    expected_variant = _target_variant(pair, pfm)
    expected_split = pair.get("split") or pfm.get("split") or ""
    teacher_base_id = teacher.get("base_id") or teacher.get("reference_base_id") or ""
    teacher_variant = teacher.get("target_variant") or ""
    teacher_split = teacher.get("split") or ""
    if expected_base_id and teacher_base_id and teacher_base_id not in expected_base_ids:
        return False
    if expected_variant and teacher_variant and expected_variant != teacher_variant:
        return False
    if expected_split and teacher_split and expected_split != teacher_split:
        return False
    return True


def _select_teacher_row(
    teacher_by_pair_index: dict[str, dict[str, str]],
    *,
    pair: dict[str, str],
    pfm: dict[str, str],
    pair_index: str,
    ordinal: int,
) -> dict[str, str] | None:
    direct = teacher_by_pair_index.get(pair_index)
    if _teacher_matches_pair(direct, pair, pfm):
        return direct
    ordinal_row = teacher_by_pair_index.get(str(ordinal))
    if _teacher_matches_pair(ordinal_row, pair, pfm):
        return ordinal_row
    if direct is not None and ordinal_row is None:
        return direct
    return None


def _detail_rows_by_pair_index(
    detail_rows: Sequence[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    indexed: dict[str, list[dict[str, str]]] = {}
    for row in detail_rows:
        key = row.get("pair_index", "")
        if key:
            indexed.setdefault(key, []).append(row)
    return indexed


def _detail_rows_match_pair(
    details: Sequence[dict[str, str]] | None,
    pair: dict[str, str],
    pfm: dict[str, str],
) -> bool:
    if not details:
        return False
    first = details[0]
    expected_base_id = _base_id(pair, pfm)
    expected_variant = _target_variant(pair, pfm)
    expected_split = pair.get("split") or pfm.get("split") or ""
    detail_base_id = first.get("base_id") or ""
    detail_variant = first.get("target_variant") or ""
    detail_split = first.get("split") or ""
    if expected_base_id and detail_base_id and expected_base_id != detail_base_id:
        return False
    if expected_variant and detail_variant and expected_variant != detail_variant:
        return False
    if expected_split and detail_split and expected_split != detail_split:
        return False
    return True


def _select_detail_rows(
    details_by_pair_index: dict[str, list[dict[str, str]]],
    *,
    pair: dict[str, str],
    pfm: dict[str, str],
    pair_index: str,
    ordinal: int,
) -> list[dict[str, str]]:
    direct = details_by_pair_index.get(pair_index, [])
    if _detail_rows_match_pair(direct, pair, pfm):
        return direct
    ordinal_rows = details_by_pair_index.get(str(ordinal), [])
    if _detail_rows_match_pair(ordinal_rows, pair, pfm):
        return ordinal_rows
    if direct and not ordinal_rows:
        return direct
    return []


def _match_detail_features(detail_rows: Sequence[dict[str, str]]) -> dict[str, str]:
    def values(name: str) -> list[float]:
        return [_float_value(row, name) for row in detail_rows]

    count = len(detail_rows)
    if count == 0:
        return {
            "feature_detail_count": "0",
            "feature_detail_score_min": _fmt_float(0.0),
            "feature_detail_score_mean": _fmt_float(0.0),
            "feature_detail_score_median": _fmt_float(0.0),
            "feature_detail_score_max": _fmt_float(0.0),
            "feature_detail_raw_margin_min": _fmt_float(0.0),
            "feature_detail_raw_margin_mean": _fmt_float(0.0),
            "feature_detail_raw_margin_median": _fmt_float(0.0),
            "feature_detail_accept_probability_min": _fmt_float(0.0),
            "feature_detail_accept_probability_mean": _fmt_float(0.0),
            "feature_detail_accept_probability_median": _fmt_float(0.0),
            "feature_detail_low_accept_fraction": _fmt_float(0.0),
            "feature_detail_low_raw_margin_fraction": _fmt_float(0.0),
            "feature_detail_positive_vs_dustbin_margin_min": _fmt_float(0.0),
            "feature_detail_positive_vs_dustbin_margin_mean": _fmt_float(0.0),
            "feature_detail_positive_vs_dustbin_margin_median": _fmt_float(0.0),
            "feature_detail_pair_logit_mean": _fmt_float(0.0),
            "feature_detail_accept_logit_mean": _fmt_float(0.0),
            "feature_detail_displacement_dx_median_px": _fmt_float(0.0),
            "feature_detail_displacement_dy_median_px": _fmt_float(0.0),
            "feature_detail_displacement_dx_mad_px": _fmt_float(0.0),
            "feature_detail_displacement_dy_mad_px": _fmt_float(0.0),
        }

    scores = values("score")
    raw_margins = values("raw_margin")
    accept_probabilities = values("accept_probability")
    dustbin_margins = values("positive_vs_dustbin_margin")
    pair_logits = values("pair_logit")
    accept_logits = values("accept_logit")
    dx_values = [
        _float_value(row, "point_b_x_px") - _float_value(row, "point_a_x_px")
        for row in detail_rows
    ]
    dy_values = [
        _float_value(row, "point_b_y_px") - _float_value(row, "point_a_y_px")
        for row in detail_rows
    ]
    return {
        "feature_detail_count": _fmt_int(count),
        "feature_detail_score_min": _fmt_float(min(scores)),
        "feature_detail_score_mean": _fmt_float(_mean(scores)),
        "feature_detail_score_median": _fmt_float(_median(scores)),
        "feature_detail_score_max": _fmt_float(max(scores)),
        "feature_detail_raw_margin_min": _fmt_float(min(raw_margins)),
        "feature_detail_raw_margin_mean": _fmt_float(_mean(raw_margins)),
        "feature_detail_raw_margin_median": _fmt_float(_median(raw_margins)),
        "feature_detail_accept_probability_min": _fmt_float(min(accept_probabilities)),
        "feature_detail_accept_probability_mean": _fmt_float(_mean(accept_probabilities)),
        "feature_detail_accept_probability_median": _fmt_float(_median(accept_probabilities)),
        "feature_detail_low_accept_fraction": _fmt_float(
            sum(1 for value in accept_probabilities if value < 0.5) / count
        ),
        "feature_detail_low_raw_margin_fraction": _fmt_float(
            sum(1 for value in raw_margins if value <= 0.05) / count
        ),
        "feature_detail_positive_vs_dustbin_margin_min": _fmt_float(min(dustbin_margins)),
        "feature_detail_positive_vs_dustbin_margin_mean": _fmt_float(_mean(dustbin_margins)),
        "feature_detail_positive_vs_dustbin_margin_median": _fmt_float(_median(dustbin_margins)),
        "feature_detail_pair_logit_mean": _fmt_float(_mean(pair_logits)),
        "feature_detail_accept_logit_mean": _fmt_float(_mean(accept_logits)),
        "feature_detail_displacement_dx_median_px": _fmt_float(_median(dx_values)),
        "feature_detail_displacement_dy_median_px": _fmt_float(_median(dy_values)),
        "feature_detail_displacement_dx_mad_px": _fmt_float(_mad(dx_values)),
        "feature_detail_displacement_dy_mad_px": _fmt_float(_mad(dy_values)),
    }


def _bbox_ratio(area_a: float, area_b: float) -> tuple[float, float]:
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0, 0.0
    smaller = min(area_a, area_b)
    larger = max(area_a, area_b)
    return smaller, smaller / larger if larger > 0.0 else 0.0


def _reject_reasons(
    *,
    pfm_wrong: int,
    pfm_precision: float,
    teacher_wrong_delta: int,
    teacher_precision_delta: float,
    has_teacher: bool,
    config: RejectionLabelConfig,
) -> list[str]:
    reasons: list[str] = []
    if pfm_wrong >= config.reject_wrong_threshold:
        reasons.append("pfm_wrong")
    if pfm_precision < config.reject_precision_threshold:
        reasons.append("pfm_low_precision")
    if has_teacher and teacher_wrong_delta >= config.teacher_wrong_excess_threshold:
        reasons.append("teacher_wrong_excess")
    if (
        has_teacher
        and -teacher_precision_delta >= config.teacher_precision_advantage_threshold
    ):
        reasons.append("teacher_precision_advantage")
    return reasons


def build_rejection_rows(
    pair_rows: Sequence[dict[str, str]],
    pfm_rows: Sequence[dict[str, str]],
    lightglue_rows: Sequence[dict[str, str]] = (),
    match_detail_rows: Sequence[dict[str, str]] = (),
    *,
    split: str,
    source_name: str = "",
    teacher_label: str = "LightGlue-SIFT-MAGSAC-min16",
    config: RejectionLabelConfig = RejectionLabelConfig(),
) -> list[dict[str, str]]:
    if len(pfm_rows) > len(pair_rows):
        raise ValueError("pair manifest must contain at least as many rows as the PFM summary")

    teacher_by_pair_index = _teacher_rows_by_pair_index(lightglue_rows, teacher_label=teacher_label)
    details_by_pair_index = _detail_rows_by_pair_index(match_detail_rows)
    rows: list[dict[str, str]] = []
    for ordinal, (pair, pfm) in enumerate(zip(pair_rows, pfm_rows)):
        pair_index = _pair_index(pair, ordinal)
        teacher = _select_teacher_row(
            teacher_by_pair_index,
            pair=pair,
            pfm=pfm,
            pair_index=pair_index,
            ordinal=ordinal,
        )
        has_teacher = teacher is not None

        pfm_matches = _pfm_metric(pfm, "matches")
        pfm_correct = _pfm_metric(pfm, "correct")
        pfm_wrong = _pfm_metric(pfm, "wrong")
        pfm_precision = _pfm_precision(pfm, pfm_matches, pfm_correct)

        lightglue_matches = _int_value(teacher, "matches") if teacher is not None else 0
        lightglue_correct = _int_value(teacher, "correct") if teacher is not None else 0
        lightglue_wrong = _int_value(teacher, "wrong") if teacher is not None else 0
        lightglue_precision = (
            _float_value(teacher, "precision", lightglue_correct / lightglue_matches if lightglue_matches else 0.0)
            if teacher is not None
            else 0.0
        )

        teacher_match_delta = pfm_matches - lightglue_matches if has_teacher else 0
        teacher_correct_delta = pfm_correct - lightglue_correct if has_teacher else 0
        teacher_wrong_delta = pfm_wrong - lightglue_wrong if has_teacher else 0
        teacher_precision_delta = pfm_precision - lightglue_precision if has_teacher else 0.0
        reasons = _reject_reasons(
            pfm_wrong=pfm_wrong,
            pfm_precision=pfm_precision,
            teacher_wrong_delta=teacher_wrong_delta,
            teacher_precision_delta=teacher_precision_delta,
            has_teacher=has_teacher,
            config=config,
        )
        keep_label = (
            not reasons
            and pfm_wrong <= config.keep_max_wrong
            and pfm_precision >= config.keep_min_precision
        )
        area_min, area_ratio = _bbox_ratio(
            _float_value(pfm, "bbox_area_a_px2"),
            _float_value(pfm, "bbox_area_b_px2"),
        )
        target_variant = _target_variant(pair, pfm)
        base_id = _base_id(pair, pfm)
        detail_features = _match_detail_features(
            _select_detail_rows(
                details_by_pair_index,
                pair=pair,
                pfm=pfm,
                pair_index=pair_index,
                ordinal=ordinal,
            )
        )
        row = {
            "source_name": source_name,
            "split": split or pair.get("split", "") or pfm.get("split", ""),
            "pair_index": pair_index,
            "pair_type": pair.get("pair_type", ""),
            "base_id": base_id,
            "reference_variant": _reference_variant(pair),
            "target_variant": target_variant,
            "pfm_matches": _fmt_int(pfm_matches),
            "pfm_correct": _fmt_int(pfm_correct),
            "pfm_wrong": _fmt_int(pfm_wrong),
            "pfm_precision": _fmt_float(pfm_precision),
            "lightglue_matches": _fmt_int(lightglue_matches),
            "lightglue_correct": _fmt_int(lightglue_correct),
            "lightglue_wrong": _fmt_int(lightglue_wrong),
            "lightglue_precision": _fmt_float(lightglue_precision),
            "teacher_match_delta": _fmt_int(teacher_match_delta),
            "teacher_correct_delta": _fmt_int(teacher_correct_delta),
            "teacher_wrong_delta": _fmt_int(teacher_wrong_delta),
            "teacher_precision_delta": _fmt_float(teacher_precision_delta),
            "reject_label": "1" if reasons else "0",
            "reject_reasons": "|".join(reasons),
            "keep_label": "1" if keep_label else "0",
            "feature_valid_fraction": _fmt_float(_float_value(pfm, "valid_fraction", _float_value(pair, "valid_fraction"))),
            "feature_matches": _fmt_int(pfm_matches),
            "feature_score_min": _fmt_float(_float_value(pfm, "score_min")),
            "feature_score_mean": _fmt_float(_float_value(pfm, "score_mean")),
            "feature_score_median": _fmt_float(_float_value(pfm, "score_median")),
            "feature_score_max": _fmt_float(_float_value(pfm, "score_max")),
            "feature_bbox_area_min_px2": _fmt_float(area_min),
            "feature_bbox_area_ratio": _fmt_float(area_ratio),
            "feature_displacement_median_px": _fmt_float(_float_value(pfm, "displacement_median_px")),
            "feature_displacement_mad_px": _fmt_float(_float_value(pfm, "displacement_mad_px")),
            "feature_homography_residual_valid": _fmt_int(_int_value(pfm, "homography_residual_valid")),
            "feature_homography_residual_median_px": _fmt_float(_float_value(pfm, "homography_residual_median_px")),
            "feature_homography_residual_p90_px": _fmt_float(_float_value(pfm, "homography_residual_p90_px")),
            "feature_target_is_extreme": "1" if target_variant.startswith("extreme") else "0",
            "feature_target_is_extreme_01": "1" if target_variant == "extreme_01" else "0",
            "feature_target_is_extreme_02": "1" if target_variant == "extreme_02" else "0",
            "feature_target_is_extreme_03": "1" if target_variant == "extreme_03" else "0",
        }
        row.update(detail_features)
        rows.append(row)
    return rows


def build_from_sources(
    sources: Sequence[RejectionDatasetSource],
    *,
    source_name: str,
    teacher_label: str,
    config: RejectionLabelConfig,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in sources:
        lightglue_rows = _read_csv_rows(source.lightglue_metrics) if source.lightglue_metrics is not None else []
        match_detail_rows = _read_csv_rows(source.match_details) if source.match_details is not None else []
        rows.extend(
            build_rejection_rows(
                _read_csv_rows(source.pair_manifest),
                _read_csv_rows(source.pfm_summary),
                lightglue_rows,
                match_detail_rows,
                split=source.split,
                source_name=source_name or source.split,
                teacher_label=teacher_label,
                config=config,
            )
        )
    return rows


def write_dataset_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    def int_field(row: dict[str, str], key: str) -> int:
        return _int_value(row, key)

    summary: dict[str, object] = {
        "rows": len(rows),
        "reject_rows": sum(1 for row in rows if row.get("reject_label") == "1"),
        "keep_rows": sum(1 for row in rows if row.get("keep_label") == "1"),
        "pfm_matches": sum(int_field(row, "pfm_matches") for row in rows),
        "pfm_correct": sum(int_field(row, "pfm_correct") for row in rows),
        "pfm_wrong": sum(int_field(row, "pfm_wrong") for row in rows),
        "lightglue_matches": sum(int_field(row, "lightglue_matches") for row in rows),
        "lightglue_correct": sum(int_field(row, "lightglue_correct") for row in rows),
        "lightglue_wrong": sum(int_field(row, "lightglue_wrong") for row in rows),
        "reject_reasons": {},
        "by_split": {},
        "by_variant": {},
    }
    reason_counter: Counter[str] = Counter()
    for row in rows:
        for reason in row.get("reject_reasons", "").split("|"):
            if reason:
                reason_counter[reason] += 1
    summary["reject_reasons"] = dict(sorted(reason_counter.items()))

    for key, bucket_name in (("split", "by_split"), ("target_variant", "by_variant")):
        bucket = summary[bucket_name]
        assert isinstance(bucket, dict)
        for row in rows:
            name = row.get(key, "")
            item = bucket.setdefault(
                name,
                {
                    "rows": 0,
                    "reject_rows": 0,
                    "keep_rows": 0,
                    "pfm_matches": 0,
                    "pfm_correct": 0,
                    "pfm_wrong": 0,
                    "lightglue_wrong": 0,
                },
            )
            item["rows"] += 1
            item["reject_rows"] += 1 if row.get("reject_label") == "1" else 0
            item["keep_rows"] += 1 if row.get("keep_label") == "1" else 0
            item["pfm_matches"] += int_field(row, "pfm_matches")
            item["pfm_correct"] += int_field(row, "pfm_correct")
            item["pfm_wrong"] += int_field(row, "pfm_wrong")
            item["lightglue_wrong"] += int_field(row, "lightglue_wrong")
    return summary


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _render_bucket_table(title: str, bucket: dict[str, object]) -> str:
    rows = []
    for name, payload in sorted(bucket.items()):
        assert isinstance(payload, dict)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{payload.get('rows', 0)}</td>"
            f"<td>{payload.get('reject_rows', 0)}</td>"
            f"<td>{payload.get('keep_rows', 0)}</td>"
            f"<td>{payload.get('pfm_correct', 0)}</td>"
            f"<td>{payload.get('pfm_wrong', 0)}</td>"
            f"<td>{payload.get('lightglue_wrong', 0)}</td>"
            "</tr>"
        )
    return (
        f"<h2>{html.escape(title)}</h2>"
        '<table border="1" cellspacing="0" cellpadding="4">'
        "<tr><th>name</th><th>rows</th><th>reject</th><th>keep</th>"
        "<th>pfm_correct</th><th>pfm_wrong</th><th>lightglue_wrong</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def write_report_html(path: Path, *, output_csv: Path, summary: dict[str, object], sources: Sequence[RejectionDatasetSource]) -> None:
    source_payload = [
        {
            "split": source.split,
            "pair_manifest": str(source.pair_manifest),
            "pfm_summary": str(source.pfm_summary),
            "lightglue_metrics": "" if source.lightglue_metrics is None else str(source.lightglue_metrics),
            "match_details": "" if source.match_details is None else str(source.match_details),
        }
        for source in sources
    ]
    by_split = summary.get("by_split", {})
    by_variant = summary.get("by_variant", {})
    assert isinstance(by_split, dict)
    assert isinstance(by_variant, dict)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Match-set rejection dataset</title>",
                "<h1>Match-set rejection dataset</h1>",
                f"<p>output_csv={html.escape(str(output_csv))}</p>",
                "<h2>Overall</h2>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
                _render_bucket_table("By split", by_split),
                _render_bucket_table("By target variant", by_variant),
                "<h2>Sources</h2>",
                f"<pre>{html.escape(json.dumps(source_payload, ensure_ascii=False, indent=2))}</pre>",
            ]
        ),
        encoding="utf-8",
    )


def parse_source(value: str) -> RejectionDatasetSource:
    parts = value.split(",", 4)
    if len(parts) not in {3, 4, 5}:
        raise argparse.ArgumentTypeError(
            "--source must be split,pair_manifest,pfm_summary[,lightglue_metrics[,match_details]]"
        )
    split, pair_manifest, pfm_summary = parts[:3]
    lightglue_metrics = Path(parts[3]) if len(parts) == 4 and parts[3] else None
    if len(parts) == 5:
        lightglue_metrics = Path(parts[3]) if parts[3] else None
    match_details = Path(parts[4]) if len(parts) == 5 and parts[4] else None
    return RejectionDatasetSource(
        split=split,
        pair_manifest=Path(pair_manifest),
        pfm_summary=Path(pfm_summary),
        lightglue_metrics=lightglue_metrics,
        match_details=match_details,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=parse_source, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--source-name", default="")
    parser.add_argument("--teacher-label", default="LightGlue-SIFT-MAGSAC-min16")
    parser.add_argument("--reject-wrong-threshold", type=int, default=3)
    parser.add_argument("--reject-precision-threshold", type=float, default=0.995)
    parser.add_argument("--teacher-wrong-excess-threshold", type=int, default=2)
    parser.add_argument("--teacher-precision-advantage-threshold", type=float, default=0.005)
    parser.add_argument("--keep-max-wrong", type=int, default=1)
    parser.add_argument("--keep-min-precision", type=float, default=0.995)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RejectionLabelConfig:
    if args.reject_wrong_threshold < 0:
        raise ValueError("--reject-wrong-threshold must be nonnegative")
    if args.teacher_wrong_excess_threshold < 0:
        raise ValueError("--teacher-wrong-excess-threshold must be nonnegative")
    if args.keep_max_wrong < 0:
        raise ValueError("--keep-max-wrong must be nonnegative")
    return RejectionLabelConfig(
        reject_wrong_threshold=args.reject_wrong_threshold,
        reject_precision_threshold=args.reject_precision_threshold,
        teacher_wrong_excess_threshold=args.teacher_wrong_excess_threshold,
        teacher_precision_advantage_threshold=args.teacher_precision_advantage_threshold,
        keep_max_wrong=args.keep_max_wrong,
        keep_min_precision=args.keep_min_precision,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    rows = build_from_sources(
        args.source,
        source_name=str(args.source_name),
        teacher_label=str(args.teacher_label),
        config=config,
    )
    summary = summarize_rows(rows)
    write_dataset_csv(args.output_csv, rows)
    write_summary_json(args.summary_json, summary)
    write_report_html(args.output_html, output_csv=args.output_csv, summary=summary, sources=args.source)
    print(f"rejection_rows={len(rows)} reject_rows={summary['reject_rows']} output={args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
