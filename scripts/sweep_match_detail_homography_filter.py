#!/usr/bin/env python3
"""Sweep homography residual filters over raw PFM match details."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


LIGHTGLUE_LABEL = "LightGlue-SIFT-MAGSAC-min16"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    split: str
    match_details: Path
    lightglue_metrics: Path


def float_value(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        parsed = float(text)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def int_value(row: dict[str, object], key: str, default: int = 0) -> int:
    return int(round(float_value(row, key, float(default))))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[dict[str, object]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for field in preferred_fields or []:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_source_spec(text: str) -> SourceSpec:
    parts = [part.strip() for part in text.split(",", 3)]
    if len(parts) != 4 or any(not part for part in parts):
        raise argparse.ArgumentTypeError("--source must be name,split,match_details,lightglue_metrics")
    name, split, match_details, lightglue_metrics = parts
    return SourceSpec(name=name, split=split, match_details=Path(match_details), lightglue_metrics=Path(lightglue_metrics))


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value) or value < 0.0:
            raise argparse.ArgumentTypeError(f"threshold must be finite and nonnegative: {item}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise argparse.ArgumentTypeError(f"min matches must be nonnegative: {item}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def parse_variant_list(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def select_lightglue_rows(rows: Sequence[dict[str, str]], *, lightglue_label: str) -> list[dict[str, str]]:
    if not rows:
        return []
    if "label" not in rows[0]:
        return list(rows)
    selected = [row for row in rows if row.get("label", "").strip() == lightglue_label]
    if not selected:
        raise ValueError(f"no LightGlue rows found for label={lightglue_label!r}")
    return selected


def detail_groups_by_pair(details: Sequence[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    for row in details:
        groups.setdefault(int_value(row, "pair_index"), []).append(row)
    return groups


def lightglue_pair_index(row: dict[str, str], fallback_index: int) -> int:
    if str(row.get("pair_index", "")).strip():
        return int_value(row, "pair_index")
    if str(row.get("manifest_pair_index", "")).strip():
        return int_value(row, "manifest_pair_index")
    return fallback_index


def homography_keep_mask(details: Sequence[dict[str, str]], threshold_px: float) -> list[bool]:
    if len(details) < 4:
        return [False] * len(details)
    points_a = np.array(
        [[float_value(row, "point_a_x_px"), float_value(row, "point_a_y_px")] for row in details],
        dtype=np.float32,
    )
    points_b = np.array(
        [[float_value(row, "point_b_x_px"), float_value(row, "point_b_y_px")] for row in details],
        dtype=np.float32,
    )
    homography, _mask = cv2.findHomography(
        points_a,
        points_b,
        cv2.USAC_MAGSAC,
        5.0,
        maxIters=10000,
        confidence=0.999,
    )
    if homography is None:
        return [False] * len(details)
    homogeneous = np.concatenate([points_a, np.ones((points_a.shape[0], 1), dtype=np.float32)], axis=1)
    projected = (homography @ homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    residual = np.linalg.norm(projected - points_b, axis=1)
    return [bool(math.isfinite(float(value)) and value <= threshold_px) for value in residual]


def filtered_metrics(
    details: Sequence[dict[str, str]],
    *,
    threshold_px: float,
    min_matches: int,
    min_score: float,
) -> tuple[int, int, int]:
    keep_mask = homography_keep_mask(details, threshold_px)
    kept = [row for row, keep in zip(details, keep_mask) if keep and float_value(row, "score") >= min_score]
    if len(kept) < min_matches:
        return 0, 0, 0
    matches = len(kept)
    correct = sum(1 for row in kept if str(row.get("correct", "")).strip() == "1")
    return matches, correct, matches - correct


def _group_variant(details: Sequence[dict[str, str]], lightglue_row: dict[str, str]) -> str:
    if details:
        return str(details[0].get("target_variant", "")).strip() or str(lightglue_row.get("target_variant", "")).strip()
    return str(lightglue_row.get("target_variant", "")).strip() or "unknown"


def _group_base_id(details: Sequence[dict[str, str]], lightglue_row: dict[str, str]) -> str:
    if details:
        return str(details[0].get("base_id", "")).strip()
    return str(lightglue_row.get("base_id", "")).strip()


def build_pair_rows(
    detail_rows: Sequence[dict[str, str]],
    lightglue_rows: Sequence[dict[str, str]],
    *,
    source: str,
    split: str,
    lightglue_label: str,
    threshold_px: float,
    min_matches: int,
    min_score: float = 0.0,
    target_variants: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    selected_lightglue_rows = select_lightglue_rows(lightglue_rows, lightglue_label=lightglue_label)
    detail_groups = detail_groups_by_pair(detail_rows)
    target_variant_set = set(target_variants)
    rows: list[dict[str, object]] = []
    for ordinal, lightglue_row in enumerate(selected_lightglue_rows):
        pair_index = lightglue_pair_index(lightglue_row, ordinal)
        details = detail_groups.get(pair_index, [])
        target_variant = _group_variant(details, lightglue_row)
        if target_variant_set and target_variant not in target_variant_set:
            continue
        pfm_matches, pfm_correct, pfm_wrong = filtered_metrics(
            details,
            threshold_px=threshold_px,
            min_matches=min_matches,
            min_score=min_score,
        )
        lightglue_matches = int_value(lightglue_row, "matches")
        lightglue_correct = int_value(lightglue_row, "correct")
        lightglue_wrong = int_value(lightglue_row, "wrong")
        rows.append(
            {
                "source": source,
                "split": split,
                "pair_index": pair_index,
                "base_id": _group_base_id(details, lightglue_row),
                "lightglue_base_id": lightglue_row.get("base_id", ""),
                "target_variant": target_variant,
                "threshold_px": f"{threshold_px:.6f}",
                "min_matches": min_matches,
                "min_score": f"{min_score:.6f}",
                "raw_detail_matches": len(details),
                "pfm_matches": pfm_matches,
                "pfm_correct": pfm_correct,
                "pfm_wrong": pfm_wrong,
                "pfm_precision": pfm_correct / pfm_matches if pfm_matches else 0.0,
                "lightglue_matches": lightglue_matches,
                "lightglue_correct": lightglue_correct,
                "lightglue_wrong": lightglue_wrong,
                "lightglue_precision": lightglue_correct / lightglue_matches if lightglue_matches else 0.0,
                "match_delta_vs_lightglue": pfm_matches - lightglue_matches,
                "correct_delta_vs_lightglue": pfm_correct - lightglue_correct,
                "wrong_delta_vs_lightglue": pfm_wrong - lightglue_wrong,
                "lightglue_label": lightglue_label,
            }
        )
    return rows


def summarize_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    materialized = list(rows)
    pfm_matches = sum(int_value(row, "pfm_matches") for row in materialized)
    pfm_correct = sum(int_value(row, "pfm_correct") for row in materialized)
    pfm_wrong = sum(int_value(row, "pfm_wrong") for row in materialized)
    lightglue_matches = sum(int_value(row, "lightglue_matches") for row in materialized)
    lightglue_correct = sum(int_value(row, "lightglue_correct") for row in materialized)
    lightglue_wrong = sum(int_value(row, "lightglue_wrong") for row in materialized)
    return {
        "rows": len(materialized),
        "pfm_matches": pfm_matches,
        "pfm_correct": pfm_correct,
        "pfm_wrong": pfm_wrong,
        "pfm_precision": pfm_correct / pfm_matches if pfm_matches else 0.0,
        "lightglue_matches": lightglue_matches,
        "lightglue_correct": lightglue_correct,
        "lightglue_wrong": lightglue_wrong,
        "lightglue_precision": lightglue_correct / lightglue_matches if lightglue_matches else 0.0,
        "match_delta_vs_lightglue": pfm_matches - lightglue_matches,
        "correct_delta_vs_lightglue": pfm_correct - lightglue_correct,
        "wrong_delta_vs_lightglue": pfm_wrong - lightglue_wrong,
    }


def summarize_by(keys: list[str], rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")).strip() or "unknown" for field in keys)
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for key, group_rows in sorted(groups.items()):
        summary = summarize_rows(group_rows)
        for field, value in zip(keys, key):
            summary[field] = value
        summaries.append(summary)
    return summaries


def best_summary(summaries: Sequence[dict[str, object]]) -> dict[str, object]:
    if not summaries:
        return {}
    return max(
        summaries,
        key=lambda row: (
            int_value(row, "wrong_delta_vs_lightglue") <= 0,
            int_value(row, "correct_delta_vs_lightglue"),
            -int_value(row, "wrong_delta_vs_lightglue"),
            int_value(row, "pfm_correct"),
        ),
    )


def write_html_report(
    path: Path,
    *,
    sources: Sequence[SourceSpec],
    lightglue_label: str,
    sweep_summary: Sequence[dict[str, object]],
    by_split_variant: Sequence[dict[str, object]],
    best_aggregate: dict[str, object],
) -> None:
    def table(rows: Sequence[dict[str, object]], columns: list[str]) -> str:
        header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        body = []
        for row in rows:
            body.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
                + "</tr>"
            )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    source_items = "".join(
        "<li>"
        f"{html.escape(source.name)}/{html.escape(source.split)}: "
        f"<code>{html.escape(str(source.match_details))}</code> vs "
        f"<code>{html.escape(str(source.lightglue_metrics))}</code>"
        "</li>"
        for source in sources
    )
    columns = [
        "threshold_px",
        "min_matches",
        "min_score",
        "split",
        "target_variant",
        "rows",
        "pfm_correct",
        "pfm_wrong",
        "lightglue_correct",
        "lightglue_wrong",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
    ]
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>PFM match-detail homography filter sweep</title>
  <style>
    body {{ font-family: sans-serif; line-height: 1.45; margin: 24px; }}
    code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 5px 7px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .note {{ color: #555; }}
  </style>
</head>
<body>
  <h1>PFM match-detail homography filter sweep</h1>
  <p class="note">过滤只使用匹配点坐标估计的 homography residual；correct/wrong 和 LightGlue 只用于离线评估。</p>
  <h2>输入</h2>
  <ul>{source_items}</ul>
  <p>LightGlue label: <code>{html.escape(lightglue_label)}</code></p>
  <h2>Best aggregate</h2>
  <pre>{html.escape(json.dumps(best_aggregate, ensure_ascii=False, indent=2))}</pre>
  <h2>Sweep aggregate</h2>
  {table(sweep_summary, columns)}
  <h2>By split and variant</h2>
  {table(by_split_variant, columns)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=parse_float_list, default=parse_float_list("1.5,2,2.5,3,4,5,8,10"))
    parser.add_argument("--min-matches", type=parse_int_list, default=parse_int_list("0,8,12,16"))
    parser.add_argument("--min-scores", type=parse_float_list, default=parse_float_list("0"))
    parser.add_argument("--target-variants", default="")
    parser.add_argument("--lightglue-label", default=LIGHTGLUE_LABEL)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    target_variants = parse_variant_list(args.target_variants)
    source_rows = [
        (
            source,
            read_csv_rows(source.match_details),
            read_csv_rows(source.lightglue_metrics),
        )
        for source in args.source
    ]

    sweep_summary: list[dict[str, object]] = []
    all_pair_rows: list[dict[str, object]] = []
    all_by_split_variant: list[dict[str, object]] = []
    for threshold_px in args.thresholds:
        for min_matches in args.min_matches:
            for min_score in args.min_scores:
                config_rows: list[dict[str, object]] = []
                for source, detail_rows, lightglue_rows in source_rows:
                    config_rows.extend(
                        build_pair_rows(
                            detail_rows,
                            lightglue_rows,
                            source=source.name,
                            split=source.split,
                            lightglue_label=args.lightglue_label,
                            threshold_px=threshold_px,
                            min_matches=min_matches,
                            min_score=min_score,
                            target_variants=target_variants,
                        )
                    )
                summary = summarize_rows(config_rows)
                summary.update(
                    {
                        "threshold_px": f"{threshold_px:.6f}",
                        "min_matches": min_matches,
                        "min_score": f"{min_score:.6f}",
                    }
                )
                sweep_summary.append(summary)
                all_pair_rows.extend(config_rows)
                for split_variant_summary in summarize_by(["split", "target_variant"], config_rows):
                    split_variant_summary.update(
                        {
                            "threshold_px": f"{threshold_px:.6f}",
                            "min_matches": min_matches,
                            "min_score": f"{min_score:.6f}",
                        }
                    )
                    all_by_split_variant.append(split_variant_summary)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_fields = [
        "source",
        "split",
        "pair_index",
        "base_id",
        "lightglue_base_id",
        "target_variant",
        "threshold_px",
        "min_matches",
        "min_score",
        "raw_detail_matches",
        "pfm_matches",
        "pfm_correct",
        "pfm_wrong",
        "pfm_precision",
        "lightglue_matches",
        "lightglue_correct",
        "lightglue_wrong",
        "lightglue_precision",
        "match_delta_vs_lightglue",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "lightglue_label",
    ]
    summary_fields = [
        "threshold_px",
        "min_matches",
        "min_score",
        "split",
        "target_variant",
        "rows",
        "pfm_matches",
        "pfm_correct",
        "pfm_wrong",
        "pfm_precision",
        "lightglue_matches",
        "lightglue_correct",
        "lightglue_wrong",
        "lightglue_precision",
        "match_delta_vs_lightglue",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
    ]
    write_csv_rows(output_dir / "pair_rows.csv", all_pair_rows, pair_fields)
    write_csv_rows(output_dir / "sweep_summary.csv", sweep_summary, summary_fields)
    write_csv_rows(output_dir / "sweep_by_split_variant.csv", all_by_split_variant, summary_fields)

    best_aggregate = best_summary(sweep_summary)
    payload = {
        "sources": [
            asdict(source) | {"match_details": str(source.match_details), "lightglue_metrics": str(source.lightglue_metrics)}
            for source in args.source
        ],
        "lightglue_label": args.lightglue_label,
        "thresholds": args.thresholds,
        "min_matches": args.min_matches,
        "min_scores": args.min_scores,
        "target_variants": target_variants,
        "best_aggregate": best_aggregate,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_html_report(
        output_dir / "index.html",
        sources=args.source,
        lightglue_label=args.lightglue_label,
        sweep_summary=sweep_summary,
        by_split_variant=all_by_split_variant,
        best_aggregate=best_aggregate,
    )
    print(
        f"wrote homography sweep to {output_dir} "
        f"best_correct_delta={best_aggregate.get('correct_delta_vs_lightglue')} "
        f"best_wrong_delta={best_aggregate.get('wrong_delta_vs_lightglue')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
