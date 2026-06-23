#!/usr/bin/env python3
"""Analyze row-level recall and wrong-match gaps between PFM and LightGlue."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


LIGHTGLUE_LABEL = "LightGlue-SIFT-MAGSAC-min16"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    split: str
    pfm_summary: Path
    lightglue_metrics: Path


def int_value(row: dict[str, object], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, object]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for field in preferred_fields or []:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_source_spec(text: str) -> SourceSpec:
    parts = [part.strip() for part in text.split(",", 3)]
    if len(parts) != 4 or any(not part for part in parts):
        raise argparse.ArgumentTypeError("--source must be name,split,pfm_summary,lightglue_metrics")
    name, split, pfm_summary, lightglue_metrics = parts
    return SourceSpec(
        name=name,
        split=split,
        pfm_summary=Path(pfm_summary),
        lightglue_metrics=Path(lightglue_metrics),
    )


def select_pfm_rows(rows: list[dict[str, str]], *, source: str, split: str) -> list[dict[str, str]]:
    if not rows:
        return []
    has_source = any("source" in row and str(row.get("source", "")).strip() for row in rows)
    has_split = any("split" in row and str(row.get("split", "")).strip() for row in rows)
    if not has_source and not has_split:
        return rows

    selected = [
        row
        for row in rows
        if (not has_source or str(row.get("source", "")).strip() == source)
        and (not has_split or str(row.get("split", "")).strip() == split)
    ]
    if selected:
        return selected
    raise ValueError(f"no PFM rows found for source={source!r} split={split!r}")


def select_lightglue_rows(rows: list[dict[str, str]], *, lightglue_label: str) -> list[dict[str, str]]:
    if not rows:
        return []
    if "label" not in rows[0]:
        return rows
    selected = [row for row in rows if str(row.get("label", "")).strip() == lightglue_label]
    if selected:
        return selected
    raise ValueError(f"no LightGlue rows found for label={lightglue_label!r}")


def classify_gap(correct_delta: int, wrong_delta: int) -> str:
    if wrong_delta > 0:
        return "pfm_wrong_risk"
    if correct_delta < 0:
        return "lightglue_recall_gap"
    if correct_delta > 0:
        return "pfm_clean_win"
    return "tie"


def build_gap_rows(
    pfm_rows: list[dict[str, object]],
    lightglue_rows: list[dict[str, object]],
    *,
    source: str,
    split: str,
    lightglue_label: str,
) -> list[dict[str, object]]:
    selected_lightglue_rows = select_lightglue_rows(
        [dict(row) for row in lightglue_rows],
        lightglue_label=lightglue_label,
    )
    if len(pfm_rows) != len(selected_lightglue_rows):
        raise ValueError(
            f"{source}/{split}: PFM/LightGlue row count mismatch: "
            f"{len(pfm_rows)} vs {len(selected_lightglue_rows)}"
        )

    gap_rows: list[dict[str, object]] = []
    for row_index, (pfm_row, lightglue_row) in enumerate(zip(pfm_rows, selected_lightglue_rows)):
        pfm_variant = str(pfm_row.get("target_variant", "")).strip()
        lightglue_variant = str(lightglue_row.get("target_variant", "")).strip()
        if pfm_variant and lightglue_variant and pfm_variant != lightglue_variant:
            raise ValueError(
                f"{source}/{split}: row {row_index} variant mismatch: "
                f"PFM={pfm_variant!r} LightGlue={lightglue_variant!r}"
            )

        pfm_matches = int_value(pfm_row, "matches")
        pfm_correct = int_value(pfm_row, "correct")
        pfm_wrong = int_value(pfm_row, "wrong")
        lightglue_matches = int_value(lightglue_row, "matches")
        lightglue_correct = int_value(lightglue_row, "correct")
        lightglue_wrong = int_value(lightglue_row, "wrong")
        match_delta = pfm_matches - lightglue_matches
        correct_delta = pfm_correct - lightglue_correct
        wrong_delta = pfm_wrong - lightglue_wrong
        lightglue_correct_gap = max(0, lightglue_correct - pfm_correct)
        pfm_wrong_excess = max(0, pfm_wrong - lightglue_wrong)

        gap_rows.append(
            {
                "source": source,
                "split": split,
                "row_index": row_index,
                "pfm_row_index": pfm_row.get("row_index", row_index),
                "pfm_base_id": str(pfm_row.get("base_id", "")).strip(),
                "lightglue_base_id": str(lightglue_row.get("base_id", "")).strip(),
                "target_variant": pfm_variant or lightglue_variant or "unknown",
                "gap_bucket": classify_gap(correct_delta, wrong_delta),
                "pfm_matches": pfm_matches,
                "pfm_correct": pfm_correct,
                "pfm_wrong": pfm_wrong,
                "lightglue_matches": lightglue_matches,
                "lightglue_correct": lightglue_correct,
                "lightglue_wrong": lightglue_wrong,
                "match_delta_vs_lightglue": match_delta,
                "correct_delta_vs_lightglue": correct_delta,
                "wrong_delta_vs_lightglue": wrong_delta,
                "lightglue_correct_gap": lightglue_correct_gap,
                "pfm_wrong_excess": pfm_wrong_excess,
                "pfm_score_mean": pfm_row.get("score_mean", ""),
                "pfm_valid_fraction": pfm_row.get("valid_fraction", lightglue_row.get("valid_fraction", "")),
                "pfm_homography_residual_p90_px": pfm_row.get("homography_residual_p90_px", ""),
                "pfm_homography_residual_median_px": pfm_row.get("homography_residual_median_px", ""),
                "pfm_displacement_mad_px": pfm_row.get("displacement_mad_px", ""),
                "pfm_selected_model": pfm_row.get("selected_model", ""),
                "pfm_selector_reason": pfm_row.get("selector_reason", ""),
                "lightglue_label": lightglue_label,
            }
        )
    return gap_rows


def summarize_total(rows: Iterable[dict[str, object]]) -> dict[str, object]:
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
        "lightglue_correct_gap": sum(int_value(row, "lightglue_correct_gap") for row in materialized),
        "pfm_wrong_excess": sum(int_value(row, "pfm_wrong_excess") for row in materialized),
        "lightglue_gap_rows": sum(1 for row in materialized if int_value(row, "lightglue_correct_gap") > 0),
        "pfm_wrong_risk_rows": sum(1 for row in materialized if int_value(row, "pfm_wrong_excess") > 0),
        "pfm_clean_win_rows": sum(1 for row in materialized if row.get("gap_bucket") == "pfm_clean_win"),
    }


def summarize_by(keys: list[str], rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(column, "")).strip() or "unknown" for column in keys)
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        summary = summarize_total(group_rows)
        for column, value in zip(keys, key):
            summary[column] = value
        summaries.append(summary)
    return summaries


def write_html_report(
    path: Path,
    *,
    sources: list[SourceSpec],
    lightglue_label: str,
    summary: dict[str, object],
    by_split: list[dict[str, object]],
    by_variant: list[dict[str, object]],
    top_gaps: list[dict[str, object]],
    top_wrong_risks: list[dict[str, object]],
) -> None:
    def table(rows: list[dict[str, object]], columns: list[str]) -> str:
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
        f"<code>{html.escape(str(source.pfm_summary))}</code> vs "
        f"<code>{html.escape(str(source.lightglue_metrics))}</code>"
        "</li>"
        for source in sources
    )
    summary_columns = [
        "split",
        "target_variant",
        "rows",
        "pfm_correct",
        "pfm_wrong",
        "lightglue_correct",
        "lightglue_wrong",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "lightglue_correct_gap",
        "pfm_wrong_excess",
    ]
    row_columns = [
        "source",
        "split",
        "row_index",
        "target_variant",
        "gap_bucket",
        "pfm_base_id",
        "lightglue_base_id",
        "pfm_correct",
        "pfm_wrong",
        "lightglue_correct",
        "lightglue_wrong",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "lightglue_correct_gap",
        "pfm_wrong_excess",
        "pfm_score_mean",
        "pfm_valid_fraction",
        "pfm_homography_residual_p90_px",
        "pfm_displacement_mad_px",
        "pfm_selected_model",
    ]
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>PFM vs LightGlue gap analysis</title>
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
  <h1>PFM vs LightGlue gap analysis</h1>
  <p class="note">该报告只用于诊断和评估。后续 selector 仍必须只使用 PFM 推理可观测字段，不能在推理时读取 LightGlue correct/wrong。</p>
  <h2>输入</h2>
  <ul>{source_items}</ul>
  <p>LightGlue label: <code>{html.escape(lightglue_label)}</code></p>
  <h2>总体</h2>
  <pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
  <h2>按 split</h2>
  {table(by_split, summary_columns)}
  <h2>按 variant</h2>
  {table(by_variant, summary_columns)}
  <h2>LightGlue recall gap Top</h2>
  {table(top_gaps, row_columns)}
  <h2>PFM wrong risk Top</h2>
  {table(top_wrong_risks, row_columns)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lightglue-label", default=LIGHTGLUE_LABEL)
    parser.add_argument("--top-k", type=int, default=80)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    all_gap_rows: list[dict[str, object]] = []
    for source in args.source:
        pfm_rows = select_pfm_rows(
            read_csv_rows(source.pfm_summary),
            source=source.name,
            split=source.split,
        )
        lightglue_rows = read_csv_rows(source.lightglue_metrics)
        all_gap_rows.extend(
            build_gap_rows(
                pfm_rows,
                lightglue_rows,
                source=source.name,
                split=source.split,
                lightglue_label=args.lightglue_label,
            )
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preferred_fields = [
        "source",
        "split",
        "row_index",
        "pfm_row_index",
        "target_variant",
        "gap_bucket",
        "pfm_base_id",
        "lightglue_base_id",
        "pfm_matches",
        "pfm_correct",
        "pfm_wrong",
        "lightglue_matches",
        "lightglue_correct",
        "lightglue_wrong",
        "match_delta_vs_lightglue",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "lightglue_correct_gap",
        "pfm_wrong_excess",
        "pfm_score_mean",
        "pfm_valid_fraction",
        "pfm_homography_residual_p90_px",
        "pfm_homography_residual_median_px",
        "pfm_displacement_mad_px",
        "pfm_selected_model",
        "pfm_selector_reason",
        "lightglue_label",
    ]
    write_csv_rows(output_dir / "gap_rows.csv", all_gap_rows, preferred_fields)

    by_split = summarize_by(["split"], all_gap_rows)
    by_variant = summarize_by(["target_variant"], all_gap_rows)
    by_split_variant = summarize_by(["split", "target_variant"], all_gap_rows)
    by_bucket = summarize_by(["gap_bucket"], all_gap_rows)
    write_csv_rows(output_dir / "summary_by_split.csv", by_split)
    write_csv_rows(output_dir / "summary_by_variant.csv", by_variant)
    write_csv_rows(output_dir / "summary_by_split_variant.csv", by_split_variant)
    write_csv_rows(output_dir / "summary_by_bucket.csv", by_bucket)

    top_gaps = sorted(
        [row for row in all_gap_rows if int_value(row, "lightglue_correct_gap") > 0],
        key=lambda row: (
            -int_value(row, "lightglue_correct_gap"),
            int_value(row, "wrong_delta_vs_lightglue"),
            str(row.get("target_variant", "")),
            str(row.get("pfm_base_id", "")),
        ),
    )[: args.top_k]
    top_wrong_risks = sorted(
        [row for row in all_gap_rows if int_value(row, "pfm_wrong_excess") > 0],
        key=lambda row: (
            -int_value(row, "pfm_wrong_excess"),
            -int_value(row, "correct_delta_vs_lightglue"),
            str(row.get("target_variant", "")),
            str(row.get("pfm_base_id", "")),
        ),
    )[: args.top_k]
    write_csv_rows(output_dir / "top_lightglue_gaps.csv", top_gaps, preferred_fields)
    write_csv_rows(output_dir / "top_pfm_wrong_risks.csv", top_wrong_risks, preferred_fields)

    summary_payload = {
        "sources": [asdict(source) | {"pfm_summary": str(source.pfm_summary), "lightglue_metrics": str(source.lightglue_metrics)} for source in args.source],
        "lightglue_label": args.lightglue_label,
        "aggregate": summarize_total(all_gap_rows),
        "by_split": {str(row["split"]): row for row in by_split},
        "by_variant": {str(row["target_variant"]): row for row in by_variant},
        "by_bucket": {str(row["gap_bucket"]): row for row in by_bucket},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html_report(
        output_dir / "index.html",
        sources=args.source,
        lightglue_label=args.lightglue_label,
        summary=summary_payload["aggregate"],
        by_split=by_split,
        by_variant=by_variant,
        top_gaps=top_gaps,
        top_wrong_risks=top_wrong_risks,
    )

    print(
        f"wrote {len(all_gap_rows)} rows to {output_dir} "
        f"correct_delta={summary_payload['aggregate']['correct_delta_vs_lightglue']} "
        f"wrong_delta={summary_payload['aggregate']['wrong_delta_vs_lightglue']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
