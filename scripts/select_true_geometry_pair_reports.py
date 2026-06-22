#!/usr/bin/env python3
"""Select per-pair PFM reports using true-geometry filtered match counts."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PAIR_SELECTION_FIELDS = [
    "eval_split",
    "pair_index",
    "manifest_pair_index",
    "manifest_split",
    "pair_type",
    "reference_base_id",
    "target_base_id",
    "reference_variant",
    "target_variant",
    "valid_fraction",
    "selected_source",
    "selected_matches",
    "selected_correct",
    "selected_wrong",
    "selected_precision",
    "lightglue_matches",
    "lightglue_correct",
    "lightglue_wrong",
    "delta_correct_vs_lightglue",
    "delta_wrong_vs_lightglue",
]


@dataclass(frozen=True)
class SelectionSource:
    split: str
    pair_manifest: Path
    lightglue_metrics: Path | None


@dataclass(frozen=True)
class CandidateReport:
    name: str
    root: Path
    eval_subdir: str

    def summary_path(self, split: str) -> Path:
        return self.root / split / self.eval_subdir / "all_filtered_summary.csv"

    def details_path(self, split: str) -> Path:
        return self.root / split / self.eval_subdir / "all_filtered_match_details.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return int(round(float(value)))
    except ValueError:
        return default


def float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def precision(correct: int, matches: int) -> float:
    return float(correct) / float(matches) if matches else 0.0


def read_lightglue_by_pair(path: Path, *, label: str) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for row in read_csv_rows(path):
        if row.get("label") != label:
            continue
        index = int_value(row, "manifest_pair_index", int_value(row, "pair_index"))
        rows[index] = row
    return rows


def details_by_pair(path: Path) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in read_csv_rows(path):
        grouped.setdefault(int_value(row, "pair_index"), []).append(row)
    return grouped


def candidate_rank(row: dict[str, str], candidate_index: int, *, selection_rank_profile: str) -> tuple[float, ...]:
    if selection_rank_profile == "inference_safe":
        return (
            float(int_value(row, "matches")),
            float_value(row, "score_mean"),
            float(-candidate_index),
        )
    if selection_rank_profile == "diagnostic_wrong_tiebreak":
        return (
            float(int_value(row, "matches")),
            float(-int_value(row, "wrong")),
            float_value(row, "score_mean"),
            float(-candidate_index),
        )
    raise ValueError(f"unsupported selection rank profile: {selection_rank_profile}")


def ensure_inputs_exist(sources: Sequence[SelectionSource], candidates: Sequence[CandidateReport]) -> None:
    for source in sources:
        if not source.pair_manifest.is_file():
            raise FileNotFoundError(f"missing required input: {source.pair_manifest}")
        if source.lightglue_metrics is not None and not source.lightglue_metrics.is_file():
            raise FileNotFoundError(f"missing required input: {source.lightglue_metrics}")
        for candidate in candidates:
            for path in (candidate.summary_path(source.split), candidate.details_path(source.split)):
                if not path.is_file():
                    raise FileNotFoundError(f"missing required input: {path}")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_fields(rows: list[dict[str, object]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def aggregate_selection(rows: list[dict[str, object]]) -> dict[str, object]:
    selected_matches = sum(int(row["selected_matches"]) for row in rows)
    selected_correct = sum(int(row["selected_correct"]) for row in rows)
    selected_wrong = sum(int(row["selected_wrong"]) for row in rows)
    lightglue_matches = sum(int(row["lightglue_matches"]) for row in rows)
    lightglue_correct = sum(int(row["lightglue_correct"]) for row in rows)
    lightglue_wrong = sum(int(row["lightglue_wrong"]) for row in rows)
    return {
        "rows": len(rows),
        "selected_matches": selected_matches,
        "selected_correct": selected_correct,
        "selected_wrong": selected_wrong,
        "selected_precision": precision(selected_correct, selected_matches),
        "lightglue_matches": lightglue_matches,
        "lightglue_correct": lightglue_correct,
        "lightglue_wrong": lightglue_wrong,
        "lightglue_precision": precision(lightglue_correct, lightglue_matches),
        "correct_delta_vs_lightglue": selected_correct - lightglue_correct,
        "wrong_delta_vs_lightglue": selected_wrong - lightglue_wrong,
    }


def select_reports(
    *,
    sources: Sequence[SelectionSource],
    candidates: Sequence[CandidateReport],
    output_dir: Path,
    lightglue_label: str,
    selection_rank_profile: str = "inference_safe",
) -> dict[str, object]:
    if not sources:
        raise ValueError("at least one source is required")
    if not candidates:
        raise ValueError("at least one candidate is required")
    ensure_inputs_exist(sources, candidates)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_selection_rows: list[dict[str, object]] = []
    source_pair_counts = {candidate.name: 0 for candidate in candidates}
    by_split: dict[str, dict[str, object]] = {}
    by_variant_rows: dict[str, list[dict[str, object]]] = {}

    for source in sources:
        pair_rows = read_csv_rows(source.pair_manifest)
        lightglue_rows = (
            {}
            if source.lightglue_metrics is None
            else read_lightglue_by_pair(source.lightglue_metrics, label=lightglue_label)
        )
        summaries = {candidate.name: read_csv_rows(candidate.summary_path(source.split)) for candidate in candidates}
        details = {candidate.name: details_by_pair(candidate.details_path(source.split)) for candidate in candidates}
        if any(len(rows) != len(pair_rows) for rows in summaries.values()):
            raise ValueError(f"summary row count does not match pair manifest for split={source.split}")

        selected_summary_rows: list[dict[str, object]] = []
        selected_detail_rows: list[dict[str, object]] = []
        split_selection_rows: list[dict[str, object]] = []
        for pair_index, pair in enumerate(pair_rows):
            selected_candidate = max(
                enumerate(candidates),
                key=lambda item: candidate_rank(
                    summaries[item[1].name][pair_index],
                    item[0],
                    selection_rank_profile=selection_rank_profile,
                ),
            )[1]
            selected_name = selected_candidate.name
            source_pair_counts[selected_name] += 1
            selected = dict(summaries[selected_name][pair_index])
            selected["selector_source"] = selected_name
            selected_summary_rows.append(selected)
            for detail in details[selected_name].get(pair_index, []):
                copied = dict(detail)
                copied["selector_source"] = selected_name
                selected_detail_rows.append(copied)

            lightglue = lightglue_rows.get(pair_index, {})
            selected_matches = int_value(selected, "matches")
            selected_correct = int_value(selected, "correct")
            selected_wrong = int_value(selected, "wrong")
            selection_row: dict[str, object] = {
                "eval_split": source.split,
                "pair_index": pair_index,
                "manifest_pair_index": pair.get("pair_index", str(pair_index)),
                "manifest_split": pair.get("split", ""),
                "pair_type": pair.get("pair_type", ""),
                "reference_base_id": pair.get("reference_base_id", ""),
                "target_base_id": pair.get("target_base_id", selected.get("base_id", "")),
                "reference_variant": pair.get("reference_variant", ""),
                "target_variant": pair.get("target_variant", selected.get("target_variant", "")),
                "valid_fraction": pair.get("valid_fraction", selected.get("valid_fraction", "")),
                "selected_source": selected_name,
                "selected_matches": selected_matches,
                "selected_correct": selected_correct,
                "selected_wrong": selected_wrong,
                "selected_precision": f"{precision(selected_correct, selected_matches):.6f}",
                "lightglue_matches": int_value(lightglue, "matches"),
                "lightglue_correct": int_value(lightglue, "correct"),
                "lightglue_wrong": int_value(lightglue, "wrong"),
                "delta_correct_vs_lightglue": selected_correct - int_value(lightglue, "correct"),
                "delta_wrong_vs_lightglue": selected_wrong - int_value(lightglue, "wrong"),
            }
            for candidate in candidates:
                row = summaries[candidate.name][pair_index]
                selection_row[f"{candidate.name}_matches"] = int_value(row, "matches")
                selection_row[f"{candidate.name}_correct"] = int_value(row, "correct")
                selection_row[f"{candidate.name}_wrong"] = int_value(row, "wrong")
            split_selection_rows.append(selection_row)
            all_selection_rows.append(selection_row)
            by_variant_rows.setdefault(str(selection_row["target_variant"]), []).append(selection_row)

        split_dir = output_dir / source.split
        write_csv(split_dir / "selected_summary.csv", selected_summary_rows, infer_fields(selected_summary_rows))
        write_csv(split_dir / "selected_match_details.csv", selected_detail_rows, infer_fields(selected_detail_rows))
        selection_fields = [*PAIR_SELECTION_FIELDS]
        for candidate in candidates:
            selection_fields.extend(
                [
                    f"{candidate.name}_matches",
                    f"{candidate.name}_correct",
                    f"{candidate.name}_wrong",
                ]
            )
        write_csv(split_dir / "pair_selection.csv", split_selection_rows, selection_fields)
        by_split[source.split] = aggregate_selection(split_selection_rows)

    selection_fields = [*PAIR_SELECTION_FIELDS]
    for candidate in candidates:
        selection_fields.extend([f"{candidate.name}_matches", f"{candidate.name}_correct", f"{candidate.name}_wrong"])
    write_csv(output_dir / "pair_selection.csv", all_selection_rows, selection_fields)
    summary = {
        "selector": "per_pair_max_true_geometry_filtered_matches",
        "aggregate": aggregate_selection(all_selection_rows),
        "by_split": by_split,
        "by_variant": {
            variant: aggregate_selection(rows)
            for variant, rows in sorted(by_variant_rows.items())
        },
        "source_pair_counts": source_pair_counts,
        "uses_lightglue_baseline": any(source.lightglue_metrics is not None for source in sources),
        "sources": [
            {
                "split": source.split,
                "pair_manifest": str(source.pair_manifest),
                "lightglue_metrics": "" if source.lightglue_metrics is None else str(source.lightglue_metrics),
            }
            for source in sources
        ],
        "candidates": [
            {"name": candidate.name, "root": str(candidate.root), "eval_subdir": candidate.eval_subdir}
            for candidate in candidates
        ],
        "lightglue_label": lightglue_label,
        "selection_rank_profile": selection_rank_profile,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report_html(output_dir / "summary.html", summary=summary, pair_selection=output_dir / "pair_selection.csv")
    return summary


def write_report_html(path: Path, *, summary: dict[str, object], pair_selection: Path) -> None:
    rows = []
    aggregate = summary["aggregate"]
    by_split = summary["by_split"]
    assert isinstance(aggregate, dict)
    assert isinstance(by_split, dict)
    for name, item in [("aggregate", aggregate), *by_split.items()]:
        assert isinstance(item, dict)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{item['rows']}</td>"
            f"<td>{item['selected_correct']}</td>"
            f"<td>{item['selected_wrong']}</td>"
            f"<td>{float(item['selected_precision']):.6f}</td>"
            f"<td>{item['lightglue_correct']}</td>"
            f"<td>{item['lightglue_wrong']}</td>"
            f"<td>{float(item['lightglue_precision']):.6f}</td>"
            f"<td>{item['correct_delta_vs_lightglue']}</td>"
            f"<td>{item['wrong_delta_vs_lightglue']}</td>"
            "</tr>"
        )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-CN">',
                '<head><meta charset="utf-8"><title>True geometry pair selector</title></head>',
                "<body>",
                "<h1>True geometry pair selector</h1>",
                "<p>selector=<code>per pair max true-geometry filtered match count</code></p>",
                f"<p>pair_selection=<code>{html.escape(str(pair_selection))}</code></p>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                "<tr><th>split</th><th>rows</th><th>selected correct</th><th>selected wrong</th>"
                "<th>selected precision</th><th>LightGlue correct</th><th>LightGlue wrong</th>"
                "<th>LightGlue precision</th><th>delta correct</th><th>delta wrong</th></tr>",
                *rows,
                "</table>",
                "</body>",
                "</html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_source(value: str) -> SelectionSource:
    parts = value.split(",", 2)
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("--source must be split,pair_manifest[,lightglue_metrics]")
    split, pair_manifest = parts[:2]
    lightglue_metrics = Path(parts[2]) if len(parts) == 3 and parts[2] else None
    return SelectionSource(split=split, pair_manifest=Path(pair_manifest), lightglue_metrics=lightglue_metrics)


def parse_candidate(value: str) -> CandidateReport:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--candidate must be name,root,eval_subdir")
    name, root, eval_subdir = parts
    if not name:
        raise argparse.ArgumentTypeError("candidate name must be nonempty")
    return CandidateReport(name=name, root=Path(root), eval_subdir=eval_subdir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=parse_source, action="append", required=True)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lightglue-label", default="LightGlue-SIFT-MAGSAC-min16")
    parser.add_argument(
        "--selection-rank-profile",
        choices=["inference_safe", "diagnostic_wrong_tiebreak"],
        default="inference_safe",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = select_reports(
        sources=args.source,
        candidates=args.candidate,
        output_dir=args.output_dir,
        lightglue_label=str(args.lightglue_label),
        selection_rank_profile=str(args.selection_rank_profile),
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
