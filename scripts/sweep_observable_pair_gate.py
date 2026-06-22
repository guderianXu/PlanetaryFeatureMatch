#!/usr/bin/env python3
"""Sweep observable pair-level PFM/LightGlue hybrid gates."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


IDENTITY_FIELDS = [
    "source_name",
    "split",
    "pair_index",
    "pair_type",
    "base_id",
    "reference_variant",
    "target_variant",
]

FORBIDDEN_FIELD_FRAGMENTS = (
    "correct",
    "wrong",
    "label",
    "teacher",
    "oracle",
    "delta",
)


@dataclass(frozen=True)
class Rule:
    name: str
    masks_by_split: dict[str, int]


@dataclass
class _SplitMetricCache:
    row_count: int
    pfm_matches: int
    pfm_correct: int
    pfm_wrong: int
    lightglue_matches: int
    lightglue_correct: int
    lightglue_wrong: int
    match_delta_by_index: list[int]
    correct_delta_by_index: list[int]
    wrong_delta_by_index: list[int]
    cache: dict[int, dict[str, int | float]]

    @classmethod
    def from_rows(cls, rows: Sequence[dict[str, str]]) -> "_SplitMetricCache":
        pfm_matches = sum(_int_value(row, "pfm_matches") for row in rows)
        pfm_correct = sum(_int_value(row, "pfm_correct") for row in rows)
        pfm_wrong = sum(_int_value(row, "pfm_wrong") for row in rows)
        lightglue_matches = sum(_int_value(row, "lightglue_matches") for row in rows)
        lightglue_correct = sum(_int_value(row, "lightglue_correct") for row in rows)
        lightglue_wrong = sum(_int_value(row, "lightglue_wrong") for row in rows)
        return cls(
            row_count=len(rows),
            pfm_matches=pfm_matches,
            pfm_correct=pfm_correct,
            pfm_wrong=pfm_wrong,
            lightglue_matches=lightglue_matches,
            lightglue_correct=lightglue_correct,
            lightglue_wrong=lightglue_wrong,
            match_delta_by_index=[
                _int_value(row, "pfm_matches") - _int_value(row, "lightglue_matches")
                for row in rows
            ],
            correct_delta_by_index=[
                _int_value(row, "pfm_correct") - _int_value(row, "lightglue_correct")
                for row in rows
            ],
            wrong_delta_by_index=[
                _int_value(row, "pfm_wrong") - _int_value(row, "lightglue_wrong")
                for row in rows
            ],
            cache={},
        )

    def summary_for_mask(self, mask: int) -> dict[str, int | float]:
        cached = self.cache.get(mask)
        if cached is not None:
            return cached
        match_delta = 0
        correct_delta = 0
        wrong_delta = 0
        active = mask
        while active:
            bit = active & -active
            index = bit.bit_length() - 1
            match_delta += self.match_delta_by_index[index]
            correct_delta += self.correct_delta_by_index[index]
            wrong_delta += self.wrong_delta_by_index[index]
            active ^= bit
        hybrid_matches = self.lightglue_matches + match_delta
        hybrid_correct = self.lightglue_correct + correct_delta
        hybrid_wrong = self.lightglue_wrong + wrong_delta
        summary = {
            "rows": self.row_count,
            "use_pfm_pairs": mask.bit_count(),
            "fallback_lightglue_pairs": self.row_count - mask.bit_count(),
            "pfm_matches": self.pfm_matches,
            "pfm_correct": self.pfm_correct,
            "pfm_wrong": self.pfm_wrong,
            "pfm_precision": self.pfm_correct / self.pfm_matches if self.pfm_matches else 0.0,
            "lightglue_matches": self.lightglue_matches,
            "lightglue_correct": self.lightglue_correct,
            "lightglue_wrong": self.lightglue_wrong,
            "lightglue_precision": self.lightglue_correct / self.lightglue_matches if self.lightglue_matches else 0.0,
            "hybrid_matches": hybrid_matches,
            "hybrid_correct": hybrid_correct,
            "hybrid_wrong": hybrid_wrong,
            "hybrid_precision": hybrid_correct / hybrid_matches if hybrid_matches else 0.0,
            "correct_delta_vs_lightglue": correct_delta,
            "wrong_delta_vs_lightglue": wrong_delta,
            "correct_delta_vs_pfm": hybrid_correct - self.pfm_correct,
            "wrong_delta_vs_pfm": hybrid_wrong - self.pfm_wrong,
        }
        self.cache[mask] = summary
        return summary


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def group_rows_by_split(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("split", ""), []).append(dict(row))
    return grouped


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _fmt_float(value: float) -> str:
    return f"{value:.6f}"


def _is_observable_feature(name: str) -> bool:
    if not name.startswith("feature_"):
        return False
    lowered = name.lower()
    return not any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS)


def _thresholds(values: Sequence[float], max_thresholds: int) -> list[float]:
    finite = sorted({value for value in values if math.isfinite(value)})
    if not finite:
        return []
    if len(finite) <= max_thresholds:
        return finite
    if max_thresholds <= 1:
        return [finite[len(finite) // 2]]
    indexes = {
        round(index * (len(finite) - 1) / (max_thresholds - 1))
        for index in range(max_thresholds)
    }
    return [finite[index] for index in sorted(indexes)]


def _mask_for_rows(rows: Sequence[dict[str, str]], predicate: Callable[[dict[str, str]], bool]) -> int:
    mask = 0
    for index, row in enumerate(rows):
        if predicate(row):
            mask |= 1 << index
    return mask


def _all_rows(grouped: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split_rows in grouped.values():
        rows.extend(split_rows)
    return rows


def _observable_feature_fields(grouped: dict[str, list[dict[str, str]]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in _all_rows(grouped):
        for key in row:
            if key in seen or not _is_observable_feature(key):
                continue
            fields.append(key)
            seen.add(key)
    return fields


def build_base_rules(
    grouped: dict[str, list[dict[str, str]]],
    *,
    max_thresholds: int = 36,
) -> list[Rule]:
    rules: list[Rule] = []
    seen_signatures: set[tuple[tuple[str, int], ...]] = set()
    all_rows = _all_rows(grouped)

    def add_rule(name: str, masks_by_split: dict[str, int], *, dedupe: bool = True) -> None:
        if not any(mask != 0 for mask in masks_by_split.values()):
            return
        signature = tuple(sorted(masks_by_split.items()))
        if dedupe and signature in seen_signatures:
            return
        if dedupe:
            seen_signatures.add(signature)
        rules.append(Rule(name=name, masks_by_split=masks_by_split))

    for field in _observable_feature_fields(grouped):
        values = [_float_value(row, field, float("nan")) for row in all_rows]
        for threshold in _thresholds(values, max_thresholds):
            add_rule(
                f"{field} >= {threshold:.6g}",
                {
                    split: _mask_for_rows(
                        rows,
                        lambda row, field=field, threshold=threshold: _float_value(row, field, float("-inf"))
                        >= threshold,
                    )
                    for split, rows in grouped.items()
                },
            )
            add_rule(
                f"{field} <= {threshold:.6g}",
                {
                    split: _mask_for_rows(
                        rows,
                        lambda row, field=field, threshold=threshold: _float_value(row, field, float("inf"))
                        <= threshold,
                    )
                    for split, rows in grouped.items()
                },
            )

    variants = sorted({row.get("target_variant", "") for row in all_rows if row.get("target_variant", "")})
    for variant in variants:
        add_rule(
            f"target_variant == {variant}",
            {
                split: _mask_for_rows(rows, lambda row, variant=variant: row.get("target_variant", "") == variant)
                for split, rows in grouped.items()
            },
            dedupe=False,
        )
    return rules


def build_pairwise_and_rules(base_rules: Sequence[Rule]) -> list[Rule]:
    rules = list(base_rules)
    seen_signatures: set[tuple[tuple[str, int], ...]] = {
        tuple(sorted(rule.masks_by_split.items()))
        for rule in base_rules
    }
    for left_index, left in enumerate(base_rules):
        for right in base_rules[left_index + 1 :]:
            masks = {
                split: left.masks_by_split.get(split, 0) & right.masks_by_split.get(split, 0)
                for split in left.masks_by_split
            }
            if not any(mask != 0 for mask in masks.values()):
                continue
            signature = tuple(sorted(masks.items()))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            rules.append(Rule(name=f"{left.name} AND {right.name}", masks_by_split=masks))
    return rules


def _is_selected(selected: set[int] | int, index: int) -> bool:
    if isinstance(selected, int):
        return bool(selected & (1 << index))
    return index in selected


def hybrid_summary_for_selected(rows: Sequence[dict[str, str]], selected: set[int] | int) -> dict[str, int | float]:
    pfm_matches = sum(_int_value(row, "pfm_matches") for row in rows)
    pfm_correct = sum(_int_value(row, "pfm_correct") for row in rows)
    pfm_wrong = sum(_int_value(row, "pfm_wrong") for row in rows)
    lightglue_matches = sum(_int_value(row, "lightglue_matches") for row in rows)
    lightglue_correct = sum(_int_value(row, "lightglue_correct") for row in rows)
    lightglue_wrong = sum(_int_value(row, "lightglue_wrong") for row in rows)

    hybrid_matches = 0
    hybrid_correct = 0
    hybrid_wrong = 0
    use_pfm_pairs = 0
    for index, row in enumerate(rows):
        prefix = "pfm" if _is_selected(selected, index) else "lightglue"
        if prefix == "pfm":
            use_pfm_pairs += 1
        hybrid_matches += _int_value(row, f"{prefix}_matches")
        hybrid_correct += _int_value(row, f"{prefix}_correct")
        hybrid_wrong += _int_value(row, f"{prefix}_wrong")

    return {
        "rows": len(rows),
        "use_pfm_pairs": use_pfm_pairs,
        "fallback_lightglue_pairs": len(rows) - use_pfm_pairs,
        "pfm_matches": pfm_matches,
        "pfm_correct": pfm_correct,
        "pfm_wrong": pfm_wrong,
        "pfm_precision": pfm_correct / pfm_matches if pfm_matches else 0.0,
        "lightglue_matches": lightglue_matches,
        "lightglue_correct": lightglue_correct,
        "lightglue_wrong": lightglue_wrong,
        "lightglue_precision": lightglue_correct / lightglue_matches if lightglue_matches else 0.0,
        "hybrid_matches": hybrid_matches,
        "hybrid_correct": hybrid_correct,
        "hybrid_wrong": hybrid_wrong,
        "hybrid_precision": hybrid_correct / hybrid_matches if hybrid_matches else 0.0,
        "correct_delta_vs_lightglue": hybrid_correct - lightglue_correct,
        "wrong_delta_vs_lightglue": hybrid_wrong - lightglue_wrong,
        "correct_delta_vs_pfm": hybrid_correct - pfm_correct,
        "wrong_delta_vs_pfm": hybrid_wrong - pfm_wrong,
    }


def _score_rule(
    grouped: dict[str, list[dict[str, str]]],
    rule: Rule,
    *,
    metric_cache_by_split: dict[str, _SplitMetricCache] | None = None,
    train_split: str,
    eval_split: str,
    allow_wrong_delta: int,
    min_correct_delta: int,
) -> dict[str, object]:
    item: dict[str, object] = {"gate": rule.name}
    valid = True
    for split in sorted(grouped):
        mask = rule.masks_by_split.get(split, 0)
        metrics = (
            metric_cache_by_split[split].summary_for_mask(mask)
            if metric_cache_by_split is not None
            else hybrid_summary_for_selected(grouped[split], mask)
        )
        for key, value in metrics.items():
            item[f"{split}_{key}"] = value
    for split in [train_split, eval_split]:
        wrong_delta = int(item.get(f"{split}_wrong_delta_vs_lightglue", 0))
        correct_delta = int(item.get(f"{split}_correct_delta_vs_lightglue", 0))
        if wrong_delta > allow_wrong_delta or correct_delta < min_correct_delta:
            valid = False
    item["valid_train_eval"] = valid
    return item


def _output_fields(grouped: dict[str, list[dict[str, str]]]) -> list[str]:
    fields = ["gate", "valid_train_eval"]
    metric_fields = [
        "use_pfm_pairs",
        "fallback_lightglue_pairs",
        "pfm_correct",
        "pfm_wrong",
        "lightglue_correct",
        "lightglue_wrong",
        "hybrid_correct",
        "hybrid_wrong",
        "correct_delta_vs_lightglue",
        "wrong_delta_vs_lightglue",
        "hybrid_precision",
    ]
    for split in sorted(grouped):
        fields.extend(f"{split}_{field}" for field in metric_fields)
    return fields


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_html(path: Path, *, summary: dict[str, object], rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    table_rows = []
    for row in rows[:50]:
        cells = "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
        table_rows.append(f"<tr>{cells}</tr>")
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                "<title>Observable Pair Gate Sweep</title>",
                "<h1>Observable Pair Gate Sweep</h1>",
                f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))}</pre>",
                '<table border="1" cellspacing="0" cellpadding="4">',
                f"<tr>{header}</tr>",
                *table_rows,
                "</table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def sweep_observable_rules(
    *,
    dataset_csv: Path,
    output_dir: Path,
    train_split: str,
    eval_split: str,
    allow_wrong_delta: int = 0,
    min_correct_delta: int = 1,
    max_thresholds: int = 36,
) -> dict[str, object]:
    rows = read_csv_rows(dataset_csv)
    if not rows:
        raise ValueError(f"dataset is empty: {dataset_csv}")
    grouped = group_rows_by_split(rows)
    for split in [train_split, eval_split]:
        if split not in grouped:
            raise ValueError(f"split {split!r} not found in {dataset_csv}")

    base_rules = build_base_rules(grouped, max_thresholds=max_thresholds)
    rules = build_pairwise_and_rules(base_rules)
    metric_cache_by_split = {
        split: _SplitMetricCache.from_rows(split_rows)
        for split, split_rows in grouped.items()
    }
    scored = [
        _score_rule(
            grouped,
            rule,
            metric_cache_by_split=metric_cache_by_split,
            train_split=train_split,
            eval_split=eval_split,
            allow_wrong_delta=allow_wrong_delta,
            min_correct_delta=min_correct_delta,
        )
        for rule in rules
    ]
    valid = [row for row in scored if row["valid_train_eval"]]
    valid_sorted = sorted(
        valid,
        key=lambda row: (
            int(row.get(f"{eval_split}_correct_delta_vs_lightglue", -10**9)),
            int(row.get(f"{train_split}_correct_delta_vs_lightglue", -10**9)),
            -int(row.get(f"{eval_split}_wrong_delta_vs_lightglue", 10**9)),
        ),
        reverse=True,
    )
    top_sorted = sorted(
        scored,
        key=lambda row: (
            int(row.get(f"{eval_split}_wrong_delta_vs_lightglue", 10**9)) <= allow_wrong_delta,
            int(row.get(f"{eval_split}_correct_delta_vs_lightglue", -10**9)),
            -int(row.get(f"{eval_split}_wrong_delta_vs_lightglue", 10**9)),
        ),
        reverse=True,
    )

    summary: dict[str, object] = {
        "dataset_csv": str(dataset_csv),
        "train_split": train_split,
        "eval_split": eval_split,
        "allow_wrong_delta": allow_wrong_delta,
        "min_correct_delta": min_correct_delta,
        "max_thresholds": max_thresholds,
        "base_rule_count": len(base_rules),
        "scored_rule_count": len(scored),
        "valid_rule_count": len(valid_sorted),
        "best_valid": valid_sorted[0] if valid_sorted else None,
        "best_eval_any": top_sorted[0] if top_sorted else None,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = _output_fields(grouped)
    _write_csv(output_dir / "valid_rules.csv", valid_sorted[:500], fields)
    _write_csv(output_dir / "top_rules.csv", top_sorted[:500], fields)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_html(output_dir / "index.html", summary=summary, rows=valid_sorted, fields=fields)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="dev_train")
    parser.add_argument("--eval-split", default="dev_val")
    parser.add_argument("--max-thresholds", type=int, default=36)
    parser.add_argument("--allow-wrong-delta", type=int, default=0)
    parser.add_argument("--min-correct-delta", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = sweep_observable_rules(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        train_split=args.train_split,
        eval_split=args.eval_split,
        allow_wrong_delta=args.allow_wrong_delta,
        min_correct_delta=args.min_correct_delta,
        max_thresholds=args.max_thresholds,
    )
    print(
        "observable_pair_gate_sweep "
        f"base_rules={summary['base_rule_count']} "
        f"scored_rules={summary['scored_rule_count']} "
        f"valid_rules={summary['valid_rule_count']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
