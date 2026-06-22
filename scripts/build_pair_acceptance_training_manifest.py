#!/usr/bin/env python3
"""Merge pair manifests with rejection labels for pair-acceptance training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


ACCEPTANCE_FIELDS = [
    "pair_accept_label",
    "pair_accept_weight",
    "pair_accept_source_wrong",
    "pair_accept_source_precision",
]

PairKey = tuple[str, str, str, str]
LoosePairKey = tuple[str, str, str]


def _clean(value: str | None) -> str:
    return "" if value is None else value.strip()


def _first_nonempty(row: dict[str, str], fields: Sequence[str]) -> str:
    for field in fields:
        value = _clean(row.get(field, ""))
        if value:
            return value
    return ""


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        rows = [
            {key: ("" if value is None else value) for key, value in row.items() if key is not None}
            for row in reader
        ]
    return list(reader.fieldnames), rows


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = _clean(row.get(key, ""))
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float_value(row, key, float(default))))


def _pair_key(row: dict[str, str]) -> PairKey:
    return (
        _clean(row.get("split", "")),
        _clean(row.get("pair_index", "")),
        _first_nonempty(row, ("reference_base_id", "target_base_id")),
        _clean(row.get("target_variant", "")),
    )


def _loose_pair_key(row: dict[str, str]) -> LoosePairKey:
    return (
        _clean(row.get("pair_index", "")),
        _first_nonempty(row, ("reference_base_id", "target_base_id")),
        _clean(row.get("target_variant", "")),
    )


def _rejection_keys(row: dict[str, str]) -> list[PairKey]:
    split = _clean(row.get("split", ""))
    pair_index = _clean(row.get("pair_index", ""))
    target_variant = _clean(row.get("target_variant", ""))
    base_ids: list[str] = []
    for field in ("base_id", "reference_base_id", "target_base_id"):
        base_id = _clean(row.get(field, ""))
        if base_id and base_id not in base_ids:
            base_ids.append(base_id)
    if not base_ids:
        base_ids.append("")
    return [(split, pair_index, base_id, target_variant) for base_id in base_ids]


def _loose_rejection_keys(row: dict[str, str]) -> list[LoosePairKey]:
    pair_index = _clean(row.get("pair_index", ""))
    target_variant = _clean(row.get("target_variant", ""))
    base_ids: list[str] = []
    for field in ("base_id", "reference_base_id", "target_base_id"):
        base_id = _clean(row.get(field, ""))
        if base_id and base_id not in base_ids:
            base_ids.append(base_id)
    if not base_ids:
        base_ids.append("")
    return [(pair_index, base_id, target_variant) for base_id in base_ids]


def _build_rejection_indexes(
    rows: Sequence[dict[str, str]],
) -> tuple[dict[PairKey, dict[str, str]], dict[LoosePairKey, dict[str, str]]]:
    index: dict[PairKey, dict[str, str]] = {}
    loose_candidates: dict[LoosePairKey, list[dict[str, str]]] = {}
    for row in rows:
        for key in _rejection_keys(row):
            index.setdefault(key, row)
        for key in _loose_rejection_keys(row):
            loose_candidates.setdefault(key, []).append(row)
    loose_index = {key: values[0] for key, values in loose_candidates.items() if len(values) == 1}
    return index, loose_index


def _accept_label(row: dict[str, str]) -> str:
    if _clean(row.get("keep_label", "")) == "1":
        return "1"
    if _clean(row.get("reject_label", "")) == "1":
        return "0"
    return "1" if _float_value(row, "pfm_wrong") <= 1.0 else "0"


def build_pair_acceptance_rows(
    pair_rows: Sequence[dict[str, str]],
    rejection_rows: Sequence[dict[str, str]],
    *,
    accept_weight: float = 1.0,
    reject_weight: float = 3.0,
) -> list[dict[str, str]]:
    rejection_index, loose_rejection_index = _build_rejection_indexes(rejection_rows)
    output_rows: list[dict[str, str]] = []
    for pair_row in pair_rows:
        key = _pair_key(pair_row)
        rejection_row = rejection_index.get(key)
        if rejection_row is None:
            rejection_row = loose_rejection_index.get(_loose_pair_key(pair_row))
        if rejection_row is None:
            raise ValueError(f"missing rejection row for key {key}")

        label = _accept_label(rejection_row)
        output_row = dict(pair_row)
        output_row["pair_accept_label"] = label
        output_row["pair_accept_weight"] = f"{accept_weight if label == '1' else reject_weight:.6f}"
        output_row["pair_accept_source_wrong"] = str(_int_value(rejection_row, "pfm_wrong"))
        output_row["pair_accept_source_precision"] = f"{_float_value(rejection_row, 'pfm_precision'):.6f}"
        output_rows.append(output_row)
    return output_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append pair_accept_label and pair_accept_weight to a pair manifest."
    )
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rejection-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-weight", type=float, default=1.0)
    parser.add_argument("--reject-weight", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pair_fields, pair_rows = _read_csv(args.pair_manifest)
    _, rejection_rows = _read_csv(args.rejection_dataset)
    output_rows = build_pair_acceptance_rows(
        pair_rows,
        rejection_rows,
        accept_weight=args.accept_weight,
        reject_weight=args.reject_weight,
    )

    output_fields = list(pair_fields)
    for field in ACCEPTANCE_FIELDS:
        if field not in output_fields:
            output_fields.append(field)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
