#!/usr/bin/env python3
"""Validate the current fov76 active selector config against promotion evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ActiveSelectorValidation:
    valid: bool
    active_selector: str
    active_label: str
    active_score: dict[str, Any]
    backup_scores: dict[str, dict[str, Any]]
    errors: list[str] = field(default_factory=list)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _comparison(decision: dict[str, Any], context: str, split: str) -> dict[str, Any] | None:
    for item in decision.get("comparisons", []):
        if item.get("context") == context and item.get("split") == split:
            return item
    return None


def _total_score(decision: dict[str, Any]) -> dict[str, Any]:
    item = _comparison(decision, "formal_target_total", "all")
    if item is None:
        raise ValueError("missing formal_target_total/all comparison")
    return {
        "correct_delta": int(item.get("correct_delta", 0)),
        "wrong_delta": int(item.get("wrong_delta", 0)),
        "precision_delta": float(item.get("precision_delta", 0.0)),
    }


def _regression_guard_clean(decision: dict[str, Any]) -> bool:
    for split in ("val", "test"):
        item = _comparison(decision, "regression_guard", split)
        if item is None:
            return False
        if int(item.get("correct_delta", 0)) < 0:
            return False
        if int(item.get("wrong_delta", 0)) > 0:
            return False
        if float(item.get("precision_delta", 0.0)) < 0.0:
            return False
    return True


def _selector_config(metadata: dict[str, Any]) -> dict[str, Any]:
    if "dual_checkpoint_rescue" in metadata:
        nested = metadata["dual_checkpoint_rescue"]
        if isinstance(nested, dict):
            metadata = nested
    config = metadata.get("config", metadata)
    if not isinstance(config, dict):
        raise ValueError("selector metadata does not contain a config object")
    return config


def _candidate_score(candidate: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    decision_path = Path(candidate["decision_path"])
    metadata_path = Path(candidate["metadata_path"])
    if not decision_path.is_absolute():
        decision_path = config_dir / decision_path
    if not metadata_path.is_absolute():
        metadata_path = config_dir / metadata_path
    decision = load_json(decision_path)
    metadata = load_json(metadata_path)
    score = _total_score(decision)
    score.update(
        {
            "promote": bool(decision.get("promote")),
            "failed_reasons": list(decision.get("failed_reasons", [])),
            "regression_guard_clean": _regression_guard_clean(decision),
            "selector_config": _selector_config(metadata),
            "decision_path": str(decision_path),
            "metadata_path": str(metadata_path),
        }
    )
    return score


def _rank_key(score: dict[str, Any]) -> tuple[bool, int, int, float]:
    return (
        bool(score.get("promote")),
        -int(score.get("wrong_delta", 0)),
        int(score.get("correct_delta", 0)),
        float(score.get("precision_delta", 0.0)),
    )


def validate_active_selector_config(path: Path) -> ActiveSelectorValidation:
    config = load_json(path)
    config_dir = path.parent
    active_selector = str(config.get("active_selector", ""))
    active_label = str(config.get("active_label", ""))
    candidates = config.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return ActiveSelectorValidation(
            valid=False,
            active_selector=active_selector,
            active_label=active_label,
            active_score={},
            backup_scores={},
            errors=["config must contain non-empty candidates list"],
        )

    errors: list[str] = []
    scores: dict[str, dict[str, Any]] = {}
    labels: dict[str, str] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            errors.append("candidate entry must be an object")
            continue
        name = str(candidate.get("name", ""))
        label = str(candidate.get("label", ""))
        if not name:
            errors.append("candidate missing name")
            continue
        labels[name] = label
        try:
            scores[name] = _candidate_score(candidate, config_dir=config_dir)
        except Exception as exc:  # noqa: BLE001 - include path-specific validation detail.
            errors.append(f"{name}: {exc}")

    if active_selector not in scores:
        errors.append(f"active_selector not found in candidates: {active_selector}")
        active_score: dict[str, Any] = {}
    else:
        active_score = scores[active_selector]
        if not active_score.get("promote"):
            errors.append(f"active selector did not pass promotion gate: {active_selector}")
        if active_score.get("failed_reasons"):
            errors.append(f"active selector has failed reasons: {active_selector}")
        if not active_score.get("regression_guard_clean"):
            errors.append(f"active selector regression guard is not clean: {active_selector}")
        if active_label and labels.get(active_selector) != active_label:
            errors.append(
                f"active_label mismatch: config={active_label} candidate={labels.get(active_selector, '')}"
            )

    if scores and active_selector in scores:
        best_name, best_score = max(scores.items(), key=lambda item: _rank_key(item[1]))
        if best_name != active_selector:
            errors.append(
                f"active selector is not top-ranked by promotion score: active={active_selector} best={best_name}"
            )
        if _rank_key(active_score) != _rank_key(best_score):
            errors.append(
                f"active selector score {active_score} does not match best score {best_score}"
            )

    backup_scores = {name: score for name, score in scores.items() if name != active_selector}
    return ActiveSelectorValidation(
        valid=not errors,
        active_selector=active_selector,
        active_label=active_label,
        active_score=active_score,
        backup_scores=backup_scores,
        errors=errors,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_active_selector_config(args.config)
    payload = {
        "valid": result.valid,
        "active_selector": result.active_selector,
        "active_label": result.active_label,
        "active_score": result.active_score,
        "backup_scores": result.backup_scores,
        "errors": result.errors,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
