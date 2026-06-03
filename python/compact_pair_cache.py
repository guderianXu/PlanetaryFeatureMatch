#!/usr/bin/env python3
"""Utilities for compact synthetic pair caches with de-duplicated views."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

COMPACT_PAIR_FORMAT = "pfm_compact_pair_v1"


def tensor_content_key(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    header = f"{tuple(cpu.shape)}|{cpu.dtype}|".encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def save_shared_image(tensor: torch.Tensor, image_store: Path) -> Path:
    key = tensor_content_key(tensor)
    out_path = image_store / f"{key}.pt"
    if out_path.exists():
        return out_path
    image_store.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    torch.save({"format": "pfm_shared_image_v1", "view": tensor.detach().cpu().contiguous()}, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path


def make_compact_pair_payload(
    *,
    pair_path: Path,
    image_a_path: Path,
    image_b_path: Path,
    warp_a_to_b: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    return {
        "format": COMPACT_PAIR_FORMAT,
        "image_a": os.path.relpath(image_a_path, pair_path.parent),
        "image_b": os.path.relpath(image_b_path, pair_path.parent),
        "warp_a_to_b": warp_a_to_b.detach().cpu().contiguous(),
        "valid_mask": valid_mask.detach().cpu().contiguous(),
    }


def is_compact_pair_payload(obj: object) -> bool:
    return isinstance(obj, dict) and obj.get("format") == COMPACT_PAIR_FORMAT


def load_shared_image(path: Path, *, device: str | torch.device = "cpu") -> torch.Tensor:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "view" not in payload:
        raise ValueError(f"{path} is not a PFM shared image cache")
    return payload["view"].to(device=device, dtype=torch.float32).contiguous()


def resolve_compact_image_path(pair_path: Path, value: str) -> Path:
    image_path = Path(value)
    if image_path.is_absolute():
        return image_path
    return (pair_path.parent / image_path).resolve(strict=False)

