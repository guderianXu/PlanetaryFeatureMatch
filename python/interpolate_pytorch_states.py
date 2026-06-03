#!/usr/bin/env python3
"""Interpolate two PyTorch PFM checkpoints with identical architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import pfm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linearly interpolate two PyTorch PFM state checkpoints")
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    first = torch.load(str(args.first), map_location="cpu", weights_only=False)
    second = torch.load(str(args.second), map_location="cpu", weights_only=False)
    first["source_checkpoint"] = str(args.first)
    second["source_checkpoint"] = str(args.second)
    mixed = pfm_model.interpolate_pytorch_state_payloads(first, second, alpha=args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mixed, args.output)
    print(f"checkpoint={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
