#!/usr/bin/env python3
"""Export a Python PFM training state into the LibTorch archive layout used by C++."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn


class ArchiveNode(nn.Module):
    def forward(self) -> torch.Tensor:
        return torch.empty(0)


def _as_int(config: dict[str, Any], name: str, fallback: int | None = None) -> int:
    if name in config:
        return int(config[name])
    if fallback is not None:
        return int(fallback)
    raise KeyError(f"missing config field: {name}")


def _child(node: ArchiveNode, name: str) -> ArchiveNode:
    existing = node._modules.get(name)
    if existing is not None:
        if not isinstance(existing, ArchiveNode):
            raise TypeError(f"archive path component is not an ArchiveNode: {name}")
        return existing
    created = ArchiveNode()
    node.add_module(name, created)
    return created


def _register_nested_tensor(root: ArchiveNode, key: str, tensor: torch.Tensor) -> None:
    parts = key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid state key: {key}")
    node = root
    for part in parts[:-1]:
        node = _child(node, part)
    node.register_buffer(parts[-1], tensor.detach().cpu().clone())


def _ensure_module_path(root: ArchiveNode, key: str) -> None:
    node = root
    for part in key.split("."):
        if part:
            node = _child(node, part)


def _add_config(root: ArchiveNode, config: dict[str, Any]) -> None:
    descriptor_dim = _as_int(config, "descriptor_dim")
    config_node = _child(root, "config")
    values = {
        "checkpoint_version": 3,
        "base_channels": _as_int(config, "base_channels"),
        "descriptor_dim": descriptor_dim,
        "graph_hidden_dim": _as_int(config, "graph_hidden_dim", max(32, descriptor_dim)),
        "graph_attention_layers": _as_int(config, "graph_attention_layers", 1),
        "graph_keypoint_meta_dim": _as_int(config, "graph_keypoint_meta_dim", 16),
        "seed": _as_int(config, "seed", 0),
        "training_profile_id": _as_int(config, "training_profile_id", 0),
        "input_channels": _as_int(config, "input_channels"),
    }
    for name, value in values.items():
        config_node.register_buffer(name, torch.tensor([value], dtype=torch.int64))


def _add_model_module_tree(root: ArchiveNode, config: dict[str, Any]) -> None:
    import pfm_model

    descriptor_dim = _as_int(config, "descriptor_dim")
    model = pfm_model.PlanetaryFeatureMatcher(
        input_channels=_as_int(config, "input_channels"),
        base_channels=_as_int(config, "base_channels"),
        descriptor_dim=descriptor_dim,
        graph_hidden_dim=_as_int(config, "graph_hidden_dim", max(32, descriptor_dim)),
        graph_attention_layers=_as_int(config, "graph_attention_layers", 1),
        graph_keypoint_meta_dim=_as_int(config, "graph_keypoint_meta_dim", 16),
    )
    for name, _ in model.named_modules():
        _ensure_module_path(root, name)


def make_libtorch_archive_module(payload: dict[str, Any]) -> ArchiveNode:
    if "config" not in payload or "model" not in payload:
        raise KeyError("pytorch state must contain 'config' and 'model'")
    root = ArchiveNode()
    config = dict(payload["config"])
    _add_config(root, config)
    _add_model_module_tree(root, config)
    for key, tensor in payload["model"].items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model state entry is not a tensor: {key}")
        _register_nested_tensor(root, key, tensor)
    return root


def export_pytorch_state_to_libtorch(source: Path | str, output: Path | str) -> None:
    source_path = Path(source)
    output_path = Path(output)
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    archive = make_libtorch_archive_module(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(archive).save(str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-state", required=True, type=Path, help="Python pytorch_pfm_state.pt path")
    parser.add_argument("--output", required=True, type=Path, help="Output LibTorch checkpoint path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_pytorch_state_to_libtorch(args.pytorch_state, args.output)
    print(f"exported_libtorch_checkpoint={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
