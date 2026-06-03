"""Named GraphMatcher evaluation profiles for PFM v2.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GraphMatcherModeConfig:
    name: str
    pytorch_state: str
    matcher_mode: str = "graph_matcher"
    graph_metadata_mode: str = "no_xy"
    descriptor_mode: str = "blend"
    keypoint_score_mode: str = "learned"
    texture_blend_weight: float = 0.35
    max_keypoints: int = 2048
    max_matches: int = 512
    texture_fraction: float = 0.4
    weak_texture_fraction: float = 0.4
    spatial_bins: int = 8
    keypoint_cell_cap: int = 48
    min_score: float = -1.0
    min_margin: float = 0.0
    graph_dustbin_delta: float = 0.0
    graph_acceptance_margin: float = 0.0
    graph_min_raw_score: float = -1.0
    graph_min_raw_margin: float = 0.0

    def resolved_state_path(self, root: Path | str = ".") -> Path:
        state = Path(self.pytorch_state)
        if state.is_absolute():
            return state
        return Path(root) / state

    def training_visual_report_args(self) -> list[str]:
        return [
            "--pytorch-state",
            self.pytorch_state,
            "--matcher-mode",
            self.matcher_mode,
            "--graph-metadata-mode",
            self.graph_metadata_mode,
            "--mode",
            self.descriptor_mode,
            "--keypoint-score-mode",
            self.keypoint_score_mode,
            "--texture-blend-weight",
            str(self.texture_blend_weight),
            "--max-keypoints",
            str(self.max_keypoints),
            "--max-matches",
            str(self.max_matches),
            "--texture-keypoint-fraction",
            str(self.texture_fraction),
            "--weak-texture-keypoint-fraction",
            str(self.weak_texture_fraction),
            "--keypoint-spatial-bins",
            str(self.spatial_bins),
            "--keypoint-cell-cap",
            str(self.keypoint_cell_cap),
            "--min-score",
            str(self.min_score),
            "--min-margin",
            str(self.min_margin),
            "--graph-dustbin-delta",
            str(self.graph_dustbin_delta),
            "--graph-acceptance-margin",
            str(self.graph_acceptance_margin),
            "--graph-min-raw-score",
            str(self.graph_min_raw_score),
            "--graph-min-raw-margin",
            str(self.graph_min_raw_margin),
        ]

    def report_args(self) -> list[str]:
        return self.training_visual_report_args()


_MODES: dict[str, GraphMatcherModeConfig] = {
    "high_precision": GraphMatcherModeConfig(
        name="high_precision",
        pytorch_state=(
            "runs/graphmatcher_no_xy_dustbin_v21full256_b1_1epoch_s512_nm128_eval512_20260530_164330/"
            "pytorch_pfm_state.pt"
        ),
    ),
    "balanced": GraphMatcherModeConfig(
        name="balanced",
        pytorch_state=(
            "runs/graphmatcher_hardnegdustbin_light_from_rawpreserve_v21full256_b1_600_s512_20260531_101916/"
            "pytorch_pfm_state.pt"
        ),
        graph_min_raw_score=0.4,
        graph_min_raw_margin=0.01,
    ),
}


def graph_matcher_mode_config(name: str) -> GraphMatcherModeConfig:
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in _MODES:
        available = ", ".join(sorted(_MODES))
        raise ValueError(f"unknown GraphMatcher mode {name!r}; available modes: {available}")
    return _MODES[normalized]


def graph_matcher_mode_names() -> tuple[str, ...]:
    return tuple(sorted(_MODES))
