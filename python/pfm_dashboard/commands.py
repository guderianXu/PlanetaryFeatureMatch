from __future__ import annotations

import html
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path


PLASCAN_PYTHON = Path("/home/xjw/.local/share/mamba/envs/plascan/bin/python")
GRAPH_INFERENCE_PRESETS = {
    "off": (-1.0, -1.0),
    "fast": (0.25, 0.85),
    "high_precision": (0.5, 0.85),
}


@dataclass(frozen=True)
class TrainingRequest:
    experiment_name: str
    backend: str
    cache_dirs: list[str]
    output_root: Path = Path("runs")
    validation_cache_dirs: list[str] = field(default_factory=list)
    init_checkpoint: str = ""
    device: str = "cuda"
    epochs: int = 1
    batch_size: int = 1
    resize: int = 512
    training_crop_size: int = 512
    samples_per_pair: int = 512
    learning_rate: float = 3.0e-5
    weight_decay: float = 1.0e-4
    profile: str = "full"
    full_v21: bool = True
    memory_cache_items: int = 64
    prefetch_batches: int = 4
    prefetch_workers: int = 2
    dataloader_workers: int = 2
    max_train_batches: int = 0
    pair_cache_limit: int = 0
    synthetic_loss_weight: float = 0.1
    graph_matcher_loss_weight: float = 1.0
    temperature: float = 0.07
    min_intensity: float = 0.01
    seed: int = 20260603
    graph_inference_preset: str = "fast"
    graph_min_accept_probability: float = -1.0


@dataclass(frozen=True)
class GeneratedRun:
    backend: str
    run_dir: Path
    script_path: Path
    html_path: Path
    script_text: str


def _quote(value: str | Path | int | float) -> str:
    return shlex.quote(str(value))


def _safe_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in name.strip())
    return cleaned or f"dashboard_run_{int(time.time())}"


def _unique_run_dir(root: Path, name: str) -> Path:
    candidate = root / name
    if not candidate.exists():
        return candidate
    suffix = time.strftime("%Y%m%d_%H%M%S")
    return root / f"{name}_{suffix}"


def _write_run_html(path: Path, request: TrainingRequest, backend: str, script_path: Path) -> None:
    cache_items = "".join(f"<li><code>{html.escape(cache)}</code></li>" for cache in request.cache_dirs)
    content = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>{html.escape(request.experiment_name)} {backend}</title></head>
<body>
<h1>{html.escape(request.experiment_name)} {backend}</h1>
<p>Created by PFM Lab Dashboard.</p>
<h2>Backend</h2><p>{html.escape(backend)}</p>
<h2>Cache dirs</h2><ul>{cache_items}</ul>
<h2>Script</h2><p><code>{html.escape(str(script_path))}</code></p>
<h2>Key parameters</h2>
<ul>
<li>device={html.escape(request.device)}</li>
<li>epochs={request.epochs}</li>
<li>batch_size={request.batch_size}</li>
<li>crop={request.training_crop_size}</li>
<li>resize={request.resize}</li>
<li>samples_per_pair={request.samples_per_pair}</li>
<li>learning_rate={request.learning_rate}</li>
<li>graph_inference_preset={html.escape(request.graph_inference_preset)}</li>
<li>graph_min_accept_probability={request.graph_min_accept_probability}</li>
</ul>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def _graph_inference_thresholds(preset: str) -> tuple[float, float]:
    try:
        return GRAPH_INFERENCE_PRESETS[preset]
    except KeyError as exc:
        allowed = ", ".join(sorted(GRAPH_INFERENCE_PRESETS))
        raise ValueError(f"graph_inference_preset must be one of: {allowed}") from exc


def build_python_training_script(request: TrainingRequest, run_dir: Path) -> str:
    width_prune_min_score, early_stop_min_confidence = _graph_inference_thresholds(request.graph_inference_preset)
    parts: list[str] = [
        _quote(PLASCAN_PYTHON),
        "-u",
        "python/pfm_pytorch_training.py",
        "--output-dir",
        _quote(run_dir),
        "--device",
        _quote(request.device),
        "--epochs",
        str(request.epochs),
        "--batch-pairs",
        str(request.batch_size),
        "--samples-per-pair",
        str(request.samples_per_pair),
        "--training-crop-size",
        str(request.training_crop_size),
        "--training-max-image-size",
        str(request.resize),
        "--learning-rate",
        str(request.learning_rate),
        "--teacher-weight",
        "0.0",
        "--synthetic-loss-weight",
        str(request.synthetic_loss_weight),
        "--hard-negative-weight",
        "0.0",
        "--diversity-weight",
        "0.0",
        "--graph-matcher-loss-weight",
        str(request.graph_matcher_loss_weight),
        "--graph-matcher-metadata-mode",
        "full",
        "--temperature",
        str(request.temperature),
        "--min-intensity",
        str(request.min_intensity),
        "--max-grad-norm",
        "1.0",
        "--skip-nonfinite-steps",
        "--memory-cache-items",
        str(request.memory_cache_items),
        "--prefetch-batches",
        str(request.prefetch_batches),
        "--prefetch-workers",
        str(request.prefetch_workers),
        "--epoch-shuffle-sampling",
        "--save-every-epoch",
        "--seed",
        str(request.seed),
    ]
    if request.init_checkpoint:
        parts.extend(["--init-pytorch-state", _quote(request.init_checkpoint)])
    else:
        parts.append("--init-random")
    for cache_dir in request.cache_dirs:
        parts.extend(["--cache-dir", _quote(cache_dir)])
    for cache_dir in request.validation_cache_dirs:
        parts.extend(["--validation-cache-dir", _quote(cache_dir)])
    if request.validation_cache_dirs:
        parts.extend(
            [
                "--generate-training-report",
                "--report-matcher-mode",
                "graph_matcher",
                "--report-graph-inference-preset",
                request.graph_inference_preset,
                "--report-graph-width-prune-min-score",
                str(width_prune_min_score),
                "--report-graph-early-stop-min-confidence",
                str(early_stop_min_confidence),
                "--report-graph-min-accept-probability",
                str(request.graph_min_accept_probability),
            ]
        )
    if request.max_train_batches > 0:
        parts.extend(["--steps", str(request.max_train_batches)])
    train_flags = [
        "--train-backbone",
        "--train-dual-fpn",
        "--train-sparse-context",
        "--train-geometry-head",
        "--train-blended-descriptors",
        "--train-texture-adapter",
        "--train-descriptor-fusion",
        "--train-quality-head",
        "--train-graph-matcher",
    ]
    parts.extend(train_flags)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "cd /home/xjw/code/deeplearning/PlanetaryFeatureMatch",
            "export PYTHONPATH=python:scripts",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            " ".join(parts),
        ]
    ) + "\n"


def build_cpp_training_script(request: TrainingRequest, run_dir: Path) -> str:
    checkpoint = run_dir / "model_final.pt"
    parts: list[str] = [
        "./build/pfm_cli",
        "train",
        "--checkpoint",
        _quote(checkpoint),
        "--device",
        _quote(request.device),
        "--training-profile",
        _quote(request.profile),
        "--epochs",
        str(request.epochs),
        "--batch-size",
        str(request.batch_size),
        "--resize",
        str(request.resize),
        "--training-crop-size",
        str(request.training_crop_size),
        "--samples-per-pair",
        str(request.samples_per_pair),
        "--synthetic-loss-weight",
        str(request.synthetic_loss_weight),
        "--graph-matcher-loss-weight",
        str(request.graph_matcher_loss_weight),
        "--temperature",
        str(request.temperature),
        "--learning-rate",
        str(request.learning_rate),
        "--weight-decay",
        str(request.weight_decay),
        "--memory-cache-items",
        str(request.memory_cache_items),
        "--dataloader-workers",
        str(request.dataloader_workers),
        "--prefetch-batches",
        str(request.prefetch_batches),
        "--min-keypoint-intensity",
        str(request.min_intensity),
        "--log-csv",
        _quote(run_dir / "metrics.csv"),
    ]
    if request.profile == "python-compare":
        parts.extend(["--min-learning-rate-ratio", "1.0"])
    if request.full_v21:
        parts.append("--full-v21")
    if request.init_checkpoint:
        parts.extend(["--init-checkpoint", _quote(request.init_checkpoint)])
    if request.max_train_batches > 0:
        parts.extend(["--max-train-batches", str(request.max_train_batches)])
    if request.pair_cache_limit > 0:
        parts.extend(["--pair-cache-limit", str(request.pair_cache_limit)])
    train_flags = [
        "--train-backbone",
        "--train-dual-fpn",
        "--train-sparse-context",
        "--train-geometry-head",
        "--train-blended-descriptors",
        "--train-texture-adapter",
        "--train-descriptor-fusion",
        "--train-quality-head",
        "--train-graph-matcher",
    ]
    parts.extend(train_flags)
    for cache_dir in request.cache_dirs:
        parts.extend(["--pair-cache-dir", _quote(cache_dir)])
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "cd /home/xjw/code/deeplearning/PlanetaryFeatureMatch",
            " ".join(parts),
        ]
    ) + "\n"


def create_training_runs(request: TrainingRequest) -> list[GeneratedRun]:
    if not request.cache_dirs:
        raise ValueError("at least one cache dir is required")
    _graph_inference_thresholds(request.graph_inference_preset)
    if request.graph_min_accept_probability < -1.0 or request.graph_min_accept_probability > 1.0:
        raise ValueError("graph_min_accept_probability must be in [-1, 1]")
    backends = [request.backend]
    if any(backend not in {"python", "cpp"} for backend in backends):
        raise ValueError("backend must be python or cpp")
    request.output_root.mkdir(parents=True, exist_ok=True)
    generated: list[GeneratedRun] = []
    for backend in backends:
        run_name = _safe_name(request.experiment_name)
        run_dir = _unique_run_dir(request.output_root, run_name)
        run_dir.mkdir(parents=True)
        script_text = build_python_training_script(request, run_dir) if backend == "python" else build_cpp_training_script(request, run_dir)
        script_path = run_dir / "train.sh"
        script_path.write_text(script_text, encoding="utf-8")
        script_path.chmod(0o755)
        html_path = run_dir / "run.html"
        _write_run_html(html_path, request, backend, script_path)
        generated.append(
            GeneratedRun(
                backend=backend,
                run_dir=run_dir,
                script_path=script_path,
                html_path=html_path,
                script_text=script_text,
            )
        )
    return generated


def start_generated_run(run: GeneratedRun) -> int:
    pid = os.fork()
    if pid == 0:
        os.setsid()
        log_path = run.run_dir / "train.log"
        with log_path.open("ab", buffering=0) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            os.execv("/bin/bash", ["/bin/bash", str(run.script_path)])
    (run.run_dir / "train.pid").write_text(str(pid), encoding="utf-8")
    return pid
