from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
EXCLUDED_DIRS = {"annotations", "__pycache__"}


@dataclass(frozen=True)
class ImagePair:
    pair_id: str
    name: str
    image_a: str
    image_b: str
    size_a: tuple[int, int]
    size_b: tuple[int, int]
    annotation_path: str
    annotation_count: int


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def require_inside_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("path escapes annotation root") from exc
    return candidate


def stable_pair_id(image_a: str, image_b: str, preferred_name: str = "") -> str:
    digest = hashlib.sha1(f"{image_a}\n{image_b}".encode("utf-8")).hexdigest()[:10]
    stem = preferred_name.strip().replace("\\", "/").split("/")[-1]
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._-")
    if not stem:
        stem = Path(image_a).stem
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._-") or "pair"
    return f"{stem}_{digest}"


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def annotation_file(root: Path, pair_id: str) -> Path:
    safe_id = re.sub(r"[^0-9A-Za-z._-]+", "_", pair_id).strip("._-") or "pair"
    return root / "annotations" / f"{safe_id}.json"


def annotation_count(root: Path, pair_id: str) -> int:
    path = annotation_file(root, pair_id)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    matches = data.get("matches", [])
    return len(matches) if isinstance(matches, list) else 0


def default_pair_name(image_a: str, image_b: str) -> str:
    path_a = Path(image_a)
    path_b = Path(image_b)
    if path_a.parent == path_b.parent and str(path_a.parent) not in {"", "."}:
        return path_a.parent.name
    return f"{path_a.stem}_{path_b.stem}"


def make_image_pair(
    root: Path,
    image_a: str,
    image_b: str,
    pair_id: str | None = None,
    name: str | None = None,
) -> ImagePair:
    root = root.resolve()
    path_a = require_inside_root(root, image_a)
    path_b = require_inside_root(root, image_b)
    if path_a == path_b:
        raise ValueError("left and right images must be different")
    if not is_image_file(path_a):
        raise ValueError(f"not an image file: {image_a}")
    if not is_image_file(path_b):
        raise ValueError(f"not an image file: {image_b}")

    rel_a = relative_path(root, path_a)
    rel_b = relative_path(root, path_b)
    pair_name = name or default_pair_name(rel_a, rel_b)
    final_pair_id = pair_id or stable_pair_id(rel_a, rel_b, pair_name)
    ann = annotation_file(root, final_pair_id)
    return ImagePair(
        pair_id=final_pair_id,
        name=pair_name,
        image_a=rel_a,
        image_b=rel_b,
        size_a=image_size(path_a),
        size_b=image_size(path_b),
        annotation_path=relative_path(root, ann),
        annotation_count=annotation_count(root, final_pair_id),
    )


def side_token(path: Path) -> str | None:
    name = path.stem.lower()
    tokens = [token for token in re.split(r"[^0-9a-z]+", name) if token]
    if not tokens:
        return None
    token = tokens[-1]
    if token in {"a", "left", "l", "0", "src", "source", "ref", "fixed"}:
        return "a"
    if token in {"b", "right", "r", "1", "dst", "target", "moving"}:
        return "b"
    for suffix, side in (
        ("_a", "a"),
        ("-a", "a"),
        (".a", "a"),
        ("_b", "b"),
        ("-b", "b"),
        (".b", "b"),
        ("_left", "a"),
        ("_right", "b"),
    ):
        if name.endswith(suffix):
            return side
    return None


def pair_group_key(path: Path) -> str | None:
    side = side_token(path)
    if side is None:
        return None
    name = path.stem.lower()
    stripped = re.sub(
        r"([_. -]?(a|b|left|right|l|r|0|1|src|source|ref|fixed|dst|target|moving))$",
        "",
        name,
    )
    if stripped == name:
        return None
    return f"{path.parent.as_posix()}/{stripped}"


def choose_pair_images(files: list[Path]) -> tuple[Path, Path]:
    by_side: dict[str, list[Path]] = {"a": [], "b": []}
    for path in files:
        side = side_token(path)
        if side in by_side:
            by_side[side].append(path)
    if by_side["a"] and by_side["b"]:
        return sorted(by_side["a"])[0], sorted(by_side["b"])[0]
    ordered = sorted(files)
    return ordered[0], ordered[1]


def read_manifest_pairs(root: Path) -> list[tuple[str, str, str, str]]:
    manifest = root / "pairs.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    items = data.get("pairs", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("pairs.json must contain a list or an object with a 'pairs' list")
    pairs: list[tuple[str, str, str, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("each pairs.json entry must be an object")
        image_a = item.get("image_a") or item.get("a")
        image_b = item.get("image_b") or item.get("b")
        if not isinstance(image_a, str) or not isinstance(image_b, str):
            raise ValueError("each pairs.json entry needs image_a/image_b")
        require_inside_root(root, image_a)
        require_inside_root(root, image_b)
        pair_id = str(item.get("id") or stable_pair_id(image_a, image_b, f"manifest_{index:04d}"))
        name = str(item.get("name") or pair_id)
        pairs.append((pair_id, name, image_a, image_b))
    return pairs


def read_annotation_pairs(root: Path) -> list[tuple[str, str, str, str]]:
    pairs: list[tuple[str, str, str, str]] = []
    for path in sorted((root / "annotations").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        image_a = data.get("image_a")
        image_b = data.get("image_b")
        if not isinstance(image_a, str) or not isinstance(image_b, str):
            continue
        pair_id = str(data.get("pair_id") or data.get("id") or path.stem)
        name = str(data.get("name") or default_pair_name(image_a, image_b))
        pairs.append((pair_id, name, image_a, image_b))
    return pairs


def candidate_image_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in EXCLUDED_DIRS or part.startswith(".") for part in path.relative_to(root).parts[:-1]):
            continue
        if is_image_file(path):
            files.append(path)
    return sorted(files)


def discover_pair_paths(root: Path) -> list[tuple[str, str, str, str]]:
    manifest_pairs = read_manifest_pairs(root)
    if manifest_pairs:
        seen = {(image_a, image_b) for _, _, image_a, image_b in manifest_pairs}
        pairs = list(manifest_pairs)
        for pair_id, name, rel_a, rel_b in read_annotation_pairs(root):
            pair_key = (rel_a, rel_b)
            if pair_key in seen:
                continue
            pairs.append((pair_id, name, rel_a, rel_b))
            seen.add(pair_key)
        return pairs

    files = candidate_image_files(root)
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, str, str]] = []

    for directory in sorted({path.parent for path in files}):
        direct_images = sorted(path for path in files if path.parent == directory)
        if len(direct_images) < 2:
            continue
        image_a, image_b = choose_pair_images(direct_images)
        rel_a = relative_path(root, image_a)
        rel_b = relative_path(root, image_b)
        key = (rel_a, rel_b)
        if key not in seen:
            rel_dir = relative_path(root, directory) if directory != root else "root"
            pair_id = stable_pair_id(rel_a, rel_b, rel_dir)
            pairs.append((pair_id, rel_dir, rel_a, rel_b))
            seen.add(key)

    grouped: dict[str, dict[str, Path]] = {}
    for path in files:
        key = pair_group_key(path)
        side = side_token(path)
        if key is None or side is None:
            continue
        grouped.setdefault(key, {})[side] = path
    for key, sides in sorted(grouped.items()):
        if "a" not in sides or "b" not in sides:
            continue
        rel_a = relative_path(root, sides["a"])
        rel_b = relative_path(root, sides["b"])
        pair_key = (rel_a, rel_b)
        if pair_key in seen:
            continue
        name = Path(key).name or "pair"
        pairs.append((stable_pair_id(rel_a, rel_b, name), name, rel_a, rel_b))
        seen.add(pair_key)

    for pair_id, name, rel_a, rel_b in read_annotation_pairs(root):
        pair_key = (rel_a, rel_b)
        if pair_key in seen:
            continue
        pairs.append((pair_id, name, rel_a, rel_b))
        seen.add(pair_key)

    if pairs:
        return pairs

    for index in range(0, len(files) - 1, 2):
        image_a = files[index]
        image_b = files[index + 1]
        rel_a = relative_path(root, image_a)
        rel_b = relative_path(root, image_b)
        name = f"pair_{index // 2:04d}"
        pairs.append((stable_pair_id(rel_a, rel_b, name), name, rel_a, rel_b))
    return pairs


def discover_pairs(root: Path) -> list[ImagePair]:
    root = root.resolve()
    result: list[ImagePair] = []
    for pair_id, name, rel_a, rel_b in discover_pair_paths(root):
        path_a = require_inside_root(root, rel_a)
        path_b = require_inside_root(root, rel_b)
        if not is_image_file(path_a) or not is_image_file(path_b):
            continue
        ann = annotation_file(root, pair_id)
        result.append(
            ImagePair(
                pair_id=pair_id,
                name=name,
                image_a=rel_a,
                image_b=rel_b,
                size_a=image_size(path_a),
                size_b=image_size(path_b),
                annotation_path=relative_path(root, ann),
                annotation_count=annotation_count(root, pair_id),
            )
        )
    return sorted(result, key=lambda pair: pair.name)


def scale_to_uint8(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    valid = values[finite]
    low, high = np.percentile(valid, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(valid.min())
        high = float(valid.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def to_display_image(source: Image.Image, max_side: int) -> Image.Image:
    image = ImageOps.exif_transpose(source)
    array = np.asarray(image)
    if array.ndim == 2:
        display = Image.fromarray(scale_to_uint8(array), mode="L")
    elif array.ndim == 3:
        channels = array[:, :, :3]
        if channels.dtype == np.uint8:
            display = Image.fromarray(channels, mode="RGB")
        else:
            scaled_channels = [scale_to_uint8(channels[:, :, index]) for index in range(channels.shape[2])]
            while len(scaled_channels) < 3:
                scaled_channels.append(scaled_channels[-1])
            display = Image.fromarray(np.stack(scaled_channels[:3], axis=2), mode="RGB")
    else:
        display = image.convert("L")

    if max_side > 0 and max(display.size) > max_side:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        display.thumbnail((max_side, max_side), resampling)
    return display


def clean_match(match: Any) -> dict[str, Any]:
    if not isinstance(match, dict):
        raise ValueError("match entries must be objects")
    point_a = match.get("a")
    point_b = match.get("b")
    if not isinstance(point_a, dict) or not isinstance(point_b, dict):
        raise ValueError("match entries need a and b points")
    clean: dict[str, Any] = {
        "id": int(match.get("id", 0)),
        "a": {"x": float(point_a["x"]), "y": float(point_a["y"])},
        "b": {"x": float(point_b["x"]), "y": float(point_b["y"])},
        "label": str(match.get("label", "match")),
    }
    return clean


def save_annotation(root: Path, payload: dict[str, Any]) -> Path:
    pair_id = payload.get("pair_id") or payload.get("id")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("annotation payload needs pair_id")
    image_a = payload.get("image_a")
    image_b = payload.get("image_b")
    if not isinstance(image_a, str) or not isinstance(image_b, str):
        raise ValueError("annotation payload needs image_a/image_b")
    require_inside_root(root, image_a)
    require_inside_root(root, image_b)
    matches = [clean_match(match) for match in payload.get("matches", [])]
    output = {
        "version": 1,
        "pair_id": pair_id,
        "image_a": image_a,
        "image_b": image_b,
        "image_size_a": list(image_size(require_inside_root(root, image_a))),
        "image_size_b": list(image_size(require_inside_root(root, image_b))),
        "matches": matches,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = annotation_file(root, pair_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_annotation(root: Path, pair_id: str) -> dict[str, Any]:
    path = annotation_file(root, pair_id)
    if not path.exists():
        return {"version": 1, "pair_id": pair_id, "matches": []}
    return json.loads(path.read_text(encoding="utf-8"))


def export_annotations(root: Path) -> dict[str, Any]:
    annotations = []
    for path in sorted((root / "annotations").glob("*.json")):
        try:
            annotations.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return {"version": 1, "count": len(annotations), "annotations": annotations}
