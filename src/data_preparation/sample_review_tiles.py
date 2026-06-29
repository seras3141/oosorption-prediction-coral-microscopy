"""Sample blinded tile-review sessions from cut-local TIFFs.

This module implements the backend for the collaborator-facing review tool.
It samples balanced positive and negative tiles across requested resolutions,
exports opaque PNG filenames, and writes a session manifest that the local
review app can update in place.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from src.data_preparation.extract_ndpi_cuts import create_tissue_mask
from src.data_preparation.generate_tiles import (
    _coerce_rgb_array,
    _compute_tile_grid,
    _level0_shape,
    _open_level0_array,
    _pad_tile_array,
    _read_json,
    _read_tile_region,
    _resolve_repo_path,
    _write_tile_png,
)
from src.data_preparation.remap_annotations import _compute_centroid, _extract_ring_coords

LOG = logging.getLogger(__name__)

DEFAULT_TILE_SIZES: tuple[int, ...] = (128, 256, 512, 1024)
DEFAULT_N_POSITIVE_PER_SIZE: int = 5
DEFAULT_N_NEGATIVE_PER_SIZE: int = 5
DEFAULT_MIN_TISSUE_FRACTION: float = 0.20
DEFAULT_NEGATIVE_BUFFER_FRACTION: float = 0.10
DEFAULT_SEED: int = 42
SESSION_PREFIX: str = "review"
SESSION_COUNTER_WIDTH: int = 3
PNG_COUNTER_WIDTH: int = 3
MAX_DISPLAY_SHUFFLE_ATTEMPTS: int = 512
FEATURE_COLLECTION: str = "FeatureCollection"
SESSION_META_FILENAME: str = "session_meta.json"
LABELS_FILENAME: str = "labels.json"


def sample_review_session(
    cuts_dir: str | Path,
    output_dir: str | Path,
    *,
    session_id: str | None = None,
    tile_sizes: tuple[int, ...] = DEFAULT_TILE_SIZES,
    n_positive_per_size: int = DEFAULT_N_POSITIVE_PER_SIZE,
    n_negative_per_size: int = DEFAULT_N_NEGATIVE_PER_SIZE,
    min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION,
    negative_buffer_fraction: float = DEFAULT_NEGATIVE_BUFFER_FRACTION,
    seed: int | None = DEFAULT_SEED,
) -> dict[str, Any]:
    """Sample a balanced tile review session from pyramidal cut TIFFs.

    For each tile size, samples ``n_positive_per_size`` tiles centred on
    annotated oocyte centroids and ``n_negative_per_size`` tiles from
    unannotated tissue regions. Exports PNG images and writes
    ``session_meta.json`` and ``labels.json`` to ``output_dir / session_id``.

    Parameters
    ----------
    cuts_dir : str or Path
        Root of the cut-local output tree. Must contain per-stem
        subdirectories with ``*_cut*.tif`` pyramidal TIFFs and
        ``*_cut*_annotations.geojson`` files.
    output_dir : str or Path
        Parent directory for review session folders.
    session_id : str or None, optional
        Identifier string for the output session. Defaults to a
        date-based auto-incremented identifier.
    tile_sizes : tuple of int, optional
        Tile edge lengths in level-0 pixels.
    n_positive_per_size : int, optional
        Number of positive tiles per tile size.
    n_negative_per_size : int, optional
        Number of negative tiles per tile size.
    min_tissue_fraction : float, optional
        Minimum saturation-mask tissue fraction required for acceptance.
    negative_buffer_fraction : float, optional
        Reject negative tiles when any annotation centroid falls within this
        fraction of one tile width outside the tile boundary.
    seed : int or None, optional
        Random seed for reproducibility. ``None`` uses system entropy.

    Returns
    -------
    dict of str to Any
        In-memory session metadata dict. A ``labels`` key is included for CLI
        summaries, but the written ``session_meta.json`` keeps labels separate.

    Examples
    --------
    >>> manifest = {"n_tiles_per_size": 10, "tile_sizes": [128, 256]}
    >>> len(manifest["tile_sizes"]) == 2
    True
    """
    cuts_dir = _resolve_repo_path(cuts_dir)
    output_dir = _resolve_repo_path(output_dir)

    _validate_sampling_inputs(
        cuts_dir=cuts_dir,
        tile_sizes=tile_sizes,
        n_positive_per_size=n_positive_per_size,
        n_negative_per_size=n_negative_per_size,
        min_tissue_fraction=min_tissue_fraction,
        negative_buffer_fraction=negative_buffer_fraction,
    )

    rng = random.Random(seed)
    cut_sources = _discover_cut_sources(cuts_dir)
    positive_candidates = _collect_positive_candidates(cut_sources)

    tiles: list[dict[str, Any]] = []
    for tile_size in tile_sizes:
        size_positive_tiles = _sample_positive_tiles(
            positive_candidates,
            tile_size=tile_size,
            n_required=n_positive_per_size,
            min_tissue_fraction=min_tissue_fraction,
            rng=rng,
        )
        size_negative_tiles = _collect_negative_candidates(
            cut_sources,
            tile_size=tile_size,
            n_required=n_negative_per_size,
            min_tissue_fraction=min_tissue_fraction,
            buffer_fraction=negative_buffer_fraction,
            rng=rng,
        )
        tiles.extend(size_positive_tiles)
        tiles.extend(size_negative_tiles)

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_session_id = _resolve_session_id(output_dir, session_id=session_id)
    session_dir = output_dir / resolved_session_id
    session_dir.mkdir(parents=True, exist_ok=False)

    ordered_tiles = _assign_png_filenames(tiles, rng)
    for tile in ordered_tiles:
        png_rel_path = Path(tile["png_filename"])
        png_abs_path = session_dir / png_rel_path
        _write_tile_png(tile["tile_array"], png_abs_path)
        tile.pop("tile_array")

    manifest = {
        "session_id": resolved_session_id,
        "created_at": _timestamp_now(),
        "n_tiles_per_size": n_positive_per_size + n_negative_per_size,
        "n_positive_per_size": n_positive_per_size,
        "n_negative_per_size": n_negative_per_size,
        "tile_sizes": list(tile_sizes),
        "seed": seed,
        "tiles": ordered_tiles,
    }

    _write_session(session_dir, manifest)
    LOG.info(
        "Prepared review session %s with %d tiles at %s",
        resolved_session_id,
        len(ordered_tiles),
        _display_path(session_dir),
    )
    return manifest


def _discover_cut_sources(cuts_dir: Path) -> list[dict[str, Any]]:
    """Return cut-local TIFF and annotation sources available for review sampling."""
    cut_sources: list[dict[str, Any]] = []
    for stem_dir in sorted(path for path in cuts_dir.iterdir() if path.is_dir()):
        for annotations_path in sorted(stem_dir.glob("*_cut[0-9][0-9][0-9]_annotations.geojson")):
            cut_name = annotations_path.stem[: -len("_annotations")]
            cut_tiff_path = stem_dir / f"{cut_name}.tif"
            if not cut_tiff_path.exists():
                LOG.warning(
                    "Skipping %s because the sibling TIFF is missing: %s",
                    annotations_path.name,
                    cut_tiff_path,
                )
                continue
            annotations = _load_cut_annotations(annotations_path)
            if not annotations:
                LOG.warning("Skipping %s because it contains no supported annotations", annotations_path)
                continue
            cut_w, cut_h = _read_cut_dimensions(cut_tiff_path)
            cut_sources.append(
                {
                    "stem": stem_dir.name,
                    "cut_name": cut_name,
                    "cut_tiff_path": cut_tiff_path,
                    "annotations_path": annotations_path,
                    "annotations": annotations,
                    "cut_w": cut_w,
                    "cut_h": cut_h,
                }
            )

    if not cut_sources:
        raise ValueError(f"No cut TIFF + annotation pairs found in {cuts_dir}")
    return cut_sources


def _collect_positive_candidates(cut_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one positive sampling candidate per available annotation centroid."""
    candidates: list[dict[str, Any]] = []
    for cut_source in cut_sources:
        for annotation in cut_source["annotations"]:
            candidates.append(
                {
                    "stem": cut_source["stem"],
                    "cut_name": cut_source["cut_name"],
                    "cut_tiff_path": cut_source["cut_tiff_path"],
                    "annotations": cut_source["annotations"],
                    "annotation_id": annotation["annotation_id"],
                    "stage": annotation["stage"],
                    "cx": annotation["cx"],
                    "cy": annotation["cy"],
                    "cut_w": cut_source["cut_w"],
                    "cut_h": cut_source["cut_h"],
                }
            )
    if not candidates:
        raise ValueError("No positive annotation candidates available for review sampling")
    return candidates


def _sample_positive_tiles(
    candidates: list[dict[str, Any]],
    *,
    tile_size: int,
    n_required: int,
    min_tissue_fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample centred positive review tiles for one tile size."""
    shuffled_candidates = list(candidates)
    rng.shuffle(shuffled_candidates)

    accepted: list[dict[str, Any]] = []
    used_annotation_ids: set[str] = set()
    for candidate in shuffled_candidates:
        if candidate["annotation_id"] in used_annotation_ids:
            continue
        bbox = _centre_bbox_on_point(
            candidate["cx"],
            candidate["cy"],
            tile_size,
            candidate["cut_w"],
            candidate["cut_h"],
        )
        tile_arr = _extract_tile_array(
            candidate["cut_tiff_path"],
            bbox["x0"],
            bbox["y0"],
            tile_size,
        )
        tissue_fraction = _tissue_fraction(tile_arr)
        if tissue_fraction < min_tissue_fraction:
            continue

        assigned_annotations = _annotations_in_bbox(candidate["annotations"], bbox)
        if not assigned_annotations:
            continue

        used_annotation_ids.add(candidate["annotation_id"])
        accepted.append(
            {
                "tile_id": _build_review_tile_id(
                    candidate["cut_name"],
                    tile_size,
                    bbox["x0"],
                    bbox["y0"],
                ),
                "cut_name": candidate["cut_name"],
                "tile_size": tile_size,
                "cut_local_bbox": bbox,
                "tissue_fraction": round(tissue_fraction, 6),
                "ground_truth": True,
                "n_oocytes_ground_truth": len(assigned_annotations),
                "annotation_ids": [item["annotation_id"] for item in assigned_annotations],
                "tile_array": tile_arr,
            }
        )
        if len(accepted) == n_required:
            return accepted

    raise ValueError(
        f"Could only sample {len(accepted)} positive tiles for size {tile_size}; "
        f"required {n_required}"
    )


def _collect_negative_candidates(
    cut_sources: list[dict[str, Any]],
    *,
    tile_size: int,
    n_required: int,
    min_tissue_fraction: float,
    buffer_fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Return sampled negative review tiles for one tile size."""
    candidate_tiles: list[dict[str, Any]] = []
    cut_sources_shuffled = list(cut_sources)
    rng.shuffle(cut_sources_shuffled)
    buffer_px = tile_size * buffer_fraction

    for cut_source in cut_sources_shuffled:
        grid = _compute_tile_grid(
            cut_h=cut_source["cut_h"],
            cut_w=cut_source["cut_w"],
            tile_size=tile_size,
            stride=tile_size,
        )
        rng.shuffle(grid)
        with tifffile.TiffFile(cut_source["cut_tiff_path"]) as tif:
            level0, axes = _open_level0_array(tif)
            for tile_bbox in grid:
                bbox = {
                    "x0": tile_bbox["x0"],
                    "y0": tile_bbox["y0"],
                    "x1": tile_bbox["x1"],
                    "y1": tile_bbox["y1"],
                }
                if not _annotation_proximity_check(bbox, cut_source["annotations"], buffer_px):
                    continue

                tile_arr = _extract_tile_array_from_level0(
                    level0,
                    axes,
                    bbox,
                    tile_size=tile_size,
                )
                tissue_fraction = _tissue_fraction(tile_arr)
                if tissue_fraction < min_tissue_fraction:
                    continue

                candidate_tiles.append(
                    {
                        "tile_id": _build_review_tile_id(
                            cut_source["cut_name"],
                            tile_size,
                            bbox["x0"],
                            bbox["y0"],
                        ),
                        "cut_name": cut_source["cut_name"],
                        "tile_size": tile_size,
                        "cut_local_bbox": bbox,
                        "tissue_fraction": round(tissue_fraction, 6),
                        "ground_truth": False,
                        "n_oocytes_ground_truth": 0,
                        "annotation_ids": [],
                        "tile_array": tile_arr,
                    }
                )

    if len(candidate_tiles) < n_required:
        raise ValueError(
            f"Could only sample {len(candidate_tiles)} negative tiles for size {tile_size}; "
            f"required {n_required}"
        )

    rng.shuffle(candidate_tiles)
    return candidate_tiles[:n_required]


def _load_cut_annotations(annotations_path: Path) -> list[dict[str, Any]]:
    """Load cut-local annotations with centroid metadata for review sampling."""
    geojson = _read_json(annotations_path)
    if geojson.get("type") != FEATURE_COLLECTION:
        raise ValueError(f"{annotations_path} is not a GeoJSON FeatureCollection")

    annotations: list[dict[str, Any]] = []
    for feature_idx, feature in enumerate(geojson.get("features", [])):
        coords = _extract_ring_coords(feature.get("geometry") or {})
        if not coords:
            continue
        cx, cy = _compute_centroid(coords)
        annotations.append(
            {
                "annotation_id": str(feature.get("id", f"{annotations_path.stem}:{feature_idx}")),
                "coords": coords,
                "stage": _extract_stage(feature),
                "cx": cx,
                "cy": cy,
            }
        )
    return annotations


def _extract_tile_array(tiff_path: Path, x0: int, y0: int, tile_size: int) -> np.ndarray:
    """Read a square level-0 tile and zero-pad any cut-edge truncation."""
    with tifffile.TiffFile(tiff_path) as tif:
        level0, axes = _open_level0_array(tif)
        cut_h, cut_w = _level0_shape(level0, axes)
        bbox = {
            "x0": x0,
            "y0": y0,
            "x1": min(x0 + tile_size, cut_w),
            "y1": min(y0 + tile_size, cut_h),
        }
        return _extract_tile_array_from_level0(level0, axes, bbox, tile_size=tile_size)


def _extract_tile_array_from_level0(
    level0: Any,
    axes: str,
    bbox: dict[str, int],
    *,
    tile_size: int,
) -> np.ndarray:
    """Read a square tile from an already-open level-0 array."""
    tile_arr = _read_tile_region(level0, axes, bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
    tile_arr = _coerce_rgb_array(tile_arr, axes="YXS")
    return _pad_tile_array(
        tile_arr,
        pad_right=tile_size - (bbox["x1"] - bbox["x0"]),
        pad_bottom=tile_size - (bbox["y1"] - bbox["y0"]),
    )


def _centre_bbox_on_point(
    cx: float,
    cy: float,
    tile_size: int,
    cut_w: int,
    cut_h: int,
) -> dict[str, int]:
    """Return a cut-local tile bbox centred on a point and shifted in-bounds."""
    max_x0 = max(cut_w - tile_size, 0)
    max_y0 = max(cut_h - tile_size, 0)
    x0 = min(max(int(round(cx - tile_size / 2)), 0), max_x0)
    y0 = min(max(int(round(cy - tile_size / 2)), 0), max_y0)
    x1 = min(x0 + tile_size, cut_w)
    y1 = min(y0 + tile_size, cut_h)
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _annotation_proximity_check(
    tile_bbox: dict[str, int],
    annotations: list[dict[str, Any]],
    buffer_px: float,
) -> bool:
    """Return True when no annotation centroid falls inside or near a tile."""
    x0 = tile_bbox["x0"] - buffer_px
    y0 = tile_bbox["y0"] - buffer_px
    x1 = tile_bbox["x1"] + buffer_px
    y1 = tile_bbox["y1"] + buffer_px
    for annotation in annotations:
        if x0 <= annotation["cx"] <= x1 and y0 <= annotation["cy"] <= y1:
            return False
    return True


def _annotations_in_bbox(
    annotations: list[dict[str, Any]],
    bbox: dict[str, int],
) -> list[dict[str, Any]]:
    """Return annotations whose centroids fall within a cut-local bbox."""
    assigned: list[dict[str, Any]] = []
    for annotation in annotations:
        if bbox["x0"] <= annotation["cx"] <= bbox["x1"] and bbox["y0"] <= annotation["cy"] <= bbox["y1"]:
            assigned.append(annotation)
    return assigned


def _assign_png_filenames(
    tiles: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Assign opaque filenames and a randomized display order to sampled tiles."""
    if not tiles:
        raise ValueError("Cannot assign display order for an empty tile list")

    best_order = list(tiles)
    best_conflicts = _adjacent_conflict_count(best_order)
    for _attempt in range(MAX_DISPLAY_SHUFFLE_ATTEMPTS):
        candidate = list(tiles)
        rng.shuffle(candidate)
        conflicts = _adjacent_conflict_count(candidate)
        if conflicts < best_conflicts:
            best_order = candidate
            best_conflicts = conflicts
        if conflicts == 0:
            best_order = candidate
            break

    if best_conflicts:
        LOG.warning("Display order still contains %d adjacent conflicts", best_conflicts)

    assigned_tiles: list[dict[str, Any]] = []
    for index, tile in enumerate(best_order):
        tile_copy = dict(tile)
        tile_copy["display_index"] = index
        tile_copy["png_filename"] = f"tiles/tile_{index + 1:0{PNG_COUNTER_WIDTH}d}.png"
        assigned_tiles.append(tile_copy)
    return assigned_tiles


def _adjacent_conflict_count(tiles: list[dict[str, Any]]) -> int:
    """Return the number of adjacent tile pairs sharing size or class."""
    conflicts = 0
    for first, second in zip(tiles, tiles[1:]):
        same_size = first["tile_size"] == second["tile_size"]
        same_class = first["ground_truth"] == second["ground_truth"]
        if same_size or same_class:
            conflicts += 1
    return conflicts


def _write_session(session_dir: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    """Write split review-session JSON files and return their paths."""
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "tiles").mkdir(parents=True, exist_ok=True)
    labels = _initial_labels_manifest(manifest)
    meta_path = session_dir / SESSION_META_FILENAME
    labels_path = session_dir / LABELS_FILENAME
    _write_json_atomic(meta_path, manifest)
    _write_json_atomic(labels_path, labels)
    manifest["labels"] = labels["labels"]
    return meta_path, labels_path


def _initial_labels_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the collaborator-facing labels manifest for a session."""
    return {
        "session_id": manifest["session_id"],
        "labels": [
            {
                "display_index": tile["display_index"],
                "tile_id": tile["tile_id"],
                "collaborator_label": None,
                "labelled_at": None,
            }
            for tile in sorted(manifest["tiles"], key=lambda item: item["display_index"])
        ],
    }


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Atomically write a JSON object using a sibling temporary file."""
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
        fp.write("\n")
        fp.flush()
    tmp_path.replace(path)


def _resolve_session_id(output_dir: Path, *, session_id: str | None) -> str:
    """Return a unique session identifier inside ``output_dir``."""
    if session_id:
        if (output_dir / session_id).exists():
            raise ValueError(f"Session directory already exists: {output_dir / session_id}")
        return session_id

    import re

    date_prefix = datetime.now().date().isoformat()
    pattern = re.compile(rf"^\d{{4}}-\d{{2}}-\d{{2}}_{re.escape(SESSION_PREFIX)}(\d+)$")
    max_counter = 0
    if output_dir.exists():
        for entry in output_dir.iterdir():
            m = pattern.match(entry.name)
            if m:
                max_counter = max(max_counter, int(m.group(1)))
    counter = max_counter + 1
    return f"{date_prefix}_{SESSION_PREFIX}{counter:0{SESSION_COUNTER_WIDTH}d}"


def _display_path(path: Path) -> str:
    """Return a stable display string for paths inside or outside the repo root."""
    try:
        return path.relative_to(Path(__file__).resolve().parents[2]).as_posix()
    except ValueError:
        return str(path)


def _read_cut_dimensions(cut_tiff_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for a cut-local TIFF."""
    with tifffile.TiffFile(cut_tiff_path) as tif:
        level0, axes = _open_level0_array(tif)
        cut_h, cut_w = _level0_shape(level0, axes)
    return cut_w, cut_h


def _extract_stage(feature: dict[str, Any]) -> str:
    """Return a stage label from a GeoJSON feature."""
    properties = feature.get("properties") or {}
    classification = properties.get("classification") or {}
    stage = classification.get("name")
    return str(stage) if stage else "Unknown"


def _timestamp_now() -> str:
    """Return the current local timestamp as an ISO-8601 string."""
    return datetime.now().isoformat(timespec="seconds")


def _tissue_fraction(tile_arr: np.ndarray) -> float:
    """Return the saturation-mask tissue fraction for one RGB tile."""
    return float(create_tissue_mask(tile_arr, method="saturation", tissue_sat_min=15).mean())


def _build_review_tile_id(cut_name: str, tile_size: int, x0: int, y0: int) -> str:
    """Return a stable review tile identifier derived from cut-local origin."""
    return f"{cut_name}_s{tile_size:04d}_x{x0:06d}_y{y0:06d}"


def _validate_sampling_inputs(
    *,
    cuts_dir: Path,
    tile_sizes: tuple[int, ...],
    n_positive_per_size: int,
    n_negative_per_size: int,
    min_tissue_fraction: float,
    negative_buffer_fraction: float,
) -> None:
    """Validate review-session sampling inputs."""
    if not cuts_dir.exists():
        raise FileNotFoundError(f"Cuts directory not found: {cuts_dir}")
    if not cuts_dir.is_dir():
        raise ValueError(f"Cuts directory must be a directory: {cuts_dir}")
    if not tile_sizes:
        raise ValueError("tile_sizes must contain at least one tile size")
    if any(size <= 0 for size in tile_sizes):
        raise ValueError("tile_sizes must contain only positive integers")
    if n_positive_per_size <= 0:
        raise ValueError("n_positive_per_size must be positive")
    if n_negative_per_size <= 0:
        raise ValueError("n_negative_per_size must be positive")
    if not 0 <= min_tissue_fraction <= 1:
        raise ValueError("min_tissue_fraction must be in the range [0, 1]")
    if negative_buffer_fraction < 0:
        raise ValueError("negative_buffer_fraction must be non-negative")
