"""Generate multi-scale training tiles from cut-local TIFFs.

This module implements Milestone 4 (M4) of the coral microscopy pipeline.
It grids cut-local pyramidal TIFFs into fixed-size PNG tiles, assigns
cut-local annotations to tiles by centroid, and writes one manifest per cut.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import zarr
from PIL import Image

from src.data_preparation.extract_ndpi_cuts import create_tissue_mask
from src.data_preparation.remap_annotations import _compute_centroid, _extract_ring_coords

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TILE_SIZES: tuple[int, ...] = (128, 256, 512, 1024)
DEFAULT_OVERLAP: float = 0.20
DEFAULT_MIN_TISSUE_FRACTION: float = 0.20
TILE_SIZE_DIR_WIDTH: int = 4
FEATURE_COLLECTION = "FeatureCollection"


def generate_tiles_for_cut(
    cut_tiff_path: str | Path,
    annotations_geojson_path: str | Path,
    cuts_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    tile_sizes: tuple[int, ...] = DEFAULT_TILE_SIZES,
    overlap: float = DEFAULT_OVERLAP,
    min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION,
    skip_if_exists: bool = False,
) -> dict[str, Any]:
    """Extract multi-scale tiles from a single pyramidal TIFF cut.

    Reads the cut at level-0 resolution, grids it into fixed-size tiles at
    each requested size, assigns cut-local annotations to tiles by centroid,
    filters low-tissue tiles, writes PNG images, and returns the tile manifest.

    Parameters
    ----------
    cut_tiff_path : str or Path
        Path to the cut-local pyramidal TIFF.
    annotations_geojson_path : str or Path
        Path to the cut-local annotation GeoJSON.
    cuts_manifest_path : str or Path
        Path to the slide-level ``_cuts.json`` manifest.
    output_dir : str or Path
        Root output directory for tile PNGs and the cut manifest.
    tile_sizes : tuple of int, optional
        Tile edge lengths in level-0 cut-local pixels.
    overlap : float, optional
        Fractional overlap between adjacent tiles.
    min_tissue_fraction : float, optional
        Minimum tissue fraction required to retain a tile.
    skip_if_exists : bool, optional
        If True, return the cached manifest when present.

    Returns
    -------
    dict of str to Any
        Tile manifest dict for the processed cut.

    Examples
    --------
    >>> manifest = {"n_tiles_total": 100, "n_tiles_with_oocyte": 10}
    >>> manifest["n_tiles_with_oocyte"] <= manifest["n_tiles_total"]
    True
    """
    cut_tiff_path = _resolve_repo_path(cut_tiff_path)
    annotations_geojson_path = _resolve_repo_path(annotations_geojson_path)
    cuts_manifest_path = _resolve_repo_path(cuts_manifest_path)
    output_dir = _resolve_repo_path(output_dir)

    _validate_generation_inputs(
        cut_tiff_path=cut_tiff_path,
        annotations_geojson_path=annotations_geojson_path,
        cuts_manifest_path=cuts_manifest_path,
        tile_sizes=tile_sizes,
        overlap=overlap,
        min_tissue_fraction=min_tissue_fraction,
    )

    cut_name = cut_tiff_path.stem
    stem = _stem_from_cut_name(cut_name)
    tile_sizes = tuple(int(size) for size in tile_sizes)
    cut_output_dir = output_dir / stem / cut_name
    manifest_path = cut_output_dir / f"{cut_name}_tile_manifest.json"

    if skip_if_exists and manifest_path.exists():
        LOG.info("Skipping %s because %s already exists", cut_name, manifest_path)
        return _read_json(manifest_path)

    annotations = _load_annotations(annotations_geojson_path)
    cuts_manifest = _load_cuts_manifest(cuts_manifest_path)
    cut_entry = _cut_entry_by_name(cuts_manifest, cut_name)

    with tifffile.TiffFile(cut_tiff_path) as tif:
        level0, axes = _open_level0_array(tif)
        cut_h, cut_w = _level0_shape(level0, axes)
        expected_w, expected_h = cut_entry["level0_size"]
        if (cut_w, cut_h) != (expected_w, expected_h):
            LOG.warning(
                "Cut dimensions for %s differ from manifest: TIFF=%sx%s manifest=%sx%s",
                cut_name,
                cut_w,
                cut_h,
                expected_w,
                expected_h,
            )

        tiles: list[dict[str, Any]] = []
        n_tiles_total = 0
        n_tiles_with_oocyte = 0
        n_tiles_skipped_tissue = 0

        for tile_size in tile_sizes:
            stride = max(1, round(tile_size * (1 - overlap)))
            grid = _compute_tile_grid(cut_h=cut_h, cut_w=cut_w, tile_size=tile_size, stride=stride)
            for tile_bbox in grid:
                tile_id = _build_tile_id(cut_name, tile_size, tile_bbox["row"], tile_bbox["col"])
                tile_arr = _read_tile_region(
                    level0,
                    axes,
                    tile_bbox["x0"],
                    tile_bbox["y0"],
                    tile_bbox["x1"],
                    tile_bbox["y1"],
                )
                tile_arr = _pad_tile_array(
                    tile_arr,
                    pad_right=tile_bbox["pad_right"],
                    pad_bottom=tile_bbox["pad_bottom"],
                )

                tissue_fraction = float(
                    create_tissue_mask(
                        tile_arr,
                        method="saturation",
                        tissue_sat_min=15,
                    ).mean()
                )
                if tissue_fraction < min_tissue_fraction:
                    n_tiles_skipped_tissue += 1
                    LOG.debug(
                        "Skipping %s due to tissue_fraction=%.4f < %.4f",
                        tile_id,
                        tissue_fraction,
                        min_tissue_fraction,
                    )
                    continue

                assigned_annotations = _assign_annotations_to_tile(annotations, tile_bbox)
                tile_annotations = [
                    {
                        "annotation_id": annotation["annotation_id"],
                        "stage": annotation["stage"],
                        "tile_local_coords": _to_tile_local_coords(
                            annotation["coords"],
                            tile_bbox["x0"],
                            tile_bbox["y0"],
                        ),
                    }
                    for annotation in assigned_annotations
                ]
                stages = list(dict.fromkeys(item["stage"] for item in tile_annotations))

                png_dir = cut_output_dir / f"{tile_size:0{TILE_SIZE_DIR_WIDTH}d}"
                png_path = png_dir / f"{tile_id}.png"
                _write_tile_png(tile_arr, png_path)

                tiles.append(
                    {
                        "tile_id": tile_id,
                        "tile_size": tile_size,
                        "row": tile_bbox["row"],
                        "col": tile_bbox["col"],
                        "cut_local_bbox": {
                            "x0": tile_bbox["x0"],
                            "y0": tile_bbox["y0"],
                            "x1": tile_bbox["x1"],
                            "y1": tile_bbox["y1"],
                        },
                        "pad_right": tile_bbox["pad_right"],
                        "pad_bottom": tile_bbox["pad_bottom"],
                        "tissue_fraction": round(tissue_fraction, 6),
                        "has_oocyte": bool(tile_annotations),
                        "n_oocytes": len(tile_annotations),
                        "stages": stages,
                        "annotations": tile_annotations,
                        "png_path": _path_relative_to_repo(png_path),
                    }
                )
                n_tiles_total += 1
                if tile_annotations:
                    n_tiles_with_oocyte += 1

    manifest = {
        "cut_name": cut_name,
        "stem": stem,
        "cut_index": cut_entry["index"],
        "mpp_x": cuts_manifest.get("mpp_x"),
        "mpp_y": cuts_manifest.get("mpp_y"),
        "level0_bbox": cut_entry["level0_bbox"],
        "tile_sizes": list(tile_sizes),
        "overlap": overlap,
        "min_tissue_fraction": min_tissue_fraction,
        "n_tiles_total": n_tiles_total,
        "n_tiles_with_oocyte": n_tiles_with_oocyte,
        "n_tiles_skipped_tissue": n_tiles_skipped_tissue,
        "tiles": tiles,
    }
    _write_tile_manifest(manifest, cut_output_dir, cut_name)
    return manifest


def generate_tiles_batch(
    cuts_dir: str | Path,
    output_dir: str | Path,
    *,
    tile_sizes: tuple[int, ...] = DEFAULT_TILE_SIZES,
    overlap: float = DEFAULT_OVERLAP,
    min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION,
    skip_if_exists: bool = True,
    stem_glob: str = "*",
) -> list[dict[str, Any]]:
    """Run ``generate_tiles_for_cut`` for every cut under ``cuts_dir``.

    Parameters
    ----------
    cuts_dir : str or Path
        Root directory containing cut TIFFs and manifests.
    output_dir : str or Path
        Root directory for tile output.
    tile_sizes : tuple of int, optional
        Tile edge lengths in level-0 cut-local pixels.
    overlap : float, optional
        Fractional overlap between adjacent tiles.
    min_tissue_fraction : float, optional
        Minimum tissue fraction required to retain a tile.
    skip_if_exists : bool, optional
        If True, reuse cached manifests where possible.
    stem_glob : str, optional
        Glob pattern to restrict which stem directories are processed.

    Returns
    -------
    list of dict
        One manifest dict per processed cut.

    Examples
    --------
    >>> isinstance([], list)
    True
    """
    cuts_dir = _resolve_repo_path(cuts_dir)
    output_dir = _resolve_repo_path(output_dir)

    manifests: list[dict[str, Any]] = []
    for cut_tiff_path in sorted(cuts_dir.glob(f"{stem_glob}/*_cut[0-9][0-9][0-9].tif")):
        cut_name = cut_tiff_path.stem
        stem = _stem_from_cut_name(cut_name)
        annotations_geojson_path = cut_tiff_path.with_name(f"{cut_name}_annotations.geojson")
        cuts_manifest_path = cuts_dir / stem / f"{stem}_cuts.json"

        if not annotations_geojson_path.exists():
            LOG.warning(
                "Skipping %s because %s is missing",
                cut_name,
                annotations_geojson_path,
            )
            continue
        if not cuts_manifest_path.exists():
            LOG.warning(
                "Skipping %s because %s is missing",
                cut_name,
                cuts_manifest_path,
            )
            continue

        manifests.append(
            generate_tiles_for_cut(
                cut_tiff_path,
                annotations_geojson_path,
                cuts_manifest_path,
                output_dir,
                tile_sizes=tile_sizes,
                overlap=overlap,
                min_tissue_fraction=min_tissue_fraction,
                skip_if_exists=skip_if_exists,
            )
        )
    return manifests


def _read_cut_level0(tiff_path: Path) -> np.ndarray:
    """Open a cut TIFF and return the level-0 RGB image.

    Parameters
    ----------
    tiff_path : Path
        Path to a pyramidal or single-level TIFF cut.

    Returns
    -------
    np.ndarray
        Level-0 RGB image as ``uint8`` with shape ``(H, W, 3)``.

    Examples
    --------
    >>> isinstance(np.zeros((2, 2, 3), dtype=np.uint8), np.ndarray)
    True
    """
    with tifffile.TiffFile(tiff_path) as tif:
        level0, axes = _open_level0_array(tif)
        array = np.array(level0)
    return _coerce_rgb_array(array, axes)


def _compute_tile_grid(
    cut_h: int,
    cut_w: int,
    tile_size: int,
    stride: int,
) -> list[dict[str, int]]:
    """Return tile grid metadata for a cut-local image.

    Parameters
    ----------
    cut_h : int
        Cut height in level-0 pixels.
    cut_w : int
        Cut width in level-0 pixels.
    tile_size : int
        Tile edge length in pixels.
    stride : int
        Grid stride in pixels.

    Returns
    -------
    list of dict
        One dict per tile with row, col, bbox, and padding metadata.

    Examples
    --------
    >>> grid = _compute_tile_grid(cut_h=100, cut_w=100, tile_size=256, stride=205)
    >>> len(grid)
    1
    >>> grid[0]["pad_right"], grid[0]["pad_bottom"]
    (156, 156)
    """
    if cut_h <= 0 or cut_w <= 0:
        raise ValueError("cut_h and cut_w must be positive integers")
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive integers")

    grid: list[dict[str, int]] = []
    for row, y0 in enumerate(range(0, cut_h, stride)):
        y1 = min(y0 + tile_size, cut_h)
        pad_bottom = tile_size - (y1 - y0)
        for col, x0 in enumerate(range(0, cut_w, stride)):
            x1 = min(x0 + tile_size, cut_w)
            pad_right = tile_size - (x1 - x0)
            grid.append(
                {
                    "row": row,
                    "col": col,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "pad_right": pad_right,
                    "pad_bottom": pad_bottom,
                }
            )
    return grid


def _load_annotations(geojson_path: Path) -> list[dict[str, Any]]:
    """Load cut-local annotations from a GeoJSON file.

    Parameters
    ----------
    geojson_path : Path
        Path to the cut-local annotation GeoJSON.

    Returns
    -------
    list of dict
        Flat annotation records with ``annotation_id``, ``stage``, and ``coords``.

    Examples
    --------
    >>> isinstance([], list)
    True
    """
    geojson = _read_json(geojson_path)
    if geojson.get("type") != FEATURE_COLLECTION:
        raise ValueError(f"{geojson_path} is not a GeoJSON FeatureCollection")

    stem = _stem_from_cut_annotations_stem(geojson_path.stem)
    annotations: list[dict[str, Any]] = []
    for feature_pos, feature in enumerate(geojson.get("features", [])):
        coords = _extract_ring_coords(feature.get("geometry") or {})
        if not coords:
            LOG.warning("Skipping empty annotation geometry at position %d in %s", feature_pos, geojson_path)
            continue
        annotations.append(
            {
                "annotation_id": str(feature.get("id", f"{stem}:{feature_pos}")),
                "stage": _extract_stage(feature),
                "coords": coords,
            }
        )
    return annotations


def _assign_annotations_to_tile(
    annotations: list[dict[str, Any]],
    tile_bbox: dict[str, int],
) -> list[dict[str, Any]]:
    """Return annotations whose centroid falls within ``tile_bbox``.

    Parameters
    ----------
    annotations : list of dict
        Annotation records containing a ``coords`` key with ``[x, y]`` pairs.
    tile_bbox : dict of str to int
        Inclusive cut-local tile bounds with ``x0``, ``y0``, ``x1``, ``y1`` keys.

    Returns
    -------
    list of dict
        Annotations assigned to the tile.

    Examples
    --------
    >>> ann = {"annotation_id": "a", "stage": "Stage 0", "coords": [[0.0, 0.0], [10.0, 10.0]]}
    >>> bbox = {"x0": 0, "y0": 0, "x1": 5, "y1": 5}
    >>> [item["annotation_id"] for item in _assign_annotations_to_tile([ann], bbox)]
    ['a']
    """
    assigned: list[dict[str, Any]] = []
    for annotation in annotations:
        cx, cy = _compute_centroid(annotation["coords"])
        if (
            tile_bbox["x0"] <= cx <= tile_bbox["x1"]
            and tile_bbox["y0"] <= cy <= tile_bbox["y1"]
        ):
            assigned.append(annotation)
    return assigned


def _to_tile_local_coords(
    coords: list[list[float]],
    x0: int,
    y0: int,
) -> list[list[float]]:
    """Translate cut-local coordinates into tile-local coordinates.

    Parameters
    ----------
    coords : list of list of float
        Cut-local ``[x, y]`` coordinate pairs.
    x0 : int
        Tile origin x-coordinate in cut-local space.
    y0 : int
        Tile origin y-coordinate in cut-local space.

    Returns
    -------
    list of list of float
        Tile-local coordinates.

    Examples
    --------
    >>> _to_tile_local_coords([[100.0, 200.0], [150.0, 250.0]], 50, 80)
    [[50.0, 120.0], [100.0, 170.0]]
    """
    return [[round(float(x) - x0, 10), round(float(y) - y0, 10)] for x, y in coords]


def _extract_stage(feature: dict[str, Any]) -> str:
    """Return the annotation stage name from a GeoJSON feature.

    Parameters
    ----------
    feature : dict of str to Any
        GeoJSON feature or feature-like annotation record.

    Returns
    -------
    str
        Classification name when present, otherwise ``"Unknown"``.

    Examples
    --------
    >>> _extract_stage({"properties": {"classification": {"name": "Stage 3"}}})
    'Stage 3'
    >>> _extract_stage({"properties": {}})
    'Unknown'
    """
    properties = feature.get("properties") or {}
    classification = properties.get("classification") or {}
    stage = classification.get("name")
    return str(stage) if stage else "Unknown"


def _build_tile_id(cut_name: str, size: int, row: int, col: int) -> str:
    """Return the canonical tile identifier for one grid position.

    Parameters
    ----------
    cut_name : str
        Cut name, for example ``"CHN_AU_10_19-21_cut000"``.
    size : int
        Tile edge length in pixels.
    row : int
        Zero-indexed grid row.
    col : int
        Zero-indexed grid column.

    Returns
    -------
    str
        Canonical tile identifier.

    Examples
    --------
    >>> _build_tile_id("CHN_AU_10_19-21_cut000", 256, 3, 12)
    'CHN_AU_10_19-21_cut000_s0256_r0003_c0012'
    """
    return f"{cut_name}_s{size:04d}_r{row:04d}_c{col:04d}"


def _write_tile_png(arr: np.ndarray, path: Path) -> None:
    """Write one RGB tile as a PNG image.

    Parameters
    ----------
    arr : np.ndarray
        ``uint8`` RGB tile with shape ``(H, W, 3)``.
    path : Path
        Output path for the PNG image.

    Returns
    -------
    None
        Writes the image to disk.

    Examples
    --------
    >>> isinstance(np.zeros((1, 1, 3), dtype=np.uint8), np.ndarray)
    True
    """
    if arr.dtype != np.uint8:
        raise ValueError("Tile array must be uint8 before PNG export")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Tile array must have shape (H, W, 3)")

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


def _write_tile_manifest(manifest: dict[str, Any], output_dir: Path, cut_name: str) -> Path:
    """Write a cut tile manifest as JSON.

    Parameters
    ----------
    manifest : dict of str to Any
        Tile manifest data to write.
    output_dir : Path
        Directory to write the manifest into.
    cut_name : str
        Cut name used to construct the output filename.

    Returns
    -------
    Path
        Path to the written manifest.

    Examples
    --------
    >>> path = Path("example") / "out"
    >>> path.name
    'out'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{cut_name}_tile_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
        fp.write("\n")
    return manifest_path


def _resolve_repo_path(path: str | Path) -> Path:
    """Resolve a repository-relative path into an absolute path.

    Parameters
    ----------
    path : str or Path
        Relative or absolute path.

    Returns
    -------
    Path
        Absolute path within the repository when relative input is given.

    Examples
    --------
    >>> _resolve_repo_path(Path("src")).is_absolute()
    True
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _validate_generation_inputs(
    *,
    cut_tiff_path: Path,
    annotations_geojson_path: Path,
    cuts_manifest_path: Path,
    tile_sizes: tuple[int, ...],
    overlap: float,
    min_tissue_fraction: float,
) -> None:
    """Validate required file paths and generation parameters.

    Parameters
    ----------
    cut_tiff_path : Path
        Path to the cut TIFF.
    annotations_geojson_path : Path
        Path to the cut-local annotation GeoJSON.
    cuts_manifest_path : Path
        Path to the cuts manifest.
    tile_sizes : tuple of int
        Requested tile sizes.
    overlap : float
        Fractional overlap between neighboring tiles.
    min_tissue_fraction : float
        Minimum tissue-fraction threshold.

    Returns
    -------
    None
        Raises on invalid inputs.

    Examples
    --------
    >>> _validate_generation_inputs  # doctest: +ELLIPSIS
    <function _validate_generation_inputs ...>
    """
    if not cut_tiff_path.exists():
        raise FileNotFoundError(f"Cut TIFF not found: {cut_tiff_path}")
    if not annotations_geojson_path.exists():
        raise FileNotFoundError(f"Annotation GeoJSON not found: {annotations_geojson_path}")
    if not cuts_manifest_path.exists():
        raise FileNotFoundError(f"Cuts manifest not found: {cuts_manifest_path}")
    if not tile_sizes:
        raise ValueError("tile_sizes must contain at least one tile size")
    if any(size <= 0 for size in tile_sizes):
        raise ValueError("tile_sizes must contain only positive integers")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in the range [0, 1)")
    if not 0 <= min_tissue_fraction <= 1:
        raise ValueError("min_tissue_fraction must be in the range [0, 1]")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a Python dict.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict of str to Any
        Parsed JSON content.

    Examples
    --------
    >>> isinstance({}, dict)
    True
    """
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_cuts_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate a slide-level cuts manifest.

    Parameters
    ----------
    manifest_path : Path
        Path to the ``_cuts.json`` file.

    Returns
    -------
    dict of str to Any
        Parsed cuts manifest.

    Examples
    --------
    >>> isinstance({"cuts": []}, dict)
    True
    """
    manifest = _read_json(manifest_path)
    cuts = manifest.get("cuts")
    if not isinstance(cuts, list) or not cuts:
        raise ValueError(f"{manifest_path} must contain a non-empty 'cuts' list")
    return manifest


def _cut_entry_by_name(manifest: dict[str, Any], cut_name: str) -> dict[str, Any]:
    """Return the cut entry matching ``cut_name``.

    Parameters
    ----------
    manifest : dict of str to Any
        Slide-level cuts manifest.
    cut_name : str
        Exact cut name to resolve.

    Returns
    -------
    dict of str to Any
        Matching cut entry.

    Examples
    --------
    >>> entry = _cut_entry_by_name({"cuts": [{"name": "a", "index": 0}]}, "a")
    >>> entry["index"]
    0
    """
    for cut in manifest.get("cuts", []):
        if cut.get("name") == cut_name:
            return cut
    raise ValueError(f"Cut {cut_name!r} is not present in the supplied cuts manifest")


def _stem_from_cut_name(cut_name: str) -> str:
    """Return the slide stem portion of a cut name.

    Parameters
    ----------
    cut_name : str
        Cut name such as ``"CHN_AU_10_19-21_cut000"``.

    Returns
    -------
    str
        Slide stem without the ``_cutNNN`` suffix.

    Examples
    --------
    >>> _stem_from_cut_name("CHN_AU_10_19-21_cut000")
    'CHN_AU_10_19-21'
    """
    stem, sep, _suffix = cut_name.rpartition("_cut")
    if not sep:
        raise ValueError(f"Cut name does not follow the expected pattern: {cut_name!r}")
    return stem


def _stem_from_cut_annotations_stem(annotations_stem: str) -> str:
    """Return the slide stem from an annotations filename stem.

    Parameters
    ----------
    annotations_stem : str
        Filename stem from a cut annotations GeoJSON.

    Returns
    -------
    str
        Slide stem used for fallback annotation IDs.

    Examples
    --------
    >>> _stem_from_cut_annotations_stem("CHN_AU_10_19-21_cut000_annotations")
    'CHN_AU_10_19-21'
    """
    if annotations_stem.endswith("_annotations"):
        cut_name = annotations_stem[: -len("_annotations")]
        return _stem_from_cut_name(cut_name)
    return annotations_stem


def _open_level0_array(tif: tifffile.TiffFile) -> tuple[Any, str]:
    """Return the zarr-backed level-0 array and its axes string.

    Parameters
    ----------
    tif : tifffile.TiffFile
        Open TIFF handle.

    Returns
    -------
    tuple
        ``(level0_array, axes)`` for the TIFF's first series.

    Examples
    --------
    >>> _open_level0_array  # doctest: +ELLIPSIS
    <function _open_level0_array ...>
    """
    series = tif.series[0]
    store = series.aszarr()
    root = zarr.open(store, mode="r")
    if isinstance(root, zarr.Array):
        return root, series.axes
    if "0" in root:
        return root["0"], series.levels[0].axes
    first_key = sorted(root.keys(), key=int)[0]
    return root[first_key], series.levels[0].axes


def _level0_shape(level0: Any, axes: str) -> tuple[int, int]:
    """Return ``(height, width)`` for a level-0 array.

    Parameters
    ----------
    level0 : Any
        Zarr-backed array or array-like level-0 image.
    axes : str
        TIFF axes string describing the array layout.

    Returns
    -------
    tuple of int
        Height and width in pixels.

    Examples
    --------
    >>> _level0_shape(np.zeros((10, 20, 3), dtype=np.uint8), "YXS")
    (10, 20)
    """
    shape = level0.shape
    if axes == "YXS":
        return int(shape[0]), int(shape[1])
    if axes in {"SYX", "CYX"}:
        return int(shape[1]), int(shape[2])
    raise ValueError(f"Unsupported TIFF axes for tile generation: {axes!r}")


def _read_tile_region(level0: Any, axes: str, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Read one tile-sized region from the level-0 image.

    Parameters
    ----------
    level0 : Any
        Zarr-backed level-0 array.
    axes : str
        TIFF axes string.
    x0, y0, x1, y1 : int
        Cut-local tile bounds.

    Returns
    -------
    np.ndarray
        RGB image region with shape ``(H, W, 3)``.

    Examples
    --------
    >>> arr = np.zeros((4, 5, 3), dtype=np.uint8)
    >>> _read_tile_region(arr, "YXS", 0, 0, 2, 3).shape
    (3, 2, 3)
    """
    if axes == "YXS":
        array = np.array(level0[y0:y1, x0:x1])
    elif axes in {"SYX", "CYX"}:
        array = np.array(level0[:, y0:y1, x0:x1])
    else:
        raise ValueError(f"Unsupported TIFF axes for tile generation: {axes!r}")
    return _coerce_rgb_array(array, axes)


def _coerce_rgb_array(array: np.ndarray, axes: str) -> np.ndarray:
    """Coerce an array from TIFF axis order into ``(H, W, 3)`` RGB.

    Parameters
    ----------
    array : np.ndarray
        Raw TIFF array data.
    axes : str
        TIFF axes string.

    Returns
    -------
    np.ndarray
        RGB ``uint8`` image with shape ``(H, W, 3)``.

    Examples
    --------
    >>> _coerce_rgb_array(np.zeros((3, 4, 5), dtype=np.uint8), "CYX").shape
    (4, 5, 3)
    """
    if axes in {"SYX", "CYX"}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D RGB array, got shape {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] != 3:
        raise ValueError(f"Expected RGB channels in the last dimension, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _pad_tile_array(arr: np.ndarray, *, pad_right: int, pad_bottom: int) -> np.ndarray:
    """Pad a partial edge tile up to the target tile size.

    Parameters
    ----------
    arr : np.ndarray
        RGB tile array.
    pad_right : int
        Number of zero columns to append at the right edge.
    pad_bottom : int
        Number of zero rows to append at the bottom edge.

    Returns
    -------
    np.ndarray
        Padded RGB tile.

    Examples
    --------
    >>> _pad_tile_array(np.zeros((2, 2, 3), dtype=np.uint8), pad_right=1, pad_bottom=1).shape
    (3, 3, 3)
    """
    if pad_right == 0 and pad_bottom == 0:
        return arr
    return np.pad(arr, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="constant")


def _path_relative_to_repo(path: Path) -> str:
    """Return a repository-relative POSIX path string.

    Parameters
    ----------
    path : Path
        Absolute path inside the repository.

    Returns
    -------
    str
        POSIX-style path relative to the repository root.

    Examples
    --------
    >>> isinstance(_path_relative_to_repo(REPO_ROOT / "README.md"), str)
    True
    """
    return path.relative_to(REPO_ROOT).as_posix()