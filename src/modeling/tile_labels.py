"""Derive tile-level oocyte labels from annotation geometry.

Tile manifests carry a ``has_oocyte`` flag, but it marks only the tile that contains an
annotation's centroid. Oocytes in this corpus have a median width of about 938 level-0
pixels, so a single oocyte spans several 512 px tiles and the surrounding tiles of a
large oocyte are flagged as having none -- across the corpus 5,858 of the 512 px tiles
that contain oocyte tissue are flagged negative, 2,491 of them more than half covered.
That makes ``has_oocyte`` unusable as a "does this tile contain an oocyte" label.

This module recomputes the label from the geometry instead: the fraction of a tile's
area covered by annotation polygons. It is a metadata join over the tile manifests and
the cut-local annotation GeoJSONs, so it needs no tile images and no re-tiling.

Three traps this module exists to avoid, each of which understates coverage silently:

* The GeoJSON features are ``LineString`` rings, not ``Polygon``s. Passing them to
  ``shapely.geometry.shape`` yields a zero-area LineString, so every area and
  intersection computes as 0 and the labels come out uniformly negative with no error
  raised. Rings are closed into polygons here, and :func:`load_annotation_polygons`
  raises rather than returning an empty result when a GeoJSON has features it could
  not use.
* 57% of these hand-drawn rings self-intersect. The usual ``buffer(0)`` repair keeps
  only the correctly-oriented lobes and drops the rest -- 44% of one annotation in this
  corpus -- so ``make_valid`` is used instead, which preserves the whole area.
* Overlapping annotations would double-count a tile's covered area. The geometries are
  unioned once per cut before measuring, so the parts are disjoint and their
  intersection areas can be summed.

Deliberately free of ``torch``: the detection pipelines need this corrected label too,
and should not have to install the ML stack to get it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from shapely import make_valid
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.data_preparation.remap_annotations import _extract_ring_coords

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MIN_OOCYTE_AREA_FRACTION: float = 0.05
FEATURE_COLLECTION = "FeatureCollection"
POLYGONAL_TYPES = ("Polygon", "MultiPolygon")

# Polygon() needs three distinct vertices and closes the ring itself. Rings arrive
# unclosed from some sources, so this matches what generate_tiles.py accepts rather
# than assuming a closing vertex is present.
MIN_RING_COORDS: int = 3

LABEL_POSITIVE = "positive"
LABEL_AMBIGUOUS = "ambiguous"
LABEL_NEGATIVE = "negative"


def load_annotation_polygons(annotations_geojson_path: str | Path) -> list[BaseGeometry]:
    """Load cut-local annotations as valid, non-degenerate polygonal geometries.

    Rings are extracted with the same helper ``generate_tiles.py`` uses for centroid
    assignment, so both labels are derived from identical geometry.

    Parameters
    ----------
    annotations_geojson_path : str or Path
        Path to a cut-local annotation GeoJSON.

    Returns
    -------
    list of BaseGeometry
        One geometry per usable annotation, in feature order. Usually a ``Polygon``,
        but repairing a self-intersecting ring can split it, so a ``MultiPolygon`` is
        also possible -- callers reaching for ``.exterior`` must handle both. Empty
        only when the GeoJSON itself has no features.

    Raises
    ------
    ValueError
        If the file is not a FeatureCollection, or if it has features but none of them
        yielded a usable polygon -- the signature of the LineString trap described in
        the module docstring, which would otherwise pass silently as "no oocytes here".

    Examples
    --------
    >>> MIN_RING_COORDS
    3
    """
    path = _resolve_repo_path(annotations_geojson_path)
    geojson = _read_json(path)
    if geojson.get("type") != FEATURE_COLLECTION:
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection")

    features = geojson.get("features") or []
    polygons: list[BaseGeometry] = []
    for feature_pos, feature in enumerate(features):
        polygon = _polygon_from_feature(feature, path, feature_pos)
        if polygon is not None:
            polygons.append(polygon)

    if features and not polygons:
        raise ValueError(
            f"{path} has {len(features)} feature(s) but none yielded a usable polygon; "
            "refusing to report zero oocyte coverage for a cut that has annotations"
        )
    if not features:
        LOG.info("%s has no annotation features", path.name)
    return polygons


def compute_oocyte_area_fractions(
    tile_manifest_path: str | Path,
    annotations_geojson_path: str | Path,
    tile_sizes: tuple[int, ...] | None = None,
) -> dict[str, float]:
    """Map each tile in a manifest to the fraction of its area covered by oocytes.

    The denominator is the tile's ``cut_local_bbox`` area, which ``generate_tiles.py``
    clips at the cut edge, so it measures coverage of the tile's real image content
    rather than of the padded region. Padding is rare -- 22 of the 55,070 512 px and
    1024 px tiles in the corpus, none of them less than half real -- so the choice
    barely moves any label, but it is made here explicitly rather than by accident.

    Parameters
    ----------
    tile_manifest_path : str or Path
        Path to a ``{cut}_tile_manifest.json``.
    annotations_geojson_path : str or Path
        Path to the matching cut-local annotation GeoJSON.
    tile_sizes : tuple of int, optional
        Restrict measurement to these tile sizes. A manifest holds roughly 18x more
        tiles across all four sizes than at 512 and 1024 alone, and the geometry work
        is per tile, so passing the sizes actually wanted is worth it on a path that
        reruns for every training configuration. Defaults to every size present.

    Returns
    -------
    dict of str to float
        ``tile_id`` to covered fraction in ``[0.0, 1.0]``, one entry per measured tile.

    Raises
    ------
    ValueError
        If the manifest and the GeoJSON describe different cuts, or if a tile bbox is
        degenerate.

    Examples
    --------
    >>> DEFAULT_MIN_OOCYTE_AREA_FRACTION
    0.05
    """
    manifest_path = _resolve_repo_path(tile_manifest_path)
    return oocyte_area_fractions_for_manifest(
        _read_json(manifest_path),
        annotations_geojson_path,
        tile_sizes=tile_sizes,
        manifest_path=manifest_path,
    )


def oocyte_area_fractions_for_manifest(
    manifest: dict[str, Any],
    annotations_geojson_path: str | Path,
    tile_sizes: tuple[int, ...] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, float]:
    """Measure coverage from an already-parsed tile manifest.

    Same contract as :func:`compute_oocyte_area_fractions`, which is a thin wrapper over
    this. Callers that have already read the manifest use this instead: the manifests
    embed per-tile annotation coordinates and run to hundreds of megabytes in total, so
    parsing them twice is the dominant cost of building the tile index.

    Parameters
    ----------
    manifest : dict
        Parsed ``{cut}_tile_manifest.json`` content.
    annotations_geojson_path : str or Path
        Path to the matching cut-local annotation GeoJSON.
    tile_sizes : tuple of int, optional
        Restrict measurement to these tile sizes.
    manifest_path : Path, optional
        Only used to make error messages name the offending file.

    Returns
    -------
    dict of str to float
        ``tile_id`` to covered fraction in ``[0.0, 1.0]``, one entry per measured tile.
    """
    geojson_path = _resolve_repo_path(annotations_geojson_path)
    _check_same_cut(manifest, manifest_path, geojson_path)
    polygons = load_annotation_polygons(geojson_path)

    tiles = [
        tile
        for tile in manifest.get("tiles", [])
        if tile_sizes is None or tile["tile_size"] in tile_sizes
    ]
    # Validate every bbox before measuring, so a malformed manifest fails the same way
    # whether or not its cut happens to have annotations.
    tile_boxes = [_tile_box(tile) for tile in tiles]
    if not polygons:
        return {tile["tile_id"]: 0.0 for tile in tiles}

    # Union once so the parts are disjoint; overlapping annotations would otherwise be
    # counted twice in a tile that both cover.
    merged = unary_union(polygons)
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    tree = STRtree(parts)

    fractions: dict[str, float] = {}
    for tile, tile_box in zip(tiles, tile_boxes):
        covered = 0.0
        for index in tree.query(tile_box):
            covered += parts[index].intersection(tile_box).area
        fractions[tile["tile_id"]] = min(covered / tile_box.area, 1.0)
    return fractions


def label_for_area_fraction(
    fraction: float,
    min_oocyte_area_fraction: float = DEFAULT_MIN_OOCYTE_AREA_FRACTION,
) -> str:
    """Classify a covered-area fraction as positive, ambiguous, or negative.

    A tile touching an oocyte only at its edge shows a sliver that is genuinely hard to
    call, so the band between "touches an oocyte" and the threshold is reported as
    ambiguous and excluded from training rather than forced into either class.

    Parameters
    ----------
    fraction : float
        Covered fraction from :func:`compute_oocyte_area_fractions`.
    min_oocyte_area_fraction : float, optional
        Coverage at or above which a tile counts as positive.

    Returns
    -------
    str
        One of ``"positive"``, ``"ambiguous"``, or ``"negative"``.

    Examples
    --------
    >>> label_for_area_fraction(0.0)
    'negative'
    >>> label_for_area_fraction(0.01)
    'ambiguous'
    >>> label_for_area_fraction(0.5)
    'positive'
    """
    if not 0.0 <= min_oocyte_area_fraction <= 1.0:
        raise ValueError("min_oocyte_area_fraction must be in the range [0, 1]")
    if fraction < 0.0:
        raise ValueError("fraction must be non-negative")
    if fraction >= min_oocyte_area_fraction and fraction > 0.0:
        return LABEL_POSITIVE
    if fraction > 0.0:
        return LABEL_AMBIGUOUS
    return LABEL_NEGATIVE


def _polygon_from_feature(
    feature: dict[str, Any],
    path: Path,
    feature_pos: int,
) -> BaseGeometry | None:
    """Build a valid, positive-area polygonal geometry from one feature, or None."""
    geometry = feature.get("geometry") or {}
    _reject_lossy_geometry(geometry, path, feature_pos)
    try:
        coords = _extract_ring_coords(geometry)
    except ValueError as exc:
        LOG.warning(
            "%s feature %d (geometry %r) could not be read as a ring: %s; skipped",
            path.name,
            feature_pos,
            geometry.get("type"),
            exc,
        )
        return None

    if len(coords) < MIN_RING_COORDS:
        LOG.warning(
            "%s feature %d has %d coordinate(s), too few for a polygon; skipped",
            path.name,
            feature_pos,
            len(coords),
        )
        return None

    polygon = Polygon(coords)
    if not polygon.is_valid:
        # 57% of this corpus's hand-drawn rings self-intersect. buffer(0) is the usual
        # idiom but it keeps only the correctly-oriented lobes and silently discards
        # the rest -- up to 44% of a single annotation here -- which would understate
        # coverage and flip tiles to negative, the exact error this module prevents.
        # make_valid keeps the whole area, at the cost of sometimes returning a
        # MultiPolygon or a GeometryCollection.
        repaired = _polygonal_parts(make_valid(polygon))
    else:
        repaired = polygon

    if repaired is None or repaired.is_empty or repaired.area <= 0.0:
        LOG.warning(
            "%s feature %d has zero polygonal area after repair; skipped",
            path.name,
            feature_pos,
        )
        return None
    return repaired


def _reject_lossy_geometry(
    geometry: dict[str, Any],
    path: Path,
    feature_pos: int,
) -> None:
    """Refuse geometry whose area the shared ring extractor cannot represent.

    ``_extract_ring_coords`` is reused so that this label and the manifest's centroid
    label derive from identical geometry, but it keeps only a Polygon's first ring and
    only a MultiPolygon's largest part. For a centroid that is immaterial; for an area
    measurement it would over-count holes and under-count multi-part annotations -- a
    silent understatement of exactly the kind this module exists to prevent. Today's
    corpus has neither (1,129 LineStrings and 79 single-ring Polygons, no
    MultiPolygons), so this stops a future QuPath export rather than current data.
    """
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geometry_type == "Polygon" and len(coordinates) > 1:
        raise ValueError(
            f"{path.name} feature {feature_pos} is a Polygon with "
            f"{len(coordinates) - 1} interior ring(s); measuring its area would ignore "
            "the holes and overstate coverage"
        )
    if geometry_type == "MultiPolygon":
        raise ValueError(
            f"{path.name} feature {feature_pos} is a MultiPolygon; measuring its area "
            "would keep only the largest part and understate coverage"
        )


def _polygonal_parts(geometry: BaseGeometry) -> BaseGeometry | None:
    """Reduce a repaired geometry to its polygonal content.

    ``make_valid`` can return a GeometryCollection mixing polygons with the degenerate
    lines and points a self-touching ring collapses to. Only the polygonal parts carry
    area, so the rest is dropped.
    """
    if geometry.geom_type in POLYGONAL_TYPES:
        return geometry
    if geometry.geom_type == "GeometryCollection":
        parts = [part for part in geometry.geoms if part.geom_type in POLYGONAL_TYPES]
        return unary_union(parts) if parts else None
    return None


def _check_same_cut(
    manifest: dict[str, Any],
    manifest_path: Path | None,
    geojson_path: Path,
) -> None:
    """Raise if a manifest is paired with another cut's annotations.

    Silently mislabelling every tile is the cost of getting this pairing wrong in a
    loop over the corpus, and the file names carry enough information to catch it.
    """
    cut_name = manifest.get("cut_name")
    if not cut_name:
        raise ValueError(f"{manifest_path or 'tile manifest'} has no cut_name")
    expected = f"{cut_name}_annotations"
    if geojson_path.stem != expected:
        raise ValueError(
            f"{geojson_path.name} does not belong to cut {cut_name!r} "
            f"(expected {expected}.geojson)"
        )


def _tile_box(tile: dict[str, Any]):
    """Return the shapely box for a tile's cut-local bbox."""
    bbox = tile["cut_local_bbox"]
    tile_box = box(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
    if tile_box.area <= 0.0:
        raise ValueError(f"Tile {tile.get('tile_id')!r} has a degenerate bbox: {bbox}")
    return tile_box


def _resolve_repo_path(path: str | Path) -> Path:
    """Resolve a repository-relative path into an absolute path.

    Examples
    --------
    >>> _resolve_repo_path(Path("src")).is_absolute()
    True
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a Python dict.

    Examples
    --------
    >>> isinstance({}, dict)
    True
    """
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
