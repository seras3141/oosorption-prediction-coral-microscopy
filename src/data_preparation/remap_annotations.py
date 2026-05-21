"""Remap QuPath GeoJSON annotations from slide to cut-local coordinates.

The cut manifests produced by :mod:`src.data_preparation.extract_ndpi_cuts`
store each tissue cut's level-0 bounding box in whole-slide coordinates.  This
module assigns each oocyte annotation to one cut, subtracts that cut origin from
the annotation coordinates, and writes one cut-local GeoJSON per cut.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

FEATURE_COLLECTION = "FeatureCollection"
SUPPORTED_GEOMETRY_TYPES = {"LineString", "Polygon", "MultiPolygon"}


def remap_annotations(
    geojson_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    skip_if_exists: bool = False,
) -> dict[str, Any]:
    """Remap a GeoJSON annotation file from whole-slide to cut-local coordinates.

    Reads the annotation file and ``_cuts.json`` manifest, assigns each annotation
    to the cut whose bounding box contains the annotation's centroid, translates
    coordinates to cut-local space, and writes one output GeoJSON per cut plus a
    remapping report.

    Parameters
    ----------
    geojson_path : str or Path
        Path to the source GeoJSON file in whole-slide level-0 coordinates.
    manifest_path : str or Path
        Path to the ``{stem}_cuts.json`` manifest produced by cut extraction.
    output_dir : str or Path
        Directory to write output files into. Created if absent.
    skip_if_exists : bool, optional
        If True, skip slides whose report JSON already exists.

    Returns
    -------
    dict
        The remapping report, matching the JSON written to disk.

    Examples
    --------
    >>> report = {"n_assigned": 2, "n_unassigned": 0, "n_annotations_total": 2}
    >>> report["n_assigned"] + report["n_unassigned"] == report["n_annotations_total"]
    True
    """
    geojson_path = Path(geojson_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    stem = geojson_path.stem
    report_path = output_dir / f"{stem}_remap_report.json"

    if skip_if_exists and report_path.exists():
        LOG.info("Skipping %s because %s already exists", stem, report_path)
        return _read_json(report_path)

    manifest = _load_manifest(manifest_path)
    cuts = manifest["cuts"]

    geojson = _read_json(geojson_path)
    if geojson.get("type") != FEATURE_COLLECTION:
        raise ValueError(f"{geojson_path} is not a GeoJSON FeatureCollection")

    output_dir.mkdir(parents=True, exist_ok=True)
    features_by_cut: dict[int, list[dict[str, Any]]] = {cut["index"]: [] for cut in cuts}
    unassigned_ids: list[str] = []

    for feature_pos, feature in enumerate(geojson.get("features", [])):
        feature_id = str(feature.get("id", f"{stem}:{feature_pos}"))
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in SUPPORTED_GEOMETRY_TYPES:
            LOG.error(
                "Feature %s has unsupported geometry type %r",
                feature_id,
                geometry.get("type"),
            )
            unassigned_ids.append(feature_id)
            continue

        coords = _extract_ring_coords(geometry)
        cut_index, assignment_method = _assign_to_cut(coords, cuts)
        if cut_index is None:
            LOG.error("Feature %s could not be assigned to any cut", feature_id)
            unassigned_ids.append(feature_id)
            continue
        if assignment_method != "centroid_in_bbox":
            LOG.warning(
                "Feature %s assigned to cut %s with %s",
                feature_id,
                cut_index,
                assignment_method,
            )

        cut = _cut_by_index(cuts, cut_index)
        bbox = cut["level0_bbox"]
        local_coords = _translate_coords(coords, bbox["x0"], bbox["y0"])
        output_feature = _build_output_feature(
            feature,
            local_coords,
            cut_index,
            assignment_method,
            coords,
        )
        features_by_cut[cut_index].append(output_feature)

    n_total = len(geojson.get("features", []))
    n_unassigned = len(unassigned_ids)
    n_assigned = n_total - n_unassigned

    for cut in cuts:
        cut_features = features_by_cut[cut["index"]]
        cut_geojson = {
            "type": FEATURE_COLLECTION,
            "properties": {
                "source_geojson": geojson_path.name,
                "cut_name": cut["name"],
                "cut_index": cut["index"],
                "level0_bbox": cut["level0_bbox"],
                "n_annotations": len(cut_features),
                "n_unassigned": n_unassigned,
            },
            "features": cut_features,
        }
        output_path = output_dir / f"{cut['name']}_annotations.geojson"
        _write_json(output_path, cut_geojson)

    report = {
        "stem": stem,
        "source_geojson": str(geojson_path),
        "n_annotations_total": n_total,
        "n_assigned": n_assigned,
        "n_unassigned": n_unassigned,
        "unassigned_ids": unassigned_ids,
        "cuts": [
            {
                "cut_index": cut["index"],
                "cut_name": cut["name"],
                "n_assigned": len(features_by_cut[cut["index"]]),
            }
            for cut in cuts
        ],
    }
    _write_json(report_path, report)
    return report


def remap_annotations_batch(
    geojson_dir: str | Path,
    cuts_dir: str | Path,
    *,
    skip_if_exists: bool = True,
    glob_pattern: str = "*.geojson",
) -> list[dict[str, Any]]:
    """Run :func:`remap_annotations` for every GeoJSON in a directory.

    Parameters
    ----------
    geojson_dir : str or Path
        Directory containing source GeoJSON files, e.g. ``data/dataset_28_04/``.
    cuts_dir : str or Path
        Root directory containing per-stem subdirectories with ``_cuts.json``
        manifests, e.g. ``data/cuts/``.
    skip_if_exists : bool, optional
        Passed through to :func:`remap_annotations`. Default True for batch runs.
    glob_pattern : str, optional
        Glob pattern to select GeoJSON files.

    Returns
    -------
    list of dict
        One report dict per processed slide.

    Examples
    --------
    >>> isinstance([], list)
    True
    """
    geojson_dir = Path(geojson_dir)
    cuts_dir = Path(cuts_dir)
    reports: list[dict[str, Any]] = []

    for geojson_path in sorted(geojson_dir.glob(glob_pattern)):
        stem = geojson_path.stem
        manifest_path = cuts_dir / stem / f"{stem}_cuts.json"
        output_dir = cuts_dir / stem
        reports.append(
            remap_annotations(
                geojson_path,
                manifest_path,
                output_dir,
                skip_if_exists=skip_if_exists,
            )
        )
    return reports


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate a ``_cuts.json`` manifest. Raises ValueError if malformed."""
    manifest = _read_json(manifest_path)
    cuts = manifest.get("cuts")
    if not isinstance(cuts, list) or not cuts:
        raise ValueError(f"{manifest_path} must contain a non-empty 'cuts' list")

    for cut in cuts:
        if not isinstance(cut.get("index"), int):
            raise ValueError(f"Cut entry in {manifest_path} is missing integer 'index'")
        if not cut.get("name"):
            raise ValueError(f"Cut {cut.get('index')} in {manifest_path} is missing 'name'")
        bbox = cut.get("level0_bbox")
        if not isinstance(bbox, dict):
            raise ValueError(f"Cut {cut.get('index')} in {manifest_path} is missing bbox")
        for key in ("x0", "y0", "x1", "y1"):
            if not isinstance(bbox.get(key), int):
                raise ValueError(f"Cut {cut.get('index')} bbox key {key!r} must be an int")
    return manifest


def _extract_ring_coords(geometry: dict[str, Any]) -> list[list[float]]:
    """Return the flat list of ``[x, y]`` pairs for any supported geometry type.

    Parameters
    ----------
    geometry : dict
        GeoJSON geometry with type ``LineString``, ``Polygon``, or ``MultiPolygon``.

    Returns
    -------
    list of list of float
        Flat coordinate list with any z-values dropped.

    Examples
    --------
    >>> _extract_ring_coords({"type": "LineString", "coordinates": [[1, 2, 3], [4, 5]]})
    [[1.0, 2.0], [4.0, 5.0]]
    """
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if geometry_type == "LineString":
        return [_xy_pair(coord) for coord in coordinates or []]
    if geometry_type == "Polygon":
        rings = coordinates or []
        return [_xy_pair(coord) for coord in (rings[0] if rings else [])]
    if geometry_type == "MultiPolygon":
        rings = _multipolygon_outer_rings(coordinates or [])
        if not rings:
            return []
        areas = [_ring_area(ring) for ring in rings]
        largest_idx = max(range(len(areas)), key=areas.__getitem__)
        if len(rings) > 1:
            LOG.warning(
                "MultiPolygon with %d rings converted to Polygon; discarded %d smaller ring(s)",
                len(rings),
                len(rings) - 1,
            )
        return [_xy_pair(coord) for coord in rings[largest_idx]]

    raise ValueError(f"Unsupported geometry type: {geometry_type!r}")


def _compute_centroid(coords: list[list[float]]) -> tuple[float, float]:
    """Return the arithmetic centroid of a coordinate list.

    Examples
    --------
    >>> _compute_centroid([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    (5.0, 5.0)
    """
    if not coords:
        raise ValueError("Cannot compute centroid of an empty coordinate list")
    x_total = sum(float(x) for x, _ in coords)
    y_total = sum(float(y) for _, y in coords)
    return x_total / len(coords), y_total / len(coords)


def _point_in_bbox(x: float, y: float, bbox: dict[str, int]) -> bool:
    """Return True if ``(x, y)`` falls within ``bbox`` inclusively.

    Examples
    --------
    >>> _point_in_bbox(10.0, 5.0, {"x0": 0, "y0": 0, "x1": 10, "y1": 10})
    True
    """
    return bbox["x0"] <= x <= bbox["x1"] and bbox["y0"] <= y <= bbox["y1"]


def _assign_to_cut(
    coords: list[list[float]],
    cuts: list[dict[str, Any]],
) -> tuple[int | None, str]:
    """Return ``(cut_index, assignment_method)`` for a single annotation."""
    cx, cy = _compute_centroid(coords)
    centroid_matches = [
        cut for cut in cuts if _point_in_bbox(cx, cy, cut["level0_bbox"])
    ]
    if len(centroid_matches) == 1:
        return centroid_matches[0]["index"], "centroid_in_bbox"
    if len(centroid_matches) > 1:
        annotation_bbox = _coords_bbox(coords)
        selected = max(
            centroid_matches,
            key=lambda cut: _bbox_intersection_area(annotation_bbox, cut["level0_bbox"]),
        )
        LOG.warning(
            "Centroid %.2f, %.2f falls in %d cut bboxes; assigned to cut %s by overlap",
            cx,
            cy,
            len(centroid_matches),
            selected["index"],
        )
        return selected["index"], "centroid_in_bbox_multiple"

    vertex_counts = []
    for cut in cuts:
        bbox = cut["level0_bbox"]
        n_vertices = sum(_point_in_bbox(x, y, bbox) for x, y in coords)
        if n_vertices:
            vertex_counts.append((n_vertices, cut))
    if vertex_counts:
        n_vertices, selected = max(vertex_counts, key=lambda item: item[0])
        LOG.warning(
            "Centroid %.2f, %.2f outside all cut bboxes; assigned to cut %s by %d vertices",
            cx,
            cy,
            selected["index"],
            n_vertices,
        )
        return selected["index"], "vertex_in_bbox"

    LOG.error("Centroid %.2f, %.2f and all vertices are outside cut bboxes", cx, cy)
    return None, "unassigned"


def _translate_coords(
    coords: list[list[float]],
    x0: int,
    y0: int,
) -> list[list[float]]:
    """Subtract cut origin from every coordinate pair.

    Parameters
    ----------
    coords : list of list of float
        Slide-level ``[x, y]`` coordinates.
    x0, y0 : int
        Cut's level-0 bbox origin.

    Returns
    -------
    list of list of float
        Cut-local coordinates.

    Examples
    --------
    >>> _translate_coords([[38541.76, 9591.36]], 4200, 1800)
    [[34341.76, 7791.36]]
    """
    return [[round(float(x) - x0, 10), round(float(y) - y0, 10)] for x, y in coords]


def _build_output_feature(
    original_feature: dict[str, Any],
    local_coords: list[list[float]],
    cut_index: int,
    assignment_method: str,
    original_coords: list[list[float]],
) -> dict[str, Any]:
    """Construct a GeoJSON feature with cut-local coordinates."""
    output = {
        "type": "Feature",
        "geometry": _build_output_geometry(original_feature.get("geometry") or {}, local_coords),
        "properties": copy.deepcopy(original_feature.get("properties") or {}),
    }
    if "id" in original_feature:
        output["id"] = original_feature["id"]

    output["properties"].update(
        {
            "source_slide_coords": copy.deepcopy(original_coords),
            "cut_index": cut_index,
            "assignment_method": assignment_method,
        }
    )
    return output


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
        fp.write("\n")


def _xy_pair(coord: list[float] | tuple[float, ...]) -> list[float]:
    if len(coord) < 2:
        raise ValueError(f"Coordinate must contain at least x and y values: {coord!r}")
    return [float(coord[0]), float(coord[1])]


def _multipolygon_outer_rings(coordinates: list[Any]) -> list[list[Any]]:
    rings = []
    for polygon in coordinates:
        if polygon:
            rings.append(polygon[0])
    return rings


def _ring_area(ring: list[Any]) -> float:
    points = [_xy_pair(coord) for coord in ring]
    if len(points) < 3:
        return 0.0
    twice_area = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        twice_area += x0 * y1 - y0 * x1
    return abs(twice_area) / 2.0


def _coords_bbox(coords: list[list[float]]) -> dict[str, float]:
    xs = [float(x) for x, _ in coords]
    ys = [float(y) for _, y in coords]
    return {
        "x0": min(xs),
        "y0": min(ys),
        "x1": max(xs),
        "y1": max(ys),
    }


def _bbox_intersection_area(a: dict[str, float], b: dict[str, int]) -> float:
    width = max(0.0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    height = max(0.0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    return width * height


def _cut_by_index(cuts: list[dict[str, Any]], cut_index: int) -> dict[str, Any]:
    for cut in cuts:
        if cut["index"] == cut_index:
            return cut
    raise ValueError(f"Cut index {cut_index} not found in manifest")


def _build_output_geometry(
    original_geometry: dict[str, Any],
    local_coords: list[list[float]],
) -> dict[str, Any]:
    original_type = original_geometry.get("type")
    if original_type == "LineString":
        return {"type": "LineString", "coordinates": local_coords}
    if original_type in {"Polygon", "MultiPolygon"}:
        return {"type": "Polygon", "coordinates": [local_coords]}
    raise ValueError(f"Unsupported geometry type: {original_type!r}")
