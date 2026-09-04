"""Synthetic tests for geometry-derived tile oocyte labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.tile_labels import (
    LABEL_AMBIGUOUS,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    compute_oocyte_area_fractions,
    label_for_area_fraction,
    load_annotation_polygons,
    oocyte_area_fractions_for_manifest,
)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp)


def _square(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """Closed axis-aligned ring."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _feature(coords: list[list[float]], geometry_type: str = "LineString") -> dict:
    if geometry_type == "LineString":
        geometry = {"type": "LineString", "coordinates": coords}
    elif geometry_type == "Polygon":
        geometry = {"type": "Polygon", "coordinates": [coords]}
    else:
        raise AssertionError(f"unexpected geometry_type {geometry_type!r}")
    return {
        "type": "Feature",
        "properties": {"name": "Oosorption Stage 0"},
        "geometry": geometry,
    }


def _geojson(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def _tile(tile_id: str, x0: int, y0: int, size: int, *, x1=None, y1=None) -> dict:
    x1 = x0 + size if x1 is None else x1
    y1 = y0 + size if y1 is None else y1
    return {
        "tile_id": tile_id,
        "tile_size": size,
        "cut_local_bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "pad_right": size - (x1 - x0),
        "pad_bottom": size - (y1 - y0),
        "has_oocyte": False,
        "n_oocytes": 0,
    }


def _manifest(tiles: list[dict]) -> dict:
    return {"cut_name": "TEST_cut000", "stem": "TEST", "tiles": tiles}


def _fixture(tmp_path: Path, tiles: list[dict], features: list[dict]) -> tuple[Path, Path]:
    manifest_path = tmp_path / "TEST_cut000_tile_manifest.json"
    geojson_path = tmp_path / "TEST_cut000_annotations.geojson"
    _write_json(manifest_path, _manifest(tiles))
    _write_json(geojson_path, _geojson(features))
    return manifest_path, geojson_path


def test_linestring_ring_is_closed_into_a_polygon(tmp_path: Path) -> None:
    """The trap this module exists for: LineString rings must not measure as zero area.

    shapely.geometry.shape() on a LineString gives a zero-area geometry, which would
    make every label negative with no error raised.
    """
    polygons = load_annotation_polygons(
        _fixture(tmp_path, [], [_feature(_square(0, 0, 10, 10))])[1]
    )
    assert len(polygons) == 1
    assert polygons[0].area == pytest.approx(100.0)


def test_tile_fully_inside_a_large_annotation_is_fully_covered(tmp_path: Path) -> None:
    """The case the centroid flag gets wrong.

    A 512 px tile in the middle of a 2000 px oocyte contains no centroid, so the
    manifest calls it negative, but it is entirely oocyte.
    """
    tiles = [_tile("inside", 700, 700, 512)]
    features = [_feature(_square(0, 0, 2000, 2000))]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, features)

    fractions = compute_oocyte_area_fractions(manifest_path, geojson_path)

    assert fractions["inside"] == pytest.approx(1.0)
    assert tiles[0]["has_oocyte"] is False
    assert label_for_area_fraction(fractions["inside"]) == LABEL_POSITIVE


def test_partial_overlap_reports_the_covered_fraction(tmp_path: Path) -> None:
    tiles = [_tile("quarter", 0, 0, 100)]
    features = [_feature(_square(0, 0, 50, 50))]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, features)

    fractions = compute_oocyte_area_fractions(manifest_path, geojson_path)

    assert fractions["quarter"] == pytest.approx(0.25)


def test_overlapping_annotations_are_not_double_counted(tmp_path: Path) -> None:
    """Two annotations covering the same half of a tile give 0.5, not 1.0."""
    tiles = [_tile("half", 0, 0, 100)]
    features = [
        _feature(_square(0, 0, 100, 50)),
        _feature(_square(0, 0, 100, 50)),
    ]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, features)

    fractions = compute_oocyte_area_fractions(manifest_path, geojson_path)

    assert fractions["half"] == pytest.approx(0.5)


def test_disjoint_annotations_sum(tmp_path: Path) -> None:
    tiles = [_tile("two", 0, 0, 100)]
    features = [
        _feature(_square(0, 0, 20, 20)),
        _feature(_square(80, 80, 100, 100)),
    ]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, features)

    fractions = compute_oocyte_area_fractions(manifest_path, geojson_path)

    assert fractions["two"] == pytest.approx(0.08)


def test_tile_away_from_every_annotation_is_zero(tmp_path: Path) -> None:
    tiles = [_tile("far", 5000, 5000, 512)]
    features = [_feature(_square(0, 0, 100, 100))]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, features)

    fractions = compute_oocyte_area_fractions(manifest_path, geojson_path)

    assert fractions["far"] == 0.0
    assert label_for_area_fraction(fractions["far"]) == LABEL_NEGATIVE


def test_polygon_geometry_is_supported(tmp_path: Path) -> None:
    tiles = [_tile("poly", 0, 0, 100)]
    features = [_feature(_square(0, 0, 50, 100), geometry_type="Polygon")]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, features)

    fractions = compute_oocyte_area_fractions(manifest_path, geojson_path)

    assert fractions["poly"] == pytest.approx(0.5)


def test_features_present_but_none_usable_raises(tmp_path: Path) -> None:
    """Silently reporting no oocytes for an annotated cut is the failure to prevent."""
    _, geojson_path = _fixture(tmp_path, [], [_feature([[0, 0], [1, 1]])])

    with pytest.raises(ValueError, match="none yielded a usable polygon"):
        load_annotation_polygons(geojson_path)


def test_no_features_is_not_an_error(tmp_path: Path) -> None:
    """One cut in the corpus genuinely has no annotations."""
    tiles = [_tile("empty", 0, 0, 512)]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, [])

    assert load_annotation_polygons(geojson_path) == []
    assert compute_oocyte_area_fractions(manifest_path, geojson_path) == {"empty": 0.0}


def test_not_a_feature_collection_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.geojson"
    _write_json(path, {"type": "Feature", "features": []})

    with pytest.raises(ValueError, match="not a GeoJSON FeatureCollection"):
        load_annotation_polygons(path)


def test_self_intersecting_ring_keeps_its_whole_area(tmp_path: Path) -> None:
    """57% of the corpus's rings self-intersect, so the repair must not lose area.

    buffer(0) keeps only the correctly-oriented lobe of this bowtie and returns 25 of
    its 50 px. Asserting merely area > 0 would pass while half the oocyte vanished,
    understating coverage and flipping tiles to negative.
    """
    bowtie = [[0, 0], [10, 10], [10, 0], [0, 10], [0, 0]]
    polygons = load_annotation_polygons(_fixture(tmp_path, [], [_feature(bowtie)])[1])

    assert len(polygons) == 1
    assert polygons[0].is_valid
    assert polygons[0].area == pytest.approx(50.0)


def test_unclosed_three_vertex_ring_is_accepted(tmp_path: Path) -> None:
    """generate_tiles.py counts these toward has_oocyte, so they must not be dropped."""
    triangle = [[0, 0], [10, 0], [0, 10]]
    polygons = load_annotation_polygons(_fixture(tmp_path, [], [_feature(triangle)])[1])

    assert len(polygons) == 1
    assert polygons[0].area == pytest.approx(50.0)


def test_degenerate_bbox_raises_even_without_annotations(tmp_path: Path) -> None:
    """A malformed manifest must fail the same way regardless of its cut's annotations."""
    tiles = [_tile("flat", 0, 0, 100, x1=0, y1=100)]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, [])

    with pytest.raises(ValueError, match="degenerate bbox"):
        compute_oocyte_area_fractions(manifest_path, geojson_path)


def test_padded_edge_tile_measures_against_its_clipped_bbox(tmp_path: Path) -> None:
    """The denominator is the real image area, not the padded tile size."""
    tiles = [_tile("edge", 0, 0, 100, x1=50, y1=100)]
    features = [_feature(_square(0, 0, 25, 100))]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, features)

    fractions = compute_oocyte_area_fractions(manifest_path, geojson_path)

    # Oocyte covers 2500 of the tile's real 5000 px, not 2500 of a nominal 10000.
    assert tiles[0]["pad_right"] == 50
    assert fractions["edge"] == pytest.approx(0.5)


def test_degenerate_bbox_raises(tmp_path: Path) -> None:
    tiles = [_tile("flat", 0, 0, 100, x1=0, y1=100)]
    features = [_feature(_square(0, 0, 50, 50))]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, features)

    with pytest.raises(ValueError, match="degenerate bbox"):
        compute_oocyte_area_fractions(manifest_path, geojson_path)


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.0, LABEL_NEGATIVE),
        (0.0001, LABEL_AMBIGUOUS),
        (0.049, LABEL_AMBIGUOUS),
        (0.05, LABEL_POSITIVE),
        (0.5, LABEL_POSITIVE),
        (1.0, LABEL_POSITIVE),
    ],
)
def test_label_bands_at_the_default_threshold(fraction: float, expected: str) -> None:
    assert label_for_area_fraction(fraction) == expected


def test_threshold_of_zero_makes_any_contact_positive() -> None:
    assert label_for_area_fraction(0.0, min_oocyte_area_fraction=0.0) == LABEL_NEGATIVE
    assert label_for_area_fraction(1e-9, min_oocyte_area_fraction=0.0) == LABEL_POSITIVE


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError, match=r"must be in the range \[0, 1\]"):
        label_for_area_fraction(0.5, min_oocyte_area_fraction=1.5)


def test_negative_fraction_raises() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        label_for_area_fraction(-0.1)


def test_mismatched_cut_raises(tmp_path: Path) -> None:
    """Pairing a manifest with another cut's annotations must not label silently."""
    manifest_path, _ = _fixture(tmp_path, [_tile("t", 0, 0, 100)], [])
    other = tmp_path / "OTHER_cut001_annotations.geojson"
    _write_json(other, _geojson([_feature(_square(0, 0, 50, 50))]))

    with pytest.raises(ValueError, match="does not belong to cut 'TEST_cut000'"):
        compute_oocyte_area_fractions(manifest_path, other)


def test_manifest_without_cut_name_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "TEST_cut000_tile_manifest.json"
    geojson_path = tmp_path / "TEST_cut000_annotations.geojson"
    _write_json(manifest_path, {"stem": "TEST", "tiles": []})
    _write_json(geojson_path, _geojson([]))

    with pytest.raises(ValueError, match="has no cut_name"):
        compute_oocyte_area_fractions(manifest_path, geojson_path)


def test_polygon_with_interior_rings_is_refused(tmp_path: Path) -> None:
    """The shared ring extractor keeps only ring 0, which would ignore holes."""
    path = tmp_path / "TEST_cut000_annotations.geojson"
    _write_json(
        path,
        _geojson(
            [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            _square(0, 0, 100, 100),
                            _square(40, 40, 60, 60),
                        ],
                    },
                }
            ]
        ),
    )

    with pytest.raises(ValueError, match="interior ring"):
        load_annotation_polygons(path)


def test_multipolygon_is_refused(tmp_path: Path) -> None:
    """The extractor keeps only the largest part, which would understate coverage."""
    path = tmp_path / "TEST_cut000_annotations.geojson"
    _write_json(
        path,
        _geojson(
            [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [
                            [_square(0, 0, 10, 10)],
                            [_square(50, 50, 90, 90)],
                        ],
                    },
                }
            ]
        ),
    )

    with pytest.raises(ValueError, match="MultiPolygon"):
        load_annotation_polygons(path)


def test_parsed_manifest_entry_point_matches_the_path_one(tmp_path: Path) -> None:
    """load_tile_index uses the parsed-manifest form to avoid a second 636 MB parse."""
    tiles = [_tile("a", 0, 0, 100), _tile("b", 500, 500, 100)]
    features = [_feature(_square(0, 0, 50, 50))]
    manifest_path, geojson_path = _fixture(tmp_path, tiles, features)

    from_path = compute_oocyte_area_fractions(manifest_path, geojson_path)
    from_dict = oocyte_area_fractions_for_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")), geojson_path
    )

    assert from_path == from_dict
