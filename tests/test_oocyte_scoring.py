"""Per-oocyte scoring geometry on a hand-built grid whose numbers can be checked on paper."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.oocyte_scoring import (
    CellGrid,
    cell_grid,
    cut_geometry,
    load_oocytes,
    oocyte_coverage,
    tile_stride,
)
from src.modeling.tile_labels import load_annotation_features, load_annotation_polygons

# Stride 8, offset 1; 4x3 cells.
TILE = 10
WIDTH, HEIGHT = 30, 20
MISSING_TILE = (2, 3)


def _square(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _manifest(skip: tuple[int, int] | None = MISSING_TILE, shift: tuple[int, int] = (0, 0)) -> dict:
    tiles = []
    for row in range(3):
        for col in range(4):
            if (row, col) == skip:
                continue
            x0, y0 = col * 8, row * 8
            if (row, col) == (0, 0):
                x0, y0 = x0 + shift[0], y0 + shift[1]
            x1, y1 = min(x0 + TILE, WIDTH), min(y0 + TILE, HEIGHT)
            tiles.append({"tile_id": f"t_r{row}_c{col}", "tile_size": TILE, "row": row, "col": col,
                          "cut_local_bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                          "pad_right": TILE - (x1 - x0), "pad_bottom": TILE - (y1 - y0)})
    return {"cut_name": "S_A_1_1-2_cut000", "stem": "S_A_1_1-2", "overlap": 0.2,
            "mpp_x": 0.5, "mpp_y": 0.5,
            "level0_bbox": {"x0": 100, "y0": 200, "x1": 100 + WIDTH, "y1": 200 + HEIGHT},
            "tiles": tiles}


def _geojson(tmp_path: Path, polygons: dict[str, list[list[float]]]) -> Path:
    features = [{"type": "Feature", "properties": {"name": name},
                 "geometry": {"type": "Polygon", "coordinates": [ring]}}
                for name, ring in polygons.items()]
    path = tmp_path / "S_A_1_1-2_cut000_annotations.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}),
                    encoding="utf-8")
    return path


OOCYTES = {
    "Oosorption Stage 0": _square(10, 10, 14, 14),
    "Oosorption Stage 4": _square(7, 7, 11, 11),
    "Oosorption Stage 2": _square(26, 17, 30, 20),
}


@pytest.fixture
def geometry(tmp_path: Path):
    manifest = _manifest()
    scored = {tile["tile_id"] for tile in manifest["tiles"]}
    return cut_geometry(manifest, _geojson(tmp_path, OOCYTES), TILE, scored)


def test_cells_partition_the_cut_and_stretch_only_at_its_edges() -> None:
    grid = CellGrid(tile_size=TILE, stride=8, width=WIDTH, height=HEIGHT)

    assert (grid.n_rows, grid.n_cols, grid.offset) == (3, 4, 1.0)
    assert [grid.cell_box(0, col)[::2] for col in range(4)] == [(0, 9), (9, 17), (17, 25), (25, 30)]
    assert [grid.cell_box(row, 0)[1::2] for row in range(3)] == [(0, 9), (9, 17), (17, 20)]
    total = sum((x1 - x0) * (y1 - y0) for row in range(3) for col in range(4)
                for x0, y0, x1, y1 in [grid.cell_box(row, col)])
    assert total == WIDTH * HEIGHT


def test_a_single_column_cut_has_one_cell_spanning_its_width() -> None:
    grid = CellGrid(tile_size=TILE, stride=8, width=5, height=5)

    assert grid.cell_box(0, 0) == (0.0, 0.0, 5.0, 5.0)


def test_the_stride_matches_tile_generation() -> None:
    assert (tile_stride(512, 0.2), tile_stride(1024, 0.2), tile_stride(TILE, 0.2)) == (410, 819, 8)


def test_a_tile_off_its_grid_position_is_refused() -> None:
    with pytest.raises(ValueError, match="not at its grid position"):
        cell_grid(_manifest(shift=(1, 0)), TILE)


def test_a_kept_tile_without_a_probability_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    scored = {tile["tile_id"] for tile in manifest["tiles"]} - {"t_r1_c1"}

    with pytest.raises(ValueError, match="have no held-out probability"):
        cut_geometry(manifest, _geojson(tmp_path, OOCYTES), TILE, scored)


def test_a_probability_for_a_tile_not_in_the_manifest_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    scored = {tile["tile_id"] for tile in manifest["tiles"]} | {"t_r2_c3"}

    with pytest.raises(ValueError, match="not in its manifest"):
        cut_geometry(manifest, _geojson(tmp_path, OOCYTES), TILE, scored)


def test_overlaps_follow_the_cells_and_skip_unscored_ones(geometry) -> None:
    overlaps = {(row.tile_id, row.oocyte): row.area for row in geometry.overlaps.itertuples()}

    assert overlaps[("t_r1_c1", 0)] == 16
    assert {k: v for k, v in overlaps.items() if k[1] == 1} == {
        ("t_r0_c0", 1): 4, ("t_r0_c1", 1): 4, ("t_r1_c0", 1): 4, ("t_r1_c1", 1): 4,
    }
    assert not any(oocyte == 2 for _, oocyte in overlaps)
    oocytes = geometry.oocytes.set_index("oocyte")
    assert list(oocytes["area"]) == [16, 16, 12]
    assert list(oocytes["scorable_area"]) == [16, 16, 0]


def test_annotation_area_counts_overlapping_oocytes_once(geometry) -> None:
    cells = geometry.cells.set_index("tile_id")

    assert cells.loc["t_r1_c1", "annotation_area"] == 16 + 4 - 1
    assert cells.loc["t_r1_c1", "area"] == 64
    assert geometry.cells["area"].sum() == WIDTH * HEIGHT - 5 * 3


def test_oocytes_keep_their_stage_and_size(geometry) -> None:
    oocytes = geometry.oocytes.set_index("oocyte")

    assert list(oocytes["stage"]) == [0, 4, 2]
    assert oocytes.loc[0, "equivalent_diameter_um"] == pytest.approx(2 * math.sqrt(16 * 0.25 / math.pi))
    assert set(oocytes["cut_name"]) == {"S_A_1_1-2_cut000"}


def test_coverage_counts_positive_cells_against_the_scorable_area(geometry) -> None:
    probabilities = pd.Series({tile: 0.1 for tile in geometry.cells["tile_id"]})
    probabilities[["t_r1_c1", "t_r0_c0"]] = 0.9

    coverage = oocyte_coverage(geometry, probabilities, threshold=0.5)

    assert coverage[0] == 1.0
    assert coverage[1] == 0.5
    assert math.isnan(coverage[2])


def test_coverage_includes_cells_exactly_at_the_threshold(geometry) -> None:
    probabilities = pd.Series({tile: 0.5 for tile in geometry.cells["tile_id"]})

    assert oocyte_coverage(geometry, probabilities, threshold=0.5)[1] == 1.0
    assert oocyte_coverage(geometry, probabilities, threshold=0.5000001)[1] == 0.0


def test_annotation_features_match_the_polygons_tile_labels_measure(tmp_path: Path) -> None:
    path = _geojson(tmp_path, OOCYTES)

    features = load_annotation_features(path)

    assert [feature.polygon.equals(other) for feature, other in
            zip(features, load_annotation_polygons(path))] == [True] * 3
    assert [feature.properties["name"] for feature in features] == list(OOCYTES)
    assert [feature.position for feature in features] == [0, 1, 2]


def test_an_annotation_without_a_name_has_an_unknown_stage(tmp_path: Path) -> None:
    path = tmp_path / "S_A_1_1-2_cut000_annotations.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Polygon", "coordinates": [_square(0, 0, 4, 4)]}}]}),
        encoding="utf-8")

    assert load_oocytes(path, 0.5, 0.5)["stage"].isna().all()


@pytest.mark.parametrize("width", [25, 26, 17, 9, 33])
def test_cells_tile_the_cut_exactly_when_the_last_tile_starts_near_the_edge(width: int) -> None:
    grid = CellGrid(tile_size=TILE, stride=8, width=width, height=HEIGHT)

    spans = [grid.cell_box(0, col)[::2] for col in range(grid.n_cols)]

    assert spans[0][0] == 0 and spans[-1][1] == width
    assert all(low <= high <= width for low, high in spans)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
    assert sum(high - low for low, high in spans) == width


def test_a_last_tile_starting_within_the_offset_of_the_edge_owns_an_empty_cell() -> None:
    grid = CellGrid(tile_size=512, stride=410, width=850, height=430)

    assert [grid.cell_box(0, col)[::2] for col in range(grid.n_cols)] == [
        (0.0, 461.0), (461.0, 850.0), (850.0, 850.0)]
    assert [grid.cell_box(row, 0)[1::2] for row in range(grid.n_rows)] == [(0.0, 430.0), (430.0, 430.0)]


def test_stage_follows_the_project_parser_when_name_and_description_disagree(tmp_path: Path) -> None:
    path = tmp_path / "S_A_1_1-2_cut000_annotations.geojson"
    properties = {"name": "Oosorption Stage 3",
                  "metadata": {"ANNOTATION_DESCRIPTION": "Oosorption Stage 0"}}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": properties,
         "geometry": {"type": "Polygon", "coordinates": [_square(0, 0, 4, 4)]}}]}),
        encoding="utf-8")

    assert list(load_oocytes(path, 0.5, 0.5)["stage"]) == [0]


def test_oocyte_ids_keep_their_geojson_position_when_a_feature_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "S_A_1_1-2_cut000_annotations.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "Oosorption Stage 0"},
         "geometry": {"type": "Polygon", "coordinates": [_square(0, 0, 4, 4)]}},
        {"type": "Feature", "properties": {"name": "Oosorption Stage 1"},
         "geometry": {"type": "Polygon", "coordinates": [[[5, 5], [6, 6]]]}},
        {"type": "Feature", "properties": {"name": "Oosorption Stage 4"},
         "geometry": {"type": "Polygon", "coordinates": [_square(10, 10, 14, 14)]}}]}),
        encoding="utf-8")

    oocytes = load_oocytes(path, 0.5, 0.5)

    assert list(oocytes["oocyte"]) == [0, 2]
    assert list(oocytes["stage"]) == [0, 4]


@pytest.mark.parametrize("gap", ["missing", "nan"])
def test_coverage_refuses_a_cell_under_an_oocyte_without_a_probability(geometry, gap: str) -> None:
    probabilities = pd.Series({tile: 0.9 for tile in geometry.cells["tile_id"]})
    if gap == "missing":
        probabilities = probabilities.drop("t_r0_c1")
    else:
        probabilities["t_r0_c1"] = float("nan")

    with pytest.raises(ValueError, match="have no probability"):
        oocyte_coverage(geometry, probabilities, threshold=0.5)


def test_a_tile_listed_beyond_the_cut_is_refused() -> None:
    manifest = _manifest()
    manifest["tiles"].append({"tile_id": "t_r0_c4", "tile_size": TILE, "row": 0, "col": 4,
                              "cut_local_bbox": {"x0": 32, "y0": 0, "x1": 42, "y1": 10}})

    with pytest.raises(ValueError, match="outside the cut"):
        cell_grid(manifest, TILE)


@pytest.mark.parametrize("change", ["narrower_image", "wider_image"])
def test_tiles_cut_from_an_image_of_another_size_are_refused(change: str) -> None:
    manifest = _manifest()
    edge = next(t for t in manifest["tiles"] if t["tile_id"] == "t_r0_c3")
    if change == "narrower_image":
        edge["cut_local_bbox"]["x1"] = 28
        edge["pad_right"] = TILE - (28 - 24)
    else:
        edge["cut_local_bbox"]["x1"] = 34
        edge["pad_right"] = 0

    with pytest.raises(ValueError, match="tiled at another size"):
        cell_grid(manifest, TILE)


def test_a_manifest_without_a_pixel_size_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["mpp_x"] = None
    scored = {tile["tile_id"] for tile in manifest["tiles"]}

    with pytest.raises(ValueError, match="records no pixel size"):
        cut_geometry(manifest, _geojson(tmp_path, OOCYTES), TILE, scored)


def test_coverage_accepts_probabilities_with_a_blank_for_a_cell_under_no_oocyte(geometry) -> None:
    probabilities = pd.Series({tile: 0.9 for tile in geometry.cells["tile_id"]}, dtype=object)
    probabilities["t_r2_c0"] = None

    assert oocyte_coverage(geometry, probabilities, threshold=0.5)[0] == 1.0


def test_coverage_is_numeric_when_no_oocyte_has_a_scored_cell(tmp_path: Path) -> None:
    manifest = _manifest()
    scored = {tile["tile_id"] for tile in manifest["tiles"]}
    geometry = cut_geometry(manifest, _geojson(tmp_path, {"Oosorption Stage 2": _square(26, 17, 30, 20)}),
                            TILE, scored)
    probabilities = pd.Series({tile: 0.9 for tile in scored})

    coverage = oocyte_coverage(geometry, probabilities, threshold=0.5)

    assert coverage.dtype == "float64" and coverage.isna().all()
    assert geometry.overlaps["area"].dtype == "float64"
