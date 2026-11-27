"""Per-oocyte scoring geometry on a hand-built grid whose numbers can be checked on paper."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.oocyte_scoring import (
    CellGrid,
    CutGeometry,
    cell_grid,
    cut_geometry,
    false_blob_counts,
    fold_curve,
    found_thresholds,
    load_oocytes,
    oocyte_coverage,
    operating_point,
    per_oocyte_table,
    recall_by,
    size_bin_edges,
    summarise_operating_points,
    threshold_grid,
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



SCENARIO = {"t_r1_c1": 0.9, "t_r0_c0": 0.8, "t_r0_c1": 0.3, "t_r1_c0": 0.2,
            "t_r0_c3": 0.7, "t_r1_c3": 0.7}
SCORED_PX = WIDTH * HEIGHT - 5 * 3
UM2_PER_PX = 0.25


def _scenario(geometry) -> pd.Series:
    return pd.Series({tile: SCENARIO.get(tile, 0.1) for tile in geometry.cells["tile_id"]})


def test_found_threshold_is_the_probability_where_coverage_reaches_half(geometry) -> None:
    found = found_thresholds(geometry, _scenario(geometry))

    assert found[0] == 0.9
    assert found[1] == 0.8
    assert math.isnan(found[2])


def test_coverage_of_exactly_half_counts_as_found(geometry) -> None:
    probabilities = _scenario(geometry)
    probabilities["t_r0_c0"] = 0.85

    assert oocyte_coverage(geometry, probabilities, threshold=0.85)[1] == 0.5
    assert found_thresholds(geometry, probabilities)[1] == 0.85


def test_found_threshold_refuses_a_cell_under_an_oocyte_without_a_probability(geometry) -> None:
    with pytest.raises(ValueError, match="have no probability"):
        found_thresholds(geometry, _scenario(geometry).drop("t_r0_c1"))


def test_per_oocyte_table_carries_fold_stage_size_and_found_threshold(geometry) -> None:
    table = per_oocyte_table(geometry, _scenario(geometry), fold=3)

    assert list(table["fold"]) == [3, 3, 3]
    assert list(table["found_threshold"].fillna(-1)) == [0.9, 0.8, -1]
    assert "polygon" not in table


def test_false_blobs_are_connected_positive_groups_touching_no_annotation(geometry) -> None:
    counts, multi_cell = false_blob_counts(geometry, _scenario(geometry),
                                           np.array([0.95, 0.75, 0.65, 0.05]))

    assert list(counts) == [0, 0, 1, 0]
    assert list(multi_cell) == [0, 0, 1, 0]


def test_fold_curve_gives_recall_false_blob_density_and_spill(geometry) -> None:
    probabilities = _scenario(geometry)
    oocytes = per_oocyte_table(geometry, probabilities, fold=1)

    curve = fold_curve([geometry], probabilities, oocytes, np.array([0.65, 0.75, 0.85]))

    assert list(curve["n_scorable"]) == [2, 2, 2]
    assert list(curve["recall"]) == [1.0, 1.0, 0.5]
    assert list(curve["false_blobs"]) == [1, 0, 0]
    assert list(curve["false_blobs_2plus"]) == [1, 0, 0]
    assert curve.loc[0, "false_blobs_2plus_per_cm2"] == curve.loc[0, "false_blobs_per_cm2"]
    assert curve.loc[0, "false_blobs_per_cm2"] == pytest.approx(1 / (SCORED_PX * UM2_PER_PX / 1e8))
    positive_px = {0.65: 81 + 64 + 45 + 40, 0.75: 81 + 64}
    annotation_px = 4 + 19
    for row, threshold in enumerate((0.65, 0.75)):
        assert curve.loc[row, "positive_um2"] == pytest.approx(positive_px[threshold] * UM2_PER_PX)
        assert curve.loc[row, "spill"] == pytest.approx(
            (positive_px[threshold] - annotation_px) / positive_px[threshold])


def test_spill_is_undefined_when_no_cell_is_positive(geometry) -> None:
    probabilities = _scenario(geometry)
    oocytes = per_oocyte_table(geometry, probabilities, fold=1)

    curve = fold_curve([geometry], probabilities, oocytes, np.array([1.0]))

    assert math.isnan(curve.loc[0, "spill"]) and curve.loc[0, "positive_um2"] == 0


def test_operating_point_takes_the_largest_threshold_reaching_the_recall(geometry) -> None:
    probabilities = _scenario(geometry)
    curve = fold_curve([geometry], probabilities, per_oocyte_table(geometry, probabilities, 1),
                       np.array([0.65, 0.75, 0.85]))

    point = operating_point(curve, 0.8)

    assert point["threshold"] == 0.75 and point["recall"] == 1.0
    assert point["false_blobs_per_cm2"] == 0 and point["false_blobs_2plus_per_cm2"] == 0
    assert point["spill"] == pytest.approx((145 - 23) / 145)
    assert operating_point(curve, 0.5)["threshold"] == 0.85
    assert operating_point(curve, 1.01) is None


def test_summary_names_folds_that_never_reach_a_target_and_leaves_them_out() -> None:
    curves = {
        1: pd.DataFrame({"threshold": [0.2, 0.6], "recall": [0.95, 0.85], "n_scorable": [5, 5],
                         "false_blobs_per_cm2": [4.0, 2.0], "spill": [0.5, 0.3],
                         "false_blobs_2plus_per_cm2": [1.0, 0.5]}),
        2: pd.DataFrame({"threshold": [0.2, 0.6], "recall": [0.82, 0.7], "n_scorable": [5, 5],
                         "false_blobs_per_cm2": [6.0, 3.0], "spill": [0.6, 0.4],
                         "false_blobs_2plus_per_cm2": [3.0, 1.5]}),
        3: pd.DataFrame({"threshold": [0.2, 0.6], "recall": [np.nan, np.nan], "n_scorable": [0, 0],
                         "false_blobs_per_cm2": [1.0, 1.0], "spill": [0.1, 0.1],
                         "false_blobs_2plus_per_cm2": [1.0, 1.0]}),
    }

    summary = summarise_operating_points(curves, targets=(0.8, 0.9))

    assert summary["recall_0.80"]["false_blobs_per_cm2"] == {"mean": 4.0, "min": 2.0, "max": 6.0}
    assert summary["recall_0.80"]["false_blobs_2plus_per_cm2"] == {"mean": 1.75, "min": 0.5, "max": 3.0}
    assert summary["recall_0.80"]["folds_not_reaching"] == []
    assert summary["recall_0.90"]["folds_not_reaching"] == [2]
    assert summary["recall_0.90"]["spill"] == {"mean": 0.5, "min": 0.5, "max": 0.5}
    assert summary["recall_0.80"]["folds_without_judged_oocytes"] == [3]


def test_the_threshold_grid_is_ascending_logit_spaced_and_closed() -> None:
    grid = threshold_grid(n=5, low=0.01)

    assert grid[0] == 0.0 and grid[-1] == 1.0 and len(grid) == 7
    assert np.all(np.diff(grid) > 0)
    logits = np.log(grid[1:-1] / (1 - grid[1:-1]))
    assert np.allclose(np.diff(logits), np.diff(logits)[0])
    assert grid[1] == pytest.approx(0.01) and grid[-2] == pytest.approx(0.99)


def test_size_bins_are_quartiles_open_at_both_ends() -> None:
    edges = size_bin_edges(pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, np.nan]))

    assert edges == [-math.inf, 20.0, 30.0, 40.0, math.inf]


def test_recall_by_judges_each_oocyte_at_its_folds_threshold() -> None:
    oocytes = pd.DataFrame({"fold": [1, 1, 2, 2, 3], "stage": [0, 4, 0, 0, 0],
                            "found_threshold": [0.5, 0.2, 0.7, np.nan, 0.9]})

    table = recall_by(oocytes, {1: 0.4, 2: 0.8}, "stage")

    assert table.loc[0].to_dict() == {"n_scorable": 2, "n_found": 1, "recall": 0.5}
    assert table.loc[4].to_dict() == {"n_scorable": 1, "n_found": 0, "recall": 0.0}



def test_coverage_between_forty_and_fifty_percent_is_not_found(tmp_path: Path) -> None:
    manifest = _manifest()
    scored = {tile["tile_id"] for tile in manifest["tiles"]}
    geometry = cut_geometry(manifest, _geojson(tmp_path, {"Oosorption Stage 0": _square(5, 10, 14, 14)}),
                            TILE, scored)
    probabilities = pd.Series({tile: 0.1 for tile in scored})
    probabilities["t_r1_c0"] = 0.9

    assert oocyte_coverage(geometry, probabilities, threshold=0.9)[0] == pytest.approx(16 / 36)
    assert found_thresholds(geometry, probabilities)[0] == 0.1


def test_exactly_half_survives_floating_point_rounding() -> None:
    overlaps = pd.DataFrame({"tile_id": ["a", "b", "c"], "oocyte": [0, 0, 0],
                             "area": [0.3, 0.2, 0.1]})
    geometry = CutGeometry(
        cut_name="c", tile_size=TILE, um2_per_px=1.0,
        cells=pd.DataFrame(columns=["tile_id", "row", "col", "area", "annotation_area"]),
        oocytes=pd.DataFrame({"oocyte": [0], "area": [0.6], "scorable_area": [0.1 + 0.2 + 0.3]}),
        overlaps=overlaps)

    assert 0.3 < 0.5 * (0.1 + 0.2 + 0.3)
    assert found_thresholds(geometry, pd.Series({"a": 0.9, "b": 0.5, "c": 0.2}))[0] == 0.9


def test_a_threshold_equal_to_a_probability_counts_that_cell_and_oocyte(geometry) -> None:
    probabilities = _scenario(geometry)
    oocytes = per_oocyte_table(geometry, probabilities, fold=1)

    curve = fold_curve([geometry], probabilities, oocytes, np.array([0.8]))

    assert curve.loc[0, "recall"] == 1.0
    assert curve.loc[0, "positive_um2"] == pytest.approx((81 + 64) * UM2_PER_PX)


def test_cells_meeting_only_at_a_corner_are_separate_blobs(geometry) -> None:
    probabilities = pd.Series({tile: 0.1 for tile in geometry.cells["tile_id"]})
    probabilities[["t_r0_c3", "t_r1_c2"]] = 0.7

    counts, multi_cell = false_blob_counts(geometry, probabilities, np.array([0.5]))

    assert list(counts) == [2]
    assert list(multi_cell) == [0]



def test_size_bins_merge_tied_quartiles_and_refuse_no_diameters() -> None:
    assert size_bin_edges(pd.Series([10.0, 10.0, 10.0, 10.0, 50.0])) == [-math.inf, 10.0, math.inf]
    with pytest.raises(ValueError, match="no oocyte diameters"):
        size_bin_edges(pd.Series([np.nan]))


def test_recall_by_keeps_oocytes_without_a_stage_and_empty_bins() -> None:
    oocytes = pd.DataFrame({"fold": [1, 1, 1], "stage": pd.array([0, None, None], dtype="Int64"),
                            "found_threshold": [0.5, 0.6, 0.1]})
    oocytes["size_bin"] = pd.Categorical(["small", "small", "small"], categories=["small", "large"])

    by_stage = recall_by(oocytes, {1: 0.4}, "stage")
    by_size = recall_by(oocytes, {1: 0.4}, "size_bin")

    assert by_stage["n_scorable"].sum() == 3
    assert by_stage.loc[by_stage.index.isna(), "n_found"].item() == 1
    assert by_size.loc["large", "n_scorable"] == 0


def test_fold_curve_refuses_oocytes_from_cuts_it_was_not_given(geometry) -> None:
    probabilities = _scenario(geometry)
    oocytes = per_oocyte_table(geometry, probabilities, fold=1)
    other = oocytes.assign(cut_name="S_B_2_1-2_cut000")

    with pytest.raises(ValueError, match="not among the geometries"):
        fold_curve([geometry], probabilities, pd.concat([oocytes, other]), np.array([0.5]))


def test_fold_curve_refuses_a_scored_cell_without_a_probability(geometry) -> None:
    probabilities = _scenario(geometry)
    oocytes = per_oocyte_table(geometry, probabilities, fold=1)

    with pytest.raises(ValueError, match="scored cells have no probability"):
        fold_curve([geometry], probabilities.drop("t_r2_c2"), oocytes, np.array([0.5]))



@pytest.mark.parametrize("x0, share, scorable", [(22, 9 / 24, False), (20, 0.5, True)])
def test_an_oocyte_needs_half_its_area_in_scored_cells_to_be_judged(
    tmp_path: Path, x0: int, share: float, scorable: bool
) -> None:
    manifest = _manifest()
    tiles = {tile["tile_id"] for tile in manifest["tiles"]}
    geometry = cut_geometry(manifest, _geojson(tmp_path, {"Oosorption Stage 0": _square(x0, 17, 30, 20)}),
                            TILE, tiles)
    probabilities = pd.Series({tile: 0.9 for tile in tiles})

    table = per_oocyte_table(geometry, probabilities, fold=1)

    assert table.loc[0, "scorable_share"] == pytest.approx(share)
    assert math.isnan(table.loc[0, "found_threshold"]) != scorable



def test_fold_curve_refuses_a_cut_with_oocytes_missing_from_the_table(tmp_path: Path, geometry) -> None:
    probabilities = _scenario(geometry)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    manifest = _manifest()
    manifest["cut_name"] = "S_A_1_1-2_cut001"
    other = cut_geometry(manifest, _geojson(other_dir, OOCYTES).rename(
        other_dir / "S_A_1_1-2_cut001_annotations.geojson"), TILE,
        {tile["tile_id"] for tile in manifest["tiles"]})

    with pytest.raises(ValueError, match="no rows in the per-oocyte table"):
        fold_curve([geometry, other], probabilities, per_oocyte_table(geometry, probabilities, 1),
                   np.array([0.5]))



def test_a_single_cell_false_blob_is_left_out_of_the_two_plus_count(geometry) -> None:
    probabilities = pd.Series({tile: 0.1 for tile in geometry.cells["tile_id"]})
    probabilities[["t_r0_c3", "t_r1_c3", "t_r2_c1"]] = 0.7

    counts, multi_cell = false_blob_counts(geometry, probabilities, np.array([0.5]))

    assert list(counts) == [2]
    assert list(multi_cell) == [1]



def test_an_empty_edge_cell_never_forms_or_grows_a_false_blob(tmp_path: Path) -> None:
    manifest = _manifest(skip=None)
    manifest["level0_bbox"]["x1"] = 100 + 25
    for tile in manifest["tiles"]:
        bbox = tile["cut_local_bbox"]
        bbox["x1"] = min(bbox["x1"], 25)
        tile["pad_right"] = TILE - (bbox["x1"] - bbox["x0"])
    scored = {tile["tile_id"] for tile in manifest["tiles"]}
    geometry = cut_geometry(manifest, _geojson(tmp_path, {"Oosorption Stage 0": _square(1, 1, 4, 4)}),
                            TILE, scored)
    probabilities = pd.Series({tile: 0.1 for tile in scored})
    probabilities[["t_r1_c2", "t_r1_c3", "t_r2_c3"]] = 0.9

    counts, multi_cell = false_blob_counts(geometry, probabilities, np.array([0.5]))

    assert geometry.cells.set_index("tile_id").loc["t_r1_c3", "area"] == 0
    assert list(counts) == [1]
    assert list(multi_cell) == [0]
