"""Synthetic tests for tile generation helpers and single-cut I/O."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.extract_ndpi_cuts import create_tissue_mask
from src.data_preparation.generate_tiles import (
    _assign_annotations_to_tile,
    _build_tile_id,
    _compute_tile_grid,
    _extract_stage,
    _to_tile_local_coords,
    generate_tiles_for_cut,
)


def _annotation(annotation_id: str, coords: list[list[float]]) -> dict:
    return {
        "annotation_id": annotation_id,
        "stage": "Stage 0",
        "coords": coords,
    }


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp)


def _synthetic_manifest() -> dict:
    return {
        "source_ndpi": "TEST_slide.ndpi",
        "level0_dimensions": [512, 512],
        "mpp_x": 0.23,
        "mpp_y": 0.23,
        "cuts": [
            {
                "index": 0,
                "name": "TEST_slide_cut000",
                "level0_bbox": {"x0": 1000, "y0": 2000, "x1": 1512, "y1": 2512},
                "level0_size": [512, 512],
            }
        ],
    }


def test_compute_tile_grid_exact_fit() -> None:
    grid = _compute_tile_grid(cut_h=512, cut_w=512, tile_size=256, stride=256)

    assert len(grid) == 4
    assert all(tile["pad_right"] == 0 for tile in grid)
    assert all(tile["pad_bottom"] == 0 for tile in grid)


def test_compute_tile_grid_with_padding() -> None:
    grid = _compute_tile_grid(cut_h=300, cut_w=300, tile_size=256, stride=205)

    assert len(grid) == 4
    by_position = {(tile["row"], tile["col"]): tile for tile in grid}
    assert by_position[(0, 0)]["pad_right"] == 0
    assert by_position[(0, 0)]["pad_bottom"] == 0
    assert by_position[(0, 1)]["pad_right"] == 161
    assert by_position[(1, 0)]["pad_bottom"] == 161
    assert by_position[(1, 1)]["pad_right"] == 161
    assert by_position[(1, 1)]["pad_bottom"] == 161


def test_compute_tile_grid_smaller_than_tile() -> None:
    grid = _compute_tile_grid(cut_h=100, cut_w=100, tile_size=256, stride=205)

    assert len(grid) == 1
    assert grid[0]["pad_right"] == 156
    assert grid[0]["pad_bottom"] == 156


def test_assign_annotations_to_tile_centroid_inside() -> None:
    annotations = [_annotation("inside", [[40.0, 40.0], [60.0, 60.0]])]
    bbox = {"x0": 0, "y0": 0, "x1": 256, "y1": 256}

    assigned = _assign_annotations_to_tile(annotations, bbox)

    assert [item["annotation_id"] for item in assigned] == ["inside"]


def test_assign_annotations_to_tile_centroid_outside() -> None:
    annotations = [_annotation("outside", [[290.0, 290.0], [310.0, 310.0]])]
    bbox = {"x0": 0, "y0": 0, "x1": 256, "y1": 256}

    assigned = _assign_annotations_to_tile(annotations, bbox)

    assert assigned == []


def test_assign_annotations_to_tile_centroid_on_boundary() -> None:
    annotations = [_annotation("boundary", [[256.0, 256.0], [256.0, 256.0]])]
    bbox = {"x0": 0, "y0": 0, "x1": 256, "y1": 256}

    assigned = _assign_annotations_to_tile(annotations, bbox)

    assert [item["annotation_id"] for item in assigned] == ["boundary"]


def test_to_tile_local_coords() -> None:
    coords = [[100.0, 200.0], [150.0, 250.0]]

    assert _to_tile_local_coords(coords, 50, 80) == [[50.0, 120.0], [100.0, 170.0]]


def test_build_tile_id() -> None:
    tile_id = _build_tile_id("CHN_AU_10_19-21_cut000", 256, 3, 12)

    assert tile_id == "CHN_AU_10_19-21_cut000_s0256_r0003_c0012"


def test_extract_stage_present() -> None:
    feature = {"properties": {"classification": {"name": "Stage 3"}}}

    assert _extract_stage(feature) == "Stage 3"


def test_extract_stage_absent() -> None:
    feature = {"properties": {}}

    assert _extract_stage(feature) == "Unknown"


def test_generate_tiles_for_cut_synthetic(tmp_path: Path) -> None:
    cut_tiff_path = tmp_path / "TEST_slide_cut000.tif"
    annotations_geojson_path = tmp_path / "TEST_slide_cut000_annotations.geojson"
    cuts_manifest_path = tmp_path / "TEST_slide_cuts.json"
    output_dir = Path("tmp_test_tiles") / tmp_path.name

    arr = np.full((512, 512, 3), fill_value=[200, 150, 180], dtype=np.uint8)
    tifffile.imwrite(cut_tiff_path, arr)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "inside-first-tile",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]],
                },
                "properties": {"classification": {"name": "Stage 0"}},
            },
            {
                "type": "Feature",
                "id": "outside-cut",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[600.0, 600.0], [620.0, 600.0], [620.0, 620.0]],
                },
                "properties": {"classification": {"name": "Stage 3"}},
            },
        ],
    }
    _write_json(annotations_geojson_path, geojson)
    _write_json(cuts_manifest_path, _synthetic_manifest())

    manifest = generate_tiles_for_cut(
        cut_tiff_path,
        annotations_geojson_path,
        cuts_manifest_path,
        output_dir,
        tile_sizes=(256,),
        min_tissue_fraction=0.0,
    )

    manifest_path = REPO_ROOT / output_dir / "TEST_slide" / "TEST_slide_cut000" / "TEST_slide_cut000_tile_manifest.json"
    assert manifest_path.exists()
    assert manifest["n_tiles_total"] == len(manifest["tiles"])
    assert manifest["n_tiles_with_oocyte"] >= 1
    assert all((REPO_ROOT / tile["png_path"]).exists() for tile in manifest["tiles"])

    first_tile = next(tile for tile in manifest["tiles"] if tile["row"] == 0 and tile["col"] == 0)
    assert first_tile["annotations"][0]["annotation_id"] == "inside-first-tile"
    assert first_tile["annotations"][0]["stage"] == "Stage 0"
    assert first_tile["annotations"][0]["tile_local_coords"][0] == [40.0, 40.0]


def test_skip_if_exists(tmp_path: Path) -> None:
    cut_tiff_path = tmp_path / "TEST_slide_cut000.tif"
    annotations_geojson_path = tmp_path / "TEST_slide_cut000_annotations.geojson"
    cuts_manifest_path = tmp_path / "TEST_slide_cuts.json"
    output_dir = Path("tmp_test_tiles") / tmp_path.name

    arr = np.full((512, 512, 3), fill_value=[200, 150, 180], dtype=np.uint8)
    tifffile.imwrite(cut_tiff_path, arr)
    _write_json(
        annotations_geojson_path,
        {
            "type": "FeatureCollection",
            "features": [],
        },
    )
    _write_json(cuts_manifest_path, _synthetic_manifest())

    first_manifest = generate_tiles_for_cut(
        cut_tiff_path,
        annotations_geojson_path,
        cuts_manifest_path,
        output_dir,
        tile_sizes=(256,),
        min_tissue_fraction=0.0,
    )
    manifest_path = REPO_ROOT / output_dir / "TEST_slide" / "TEST_slide_cut000" / "TEST_slide_cut000_tile_manifest.json"
    before_mtime = manifest_path.stat().st_mtime_ns

    second_manifest = generate_tiles_for_cut(
        cut_tiff_path,
        annotations_geojson_path,
        cuts_manifest_path,
        output_dir,
        tile_sizes=(256,),
        min_tissue_fraction=0.0,
        skip_if_exists=True,
    )

    assert second_manifest == first_manifest
    assert manifest_path.stat().st_mtime_ns == before_mtime


def test_tissue_fraction_filter() -> None:
    white_tile = np.full((256, 256, 3), fill_value=255, dtype=np.uint8)
    pink_tile = np.full((256, 256, 3), fill_value=[200, 150, 180], dtype=np.uint8)

    assert create_tissue_mask(white_tile, method="saturation").mean() < 0.20
    assert create_tissue_mask(pink_tile, method="saturation").mean() > 0.20