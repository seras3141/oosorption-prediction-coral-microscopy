"""Synthetic tests for review-session tile sampling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tifffile
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.sample_review_tiles import (
    _annotation_proximity_check,
    _centre_bbox_on_point,
    _collect_positive_candidates,
    _discover_cut_sources,
    _extract_tile_array,
    _resolve_session_id,
    sample_review_session,
)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp)


def _synthetic_feature(annotation_id: str, coords: list[list[float]]) -> dict:
    return {
        "type": "Feature",
        "id": annotation_id,
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
        "properties": {
            "classification": {
                "name": "Stage 0",
            }
        },
    }


def _write_synthetic_cut(
    root: Path,
    *,
    stem: str = "TEST_slide",
    cut_name: str = "TEST_slide_cut000",
) -> tuple[Path, Path, Path]:
    stem_dir = root / stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    cut_tiff_path = stem_dir / f"{cut_name}.tif"
    annotations_path = stem_dir / f"{cut_name}_annotations.geojson"

    arr = np.full((512, 512, 3), fill_value=[200, 150, 180], dtype=np.uint8)
    tifffile.imwrite(cut_tiff_path, arr)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            _synthetic_feature(
                "inside-first-tile",
                [[100.0, 100.0], [120.0, 100.0], [120.0, 120.0], [100.0, 120.0]],
            )
        ],
    }
    _write_json(annotations_path, geojson)
    return stem_dir, cut_tiff_path, annotations_path


def test_centre_bbox_on_point_shifts_in_bounds() -> None:
    bbox = _centre_bbox_on_point(20.0, 20.0, 128, 256, 256)

    assert bbox == {"x0": 0, "y0": 0, "x1": 128, "y1": 128}


def test_centre_bbox_on_point_handles_small_cut() -> None:
    bbox = _centre_bbox_on_point(50.0, 50.0, 256, 100, 120)

    assert bbox == {"x0": 0, "y0": 0, "x1": 100, "y1": 120}


def test_annotation_proximity_check_rejects_buffer_overlap() -> None:
    bbox = {"x0": 0, "y0": 0, "x1": 256, "y1": 256}
    annotations = [{"cx": 270.0, "cy": 128.0}]

    assert not _annotation_proximity_check(bbox, annotations, 20.0)


def test_extract_tile_array_zero_pads_partial_tile(tmp_path: Path) -> None:
    _stem_dir, cut_tiff_path, _annotations_path = _write_synthetic_cut(tmp_path)

    tile_arr = _extract_tile_array(cut_tiff_path, 400, 400, 256)

    assert tile_arr.shape == (256, 256, 3)
    assert np.all(tile_arr[:112, :112] == [200, 150, 180])
    assert np.all(tile_arr[112:, 112:] == 0)


def test_collect_positive_candidates_matches_annotation_count(tmp_path: Path) -> None:
    cuts_dir = tmp_path / "cuts"
    _write_synthetic_cut(cuts_dir)

    cut_sources = _discover_cut_sources(cuts_dir)
    candidates = _collect_positive_candidates(cut_sources)

    assert len(candidates) == 1
    assert candidates[0]["annotation_id"] == "inside-first-tile"


def test_sample_review_session_writes_manifest_and_pngs(tmp_path: Path) -> None:
    cuts_dir = tmp_path / "cuts"
    output_dir = tmp_path / "review_sessions"
    _write_synthetic_cut(cuts_dir)

    manifest = sample_review_session(
        cuts_dir,
        output_dir,
        session_id="2026-05-20_review001",
        tile_sizes=(256,),
        n_positive_per_size=1,
        n_negative_per_size=1,
        min_tissue_fraction=0.0,
        negative_buffer_fraction=0.0,
        seed=7,
    )

    session_dir = output_dir / "2026-05-20_review001"
    meta_path = session_dir / "session_meta.json"
    labels_path = session_dir / "labels.json"
    assert meta_path.exists()
    assert labels_path.exists()
    assert not (session_dir / "session.json").exists()

    with meta_path.open("r", encoding="utf-8") as fp:
        session_meta = json.load(fp)
    with labels_path.open("r", encoding="utf-8") as fp:
        labels = json.load(fp)

    assert manifest["n_tiles_per_size"] == 2
    assert len(manifest["tiles"]) == 2
    assert len(manifest["labels"]) == 2
    assert session_meta["session_id"] == "2026-05-20_review001"
    assert labels["session_id"] == "2026-05-20_review001"
    assert len(session_meta["tiles"]) == 2
    assert len(labels["labels"]) == 2
    assert all("ground_truth" in tile for tile in session_meta["tiles"])
    assert all("cut_name" in tile for tile in session_meta["tiles"])
    assert all("cut_local_bbox" in tile for tile in session_meta["tiles"])
    assert all("annotation_ids" in tile for tile in session_meta["tiles"])
    assert all("collaborator_label" not in tile for tile in session_meta["tiles"])
    assert all("labelled_at" not in tile for tile in session_meta["tiles"])
    assert all(
        set(label_row) == {"display_index", "tile_id", "collaborator_label", "labelled_at"}
        for label_row in labels["labels"]
    )
    assert all(label_row["collaborator_label"] is None for label_row in labels["labels"])
    assert all(label_row["labelled_at"] is None for label_row in labels["labels"])
    assert sorted(tile["png_filename"] for tile in manifest["tiles"]) == [
        "tiles/tile_001.png",
        "tiles/tile_002.png",
    ]
    assert all((session_dir / tile["png_filename"]).exists() for tile in manifest["tiles"])

    by_truth = {tile["ground_truth"]: tile for tile in manifest["tiles"]}
    assert by_truth[True]["n_oocytes_ground_truth"] >= 1
    assert by_truth[False]["annotation_ids"] == []


def test_resolve_session_id_increments_existing_directories(tmp_path: Path) -> None:
    output_dir = tmp_path / "review_sessions"
    (output_dir / "2026-05-20_review001").mkdir(parents=True)

    session_id = _resolve_session_id(output_dir, session_id=None)

    assert session_id.endswith("review002")


def test_sample_review_session_raises_when_negative_tiles_unavailable(tmp_path: Path) -> None:
    cuts_dir = tmp_path / "cuts"
    output_dir = tmp_path / "review_sessions"
    stem_dir, cut_tiff_path, annotations_path = _write_synthetic_cut(cuts_dir)

    arr = np.full((256, 256, 3), fill_value=[200, 150, 180], dtype=np.uint8)
    tifffile.imwrite(cut_tiff_path, arr)
    _write_json(
        annotations_path,
        {
            "type": "FeatureCollection",
            "features": [
                _synthetic_feature(
                    "covers-only-tile",
                    [[120.0, 120.0], [130.0, 120.0], [130.0, 130.0], [120.0, 130.0]],
                )
            ],
        },
    )

    try:
        sample_review_session(
            cuts_dir,
            output_dir,
            session_id="2026-05-20_review001",
            tile_sizes=(256,),
            n_positive_per_size=1,
            n_negative_per_size=1,
            min_tissue_fraction=0.0,
            negative_buffer_fraction=0.0,
            seed=7,
        )
    except ValueError as exc:
        assert "negative tiles" in str(exc)
    else:
        raise AssertionError("Expected negative tile sampling to fail")
