"""Cut-level maps: outcome classification, cut loading, figures and run lookup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.cut_maps import (
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    NOT_SCORED,
    TRUE_NEGATIVE,
    TRUE_POSITIVE,
    classify_outcomes,
    find_default_cv_results,
    list_cuts,
    load_cut_view,
    plot_oocyte_overlay,
    plot_prediction_errors,
    plot_tile_labels,
    save,
)

STEM, CUT = "LHP_A_1_1-2", "LHP_A_1_1-2_cut000"


def test_outcomes_follow_label_probability_and_threshold() -> None:
    labels = pd.Series(["positive", "positive", "negative", "negative", "ambiguous", "positive"])
    probs = pd.Series([0.9, 0.2, 0.7, 0.1, 0.9, np.nan])
    outcomes = classify_outcomes(labels, probs, pd.Series([0.5] * 6))
    assert list(outcomes) == [TRUE_POSITIVE, FALSE_NEGATIVE, FALSE_POSITIVE, TRUE_NEGATIVE,
                              NOT_SCORED, NOT_SCORED]


@pytest.fixture
def cut_dirs(tmp_path: Path) -> tuple[Path, Path]:
    cuts, tiles = tmp_path / "cuts", tmp_path / "tiles"
    (cuts / STEM).mkdir(parents=True)
    Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8)).save(cuts / STEM / f"{CUT}_preview.png")
    (cuts / STEM / f"{STEM}_cuts.json").write_text(json.dumps({"cuts": [{
        "name": CUT, "level0_size": [1024, 1024], "preview_path": f"{CUT}_preview.png"}]}))
    square = [[0, 0], [400, 0], [400, 400], [0, 400], [0, 0]]
    (cuts / STEM / f"{CUT}_annotations.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {},
                      "geometry": {"type": "LineString", "coordinates": square}}]}))
    manifest = tiles / STEM / CUT / f"{CUT}_tile_manifest.json"
    manifest.parent.mkdir(parents=True)
    boxes = {"t_in": (0, 0), "t_edge": (390, 0), "t_out": (600, 600)}
    manifest.write_text(json.dumps({"cut_name": CUT, "stem": STEM, "tiles": [
        {"tile_id": tid, "tile_size": 256,
         "cut_local_bbox": {"x0": x, "y0": y, "x1": x + 256, "y1": y + 256}}
        for tid, (x, y) in boxes.items()]}))
    return cuts, tiles


def test_a_cut_loads_in_preview_coordinates_with_labels(cut_dirs) -> None:
    cuts, tiles = cut_dirs
    view = load_cut_view(STEM, CUT, tile_sizes=(256,), cuts_dir=cuts, tiles_dir=tiles)
    assert view.scale == pytest.approx(1 / 16)
    assert view.scale_label == "1/16"
    assert dict(zip(view.tiles.tile_id, view.tiles.label)) == {
        "t_in": "positive", "t_edge": "ambiguous", "t_out": "negative"}
    assert list_cuts(cuts) == [(STEM, CUT)]


def test_every_figure_renders(cut_dirs, tmp_path: Path) -> None:
    cuts, tiles = cut_dirs
    view = load_cut_view(STEM, CUT, tile_sizes=(256,), cuts_dir=cuts, tiles_dir=tiles)
    predictions = pd.DataFrame(
        {"prob": [0.9, 0.9, 0.8], "fold": [1, 1, 1], "threshold": [0.5] * 3, "fold_auprc": [0.9] * 3},
        index=pd.Index(["t_in", "t_edge", "t_out"], name="tile_id"),
    )
    for name, fig in (
        ("overlay", plot_oocyte_overlay(view)),
        ("labels", plot_tile_labels(view, 256)),
        ("errors", plot_prediction_errors(view, 256, predictions, "Model")),
    ):
        path = save(fig, tmp_path / f"{name}.jpg")
        assert Image.open(path).size[0] > 64


def _write_cv(root: Path, name: str, config: dict, partial: bool = False) -> None:
    base = {"epochs": 30, "patience": 5, "batch_size": 64, "weight_decay": 1e-4,
            "positive_fraction": 0.25, "hard_negative_mining": False, "head_hidden_dim": 512,
            "head_dropout": 0.25, "min_oocyte_area_fraction": 0.05, "label_rule": "area",
            "lr": 1e-3, "quadrant_pooling": "mean", "embedding_cache_dir": "cache"}
    path = root / name / "cv_results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"config": {**base, **config}, "is_partial": partial,
                                "is_subset_run": False}))


def test_default_run_lookup_skips_sweep_variants(tmp_path: Path) -> None:
    prefix = "frozen_encoder_owkin-phikon-v2_0512_area0500_seed42_cv_"
    _write_cv(tmp_path, prefix + "aaaa", {})
    _write_cv(tmp_path, prefix + "bbbb", {"hard_negative_mining": True})
    _write_cv(tmp_path, prefix + "cccc", {"embedding_cache_dir": None})
    _write_cv(tmp_path, prefix + "dddd", {}, partial=True)
    assert find_default_cv_results("phikon", 512, cv_dir=tmp_path).parent.name == prefix + "aaaa"

    _write_cv(tmp_path, prefix + "eeee", {})
    with pytest.raises(ValueError, match="found 2"):
        find_default_cv_results("phikon", 512, cv_dir=tmp_path)


def test_overlapping_annotations_are_not_double_counted(cut_dirs) -> None:
    from shapely.ops import unary_union

    cuts, tiles = cut_dirs
    geojson = cuts / STEM / f"{CUT}_annotations.geojson"
    features = json.loads(geojson.read_text())["features"]
    geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features * 2}))
    view = load_cut_view(STEM, CUT, tile_sizes=(256,), cuts_dir=cuts, tiles_dir=tiles)
    assert len(view.polygons) == 2
    assert unary_union(view.polygons).area == pytest.approx(400 * 400)


def test_oocyte_paths_keep_their_holes() -> None:
    from shapely.geometry import Polygon

    from src.visualization.cut_maps import _polygon_path

    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch

    ring = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]])
    fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(PathPatch(_polygon_path(ring, scale=1.0), facecolor="k", edgecolor="none"))
    fig.canvas.draw()
    red = np.asarray(fig.canvas.buffer_rgba())[..., 0]
    plt.close(fig)
    assert red[10, 10] == 0
    assert red[50, 50] == 255
