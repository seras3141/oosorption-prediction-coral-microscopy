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
    NO_PREDICTION,
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
    labels = pd.Series(["positive", "positive", "negative", "negative", "ambiguous", "positive",
                        "positive", "ambiguous"])
    probs = pd.Series([0.9, 0.2, 0.7, 0.1, 0.9, np.nan, 0.9, np.nan])
    thresholds = pd.Series([0.5] * 6 + [np.nan, 0.5])
    outcomes = classify_outcomes(labels, probs, thresholds)
    assert list(outcomes) == [TRUE_POSITIVE, FALSE_NEGATIVE, FALSE_POSITIVE, TRUE_NEGATIVE,
                              NOT_SCORED, NO_PREDICTION, NO_PREDICTION, NOT_SCORED]


def _legend_styles(handles: list) -> list[tuple]:
    from matplotlib.colors import to_rgba

    styles = []
    for handle in handles:
        if hasattr(handle, "get_hatch"):
            styles.append(("patch", handle.get_hatch() or "", str(handle.get_linestyle()),
                           round(to_rgba(handle.get_facecolor())[3], 1)))
        else:
            styles.append(("line", str(handle.get_linestyle())))
    return styles


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


PHIKON_PREFIX = "frozen_encoder_owkin-phikon-v2_0512_area0500_seed42_cv_"


def _folds(root: Path) -> Path:
    path = root / "fold_manifest.json"
    if not path.exists():
        path.write_text('{"folds": []}')
    return path


def _write_cv(root: Path, name: str, overrides: dict | None = None, recorded: dict | None = None,
              partial: bool = False, model: str = "phikon", fold_manifest: Path | None = None,
              fold_sha256: str | None = None) -> None:
    import hashlib
    from dataclasses import asdict

    from src.modeling.train_tile_classifier import TrainingConfig

    arm = ({"architecture": "frozen_encoder", "encoder_name": "owkin/phikon-v2", "num_workers": 3}
           if model == "phikon" else {"architecture": "resnet18", "num_workers": 7})
    config = asdict(TrainingConfig(tile_size=512, **{**arm, **(overrides or {})}))
    if model == "phikon":
        config["embedding_cache_dir"] = "data/embedding_cache/owkin-phikon-v2/0512"
    config.update(recorded or {})
    folds = fold_manifest or _folds(root)
    fold_sha256 = fold_sha256 or hashlib.sha256(folds.read_bytes()).hexdigest()
    path = root / name / "cv_results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "config": config, "is_partial": partial, "fold_manifest_path": str(folds),
        "fold_manifest_sha256": fold_sha256,
        "is_subset_run": any(config.get(key) is not None for key in
                             ("max_train_slides", "max_batches_per_epoch", "max_val_tiles")),
    }))


def _lookup(root: Path, model: str = "phikon") -> str:
    return find_default_cv_results(model, 512, cv_dir=root, fold_manifest=_folds(root)).parent.name


def test_default_run_lookup_skips_sweep_variants(tmp_path: Path) -> None:
    other_folds = tmp_path / "other_folds.json"
    other_folds.write_text('{"folds": [], "version": "other"}')
    _write_cv(tmp_path, PHIKON_PREFIX + "aaaa")
    _write_cv(tmp_path, PHIKON_PREFIX + "bbbb", {"hard_negative_mining": True})
    _write_cv(tmp_path, PHIKON_PREFIX + "cccc", recorded={"embedding_cache_dir": None})
    _write_cv(tmp_path, PHIKON_PREFIX + "dddd", partial=True)
    _write_cv(tmp_path, PHIKON_PREFIX + "ffff", {"head_hidden_dim": 256})
    _write_cv(tmp_path, PHIKON_PREFIX + "gggg", recorded={"quadrant_pooling": "max"})
    _write_cv(tmp_path, PHIKON_PREFIX + "hhhh", fold_manifest=other_folds)
    _write_cv(tmp_path, PHIKON_PREFIX + "jjjj", recorded={
        "embedding_cache_dir": "data/embedding_cache_variant/owkin-phikon-v2/0512"})
    _write_cv(tmp_path, PHIKON_PREFIX + "llll", recorded={"a_retired_field": 1})
    _write_cv(tmp_path, PHIKON_PREFIX + "mmmm", fold_sha256="0" * 64)
    assert _lookup(tmp_path) == PHIKON_PREFIX + "aaaa"

    (tmp_path / "fold_manifest.json").write_text('{"folds": [], "version": "regenerated"}')
    with pytest.raises(ValueError, match="found 0.*skipped .* as run on other folds"):
        _lookup(tmp_path)
    (tmp_path / "fold_manifest.json").write_text('{"folds": []}')

    _write_cv(tmp_path, PHIKON_PREFIX + "eeee", {"num_workers": 7})
    with pytest.raises(ValueError, match="found 2"):
        _lookup(tmp_path)


def test_stale_content_hashes_do_not_exclude_the_default_run(tmp_path: Path) -> None:
    _write_cv(tmp_path, PHIKON_PREFIX + "aaaa",
              recorded={"embedding_cache_sha256": "0" * 64, "split_manifest_sha256": "1" * 64})
    assert _lookup(tmp_path) == PHIKON_PREFIX + "aaaa"


def test_the_resnet_default_run_has_no_cache(tmp_path: Path) -> None:
    prefix = "resnet18_none_0512_area0500_seed42_cv_"
    _write_cv(tmp_path, prefix + "aaaa", model="resnet18")
    _write_cv(tmp_path, prefix + "bbbb", {"lr": 1e-3}, model="resnet18")
    _write_cv(tmp_path, prefix + "cccc", model="resnet18",
              recorded={"embedding_cache_dir": "data/embedding_cache/owkin-phikon-v2/0512"})
    assert _lookup(tmp_path, "resnet18") == prefix + "aaaa"


def test_default_run_follows_the_training_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.modeling.train_tile_classifier as train_module

    _write_cv(tmp_path, PHIKON_PREFIX + "aaaa", {"lr": 1e-3})
    _write_cv(tmp_path, PHIKON_PREFIX + "bbbb", {"lr": 5e-4})
    monkeypatch.setitem(train_module.DEFAULT_LR, "frozen_encoder", 5e-4)
    assert _lookup(tmp_path) == PHIKON_PREFIX + "bbbb"


def test_no_figure_tells_identities_apart_by_colour_alone(
    cut_dirs, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.visualization.cut_maps as cut_maps

    shown = []
    real_legend = cut_maps._legend
    monkeypatch.setattr(cut_maps, "_legend",
                        lambda fig, handles: shown.append(handles) or real_legend(fig, handles))
    cuts, tiles = cut_dirs
    view = load_cut_view(STEM, CUT, tile_sizes=(256,), cuts_dir=cuts, tiles_dir=tiles)
    predictions = pd.DataFrame(
        {"prob": [0.9, 0.9], "fold": [1, 1], "threshold": [0.5] * 2, "fold_auprc": [0.9] * 2},
        index=pd.Index(["t_in", "t_out"], name="tile_id"),
    )
    view.tiles = pd.concat([view.tiles, view.tiles.iloc[[0]].assign(tile_id="t_missing")])
    plot_tile_labels(view, 256)
    plot_prediction_errors(view, 256, predictions, "M")
    assert len(shown) == 2
    for handles in shown:
        styles = _legend_styles(handles)
        assert len(set(styles)) == len(styles), styles


def test_a_missing_prediction_is_counted_apart_from_ambiguous_tiles(cut_dirs) -> None:
    cuts, tiles = cut_dirs
    view = load_cut_view(STEM, CUT, tile_sizes=(256,), cuts_dir=cuts, tiles_dir=tiles)
    predictions = pd.DataFrame(
        {"prob": [0.9], "fold": [1], "threshold": [0.5], "fold_auprc": [0.9]},
        index=pd.Index(["t_in"], name="tile_id"),
    )
    fig = plot_prediction_errors(view, 256, predictions, "M")
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert f"{NOT_SCORED} 1" in labels
    assert f"{NO_PREDICTION} 1" in labels


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


def test_label_and_overlay_maps_do_not_need_torch() -> None:
    import subprocess

    check = ("import sys; sys.path.insert(0, sys.argv[1]); import src.visualization.cut_maps; "
             "sys.exit('torch' in sys.modules)")
    run = subprocess.run([sys.executable, "-c", check, str(REPO_ROOT)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr or "importing cut_maps loaded torch"


@pytest.mark.parametrize(("threshold", "rule"), [
    (0.1, "positive: oocyte coverage ≥ 0.1 · ambiguous: 0 < coverage < 0.1"),
    (0.0, "positive: oocyte coverage > 0"),
])
def test_the_label_caption_states_the_rule_drawn(cut_dirs, threshold: float, rule: str) -> None:
    cuts, tiles = cut_dirs
    view = load_cut_view(STEM, CUT, tile_sizes=(256,), min_oocyte_area_fraction=threshold,
                         cuts_dir=cuts, tiles_dir=tiles)
    texts = [text.get_text() for text in plot_tile_labels(view, 256).texts]
    caption = next(text for text in texts if "coverage" in text)
    assert rule in caption
    assert ("ambiguous" in caption) == (threshold > 0)


@pytest.mark.parametrize(("threshold", "unpredicted"), [(None, 2), (0.5, 0)])
def test_a_fold_with_undefined_metrics_still_renders(
    cut_dirs, tmp_path: Path, threshold: float | None, unpredicted: int
) -> None:
    from src.visualization.cut_maps import held_out_predictions

    cuts, tiles = cut_dirs
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"tile_id": ["t_in", "t_out"], "prob": [0.9, 0.1],
                  "split": ["test", "test"]}).to_csv(predictions, index=False)
    cv = tmp_path / "cv_results.json"
    cv.write_text(json.dumps({"per_fold": [{"fold": 1, "predictions_path": str(predictions),
                                            "decision_threshold": threshold, "auprc": None}]}))
    view = load_cut_view(STEM, CUT, tile_sizes=(256,), cuts_dir=cuts, tiles_dir=tiles)
    fig = plot_prediction_errors(view, 256, held_out_predictions(cv), "M")
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert (f"{NO_PREDICTION} {unpredicted}" in labels) == bool(unpredicted)
    caption = next(text.get_text() for text in fig.texts if "fold AUPRC" in text.get_text())
    assert "fold AUPRC undefined" in caption
    if threshold is None:
        assert "threshold undefined" in caption
        assert "fold 1: no threshold recorded" in caption
        assert "selected on the fold's inner validation" not in caption
    else:
        assert "threshold selected on the fold's inner validation" in caption


def test_a_fold_without_a_threshold_is_named_beside_the_selected_ones(cut_dirs) -> None:
    cuts, tiles = cut_dirs
    view = load_cut_view(STEM, CUT, tile_sizes=(256,), cuts_dir=cuts, tiles_dir=tiles)
    predictions = pd.DataFrame(
        {"prob": [0.9, 0.9, 0.1], "fold": [1, 2, 2], "threshold": [0.5, np.nan, np.nan],
         "fold_auprc": [0.9, np.nan, np.nan]},
        index=pd.Index(["t_in", "t_edge", "t_out"], name="tile_id"),
    )
    fig = plot_prediction_errors(view, 256, predictions, "M")
    caption = next(text.get_text() for text in fig.texts if "fold AUPRC" in text.get_text())
    assert "threshold selected on the fold's inner validation" in caption
    assert "fold 2: no threshold recorded, unambiguous tiles drawn as no prediction" in caption


def test_the_cli_refusal_names_the_figures_a_size_without_runs_can_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.plot_cut_maps as cli
    from src.visualization.cut_maps import NoDefaultRunError

    def no_run(model: str, tile_size: int, seed: int = 42, **kwargs) -> Path:
        raise NoDefaultRunError(f"no default {model} CV run")

    monkeypatch.setattr(cli, "find_default_cv_results", no_run)
    monkeypatch.setattr(cli, "list_cuts", lambda *args, **kwargs: [])
    with pytest.raises(SystemExit) as raised:
        cli.main(["--tile-sizes", "128", "256"])
    message = str(raised.value)
    assert "no default seed-42 CV run" in message
    for model in ("resnet18", "phikon"):
        for size in (128, 256):
            assert f"{model} at {size} px" in message
    assert "--tile-sizes" in message and "--models" in message
    assert "--figures overlay labels" in message


def test_the_cli_passes_other_lookup_errors_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.plot_cut_maps as cli

    def stale_manifest(model: str, tile_size: int, seed: int = 42, **kwargs) -> Path:
        raise ValueError("fold 2: split file does not match the hash")

    monkeypatch.setattr(cli, "find_default_cv_results", stale_manifest)
    monkeypatch.setattr(cli, "list_cuts", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="fold 2"):
        cli.main(["--tile-sizes", "512"])

    _write_cv(tmp_path, PHIKON_PREFIX + "aaaa")
    _write_cv(tmp_path, PHIKON_PREFIX + "bbbb", {"num_workers": 7})
    with pytest.raises(ValueError, match="found 2") as raised:
        _lookup(tmp_path)
    assert type(raised.value) is ValueError


def test_runs_only_on_other_folds_are_not_reported_as_missing(tmp_path: Path) -> None:
    _write_cv(tmp_path, PHIKON_PREFIX + "aaaa", fold_sha256="0" * 64)
    with pytest.raises(ValueError, match="skipped .* as run on other folds") as raised:
        _lookup(tmp_path)
    assert type(raised.value) is ValueError


def test_a_partial_default_run_is_named_rather_than_reported_missing(tmp_path: Path) -> None:
    _write_cv(tmp_path, PHIKON_PREFIX + "aaaa", partial=True)
    with pytest.raises(ValueError, match=r"skipped \['" + PHIKON_PREFIX + r"aaaa'\] as partial") as raised:
        _lookup(tmp_path)
    assert type(raised.value) is ValueError


def test_a_stale_sweep_run_does_not_hide_a_missing_default(tmp_path: Path) -> None:
    from src.visualization.cut_maps import NoDefaultRunError

    _write_cv(tmp_path, PHIKON_PREFIX + "aaaa", {"lr": 5e-4}, fold_sha256="0" * 64)
    _write_cv(tmp_path, PHIKON_PREFIX + "bbbb", {"lr": 5e-4}, partial=True)
    with pytest.raises(NoDefaultRunError):
        _lookup(tmp_path)
