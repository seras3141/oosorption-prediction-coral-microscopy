"""Cut-level maps: oocyte overlays, tile labels, and held-out prediction errors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from src.modeling.tile_labels import (
    DEFAULT_MIN_OOCYTE_AREA_FRACTION,
    LABEL_AMBIGUOUS,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    label_for_area_fraction,
    load_annotation_polygons,
    oocyte_area_fractions_for_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CUTS_DIR = REPO_ROOT / "data" / "cuts"
TILES_DIR = REPO_ROOT / "data" / "tiles"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#9a9993"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
OOCYTE_FILL = AQUA

LABEL_COLOURS = {LABEL_NEGATIVE: BLUE, LABEL_AMBIGUOUS: ORANGE, LABEL_POSITIVE: AQUA}

TRUE_POSITIVE = "true positive"
FALSE_NEGATIVE = "false negative"
FALSE_POSITIVE = "false positive"
TRUE_NEGATIVE = "true negative"
NOT_SCORED = "not scored (ambiguous)"
OUTCOME_ORDER = (TRUE_POSITIVE, FALSE_NEGATIVE, FALSE_POSITIVE, TRUE_NEGATIVE, NOT_SCORED)
OUTCOME_COLOURS = {TRUE_POSITIVE: AQUA, FALSE_NEGATIVE: ORANGE, FALSE_POSITIVE: BLUE}


@dataclass
class CutView:
    """Everything a cut-level figure draws, in preview pixel coordinates."""

    stem: str
    cut_name: str
    preview: np.ndarray
    scale: float
    polygons: list[BaseGeometry]
    cut_area_level0: float
    tiles: pd.DataFrame

    @property
    def title(self) -> str:
        return self.cut_name.replace("_", " ")

    @property
    def scale_label(self) -> str:
        return f"1/{round(1 / self.scale)}"


def list_cuts(cuts_dir: Path = CUTS_DIR) -> list[tuple[str, str]]:
    """Every ``(stem, cut_name)`` with a cut manifest, sorted."""
    cuts = []
    for manifest in sorted(cuts_dir.glob("*/*_cuts.json")):
        for cut in json.loads(manifest.read_text())["cuts"]:
            cuts.append((manifest.parent.name, cut["name"]))
    return cuts


def load_cut_view(
    stem: str,
    cut_name: str,
    tile_sizes: tuple[int, ...] = (512, 1024),
    min_oocyte_area_fraction: float = DEFAULT_MIN_OOCYTE_AREA_FRACTION,
    cuts_dir: Path = CUTS_DIR,
    tiles_dir: Path = TILES_DIR,
) -> CutView:
    """Load a cut's preview, annotations and labelled tiles at the given sizes."""
    cut_manifest = json.loads((cuts_dir / stem / f"{stem}_cuts.json").read_text())
    cut = next(c for c in cut_manifest["cuts"] if c["name"] == cut_name)
    preview = np.asarray(Image.open(cuts_dir / stem / cut["preview_path"]).convert("RGB"))
    width_level0, height_level0 = cut["level0_size"]
    scale = preview.shape[1] / width_level0

    geojson = cuts_dir / stem / f"{cut_name}_annotations.geojson"
    tile_manifest = json.loads(
        (tiles_dir / stem / cut_name / f"{cut_name}_tile_manifest.json").read_text()
    )
    fractions = oocyte_area_fractions_for_manifest(tile_manifest, geojson, tile_sizes=tile_sizes)
    rows = []
    for tile in tile_manifest["tiles"]:
        if tile["tile_size"] not in tile_sizes:
            continue
        box = tile["cut_local_bbox"]
        fraction = fractions[tile["tile_id"]]
        rows.append({
            "tile_id": tile["tile_id"], "tile_size": tile["tile_size"],
            "x0": box["x0"], "y0": box["y0"], "x1": box["x1"], "y1": box["y1"],
            "oocyte_area_fraction": fraction,
            "label": label_for_area_fraction(fraction, min_oocyte_area_fraction),
        })
    return CutView(
        stem=stem,
        cut_name=cut_name,
        preview=preview,
        scale=scale,
        polygons=load_annotation_polygons(geojson),
        cut_area_level0=float(width_level0 * height_level0),
        tiles=pd.DataFrame(rows, columns=[
            "tile_id", "tile_size", "x0", "y0", "x1", "y1", "oocyte_area_fraction", "label",
        ]),
    )


def classify_outcomes(
    labels: pd.Series, probabilities: pd.Series, thresholds: pd.Series
) -> pd.Series:
    """Map each tile to TP / FN / FP / TN, or not-scored for ambiguous or missing."""
    predicted = probabilities >= thresholds
    outcome = pd.Series(NOT_SCORED, index=labels.index, dtype=object)
    scored = probabilities.notna() & (labels != LABEL_AMBIGUOUS)
    positive = labels == LABEL_POSITIVE
    outcome[scored & positive & predicted] = TRUE_POSITIVE
    outcome[scored & positive & ~predicted] = FALSE_NEGATIVE
    outcome[scored & ~positive & predicted] = FALSE_POSITIVE
    outcome[scored & ~positive & ~predicted] = TRUE_NEGATIVE
    return outcome


def held_out_predictions(cv_results_path: str | Path) -> pd.DataFrame:
    """Each tile's held-out probability, fold and that fold's decision threshold."""
    path = Path(cv_results_path)
    path = path if path.is_absolute() else REPO_ROOT / path
    cv = json.loads(path.read_text())
    frames = []
    for fold in cv["per_fold"]:
        predictions = pd.read_csv(REPO_ROOT / fold["predictions_path"])
        held_out = predictions.loc[predictions["split"] == "test", ["tile_id", "prob"]]
        frames.append(held_out.assign(fold=fold["fold"], threshold=fold["decision_threshold"],
                                      fold_auprc=fold["auprc"]))
    return pd.concat(frames, ignore_index=True).set_index("tile_id")


MODELS = {
    "resnet18": {"label": "ResNet-18", "prefix": "resnet18_none", "cached": False},
    "phikon": {"label": "Phikon-v2 (frozen, cached)",
               "prefix": "frozen_encoder_owkin-phikon-v2", "cached": True},
}
# Declared defaults; others are sweep variants.
DEFAULT_RUN_CONFIG = {
    "epochs": 30, "patience": 5, "batch_size": 64, "weight_decay": 1e-4,
    "positive_fraction": 0.25, "hard_negative_mining": False, "head_hidden_dim": 512,
    "head_dropout": 0.25, "min_oocyte_area_fraction": DEFAULT_MIN_OOCYTE_AREA_FRACTION,
    "label_rule": "area",
}
DEFAULT_LR = {"resnet18": 1e-4, "phikon": 1e-3}


def find_default_cv_results(
    model: str, tile_size: int, seed: int = 42, cv_dir: Path = REPO_ROOT / "data" / "tile_classifier_cv"
) -> Path:
    """The one full CV run of ``model`` at ``tile_size`` under the declared defaults."""
    spec = MODELS[model]
    matches = []
    for path in sorted(cv_dir.glob(f"{spec['prefix']}_{tile_size:04d}_*_seed{seed}_cv_*/cv_results.json")):
        cv = json.loads(path.read_text())
        config = cv["config"]
        if cv["is_partial"] or cv["is_subset_run"]:
            continue
        if bool(config.get("embedding_cache_dir")) != spec["cached"]:
            continue
        if config.get("quadrant_pooling", "mean") != "mean" or config["lr"] != DEFAULT_LR[model]:
            continue
        if all(config.get(key) == value for key, value in DEFAULT_RUN_CONFIG.items()):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"expected one default {model} CV run at {tile_size} px, seed {seed}; found "
            f"{len(matches)}: {[p.parent.name for p in matches]}"
        )
    return matches[0]


def plot_oocyte_overlay(view: CutView) -> plt.Figure:
    """The cut preview with its annotated oocytes shaded."""
    fig, ax = _canvas(view)
    _draw_oocytes(ax, view, fill=True)
    oocyte_area = unary_union(view.polygons).area if view.polygons else 0.0
    count = len(view.polygons)
    _caption(fig, [
        f"{count} annotated oocyte{'s' if count != 1 else ''}",
        f"{100 * oocyte_area / view.cut_area_level0:.2f}% of the cut area",
        f"preview at {view.scale_label} scale",
    ])
    return fig


def plot_tile_labels(view: CutView, tile_size: int) -> plt.Figure:
    """Tiles at one size, coloured by label, with the oocyte boundaries on top."""
    tiles = view.tiles.loc[view.tiles["tile_size"] == tile_size]
    fig, ax = _canvas(view, subtitle=f"tile labels at {tile_size} px")
    _draw_tiles(ax, view, tiles.loc[tiles["label"] == LABEL_NEGATIVE], BLUE,
                fill_alpha=0.05, edge_alpha=0.5, linewidth=0.5)
    for label in (LABEL_AMBIGUOUS, LABEL_POSITIVE):
        _draw_tiles(ax, view, tiles.loc[tiles["label"] == label], LABEL_COLOURS[label],
                    fill_alpha=0.35)
    _draw_oocytes(ax, view, fill=False)
    counts = tiles["label"].value_counts()
    handles = [
        Patch(facecolor=matplotlib.colors.to_rgba(LABEL_COLOURS[label], alpha),
              edgecolor=LABEL_COLOURS[label], label=f"{label} {counts.get(label, 0):,}")
        for label, alpha in ((LABEL_POSITIVE, 0.5), (LABEL_AMBIGUOUS, 0.5), (LABEL_NEGATIVE, 0.08))
    ]
    handles.append(Line2D([], [], color=INK, linewidth=1.5, label="oocyte boundary"))
    _legend(fig, handles)
    _caption(fig, [
        f"{len(tiles):,} tiles kept by the tissue filter",
        "positive: oocyte coverage ≥ 0.05 · ambiguous: 0 < coverage < 0.05",
        "tiles overlap by 20%",
    ])
    return fig


def plot_prediction_errors(
    view: CutView, tile_size: int, predictions: pd.DataFrame, model_name: str
) -> plt.Figure:
    """Held-out outcomes of one model at one size, with the oocyte boundaries on top."""
    tiles = view.tiles.loc[view.tiles["tile_size"] == tile_size].copy()
    joined = tiles.join(predictions, on="tile_id")
    tiles["outcome"] = classify_outcomes(joined["label"], joined["prob"], joined["threshold"])
    folds = sorted(joined["fold"].dropna().astype(int).unique())
    fold_note = ", ".join(
        f"fold {f}: threshold {joined.loc[joined['fold'] == f, 'threshold'].iloc[0]:.3f}, "
        f"fold AUPRC {joined.loc[joined['fold'] == f, 'fold_auprc'].iloc[0]:.3f}"
        for f in folds
    ) or "no held-out predictions for this cut"

    fig, ax = _canvas(view, subtitle=f"{model_name} · held-out predictions at {tile_size} px")
    _draw_tiles(ax, view, tiles.loc[tiles["outcome"] == TRUE_NEGATIVE], INK_MUTED,
                fill_alpha=0.0, edge_alpha=0.35, linewidth=0.4)
    _draw_tiles(ax, view, tiles.loc[tiles["outcome"] == NOT_SCORED], INK_MUTED,
                fill_alpha=0.0, edge_alpha=0.8, linestyle=(0, (2, 2)))
    for outcome, hatch in ((TRUE_POSITIVE, None), (FALSE_POSITIVE, None), (FALSE_NEGATIVE, "////")):
        _draw_tiles(ax, view, tiles.loc[tiles["outcome"] == outcome], OUTCOME_COLOURS[outcome],
                    hatch=hatch)
    _draw_oocytes(ax, view, fill=False)

    counts = tiles["outcome"].value_counts()
    handles = [
        Patch(facecolor=OUTCOME_COLOURS[o], edgecolor=OUTCOME_COLOURS[o], alpha=0.6,
              hatch="////" if o == FALSE_NEGATIVE else None, label=f"{o} {counts.get(o, 0):,}")
        for o in (TRUE_POSITIVE, FALSE_NEGATIVE, FALSE_POSITIVE)
    ]
    handles += [
        Patch(facecolor="none", edgecolor=INK_MUTED, label=f"{TRUE_NEGATIVE} {counts.get(TRUE_NEGATIVE, 0):,}"),
        Patch(facecolor="none", edgecolor=INK_MUTED, linestyle="--",
              label=f"{NOT_SCORED} {counts.get(NOT_SCORED, 0):,}"),
        Line2D([], [], color=INK, linewidth=1.5, label="oocyte boundary"),
    ]
    _legend(fig, handles)
    _caption(fig, [fold_note, "threshold selected on the fold's inner validation"])
    return fig


def save(fig: plt.Figure, path: Path) -> Path:
    """Write a figure as JPEG and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=fig.dpi, facecolor=SURFACE, pil_kwargs={"quality": 88})
    plt.close(fig)
    return path


def _canvas(view: CutView, subtitle: str | None = None) -> tuple[plt.Figure, Any]:
    height, width = view.preview.shape[:2]
    dpi = 100
    top, bottom = 0.9, 1.3 if subtitle else 0.8
    fig = plt.figure(figsize=(width / dpi + 0.9, height / dpi + top + bottom), dpi=dpi,
                     facecolor=SURFACE)
    ax = fig.add_axes([
        0.45 / fig.get_figwidth(), bottom / fig.get_figheight(),
        width / dpi / fig.get_figwidth(), height / dpi / fig.get_figheight(),
    ])
    # Pixel edges at integers, matching the overlay coordinates.
    ax.imshow(view.preview, interpolation="lanczos", extent=(0, width, height, 0))
    ax.set_axis_off()
    fig.text(0.5, 1 - 0.35 / fig.get_figheight(), view.title, ha="center", va="center",
             fontsize=20, color=INK)
    if subtitle:
        fig.text(0.5, 1 - 0.7 / fig.get_figheight(), subtitle, ha="center", va="center",
                 fontsize=14, color=INK_SECONDARY)
    return fig, ax


def _draw_oocytes(ax: Any, view: CutView, fill: bool) -> None:
    patches = [
        PathPatch(_polygon_path(part, view.scale))
        for polygon in view.polygons
        for part in getattr(polygon, "geoms", [polygon])
    ]
    if fill:
        ax.add_collection(PatchCollection(patches, facecolor=OOCYTE_FILL, edgecolor=OOCYTE_FILL,
                                          alpha=0.55, linewidth=1.0))
    else:
        ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor=INK,
                                          linewidth=1.4))


def _polygon_path(polygon: BaseGeometry, scale: float) -> MplPath:
    """A path with the polygon's holes, so fills leave them empty."""
    # Nonzero fill needs holes wound opposite the exterior.
    polygon = orient(polygon, sign=1.0)
    rings = [polygon.exterior, *polygon.interiors]
    return MplPath.make_compound_path(*(
        MplPath(np.asarray(ring.coords) * scale, closed=True) for ring in rings
    ))


def _draw_tiles(
    ax: Any,
    view: CutView,
    tiles: pd.DataFrame,
    colour: str,
    fill_alpha: float = 0.28,
    edge_alpha: float = 0.9,
    linewidth: float = 0.8,
    linestyle: Any = "solid",
    hatch: str | None = None,
) -> None:
    if tiles.empty:
        return
    s = view.scale
    rectangles = [
        Rectangle((x0 * s, y0 * s), (x1 - x0) * s, (y1 - y0) * s)
        for x0, y0, x1, y1 in tiles[["x0", "y0", "x1", "y1"]].itertuples(index=False)
    ]
    face = matplotlib.colors.to_rgba(colour, fill_alpha)
    edge = matplotlib.colors.to_rgba(colour, edge_alpha)
    ax.add_collection(PatchCollection(rectangles, facecolor=face, edgecolor=edge,
                                      linewidth=linewidth, linestyle=linestyle, hatch=hatch))


def _legend(fig: plt.Figure, handles: list[Any]) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.55 / fig.get_figheight()), fontsize=12,
               labelcolor=INK_SECONDARY, handlelength=1.6)


def _caption(fig: plt.Figure, parts: list[str]) -> None:
    fig.text(0.5, 0.22 / fig.get_figheight(), "  ·  ".join(parts), ha="center", va="center",
             fontsize=12, color=INK_SECONDARY)
