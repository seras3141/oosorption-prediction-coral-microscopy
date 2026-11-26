"""Per-oocyte scoring geometry: scoring cells, oocytes and their overlaps, per cut."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.data_preparation.annotations import extract_stage
from src.modeling.tile_labels import load_annotation_features

FOUND_COVERAGE = 0.5
MIN_SCORABLE_SHARE = 0.5
OPERATING_RECALLS = (0.80, 0.90)
UM2_PER_CM2 = 1e8
# Areas summed in a different order differ by rounding.
COVERAGE_RTOL = 1e-9


@dataclass(frozen=True)
class CellGrid:
    """The scoring cells of one tile size on one cut: the tile grid's stride boxes."""

    tile_size: int
    stride: int
    width: int
    height: int

    @property
    def offset(self) -> float:
        return (self.tile_size - self.stride) / 2

    @property
    def n_cols(self) -> int:
        return math.ceil(self.width / self.stride)

    @property
    def n_rows(self) -> int:
        return math.ceil(self.height / self.stride)

    def cell_box(self, row: int, col: int) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` of the cell owned by tile ``(row, col)``."""
        x0, x1 = _span(col, self.offset, self.stride, self.width)
        y0, y1 = _span(row, self.offset, self.stride, self.height)
        return x0, y0, x1, y1


@dataclass(frozen=True)
class CutGeometry:
    """One cut's scored cells, oocytes, and the areas they share."""

    cut_name: str
    tile_size: int
    um2_per_px: float
    cells: pd.DataFrame = field(repr=False)
    oocytes: pd.DataFrame = field(repr=False)
    overlaps: pd.DataFrame = field(repr=False)


def tile_stride(tile_size: int, overlap: float) -> int:
    """The grid stride ``generate_tiles`` uses for a tile size and overlap."""
    return max(1, round(tile_size * (1 - overlap)))


def cell_grid(manifest: dict[str, Any], tile_size: int) -> CellGrid:
    """The cell grid of a tile manifest at one size, checked against its tiles."""
    bbox = manifest["level0_bbox"]
    grid = CellGrid(tile_size=tile_size, stride=tile_stride(tile_size, manifest["overlap"]),
                    width=bbox["x1"] - bbox["x0"], height=bbox["y1"] - bbox["y0"])
    for tile in manifest["tiles"]:
        if tile["tile_size"] != tile_size:
            continue
        local = tile["cut_local_bbox"]
        if (local["x0"], local["y0"]) != (tile["col"] * grid.stride, tile["row"] * grid.stride):
            raise ValueError(
                f"{manifest['cut_name']}: tile {tile['tile_id']} starts at "
                f"({local['x0']}, {local['y0']}), not at its grid position "
                f"({tile['col']} × {grid.stride}, {tile['row']} × {grid.stride})"
            )
        if not (tile["row"] < grid.n_rows and tile["col"] < grid.n_cols):
            raise ValueError(f"{manifest['cut_name']}: tile {tile['tile_id']} is outside the cut")
        for end, extent, pad in (("x1", grid.width, "pad_right"), ("y1", grid.height, "pad_bottom")):
            if local[end] > extent or (tile.get(pad, 0) > 0 and local[end] != extent):
                raise ValueError(
                    f"{manifest['cut_name']}: tile {tile['tile_id']} ends at {end}={local[end]}, "
                    f"but the manifest's cut is {extent} px; the image was tiled at another size"
                )
    return grid


def load_oocytes(annotations_geojson_path: str | Path, mpp_x: float, mpp_y: float) -> pd.DataFrame:
    """One row per usable annotation, keyed by feature position, with stage, area and size."""
    rows = []
    for feature in load_annotation_features(annotations_geojson_path):
        polygon = feature.polygon
        area = float(polygon.area)
        rows.append({
            "oocyte": feature.position,
            "stage": extract_stage(feature.properties),
            "polygon": polygon,
            "area": area,
            "equivalent_diameter_um": 2 * math.sqrt(area * mpp_x * mpp_y / math.pi),
        })
    oocytes = pd.DataFrame(rows, columns=["oocyte", "stage", "polygon", "area",
                                          "equivalent_diameter_um"])
    return oocytes.astype({"oocyte": "int64", "stage": "Int64", "area": "float64",
                           "equivalent_diameter_um": "float64"})


def cut_geometry(manifest: dict[str, Any], annotations_geojson_path: str | Path,
                 tile_size: int, scored_tile_ids: set[str]) -> CutGeometry:
    """Scored cells, oocytes and overlaps of one cut; every kept tile needs a probability."""
    cut_name = manifest["cut_name"]
    grid = cell_grid(manifest, tile_size)
    tiles = [tile for tile in manifest["tiles"] if tile["tile_size"] == tile_size]
    kept = {tile["tile_id"] for tile in tiles}
    unpredicted = sorted(kept - scored_tile_ids)
    if unpredicted:
        raise ValueError(
            f"{cut_name}: {len(unpredicted)} kept {tile_size} px tile(s) have no held-out "
            f"probability, e.g. {unpredicted[:3]}"
        )
    unknown = sorted(scored_tile_ids - kept)
    if unknown:
        raise ValueError(
            f"{cut_name}: {len(unknown)} probability row(s) name tiles not in its manifest, "
            f"e.g. {unknown[:3]}"
        )

    mpp_x, mpp_y = manifest.get("mpp_x"), manifest.get("mpp_y")
    if not mpp_x or not mpp_y:
        raise ValueError(f"{cut_name}: the tile manifest records no pixel size (mpp_x, mpp_y)")
    oocytes = load_oocytes(annotations_geojson_path, mpp_x, mpp_y)
    cell_boxes = [box(*grid.cell_box(tile["row"], tile["col"])) for tile in tiles]
    tree = STRtree(cell_boxes)
    annotation_area = np.zeros(len(cell_boxes))
    if len(oocytes):
        union = unary_union(list(oocytes["polygon"]))
        for position in tree.query(union):
            annotation_area[position] = cell_boxes[position].intersection(union).area

    cells = pd.DataFrame({
        "tile_id": [tile["tile_id"] for tile in tiles],
        "row": [tile["row"] for tile in tiles],
        "col": [tile["col"] for tile in tiles],
        "area": [cell.area for cell in cell_boxes],
        "annotation_area": annotation_area,
    })

    overlap_rows = []
    if len(oocytes):
        for oocyte, polygon in zip(oocytes["oocyte"], oocytes["polygon"]):
            for position in tree.query(polygon):
                area = cell_boxes[position].intersection(polygon).area
                if area > 0:
                    overlap_rows.append({"tile_id": tiles[position]["tile_id"],
                                         "oocyte": int(oocyte), "area": float(area)})
    overlaps = pd.DataFrame(overlap_rows, columns=["tile_id", "oocyte", "area"]).astype(
        {"tile_id": "object", "oocyte": "int64", "area": "float64"})

    scorable = overlaps.groupby("oocyte")["area"].sum()
    oocytes = oocytes.assign(
        scorable_area=oocytes["oocyte"].map(scorable).fillna(0.0).to_numpy(dtype=float),
        cut_name=cut_name,
    )
    return CutGeometry(cut_name=cut_name, tile_size=tile_size, um2_per_px=mpp_x * mpp_y,
                       cells=cells, oocytes=oocytes, overlaps=overlaps)


def oocyte_coverage(geometry: CutGeometry, probabilities: pd.Series, threshold: float) -> pd.Series:
    """Share of each oocyte's scorable area covered by cells with ``prob >= threshold``."""
    touched = set(geometry.overlaps["tile_id"])
    missing = sorted(touched - set(probabilities.index[probabilities.notna()]))
    if missing:
        raise ValueError(
            f"{geometry.cut_name}: {len(missing)} cell(s) under an oocyte have no probability, "
            f"e.g. {missing[:3]}"
        )
    numeric = pd.to_numeric(probabilities)
    positive = set(numeric.index[numeric.to_numpy() >= threshold])
    covered = geometry.overlaps.loc[geometry.overlaps["tile_id"].isin(positive)]
    covered_area = covered.groupby("oocyte")["area"].sum()
    oocytes = geometry.oocytes.set_index("oocyte")
    scorable = oocytes["scorable_area"].replace(0.0, np.nan)
    return covered_area.reindex(oocytes.index, fill_value=0.0) / scorable


def _span(index: int, offset: float, stride: int, extent: int) -> tuple[float, float]:
    # Last tile within offset owns empty cell.
    low = 0.0 if index == 0 else min(offset + index * stride, extent)
    high = min(offset + (index + 1) * stride, extent)
    return low, high


def threshold_grid(n: int = 2001, low: float = 1e-6) -> np.ndarray:
    """Ascending thresholds, evenly spaced in logit between ``low`` and ``1 - low``, plus 0 and 1."""
    logits = np.linspace(math.log(low / (1 - low)), math.log((1 - low) / low), n)
    return np.concatenate([[0.0], 1 / (1 + np.exp(-logits)), [1.0]])


def found_thresholds(geometry: CutGeometry, probabilities: pd.Series) -> pd.Series:
    """Each oocyte's largest threshold reaching ``FOUND_COVERAGE``; NaN if under-scorable."""
    numeric = pd.to_numeric(probabilities)
    overlaps = geometry.overlaps.assign(prob=geometry.overlaps["tile_id"].map(numeric))
    if overlaps["prob"].isna().any():
        missing = sorted(overlaps.loc[overlaps["prob"].isna(), "tile_id"].unique())
        raise ValueError(f"{geometry.cut_name}: cells under an oocyte have no probability, "
                         f"e.g. {missing[:3]}")
    oocytes = geometry.oocytes.set_index("oocyte")
    scorable = oocytes["scorable_area"]
    judged = scorable >= MIN_SCORABLE_SHARE * oocytes["area"] * (1 - COVERAGE_RTOL)
    found = pd.Series(np.nan, index=scorable.index, dtype="float64")
    overlaps = overlaps.loc[overlaps["oocyte"].map(judged)]
    for oocyte, cells in overlaps.sort_values("prob", ascending=False).groupby("oocyte", sort=False):
        target = FOUND_COVERAGE * scorable[oocyte] * (1 - COVERAGE_RTOL)
        reached = np.flatnonzero(cells["area"].cumsum().to_numpy() >= target)
        found[oocyte] = cells["prob"].to_numpy()[reached[0]]
    return found


def per_oocyte_table(geometry: CutGeometry, probabilities: pd.Series, fold: int) -> pd.DataFrame:
    """One row per oocyte of a cut: its fold, stage, size, scorable share and found threshold."""
    oocytes = geometry.oocytes.drop(columns="polygon").assign(fold=fold)
    oocytes = oocytes.assign(scorable_share=oocytes["scorable_area"] / oocytes["area"])
    found = found_thresholds(geometry, probabilities)
    return oocytes.assign(found_threshold=oocytes["oocyte"].map(found).astype("float64"))


def false_blob_counts(geometry: CutGeometry, probabilities: pd.Series,
                      thresholds: np.ndarray) -> np.ndarray:
    """Per threshold, the 4-connected groups of positive cells that touch no annotation."""
    cells = geometry.cells
    numeric = pd.to_numeric(probabilities).reindex(cells["tile_id"]).to_numpy()
    if np.isnan(numeric).any():
        raise ValueError(f"{geometry.cut_name}: scored cells have no probability")
    shape = (int(cells["row"].max()) + 1, int(cells["col"].max()) + 1) if len(cells) else (0, 0)
    rows, cols = cells["row"].to_numpy(), cells["col"].to_numpy()
    touches = np.zeros(shape, dtype=bool)
    touches[rows, cols] = cells["annotation_area"].to_numpy() > 0
    counts = np.zeros(len(thresholds), dtype=np.int64)
    for i, threshold in enumerate(thresholds):
        positive = np.zeros(shape, dtype=bool)
        positive[rows, cols] = numeric >= threshold
        labels, n_blobs = ndimage.label(positive)
        if n_blobs:
            touched = np.bincount(labels[touches & positive], minlength=n_blobs + 1)[1:]
            counts[i] = int((touched == 0).sum())
    return counts


def fold_curve(geometries: list[CutGeometry], probabilities: pd.Series,
               oocytes: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Recall, false blobs per cm² and spill over ``thresholds`` for one fold's cuts."""
    numeric = pd.to_numeric(probabilities)
    cut_names = {geometry.cut_name for geometry in geometries}
    if set(oocytes["cut_name"]) - cut_names:
        raise ValueError(
            f"the per-oocyte table has cuts {sorted(set(oocytes['cut_name']) - cut_names)} that "
            "are not among the geometries"
        )
    untabled = sorted(geometry.cut_name for geometry in geometries
                      if len(geometry.oocytes) and geometry.cut_name not in set(oocytes["cut_name"]))
    if untabled:
        raise ValueError(f"cuts {untabled} have oocytes but no rows in the per-oocyte table")
    scorable = oocytes.dropna(subset=["found_threshold"])
    found = np.sort(scorable["found_threshold"].to_numpy())
    n_found = len(found) - np.searchsorted(found, thresholds, side="left")

    positive_um2 = np.zeros(len(thresholds))
    outside_um2 = np.zeros(len(thresholds))
    false_blobs = np.zeros(len(thresholds), dtype=np.int64)
    scored_um2 = 0.0
    for geometry in geometries:
        cells = geometry.cells
        prob = numeric.reindex(cells["tile_id"]).to_numpy()
        area = cells["area"].to_numpy() * geometry.um2_per_px
        outside = (cells["area"] - cells["annotation_area"]).to_numpy() * geometry.um2_per_px
        order = np.argsort(prob)
        sorted_prob = prob[order]
        first_positive = np.searchsorted(sorted_prob, thresholds, side="left")
        area_from = np.concatenate([np.cumsum(area[order][::-1])[::-1], [0.0]])
        outside_from = np.concatenate([np.cumsum(outside[order][::-1])[::-1], [0.0]])
        positive_um2 += area_from[first_positive]
        outside_um2 += outside_from[first_positive]
        false_blobs += false_blob_counts(geometry, numeric, thresholds)
        scored_um2 += float(area.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        spill = np.where(positive_um2 > 0, outside_um2 / positive_um2, np.nan)
    return pd.DataFrame({
        "threshold": thresholds,
        "n_scorable": len(found),
        "n_found": n_found,
        "recall": n_found / len(found) if len(found) else np.nan,
        "positive_um2": positive_um2,
        "false_blobs": false_blobs,
        "false_blobs_per_cm2": false_blobs / (scored_um2 / UM2_PER_CM2) if scored_um2 else np.nan,
        "spill": spill,
    })


def operating_point(curve: pd.DataFrame, target_recall: float) -> dict[str, float] | None:
    """The curve at its largest threshold whose recall reaches ``target_recall``, or None."""
    reached = curve.loc[curve["recall"] >= target_recall]
    if reached.empty:
        return None
    row = reached.loc[reached["threshold"].idxmax()]
    return {key: float(row[key]) for key in
            ("threshold", "recall", "false_blobs_per_cm2", "spill")}


def summarise_operating_points(curves: dict[int, pd.DataFrame],
                               targets: tuple[float, ...] = OPERATING_RECALLS) -> dict[str, Any]:
    """Per target recall: each fold's operating point, and the mean and range over folds."""
    summary = {}
    for target in targets:
        judged = {fold: curve for fold, curve in sorted(curves.items())
                  if (curve["n_scorable"] > 0).all()}
        points = {fold: operating_point(curve, target) for fold, curve in judged.items()}
        reached = {fold: point for fold, point in points.items() if point is not None}
        metrics = {}
        for metric in ("false_blobs_per_cm2", "spill", "threshold"):
            values = [point[metric] for point in reached.values()]
            metrics[metric] = ({"mean": float(np.mean(values)), "min": float(np.min(values)),
                                "max": float(np.max(values))} if values else None)
        summary[f"recall_{target:.2f}"] = {
            "per_fold": reached,
            "folds_not_reaching": [fold for fold, point in points.items() if point is None],
            "folds_without_judged_oocytes": sorted(set(curves) - set(judged)),
            **metrics,
        }
    return summary


def size_bin_edges(diameters_um: pd.Series) -> list[float]:
    """Quartile edges of oocyte equivalent diameters, open at both ends; tied edges merge."""
    diameters = diameters_um.dropna().to_numpy()
    if not len(diameters):
        raise ValueError("no oocyte diameters to bin")
    inner = np.unique(np.quantile(diameters, [0.25, 0.5, 0.75]))
    return [-math.inf, *map(float, inner), math.inf]


def recall_by(oocytes: pd.DataFrame, fold_thresholds: dict[int, float], column: str) -> pd.DataFrame:
    """Recall per value of ``column``, each oocyte judged at its fold's threshold."""
    judged = oocytes.dropna(subset=["found_threshold"])
    judged = judged.loc[judged["fold"].isin(fold_thresholds)]
    found = judged["found_threshold"] >= judged["fold"].map(fold_thresholds)
    grouped = found.groupby(judged[column], observed=False, dropna=False)
    return pd.DataFrame({"n_scorable": grouped.size(), "n_found": grouped.sum(),
                         "recall": grouped.mean()})

