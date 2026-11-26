"""Per-oocyte scoring geometry: scoring cells, oocytes and their overlaps, per cut."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.data_preparation.annotations import extract_stage
from src.modeling.tile_labels import load_annotation_features


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
    return CutGeometry(cut_name=cut_name, tile_size=tile_size, cells=cells, oocytes=oocytes,
                       overlaps=overlaps)


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
