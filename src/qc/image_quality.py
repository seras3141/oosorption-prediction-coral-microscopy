"""Per-cut image quality of the kept tiles: brightness, colour and sharpness."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from src.modeling.cut_exclusions import EXCLUDED, CutExclusions
from src.modeling.tile_index import _resolve_repo_path

QUALITY_TILES_PER_CUT = 60
QUALITY_SEED = 2026
BACKGROUND_GREY = 0.8
# Tiling's tissue_sat_min, on 0-255.
TISSUE_SAT_MIN = 15
MIN_TISSUE_SHARE = 0.05
EDGE_ERODE_PX = 2
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
STATS = ("brightness", "saturation", "red", "blue", "sharpness")
DEFAULT_CUTOFF_FRACTION = 0.3
DEFAULT_MIN_SHARPNESS_TILES = 10
BLURRED = "blurred"
TOO_FEW_TILES = "too few tiles"


def tissue_mask(image: np.ndarray) -> np.ndarray:
    """Tissue pixels of an RGB image on 0-255: stained, with HSV saturation of at least 15
    (tiling's threshold, without its pre-blur), and darker than glass. Neutral grey glass or
    medium, anything as bright as glass, and black padding are not tissue."""
    rgb = _unit_rgb(image)
    return _tissue(_saturation(rgb), rgb @ LUMA)


def _tissue(saturation: np.ndarray, grey: np.ndarray) -> np.ndarray:
    # OpenCV rounds S before tiling compares it.
    return (saturation >= (TISSUE_SAT_MIN - 0.5) / 255) & (grey < BACKGROUND_GREY)


def _unit_rgb(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)[..., :3] / 255


def _saturation(rgb: np.ndarray) -> np.ndarray:
    high, low = rgb.max(axis=2), rgb.min(axis=2)
    return np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0)


def tile_image_stats(image: np.ndarray) -> dict[str, float] | None:
    """Brightness, saturation, red and blue over the :func:`tissue_mask` pixels of an RGB tile,
    and sharpness (variance of the Laplacian of grey) over tissue away from its edges, so tissue
    outline does not count as focus — NaN when no tissue lies that far from an edge; ``None``
    when under 5 % is tissue."""
    rgb = _unit_rgb(image)
    grey = rgb @ LUMA
    saturation = _saturation(rgb)
    tissue = _tissue(saturation, grey)
    if tissue.mean() < MIN_TISSUE_SHARE:
        return None
    interior = ndimage.binary_erosion(tissue, iterations=EDGE_ERODE_PX)
    return {"brightness": float(grey[tissue].mean()),
            "saturation": float(saturation[tissue].mean()),
            "red": float(rgb[..., 0][tissue].mean()),
            "blue": float(rgb[..., 2][tissue].mean()),
            "sharpness": float(ndimage.laplace(grey)[interior].var()) if interior.any() else np.nan}


def cut_image_quality(tiles: pd.DataFrame, per_cut: int = QUALITY_TILES_PER_CUT,
                      seed: int = QUALITY_SEED) -> pd.DataFrame:
    """Per cut, the median of each statistic over a seeded sample of up to ``per_cut`` of its
    tiles, with each median's percentile rank among the cuts. ``tiles`` are tile index rows of
    one tile size; a relative ``png_path`` is read from the repo root. ``sharpness_tiles``
    counts the measured tiles with tissue away from its edges, which alone give a sharpness."""
    if tiles["tile_id"].duplicated().any():
        raise ValueError("tiles list a tile more than once")
    sizes = sorted(int(size) for size in tiles["tile_size"].unique())
    if len(sizes) > 1:
        raise ValueError(f"tiles mix tile sizes {sizes}; measure one size at a time")
    rows = []
    for cut_name, group in tiles.sort_values("tile_id").groupby("cut_name", sort=True):
        if len(group) > per_cut:
            group = group.sample(per_cut, random_state=seed)
        stats = []
        for png_path in group["png_path"]:
            with Image.open(_resolve_repo_path(png_path)) as image:
                measured = tile_image_stats(np.asarray(image.convert("RGB")))
            if measured is not None:
                stats.append(measured)
        cut_stats = pd.DataFrame(stats, columns=list(STATS))
        rows.append({"cut_name": cut_name, "tiles_measured": len(cut_stats),
                     "sharpness_tiles": int(cut_stats["sharpness"].notna().sum()),
                     **cut_stats.median().to_dict()})
    table = pd.DataFrame(rows, columns=["cut_name", "tiles_measured", "sharpness_tiles", *STATS])
    for stat in STATS:
        table[f"{stat}_percentile"] = 100 * table[stat].rank(pct=True)
    return table


def flag_cuts(quality: pd.DataFrame, exclusions: CutExclusions,
              cutoff_fraction: float = DEFAULT_CUTOFF_FRACTION,
              min_sharpness_tiles: int = DEFAULT_MIN_SHARPNESS_TILES) -> pd.DataFrame:
    """``quality`` with its ``listed`` status in ``exclusions``, each cut's sharpness over the
    median of the cuts not excluded with at least ``min_sharpness_tiles`` sharpness tiles, and a
    ``flag``: ``blurred`` below ``cutoff_fraction`` of that median, ``too few tiles`` under the
    minimum or without a sharpness; least sharp first.

    Raises
    ------
    ValueError
        If no cut can set the median, or it is 0.
    """
    listed = quality["cut_name"].map(exclusions.statuses).fillna("")
    enough = (quality["sharpness_tiles"] >= min_sharpness_tiles) & quality["sharpness"].notna()
    median = quality.loc[enough & (listed != EXCLUDED), "sharpness"].median()
    if not median > 0:
        raise ValueError(f"no reference sharpness: the cuts not excluded with at least "
                         f"{min_sharpness_tiles} sharpness tiles have median {median}")
    report = quality.assign(listed=listed, sharpness_vs_median=quality["sharpness"] / median)
    report["flag"] = np.select([~enough, report["sharpness_vs_median"] < cutoff_fraction],
                               [TOO_FEW_TILES, BLURRED], "")
    return report.sort_values(["sharpness", "cut_name"], na_position="first").reset_index(drop=True)
