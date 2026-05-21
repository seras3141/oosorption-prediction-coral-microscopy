from __future__ import annotations

"""
Extract tissue cuts from NDPI whole-slide images.

Detection runs at the highest (smallest) pyramid level for speed.  Tissue is
located by saturation thresholding (default) or white-background subtraction,
making the approach robust to stain-intensity and background-colour variation
across slides.

Public API
----------
detect_and_save_cut_previews(ndpi_path, output_dir, ...)
    Detects cuts, saves downsampled preview images, and writes a JSON manifest.

create_pyramidal_tiffs_from_manifest(manifest_path, ndpi_path, output_dir, ...)
    Reads a verified manifest and writes a pyramidal TIFF per cut.

extract_specimen_cuts(ndpi_path, output_dir, ...)
    Backward-compatible wrapper that runs both phases.

parse_n_cuts_from_stem(stem)
    Returns the expected cut count encoded in the filename (e.g. ``_19-21`` → 3).
"""

import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import openslide
import tifffile
from skimage.measure import label, regionprops

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PREVIEW_TARGET_LONG_SIDE_PX = 2_000
SUPPORTED_PREVIEW_FORMATS = {"png", "tif"}
TISSUE_MASK_METHODS = {"saturation", "brightness"}


# ─────────────────────────── Geometry helpers ────────────────────────────── #

def box_area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def intersection_area(a, b):
    return (
        max(0, min(a[2], b[2]) - max(a[0], b[0])) *
        max(0, min(a[3], b[3]) - max(a[1], b[1]))
    )


def overlap_fraction(smaller, bigger):
    """Fraction of *smaller*'s area covered by *bigger*."""
    area = box_area(smaller)
    return intersection_area(smaller, bigger) / area if area else 0.0


def filter_by_asymmetric_overlap(boxes, overlap_thresh=0.7, min_area_ratio=2.0):
    """Return indices of boxes not largely contained within a much larger neighbour."""
    if not boxes:
        return []
    boxes_arr = np.asarray(boxes, dtype=float)
    areas = np.array([box_area(b) for b in boxes])
    order = np.argsort(-areas)
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i, idx_big in enumerate(order):
        if suppressed[idx_big]:
            continue
        for idx_small in order[i + 1:]:
            if suppressed[idx_small]:
                continue
            if areas[idx_big] < min_area_ratio * areas[idx_small]:
                continue
            if overlap_fraction(boxes_arr[idx_small], boxes_arr[idx_big]) >= overlap_thresh:
                suppressed[idx_small] = True

    return [i for i in range(len(boxes)) if not suppressed[i]]


# ───────────────────────── Filename utilities ────────────────────────────── #

def parse_n_cuts_from_stem(stem: str) -> Optional[int]:
    """Return the expected cut count encoded in a filename stem.

    The convention is ``{LOC}_{SEASON}_{COLONY}_{A}-{B}`` where the number
    of cuts equals ``B − A + 1``.  Returns ``None`` when the pattern is absent.

    Examples
    --------
    >>> parse_n_cuts_from_stem("CHN_AU_10_19-21")
    3
    >>> parse_n_cuts_from_stem("LHP_SP_6_3-4")
    2
    """
    m = re.search(r"_(\d+)-(\d+)$", stem)
    return int(m.group(2)) - int(m.group(1)) + 1 if m else None


# ──────────────────────── Slide / thumbnail I/O ──────────────────────────── #

def open_slide_and_thumbnail(ndpi_path, thumbnail_max_dim=None):
    """Open an NDPI slide and return its highest pyramid level as the detection image.

    Parameters
    ----------
    ndpi_path : str or Path
    thumbnail_max_dim : deprecated, ignored.

    Returns
    -------
    slide, thumb_rgb (HxWx3 uint8), (w0, h0), (thumb_w, thumb_h), scale
        *scale* converts thumbnail → level-0: ``level0_coord = thumb_coord / scale``
    """
    if thumbnail_max_dim is not None:
        warnings.warn(
            "thumbnail_max_dim is deprecated; the highest pyramid level is used "
            "directly for detection.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Check if file is NDPI and file exists
    if not str(ndpi_path).lower().endswith(".ndpi"):
        raise ValueError(f"File {ndpi_path} does not have .ndpi extension.")

    if not Path(ndpi_path).exists():
        raise FileNotFoundError(f"File {ndpi_path} not found.")

    slide = openslide.OpenSlide(str(ndpi_path))
    w0, h0 = slide.dimensions

    top_level = len(slide.level_dimensions) - 1
    thumb_w, thumb_h = slide.level_dimensions[top_level]
    downsample = slide.level_downsamples[top_level]

    thumb_rgb = np.array(
        slide.read_region((0, 0), top_level, (thumb_w, thumb_h)).convert("RGB")
    )

    logging.info(
        "%s  level-0: %dx%d  |  detection level %d: %dx%d (x%.0f downsample)",
        Path(ndpi_path).name, w0, h0, top_level, thumb_w, thumb_h, downsample,
    )

    return slide, thumb_rgb, (w0, h0), (thumb_w, thumb_h), 1.0 / downsample


# ────────────────────────── Tissue detection ─────────────────────────────── #

def create_tissue_mask(
    thumb_rgb: np.ndarray,
    background_brightness: int = 220,
    saturation_min: int = 10,
    method: str = "saturation",
    tissue_sat_min: int = 15,
    blur_ksize: int = 5,
) -> np.ndarray:
    """Return a binary tissue mask from a detection thumbnail.

    Two methods are available:

    ``'saturation'`` (default)
        Classifies a pixel as tissue when its HSV saturation exceeds
        *tissue_sat_min*.  Robust to gray or off-white slide backgrounds
        that the brightness method misclassifies as tissue.  H&E-stained
        tissue is distinctly pink/purple (high saturation) whereas glass,
        gray mounting medium, and debris are near-achromatic (low saturation).
        An optional Gaussian pre-blur (*blur_ksize*) suppresses isolated noise
        particles before thresholding.

    ``'brightness'``
        Legacy method: background = mean-RGB ≥ *background_brightness* AND
        HSV saturation < *saturation_min*.  Only appropriate when the slide
        background is near-white (mean RGB ≥ ~215).

    Parameters
    ----------
    thumb_rgb : np.ndarray
        H×W×3 uint8 RGB thumbnail.
    background_brightness : int
        Only used when ``method='brightness'``.  Mean-RGB threshold above
        which a pixel is a background candidate.
    saturation_min : int
        Only used when ``method='brightness'``.  HSV saturation below which
        a bright pixel is classified as background.
    method : {'saturation', 'brightness'}
        Masking strategy.  ``'saturation'`` is recommended for most slides.
    tissue_sat_min : int
        Minimum HSV saturation (0–255) for tissue classification.  Only used
        when ``method='saturation'``.  Default 15 works well for H&E-stained
        coral slides with gray backgrounds.
    blur_ksize : int
        Gaussian blur kernel size applied before thresholding to suppress
        isolated noise particles.  Must be odd; 0 disables blurring.
        Default 5.

    Returns
    -------
    np.ndarray
        uint8 mask — 1 = tissue, 0 = background.
    """
    if method not in TISSUE_MASK_METHODS:
        raise ValueError(f"method must be one of {sorted(TISSUE_MASK_METHODS)!r}, got {method!r}.")

    if blur_ksize > 0:
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        thumb_rgb = cv2.GaussianBlur(thumb_rgb, (ksize, ksize), 0)

    hsv_s = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]

    if method == "saturation":
        return (hsv_s >= tissue_sat_min).astype(np.uint8)

    # method == "brightness"
    bright = thumb_rgb.mean(axis=2) >= background_brightness
    low_sat = hsv_s < saturation_min
    return (~(bright & low_sat)).astype(np.uint8)


def clean_mask(mask, thumb_w, thumb_h, min_area_fraction):
    """Morphologically close/open *mask* and remove components below area threshold.

    Returns
    -------
    cleaned : uint8 mask
    min_area : float  — pixel threshold used (for downstream re-use)
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    min_area = min_area_fraction * thumb_w * thumb_h
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)
    for lid in range(1, n_labels):
        if stats[lid, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == lid] = 1

    return cleaned, min_area


def connect_tissue_fragments(
    cleaned_mask: np.ndarray,
    thumb_w: int,
    closing_frac: float = 0.03,
) -> np.ndarray:
    """Bridge intra-cut voids with a large morphological closing.

    *closing_frac* sets the structuring-element diameter as a fraction of
    thumbnail width.  Intra-cut voids are far smaller than inter-cut white
    space, so ≈ 3% safely connects fragments within a cut without merging
    separate cuts.
    """
    ksize = max(3, int(closing_frac * thumb_w))
    ksize += ksize % 2 == 0  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)


def get_regions(mask):
    regions = regionprops(label(mask))
    logging.info("Found %d candidate region(s)", len(regions))
    return regions


def regions_to_fullres_boxes(regions, min_area, thumb_w, thumb_h, scale, margin_frac):
    """Map skimage region bounding boxes to level-0 coordinates with margin."""
    boxes = []
    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        if (maxr - minr) * (maxc - minc) < min_area:
            continue
        h, w = maxr - minr, maxc - minc
        # expand by margin, clamp to thumbnail bounds
        minr = max(0,       minr - int(margin_frac * h))
        minc = max(0,       minc - int(margin_frac * w))
        maxr = min(thumb_h, maxr + int(margin_frac * h))
        maxc = min(thumb_w, maxc + int(margin_frac * w))
        # map to level-0; scale = 1/downsample so full_coord = thumb_coord / scale
        boxes.append((
            int(minc / scale), int(minr / scale),
            int(maxc / scale), int(maxr / scale),
        ))
    logging.info("%d box(es) before overlap filter", len(boxes))
    return boxes


# ──────────────────────── Output helpers ─────────────────────────────────── #

def crop_and_save_rois_pyramidal(
    slide: openslide.OpenSlide,
    boxes: list,
    ndpi_path,
    output_dir,
    tile_size: int = 256,
    output_levels: Optional[list] = None,
    *,
    cut_indices: Optional[list[int]] = None,
    skip_existing: bool = False,
) -> list:
    """Extract each ROI across pyramid levels and save as a pyramidal TIFF.

    All pyramid levels are embedded in a single file per cut, making it
    directly readable by OpenSlide / QuPath / napari without any additional
    conversion.  Resolution metadata (µm/px) is stored in TIFF tags.

    Parameters
    ----------
    output_levels : list of int or None
        Pyramid levels to embed.  ``None`` embeds all available levels.
    cut_indices : list of int or None
        Original cut indices to use in filenames.  ``None`` uses ``range(len(boxes))``.
    skip_existing : bool
        Return existing TIFF paths without rewriting them.

    Returns
    -------
    List of saved Path objects (one per cut).
    """
    stem = Path(ndpi_path).stem
    os.makedirs(output_dir, exist_ok=True)

    levels = output_levels or list(range(len(slide.level_dimensions)))
    mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0) or 0)
    mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0) or 0)

    # pixels-per-cm for TIFF resolution tags  (1 cm = 1e4 µm)
    res_kwargs = (
        {"resolution": (1e4 / mpp_x, 1e4 / mpp_y), "resolutionunit": 3}
        if mpp_x and mpp_y else {}
    )
    tiff_opts = dict(tile=(tile_size, tile_size), compression="deflate", photometric="rgb")

    if cut_indices is None:
        cut_indices = list(range(len(boxes)))
    if len(cut_indices) != len(boxes):
        raise ValueError("cut_indices must have the same length as boxes.")

    saved = []
    for idx, (x0, y0, x1, y1) in zip(cut_indices, boxes):
        w0, h0 = x1 - x0, y1 - y0
        outpath = Path(output_dir) / f"{stem}_cut{idx:03d}.tif"

        if skip_existing and outpath.exists():
            logging.info("Skipping existing %s", outpath.name)
            saved.append(outpath)
            continue

        arrays = []
        for lvl in levels:
            D = slide.level_downsamples[lvl]
            region = slide.read_region(
                (x0, y0), lvl, (max(1, int(w0 / D)), max(1, int(h0 / D)))
            ).convert("RGB")
            arrays.append(np.array(region))

        with tifffile.TiffWriter(str(outpath), bigtiff=True) as tif:
            tif.write(arrays[0], subifds=len(arrays) - 1, **tiff_opts, **res_kwargs)
            for arr in arrays[1:]:
                tif.write(arr, subfiletype=1, **tiff_opts)

        logging.info("Saved %s (%d level(s))", outpath.name, len(arrays))
        saved.append(outpath)

    return saved


def crop_and_save_rois(slide, boxes, ndpi_path, output_dir, output_format="tif", level=0):
    """Flat single-level ROI extraction — kept for backward compatibility."""
    stem = Path(ndpi_path).stem
    os.makedirs(output_dir, exist_ok=True)

    D = slide.level_downsamples[level]
    logging.info("Extracting flat ROIs at level %d (x%.1f downsample)", level, D)

    saved = []
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        w, h = max(1, int((x1 - x0) / D)), max(1, int((y1 - y0) / D))
        arr = np.array(slide.read_region((x0, y0), level, (w, h)).convert("RGB"))

        outpath = Path(output_dir) / f"{stem}_cut{idx:03d}.{output_format}"
        if output_format in ("tif", "tiff"):
            tifffile.imwrite(str(outpath), arr)
        else:
            cv2.imwrite(str(outpath), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        logging.info("Saved %s", outpath.name)
        saved.append(outpath)

    return saved


def _select_preview_level(
    slide: openslide.OpenSlide,
    box: tuple[int, int, int, int],
    preview_level: Optional[int] = None,
) -> int:
    """Return the pyramid level whose cut long side is closest to the preview target."""
    if preview_level is not None:
        if preview_level < 0 or preview_level >= len(slide.level_dimensions):
            raise ValueError(
                f"preview_level must be between 0 and {len(slide.level_dimensions) - 1}."
            )
        return preview_level

    x0, y0, x1, y1 = box
    long_side = max(x1 - x0, y1 - y0)
    level_sizes = [
        abs((long_side / slide.level_downsamples[level]) - PREVIEW_TARGET_LONG_SIDE_PX)
        for level in range(len(slide.level_dimensions))
    ]
    return int(np.argmin(level_sizes))


def _save_cut_previews(
    slide: openslide.OpenSlide,
    boxes: list[tuple[int, int, int, int]],
    ndpi_path,
    output_dir,
    preview_format: str = "png",
    preview_level: Optional[int] = None,
) -> list[Path]:
    """Save one downsampled preview image per detected cut."""
    preview_format = preview_format.lower().lstrip(".")
    if preview_format not in SUPPORTED_PREVIEW_FORMATS:
        raise ValueError("preview_format must be one of 'png' or 'tif'.")

    stem = Path(ndpi_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_paths = []
    for idx, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        level = _select_preview_level(slide, box, preview_level)
        downsample = slide.level_downsamples[level]
        width = max(1, int((x1 - x0) / downsample))
        height = max(1, int((y1 - y0) / downsample))
        arr = np.array(slide.read_region((x0, y0), level, (width, height)).convert("RGB"))

        outpath = output_dir / f"{stem}_cut{idx:03d}_preview.{preview_format}"
        if preview_format == "png":
            ok = cv2.imwrite(str(outpath), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            if not ok:
                raise IOError(f"Could not write preview image: {outpath}")
        else:
            tifffile.imwrite(str(outpath), arr)

        logging.info(
            "Saved %s from level %d (%dx%d, x%.1f downsample)",
            outpath.name, level, width, height, downsample,
        )
        preview_paths.append(outpath)

    return preview_paths


# ─────────────────────────── Manifest ────────────────────────────────────── #

def write_cut_manifest(
    output_dir,
    ndpi_path,
    slide,
    boxes,
    saved_paths,
    preview_paths: Optional[list] = None,
) -> Path:
    """Write a JSON sidecar recording each cut's level-0 bounding box and slide metadata.

    This manifest is consumed by the annotation-remapping step to translate
    GeoJSON coordinates (whole-slide level-0 space) into cut-local pixel space.

    Schema
    ------
    {
      "source_ndpi": "CHN_AU_10_19-21.ndpi",
      "level0_dimensions": [165120, 41472],
      "mpp_x": 0.2271,
      "mpp_y": 0.2271,
      "cuts": [
        {
          "index": 0,
          "name": "CHN_AU_10_19-21_cut000",
          "level0_bbox": {"x0": ..., "y0": ..., "x1": ..., "y1": ...},
          "level0_size": [width, height],
          "preview_path": "CHN_AU_10_19-21_cut000_preview.png"
        }, ...
      ]
    }
    """
    w0, h0 = slide.dimensions
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    if len(saved_paths) != len(boxes):
        raise ValueError("saved_paths must have the same length as boxes.")
    if preview_paths is not None and len(preview_paths) != len(boxes):
        raise ValueError("preview_paths must have the same length as boxes.")

    stem = Path(ndpi_path).stem
    cuts = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cut = {
            "index":       i,
            "name":        f"{stem}_cut{i:03d}",
            "level0_bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "level0_size": [x1 - x0, y1 - y0],
        }
        if preview_paths is not None:
            cut["preview_path"] = Path(preview_paths[i]).name
        cuts.append(cut)

    manifest = {
        "source_ndpi":       Path(ndpi_path).name,
        "level0_dimensions": [w0, h0],
        "mpp_x":             float(mpp_x) if mpp_x else None,
        "mpp_y":             float(mpp_y) if mpp_y else None,
        "cuts":              cuts,
    }

    out = Path(output_dir) / f"{stem}_cuts.json"
    out.write_text(json.dumps(manifest, indent=2))
    logging.info("Manifest written → %s", out.name)
    return out


def detect_and_save_cut_previews(
    ndpi_path,
    output_dir,
    *,
    preview_format: str = "png",
    preview_level: Optional[int] = None,
    write_manifest: bool = True,
    n_boxes: Optional[int] = None,
    mask_method: str = "saturation",
    tissue_sat_min: int = 15,
    blur_ksize: int = 5,
    background_brightness: int = 220,
    saturation_min: int = 10,
    closing_frac: float = 0.03,
    min_area_fraction: float = 0.0001,
    margin_frac: float = 0.05,
    overlap_thresh: float = 0.7,
    min_area_ratio: float = 2.0,
) -> tuple[list[Path], list[tuple[int, int, int, int]]]:
    """Detect tissue cuts, save small previews, and optionally write a manifest.

    Parameters
    ----------
    ndpi_path : str or Path
        Source NDPI slide.  Coordinates in the returned boxes are level-0 pixels.
    output_dir : str or Path
        Directory for preview images and the optional ``{stem}_cuts.json`` manifest.
    preview_format : {'png', 'tif'}
        Lossless preview format.  PNG is the default for easy visual inspection.
    preview_level : int or None
        Pyramid level to read for previews.  ``None`` selects the level whose cut
        long side is closest to 2 000 px.
    write_manifest : bool
        Write a manifest containing level-0 bounding boxes and preview filenames.
    mask_method : {'saturation', 'brightness'}
        Tissue detection strategy forwarded to ``create_tissue_mask``.
        ``'saturation'`` (default) classifies pixels as tissue when their HSV
        saturation exceeds *tissue_sat_min*; robust to gray backgrounds.
        ``'brightness'`` uses the legacy brightness-threshold approach suitable
        for near-white backgrounds only.
    tissue_sat_min : int
        Minimum HSV saturation for tissue pixels.  Only used when
        ``mask_method='saturation'``.
    blur_ksize : int
        Gaussian pre-blur kernel size applied before masking.  0 disables blur.

    Returns
    -------
    preview_paths, boxes
        Preview paths and corresponding level-0 bounding boxes.

    Examples
    --------
    >>> parse_n_cuts_from_stem("CHN_AU_10_19-21")
    3
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    slide, thumb_rgb, (_w0, _h0), (thumb_w, thumb_h), scale = open_slide_and_thumbnail(ndpi_path)

    try:
        if n_boxes is None:
            n_boxes = parse_n_cuts_from_stem(Path(ndpi_path).stem)
            if n_boxes is not None:
                logging.info("Auto-derived n_boxes=%d from filename", n_boxes)
            else:
                logging.warning("Could not parse n_boxes from filename; all regions kept.")

        raw_mask = create_tissue_mask(
            thumb_rgb,
            background_brightness=background_brightness,
            saturation_min=saturation_min,
            method=mask_method,
            tissue_sat_min=tissue_sat_min,
            blur_ksize=blur_ksize,
        )
        logging.info("Tissue mask: method=%r  tissue_sat_min=%d  blur_ksize=%d",
                     mask_method, tissue_sat_min, blur_ksize)
        cleaned_mask, min_area = clean_mask(raw_mask, thumb_w, thumb_h, min_area_fraction)
        connected_mask = connect_tissue_fragments(cleaned_mask, thumb_w, closing_frac)
        regions = get_regions(connected_mask)
        if not regions:
            logging.warning("No tissue regions detected — check background_brightness threshold.")
            return [], []

        full_boxes = regions_to_fullres_boxes(
            regions, min_area, thumb_w, thumb_h, scale, margin_frac
        )
        if not full_boxes:
            logging.warning("No ROIs survived the area filter.")
            return [], []

        keep_idx = filter_by_asymmetric_overlap(full_boxes, overlap_thresh, min_area_ratio)
        full_boxes = [full_boxes[i] for i in keep_idx]
        logging.info("%d box(es) after overlap filter", len(full_boxes))

        if n_boxes is not None and len(full_boxes) > n_boxes:
            full_boxes = sorted(full_boxes, key=box_area, reverse=True)[:n_boxes]
        full_boxes = sorted(full_boxes, key=lambda b: b[0])

        logging.info("Final: %d cut(s)", len(full_boxes))
        for i, box in enumerate(full_boxes):
            logging.info("  cut %d  x0=%d  area=%d px²", i, box[0], box_area(box))

        preview_paths = _save_cut_previews(
            slide,
            full_boxes,
            ndpi_path,
            output_dir,
            preview_format=preview_format,
            preview_level=preview_level,
        )
        if write_manifest:
            write_cut_manifest(
                output_dir,
                ndpi_path,
                slide,
                full_boxes,
                preview_paths,
                preview_paths=preview_paths,
            )
        return preview_paths, full_boxes
    finally:
        slide.close()


def create_pyramidal_tiffs_from_manifest(
    manifest_path,
    ndpi_path,
    output_dir,
    *,
    output_levels: Optional[list[int]] = None,
    tile_size: int = 256,
    skip_existing: bool = True,
) -> list[Path]:
    """Create pyramidal BigTIFF cuts from a verified cut manifest.

    Parameters
    ----------
    manifest_path : str or Path
        ``{stem}_cuts.json`` manifest produced by ``detect_and_save_cut_previews``.
    ndpi_path : str or Path
        Source NDPI slide.  Its filename must match ``source_ndpi`` in the manifest.
    output_dir : str or Path
        Directory where ``{stem}_cutNNN.tif`` files are written.
    output_levels : list[int] or None
        Pyramid levels to embed.  ``None`` embeds all available levels.
    tile_size : int
        TIFF tile edge length in pixels.
    skip_existing : bool
        When true, existing TIFFs are returned without being rewritten.

    Returns
    -------
    list of Path
        One TIFF path per manifest cut, including skipped existing files.

    Examples
    --------
    >>> parse_n_cuts_from_stem("LHP_SP_6_3-4")
    2
    """
    manifest_path = Path(manifest_path)
    ndpi_path = Path(ndpi_path)
    output_dir = Path(output_dir)
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("source_ndpi") != ndpi_path.name:
        raise ValueError(
            f"Manifest source_ndpi={manifest.get('source_ndpi')!r} does not match "
            f"ndpi_path={ndpi_path.name!r}."
        )

    boxes = []
    cut_indices = []
    for cut in manifest.get("cuts", []):
        bbox = cut["level0_bbox"]
        boxes.append((bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]))
        cut_indices.append(int(cut["index"]))

    if not boxes:
        logging.warning("Manifest contains no cuts: %s", manifest_path)
        return []

    slide = openslide.OpenSlide(str(ndpi_path))
    try:
        return crop_and_save_rois_pyramidal(
            slide,
            boxes,
            ndpi_path,
            output_dir,
            tile_size=tile_size,
            output_levels=output_levels,
            cut_indices=cut_indices,
            skip_existing=skip_existing,
        )
    finally:
        slide.close()


# ─────────────────────────── Main entry point ────────────────────────────── #

def extract_specimen_cuts(
    ndpi_path,
    output_dir,
    # --- output ---
    output_format="pyramidal",    # 'pyramidal' | 'tif' | 'png' | 'jpg'
    output_levels=None,           # pyramid levels to embed (None → all); pyramidal only
    write_manifest=True,
    # --- detection ---
    n_boxes=None,                 # None → auto-derived from filename
    mask_method="saturation",     # 'saturation' (default) | 'brightness'
    tissue_sat_min=15,            # HSV-S threshold; used when mask_method='saturation'
    blur_ksize=5,                 # Gaussian pre-blur size; 0 to disable
    background_brightness=220,    # mean-RGB threshold; used when mask_method='brightness'
    saturation_min=10,            # HSV-S upper bound for background; mask_method='brightness'
    closing_frac=0.03,
    min_area_fraction=0.0001,
    margin_frac=0.05,
    overlap_thresh=0.7,
    min_area_ratio=2.0,
    # --- deprecated (kept for backward compatibility) ---
    level=0,
    thumbnail_max_dim=None,
    pink_hue_range=None,
):
    """Detect and extract tissue cuts from an NDPI whole-slide image.

    Parameters
    ----------
    ndpi_path : str or Path
    output_dir : str or Path
    output_format : {'pyramidal', 'tif', 'png', 'jpg'}
        'pyramidal' (default) writes a multi-resolution TIFF per cut.
        Other values write a flat single-level image via crop_and_save_rois.
    output_levels : list[int] or None
        Pyramid levels to embed in the pyramidal TIFF.  None embeds all levels.
    write_manifest : bool
        Write ``{stem}_cuts.json`` with level-0 bounding boxes for each cut.
    n_boxes : int or None
        Expected number of cuts.  None → auto-derived from the filename
        (e.g. ``CHN_AU_10_19-21`` → 3).
    mask_method : {'saturation', 'brightness'}
        Tissue detection strategy.  ``'saturation'`` (default) classifies tissue
        by HSV saturation — robust to gray backgrounds.  ``'brightness'`` uses
        the legacy brightness-threshold approach for near-white backgrounds.
    tissue_sat_min : int
        Minimum HSV saturation (0–255) for tissue pixels.  Only used when
        ``mask_method='saturation'`` (default 15).
    blur_ksize : int
        Gaussian pre-blur kernel applied before masking (0 = disabled, default 5).
    background_brightness : int
        Mean-RGB threshold for background pixels.  Only used when
        ``mask_method='brightness'`` (default 220).
    closing_frac : float
        Closing kernel diameter as a fraction of thumbnail width (default 0.03).
        Bridges intra-cut voids without merging separately placed cuts.
    level : int
        Pyramid level for flat output (ignored when output_format='pyramidal').
    """
    for name, val in [
        ("thumbnail_max_dim", thumbnail_max_dim),
        ("pink_hue_range", pink_hue_range),
    ]:
        if val is not None:
            warnings.warn(
                f"'{name}' is deprecated and has no effect.",
                DeprecationWarning,
                stacklevel=2,
            )

    os.makedirs(output_dir, exist_ok=True)

    preview_paths, full_boxes = detect_and_save_cut_previews(
        ndpi_path,
        output_dir,
        write_manifest=write_manifest,
        n_boxes=n_boxes,
        mask_method=mask_method,
        tissue_sat_min=tissue_sat_min,
        blur_ksize=blur_ksize,
        background_brightness=background_brightness,
        saturation_min=saturation_min,
        closing_frac=closing_frac,
        min_area_fraction=min_area_fraction,
        margin_frac=margin_frac,
        overlap_thresh=overlap_thresh,
        min_area_ratio=min_area_ratio,
    )
    if not full_boxes:
        return []

    if output_format == "pyramidal":
        if write_manifest:
            manifest_path = Path(output_dir) / f"{Path(ndpi_path).stem}_cuts.json"
            saved = create_pyramidal_tiffs_from_manifest(
                manifest_path,
                ndpi_path,
                output_dir,
                output_levels=output_levels,
            )
        else:
            slide = openslide.OpenSlide(str(ndpi_path))
            try:
                saved = crop_and_save_rois_pyramidal(
                    slide,
                    full_boxes,
                    ndpi_path,
                    output_dir,
                    output_levels=output_levels,
                )
            finally:
                slide.close()
    else:
        slide = openslide.OpenSlide(str(ndpi_path))
        try:
            saved = crop_and_save_rois(
                slide,
                full_boxes,
                ndpi_path,
                output_dir,
                output_format=output_format,
                level=level,
            )
            if write_manifest:
                write_cut_manifest(
                    output_dir,
                    ndpi_path,
                    slide,
                    full_boxes,
                    saved,
                    preview_paths=preview_paths,
                )
        finally:
            slide.close()

    logging.info("Done — %d cut(s) saved to %s", len(saved), output_dir)
    return saved
