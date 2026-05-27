#!/usr/bin/env python3
"""Extract tissue cuts from NDPI whole-slide images and write a _cuts.json manifest.

Delegates to src.data_preparation.extract_ndpi_cuts.extract_specimen_cuts, which
writes pyramidal TIFFs named ``{stem}_cut{i:03d}.tif`` (0-based) and a
``{stem}_cuts.json`` manifest consumed by coral-remap-annotations and
coral-generate-tiles.

Usage:
    uv run coral-extract-rois --input /path/to/CHN_W_2_1-3.ndpi --outdir data/cuts
"""

import argparse
import glob
import logging
import math
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import openslide
import tifffile
from PIL import Image
from skimage.measure import label, regionprops

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.extract_ndpi_cuts import extract_specimen_cuts

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def extract_specimen_rois(
    ndpi_path,
    output_dir,
    thumbnail_max_dim=4000,
    pink_hue_range=(140, 180),   # for HSV hue in OpenCV (0–180); adjust as needed
    min_area_fraction=0.0001,    # min region area relative to thumbnail area
    margin_frac=0.05,            # add 5% margin to boxes
):
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Open slide and create thumbnail ---
    slide = openslide.OpenSlide(ndpi_path)
    w0, h0 = slide.dimensions  # full-res width, height

    # Determine thumbnail size keeping aspect ratio
    if max(w0, h0) > thumbnail_max_dim:
        scale = thumbnail_max_dim / max(w0, h0)
    else:
        scale = 1.0

    thumb_w = int(w0 * scale)
    thumb_h = int(h0 * scale)

    print(f"Full-resolution: {w0} x {h0}")
    print(f"Thumbnail: {thumb_w} x {thumb_h}, scale={scale:.4f}")

    thumbnail = slide.get_thumbnail((thumb_w, thumb_h))
    thumb_rgb = np.array(thumbnail)[:, :, :3]  # drop alpha if present

    # --- 2. Convert to HSV and threshold for pink tissue ---
    thumb_bgr = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2HSV)

    # HSV channels
    h, s, v = cv2.split(hsv)

    # Basic thresholds – tweak these for your images
    h_lower, h_upper = pink_hue_range
    # Pink is often around hue 140–180 in OpenCV’s 0–180 scale, but this varies.
    pink_mask = (
        (h >= h_lower) & (h <= h_upper) &
        (s > 50) &      # saturated enough
        (v > 50)        # not too dark
    ).astype(np.uint8)

    # Alternative / additional heuristic: R > G, R > B, etc.
    # r, g, b = thumb_rgb[:, :, 0], thumb_rgb[:, :, 1], thumb_rgb[:, :, 2]
    # pink_mask_rb = ((r > g + 10) & (r > b + 10)).astype(np.uint8)
    # pink_mask = cv2.bitwise_and(pink_mask, pink_mask_rb)

    # --- 3. Morphological cleanup ---
    kernel = np.ones((5, 5), np.uint8)
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel)

    # Remove very small specks
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pink_mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    min_area = min_area_fraction * thumb_w * thumb_h
    cleaned_mask = np.zeros_like(pink_mask)
    for label_id in range(1, num_labels):  # skip background
        if areas[label_id] >= min_area:
            cleaned_mask[labels == label_id] = 1

    # Optional: show mask for debugging
    # cv2.imwrite("debug_mask.png", cleaned_mask * 255)

    # If you prefer skimage's connected components and regionprops:
    labeled = label(cleaned_mask)
    regions = regionprops(labeled)

    print(f"Found {len(regions)} candidate specimens")

    if len(regions) == 0:
        print("No regions detected. Check thresholds or staining assumptions.")
        return

    # --- 4. For each region, compute bounding box in thumb coords ---
    thumb_area = thumb_w * thumb_h
    roi_idx = 0
    for r in regions:
        minr, minc, maxr, maxc = r.bbox  # (row_min, col_min, row_max, col_max)
        # Optionally reject regions that are too small / too large:
        area = (maxr - minr) * (maxc - minc)
        if area < min_area:
            continue

        # Expand bounding box by margin_frac
        height = maxr - minr
        width = maxc - minc
        margin_y = int(margin_frac * height)
        margin_x = int(margin_frac * width)

        minr = max(minr - margin_y, 0)
        minc = max(minc - margin_x, 0)
        maxr = min(maxr + margin_y, thumb_h)
        maxc = min(maxc + margin_x, thumb_w)

        # --- 5. Map to full-res coordinates ---
        # Thumbnail was scaled by `scale` from level 0:
        # thumb_coord = full_coord * scale  =>  full_coord = thumb_coord / scale
        x0 = int(minc / scale)
        y0 = int(minr / scale)
        x1 = int(maxc / scale)
        y1 = int(maxr / scale)

        width_full = x1 - x0
        height_full = y1 - y0

        print(f"ROI {roi_idx}: full-res box (x={x0}, y={y0}, w={width_full}, h={height_full})")

        # --- 6. Read region from slide at level 0 ---
        region = slide.read_region((x0, y0), 0, (width_full, height_full))
        region_rgb = np.array(region)[:, :, :3]  # drop alpha

        out_path = os.path.join(output_dir, f"specimen_{roi_idx:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved {out_path}")
        roi_idx += 1

    slide.close()
    print("Done.")


def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL image (RGB) to OpenCV BGR numpy array."""
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    # RGB -> BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def detect_roi_bboxes(thumb: Image.Image, min_area: int = 5000, edge_thresh1: int = 50, edge_thresh2: int = 150) -> list:
    """Detect candidate bounding boxes from a thumbnail image.

    Returns list of (x,y,w,h) in thumbnail pixel coordinates.
    """
    cv_img = pil_to_cv(thumb)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if cv_img.ndim == 3 else cv_img

    # Compute edges (Canny) which tends to capture high-contrast tissue regions.
    edges = cv2.Canny(gray, edge_thresh1, edge_thresh2)

    # Dilate to join nearby edges into contiguous blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Fill holes and remove small regions
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area >= min_area:
            boxes.append((x, y, w, h))

    # Optionally we can merge overlapping boxes (simple greedy merge)
    boxes = merge_boxes(boxes)
    logging.info('Detected %d candidate boxes after filtering/merge', len(boxes))
    return boxes


def merge_boxes(boxes, iou_threshold=0.1):
    """Merge overlapping boxes using simple IoU-based greedy merging."""
    if not boxes:
        return boxes
    boxes = [tuple(b) for b in boxes]
    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]

    areas = boxes_np[:, 2] * boxes_np[:, 3]
    order = areas.argsort()[::-1]

    keep = []
    used = set()
    for idx in order:
        if idx in used:
            continue
        bx = [x1[idx], y1[idx], x2[idx], y2[idx]]
        keep_idx = [idx]
        used.add(idx)
        for j in order:
            if j in used:
                continue
            bx2 = [x1[j], y1[j], x2[j], y2[j]]
            # compute IoU
            xx1 = max(bx[0], bx2[0])
            yy1 = max(bx[1], bx2[1])
            xx2 = min(bx[2], bx2[2])
            yy2 = min(bx[3], bx2[3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[idx] + areas[j] - inter + 1e-9)
            if iou > iou_threshold:
                # merge by expanding bx
                bx[0] = min(bx[0], bx2[0])
                bx[1] = min(bx[1], bx2[1])
                bx[2] = max(bx[2], bx2[2])
                bx[3] = max(bx[3], bx2[3])
                used.add(j)
        # final box
        nx = int(bx[0])
        ny = int(bx[1])
        nw = int(bx[2] - bx[0])
        nh = int(bx[3] - bx[1])
        keep.append((nx, ny, nw, nh))
    return keep


def extract_and_save(slide: openslide.OpenSlide, region_box_level0: tuple, outpath: Path):
    """Extract region at level 0 and save as TIFF. region_box_level0 = (x,y,w,h)."""
    x, y, w, h = region_box_level0
    logging.info('Reading region at level 0: x=%d y=%d w=%d h=%d', x, y, w, h)
    # read_region returns RGBA PIL image
    region = slide.read_region((x, y), 0, (w, h)).convert('RGB')
    arr = np.array(region)
    # Save as TIFF using tifffile for reliable multi-channel TIFFs
    tifffile.imwrite(str(outpath), arr)
    logging.info('Wrote %s', outpath)


def process_file(path: Path, outdir: Path, *, min_area=5000, padding=100, thumb_max_dim=1500, edge_thresh1=50, edge_thresh2=150):
    logging.info('Processing %s', path)
    slide = openslide.OpenSlide(str(path))
    width, height = slide.dimensions
    logging.info('Level-0 dimensions: %d x %d', width, height)

    # Create thumbnail preserving aspect ratio, limiting the larger side to thumb_max_dim
    if max(width, height) > thumb_max_dim:
        scale = max(width, height) / thumb_max_dim
        thumb_size = (int(width / scale), int(height / scale))
    else:
        thumb_size = (width, height)

    logging.info('Generating thumbnail of size %s for detection', thumb_size)
    thumb = slide.get_thumbnail(thumb_size).convert('RGB')
    thumb_w, thumb_h = thumb.size
    scale_x = width / thumb_w
    scale_y = height / thumb_h

    boxes = detect_roi_bboxes(thumb, min_area=min_area, edge_thresh1=edge_thresh1, edge_thresh2=edge_thresh2)

    # For each thumbnail box, map to level-0 coords, add padding, clamp and extract
    ensure_outdir(outdir)
    basename = path.stem
    for i, (tx, ty, tw, th) in enumerate(boxes, start=1):
        # map to level-0
        x0 = max(0, int(tx * scale_x) - padding)
        y0 = max(0, int(ty * scale_y) - padding)
        w0 = int(tw * scale_x) + 2 * padding
        h0 = int(th * scale_y) + 2 * padding
        # clamp to image
        w0 = min(w0, width - x0)
        h0 = min(h0, height - y0)
        if w0 <= 0 or h0 <= 0:
            logging.warning('Skipping invalid mapped region for box %s', (tx, ty, tw, th))
            continue
        outpath = outdir / f"{basename}_roi{i:03d}.tiff"
        extract_and_save(slide, (x0, y0, w0, h0), outpath)

    slide.close()

def get_ndpi_images(input_path: Path) -> List:
    ndpi_images = sorted(glob.glob(str(input_path / '*.ndpi')))
    return ndpi_images


def parse_args():
    p = argparse.ArgumentParser(description='Extract high-contrast ROIs from NDPI whole-slide images')
    p.add_argument('--input', '-i', required=True, help='Path to NDPI file (or a directory). If directory, all .ndpi files are processed')
    p.add_argument('--outdir', '-o', required=True, help='Output directory to write ROI TIFFs')
    p.add_argument('--min-area', type=int, default=5000, help='Minimum area (px in thumbnail) for candidate boxes')
    p.add_argument('--padding', type=int, default=100, help='Padding (px at level-0) to add around extracted regions')
    p.add_argument('--thumb-max-dim', type=int, default=1500, help='Maximum dimension (px) for thumbnail used in detection')
    p.add_argument('--edge-thresh1', type=int, default=50, help='Canny edge detector threshold1')
    p.add_argument('--edge-thresh2', type=int, default=150, help='Canny edge detector threshold2')
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    outdir = Path(args.outdir)

    if inp.is_file():
        targets = [inp]
    elif inp.is_dir():
        targets = sorted(inp.glob("*.ndpi"))
    else:
        raise FileNotFoundError(f"Input path not found: {inp}")

    if not targets:
        logging.warning("No NDPI files found under %s", inp)
        return

    for ndpi_path in targets:
        logging.info("Processing %s", ndpi_path.name)
        extract_specimen_cuts(
            ndpi_path,
            outdir / ndpi_path.stem,
            output_format="pyramidal",
            write_manifest=True,
        )


if __name__ == '__main__':
    main()
