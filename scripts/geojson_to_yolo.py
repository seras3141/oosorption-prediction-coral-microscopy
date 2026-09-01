#!/usr/bin/env python3
"""Convert QuPath-style GeoJSON oocyte annotations to YOLO training labels.

Each ``*.geojson`` file in ``--input-dir`` is paired with a whole-slide
image (``.ndpi`` / ``.svs`` / ``.tif``) of the same stem.  For every
annotated oocyte we:

1.  parse the oosorption stage from the feature properties,
2.  convert the geometry (Polygon, MultiPolygon, or traced LineString)
    into a shapely polygon,
3.  reduce that polygon to either an axis-aligned bounding box
    (``--task detect``, the default) or keep its outline as a
    polygon mask (``--task segment``),
4.  write a YOLO-formatted ``.txt`` file with normalised coordinates
    next to a ``classes.txt`` and a ``dataset.yaml`` ready for
    Ultralytics.

Coordinate handling
-------------------
GeoJSON coordinates are in **slide-level-0 pixels**.  YOLO expects
coordinates normalised to the dimensions of the *training image*.

There are three supported modes (``--mode``):

``slide``
    Emit one label file per slide; coordinates are normalised over the
    full slide width / height.  Use this when you plan to train YOLO on
    downsampled whole slides (rare).

``tile`` (recommended)
    Cut the slide into fixed-size, optionally overlapping tiles of size
    ``--tile-size`` at level 0 (or at ``--tile-level`` if downsampling),
    and emit one label per tile that contains at least one annotation.
    Tile *images* can be exported in the same pass with ``--export-tiles``
    (requires ``openslide``).

``manifest``
    Don't write any image-derived metadata at all – just emit a CSV
    manifest of bounding boxes in slide pixels.  Useful when the
    image-tile production happens elsewhere.

Examples
--------

Slide-level whole-image labels (no tiling):

    python scripts/geojson_to_yolo.py \\
        --input-dir sample_data_labeled \\
        --output-dir yolo_dataset \\
        --mode slide

512 px tiles cut from each slide (level 0) with both labels and PNG
images written, in YOLO segmentation format:

    python scripts/geojson_to_yolo.py \\
        --input-dir sample_data_labeled \\
        --output-dir yolo_dataset \\
        --mode tile --tile-size 1024 --tile-overlap 128 \\
        --task segment --export-tiles

Just dump per-instance bounding boxes (no images required):

    python scripts/geojson_to_yolo.py \\
        --input-dir sample_data_labeled \\
        --output-dir yolo_dataset \\
        --mode manifest
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Polygon,
    box as shapely_box,
    shape as shapely_shape,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.annotations import extract_stage

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover - openslide is optional
    openslide = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - PIL is optional, only for tile export
    Image = None


LOG = logging.getLogger("geojson_to_yolo")

# Class index -> human-readable name.  Stage 0 -> 0 ... Stage 4 -> 4.
DEFAULT_CLASSES = ["Stage_0", "Stage_1", "Stage_2", "Stage_3", "Stage_4"]


# --------------------------------------------------------------------------- #
# Annotation parsing
# --------------------------------------------------------------------------- #


@dataclass
class Instance:
    """One annotated oocyte after geometry normalisation."""

    feature_id: str
    stage: int
    polygon: Polygon
    geom_type: str  # original GeoJSON geometry type

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.polygon.bounds  # (minx, miny, maxx, maxy)


def feature_to_polygon(feat: dict) -> Polygon | MultiPolygon | None:
    """Convert a GeoJSON feature into a shapely polygon (closed)."""
    geom = feat.get("geometry") or {}
    gtype = geom.get("type")
    if gtype in ("Polygon", "MultiPolygon"):
        try:
            poly = shapely_shape(geom)
        except Exception:
            return None
    elif gtype == "LineString":
        coords = geom.get("coordinates") or []
        if len(coords) < 3:
            return None
        if coords[0] != coords[-1]:
            coords = list(coords) + [coords[0]]
        try:
            poly = Polygon(coords)
        except Exception:
            return None
    else:
        return None

    if not poly.is_valid:
        poly = poly.buffer(0)  # repair self-intersections
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def load_geojson(path: Path) -> list[Instance]:
    with path.open("r", encoding="utf-8") as fp:
        gj = json.load(fp)
    out: list[Instance] = []
    for idx, feat in enumerate(gj.get("features", [])):
        props = feat.get("properties") or {}
        stage = extract_stage(props)
        if stage is None:
            LOG.debug("Skipping unlabelled feature %s in %s", feat.get("id"), path.name)
            continue
        poly = feature_to_polygon(feat)
        if poly is None:
            LOG.debug("Skipping invalid geometry %s in %s", feat.get("id"), path.name)
            continue
        out.append(
            Instance(
                feature_id=str(feat.get("id") or f"{path.stem}-{idx}"),
                stage=stage,
                polygon=poly,
                geom_type=(feat.get("geometry") or {}).get("type", ""),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Slide dimensions
# --------------------------------------------------------------------------- #


def find_slide_for(stem: str, slide_dirs: Iterable[Path]) -> Path | None:
    """Look for an image file (.ndpi/.svs/.tif/.tiff/.png/.jpg) with the same stem."""
    for d in slide_dirs:
        for ext in (".ndpi", ".svs", ".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def slide_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) at level 0 of the slide referenced by ``path``."""
    if path.suffix.lower() in (".ndpi", ".svs"):
        if openslide is None:
            raise RuntimeError(
                "openslide is required to read NDPI/SVS dimensions; install "
                "openslide-python or use --slide-dim explicitly."
            )
        slide = openslide.OpenSlide(str(path))
        try:
            return slide.level_dimensions[0]
        finally:
            slide.close()
    # Anything else: rely on PIL.
    if Image is None:
        raise RuntimeError("PIL is required for non-NDPI slides.")
    with Image.open(path) as im:
        return im.size  # (w, h)


# --------------------------------------------------------------------------- #
# YOLO formatting helpers
# --------------------------------------------------------------------------- #


def yolo_bbox_line(cls: int, minx: float, miny: float, maxx: float, maxy: float,
                   img_w: int, img_h: int) -> str:
    """Return one ``class cx cy w h`` line normalised to [0,1]."""
    cx = ((minx + maxx) / 2) / img_w
    cy = ((miny + maxy) / 2) / img_h
    w = (maxx - minx) / img_w
    h = (maxy - miny) / img_h
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def yolo_polygon_line(cls: int, polygon: Polygon | MultiPolygon, img_w: int,
                      img_h: int, max_points: int = 200) -> str | None:
    """Return one ``class x1 y1 x2 y2 ...`` segmentation line.

    For MultiPolygons we keep the largest sub-polygon (YOLO-seg labels
    one shape per line).
    """
    if isinstance(polygon, MultiPolygon):
        polys = [g for g in polygon.geoms if isinstance(g, Polygon) and g.area > 0]
        if not polys:
            return None
        polygon = max(polys, key=lambda g: g.area)
    if not isinstance(polygon, Polygon) or polygon.exterior is None:
        return None
    coords = list(polygon.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    # downsample if extremely dense
    if len(coords) > max_points:
        idx = np.linspace(0, len(coords) - 1, max_points).astype(int)
        coords = [coords[i] for i in idx]
    if len(coords) < 3:
        return None
    flat = []
    for x, y in coords:
        nx = float(np.clip(x / img_w, 0.0, 1.0))
        ny = float(np.clip(y / img_h, 0.0, 1.0))
        flat.append(f"{nx:.6f}")
        flat.append(f"{ny:.6f}")
    return f"{cls} " + " ".join(flat)


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #


def iterate_tiles(width: int, height: int, tile_size: int, overlap: int):
    """Yield ``(x0, y0, x1, y1)`` for tiles covering the slide."""
    step = max(1, tile_size - overlap)
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            yield x0, y0, x1, y1
            if x1 == width:
                break
        if y1 == height:
            break


def emit_tile_labels(
    instances: list[Instance],
    slide_w: int,
    slide_h: int,
    tile_size: int,
    overlap: int,
    task: str,
    min_visible: float,
):
    """Yield ``(tile_box, lines)`` for every tile that contains at least one
    visible annotation.

    ``tile_box`` is ``(x0, y0, x1, y1)`` in slide pixels.
    """
    polys = [(inst, inst.polygon) for inst in instances]
    for x0, y0, x1, y1 in iterate_tiles(slide_w, slide_h, tile_size, overlap):
        tile_geom = shapely_box(x0, y0, x1, y1)
        tile_w = x1 - x0
        tile_h = y1 - y0
        lines: list[str] = []
        for inst, poly in polys:
            if not poly.intersects(tile_geom):
                continue
            clipped = poly.intersection(tile_geom)
            if clipped.is_empty:
                continue
            visible = clipped.area / poly.area
            if visible < min_visible:
                continue

            # shift into tile-local coords
            if task == "detect":
                minx, miny, maxx, maxy = clipped.bounds
                lines.append(
                    yolo_bbox_line(inst.stage,
                                   minx - x0, miny - y0,
                                   maxx - x0, maxy - y0,
                                   tile_w, tile_h)
                )
            else:  # segment
                # ``clipped`` may be a MultiPolygon after the intersection
                geoms = [clipped] if isinstance(clipped, Polygon) else list(clipped.geoms)
                for g in geoms:
                    if not isinstance(g, Polygon) or g.area <= 0:
                        continue
                    shifted = Polygon(
                        [(x - x0, y - y0) for x, y in g.exterior.coords]
                    )
                    line = yolo_polygon_line(inst.stage, shifted, tile_w, tile_h)
                    if line:
                        lines.append(line)

        if lines:
            yield (x0, y0, x1, y1), lines


# --------------------------------------------------------------------------- #
# Main per-slide processing
# --------------------------------------------------------------------------- #


def process_slide(
    geojson_path: Path,
    slide_dirs: list[Path],
    out_root: Path,
    args: argparse.Namespace,
    manifest_writer: csv.writer | None,
) -> dict:
    stem = geojson_path.stem
    instances = load_geojson(geojson_path)

    stats = {
        "geojson": geojson_path.name,
        "n_instances": len(instances),
        "n_tiles_with_labels": 0,
        "labels_written": 0,
    }
    if not instances:
        LOG.warning("No usable annotations in %s", geojson_path.name)
        return stats

    # Resolve slide dimensions
    slide_path: Path | None = None
    if args.mode != "manifest":
        slide_path = find_slide_for(stem, slide_dirs)
        if slide_path is None and args.slide_dim is None:
            LOG.warning(
                "No slide image found for %s and --slide-dim not given; "
                "falling back to bounding rectangle of the annotations.",
                stem,
            )
            xs = [c for inst in instances for c in (inst.bbox[0], inst.bbox[2])]
            ys = [c for inst in instances for c in (inst.bbox[1], inst.bbox[3])]
            slide_w = int(np.ceil(max(xs))) + 1
            slide_h = int(np.ceil(max(ys))) + 1
        elif args.slide_dim is not None:
            slide_w, slide_h = args.slide_dim
        else:
            slide_w, slide_h = slide_dimensions(slide_path)
    else:
        slide_w = slide_h = None

    # Manifest output (always written if requested)
    if manifest_writer is not None:
        for inst in instances:
            minx, miny, maxx, maxy = inst.bbox
            manifest_writer.writerow(
                [
                    stem,
                    inst.feature_id,
                    inst.stage,
                    DEFAULT_CLASSES[inst.stage] if 0 <= inst.stage < len(DEFAULT_CLASSES) else f"Stage_{inst.stage}",
                    f"{minx:.2f}",
                    f"{miny:.2f}",
                    f"{maxx:.2f}",
                    f"{maxy:.2f}",
                    f"{maxx - minx:.2f}",
                    f"{maxy - miny:.2f}",
                    f"{inst.polygon.area:.2f}",
                    inst.geom_type,
                ]
            )

    if args.mode == "manifest":
        return stats

    labels_dir = out_root / "labels"
    images_dir = out_root / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.export_tiles:
        images_dir.mkdir(parents=True, exist_ok=True)

    # ----- mode: slide -------------------------------------------------- #
    if args.mode == "slide":
        lines: list[str] = []
        for inst in instances:
            if args.task == "detect":
                minx, miny, maxx, maxy = inst.bbox
                lines.append(
                    yolo_bbox_line(inst.stage, minx, miny, maxx, maxy,
                                   slide_w, slide_h)
                )
            else:  # segment
                line = yolo_polygon_line(inst.stage, inst.polygon, slide_w, slide_h)
                if line:
                    lines.append(line)
        out_label = labels_dir / f"{stem}.txt"
        out_label.write_text("\n".join(lines) + "\n")
        stats["labels_written"] = len(lines)
        if args.export_tiles and slide_path is not None and openslide is not None:
            LOG.info("Exporting whole-slide thumbnail for %s", stem)
            slide = openslide.OpenSlide(str(slide_path))
            try:
                level = slide.get_best_level_for_downsample(
                    max(1, slide_w / args.thumb_max_dim)
                )
                lw, lh = slide.level_dimensions[level]
                img = slide.read_region((0, 0), level, (lw, lh)).convert("RGB")
                img.save(images_dir / f"{stem}.png")
            finally:
                slide.close()
        return stats

    # ----- mode: tile --------------------------------------------------- #
    slide_handle = None
    if args.export_tiles:
        if slide_path is None:
            LOG.warning("--export-tiles requested but no slide image found for %s", stem)
        elif openslide is None and slide_path.suffix.lower() in (".ndpi", ".svs"):
            LOG.warning("openslide unavailable; cannot export tiles for %s", stem)
        else:
            if slide_path.suffix.lower() in (".ndpi", ".svs"):
                slide_handle = openslide.OpenSlide(str(slide_path))
            else:
                slide_handle = ("pil", str(slide_path))

    try:
        for (x0, y0, x1, y1), lines in emit_tile_labels(
            instances,
            slide_w,
            slide_h,
            args.tile_size,
            args.tile_overlap,
            args.task,
            args.min_visible,
        ):
            tile_name = f"{stem}_x{x0}_y{y0}_w{x1-x0}_h{y1-y0}"
            (labels_dir / f"{tile_name}.txt").write_text("\n".join(lines) + "\n")
            stats["n_tiles_with_labels"] += 1
            stats["labels_written"] += len(lines)

            if slide_handle is not None:
                tile_w, tile_h = x1 - x0, y1 - y0
                if isinstance(slide_handle, tuple) and slide_handle[0] == "pil":
                    with Image.open(slide_handle[1]) as im:
                        crop = im.crop((x0, y0, x1, y1))
                        crop.save(images_dir / f"{tile_name}.png")
                else:
                    region = slide_handle.read_region((x0, y0), 0, (tile_w, tile_h))
                    region.convert("RGB").save(images_dir / f"{tile_name}.png")
    finally:
        if slide_handle is not None and not isinstance(slide_handle, tuple):
            slide_handle.close()
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Folder containing the GeoJSON annotation files.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write the YOLO dataset.")
    p.add_argument("--slide-dir", type=Path, action="append", default=None,
                   help="Folder(s) where to look for the matching slide image. "
                        "Repeat to search several locations. "
                        "Defaults to --input-dir.")
    p.add_argument("--mode", choices=("slide", "tile", "manifest"), default="tile",
                   help="Granularity of the output labels.")
    p.add_argument("--task", choices=("detect", "segment"), default="detect",
                   help="YOLO task: bounding boxes (detect) or polygon masks (segment).")
    p.add_argument("--tile-size", type=int, default=1024,
                   help="Tile side length in slide-level-0 pixels (mode=tile).")
    p.add_argument("--tile-overlap", type=int, default=128,
                   help="Tile overlap in pixels (mode=tile).")
    p.add_argument("--min-visible", type=float, default=0.3,
                   help="Drop annotations whose visible fraction inside a tile "
                        "is below this threshold.")
    p.add_argument("--slide-dim", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Override slide dimensions (use when no image file is "
                        "available).")
    p.add_argument("--export-tiles", action="store_true",
                   help="Also save the corresponding image tiles next to the "
                        "labels (requires openslide for NDPI / SVS).")
    p.add_argument("--thumb-max-dim", type=int, default=4000,
                   help="When --mode slide --export-tiles, max thumbnail edge.")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Fraction of slides set aside for validation in "
                        "dataset.yaml.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for the train/val split.")
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p.parse_args(argv)


def write_dataset_yaml(out_root: Path, train_files: list[str], val_files: list[str]):
    """Write an Ultralytics-compatible dataset.yaml."""
    yaml_path = out_root / "dataset.yaml"
    lines = [
        f"path: {out_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for i, name in enumerate(DEFAULT_CLASSES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n")

    # Also dump the lists as plain text for convenience.
    (out_root / "train.txt").write_text("\n".join(train_files) + "\n")
    (out_root / "val.txt").write_text("\n".join(val_files) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if not args.input_dir.exists():
        LOG.error("Input directory does not exist: %s", args.input_dir)
        return 2

    geojson_files = sorted(args.input_dir.glob("*.geojson"))
    if not geojson_files:
        LOG.error("No .geojson files found in %s", args.input_dir)
        return 2

    slide_dirs = list(args.slide_dir or [args.input_dir])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classes.txt").write_text(
        "\n".join(DEFAULT_CLASSES) + "\n"
    )

    # Manifest CSV is always produced for traceability.
    manifest_path = args.output_dir / "annotations_manifest.csv"
    manifest_fp = manifest_path.open("w", newline="", encoding="utf-8")
    manifest_writer = csv.writer(manifest_fp)
    manifest_writer.writerow([
        "slide_id", "feature_id", "stage", "stage_label",
        "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
        "bbox_w", "bbox_h", "polygon_area_px2", "geom_type",
    ])

    all_stats = []
    try:
        for gj_path in geojson_files:
            LOG.info("Processing %s", gj_path.name)
            stats = process_slide(gj_path, slide_dirs, args.output_dir,
                                  args, manifest_writer)
            all_stats.append(stats)
    finally:
        manifest_fp.close()

    # Summary
    total_inst = sum(s["n_instances"] for s in all_stats)
    total_lines = sum(s["labels_written"] for s in all_stats)
    LOG.info("Done.  %d annotations across %d files; %d label lines written.",
             total_inst, len(all_stats), total_lines)
    LOG.info("Manifest: %s", manifest_path)

    # Build a simple slide-level train/val split for dataset.yaml when we
    # actually wrote labels.
    if args.mode != "manifest":
        rng = np.random.default_rng(args.seed)
        slide_ids = [g.stem for g in geojson_files]
        rng.shuffle(slide_ids)
        n_val = max(1, int(len(slide_ids) * args.val_fraction))
        val = slide_ids[:n_val]
        train = slide_ids[n_val:]
        write_dataset_yaml(args.output_dir, train, val)
        LOG.info("dataset.yaml written – train: %d slides, val: %d slides",
                 len(train), len(val))

    return 0


if __name__ == "__main__":
    sys.exit(main())
