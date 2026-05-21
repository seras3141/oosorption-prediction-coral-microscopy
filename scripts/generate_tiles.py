#!/usr/bin/env python3
"""Generate multi-scale PNG tiles from cut-local TIFFs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.generate_tiles import (
    DEFAULT_MIN_TISSUE_FRACTION,
    DEFAULT_OVERLAP,
    DEFAULT_TILE_SIZES,
    generate_tiles_for_cut,
)

LOG_FORMAT = "[%(levelname)s] %(message)s"
DEFAULT_CUTS_DIR = Path("data/cuts")
DEFAULT_OUTPUT_DIR = Path("data/tiles")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multi-scale PNG tiles from cut-local TIFFs."
    )
    parser.add_argument(
        "--cuts-dir",
        type=Path,
        default=DEFAULT_CUTS_DIR,
        help=f"Root cuts directory. Default: {DEFAULT_CUTS_DIR}/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root tile output directory. Default: {DEFAULT_OUTPUT_DIR}/",
    )
    parser.add_argument(
        "--stem",
        help="Process a single stem only, e.g. CHN_AU_10_19-21.",
    )
    parser.add_argument(
        "--tile-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_TILE_SIZES),
        help="One or more tile sizes in pixels. Default: 128 256 512 1024.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help=f"Fractional tile overlap. Default: {DEFAULT_OVERLAP:.2f}.",
    )
    parser.add_argument(
        "--min-tissue",
        type=float,
        default=DEFAULT_MIN_TISSUE_FRACTION,
        help=(
            "Minimum tissue fraction required to keep a tile. "
            f"Default: {DEFAULT_MIN_TISSUE_FRACTION:.2f}."
        ),
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-extract tiles even if the cut tile manifest already exists.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity. Default: INFO.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the tile-generation CLI."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)

    failures = 0
    manifests = []
    for cut_tiff_path, annotations_geojson_path, cuts_manifest_path in _discover_jobs(
        args.cuts_dir,
        stem=args.stem,
    ):
        try:
            manifests.append(
                generate_tiles_for_cut(
                    cut_tiff_path,
                    annotations_geojson_path,
                    cuts_manifest_path,
                    args.output_dir,
                    tile_sizes=tuple(args.tile_sizes),
                    overlap=args.overlap,
                    min_tissue_fraction=args.min_tissue,
                    skip_if_exists=not args.no_skip,
                )
            )
        except (FileNotFoundError, ValueError, ImportError) as exc:
            logging.error("%s", exc)
            failures += 1

    _print_summary(manifests)
    return 1 if failures else 0


def _discover_jobs(cuts_dir: Path, *, stem: str | None) -> list[tuple[Path, Path, Path]]:
    jobs: list[tuple[Path, Path, Path]] = []
    stems = [stem] if stem else sorted(path.name for path in cuts_dir.iterdir() if path.is_dir())
    for current_stem in stems:
        stem_dir = cuts_dir / current_stem
        manifest_path = stem_dir / f"{current_stem}_cuts.json"
        for cut_tiff_path in sorted(stem_dir.glob(f"{current_stem}_cut[0-9][0-9][0-9].tif")):
            cut_name = cut_tiff_path.stem
            annotations_geojson_path = stem_dir / f"{cut_name}_annotations.geojson"
            jobs.append((cut_tiff_path, annotations_geojson_path, manifest_path))
    return jobs


def _print_summary(manifests: list[dict]) -> None:
    print(f"{'stem':28} {'cut':>5} {'tiles_total':>12} {'tiles_oocyte':>13} {'skipped_tissue':>15}")
    for manifest in manifests:
        print(
            f"{manifest['stem']:28} "
            f"{manifest['cut_index']:03d} "
            f"{manifest['n_tiles_total']:12d} "
            f"{manifest['n_tiles_with_oocyte']:13d} "
            f"{manifest.get('n_tiles_skipped_tissue', 0):15d}"
        )


if __name__ == "__main__":
    sys.exit(main())