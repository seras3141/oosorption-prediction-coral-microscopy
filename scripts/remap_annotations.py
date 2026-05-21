#!/usr/bin/env python3
"""Remap QuPath GeoJSON annotations from whole-slide to cut-local coordinates."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.remap_annotations import (
    remap_annotations,
    remap_annotations_batch,
)

LOG_FORMAT = "[%(levelname)s] %(message)s"
DEFAULT_GEOJSON_DIR = Path("data/dataset_28_04")
DEFAULT_CUTS_DIR = Path("data/cuts")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Remap QuPath GeoJSON annotations from whole-slide to cut-local coordinates."
    )
    parser.add_argument(
        "--geojson-dir",
        type=Path,
        default=DEFAULT_GEOJSON_DIR,
        help=f"Source GeoJSON directory. Default: {DEFAULT_GEOJSON_DIR}/",
    )
    parser.add_argument(
        "--cuts-dir",
        type=Path,
        default=DEFAULT_CUTS_DIR,
        help=f"Root cuts directory. Default: {DEFAULT_CUTS_DIR}/",
    )
    parser.add_argument(
        "--stem",
        help="Process a single stem only, e.g. CHN_AU_10_19-21.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Reprocess even if the remapping report already exists.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity. Default: INFO.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the remapping CLI."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)

    try:
        if args.stem:
            geojson_path = args.geojson_dir / f"{args.stem}.geojson"
            manifest_path = args.cuts_dir / args.stem / f"{args.stem}_cuts.json"
            output_dir = args.cuts_dir / args.stem
            reports = [
                remap_annotations(
                    geojson_path,
                    manifest_path,
                    output_dir,
                    skip_if_exists=not args.no_skip,
                )
            ]
        else:
            reports = remap_annotations_batch(
                args.geojson_dir,
                args.cuts_dir,
                skip_if_exists=not args.no_skip,
            )
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1
    except ValueError as exc:
        logging.error("%s", exc)
        return 1

    _print_summary(reports)
    return 1 if any(report["n_unassigned"] for report in reports) else 0


def _print_summary(reports: list[dict]) -> None:
    print(f"{'Stem':28} {'Total':>7} {'Assigned':>9} {'Unassigned':>11} {'Cuts':>5}")
    total = assigned = unassigned = 0
    for report in reports:
        total += report["n_annotations_total"]
        assigned += report["n_assigned"]
        unassigned += report["n_unassigned"]
        print(
            f"{report['stem']:28} "
            f"{report['n_annotations_total']:7d} "
            f"{report['n_assigned']:9d} "
            f"{report['n_unassigned']:11d} "
            f"{len(report['cuts']):5d}"
        )
    print(f"{'TOTAL':28} {total:7d} {assigned:9d} {unassigned:11d} {'--':>5}")


if __name__ == "__main__":
    sys.exit(main())
