#!/usr/bin/env python3
"""Generate the deterministic slide-level train/val/test split manifest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.generate_split_manifest import (
    DEFAULT_SEED,
    build_split_manifest,
    format_split_summary,
    write_split_manifest,
)

LOG_FORMAT = "[%(levelname)s] %(message)s"
DEFAULT_GEOJSON_DIR = Path("data/dataset_28_04")
DEFAULT_OUTPUT = Path("data/splits/split_manifest.json")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the deterministic slide-level train/val/test split manifest."
    )
    parser.add_argument(
        "--geojson-dir",
        type=Path,
        default=DEFAULT_GEOJSON_DIR,
        help=f"Source GeoJSON directory. Default: {DEFAULT_GEOJSON_DIR}/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Manifest output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for split assignment. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the summary table without writing the manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --output if it already exists.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity. Default: INFO.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the split manifest generation CLI."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)

    manifest = build_split_manifest(args.geojson_dir, seed=args.seed)
    print(format_split_summary(manifest))

    if args.dry_run:
        return 0

    if args.output.exists() and not args.force:
        logging.error("%s already exists; pass --force to overwrite", args.output)
        return 1

    write_split_manifest(manifest, args.output)
    logging.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
