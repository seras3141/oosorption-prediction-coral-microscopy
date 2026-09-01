#!/usr/bin/env python3
"""Generate the deterministic slide-level train/val/test split manifest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.generate_split_manifest import (
    DEFAULT_SEED,
    VALID_STAGES,
    build_split_manifest,
    write_split_manifest,
)

LOG_FORMAT = "[%(levelname)s] %(message)s"
DEFAULT_GEOJSON_DIR = Path("data/dataset_28_04")
DEFAULT_OUTPUT = Path("data/splits/split_manifest.json")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING")
SPLITS = ("train", "val", "test")
STAGES = tuple(sorted(VALID_STAGES))


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


def _print_summary(manifest: dict[str, Any]) -> None:
    header = (
        f"{'split':<7} {'n_slides':>8} {'CHN':>5} {'LHP':>5} "
        + " ".join(f"{'stage' + str(s):>7}" for s in STAGES)
    )
    print(header)
    for split in SPLITS:
        slides = [s for s in manifest["slides"].values() if s["split"] == split]
        n_slides = len(slides)
        n_chn = sum(1 for s in slides if s["location"] == "CHN")
        n_lhp = sum(1 for s in slides if s["location"] == "LHP")
        stage_totals = {stage: 0 for stage in STAGES}
        for s in slides:
            for stage, count in s["stage_counts"].items():
                # stage_counts keys are int in-memory, str once JSON-round-tripped.
                stage_totals[int(stage)] += count
        row = (
            f"{split:<7} {n_slides:>8} {n_chn:>5} {n_lhp:>5} "
            + " ".join(f"{stage_totals[s]:>7}" for s in STAGES)
        )
        print(row)


def main() -> int:
    """Run the split manifest generation CLI."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)

    manifest = build_split_manifest(args.geojson_dir, seed=args.seed)
    _print_summary(manifest)

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
