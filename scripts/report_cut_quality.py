#!/usr/bin/env python3
"""Report each cut's image quality and flag the cuts that look out of focus."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.cut_exclusions import (
    DEFAULT_CUT_EXCLUSIONS,
    EXCLUDED,
    LEGACY_CORPUS,
    CutExclusions,
    cut_exclusions_from_flag,
)
from src.modeling.tile_index import (
    DEFAULT_SPLIT_MANIFEST,
    LABEL_RULE_CENTROID,
    _resolve_repo_path,
    load_tile_index,
)
from src.qc.image_quality import (
    BLURRED,
    DEFAULT_CUTOFF_FRACTION,
    DEFAULT_MIN_SHARPNESS_TILES,
    QUALITY_SEED,
    QUALITY_TILES_PER_CUT,
    cut_image_quality,
    flag_cuts,
)

LOG_FORMAT = "[%(levelname)s] %(message)s"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING")
QUALITY_TILE_SIZE = 512
DEFAULT_OUTPUT_DIR = Path("data/qc/cut_quality")
MARKDOWN_COLUMNS = {
    "cut_name": "{}",
    "sharpness": "{:.5f}",
    "sharpness_vs_median": "{:.2f}",
    "sharpness_percentile": "{:.0f}",
    "tiles_measured": "{}",
    "sharpness_tiles": "{}",
    "flag": "{}",
    "listed": "{}",
}

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tile-size", type=int, default=QUALITY_TILE_SIZE,
                        help="Tile size measured. Default: %(default)s.")
    parser.add_argument("--per-cut", type=int, default=QUALITY_TILES_PER_CUT,
                        help="Tiles sampled per cut. Default: %(default)s.")
    parser.add_argument("--seed", type=int, default=QUALITY_SEED,
                        help="Seed of the per-cut tile sample. Default: %(default)s.")
    parser.add_argument(
        "--cutoff-fraction", type=float, default=DEFAULT_CUTOFF_FRACTION,
        help="A cut is flagged blurred below this fraction of the median sharpness. "
             "Default: %(default)s.",
    )
    parser.add_argument(
        "--min-sharpness-tiles", type=int, default=DEFAULT_MIN_SHARPNESS_TILES,
        help="Tiles that must give a sharpness for a cut to be judged; fewer flags it "
             "for a look. Default: %(default)s.",
    )
    parser.add_argument(
        "--cut-exclusions", default=DEFAULT_CUT_EXCLUSIONS,
        help="CSV whose listed cuts are marked in the report; 'none' marks none. Listed "
             "cuts are still measured. Default: %(default)s.",
    )
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST,
                        help="Slide manifest the tile index is built against. "
                             "Default: %(default)s.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Where cut_quality.csv and cut_quality.md go. Default: %(default)s.")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    args = parser.parse_args(argv)
    if args.min_sharpness_tiles < 1:
        parser.error(f"--min-sharpness-tiles must be at least 1, got {args.min_sharpness_tiles}")
    if not 0 < args.cutoff_fraction < 1:
        parser.error(f"--cutoff-fraction must lie between 0 and 1, got {args.cutoff_fraction}")
    if args.per_cut < args.min_sharpness_tiles:
        parser.error(f"--per-cut {args.per_cut} samples fewer tiles than --min-sharpness-tiles "
                     f"{args.min_sharpness_tiles}, so no cut could be judged")
    return args


def markdown_report(report: pd.DataFrame, args: argparse.Namespace,
                    exclusions: CutExclusions) -> str:
    """The report as a Markdown table under a header naming its parameters."""
    listed = (f"`listed` is its status in {args.cut_exclusions}" if exclusions.applied
              else "`listed` is empty: no exclusion list was given")
    lines = [
        "# Cut image quality",
        "",
        f"Up to {args.per_cut} tiles of {args.tile_size} px per cut, seed {args.seed}. A cut "
        f"is `blurred` below {args.cutoff_fraction} × the median sharpness of the cuts not "
        f"excluded with at least {args.min_sharpness_tiles} tiles that give one, and `too few "
        f"tiles` under that count. {listed}. Percentiles rank all measured cuts. Least sharp first.",
        "",
        "| " + " | ".join(MARKDOWN_COLUMNS) + " |",
        "|" + "---|" * len(MARKDOWN_COLUMNS),
    ]
    for row in report.itertuples(index=False):
        cells = [_cell(getattr(row, column), template)
                 for column, template in MARKDOWN_COLUMNS.items()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _cell(value: object, template: str) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "–"
    return template.format(value)


def main(argv: list[str] | None = None) -> int:
    """Measure every tiled cut, flag the outliers and write the report."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)
    exclusions = cut_exclusions_from_flag(args.cut_exclusions)
    # Legacy corpus keeps listed cuts measured.
    tiles = load_tile_index(split_manifest_path=args.split_manifest,
                            tile_sizes=(args.tile_size,), label_rule=LABEL_RULE_CENTROID,
                            cut_exclusions=LEGACY_CORPUS)
    quality = cut_image_quality(tiles, per_cut=args.per_cut, seed=args.seed)
    report = flag_cuts(quality, exclusions, args.cutoff_fraction, args.min_sharpness_tiles)

    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "cut_quality.csv", index=False)
    (output_dir / "cut_quality.md").write_text(markdown_report(report, args, exclusions), encoding="utf-8")
    for row in report[(report["flag"] != "") & (report["listed"] != EXCLUDED)].itertuples():
        if row.flag == BLURRED:
            logger.warning("%s looks out of focus (%.2f x the median sharpness) and is not "
                           "excluded", row.cut_name, row.sharpness_vs_median)
        else:
            logger.warning("%s has %d tiles that give a sharpness, too few to judge; look at it",
                           row.cut_name, row.sharpness_tiles)
    unmatched = sorted(set(exclusions.statuses) - set(report["cut_name"]))
    if unmatched:
        logger.warning("%s listed in the cut exclusions but no %d px tile belongs to it; a typo?",
                       ", ".join(unmatched), args.tile_size)
    logger.info("Wrote the report on %d cuts (%d flagged) to %s", len(report),
                int((report["flag"] != "").sum()), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
