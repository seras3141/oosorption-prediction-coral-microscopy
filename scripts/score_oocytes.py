#!/usr/bin/env python3
"""Score CV runs per oocyte from their held-out predictions, or their noise floor."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.cut_exclusions import DEFAULT_CUT_EXCLUSIONS, cut_exclusions_from_flag
from src.modeling.oocyte_scoring import (
    DEFAULT_OUTPUT_ROOT,
    noise_floor,
    output_dir_for_name,
    score_held_out_predictions,
    write_noise_floor,
    write_oocyte_scores,
)
from src.modeling.tile_index import _resolve_repo_path

LOG_FORMAT = "[%(levelname)s] %(message)s"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING")

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--held-out-predictions", nargs="+", metavar="DIR",
                      help="coral-predict-held-out-tiles output directories (predictions.csv "
                           "and provenance.json), one per CV run, to score.")
    mode.add_argument("--noise-floor", nargs="+", metavar="SUMMARY",
                      help="summary.json files of one tile size to compare across seeds.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Each output goes to <output-root>/<name>/. Default: %(default)s.")
    parser.add_argument("--cut-exclusions", default=None,
                        help="Cuts it lists as excluded are left out of scoring; scoring mode "
                             f"only. Default: {DEFAULT_CUT_EXCLUSIONS}.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing outputs instead of refusing.")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    args = parser.parse_args(argv)
    if args.noise_floor and args.cut_exclusions is not None:
        parser.error("--cut-exclusions applies to scoring, not to --noise-floor")
    if args.cut_exclusions == "none":
        parser.error("scoring needs an exclusion list; 'none' would score unannotated cuts")
    if args.cut_exclusions is None:
        args.cut_exclusions = DEFAULT_CUT_EXCLUSIONS
    return args


def check_outputs(held_out_dirs: list[str], output_root: str, overwrite: bool) -> None:
    """Refuse duplicate and, without overwrite, existing outputs before any scoring."""
    out_dirs = []
    for held_out in held_out_dirs:
        provenance = json.loads((_resolve_repo_path(held_out) / "provenance.json")
                                .read_text(encoding="utf-8"))
        out_dirs.append((held_out, output_dir_for_name(output_root, provenance["cv_run_id"])))
    repeated = sorted(str(d) for d, n in Counter(d for _, d in out_dirs).items() if n > 1)
    if repeated:
        raise ValueError(f"more than one input writes to {', '.join(repeated)}")
    existing = [held_out for held_out, out_dir in out_dirs if out_dir.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"outputs already exist for {', '.join(existing)}; pass "
                              "--overwrite to replace them, or leave those runs out")


def main(argv: list[str] | None = None) -> int:
    """Score each run in turn, or write one noise floor, stopping at the first refusal."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)
    if args.noise_floor:
        summaries = [json.loads(_resolve_repo_path(path).read_text(encoding="utf-8"))
                     for path in args.noise_floor]
        out_dir = write_noise_floor(noise_floor(summaries), args.output_root, args.overwrite)
        logger.info("Wrote %s", out_dir)
        return 0
    exclusions = cut_exclusions_from_flag(args.cut_exclusions)
    check_outputs(args.held_out_predictions, args.output_root, args.overwrite)
    for held_out in args.held_out_predictions:
        scored = score_held_out_predictions(held_out, exclusions)
        out_dir = write_oocyte_scores(scored, args.output_root, args.overwrite)
        logger.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
