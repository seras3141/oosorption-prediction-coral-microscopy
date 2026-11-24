#!/usr/bin/env python3
"""Predict every held-out tile of finished CV runs, ambiguous tiles included."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.held_out_predictions import (
    DEFAULT_OUTPUT_ROOT,
    output_dir_for_results,
    run_held_out_predictions,
)

LOG_FORMAT = "[%(levelname)s] %(message)s"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING")

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cv-results", nargs="+", required=True,
                        help="One or more cv_results.json files.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Each run is written to <output-root>/<cv_run_id>/. "
                             "Default: %(default)s.")
    parser.add_argument("--device", default="cuda",
                        help="Torch device. Use the kind the runs were evaluated on. "
                             "Default: %(default)s.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers. Default: %(default)s.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing outputs instead of refusing.")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error(f"--num-workers must be at least 0, got {args.num_workers}")
    return args


def check_outputs(cv_results: list[str], output_root: str, overwrite: bool) -> None:
    """Refuse duplicate and, without overwrite, existing outputs before any run starts."""
    out_dirs = [(path, output_dir_for_results(path, output_root)) for path in cv_results]
    repeated = sorted(str(d) for d, n in Counter(d for _, d in out_dirs).items() if n > 1)
    if repeated:
        raise ValueError(f"more than one --cv-results writes to {', '.join(repeated)}")
    existing = [path for path, out_dir in out_dirs if out_dir.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"outputs already exist for {', '.join(existing)}; pass --overwrite to replace "
            "them, or leave those runs out"
        )


def main(argv: list[str] | None = None) -> int:
    """Predict and write each CV run in turn, stopping at the first refusal."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)
    check_outputs(args.cv_results, args.output_root, args.overwrite)
    for cv_results in args.cv_results:
        out_dir = run_held_out_predictions(cv_results, output_root=args.output_root,
                                           device=args.device, num_workers=args.num_workers,
                                           overwrite=args.overwrite)
        logger.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
