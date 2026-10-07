#!/usr/bin/env python3
"""Build frozen-encoder embedding caches and check them against a live encoder pass."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.embedding_cache import (
    DEFAULT_CACHE_ROOT,
    build_cache,
    cache_dir,
    compare_with_live,
    load_cache,
)
from src.modeling.encoders import DEFAULT_ENCODER, FROZEN_TILE_SIZES, SUPPORTED_ENCODERS

VERIFICATION_FILE = "live_check.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER, choices=SUPPORTED_ENCODERS)
    parser.add_argument(
        "--tile-sizes", type=int, nargs="+", default=list(FROZEN_TILE_SIZES),
        choices=FROZEN_TILE_SIZES,
    )
    parser.add_argument("--root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--verify-only", action="store_true", help="Check existing caches without rebuilding.",
    )
    parser.add_argument("--verify-tiles", type=int, default=256)
    parser.add_argument(
        "--tolerance", type=float, default=1e-3,
        help="Largest live-vs-cached difference allowed, relative to the mean |embedding|.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    failed = []
    for tile_size in args.tile_sizes:
        path = cache_dir(args.encoder_name, tile_size, args.root)
        if not args.verify_only:
            build_cache(args.encoder_name, tile_size, args.root,
                        batch_size=args.batch_size, num_workers=args.num_workers)
        cache = load_cache(path, tile_size, args.encoder_name)
        check = compare_with_live(cache, n_tiles=args.verify_tiles)
        check["tolerance"] = args.tolerance
        check["passed"] = check["max_abs_diff_over_mean_magnitude"] <= args.tolerance
        (path / VERIFICATION_FILE).write_text(json.dumps(check, indent=2) + "\n")
        print(f"{tile_size:>4} px  {cache.meta['n_tiles']:,} tiles  "
              f"max |live - cached| / mean|x| = {check['max_abs_diff_over_mean_magnitude']:.2e}  "
              f"{'ok' if check['passed'] else 'FAILED'}")
        if not check["passed"]:
            failed.append(tile_size)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
