#!/usr/bin/env python3
"""Prepare blinded tile-review sessions for collaborator oocyte checks."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.build_review_html import build_review_html
from src.data_preparation.sample_review_tiles import (
    DEFAULT_MIN_TISSUE_FRACTION,
    DEFAULT_NEGATIVE_BUFFER_FRACTION,
    DEFAULT_N_NEGATIVE_PER_SIZE,
    DEFAULT_N_POSITIVE_PER_SIZE,
    DEFAULT_SEED,
    DEFAULT_TILE_SIZES,
    sample_review_session,
)

LOG_FORMAT = "[%(levelname)s] %(message)s"
DEFAULT_CUTS_DIR = Path("data/cuts")
DEFAULT_OUTPUT_DIR = Path("data/review_sessions")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare a blinded tile-review session from cut TIFFs and annotations."
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
        help=f"Parent output directory for review sessions. Default: {DEFAULT_OUTPUT_DIR}/",
    )
    parser.add_argument(
        "--session-id",
        help="Optional explicit session identifier. Default: auto-generated date-based ID.",
    )
    parser.add_argument(
        "--tile-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_TILE_SIZES),
        help="One or more tile sizes in pixels. Default: 128 256 512 1024.",
    )
    parser.add_argument(
        "--n-per-size",
        type=int,
        default=DEFAULT_N_POSITIVE_PER_SIZE + DEFAULT_N_NEGATIVE_PER_SIZE,
        help="Total tiles per size. Must be even because sampling is balanced. Default: 10.",
    )
    parser.add_argument(
        "--min-tissue",
        type=float,
        default=DEFAULT_MIN_TISSUE_FRACTION,
        help=(
            "Minimum saturation-mask tissue fraction required to keep a tile. "
            f"Default: {DEFAULT_MIN_TISSUE_FRACTION:.2f}."
        ),
    )
    parser.add_argument(
        "--negative-buffer-fraction",
        type=float,
        default=DEFAULT_NEGATIVE_BUFFER_FRACTION,
        help=(
            "Reject negative tiles within this tile-size fraction of any annotation centroid. "
            f"Default: {DEFAULT_NEGATIVE_BUFFER_FRACTION:.2f}."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a ZIP archive next to the written session directory.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Build a self-contained review.html inside the session directory (no pip install needed).",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity. Default: INFO.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the review-session preparation CLI."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)

    if args.n_per_size <= 0 or args.n_per_size % 2 != 0:
        logging.error("--n-per-size must be a positive even integer")
        return 1

    n_positive = args.n_per_size // 2
    n_negative = args.n_per_size // 2

    try:
        manifest = sample_review_session(
            args.cuts_dir,
            args.output_dir,
            session_id=args.session_id,
            tile_sizes=tuple(args.tile_sizes),
            n_positive_per_size=n_positive,
            n_negative_per_size=n_negative,
            min_tissue_fraction=args.min_tissue,
            negative_buffer_fraction=args.negative_buffer_fraction,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    session_dir = _resolve_session_dir(args.output_dir, manifest["session_id"])
    zip_path = _zip_session_dir(session_dir) if args.zip else None
    html_path = build_review_html(session_dir) if args.html else None

    _print_summary(manifest, session_dir=session_dir, zip_path=zip_path, html_path=html_path)
    return 0


def _resolve_session_dir(output_dir: Path, session_id: str) -> Path:
    """Return the written session directory path."""
    output_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir
    return output_dir / session_id


def _zip_session_dir(session_dir: Path) -> Path:
    """Create a ZIP archive for one prepared review session."""
    archive_base = session_dir.parent / session_dir.name
    archive_path = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=session_dir.parent,
        base_dir=session_dir.name,
    )
    return Path(archive_path)


def _print_summary(
    manifest: dict, *, session_dir: Path, zip_path: Path | None, html_path: Path | None
) -> None:
    """Print a compact review-session summary table."""
    print(f"Session:  {manifest['session_id']}")
    print(f"Location: {_display_path(session_dir)}")
    print(
        f"Tiles:    {len(manifest['tiles'])}  "
        f"({manifest['n_tiles_per_size']} per size x {len(manifest['tile_sizes'])} sizes; "
        f"{manifest['n_positive_per_size']} positive + {manifest['n_negative_per_size']} negative each)"
    )
    if zip_path is not None:
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"ZIP:      {_display_path(zip_path)}  ({size_mb:.1f} MB)")
    if html_path is not None:
        size_mb = html_path.stat().st_size / (1024 * 1024)
        print(f"HTML:     {_display_path(html_path)}  ({size_mb:.1f} MB)")

    print()
    print(f"{'Size':>6} {'Positive':>9} {'Negative':>9} {'Sources':>8}")
    for tile_size in manifest["tile_sizes"]:
        size_tiles = [tile for tile in manifest["tiles"] if tile["tile_size"] == tile_size]
        positives = sum(1 for tile in size_tiles if tile["ground_truth"])
        negatives = sum(1 for tile in size_tiles if not tile["ground_truth"])
        sources = len({tile["cut_name"] for tile in size_tiles})
        print(f"{tile_size:6d} {positives:9d} {negatives:9d} {sources:8d}")

    print()
    if html_path is not None:
        print("Share review.html with the collaborator — no installation needed.")
        print("They open it in any browser, label tiles, and download session.json when done.")
    else:
        print("Share the session directory or ZIP with the collaborator.")
        print("Run the review app with:")
        print(f"  streamlit run app/review_tiles.py -- --session {_display_path(session_dir)}")
        print()
        print("Or generate a self-contained HTML file (no pip install for the reviewer):")
        print(f"  python app/build_review_html.py --session {_display_path(session_dir)}")


def _display_path(path: Path) -> str:
    """Return a stable display path inside or outside the repo root."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())