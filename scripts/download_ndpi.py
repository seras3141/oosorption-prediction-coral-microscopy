#!/usr/bin/env python3
"""Download NDPI slide files from PANGAEA for every GeoJSON in a dataset folder.

Usage
-----
    python scripts/download_ndpi.py                          # uses defaults
    python scripts/download_ndpi.py --geojson-dir data/dataset_28_04
    python scripts/download_ndpi.py --out-dir data/ndpi --dry-run

PANGAEA datasets
----------------
  CHN slides → https://doi.pangaea.de/10.1594/PANGAEA.984641
  LHP slides → https://doi.pangaea.de/10.1594/PANGAEA.984640
"""

from __future__ import annotations

import argparse
import errno
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# PANGAEA download base URLs (one per location prefix)
# ---------------------------------------------------------------------------
PANGAEA_BASES: dict[str, str] = {
    "CHN": "https://download.pangaea.de/dataset/984641/files/",
    "LHP": "https://download.pangaea.de/dataset/984640/files/",
}

DEFAULT_GEOJSON_DIR = Path(__file__).parent.parent / "data" / "dataset_28_04"
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "data" / "ndpi"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
USER_AGENT = "Mozilla/5.0 (compatible; coral-microscopy-downloader/0.1)"
MAX_RETRIES = 3
RETRY_DELAYS = (10, 30, 60)  # seconds between successive attempts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def geojson_stem_to_ndpi_name(stem: str) -> str:
    """Return the expected NDPI filename for a GeoJSON stem.

    E.g. ``CHN_AU_10_19-21`` → ``CHN_AU_10_19-21.ndpi``
    """
    return stem + ".ndpi"


def location_prefix(stem: str) -> str | None:
    """Return the location code (first underscore-separated token)."""
    parts = stem.split("_")
    return parts[0] if parts else None


def download_file(url: str, dest: Path, *, verbose: bool = True) -> bool:
    """Download *url* to *dest*.  Returns True on success.

    Retries up to MAX_RETRIES times on network errors (e.g. EHOSTUNREACH).
    HTTP errors (4xx/5xx) are not retried.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(f"{dest.name}.part")

    if verbose:
        print(f"  Downloading {url}")
        print(f"          → {dest}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request) as response, tmp_dest.open("wb") as out:
                total_size = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    _print_progress(downloaded, total_size, verbose=verbose)

            tmp_dest.replace(dest)
            if verbose:
                print()  # newline after progress bar
            return True

        except urllib.error.HTTPError as exc:
            if verbose:
                print(f"\n  ERROR {exc.code}: {exc.reason}  ({url})")
            # Retry on transient server errors (5xx, 429); give up on client errors (4xx)
            if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt - 1]
                if verbose:
                    print(f"  Retrying in {delay}s (attempt {attempt}/{MAX_RETRIES})...")
                if tmp_dest.exists():
                    tmp_dest.unlink()
                time.sleep(delay)
            else:
                break

        except (urllib.error.URLError, OSError) as exc:
            if verbose:
                print(f"\n  ERROR: {exc.reason if hasattr(exc, 'reason') else exc}  ({url})")
            # Disk quota / no space — fatal, abort immediately
            if isinstance(exc, OSError) and exc.errno in (errno.EDQUOT, errno.ENOSPC):
                raise
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt - 1]
                if verbose:
                    print(f"  Retrying in {delay}s (attempt {attempt}/{MAX_RETRIES})...")
                if tmp_dest.exists():
                    tmp_dest.unlink()
                time.sleep(delay)

        finally:
            if tmp_dest.exists():
                tmp_dest.unlink()

    return False


def _print_progress(downloaded: int, total_size: int, *, verbose: bool) -> None:
    """Print an in-place progress indicator for a file download."""
    if not verbose:
        return

    if total_size <= 0:
        dl_mb = downloaded / 1_048_576
        print(f"\r    {dl_mb:.1f} MB downloaded", end="", flush=True)
        return

    pct = min(100.0, downloaded / total_size * 100)
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    size_mb = total_size / 1_048_576
    dl_mb = min(downloaded, total_size) / 1_048_576
    print(
        f"\r    [{bar}] {pct:5.1f}%  {dl_mb:.1f}/{size_mb:.1f} MB",
        end="",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_download_list(
    geojson_dir: Path,
    out_dir: Path,
) -> list[tuple[str, Path]]:
    """Return list of (url, dest_path) pairs that still need to be fetched."""
    geojson_files = sorted(geojson_dir.glob("*.geojson"))
    if not geojson_files:
        print(f"No GeoJSON files found in {geojson_dir}", file=sys.stderr)
        return []

    download_list: list[tuple[str, Path]] = []

    for gj in geojson_files:
        stem = gj.stem  # e.g. CHN_AU_10_19-21
        ndpi_name = geojson_stem_to_ndpi_name(stem)
        dest = out_dir / ndpi_name
        loc = location_prefix(stem)

        if loc not in PANGAEA_BASES:
            print(f"  [SKIP] Unknown location prefix for {gj.name!r}")
            continue

        base_url = PANGAEA_BASES[loc]
        url = base_url + ndpi_name

        if dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  [OK]   {ndpi_name}  (already present, {size_mb:.1f} MB)")
        else:
            download_list.append((url, dest))

    return download_list


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download NDPI files from PANGAEA for a GeoJSON dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--geojson-dir",
        type=Path,
        default=DEFAULT_GEOJSON_DIR,
        help="Directory containing the GeoJSON annotation files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where NDPI files will be saved.",
    )
    parser.add_argument(
        "--inter-file-delay",
        type=int,
        default=90,
        metavar="SECONDS",
        help="Seconds to wait between successive downloads (avoids server rate-limiting).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    args = parser.parse_args(argv)

    geojson_dir: Path = args.geojson_dir
    out_dir: Path = args.out_dir

    if not geojson_dir.exists():
        print(f"ERROR: GeoJSON directory not found: {geojson_dir}", file=sys.stderr)
        return 1

    print(f"GeoJSON dir : {geojson_dir.resolve()}")
    print(f"Output dir  : {out_dir.resolve()}")
    print()

    # Build the list of files to download
    todo = build_download_list(geojson_dir, out_dir)

    if not todo:
        print("\nAll NDPI files already present — nothing to download.")
        return 0

    print(f"\n{len(todo)} file(s) to download:")
    for url, dest in todo:
        print(f"  {dest.name:40s}  {url}")

    if args.dry_run:
        print("\n[dry-run] Exiting without downloading.")
        return 0

    print()
    failed: list[str] = []
    for i, (url, dest) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {dest.name}")
        try:
            ok = download_file(url, dest)
        except OSError as exc:
            print(f"\nFATAL: {exc} — aborting.", file=sys.stderr)
            print("Free up disk space or request a higher quota, then re-run.", file=sys.stderr)
            return 1
        if not ok:
            failed.append(dest.name)
        elif i < len(todo):
            print(f"  Waiting {args.inter_file_delay}s before next download...")
            time.sleep(args.inter_file_delay)

    print()
    if failed:
        print(f"FAILED ({len(failed)}/{len(todo)}):")
        for name in failed:
            print(f"  {name}")
        return 1

    print(f"Done. {len(todo)} file(s) downloaded to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
