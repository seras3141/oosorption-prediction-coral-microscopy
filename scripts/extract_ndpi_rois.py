#!/usr/bin/env python3
"""Extract tissue cuts from NDPI whole-slide images and write a _cuts.json manifest.

Delegates to src.data_preparation.extract_ndpi_cuts.extract_specimen_cuts, which
writes pyramidal TIFFs named ``{stem}_cut{i:03d}.tif`` (0-based) and a
``{stem}_cuts.json`` manifest consumed by coral-remap-annotations and
coral-generate-tiles.

Usage:
    uv run coral-extract-rois --input /path/to/CHN_W_2_1-3.ndpi --outdir data/cuts
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.extract_ndpi_cuts import extract_specimen_cuts

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def parse_args():
    p = argparse.ArgumentParser(description='Extract tissue cuts from NDPI whole-slide images')
    p.add_argument('--input', '-i', required=True, help='Path to NDPI file or directory of .ndpi files')
    p.add_argument('--outdir', '-o', required=True, help='Output directory to write cut TIFFs and manifests')
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    outdir = Path(args.outdir)

    if inp.is_file():
        targets = [inp]
    elif inp.is_dir():
        targets = sorted(inp.glob("*.ndpi"))
    else:
        raise FileNotFoundError(f"Input path not found: {inp}")

    if not targets:
        logging.warning("No NDPI files found under %s", inp)
        return

    for ndpi_path in targets:
        logging.info("Processing %s", ndpi_path.name)
        extract_specimen_cuts(
            ndpi_path,
            outdir / ndpi_path.stem,
            output_format="pyramidal",
            write_manifest=True,
        )


if __name__ == '__main__':
    main()
