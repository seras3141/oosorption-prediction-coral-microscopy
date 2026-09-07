"""Cross-check geometry-derived tile labels against pinned corpus counts.

The label computation replaces the tile manifests' ``has_oocyte`` flag, so a silent
change in it would quietly alter every downstream training label. This script recomputes
the positive counts over all 77 cuts and compares them against values pinned from the
corpus, at six coverage thresholds for both trained tile sizes and all three splits.

Needs the tiles and cuts under ``data/``, which is not synced to every checkout, so this
is a script rather than part of the pytest suite.

Usage
-----
    uv run coral-verify-tile-labels
    uv run coral-verify-tile-labels --tile-sizes 512
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.tile_labels import (
    _resolve_repo_path,
    oocyte_area_fractions_for_manifest,
)

# Pinned 2026-09-04 from the full 26-slide corpus, using make_valid to repair the 57% of
# annotation rings that self-intersect. "centroid" is the manifest flag, kept alongside
# to document how far the corrected label diverges from it.
EXPECTED: dict[tuple[int, str], dict[str, int]] = {
    (512, "train"): {"n": 32521, "centroid": 1160, "gt0": 5607, "ge05": 5063,
                     "ge10": 4786, "ge25": 4047, "ge50": 2944},
    (512, "val"): {"n": 7723, "centroid": 369, "gt0": 1389, "ge05": 1263,
                   "ge10": 1190, "ge25": 1006, "ge50": 696},
    (512, "test"): {"n": 3949, "centroid": 124, "gt0": 515, "ge05": 439,
                    "ge10": 394, "ge25": 292, "ge50": 197},
    (1024, "train"): {"n": 8066, "centroid": 908, "gt0": 1896, "ge05": 1634,
                      "ge10": 1487, "ge25": 1108, "ge50": 618},
    (1024, "val"): {"n": 1932, "centroid": 269, "gt0": 491, "ge05": 427,
                    "ge10": 379, "ge25": 276, "ge50": 148},
    (1024, "test"): {"n": 879, "centroid": 101, "gt0": 175, "ge05": 135,
                     "ge10": 113, "ge25": 70, "ge50": 36},
}
THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("gt0", 0.0), ("ge05", 0.05), ("ge10", 0.10), ("ge25", 0.25), ("ge50", 0.50),
)
FIELDS = ("n", "centroid", "gt0", "ge05", "ge10", "ge25", "ge50")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tiles-dir", default="data/tiles", help="Root of the per-cut tile manifests."
    )
    parser.add_argument(
        "--cuts-dir", default="data/cuts", help="Root of the cut-local annotation GeoJSONs."
    )
    parser.add_argument(
        "--split-manifest", default="data/splits/split_manifest.json",
        help="Slide-to-split manifest.",
    )
    parser.add_argument(
        "--tile-sizes", type=int, nargs="+", default=[512, 1024],
        help="Tile sizes to check. Default: 512 1024.",
    )
    return parser.parse_args(argv)


def _count(args: argparse.Namespace) -> dict[tuple[int, str], collections.Counter]:
    # Anchor every path the same way tile_labels does, so the script and the module can
    # never end up reading two different copies of the corpus.
    tiles_root = _resolve_repo_path(args.tiles_dir)
    cuts_root = _resolve_repo_path(args.cuts_dir)
    split_path = _resolve_repo_path(args.split_manifest)

    split = json.loads(split_path.read_text(encoding="utf-8"))["slides"]
    counts: dict[tuple[int, str], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    manifests = sorted(glob.glob(str(tiles_root / "*" / "*" / "*_tile_manifest.json")))
    if not manifests:
        raise SystemExit(f"No tile manifests found under {tiles_root}")

    for manifest_path in manifests:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        stem, cut = manifest["stem"], manifest["cut_name"]
        # Hand over the manifest already parsed here: these files total ~636 MB and
        # re-reading them is the dominant cost of this pass.
        fractions = oocyte_area_fractions_for_manifest(
            manifest,
            cuts_root / stem / f"{cut}_annotations.geojson",
            tile_sizes=tuple(args.tile_sizes),
            manifest_path=Path(manifest_path),
        )
        if stem not in split:
            raise SystemExit(
                f"{stem} has tiles under {tiles_root} but no entry in {split_path}; "
                "the split manifest and the tile tree disagree."
            )
        assigned = split[stem]["split"]
        for tile in manifest["tiles"]:
            size = tile["tile_size"]
            if size not in args.tile_sizes:
                continue
            counter = counts[(size, assigned)]
            fraction = fractions[tile["tile_id"]]
            counter["n"] += 1
            counter["centroid"] += bool(tile["has_oocyte"])
            for name, threshold in THRESHOLDS:
                hit = fraction > threshold if threshold == 0.0 else fraction >= threshold
                if hit:
                    counter[name] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    counts = _count(args)

    checked = mismatched = 0
    for key in sorted(EXPECTED, key=lambda pair: (pair[0], pair[1])):
        if key[0] not in args.tile_sizes:
            continue
        for field in FIELDS:
            checked += 1
            expected, got = EXPECTED[key][field], counts[key][field]
            if expected != got:
                mismatched += 1
                print(
                    f"MISMATCH  {key[0]:>4} px  {key[1]:<5}  {field:<8} "
                    f"expected {expected}, got {got}"
                )

    if not checked:
        # "All 0 pinned cells match" from the one tool meant to catch silent label
        # drift would be worse than no tool at all.
        print(
            f"No pinned cells cover tile sizes {args.tile_sizes}; "
            f"pinned sizes are {sorted({size for size, _ in EXPECTED})}."
        )
        return 2
    if mismatched:
        print(f"\n{mismatched} of {checked} cells differ from the pinned corpus counts.")
        return 1
    print(f"All {checked} pinned cells match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
