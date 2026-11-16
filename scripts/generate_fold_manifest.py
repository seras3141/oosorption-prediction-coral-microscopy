#!/usr/bin/env python3
"""Generate the specimen-grouped cross-validation fold manifest and its per-fold splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.specimen_groups import (
    DEFAULT_N_FOLDS,
    FOLD_MANIFEST_VERSION,
    assign_folds,
    build_fold_manifest,
    choose_inner_val,
    specimen_counts,
    straddling_specimens,
    write_fold_manifests,
)
from src.modeling.cut_exclusions import (
    DEFAULT_CUT_EXCLUSIONS,
    CutExclusions,
    cut_exclusion_fields,
    cut_exclusions_from_flag,
    cut_exclusions_recorded_in,
    describe_cut_exclusions,
)
from src.modeling.tile_index import DEFAULT_SPLIT_MANIFEST, load_tile_index

LOG_FORMAT = "[%(levelname)s] %(message)s"
DEFAULT_OUTPUT = Path("data/splits/fold_manifest.json")
DEFAULT_SPLIT_DIR = f"data/splits/{FOLD_MANIFEST_VERSION}"
DEFAULT_TILE_SIZES = (128, 256, 512, 1024)
LOG_LEVELS = ("DEBUG", "INFO", "WARNING")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split-manifest", default=DEFAULT_SPLIT_MANIFEST,
        help="Slide-level manifest the tile index is built against; only its slide list "
             "is used, and its hash is recorded as provenance. Default: %(default)s.",
    )
    parser.add_argument(
        "--cut-exclusions", default=DEFAULT_CUT_EXCLUSIONS,
        help="CSV of cuts to leave out or confirmed empty, recorded in the manifest; "
             "'none' indexes every cut. Default: %(default)s.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Fold manifest output path. Default: %(default)s.",
    )
    parser.add_argument(
        "--split-dir", default=DEFAULT_SPLIT_DIR,
        help="Repository-relative directory for the per-fold split manifests. "
             "Default: %(default)s.",
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument(
        "--tile-sizes", type=int, nargs="+", default=list(DEFAULT_TILE_SIZES),
        help="Sizes whose counts are recorded. 512 and 1024 are needed for balancing.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print the fold table without writing anything.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing manifests.",
    )
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    return parser.parse_args(argv)


def format_fold_table(fold_manifest: dict) -> str:
    """Format one summary line per fold."""
    sizes = list(fold_manifest["folds"][0]["tiles"])
    lines = []
    for fold in fold_manifest["folds"]:
        cells = "  ".join(
            f"{s}px {fold['tiles'][s]:>7,} {fold['positives'][s]:>6,} "
            + ("  n/a" if fold["positive_rate"][s] is None else f"{fold['positive_rate'][s]:.1%}")
            for s in sizes
        )
        lines.append(
            f"fold {fold['fold']}  {fold['n_slides']:>2} slides  {cells}\n"
            f"        {', '.join(fold['specimens'])}  (inner val: {fold['inner_val_specimen']})"
        )
    return "\n".join(lines)


def other_exclusions_at(
    output: Path,
    split_targets: list[Path],
    exclusions: CutExclusions,
    manifests: tuple[Path, ...] | None = None,
) -> str | None:
    """Why writing ``output`` and ``split_targets`` would orphan a fold manifest built with
    another list: one at ``output``, or one in ``manifests`` (default: every JSON file in
    ``output``'s folder and under ``data/splits``) whose split files would be overwritten."""
    output = output.resolve()
    targets = {target.resolve() for target in split_targets}
    if manifests is None:
        splits_root = REPO_ROOT / "data" / "splits"
        manifests = (*output.parent.glob("*.json"), *splits_root.rglob("*.json"))
    candidates = manifests
    for manifest_path in sorted({output, *(path.resolve() for path in candidates)}):
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "folds" not in manifest:
            continue
        recorded = cut_exclusions_recorded_in(manifest)
        if recorded == exclusions:
            continue
        uses = {(REPO_ROOT / fold["split_manifest_path"]).resolve() for fold in manifest["folds"]}
        if manifest_path == output or uses & targets:
            return (f"{manifest_path} was built with {describe_cut_exclusions(recorded)}; "
                    f"writing a fold manifest with {describe_cut_exclusions(exclusions)} over it "
                    "or its split files would orphan the runs made on it, even with --force. "
                    "Pass a new --output and --split-dir.")
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the fold manifest generation CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)

    split_path = Path(args.split_manifest)
    split_path = split_path if split_path.is_absolute() else REPO_ROOT / split_path
    slides = json.loads(split_path.read_text(encoding="utf-8"))["slides"]
    straddling = straddling_specimens({stem: entry["split"] for stem, entry in slides.items()})
    print(f"{len(straddling)} specimen(s) straddle splits in {args.split_manifest}:")
    for specimen, splits in straddling.items():
        print(f"  {specimen}: {' + '.join(splits)}")

    exclusions = cut_exclusions_from_flag(args.cut_exclusions)
    index = load_tile_index(split_manifest_path=split_path, tile_sizes=tuple(args.tile_sizes),
                            cut_exclusions=exclusions)
    counts = specimen_counts(index)
    assignment = assign_folds(counts, n_folds=args.n_folds)
    inner_val = choose_inner_val(counts, assignment)
    fold_manifest = build_fold_manifest(
        counts,
        assignment,
        inner_val,
        split_manifest_dir=args.split_dir.rstrip("/"),
        source_split_manifest_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
    )
    fold_manifest.update(cut_exclusion_fields(exclusions))
    print(f"\n{fold_manifest['n_specimens']} specimens, {fold_manifest['n_slides']} slides, "
          f"{fold_manifest['n_folds']} folds")
    print(format_fold_table(fold_manifest))

    if args.dry_run:
        return 0

    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    split_targets = [REPO_ROOT / f["split_manifest_path"] for f in fold_manifest["folds"]]
    conflict = other_exclusions_at(output, split_targets, exclusions)
    if conflict:
        logging.error(conflict)
        return 1
    targets = [output] + split_targets
    existing = [t for t in targets if t.exists()]
    if existing and not args.force:
        logging.error("%s already exist(s); pass --force to overwrite",
                      ", ".join(str(t) for t in existing))
        return 1

    for path in write_fold_manifests(fold_manifest, output, REPO_ROOT):
        logging.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
