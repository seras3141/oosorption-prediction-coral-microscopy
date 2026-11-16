"""Build the tile-level training index: one row per tile, labelled from its cut's
annotations and joined to its slide's split, so no oocyte crosses splits. Free of
``torch``, so detection work can reuse it without the ML stack."""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.modeling.cut_exclusions import DEFAULT_CUT_EXCLUSIONS, LEGACY_CORPUS, CutExclusions
from src.modeling.tile_labels import (
    DEFAULT_MIN_OOCYTE_AREA_FRACTION,
    LABEL_AMBIGUOUS,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    label_for_area_fraction,
    load_annotation_polygons,
    oocyte_area_fractions_for_manifest,
)

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TILES_DIR = "data/tiles"
DEFAULT_CUTS_DIR = "data/cuts"
DEFAULT_SPLIT_MANIFEST = "data/splits/split_manifest.json"
DEFAULT_TILE_SIZES: tuple[int, ...] = (512, 1024)

LABEL_RULE_AREA = "area"
LABEL_RULE_CENTROID = "centroid"
LABEL_RULES = (LABEL_RULE_AREA, LABEL_RULE_CENTROID)

INDEX_COLUMNS = (
    "tile_id",
    "png_path",
    "stem",
    "cut_name",
    "tile_size",
    "n_oocytes",
    "tissue_fraction",
    "oocyte_area_fraction",
    "label",
    "is_ambiguous",
    "split",
)


def load_tile_index(
    tiles_dir: str | Path = DEFAULT_TILES_DIR,
    cuts_dir: str | Path = DEFAULT_CUTS_DIR,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    tile_sizes: tuple[int, ...] = DEFAULT_TILE_SIZES,
    label_rule: str = LABEL_RULE_AREA,
    min_oocyte_area_fraction: float = DEFAULT_MIN_OOCYTE_AREA_FRACTION,
    allow_missing_slides: bool = False,
    cut_exclusions: CutExclusions = LEGACY_CORPUS,
) -> pd.DataFrame:
    """Build one indexed row per tile at the requested sizes.

    Parameters
    ----------
    tiles_dir : str or Path, optional
        Root holding ``{stem}/{cut}/{cut}_tile_manifest.json``.
    cuts_dir : str or Path, optional
        Root holding ``{stem}/{cut}_annotations.geojson``.
    split_manifest_path : str or Path, optional
        Slide-to-split manifest.
    tile_sizes : tuple of int, optional
        Tile sizes to include. Sizes absent from a manifest are simply not present.
    label_rule : str, optional
        ``"area"`` uses the geometry-derived coverage label; ``"centroid"`` reproduces
        the manifest's ``has_oocyte`` flag, for comparison only. The centroid rule never
        yields an ambiguous row.
    min_oocyte_area_fraction : float, optional
        Coverage at or above which an area-rule tile is positive. Ignored by the
        centroid rule.
    allow_missing_slides : bool, optional
        Permit split-manifest slides that have no tiles, for a deliberate subset run;
        otherwise a partially copied tile tree would silently shrink the splits.
    cut_exclusions : CutExclusions, optional
        Cuts to leave out and cuts confirmed empty. Default:
        ``LEGACY_CORPUS``, which indexes every cut, as before the list existed.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`INDEX_COLUMNS`. ``label`` is ``"positive"``, ``"ambiguous"`` or
        ``"negative"``; ``is_ambiguous`` repeats it as the flag callers filter on.

    Raises
    ------
    ValueError
        If ``label_rule`` is unknown, no manifests or no tiles at ``tile_sizes`` are
        found, a stem under ``tiles_dir`` is absent from the split manifest, or, unless
        ``allow_missing_slides``, a split-manifest slide has no tiles. With an applied
        exclusion list, also if a cut has no annotation file, an unlisted cut has no
        annotations, a known-empty cut has annotations, the list excludes every cut with
        tiles, or, unless ``allow_missing_slides``, a listed cut is not under
        ``tiles_dir`` or every cut of a split-manifest slide is excluded.

    Examples
    --------
    >>> INDEX_COLUMNS[0], INDEX_COLUMNS[-1]
    ('tile_id', 'split')
    """
    if label_rule not in LABEL_RULES:
        raise ValueError(f"label_rule must be one of {LABEL_RULES}, got {label_rule!r}")

    tiles_root = _resolve_repo_path(tiles_dir)
    cuts_root = _resolve_repo_path(cuts_dir)
    slides = _load_split_assignments(split_manifest_path)

    manifest_paths = sorted(glob.glob(str(tiles_root / "*" / "*" / "*_tile_manifest.json")))
    if not manifest_paths:
        raise ValueError(f"No tile manifests found under {tiles_root}")

    wanted = set(tile_sizes)
    seen_stems: set[str] = set()
    manifest_cuts: dict[str, set[str]] = {}
    sized_cuts: dict[str, set[str]] = {}
    left_out: list[str] = []
    rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = _read_json(Path(manifest_path))
        stem, cut_name = manifest["stem"], manifest["cut_name"]
        manifest_cuts.setdefault(stem, set()).add(cut_name)
        has_wanted = any(tile["tile_size"] in wanted for tile in manifest.get("tiles", []))
        if has_wanted:
            sized_cuts.setdefault(stem, set()).add(cut_name)
        if cut_name in cut_exclusions.excluded:
            LOG.info("Leaving out %s: %s", cut_name,
                     cut_exclusions.reasons.get(cut_name) or "listed in the cut exclusions")
            if has_wanted:
                left_out.append(cut_name)
            continue
        if stem not in slides:
            raise ValueError(
                f"{stem} has tiles under {tiles_root} but no entry in the split "
                f"manifest; refusing to guess its split"
            )
        assigned = slides[stem]
        geojson_path = cuts_root / stem / f"{cut_name}_annotations.geojson"
        checked = cut_exclusions.applied and has_wanted
        if checked and not geojson_path.exists():
            raise ValueError(
                f"{cut_name} has no annotation file ({geojson_path}); list it as excluded "
                f"in the cut exclusion list this index uses (the versioned list is "
                f"{DEFAULT_CUT_EXCLUSIONS})."
            )
        polygons = load_annotation_polygons(geojson_path) if checked else None
        if polygons and cut_name in cut_exclusions.known_empty:
            raise ValueError(
                f"{cut_name} is listed as known_empty but has {len(polygons)} annotated "
                "oocyte(s); remove it from the cut exclusion list"
            )
        if polygons == [] and cut_name not in cut_exclusions.known_empty:
            raise ValueError(
                f"{cut_name} has no annotations. A cut the annotators skipped holds "
                "oocytes nobody labelled, so it cannot be indexed as all negative: list it "
                "as excluded in the cut exclusion list this index uses (the versioned list "
                f"is {DEFAULT_CUT_EXCLUSIONS}), or as known_empty if it truly holds no "
                "oocytes. A saved run indexes with the list it recorded."
            )

        fractions: dict[str, float] = {}
        if label_rule == LABEL_RULE_AREA:
            # Reuse the parsed manifest; it is huge.
            fractions = oocyte_area_fractions_for_manifest(
                manifest,
                geojson_path,
                tile_sizes=tuple(wanted),
                manifest_path=Path(manifest_path),
            )

        for tile in manifest.get("tiles", []):
            if tile["tile_size"] not in wanted:
                continue
            # Per tile: tileless slides must trip guard.
            seen_stems.add(stem)
            fraction = fractions.get(tile["tile_id"]) if fractions else None
            label = _label_for_tile(tile, fraction, label_rule, min_oocyte_area_fraction)
            rows.append(
                {
                    "tile_id": tile["tile_id"],
                    "png_path": tile["png_path"],
                    "stem": stem,
                    "cut_name": cut_name,
                    "tile_size": tile["tile_size"],
                    "n_oocytes": tile["n_oocytes"],
                    "tissue_fraction": tile["tissue_fraction"],
                    "oocyte_area_fraction": fraction,
                    "label": label,
                    "is_ambiguous": label == LABEL_AMBIGUOUS,
                    "split": assigned,
                }
            )

    if not rows and left_out:
        raise ValueError(
            f"No tiles left at sizes {sorted(wanted)} under {tiles_root}: the cut exclusion "
            f"list left out every cut that has them ({', '.join(sorted(left_out))})"
        )
    if not rows:
        raise ValueError(
            f"No tiles at sizes {sorted(wanted)} in any manifest under {tiles_root}; "
            "the manifests in this corpus carry sizes 128, 256, 512 and 1024. A size "
            "that matches nothing would otherwise return an empty index and only fail "
            "later, where the cause is no longer visible."
        )

    listed = cut_exclusions.excluded | cut_exclusions.known_empty
    seen_cuts = set().union(*manifest_cuts.values())
    for cut_name in sorted(listed - seen_cuts):
        stem = cut_name.rsplit("_cut", 1)[0]
        if stem not in slides:
            raise ValueError(f"{cut_name} is listed in the cut exclusions, but the split "
                             f"manifest has no slide {stem}: a typo")
        message = (f"{cut_name} is listed in the cut exclusions but is not under "
                   f"{tiles_root}: a typo, or a partially copied tile tree")
        if not allow_missing_slides:
            raise ValueError(f"{message}; pass allow_missing_slides=True for a deliberate "
                             "subset run")
        LOG.warning(message)

    missing = sorted(set(slides) - seen_stems)
    # A missing slide's sized cuts were excluded.
    excluded_slides = [stem for stem in missing if stem in sized_cuts]
    if missing and not allow_missing_slides:
        raise ValueError(
            f"{len(missing)} slide(s) in the split manifest contribute no tiles at "
            f"sizes {sorted(wanted)} under {tiles_root}: {', '.join(missing)}. "
            + (f"Every cut of {', '.join(excluded_slides)} at these sizes is excluded by the "
               "cut exclusion list. " if excluded_slides else "")
            + "A partial tile tree, or a size those slides were never tiled at, would "
            "silently shrink the evaluation splits; pass allow_missing_slides=True for a "
            "deliberate subset run."
        )
    if missing:
        LOG.warning(
            "Indexing without %d split-manifest slide(s) that have no tiles: %s",
            len(missing),
            ", ".join(missing),
        )

    index = pd.DataFrame(rows, columns=list(INDEX_COLUMNS))
    LOG.info(
        "Indexed %d tiles from %d cuts at sizes %s under the %r rule",
        len(index),
        index["cut_name"].nunique(),
        sorted(wanted),
        label_rule,
    )
    return index


def training_rows(index: pd.DataFrame) -> pd.DataFrame:
    """Drop the ambiguous band, leaving rows safe to train or evaluate on.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"is_ambiguous": [False, True], "label": ["negative", "ambiguous"]})
    >>> len(training_rows(frame))
    1
    """
    return index.loc[~index["is_ambiguous"]].reset_index(drop=True)


def binary_targets(index: pd.DataFrame) -> pd.Series:
    """Map an ambiguity-free index's labels onto 1 for positive and 0 for negative.

    Raises
    ------
    ValueError
        If any ambiguous row is still present, rather than scoring it as negative.
    """
    if index["is_ambiguous"].any():
        raise ValueError(
            "index still contains ambiguous rows; call training_rows() before "
            "converting labels to binary targets"
        )
    return (index["label"] == LABEL_POSITIVE).astype("int64")


def _label_for_tile(
    tile: dict[str, Any],
    fraction: float | None,
    label_rule: str,
    min_oocyte_area_fraction: float,
) -> str:
    """Return the label band for one tile under the configured rule."""
    if label_rule == LABEL_RULE_CENTROID:
        return LABEL_POSITIVE if tile["has_oocyte"] else LABEL_NEGATIVE
    if fraction is None:
        raise ValueError(f"No coverage computed for tile {tile['tile_id']!r}")
    return label_for_area_fraction(fraction, min_oocyte_area_fraction)


def _load_split_assignments(split_manifest_path: str | Path) -> dict[str, str]:
    """Read the slide-to-split mapping out of the split manifest."""
    path = _resolve_repo_path(split_manifest_path)
    manifest = _read_json(path)
    slides = manifest.get("slides")
    if not slides:
        raise ValueError(f"{path} has no slides")
    return {stem: entry["split"] for stem, entry in slides.items()}


def _resolve_repo_path(path: str | Path) -> Path:
    """Resolve a repository-relative path into an absolute path.

    Examples
    --------
    >>> _resolve_repo_path(Path("src")).is_absolute()
    True
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a Python dict.

    Examples
    --------
    >>> isinstance({}, dict)
    True
    """
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
