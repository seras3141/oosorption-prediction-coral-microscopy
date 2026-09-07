"""Build the tile-level training index: one row per tile, joined to its slide's split.

Flattens the 77 per-cut tile manifests into a single table, attaches the
geometry-derived oocyte label from :mod:`src.modeling.tile_labels`, and joins each tile
to its parent slide's train/val/test assignment. Tiles are split by slide, never within
a slide, so no oocyte can appear on both sides of the split.

Deliberately free of ``torch`` -- see :mod:`src.modeling.tile_classification_dataset` for
the tensor-serving layer. Keeping the two apart means the detection milestones can reuse
this index and the corrected label without installing the ML stack.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.modeling.tile_labels import (
    DEFAULT_MIN_OOCYTE_AREA_FRACTION,
    LABEL_AMBIGUOUS,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    label_for_area_fraction,
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
        Permit split-manifest slides that have no tiles. Off by default: `data/` is not
        synced to every checkout, so a partially copied tile tree would otherwise yield
        a quietly smaller val or test set and confidence intervals over fewer slides
        than the split defines. Set it only for a deliberate subset run.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`INDEX_COLUMNS`. ``label`` is one of ``"positive"``,
        ``"ambiguous"``, ``"negative"``; ``is_ambiguous`` is the same information as a
        boolean, carried separately because filtering it is the common operation and a
        caller that forgets would otherwise train on tiles this rule declines to call.

    Raises
    ------
    ValueError
        If ``label_rule`` is unknown, if no manifests are found, if ``tile_sizes``
        matches no tile in any manifest, if a stem under ``tiles_dir`` is absent from the
        split manifest -- an unassigned slide is a broken split, not a slide to quietly
        drop -- or, unless ``allow_missing_slides``, if a slide in the split manifest has
        no tiles.

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
    rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = _read_json(Path(manifest_path))
        stem, cut_name = manifest["stem"], manifest["cut_name"]
        if stem not in slides:
            raise ValueError(
                f"{stem} has tiles under {tiles_root} but no entry in the split "
                f"manifest; refusing to guess its split"
            )
        assigned = slides[stem]
        geojson_path = cuts_root / stem / f"{cut_name}_annotations.geojson"

        fractions: dict[str, float] = {}
        if label_rule == LABEL_RULE_AREA:
            # Reuse the manifest already parsed above; these files embed per-tile
            # annotation coordinates and total hundreds of megabytes, so parsing them
            # twice per cut would dominate the cost of building the index.
            fractions = oocyte_area_fractions_for_manifest(
                manifest,
                geojson_path,
                tile_sizes=tuple(wanted),
                manifest_path=Path(manifest_path),
            )

        for tile in manifest.get("tiles", []):
            if tile["tile_size"] not in wanted:
                continue
            # Marked here rather than per manifest: a slide whose cuts hold none of the
            # requested sizes contributes no rows, and counting it as present would let
            # it vanish from the index without tripping the evaluation-shrink guard.
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

    if not rows:
        raise ValueError(
            f"No tiles at sizes {sorted(wanted)} in any manifest under {tiles_root}; "
            "the manifests in this corpus carry sizes 128, 256, 512 and 1024. A size "
            "that matches nothing would otherwise return an empty index and only fail "
            "later, where the cause is no longer visible."
        )

    missing = sorted(set(slides) - seen_stems)
    if missing and not allow_missing_slides:
        raise ValueError(
            f"{len(missing)} slide(s) in the split manifest contribute no tiles at "
            f"sizes {sorted(wanted)} under {tiles_root}: {', '.join(missing)}. A partial "
            "tile tree, or a size those slides were never tiled at, would silently "
            "shrink the evaluation splits; pass allow_missing_slides=True for a "
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
        len(manifest_paths),
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
        If any ambiguous row is still present -- silently scoring those as negative
        would inflate the negative pool with tiles the label rule declined to call.
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
