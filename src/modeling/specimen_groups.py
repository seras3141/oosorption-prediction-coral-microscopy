"""Group slides into specimens and deal the specimens into cross-validation folds.

A specimen is one coral polyp, identified by ``(Site, Season, Polyp No)``. One NDPI file
covers a contiguous range of cuts from a single specimen, so a specimen can produce
several slides, and those slides are serial sections through the same oocytes. The
slide-level split holds slides apart but not specimens: 26 slides are only 14 specimens,
and five of them straddle train and a held-out split. This module is the grouping that
closes that leak.

The fold assignment is a deterministic exhaustive search rather than a random deal:

* Every fold gets exactly one CHN specimen. CHN has as many specimens as there are
  folds, so this is the only arrangement in which every fold can test cross-site
  transfer. CHN specimens go to folds in sorted order.
* The LHP specimens are then assigned to minimise the spread of positive-tile rate
  across folds, taken as the worst spread over the balancing tile sizes, subject to a
  floor on positive tiles per fold and a cap on how much larger the largest fold may be
  than the smallest. Ties go to the lexicographically first assignment.

The size cap is not optional decoration. Minimising rate spread alone is solved by
parking the densest specimens together with the sparsest in one very large fold: on
this corpus the unconstrained optimum puts 20,934 of 44,193 512 px tiles in one fold,
3.5x the smallest, which balances the rates by making one fold half the data.

Each fold also names one *inner-validation* specimen. When that fold is a training
fold, its inner-val specimen is held out of training and used for early stopping and
threshold selection, so the held-out fold is never selection data.

Deliberately free of ``torch``, like the rest of the data layer: the detection
milestones need the same grouping without the ML stack.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.modeling.tile_labels import LABEL_AMBIGUOUS, LABEL_POSITIVE

FOLD_MANIFEST_VERSION = "cv-v1"
DEFAULT_N_FOLDS = 4
#: The site that must appear once in every fold.
ANCHOR_SITE = "CHN"
#: Sizes whose fold positive rates the search balances. The small scales are recorded
#: but not balanced on: their rates track the large scales, and adding them would let
#: the 741k-tile 128 px count dominate a search the plan specified at 512 and 1024 px.
DEFAULT_BALANCE_SIZES: tuple[int, ...] = (512, 1024)
DEFAULT_MIN_POSITIVES = {512: 250}
#: Largest fold's tile count over the smallest's, at :data:`SIZE_BALANCE_TILE_SIZE`.
#: 1.4 is a round cap, not a fitted one: the assignment it selects on this corpus is
#: the same for any cap from 1.38 to 1.44.
DEFAULT_MAX_FOLD_SIZE_RATIO = 1.4
SIZE_BALANCE_TILE_SIZE = 512
#: A specimen holding more than this share of all positives at the reference size is
#: never picked as inner validation: removing it would strip a large part of the
#: positive class out of every training set it would otherwise be in.
DEFAULT_MAX_INNER_VAL_POSITIVE_SHARE = 0.25
INNER_VAL_REFERENCE_SIZE = 512


def specimen_of(stem: str) -> str:
    """The specimen a slide belongs to, which is the unit that is actually independent.

    Slide stems are ``Site_Season_PolypNo_cutrange`` and one specimen -- a single polyp --
    spans several cut ranges, so dropping the last component groups serial sections of the
    same polyp together. `CHN_SP_5_22-24` and `CHN_SP_5_25-27` are consecutive sections
    through one polyp, not two independent samples.

    Examples
    --------
    >>> specimen_of("CHN_SP_5_22-24")
    'CHN_SP_5'
    """
    parts = stem.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else stem


def site_of(specimen_or_stem: str) -> str:
    """The sampling site, the first component of a specimen key or slide stem.

    Examples
    --------
    >>> site_of("LHP_SU_9")
    'LHP'
    """
    return specimen_or_stem.split("_", 1)[0]


def specimen_counts(index: pd.DataFrame) -> pd.DataFrame:
    """Tally tiles, positives and ambiguous tiles per specimen and tile size.

    Parameters
    ----------
    index : pandas.DataFrame
        Output of :func:`src.modeling.tile_index.load_tile_index`, ambiguous rows
        included -- the tile counts here are of everything indexed.

    Returns
    -------
    pandas.DataFrame
        One row per specimen, sorted, with columns ``site``, ``slides`` (a sorted tuple
        of stems) and, per tile size, ``tiles_{size}``, ``positives_{size}`` and
        ``ambiguous_{size}``.

    Examples
    --------
    >>> frame = pd.DataFrame({
    ...     "stem": ["A_X_1_1-2", "A_X_1_3-4", "B_X_2_1-2"],
    ...     "tile_size": [512, 512, 512],
    ...     "label": ["positive", "negative", "ambiguous"],
    ... })
    >>> specimen_counts(frame)[["tiles_512", "positives_512", "ambiguous_512"]].values.tolist()
    [[2, 1, 0], [1, 0, 1]]
    """
    frame = index.assign(
        specimen=index["stem"].map(specimen_of),
        is_positive=index["label"] == LABEL_POSITIVE,
        is_ambiguous_label=index["label"] == LABEL_AMBIGUOUS,
    )
    slides = frame.groupby("specimen")["stem"].agg(lambda s: tuple(sorted(set(s))))
    grouped = frame.groupby(["specimen", "tile_size"]).agg(
        tiles=("stem", "size"),
        positives=("is_positive", "sum"),
        ambiguous=("is_ambiguous_label", "sum"),
    )
    wide = grouped.unstack("tile_size", fill_value=0)
    wide.columns = [f"{name}_{size}" for name, size in wide.columns]
    out = pd.DataFrame({"site": [site_of(s) for s in wide.index], "slides": slides[wide.index]},
                       index=wide.index)
    out = out.join(wide.astype("int64"))
    out.index.name = "specimen"
    return out.sort_index()


def tile_sizes_in(counts: pd.DataFrame) -> tuple[int, ...]:
    """The tile sizes a :func:`specimen_counts` frame carries, ascending."""
    return tuple(sorted(int(c.split("_")[1]) for c in counts.columns if c.startswith("tiles_")))


def assign_folds(
    counts: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    balance_sizes: tuple[int, ...] = DEFAULT_BALANCE_SIZES,
    min_positives: dict[int, int] | None = None,
    max_size_ratio: float | None = DEFAULT_MAX_FOLD_SIZE_RATIO,
) -> dict[str, int]:
    """Deal specimens into folds: one anchor-site specimen each, the rest balanced.

    Parameters
    ----------
    counts : pandas.DataFrame
        Output of :func:`specimen_counts`.
    n_folds : int, optional
        Number of folds. Must equal the number of anchor-site specimens.
    balance_sizes : tuple of int, optional
        Tile sizes whose per-fold positive rate is balanced.
    min_positives : dict of int to int, optional
        Per tile size, the fewest positive tiles any fold may hold. Defaults to
        :data:`DEFAULT_MIN_POSITIVES`.
    max_size_ratio : float or None, optional
        Cap on the largest fold's tile count over the smallest's at
        :data:`SIZE_BALANCE_TILE_SIZE`. None disables it.

    Returns
    -------
    dict of str to int
        Specimen to fold number, folds numbered from 1.

    Raises
    ------
    ValueError
        If the anchor site does not have exactly ``n_folds`` specimens, if a requested
        size is absent from ``counts``, or if no assignment satisfies the constraints.

    Examples
    --------
    >>> counts = pd.DataFrame(
    ...     {"site": ["CHN", "CHN", "LHP", "LHP"],
    ...      "tiles_512": [10, 10, 10, 10], "positives_512": [1, 5, 5, 1]},
    ...     index=["CHN_A_1", "CHN_A_2", "LHP_A_1", "LHP_A_2"])
    >>> assign_folds(counts, n_folds=2, balance_sizes=(512,), min_positives={},
    ...              max_size_ratio=None)
    {'CHN_A_1': 1, 'CHN_A_2': 2, 'LHP_A_1': 1, 'LHP_A_2': 2}
    """
    floors = DEFAULT_MIN_POSITIVES if min_positives is None else min_positives
    needed = set(balance_sizes) | set(floors)
    if max_size_ratio is not None:
        needed.add(SIZE_BALANCE_TILE_SIZE)
    for size in needed:
        for column in (f"tiles_{size}", f"positives_{size}"):
            if column not in counts.columns:
                raise ValueError(f"counts has no {column!r} column")

    anchors = sorted(s for s in counts.index if counts.at[s, "site"] == ANCHOR_SITE)
    others = sorted(s for s in counts.index if counts.at[s, "site"] != ANCHOR_SITE)
    if len(anchors) != n_folds:
        raise ValueError(
            f"one {ANCHOR_SITE} specimen per fold needs exactly {n_folds} {ANCHOR_SITE} "
            f"specimens, found {len(anchors)}: {anchors}"
        )

    # Every assignment of the non-anchor specimens, in lexicographic order, so the first
    # minimum found is the deterministic tie-break. 4^10 is about a million rows.
    grid = np.array(list(itertools.product(range(n_folds), repeat=len(others))), dtype=np.int8)
    if grid.size == 0:
        grid = np.zeros((1, 0), dtype=np.int8)

    def per_fold(column: str) -> np.ndarray:
        """Fold totals of one count column, shape (n_assignments, n_folds)."""
        other_values = counts.loc[others, column].to_numpy(dtype=np.float64)
        anchor_values = counts.loc[anchors, column].to_numpy(dtype=np.float64)
        totals = np.stack(
            [(grid == fold) @ other_values for fold in range(n_folds)], axis=1
        ) if others else np.zeros((1, n_folds))
        return totals + anchor_values[np.newaxis, :]

    feasible = np.ones(len(grid), dtype=bool)
    for size, floor in floors.items():
        feasible &= (per_fold(f"positives_{size}") >= floor).all(axis=1)
    if max_size_ratio is not None:
        sizes = per_fold(f"tiles_{SIZE_BALANCE_TILE_SIZE}")
        feasible &= sizes.max(axis=1) <= max_size_ratio * sizes.min(axis=1)

    spread = np.zeros(len(grid))
    for size in balance_sizes:
        tiles = per_fold(f"tiles_{size}")
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.where(tiles > 0, per_fold(f"positives_{size}") / tiles, np.nan)
        # A fold with no tiles at a balanced size cannot be scored there at all.
        size_spread = np.nanmax(rates, axis=1) - np.nanmin(rates, axis=1)
        size_spread[np.isnan(rates).any(axis=1)] = np.inf
        spread = np.maximum(spread, size_spread)
    spread[~feasible] = np.inf

    best = int(np.argmin(spread))
    if not np.isfinite(spread[best]):
        raise ValueError(
            f"no fold assignment satisfies the positive-tile floor {floors} and the "
            f"fold-size cap {max_size_ratio}"
        )

    assignment = {specimen: fold + 1 for fold, specimen in enumerate(anchors)}
    assignment.update({s: int(f) + 1 for s, f in zip(others, grid[best])})
    return dict(sorted(assignment.items()))


def choose_inner_val(
    counts: pd.DataFrame,
    assignment: dict[str, int],
    max_positive_share: float = DEFAULT_MAX_INNER_VAL_POSITIVE_SHARE,
    reference_size: int = INNER_VAL_REFERENCE_SIZE,
) -> dict[int, str]:
    """Pick each fold's inner-validation specimen.

    Within each fold, the non-anchor specimen whose positive rate at
    ``reference_size`` is closest to the corpus rate, excluding any specimen that holds
    more than ``max_positive_share`` of all positives. The anchor-site specimen is never
    chosen: each fold holds exactly one, and removing it would take that site out of the
    fold's training contribution. Ties go to the specimen that sorts first.

    Rates here count every indexed tile, ambiguous included, matching
    :func:`assign_folds`.

    Returns
    -------
    dict of int to str
        Fold number to its inner-val specimen.

    Raises
    ------
    ValueError
        If a fold has no eligible specimen.

    Examples
    --------
    >>> counts = pd.DataFrame(
    ...     {"site": ["CHN", "LHP", "LHP"],
    ...      "tiles_512": [10, 10, 10], "positives_512": [2, 2, 5]},
    ...     index=["CHN_A_1", "LHP_A_1", "LHP_A_2"])
    >>> choose_inner_val(counts, {"CHN_A_1": 1, "LHP_A_1": 1, "LHP_A_2": 1})
    {1: 'LHP_A_1'}
    """
    tiles = counts[f"tiles_{reference_size}"]
    positives = counts[f"positives_{reference_size}"]
    corpus_rate = positives.sum() / tiles.sum()
    share = positives / positives.sum()

    chosen: dict[int, str] = {}
    for fold in sorted(set(assignment.values())):
        candidates = sorted(
            s for s, f in assignment.items()
            if f == fold
            and counts.at[s, "site"] != ANCHOR_SITE
            and share[s] <= max_positive_share
        )
        if not candidates:
            raise ValueError(f"fold {fold} has no specimen eligible for inner validation")
        chosen[fold] = min(
            candidates, key=lambda s: (abs(positives[s] / tiles[s] - corpus_rate), s)
        )
    return chosen


def build_fold_manifest(
    counts: pd.DataFrame,
    assignment: dict[str, int],
    inner_val: dict[int, str],
    *,
    split_manifest_dir: str = f"data/splits/{FOLD_MANIFEST_VERSION}",
    source_split_manifest_sha256: str | None = None,
    min_positives: dict[int, int] | None = None,
    max_size_ratio: float | None = DEFAULT_MAX_FOLD_SIZE_RATIO,
) -> dict[str, Any]:
    """Assemble the fold manifest: which specimens and slides each fold holds, and why.

    ``split_manifest_sha256`` is left None per fold; :func:`write_fold_manifests`
    fills it with the hash of the per-fold split manifest it actually writes.
    """
    floors = DEFAULT_MIN_POSITIVES if min_positives is None else min_positives
    sizes = tile_sizes_in(counts)
    folds = []
    for fold in sorted(set(assignment.values())):
        members = sorted(s for s, f in assignment.items() if f == fold)
        member_counts = counts.loc[members]
        tiles = {str(s): int(member_counts[f"tiles_{s}"].sum()) for s in sizes}
        positives = {str(s): int(member_counts[f"positives_{s}"].sum()) for s in sizes}
        ambiguous = {str(s): int(member_counts[f"ambiguous_{s}"].sum()) for s in sizes}
        slides = sorted(stem for s in members for stem in counts.at[s, "slides"])
        folds.append({
            "fold": fold,
            "specimens": members,
            "slides": slides,
            "n_slides": len(slides),
            "tiles": tiles,
            "positives": positives,
            "ambiguous": ambiguous,
            # Descriptive only: the denominator includes ambiguous tiles. AUPRC lift must
            # use the prevalence of the tiles actually scored, which excludes them.
            # None rather than a ZeroDivisionError when a fold has no tiles at a
            # recorded size, e.g. a scale tiled for only some slides.
            "positive_rate": {
                s: round(positives[s] / tiles[s], 4) if tiles[s] else None for s in tiles
            },
            "inner_val_specimen": inner_val[fold],
            "split_manifest_path": f"{split_manifest_dir}/fold{fold}_split_manifest.json",
            "split_manifest_sha256": None,
        })
    return {
        "manifest_version": FOLD_MANIFEST_VERSION,
        "created": date.today().isoformat(),
        "grouping": "specimen",
        "specimen_key": "site_season_polyp",
        "n_specimens": len(assignment),
        "n_slides": sum(f["n_slides"] for f in folds),
        "n_folds": len(folds),
        "assignment_rule": (
            f"one {ANCHOR_SITE} specimen per fold, in sorted order; the rest assigned by "
            "exhaustive search to minimise the worst per-size spread of fold positive "
            f"rate at {list(DEFAULT_BALANCE_SIZES)} px, first assignment on ties"
        ),
        "max_fold_size_ratio": {str(SIZE_BALANCE_TILE_SIZE): max_size_ratio},
        "min_positive_tiles_per_fold": {str(k): v for k, v in floors.items()},
        "inner_val_rule": (
            f"per fold, the non-{ANCHOR_SITE} specimen whose {INNER_VAL_REFERENCE_SIZE} px "
            "positive rate is closest to the corpus rate, excluding specimens holding more "
            f"than {DEFAULT_MAX_INNER_VAL_POSITIVE_SHARE:.0%} of all positives"
        ),
        "source_split_manifest_sha256": source_split_manifest_sha256,
        "folds": folds,
    }


def fold_split_manifest(
    fold_manifest: dict[str, Any], held_out: int
) -> dict[str, Any]:
    """The train/val/test split for one held-out fold, in the v1 split-manifest shape.

    ``test`` is the held-out fold, ``val`` is the inner-val specimen of every other
    fold, and ``train`` is everything else. Readers of the v1 manifest only use
    ``slides[stem]["split"]``, so the trainer and evaluator consume this unchanged.
    """
    folds = {f["fold"]: f for f in fold_manifest["folds"]}
    if held_out not in folds:
        raise ValueError(f"fold {held_out} is not in the fold manifest")
    slides: dict[str, Any] = {}
    for number, fold in folds.items():
        for stem in fold["slides"]:
            specimen = specimen_of(stem)
            if number == held_out:
                split = "test"
            elif specimen == fold["inner_val_specimen"]:
                split = "val"
            else:
                split = "train"
            slides[stem] = {
                "split": split,
                "location": site_of(stem),
                "specimen": specimen,
                "fold": number,
            }
    counts = {name: sum(v["split"] == name for v in slides.values())
              for name in ("train", "val", "test")}
    return {
        "manifest_version": f"{fold_manifest['manifest_version']}-fold{held_out}",
        "created": fold_manifest["created"],
        "held_out_fold": held_out,
        "n_slides": len(slides),
        "split_counts": counts,
        "slides": dict(sorted(slides.items())),
    }


def write_fold_manifests(
    fold_manifest: dict[str, Any],
    fold_manifest_path: str | Path,
    repo_root: str | Path,
) -> list[Path]:
    """Write every per-fold split manifest, record their hashes, then the fold manifest.

    Returns
    -------
    list of Path
        The per-fold split manifests, then the fold manifest, in write order.
    """
    root = Path(repo_root)
    written: list[Path] = []
    for fold in fold_manifest["folds"]:
        path = root / fold["split_manifest_path"]
        payload = _dump(fold_split_manifest(fold_manifest, fold["fold"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        fold["split_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
        written.append(path)
    out = Path(fold_manifest_path)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_dump(fold_manifest))
    written.append(out)
    return written


def straddling_specimens(split_assignments: dict[str, str]) -> dict[str, list[str]]:
    """Specimens whose slides sit in more than one split of a slide-level manifest.

    Examples
    --------
    >>> straddling_specimens({"A_X_1_1-2": "train", "A_X_1_3-4": "val", "B_X_2_1-2": "val"})
    {'A_X_1': ['train', 'val']}
    """
    by_specimen: dict[str, set[str]] = {}
    for stem, split in split_assignments.items():
        by_specimen.setdefault(specimen_of(stem), set()).add(split)
    return {s: sorted(v) for s, v in sorted(by_specimen.items()) if len(v) > 1}


def _dump(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, indent=2) + "\n").encode("utf-8")
