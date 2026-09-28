"""Group slides into specimens and deal specimens into cross-validation folds."""

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
ANCHOR_SITE = "CHN"
DEFAULT_BALANCE_SIZES: tuple[int, ...] = (512, 1024)
DEFAULT_MIN_POSITIVES = {512: 250}
# Without a cap, one fold takes half the corpus.
DEFAULT_MAX_FOLD_SIZE_RATIO = 1.4
SIZE_BALANCE_TILE_SIZE = 512
DEFAULT_MAX_INNER_VAL_POSITIVE_SHARE = 0.25
INNER_VAL_REFERENCE_SIZE = 512


def specimen_of(stem: str) -> str:
    """Return the specimen key: the stem without its cut range."""
    parts = stem.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else stem


def site_of(specimen_or_stem: str) -> str:
    """Return the sampling site of a specimen key or stem."""
    return specimen_or_stem.split("_", 1)[0]


def specimen_counts(index: pd.DataFrame) -> pd.DataFrame:
    """Tally tiles, positives and ambiguous tiles per specimen and tile size."""
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
    """Deal specimens into folds, one anchor-site specimen per fold."""
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

    # Lexicographic order makes argmin the tie-break.
    grid = np.array(list(itertools.product(range(n_folds), repeat=len(others))), dtype=np.int8)
    if grid.size == 0:
        grid = np.zeros((1, 0), dtype=np.int8)

    def per_fold(column: str) -> np.ndarray:
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
    """Pick each fold's inner-validation specimen."""
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
    """Assemble the fold manifest."""
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
            # Includes ambiguous tiles; descriptive only.
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
    """Build one held-out fold's split in the v1 manifest shape."""
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
    """Write the per-fold split manifests, then the hashed fold manifest."""
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
    """Return specimens whose slides span more than one split."""
    by_specimen: dict[str, set[str]] = {}
    for stem, split in split_assignments.items():
        by_specimen.setdefault(specimen_of(stem), set()).add(split)
    return {s: sorted(v) for s, v in sorted(by_specimen.items()) if len(v) > 1}


def _dump(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, indent=2) + "\n").encode("utf-8")
