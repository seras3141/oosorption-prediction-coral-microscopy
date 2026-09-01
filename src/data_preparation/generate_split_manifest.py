"""Deterministic slide-level train/val/test split manifest generation.

Implements the algorithm specified in ``docs_local/plan_addendum_m6_split.md``
so every downstream training step (M7 YOLOv8-seg, the tile-classification
baseline, M10 stage classifier) reads splits from one file instead of
re-deriving its own ad hoc split.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from src.data_preparation.annotations import FEATURE_COLLECTION, extract_stage

LOG = logging.getLogger(__name__)

FORCE_TRAIN_IF_STAGE_PRESENT: tuple[int, ...] = (1,)
DEFAULT_SEED: int = 42
DEFAULT_QUOTAS: dict[tuple[str, bool], tuple[int, int]] = {
    ("CHN", False): (2, 1),
    ("CHN", True): (0, 0),
    ("LHP", False): (1, 1),
    ("LHP", True): (1, 1),
}
MANIFEST_VERSION: str = "v1"

# The 5-class oosorption stage scheme (project-wide; see CLAUDE.md's project overview).
VALID_STAGES: frozenset[int] = frozenset(range(5))

# The stage whose presence defines the stratification key (location, has_this_stage),
# per plan_addendum_m6_split.md §5.2 step 3. Not currently exposed as a parameter of
# assign_splits because the algorithm spec fixes it, unlike force_train_if_stage_present.
STRATIFY_ON_STAGE: int = 2


def compute_stage_distribution(
    geojson_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Compute per-slide location and stage-annotation counts.

    Parameters
    ----------
    geojson_dir : str or Path
        Directory containing one ``{stem}.geojson`` file per slide
        (``data/dataset_28_04/``).

    Returns
    -------
    dict
        ``{stem: {"location": str, "stage_counts": dict[int, int]}}``.
        ``location`` is parsed as the first underscore-delimited token of
        the stem. Features with no extractable stage (``extract_stage``
        returns ``None``) are skipped and logged at DEBUG; a parsed stage
        outside the valid 0-4 range is skipped and logged at WARNING.

    Examples
    --------
    >>> import json, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     fc = {"type": "FeatureCollection",
    ...           "features": [{"properties": {"name": "Oosorption Stage 0"}}]}
    ...     _ = (Path(tmp) / "CHN_A_1_1-2.geojson").write_text(json.dumps(fc))
    ...     dist = compute_stage_distribution(tmp)
    >>> dist["CHN_A_1_1-2"]["location"], dist["CHN_A_1_1-2"]["stage_counts"]
    ('CHN', {0: 1})
    """
    geojson_dir = Path(geojson_dir)
    if not geojson_dir.is_dir():
        raise FileNotFoundError(f"geojson_dir does not exist: {geojson_dir}")

    distribution: dict[str, dict[str, Any]] = {}
    for geojson_path in sorted(geojson_dir.glob("*.geojson")):
        stem = geojson_path.stem
        location = stem.split("_")[0]
        with geojson_path.open("r", encoding="utf-8") as fp:
            geojson = json.load(fp)
        if geojson.get("type") != FEATURE_COLLECTION:
            raise ValueError(f"{geojson_path} is not a GeoJSON FeatureCollection")

        stage_counts: dict[int, int] = {}
        for feature in geojson.get("features") or []:
            props = feature.get("properties") or {}
            stage = extract_stage(props)
            if stage is None:
                LOG.debug(
                    "Skipping unlabelled feature %s in %s",
                    feature.get("id"),
                    geojson_path.name,
                )
                continue
            if stage not in VALID_STAGES:
                LOG.warning(
                    "Skipping feature %s in %s: parsed stage %d is outside the valid "
                    "0-4 range (likely a data-entry typo)",
                    feature.get("id"),
                    geojson_path.name,
                    stage,
                )
                continue
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        distribution[stem] = {"location": location, "stage_counts": stage_counts}
    return distribution


def _has_stage(stage_counts: dict[int, int], stage: int) -> bool:
    """Return True if stage_counts[stage] > 0 (missing key treated as 0)."""
    return stage_counts.get(stage, 0) > 0


def assign_splits(
    stage_distribution: dict[str, dict[str, Any]],
    *,
    quotas: dict[tuple[str, bool], tuple[int, int]] = DEFAULT_QUOTAS,
    force_train_if_stage_present: tuple[int, ...] = FORCE_TRAIN_IF_STAGE_PRESENT,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Deterministically assign each slide stem to train/val/test.

    Implements the algorithm in plan_addendum_m6_split.md §5.2 exactly:
    force-train slides with a scarce stage first, then a fixed-quota,
    seeded assignment within (location, has_stage2) strata.

    Parameters
    ----------
    stage_distribution : dict
        Output of ``compute_stage_distribution``.
    quotas : dict, optional
        Maps ``(location, has_stage2)`` to ``(n_val, n_test)``. Remaining
        stratum members go to train. Must cover every stratum present in
        ``stage_distribution`` after the force-train step, or a
        ``KeyError`` is raised (fail loudly rather than silently
        defaulting an unplanned-for stratum). Both values must be
        non-negative and their sum must not exceed the stratum's size, or
        a ``ValueError`` is raised (fail loudly rather than silently
        under-filling a split or overwriting an earlier assignment).
    force_train_if_stage_present : tuple of int, optional
        Any slide containing an annotation whose stage is in this tuple
        is force-assigned to train before stratification.
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Strata are shuffled in
        sorted-key order using one shared RNG instance so the result is
        fully reproducible for a given seed and input.

    Returns
    -------
    dict
        ``{stem: "train" | "val" | "test"}``, one entry per input stem.

    Examples
    --------
    >>> dist = {"A_1": {"location": "A", "stage_counts": {0: 5}},
    ...         "A_2": {"location": "A", "stage_counts": {0: 5}}}
    >>> q = {("A", False): (0, 1)}
    >>> result = assign_splits(dist, quotas=q, seed=0)
    >>> sorted(result.values())
    ['test', 'train']
    """
    quotas = dict(quotas)
    result: dict[str, str] = {}
    remaining_stems: list[str] = []
    for stem, info in stage_distribution.items():
        if any(_has_stage(info["stage_counts"], s) for s in force_train_if_stage_present):
            result[stem] = "train"
        else:
            remaining_stems.append(stem)

    strata: dict[tuple[str, bool], list[str]] = {}
    for stem in remaining_stems:
        info = stage_distribution[stem]
        key = (info["location"], _has_stage(info["stage_counts"], STRATIFY_ON_STAGE))
        strata.setdefault(key, []).append(stem)

    rng = np.random.default_rng(seed)
    for key in sorted(strata.keys()):
        try:
            n_val, n_test = quotas[key]
        except KeyError:
            raise KeyError(
                f"No quota defined for stratum {key} "
                f"({len(strata[key])} slide(s): {sorted(strata[key])})"
            ) from None
        if n_val < 0 or n_test < 0:
            raise ValueError(f"Quota for stratum {key} must be non-negative, got {(n_val, n_test)}")
        n_train_start = n_val + n_test
        if n_train_start > len(strata[key]):
            raise ValueError(
                f"Stratum {key} has only {len(strata[key])} slide(s), which cannot cover "
                f"quota (n_val={n_val}, n_test={n_test})"
            )
        shuffled = sorted(strata[key])
        rng.shuffle(shuffled)
        for stem in shuffled[:n_val]:
            result[stem] = "val"
        for stem in shuffled[n_val:n_train_start]:
            result[stem] = "test"
        for stem in shuffled[n_train_start:]:
            result[stem] = "train"

    if set(result.keys()) != set(stage_distribution.keys()):
        raise AssertionError("every slide must be assigned exactly once")
    return result


def _quota_key(location: str, has_stratify_stage: bool) -> str:
    prefix = "has" if has_stratify_stage else "no"
    return f"{location}_{prefix}_stage{STRATIFY_ON_STAGE}"


def build_split_manifest(
    geojson_dir: str | Path,
    *,
    quotas: dict[tuple[str, bool], tuple[int, int]] = DEFAULT_QUOTAS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compute distribution + assignment and assemble the manifest dict
    documented in plan_addendum_m6_split.md §6.

    Parameters
    ----------
    geojson_dir : str or Path
        Passed through to ``compute_stage_distribution``.
    quotas : dict, optional
        Passed through to ``assign_splits``.
    seed : int, optional
        Passed through to ``assign_splits``; also recorded in the
        returned manifest's ``"seed"`` field.

    Returns
    -------
    dict
        Ready to pass to ``write_split_manifest``.

    Examples
    --------
    >>> import json, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     fc = {"type": "FeatureCollection",
    ...           "features": [{"properties": {"name": "Oosorption Stage 0"}}]}
    ...     _ = (Path(tmp) / "A_1_1-2.geojson").write_text(json.dumps(fc))
    ...     _ = (Path(tmp) / "A_2_1-2.geojson").write_text(json.dumps(fc))
    ...     manifest = build_split_manifest(tmp, quotas={("A", False): (0, 1)}, seed=0)
    >>> manifest["n_slides"]
    2
    >>> sum(manifest["split_counts"].values()) == manifest["n_slides"]
    True
    """
    distribution = compute_stage_distribution(geojson_dir)
    assignment = assign_splits(distribution, quotas=quotas, seed=seed)

    split_counts = {"train": 0, "val": 0, "test": 0}
    slides: dict[str, Any] = {}
    for stem, info in distribution.items():
        split = assignment[stem]
        split_counts[split] += 1
        slides[stem] = {
            "split": split,
            "location": info["location"],
            "stage_counts": dict(info["stage_counts"]),
        }

    quotas_out = {
        _quota_key(location, has_stage2): [n_val, n_test]
        for (location, has_stage2), (n_val, n_test) in quotas.items()
    }

    return {
        "manifest_version": MANIFEST_VERSION,
        "created": date.today().isoformat(),
        "seed": seed,
        "force_train_if_stage_present": list(FORCE_TRAIN_IF_STAGE_PRESENT),
        "quotas": quotas_out,
        "n_slides": len(distribution),
        "split_counts": split_counts,
        "slides": slides,
    }


def write_split_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write *manifest* as JSON, 2-space indent, snake_case keys.

    Parameters
    ----------
    manifest : dict
        Output of ``build_split_manifest``.
    output_path : str or Path
        Typically ``data/splits/split_manifest.json``. Parent directory
        is created if absent.

    Returns
    -------
    Path
        The path written to.

    Examples
    --------
    >>> import tempfile
    >>> manifest = {"n_slides": 2, "split_counts": {"train": 1, "val": 1, "test": 0}}
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     out = write_split_manifest(manifest, Path(tmp) / "sub" / "split_manifest.json")
    ...     reloaded = json.loads(out.read_text())
    >>> reloaded == manifest
    True
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
        fp.write("\n")
    return output_path
