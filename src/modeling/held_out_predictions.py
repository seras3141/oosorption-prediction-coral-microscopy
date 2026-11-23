"""Every held-out tile of a CV run; saved predictions omit the ambiguous band."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.modeling.cross_validation import (
    _fold_entry,
    check_coverage,
    check_fold_isolation,
    load_fold_manifest,
)
from src.modeling.tile_index import _resolve_repo_path, load_tile_index

if TYPE_CHECKING:
    from src.modeling.train_tile_classifier import TrainingConfig

HELD_OUT_SPLIT = "test"
# Coverage round-trips through CSV as decimal text.
COVERAGE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FoldRun:
    """One fold model of a CV run and the files it was evaluated from."""

    fold: int
    run_id: str
    checkpoint_path: Path
    saved_predictions_path: Path
    config: "TrainingConfig"
    label_rule: str
    scoring_area_fraction: float
    n_ambiguous_held_out: int


@dataclass(frozen=True)
class CvRun:
    """A finished CV run, checked against the fold manifest it was built on."""

    cv_results_path: Path
    cv_results: dict[str, Any]
    fold_manifest: dict[str, Any]
    folds: tuple[FoldRun, ...]


def load_cv_run(cv_results_path: str | Path) -> CvRun:
    """Read a CV run and each fold's model, refusing anything that has drifted."""
    path = _resolve_repo_path(cv_results_path)
    cv_results = json.loads(path.read_text(encoding="utf-8"))
    if cv_results.get("is_partial") or cv_results.get("folds_missing"):
        raise ValueError(
            f"{path} is a partial CV run (folds missing: {cv_results.get('folds_missing')}); "
            "its held-out tiles do not cover the corpus"
        )
    fold_manifest = load_fold_manifest(cv_results["fold_manifest_path"])
    if fold_manifest["_sha256"] != cv_results["fold_manifest_sha256"]:
        raise ValueError(
            f"{cv_results['fold_manifest_path']} has changed since {path.parent.name} ran "
            f"(sha256 {cv_results['fold_manifest_sha256'][:12]} then, "
            f"{fold_manifest['_sha256'][:12]} now); its folds no longer describe this run"
        )
    folds = tuple(_load_fold_run(entry, fold_manifest) for entry in cv_results["per_fold"])
    expected = sorted(entry["fold"] for entry in fold_manifest["folds"])
    if sorted(fold.fold for fold in folds) != expected:
        raise ValueError(
            f"{path} has folds {sorted(fold.fold for fold in folds)}, but its fold manifest "
            f"has {expected}"
        )
    labellings = {(fold.label_rule, fold.scoring_area_fraction) for fold in folds}
    if len(labellings) != 1:
        raise ValueError(f"{path} has folds scored under different labellings: {sorted(labellings)}")
    return CvRun(path, cv_results, fold_manifest, folds)


def held_out_tiles(fold_run: FoldRun, fold_manifest: dict[str, Any]) -> pd.DataFrame:
    """Every tile the fold held out, labelled as the run scored it, ambiguous included."""
    config = fold_run.config
    index = load_tile_index(
        split_manifest_path=config.split_manifest_path,
        tile_sizes=(config.tile_size,),
        label_rule=fold_run.label_rule,
        min_oocyte_area_fraction=fold_run.scoring_area_fraction,
        cut_exclusions=config.cut_exclusions(),
    )
    held_out = index.loc[index["split"] == HELD_OUT_SPLIT].reset_index(drop=True)
    check_fold_isolation(held_out, fold_manifest, fold_run.fold, required_splits=(HELD_OUT_SPLIT,))
    return held_out


def check_matches_saved_predictions(fold_run: FoldRun, held_out: pd.DataFrame) -> None:
    """Refuse a held-out set differing from the run's beyond the ambiguous band."""
    saved = pd.read_csv(fold_run.saved_predictions_path)
    saved_ids = set(saved.loc[saved["split"] == HELD_OUT_SPLIT, "tile_id"])
    held_ids = set(held_out["tile_id"])
    missing = sorted(saved_ids - held_ids)
    if missing:
        raise ValueError(
            f"fold {fold_run.fold}: {len(missing)} tile(s) the run scored are not held out "
            f"now, e.g. {missing[:3]}"
        )
    extra = held_out.loc[~held_out["tile_id"].isin(saved_ids)]
    unscored = extra.loc[~extra["is_ambiguous"]]
    if len(unscored):
        raise ValueError(
            f"fold {fold_run.fold}: {len(unscored)} held-out tile(s) outside the ambiguous "
            f"band were never scored by the run, e.g. {sorted(unscored['tile_id'])[:3]}"
        )
    scored_ambiguous = held_out.loc[held_out["tile_id"].isin(saved_ids) & held_out["is_ambiguous"]]
    if len(scored_ambiguous):
        raise ValueError(
            f"fold {fold_run.fold}: {len(scored_ambiguous)} tile(s) the run scored are "
            f"ambiguous under its labelling now, e.g. {sorted(scored_ambiguous['tile_id'])[:3]}"
        )
    scored = saved.loc[saved["split"] == HELD_OUT_SPLIT].merge(
        held_out, on="tile_id", suffixes=("_saved", "_now")
    )
    relabelled = scored.loc[
        (scored["label_saved"] != scored["label_now"])
        | ~np.isclose(scored["oocyte_area_fraction_saved"].astype(float),
                      scored["oocyte_area_fraction_now"].astype(float),
                      rtol=0.0, atol=COVERAGE_TOLERANCE, equal_nan=True)
    ]
    if len(relabelled):
        raise ValueError(
            f"fold {fold_run.fold}: {len(relabelled)} tile(s) have a different label or "
            f"oocyte coverage from the one the run scored, e.g. "
            f"{sorted(relabelled['tile_id'])[:3]}; the annotations changed after the run"
        )
    n_ambiguous = int(held_out["is_ambiguous"].sum())
    if n_ambiguous != fold_run.n_ambiguous_held_out:
        raise ValueError(
            f"fold {fold_run.fold}: {n_ambiguous} held-out tile(s) are ambiguous now, but the "
            f"run excluded {fold_run.n_ambiguous_held_out} as ambiguous"
        )


def check_held_out_coverage(held_out: dict[int, pd.DataFrame], n_tiles_held_out: int) -> int:
    """Refuse a tile held out twice, or a scorable total other than the run recorded."""
    total = check_coverage(held_out)
    scorable = sum(int((~frame["is_ambiguous"]).sum()) for frame in held_out.values())
    if scorable != n_tiles_held_out:
        raise ValueError(
            f"{scorable} held-out tiles outside the ambiguous band, but the run recorded "
            f"{n_tiles_held_out}"
        )
    return total


def collect_held_out_tiles(cv_run: CvRun) -> dict[int, pd.DataFrame]:
    """Every fold's held-out tiles, after every check has passed."""
    held_out = {}
    for fold_run in cv_run.folds:
        frame = held_out_tiles(fold_run, cv_run.fold_manifest)
        check_matches_saved_predictions(fold_run, frame)
        held_out[fold_run.fold] = frame
    check_held_out_coverage(held_out, int(cv_run.cv_results["n_tiles_held_out"]))
    return held_out


def _load_fold_run(summary: dict[str, Any], fold_manifest: dict[str, Any]) -> FoldRun:
    import torch

    from src.modeling.train_tile_classifier import TrainingConfig

    fold = int(summary["fold"])
    predictions_path = _resolve_repo_path(summary["predictions_path"])
    results_path = predictions_path.with_name(
        predictions_path.name.replace("predictions", "results", 1)
    ).with_suffix(".json")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("run_id") != summary["run_id"] or _normalised(
        results.get("predictions_path", "")
    ) != _normalised(predictions_path):
        raise ValueError(
            f"fold {fold}: {results_path} describes run {results.get('run_id')} with "
            f"predictions {results.get('predictions_path')}, not the run and predictions "
            "the CV run recorded"
        )
    checkpoint_path = _resolve_repo_path(results["checkpoint_path"])
    # Primitives-only config allows weights_only loading.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TrainingConfig(**checkpoint["config"])

    if config.run_id() != summary["run_id"]:
        raise ValueError(
            f"fold {fold}: {checkpoint_path} rebuilds as run {config.run_id()}, but the CV "
            f"run recorded {summary['run_id']}"
        )
    entry = _fold_entry(fold_manifest, fold)
    if _normalised(config.split_manifest_path) != _normalised(entry["split_manifest_path"]):
        raise ValueError(
            f"fold {fold}: the checkpoint names {config.split_manifest_path}, but the fold "
            f"manifest names {entry['split_manifest_path']}"
        )
    for source, digest in (("checkpoint", checkpoint["config"].get("split_manifest_sha256")),
                           ("evaluation", results.get("split_manifest_sha256"))):
        if digest != entry["split_manifest_sha256"]:
            raise ValueError(
                f"fold {fold}: the {source} records split manifest sha256 "
                f"{str(digest)[:12]}, but the fold manifest records "
                f"{entry['split_manifest_sha256'][:12]}"
            )

    n_ambiguous_excluded = results.get("n_ambiguous_excluded", {})
    if HELD_OUT_SPLIT not in n_ambiguous_excluded:
        raise ValueError(
            f"fold {fold}: {results_path} does not record how many {HELD_OUT_SPLIT} tiles were "
            "ambiguous, so the ambiguous tiles cannot be checked against the run"
        )
    scoring_area_fraction = results.get("eval_min_oocyte_area_fraction")
    if scoring_area_fraction is None:
        scoring_area_fraction = config.min_oocyte_area_fraction
    return FoldRun(
        fold=fold,
        run_id=summary["run_id"],
        checkpoint_path=checkpoint_path,
        saved_predictions_path=predictions_path,
        config=config,
        label_rule=results.get("label_rule", config.label_rule),
        scoring_area_fraction=float(scoring_area_fraction),
        n_ambiguous_held_out=int(n_ambiguous_excluded[HELD_OUT_SPLIT]),
    )


def _normalised(path: str | Path) -> Path:
    return _resolve_repo_path(path).resolve()
