"""Flat specimen-grouped cross-validation over the tile classifier."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.modeling.specimen_groups import specimen_of
from src.modeling.tile_index import _resolve_repo_path

if TYPE_CHECKING:
    from src.modeling.train_tile_classifier import TrainingConfig

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FOLD_MANIFEST = "data/splits/fold_manifest.json"
DEFAULT_CV_OUTPUT_DIR = "data/tile_classifier_cv"
EVALUATION_DESIGN = "flat_grouped_{k}fold"
SPREAD_NOTE = "spread is a range over {k} non-independent folds, not a confidence interval"
PER_FOLD_METRICS = ("auprc", "roc_auc", "f1", "precision", "recall", "accuracy",
                    "balanced_accuracy", "mcc")


def load_fold_manifest(path: str | Path = DEFAULT_FOLD_MANIFEST) -> dict[str, Any]:
    """Read the fold manifest, refusing stale per-fold split files."""
    manifest_path = _resolve_repo_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for fold in manifest["folds"]:
        split_path = _resolve_repo_path(fold["split_manifest_path"])
        if not split_path.exists():
            raise ValueError(f"fold {fold['fold']}: {split_path} does not exist")
        digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
        if digest != fold["split_manifest_sha256"]:
            raise ValueError(
                f"fold {fold['fold']}: {fold['split_manifest_path']} does not match the "
                "hash recorded in the fold manifest; regenerate both together"
            )
    manifest["_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest["_path"] = _path_relative_to_repo(manifest_path)
    return manifest


def fold_config(
    base: "TrainingConfig", fold_manifest: dict[str, Any], fold: int
) -> "TrainingConfig":
    """Point the base configuration at one fold's split manifest."""
    from src.modeling.tile_index import DEFAULT_SPLIT_MANIFEST

    if base.split_manifest_path != DEFAULT_SPLIT_MANIFEST:
        raise ValueError(
            f"the base configuration already names {base.split_manifest_path}; "
            "cross-validation sets the split manifest per fold"
        )
    entry = _fold_entry(fold_manifest, fold)
    # load_fold_manifest already verified this file.
    return replace(base, split_manifest_path=entry["split_manifest_path"],
                   split_manifest_sha256=None)


def check_fold_isolation(
    index: pd.DataFrame,
    fold_manifest: dict[str, Any],
    fold: int,
    required_splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, list[str]]:
    """Check that each required split has exactly its assigned specimens."""
    entry = _fold_entry(fold_manifest, fold)
    held_out = set(entry["specimens"])
    inner_val = {f["inner_val_specimen"] for f in fold_manifest["folds"] if f["fold"] != fold}
    all_specimens = {s for f in fold_manifest["folds"] for s in f["specimens"]}
    expected = {
        "test": held_out,
        "val": inner_val,
        "train": all_specimens - held_out - inner_val,
    }
    unknown_splits = set(required_splits) - set(expected)
    if unknown_splits:
        raise ValueError(f"unknown required split(s): {sorted(unknown_splits)}")

    specimens = index["stem"].map(specimen_of)
    sides = pd.DataFrame({"specimen": specimens, "split": index["split"]}).drop_duplicates()
    per_specimen = sides.groupby("specimen")["split"].agg(lambda s: sorted(set(s)))
    straddling = {s: v for s, v in per_specimen.items() if len(v) > 1}
    if straddling:
        raise ValueError(f"fold {fold}: specimen(s) on more than one split: {straddling}")

    found = {
        split: sorted(sides.loc[sides["split"] == split, "specimen"])
        for split in sorted(sides["split"].unique())
    }
    for split in required_splits:
        actual = set(found.get(split, []))
        missing = sorted(expected[split] - actual)
        unexpected = sorted(actual - expected[split])
        if missing or unexpected:
            raise ValueError(
                f"fold {fold}: {split} specimens do not match the fold manifest; "
                f"missing {missing}, unexpected {unexpected}"
            )
    return found


def check_coverage(
    held_out_predictions: dict[int, pd.DataFrame], expected_tiles: int | None = None
) -> int:
    """Confirm each scored tile is held out exactly once."""
    ids = pd.concat([frame["tile_id"] for frame in held_out_predictions.values()],
                    ignore_index=True)
    duplicated = ids[ids.duplicated()]
    if len(duplicated):
        raise ValueError(
            f"{duplicated.nunique()} tile(s) held out by more than one fold, e.g. "
            f"{duplicated.iloc[0]}"
        )
    if expected_tiles is not None and len(ids) != expected_tiles:
        raise ValueError(
            f"{len(ids)} tiles held out across the folds, expected {expected_tiles}"
        )
    return int(len(ids))


def run_fold(
    base: "TrainingConfig",
    fold_manifest: dict[str, Any],
    fold: int,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Train and evaluate one fold, checking isolation before and after."""
    from src.modeling.evaluate_tile_classifier import evaluate
    from src.modeling.tile_index import load_tile_index
    from src.modeling.train_tile_classifier import input_px_for, train

    config = fold_config(base, fold_manifest, fold)
    # Reject an unsupported size before the index parse.
    input_px_for(config)
    index = load_tile_index(
        split_manifest_path=config.split_manifest_path,
        tile_sizes=(config.tile_size,),
        label_rule=config.label_rule,
        min_oocyte_area_fraction=config.min_oocyte_area_fraction,
    )
    found = check_fold_isolation(index, fold_manifest, fold)
    LOG.info("Fold %d isolated: test %s, val %s", fold, found.get("test"), found.get("val"))
    del index

    result = train(config)
    results = evaluate(
        _resolve_repo_path(config.output_dir) / result.run_id,
        bootstrap_samples=bootstrap_samples,
        num_workers=config.num_workers,
    )
    check_fold_isolation(
        pd.read_csv(_resolve_repo_path(results["predictions_path"])),
        fold_manifest,
        fold,
        required_splits=("val", "test"),
    )
    return results


def cv_run_id(base: "TrainingConfig", fold_manifest_sha256: str, length: int = 8) -> str:
    """Name a CV run by its config and fold manifest."""
    prefix = re.sub(r"_[0-9a-f]+$", "", base.run_id())
    payload = {
        key: value
        for key, value in sorted(asdict(base).items())
        if key not in base.RUN_ID_EXCLUDED
        and key not in ("split_manifest_path", "split_manifest_sha256")
    }
    payload["fold_manifest_sha256"] = fold_manifest_sha256
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return f"{prefix}_cv_{digest[:length]}"


def fold_summary(results: dict[str, Any], fold: int) -> dict[str, Any]:
    """Summarise one fold's held-out metrics and AUPRC lift."""
    test = results["metrics"]["test"]
    # Scored prevalence, so ambiguous tiles excluded.
    rate = test["trivial_baseline"]["positive_rate"]
    auprc = test["auprc"]
    defined = auprc is not None and rate not in (None, 0)
    summary = {
        "fold": fold,
        "run_id": results["run_id"],
        "n_tiles": test["n_tiles"],
        "n_positive": test["n_positive"],
        "n_slides": test["n_slides"],
        "n_specimens": test["n_specimens"],
        "positive_rate": rate,
        **{name: test.get(name) for name in PER_FOLD_METRICS},
        "auprc_lift": auprc / rate if defined else None,
        "undefined": not defined,
        "trivial_baseline": test["trivial_baseline"],
        "decision_threshold": results["decision_threshold"],
        "threshold_selection": results["threshold_selection"],
        "best_epoch": results.get("best_epoch"),
        "epochs_run": results.get("epochs_run"),
        "val_n_tiles": results["metrics"]["val"]["n_tiles"],
        "val_n_specimens": results["metrics"]["val"]["n_specimens"],
        "is_subset_run": results.get("is_subset_run"),
        "predictions_path": results["predictions_path"],
    }
    return summary


def summarise(values: list[float | None]) -> dict[str, Any]:
    """Summarise the defined values: mean, range and IQR."""
    defined = np.array([v for v in values if v is not None], dtype=float)
    if not len(defined):
        return {"mean": None, "min": None, "max": None, "iqr": [None, None], "n_defined": 0}
    return {
        "mean": float(defined.mean()),
        "min": float(defined.min()),
        "max": float(defined.max()),
        "iqr": [float(np.percentile(defined, 25)), float(np.percentile(defined, 75))],
        "n_defined": int(len(defined)),
    }


def aggregate(
    base: "TrainingConfig",
    fold_manifest: dict[str, Any],
    fold_results: dict[int, dict[str, Any]],
    *,
    allow_partial: bool = False,
    augmentation: str = "standard",
) -> dict[str, Any]:
    """Combine per-fold evaluations into the cross-validation result."""
    k = fold_manifest["n_folds"]
    expected = {f["fold"] for f in fold_manifest["folds"]}
    missing = sorted(expected - set(fold_results))
    if missing and not allow_partial:
        raise ValueError(f"results missing for fold(s) {missing}")

    held_out: dict[int, pd.DataFrame] = {}
    per_fold = []
    for fold in sorted(fold_results):
        results = fold_results[fold]
        entry = _fold_entry(fold_manifest, fold)
        if results.get("split_manifest_sha256") != entry["split_manifest_sha256"]:
            raise ValueError(
                f"fold {fold} was evaluated against a split manifest other than the one "
                "this fold manifest records"
            )
        predictions = pd.read_csv(_resolve_repo_path(results["predictions_path"]))
        check_fold_isolation(
            predictions, fold_manifest, fold, required_splits=("val", "test")
        )
        held_out[fold] = predictions.loc[predictions["split"] == "test"]
        per_fold.append(fold_summary(results, fold))

    expected_tiles = None if missing else _scorable_tiles(fold_manifest, base)
    n_held_out = check_coverage(held_out, expected_tiles)

    return {
        "cv_run_id": cv_run_id(base, fold_manifest["_sha256"]),
        "evaluation_design": EVALUATION_DESIGN.format(k=k),
        "fold_manifest_path": fold_manifest["_path"],
        "fold_manifest_version": fold_manifest["manifest_version"],
        "fold_manifest_sha256": fold_manifest["_sha256"],
        "config": {key: value for key, value in asdict(base).items()
                   if key not in ("split_manifest_path", "split_manifest_sha256")},
        "augmentation": augmentation,
        "is_partial": bool(missing),
        "is_subset_run": any(bool(f["is_subset_run"]) for f in per_fold),
        "folds_missing": missing,
        "n_tiles_held_out": n_held_out,
        "per_fold": per_fold,
        "summary": {
            "auprc_lift": summarise([f["auprc_lift"] for f in per_fold]),
            "auprc": summarise([f["auprc"] for f in per_fold]),
            "roc_auc": summarise([f["roc_auc"] for f in per_fold]),
            "f1": summarise([f["f1"] for f in per_fold]),
            "note": SPREAD_NOTE.format(k=len(per_fold)),
        },
    }


def write_cv_results(results: dict[str, Any], output_dir: str | Path = DEFAULT_CV_OUTPUT_DIR) -> Path:
    """Write ``{output_dir}/{cv_run_id}/cv_results.json`` and return its path."""
    path = _resolve_repo_path(output_dir) / results["cv_run_id"] / "cv_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _scorable_tiles(fold_manifest: dict[str, Any], base: "TrainingConfig") -> int | None:
    """Count the scorable tiles the folds should hold out."""
    from src.modeling.tile_index import LABEL_RULE_AREA
    from src.modeling.tile_labels import DEFAULT_MIN_OOCYTE_AREA_FRACTION

    if (base.label_rule, base.min_oocyte_area_fraction) != (
        LABEL_RULE_AREA, DEFAULT_MIN_OOCYTE_AREA_FRACTION
    ):
        # Manifest counts assume the default labelling.
        return None
    key = str(base.tile_size)
    expected = 0
    for fold in fold_manifest["folds"]:
        if key not in fold["tiles"] or key not in fold.get("ambiguous", {}):
            return None
        expected += fold["tiles"][key] - fold["ambiguous"][key]
    return expected


def _fold_entry(fold_manifest: dict[str, Any], fold: int) -> dict[str, Any]:
    for entry in fold_manifest["folds"]:
        if entry["fold"] == fold:
            return entry
    raise ValueError(f"fold {fold} is not in the fold manifest")


def _path_relative_to_repo(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
