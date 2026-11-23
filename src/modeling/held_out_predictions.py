"""Every held-out tile of a CV run; saved predictions omit the ambiguous band."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.modeling.cross_validation import (
    _fold_entry,
    _path_relative_to_repo,
    check_coverage,
    check_fold_isolation,
    load_fold_manifest,
)
from src.modeling.embedding_cache import _swap_into_place
from src.modeling.evaluate_tile_classifier import _predict, _restore_model
from src.modeling.tile_index import _resolve_repo_path, load_tile_index
from src.modeling.tile_labels import LABEL_NEGATIVE
from src.modeling.train_tile_classifier import TrainingConfig, load_config_cache

HELD_OUT_SPLIT = "test"
DEFAULT_OUTPUT_ROOT = "data/tile_classifier_cv/held_out_predictions"
# TF32 kernel choice varies with batch size.
REPRODUCTION_TOLERANCE = 1e-3
OUTPUT_COLUMNS = ("tile_id", "stem", "cut_name", "tile_size", "fold", "run_id", "label",
                  "oocyte_area_fraction", "is_ambiguous", "prob", "prob_source")
PROB_SAVED = "saved"
PROB_RECOMPUTED = "recomputed"
# Coverage round-trips through CSV as decimal text.
COVERAGE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FoldRun:
    """One fold model of a CV run and the files it was evaluated from."""

    fold: int
    run_id: str
    checkpoint_path: Path
    saved_predictions_path: Path
    config: TrainingConfig
    label_rule: str
    scoring_area_fraction: float
    n_ambiguous_held_out: int
    checkpoint_sha256: str
    results_sha256: str
    saved_predictions_sha256: str
    saved_held_out: pd.DataFrame = field(compare=False, repr=False)


@dataclass(frozen=True)
class CvRun:
    """A finished CV run, checked against the fold manifest it was built on."""

    cv_results_path: Path
    cv_results_sha256: str
    cv_results: dict[str, Any]
    fold_manifest: dict[str, Any]
    folds: tuple[FoldRun, ...]


def load_cv_run(cv_results_path: str | Path) -> CvRun:
    """Read a CV run and each fold's model, refusing anything that has drifted."""
    path = _resolve_repo_path(cv_results_path)
    data = path.read_bytes()
    cv_results = json.loads(data.decode("utf-8"))
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
    return CvRun(path, hashlib.sha256(data).hexdigest(), cv_results, fold_manifest, folds)


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
    saved = fold_run.saved_held_out
    saved_ids = set(saved["tile_id"])
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
    scored = saved.merge(
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


def predict_fold(fold_run: FoldRun, held_out: pd.DataFrame, device: torch.device,
                 num_workers: int) -> np.ndarray:
    """The fold model's probability for each held-out tile, in the frame's order."""
    config = fold_run.config
    checkpoint, _ = _load_checkpoint(fold_run.checkpoint_path, fold_run.checkpoint_sha256)
    cache = load_config_cache(config) if config.embedding_cache_dir else None
    model = _restore_model(config, checkpoint, cache).to(device).eval()
    # Dataset rejects ambiguous labels; targets unused.
    servable = held_out.assign(label=LABEL_NEGATIVE, is_ambiguous=False)
    return _predict(model, servable, config, device, num_workers, cache)


def saved_held_out_probabilities(fold_run: FoldRun) -> pd.Series:
    return fold_run.saved_held_out.set_index("tile_id")["prob"]


def check_reproduces_saved_predictions(fold_run: FoldRun, held_out: pd.DataFrame,
                                       probabilities: np.ndarray) -> float:
    """Refuse probabilities that differ from the run's saved ones; return the largest gap."""
    saved = saved_held_out_probabilities(fold_run).reset_index()
    if saved.empty:
        raise ValueError(
            f"fold {fold_run.fold}: the run scored no held-out tiles, so there is nothing to "
            "check its model against"
        )
    if saved["prob"].isna().any():
        raise ValueError(
            f"fold {fold_run.fold}: {int(saved['prob'].isna().sum())} saved held-out "
            "probabilities are blank"
        )
    now = pd.DataFrame({"tile_id": held_out["tile_id"].to_numpy(), "prob_now": probabilities})
    compared = saved.merge(now, on="tile_id", how="left", validate="one_to_one")
    if compared["prob_now"].isna().any():
        raise ValueError(f"fold {fold_run.fold}: some saved tiles were not predicted")
    largest = float((compared["prob"] - compared["prob_now"]).abs().max())
    if not largest <= REPRODUCTION_TOLERANCE:
        raise ValueError(
            f"fold {fold_run.fold}: probabilities differ from the run's saved ones by up to "
            f"{largest:.2e} (tolerance {REPRODUCTION_TOLERANCE:.0e}). Predict on the kind of "
            "device the run was evaluated on: GPU convolutions run in TF32, and scoring a "
            "GPU-evaluated run on CPU differs by up to about 5e-3. Otherwise this is not the "
            "model or input pipeline the run was evaluated with"
        )
    return largest


def predict_held_out_tiles(cv_run: CvRun, held_out: dict[int, pd.DataFrame],
                           device: torch.device, num_workers: int
                           ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Score every held-out tile with its fold's model, checked against the saved run."""
    frames = []
    records = []
    for fold_run in cv_run.folds:
        frame = held_out[fold_run.fold]
        probabilities = predict_fold(fold_run, frame, device, num_workers)
        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"fold {fold_run.fold}: the model returned "
                f"{int((~np.isfinite(probabilities)).sum())} non-finite probabilities"
            )
        largest = check_reproduces_saved_predictions(fold_run, frame, probabilities)
        saved = frame["tile_id"].map(saved_held_out_probabilities(fold_run))
        frames.append(frame.assign(
            fold=fold_run.fold, run_id=fold_run.run_id,
            prob=saved.fillna(pd.Series(probabilities, index=frame.index)),
            prob_source=np.where(saved.notna(), PROB_SAVED, PROB_RECOMPUTED),
        ))
        records.append({
            "fold": fold_run.fold,
            "run_id": fold_run.run_id,
            "checkpoint_path": _path_relative_to_repo(fold_run.checkpoint_path),
            "checkpoint_sha256": fold_run.checkpoint_sha256,
            "results_sha256": fold_run.results_sha256,
            "saved_predictions_path": _path_relative_to_repo(fold_run.saved_predictions_path),
            "saved_predictions_sha256": fold_run.saved_predictions_sha256,
            "split_manifest_path": fold_run.config.split_manifest_path,
            "split_manifest_sha256": fold_run.config.split_manifest_sha256,
            "label_rule": fold_run.label_rule,
            "scoring_area_fraction": fold_run.scoring_area_fraction,
            "n_tiles": int(len(frame)),
            "n_ambiguous": int(frame["is_ambiguous"].sum()),
            "max_reproduction_difference": largest,
        })
    predictions = pd.concat(frames, ignore_index=True).loc[:, list(OUTPUT_COLUMNS)]
    return predictions, records


def output_dir_for(cv_run: CvRun, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    cv_run_id = str(cv_run.cv_results.get("cv_run_id", ""))
    if not cv_run_id or Path(cv_run_id).name != cv_run_id or cv_run_id in (".", ".."):
        raise ValueError(f"{cv_run.cv_results_path} has cv_run_id {cv_run_id!r}, not a plain name")
    return _resolve_repo_path(output_root) / cv_run_id


def write_held_out_predictions(cv_run: CvRun, predictions: pd.DataFrame,
                               fold_records: list[dict[str, Any]], device: str,
                               output_root: str | Path = DEFAULT_OUTPUT_ROOT,
                               overwrite: bool = False) -> Path:
    """Write ``predictions.csv`` and ``provenance.json`` through a staging directory."""
    out_dir = output_dir_for(cv_run, output_root)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(f".{out_dir.name}.staging-{uuid.uuid4().hex}")
    # mkdir, unlike mkdtemp, keeps the group-readable umask.
    staging.mkdir()
    try:
        predictions.to_csv(staging / "predictions.csv", index=False)
        provenance = {
            "cv_run_id": cv_run.cv_results["cv_run_id"],
            "cv_results_path": _path_relative_to_repo(cv_run.cv_results_path),
            "cv_results_sha256": cv_run.cv_results_sha256,
            "fold_manifest_path": cv_run.cv_results["fold_manifest_path"],
            "fold_manifest_sha256": cv_run.cv_results["fold_manifest_sha256"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": device,
            "device_details": _device_details(device),
            "reproduction_tolerance": REPRODUCTION_TOLERANCE,
            "n_tiles": int(len(predictions)),
            "n_ambiguous": int(predictions["is_ambiguous"].sum()),
            "folds": fold_records,
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if overwrite:
            _swap_into_place(staging, out_dir)
        else:
            _refuse_existing(out_dir, overwrite)
            # rename refuses a non-empty target directory.
            os.rename(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return out_dir


def run_held_out_predictions(cv_results_path: str | Path,
                             output_root: str | Path = DEFAULT_OUTPUT_ROOT,
                             device: str = "cuda", num_workers: int = 0,
                             overwrite: bool = False) -> Path:
    """Check, predict and write every held-out tile of one CV run."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "no GPU is available. A run evaluated on GPU fails the reproduction check when "
            "scored on CPU; pass device='cpu' only for a run evaluated on CPU"
        )
    cv_run = load_cv_run(cv_results_path)
    _refuse_existing(output_dir_for(cv_run, output_root), overwrite)
    held_out = collect_held_out_tiles(cv_run)
    predictions, records = predict_held_out_tiles(cv_run, held_out, torch_device, num_workers)
    return write_held_out_predictions(cv_run, predictions, records, str(torch_device),
                                      output_root, overwrite)


def _load_checkpoint(path: Path, expected_sha256: str | None = None
                     ) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{path} has changed since its run was checked")
    # Primitives-only config allows weights_only loading.
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True), digest


def _device_details(device: str) -> dict[str, Any]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {}
    return {
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def _refuse_existing(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} already exists; pass overwrite to replace it")


def _load_fold_run(summary: dict[str, Any], fold_manifest: dict[str, Any]) -> FoldRun:
    fold = int(summary["fold"])
    predictions_path = _resolve_repo_path(summary["predictions_path"])
    results_path = predictions_path.with_name(
        predictions_path.name.replace("predictions", "results", 1)
    ).with_suffix(".json")
    results_bytes = results_path.read_bytes()
    results = json.loads(results_bytes.decode("utf-8"))
    if results.get("run_id") != summary["run_id"] or _normalised(
        results.get("predictions_path", "")
    ) != _normalised(predictions_path):
        raise ValueError(
            f"fold {fold}: {results_path} describes run {results.get('run_id')} with "
            f"predictions {results.get('predictions_path')}, not the run and predictions "
            "the CV run recorded"
        )
    checkpoint_path = _resolve_repo_path(results["checkpoint_path"])
    checkpoint, checkpoint_sha256 = _load_checkpoint(checkpoint_path)
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
    saved_bytes = predictions_path.read_bytes()
    saved = pd.read_csv(io.BytesIO(saved_bytes))
    saved_held_out = saved.loc[saved["split"] == HELD_OUT_SPLIT].reset_index(drop=True)
    duplicated = saved_held_out.loc[saved_held_out["tile_id"].duplicated(), "tile_id"]
    if len(duplicated):
        raise ValueError(
            f"fold {fold}: {predictions_path} lists {duplicated.nunique()} held-out tile(s) more "
            f"than once, e.g. {sorted(duplicated)[:3]}"
        )
    return FoldRun(
        fold=fold,
        run_id=summary["run_id"],
        checkpoint_path=checkpoint_path,
        saved_predictions_path=predictions_path,
        config=config,
        label_rule=results.get("label_rule", config.label_rule),
        scoring_area_fraction=float(scoring_area_fraction),
        n_ambiguous_held_out=int(n_ambiguous_excluded[HELD_OUT_SPLIT]),
        checkpoint_sha256=checkpoint_sha256,
        results_sha256=hashlib.sha256(results_bytes).hexdigest(),
        saved_predictions_sha256=hashlib.sha256(saved_bytes).hexdigest(),
        saved_held_out=saved_held_out,
    )


def _normalised(path: str | Path) -> Path:
    return _resolve_repo_path(path).resolve()
