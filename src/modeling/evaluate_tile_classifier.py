"""Evaluate a trained tile classifier and write its results and per-tile predictions.

Two things here are deliberate and worth reading before changing them.

**Average precision is the headline metric, not accuracy.** At this corpus's positive
rates a classifier that answers "negative" to everything scores about 85% accuracy at
512 px and 80% at 1024 px. Every accuracy is therefore reported next to that
all-negative floor, so it cannot be read as skill on its own.

**The operating point is chosen on validation, never on test.** Threshold-bound metrics --
precision, recall, F1, balanced accuracy, MCC -- all move with where the probability cut
falls, and 0.5 is not a neutral place to put it here: the sampler forces 25% positives
per training batch while val and test run at their natural ~17% and ~11%, so the model's
probabilities are on a different scale from the data it is scored on by construction. The
threshold that maximises validation F1 is selected and then applied unchanged to test, so
test never picks its own operating point and the four runs of this milestone stay
comparable. The threshold used is recorded.

**Confidence intervals resample specimens, not tiles or even slides.** Tiles overlap by
20% and one oocyte spans several, so tiles are not independent draws. Nor are slides: a
specimen is ``(Site, Season, Polyp No)`` and one specimen yields several slides that are
consecutive serial sections through the same polyp -- ``CHN_SP_5`` contributes four. The
independent unit is therefore the specimen, obtained by dropping the cut range from the
stem. On the current 19/4/3 split this changes nothing, because val and test happen to
hold one slide per specimen; it matters for the grouped cross-validation that follows this
milestone, where a fold will contain several slides of one specimen and resampling slides
would treat serial sections as independent evidence. The resulting intervals are wide --
that is the honest width, and narrowing it would assume an independence the data lacks.

The per-tile prediction log exists so the reported metrics can be re-derived from disk
rather than taken from this module's own summary.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.modeling.encoders import DEFAULT_ENCODER, encoder_spec, load_encoder
from src.modeling.tile_classification_dataset import TileClassificationDataset
from src.modeling.tile_index import (
    LABEL_AMBIGUOUS,
    binary_targets,
    load_tile_index,
    training_rows,
)
from src.modeling.train_tile_classifier import (
    ARCHITECTURE_FROZEN_ENCODER,
    TrainingConfig,
    _logits,
    _nan_to_none,
    _path_relative_to_repo,
    build_model,
    build_transforms,
    input_px_for,
    normalisation_for,
    seed_everything,
)

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPLIT_MANIFEST = "data/splits/split_manifest.json"
EVALUATION_SPLITS = ("val", "test")
DEFAULT_BOOTSTRAP_SAMPLES = 2000
#: Only a fallback. The operating point is normally selected on validation; this is what
#: is used when that selection cannot run, e.g. a single-class validation split.
FALLBACK_DECISION_THRESHOLD = 0.5
#: Cap on candidate thresholds considered during selection. Candidates come from the
#: observed probabilities, so this only bounds the work on a large split.
MAX_THRESHOLD_CANDIDATES = 512

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


PREDICTION_COLUMNS = (
    "tile_id",
    "stem",
    "cut_name",
    "split",
    "tile_size",
    "label",
    "oocyte_area_fraction",
    "prob",
)


def evaluate(
    run_dir: str | Path,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    decision_threshold: float | None = None,
    num_workers: int = 4,
) -> dict[str, Any]:
    """Score a finished run's checkpoint on val and test, and write its artifacts.

    Reconstructs the configuration from the checkpoint rather than taking it as
    arguments, so the evaluation cannot silently use a different tile size, label rule or
    threshold than the model was trained under.

    Parameters
    ----------
    run_dir : str or Path
        A training run's output directory, holding ``checkpoint.pt``.
    bootstrap_samples : int, optional
        Slide-level resamples for the confidence intervals.
    decision_threshold : float, optional
        Probability at or above which a tile is predicted positive. Left as None -- the
        normal case -- it is chosen to maximise F1 on the validation split and then
        applied unchanged to test, so test never selects its own operating point. Pass a
        value only to score an externally chosen cut. Either way the threshold used and
        how it was arrived at are both recorded.
    num_workers : int, optional
        Dataloader workers.

    Returns
    -------
    dict
        The results payload, also written to ``results.json``.

    Raises
    ------
    FileNotFoundError
        If the run directory holds no checkpoint.
    """
    if bootstrap_samples < 1:
        raise ValueError(
            f"bootstrap_samples must be at least 1, got {bootstrap_samples}; zero would "
            "complete with every confidence interval null while recording the count"
        )
    run_path = _resolve_repo_path(run_dir)
    checkpoint_path = run_path / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} does not exist; evaluation needs a finished training run"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainingConfig(**checkpoint["config"])
    seed_everything(config.seed)
    LOG.info(
        "Evaluating %s (%s)",
        config.run_id(),
        "threshold selected on validation"
        if decision_threshold is None
        else f"caller-supplied threshold {decision_threshold:.4f}",
    )

    model = _restore_model(config, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    index = load_tile_index(
        tile_sizes=(config.tile_size,),
        label_rule=config.label_rule,
        min_oocyte_area_fraction=config.min_oocyte_area_fraction,
    )
    ambiguous = _ambiguous_counts(index)
    scored = training_rows(index)

    # Score every split first, then fix the operating point, then measure. The order
    # matters: the threshold has to come from validation alone, and both splits have to
    # be measured at that same one.
    scored_splits: dict[str, pd.DataFrame] = {}
    for split in EVALUATION_SPLITS:
        frame = scored.loc[scored["split"] == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"the {split} split has no tiles at {config.tile_size} px")
        scored_splits[split] = frame.assign(
            prob=_predict(model, frame, config, device, num_workers)
        )

    validation = scored_splits["val"]
    if decision_threshold is None:
        threshold, selection = select_threshold(
            binary_targets(validation).to_numpy(), validation["prob"].to_numpy()
        )
    else:
        threshold, selection = float(decision_threshold), "caller_supplied"

    metrics: dict[str, Any] = {"train": _train_summary(scored, config)}
    for split, frame in scored_splits.items():
        metrics[split] = _split_metrics(
            frame, threshold, bootstrap_samples, config.seed
        )
    predictions = list(scored_splits.values())

    predictions_path = run_path / "predictions.csv"
    _write_predictions(predictions_path, pd.concat(predictions, ignore_index=True))

    results = _build_results(
        config=config,
        checkpoint=checkpoint,
        metrics=metrics,
        ambiguous=ambiguous,
        decision_threshold=threshold,
        threshold_selection=selection,
        bootstrap_samples=bootstrap_samples,
        run_path=run_path,
        checkpoint_path=checkpoint_path,
        predictions_path=predictions_path,
    )
    results = _nan_to_none(results)
    # Sanitise before returning, not only before writing: _nan_to_none builds a copy, so
    # returning the raw payload would let the caller print "nan" while results.json
    # correctly said null.
    _write_results(run_path / "results.json", results)
    return results


def _restore_model(config: TrainingConfig, checkpoint: dict[str, Any]) -> torch.nn.Module:
    """Rebuild the trained model from a checkpoint, head-only or whole.

    The frozen arm stores only its head, so the backbone is reloaded from the local
    HuggingFace cache and the head weights are placed onto it. ``state_dict_scope`` says
    which of the two a checkpoint holds.
    """
    scope = checkpoint.get("state_dict_scope", "model")
    state = checkpoint["model_state_dict"]
    if scope == "head":
        model = load_encoder(
            encoder_name=config.encoder_name or DEFAULT_ENCODER,
            tile_size=config.tile_size,
            hidden_dim=config.head_hidden_dim,
            dropout=config.head_dropout,
        )
        model.head.load_state_dict(state)
        return model
    # pretrained=False: the checkpoint carries every parameter, so fetching ImageNet
    # weights only to overwrite them costs a download that fails outright on a node
    # with no cache and no egress.
    model = build_model(config, pretrained=False)
    model.load_state_dict(state)
    return model


def _predict(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    config: TrainingConfig,
    device: torch.device,
    num_workers: int,
) -> np.ndarray:
    """Predicted positive probability per row, in the frame's own order."""
    mean, std = normalisation_for(config)
    dataset = TileClassificationDataset(
        frame,
        transform=build_transforms(
            train=False, input_px=input_px_for(config), mean=mean, std=std
        ),
    )
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers
    )
    probabilities: list[float] = []
    with torch.no_grad():
        for images, _ in loader:
            logits = _logits(model, images.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return np.asarray(probabilities, dtype=float)


def select_threshold(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, str]:
    """Pick the probability cut that maximises F1 on the validation split.

    Returns the threshold and how it was chosen, both of which are recorded: a reader of
    the results has to be able to tell a selected operating point from a defaulted one.

    Ties go to the lower threshold, which favours recall -- missing an oocyte is the more
    costly error for a screening stage feeding detection.
    """
    from sklearn.metrics import f1_score

    if len(np.unique(targets)) < 2:
        LOG.warning(
            "Validation split holds one class; falling back to a %.2f threshold",
            FALLBACK_DECISION_THRESHOLD,
        )
        return FALLBACK_DECISION_THRESHOLD, "fallback_single_class"

    # Candidates are the observed probabilities, not a fixed grid. F1 only changes
    # where the cut crosses an actual prediction, so an evenly spaced grid can step
    # straight over a narrow optimum: with one positive at 0.504 and one negative at
    # 0.501, every point of a 199-step grid scores both the same way and the best F1
    # found is 0.667, where cutting at 0.504 gives 1.0.
    candidates = np.unique(probabilities)
    if len(candidates) > MAX_THRESHOLD_CANDIDATES:
        # Quantiles keep the candidates where the predictions actually are, which an
        # evenly spaced grid does not.
        candidates = np.unique(
            np.quantile(candidates, np.linspace(0.0, 1.0, MAX_THRESHOLD_CANDIDATES))
        )
    scores = [f1_score(targets, (probabilities >= t).astype(int), zero_division=0)
              for t in candidates]
    best = int(np.argmax(scores))
    LOG.info(
        "Validation-selected threshold %.6f (F1 %.4f) from %d candidates; "
        "0.50 would give F1 %.4f",
        candidates[best],
        scores[best],
        len(candidates),
        f1_score(targets, (probabilities >= 0.5).astype(int), zero_division=0),
    )
    return float(candidates[best]), "max_val_f1"


def _split_metrics(
    frame: pd.DataFrame,
    threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Point metrics plus slide-clustered confidence intervals for one split."""
    targets = binary_targets(frame).to_numpy()
    probabilities = frame["prob"].to_numpy()
    point = _point_metrics(targets, probabilities, threshold)
    positive_rate = float(targets.mean()) if len(targets) else float("nan")
    return {
        "n_tiles": int(len(frame)),
        "n_positive": int(targets.sum()),
        "n_slides": int(frame["stem"].nunique()),
        # The interval rests on this, not on n_slides: they coincide on the current
        # split but will not under grouped cross-validation.
        "n_specimens": int(frame["stem"].map(specimen_of).nunique()),
        **point,
        "trivial_baseline": {
            "all_negative_accuracy": 1.0 - positive_rate,
            "positive_rate": positive_rate,
        },
        "confidence_interval_95": _bootstrap_intervals(
            frame, targets, probabilities, threshold, bootstrap_samples, seed
        ),
    }


def _point_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Average precision, ROC AUC and the threshold-dependent metrics.

    Returns NaN throughout when the split holds a single class, where none of these are
    defined. That is reported rather than raised so the rest of the evaluation still
    completes and the gap is visible in the results file.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if len(np.unique(targets)) < 2:
        LOG.warning("Split holds one class only; metrics are undefined")
        nan = float("nan")
        return {
            "auprc": nan, "roc_auc": nan, "precision": nan, "recall": nan,
            "f1": nan, "accuracy": nan, "balanced_accuracy": nan, "mcc": nan,
        }

    predicted = (probabilities >= threshold).astype(int)
    return {
        # Threshold-free: these describe the ranking and do not move with the cut.
        "auprc": float(average_precision_score(targets, probabilities)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        # Threshold-bound: all of these move with the operating point.
        "precision": float(precision_score(targets, predicted, zero_division=0)),
        "recall": float(recall_score(targets, predicted, zero_division=0)),
        "f1": float(f1_score(targets, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(targets, predicted)),
        # Balanced accuracy averages per-class recall and MCC accounts for all four
        # cells of the confusion matrix, so neither is flattered by predicting the
        # majority class the way raw accuracy is at these positive rates.
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "mcc": float(matthews_corrcoef(targets, predicted)),
    }


def _bootstrap_intervals(
    frame: pd.DataFrame,
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, list[float | None]]:
    """95% intervals for AUPRC, F1 and recall, resampling whole specimens.

    The specimen is the independent unit, not the tile and not the slide: tiles overlap
    by 20% and one oocyte spans several, while several slides can be serial sections
    through one polyp. Resampling anything finer treats correlated rows as independent
    evidence and reports intervals narrower than the data supports. Resamples that end
    up single-class are skipped rather than counted, since the metrics are undefined
    there.
    """
    specimens = np.array([specimen_of(stem) for stem in frame["stem"]])
    unique_specimens = np.unique(specimens)
    rng = np.random.default_rng(seed)
    rows_by_specimen = {
        specimen: np.flatnonzero(specimens == specimen) for specimen in unique_specimens
    }

    collected: dict[str, list[float]] = {"auprc": [], "f1": [], "recall": []}
    for _ in range(bootstrap_samples):
        drawn = rng.choice(
            unique_specimens, size=len(unique_specimens), replace=True
        )
        rows = np.concatenate([rows_by_specimen[specimen] for specimen in drawn])
        sample_targets = targets[rows]
        if len(np.unique(sample_targets)) < 2:
            continue
        sample_metrics = _point_metrics(sample_targets, probabilities[rows], threshold)
        for name in collected:
            collected[name].append(sample_metrics[name])

    intervals: dict[str, list[float | None]] = {}
    for name, values in collected.items():
        if len(values) < 2:
            LOG.warning(
                "Too few usable resamples for a %s interval (%d of %d were single-class)",
                name,
                bootstrap_samples - len(values),
                bootstrap_samples,
            )
            intervals[name] = [None, None]
        else:
            low, high = np.percentile(values, [2.5, 97.5])
            intervals[name] = [float(low), float(high)]
    return intervals


def _train_summary(scored: pd.DataFrame, config: TrainingConfig) -> dict[str, Any]:
    """Sizes of the train split the model was actually trained on.

    Honours ``max_train_slides``: a smoke-test run saw a subset, and counting the whole
    split here would report tiles and positives from slides the model never met, with
    nothing in the file marking the discrepancy. The subset itself is recorded so the
    results are self-describing.
    """
    frame = scored.loc[scored["split"] == "train"]
    if config.max_train_slides is not None:
        keep = sorted(frame["stem"].unique())[: config.max_train_slides]
        frame = frame.loc[frame["stem"].isin(keep)]
    return {
        "n_tiles": int(len(frame)),
        "n_positive": int(binary_targets(frame).sum()) if len(frame) else 0,
        "n_slides": int(frame["stem"].nunique()),
        # The interval rests on this, not on n_slides: they coincide on the current
        # split but will not under grouped cross-validation.
        "n_specimens": int(frame["stem"].map(specimen_of).nunique()),
        "max_train_slides": config.max_train_slides,
    }


def _ambiguous_counts(index: pd.DataFrame) -> dict[str, int]:
    """Tiles the label rule declined to call, per split, before they were dropped."""
    ambiguous = index.loc[index["label"] == LABEL_AMBIGUOUS]
    counts = ambiguous.groupby("split").size().to_dict()
    return {split: int(counts.get(split, 0)) for split in ("train", "val", "test")}


def _observed_positive_fraction(run_path: Path) -> float | None:
    """Realised batch positive fraction, read back from the composition log.

    Verification compares this against the configured fraction, so it must come from what
    the sampler actually delivered rather than from the flag that requested it.
    """
    log_path = run_path / "batch_composition.jsonl"
    if not log_path.exists():
        LOG.warning("%s is missing; cannot report the realised positive fraction", log_path)
        return None
    positives = total = 0
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            positives += record["n_positive"]
            total += record["batch_size"]
    return positives / total if total else None


def _build_results(
    config: TrainingConfig,
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
    ambiguous: dict[str, int],
    decision_threshold: float,
    threshold_selection: str,
    bootstrap_samples: int,
    run_path: Path,
    checkpoint_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    """Assemble the results payload."""
    manifest_path = _resolve_repo_path(DEFAULT_SPLIT_MANIFEST)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "run_id": config.run_id(),
        "architecture": config.architecture,
        "encoder_name": config.encoder_name,
        "encoder_licence": (
            encoder_spec(config.encoder_name or DEFAULT_ENCODER).licence
            if config.architecture == ARCHITECTURE_FROZEN_ENCODER
            else None
        ),
        "tile_size": config.tile_size,
        "seed": config.seed,
        "label_rule": config.label_rule,
        "min_oocyte_area_fraction": config.min_oocyte_area_fraction,
        "n_ambiguous_excluded": ambiguous,
        "decision_threshold": decision_threshold,
        "threshold_selection": threshold_selection,
        "positive_fraction_per_batch": config.positive_fraction,
        "positive_fraction_observed": _observed_positive_fraction(run_path),
        "hard_negative_mining": config.hard_negative_mining,
        "warmup_epochs": config.warmup_epochs,
        "hard_negative_pool_fraction": config.hard_negative_pool_fraction,
        "hard_negative_share": config.hard_negative_share,
        # From the training log, not the checkpoint. The checkpoint is only written on
        # an improving epoch, so its "epoch" is the best one -- reporting that as
        # epochs_run understates an early-stopped run by however long it ran after its
        # peak, and the two fields would never disagree, hiding the error.
        **_epoch_summary(run_path, checkpoint),
        "hyperparameters": {
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "head_hidden_dim": config.head_hidden_dim,
            "head_dropout": config.head_dropout,
        },
        "bootstrap_samples": bootstrap_samples,
        # Recorded because they change what a run means: a capped run is a smoke test,
        # not a result, and its results file has to say so on its own.
        "subset_caps": {
            "max_train_slides": config.max_train_slides,
            "max_batches_per_epoch": config.max_batches_per_epoch,
            "max_val_tiles": config.max_val_tiles,
        },
        "is_subset_run": any(
            value is not None
            for value in (
                config.max_train_slides,
                config.max_batches_per_epoch,
                config.max_val_tiles,
            )
        ),
        "split_manifest_version": manifest.get("manifest_version"),
        "split_manifest_sha256": _sha256(manifest_path),
        "tiling_provenance": _tiling_provenance(),
        "metrics": metrics,
        "checkpoint_path": _path_relative_to_repo(checkpoint_path),
        "predictions_path": _path_relative_to_repo(predictions_path),
        "batch_composition_log_path": _path_relative_to_repo(
            run_path / "batch_composition.jsonl"
        ),
    }


def _epoch_summary(run_path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """How long the run actually went, read from the training log beside the checkpoint.

    Falls back to the checkpoint's own epoch when no log is present, flagging that the
    count is the best epoch rather than the run length so the results file never asserts
    something it cannot know.
    """
    log_path = run_path / "training_log.json"
    if not log_path.exists():
        LOG.warning("%s is missing; epoch counts fall back to the checkpoint", log_path)
        return {
            "epochs_run": None,
            "best_epoch": checkpoint.get("epoch"),
            "early_stopped": None,
            "early_stopped_epoch": None,
        }
    log = json.loads(log_path.read_text(encoding="utf-8"))
    early_stopped = bool(log.get("early_stopped"))
    return {
        "epochs_run": log.get("epochs_run"),
        "best_epoch": log.get("best_epoch"),
        "early_stopped": early_stopped,
        "early_stopped_epoch": log.get("epochs_run") if early_stopped else None,
    }


def _tiling_provenance() -> dict[str, float | None]:
    """Tiling settings the corpus was generated with, read from one manifest.

    Recorded because the tissue-fraction threshold was a live source of confusion in this
    project; a results file that does not name it cannot be compared against another.
    """
    manifests = sorted(
        (_resolve_repo_path("data/tiles")).glob("*/*/*_tile_manifest.json")
    )
    if not manifests:
        LOG.warning("No tile manifests found; tiling provenance will be null")
        return {"min_tissue_fraction": None, "overlap": None}
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    return {
        "min_tissue_fraction": manifest.get("min_tissue_fraction"),
        "overlap": manifest.get("overlap"),
    }


def _write_predictions(path: Path, frame: pd.DataFrame) -> None:
    """Write the per-tile prediction log the verification pass re-derives metrics from.

    CSV rather than parquet: the log is a few thousand rows per run, so the size saving
    would be irrelevant, and being readable without a library matters more for an
    artifact whose whole purpose is letting someone else check this module's arithmetic.
    """
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame is missing {missing}")
    frame[list(PREDICTION_COLUMNS)].to_csv(path, index=False)
    LOG.info("Wrote %s (%d rows)", path, len(frame))


def _write_results(path: Path, results: dict[str, Any]) -> None:
    """Write results.json. Undefined metrics must already be None, not NaN."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")
    LOG.info("Wrote %s", path)


def _sha256(path: Path) -> str:
    """Content hash, so a split can be compared rather than asserted."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate
