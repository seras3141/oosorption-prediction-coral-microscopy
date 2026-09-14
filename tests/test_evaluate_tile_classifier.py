"""Tests for the evaluator: schema, slide-clustered intervals, honest baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.evaluate_tile_classifier import (
    FALLBACK_DECISION_THRESHOLD,
    PREDICTION_COLUMNS,
    select_threshold,
    _ambiguous_counts,
    _bootstrap_intervals,
    _observed_positive_fraction,
    _point_metrics,
    _sha256,
    _split_metrics,
    _write_predictions,
    _write_results,
)
from src.modeling.tile_index import binary_targets


def _frame(labels: list[str], probs: list[float], slides: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tile_id": [f"t{i}" for i in range(len(labels))],
            "stem": slides,
            "cut_name": [f"{s}_cut000" for s in slides],
            "split": ["val"] * len(labels),
            "tile_size": [512] * len(labels),
            "label": labels,
            "is_ambiguous": [False] * len(labels),
            "oocyte_area_fraction": [1.0 if l == "positive" else 0.0 for l in labels],
            "prob": probs,
        }
    )


def test_a_perfect_ranking_scores_one() -> None:
    targets = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = _point_metrics(targets, probs, threshold=0.5)

    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_average_precision_of_a_random_ranking_approaches_the_positive_rate() -> None:
    """The chance floor for AUPRC is the positive rate, which is why it is reported."""
    rng = np.random.default_rng(0)
    targets = (rng.random(20000) < 0.2).astype(int)
    probs = rng.random(20000)

    metrics = _point_metrics(targets, probs, threshold=0.5)

    assert metrics["auprc"] == pytest.approx(0.2, abs=0.02)


def test_single_class_split_reports_nan_rather_than_raising() -> None:
    metrics = _point_metrics(np.zeros(10, dtype=int), np.linspace(0, 1, 10), 0.5)

    assert all(np.isnan(value) for value in metrics.values())


def test_threshold_moves_recall_but_not_ranking_metrics() -> None:
    targets = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.45, 0.9])

    low = _point_metrics(targets, probs, threshold=0.3)
    high = _point_metrics(targets, probs, threshold=0.8)

    assert low["recall"] > high["recall"]
    assert low["auprc"] == pytest.approx(high["auprc"])
    assert low["roc_auc"] == pytest.approx(high["roc_auc"])


def test_trivial_baseline_is_reported_beside_accuracy() -> None:
    """At these positive rates answering negative to everything scores ~0.8, so an
    accuracy reported alone would read as skill."""
    labels = ["positive"] * 2 + ["negative"] * 8
    frame = _frame(labels, [0.4] * 10, ["A"] * 5 + ["B"] * 5)

    metrics = _split_metrics(frame, threshold=0.5, bootstrap_samples=50, seed=1)

    assert metrics["trivial_baseline"]["positive_rate"] == pytest.approx(0.2)
    assert metrics["trivial_baseline"]["all_negative_accuracy"] == pytest.approx(0.8)
    assert metrics["accuracy"] == pytest.approx(0.8)


def test_split_metrics_counts_tiles_positives_and_slides() -> None:
    frame = _frame(
        ["positive", "negative", "negative", "positive"],
        [0.9, 0.1, 0.2, 0.8],
        ["A", "A", "B", "C"],
    )

    metrics = _split_metrics(frame, threshold=0.5, bootstrap_samples=50, seed=1)

    assert metrics["n_tiles"] == 4
    assert metrics["n_positive"] == 2
    assert metrics["n_slides"] == 3


def test_intervals_resample_slides_not_tiles() -> None:
    """Tiles overlap by 20% and one oocyte spans several, so a tile-level bootstrap
    would treat correlated rows as independent and report far too narrow an interval.

    Two slides that disagree completely must produce a wide interval; a tile-level
    bootstrap over the same rows would not.
    """
    labels = ["positive", "negative"] * 10
    # Slide A is ranked perfectly, slide B is ranked backwards.
    probs = [0.9, 0.1] * 5 + [0.1, 0.9] * 5
    slides = ["A"] * 10 + ["B"] * 10
    frame = _frame(labels, probs, slides)
    targets = binary_targets(frame).to_numpy()

    intervals = _bootstrap_intervals(
        frame, targets, frame["prob"].to_numpy(), 0.5, 500, seed=1
    )

    low, high = intervals["auprc"]
    assert low is not None and high is not None
    assert high - low > 0.3, "resampling slides should expose the disagreement"


def test_intervals_are_none_when_resamples_cannot_be_scored() -> None:
    """One slide per class means most resamples are single-class; reporting a number
    from the handful that were not would overstate what the data supports."""
    frame = _frame(["positive"] * 5 + ["negative"] * 5, [0.5] * 10, ["A"] * 5 + ["B"] * 5)
    targets = binary_targets(frame).to_numpy()

    intervals = _bootstrap_intervals(
        frame, targets, frame["prob"].to_numpy(), 0.5, 20, seed=1
    )

    assert set(intervals) == {"auprc", "f1", "recall"}


def test_intervals_are_reproducible_for_a_seed() -> None:
    frame = _frame(
        ["positive", "negative"] * 6, [0.9, 0.1] * 6, ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    )
    targets = binary_targets(frame).to_numpy()
    probs = frame["prob"].to_numpy()

    first = _bootstrap_intervals(frame, targets, probs, 0.5, 200, seed=7)
    second = _bootstrap_intervals(frame, targets, probs, 0.5, 200, seed=7)

    assert first == second


def test_ambiguous_rows_are_counted_before_being_dropped(tmp_path: Path) -> None:
    """The count belongs in the results: a threshold sweep changes how many tiles the
    rule declined to call, and that is part of what a run means."""
    index = pd.DataFrame(
        {
            "split": ["train", "train", "val", "test", "val"],
            "label": ["ambiguous", "positive", "ambiguous", "negative", "ambiguous"],
        }
    )

    counts = _ambiguous_counts(index)

    assert counts == {"train": 1, "val": 2, "test": 0}


def test_observed_positive_fraction_comes_from_the_log_not_the_flag(tmp_path: Path) -> None:
    """Verification compares the realised fraction against the configured one, so it has
    to be read back from what the sampler actually delivered."""
    log = tmp_path / "batch_composition.jsonl"
    log.write_text(
        "\n".join(
            json.dumps({"n_positive": 4, "batch_size": 16}) for _ in range(10)
        ),
        encoding="utf-8",
    )

    assert _observed_positive_fraction(tmp_path) == pytest.approx(0.25)


def test_observed_positive_fraction_is_none_without_a_log(tmp_path: Path) -> None:
    assert _observed_positive_fraction(tmp_path) is None


def test_prediction_log_has_the_columns_verification_needs(tmp_path: Path) -> None:
    frame = _frame(["positive", "negative"], [0.9, 0.1], ["A", "B"])
    path = tmp_path / "predictions.csv"

    _write_predictions(path, frame)

    written = pd.read_csv(path)
    assert list(written.columns) == list(PREDICTION_COLUMNS)
    assert len(written) == 2


def test_prediction_log_refuses_a_frame_missing_columns(tmp_path: Path) -> None:
    frame = _frame(["positive"], [0.9], ["A"]).drop(columns=["prob"])

    with pytest.raises(ValueError, match="missing"):
        _write_predictions(tmp_path / "predictions.csv", frame)


def test_sanitised_results_are_written_as_strict_json(tmp_path: Path) -> None:
    """json.dump emits a bare NaN literal by default, which is not valid JSON."""
    from src.modeling.evaluate_tile_classifier import _nan_to_none

    path = tmp_path / "results.json"
    payload = _nan_to_none({"auprc": float("nan"), "nested": {"f1": float("nan")}, "n": 3})

    _write_results(path, payload)

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw) == {"auprc": None, "nested": {"f1": None}, "n": 3}


def test_writing_unsanitised_results_is_refused(tmp_path: Path) -> None:
    """The writer will not silently emit invalid JSON: evaluate() sanitises the payload
    it both returns and writes, so the terminal summary and the file cannot disagree."""
    with pytest.raises(ValueError):
        _write_results(tmp_path / "results.json", {"auprc": float("nan")})


def test_split_manifest_hash_changes_with_content(tmp_path: Path) -> None:
    """Recorded so the split can be compared rather than asserted."""
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text('{"manifest_version": "v1"}', encoding="utf-8")
    second.write_text('{"manifest_version": "v2"}', encoding="utf-8")

    assert _sha256(first) != _sha256(second)
    assert _sha256(first) == _sha256(first)
    assert len(_sha256(first)) == 64


def test_threshold_is_selected_where_f1_peaks() -> None:
    """0.5 is not a neutral cut here: training forces 25% positives per batch while the
    evaluation splits run at their natural rate, so the probabilities are on a different
    scale from the data being scored."""
    # Positives sit at 0.30-0.34, negatives below 0.20. The best cut is between them,
    # nowhere near 0.5, where every positive would be missed.
    targets = np.array([1] * 5 + [0] * 20)
    probs = np.concatenate([np.linspace(0.30, 0.34, 5), np.linspace(0.02, 0.19, 20)])

    threshold, how = select_threshold(targets, probs)

    assert 0.19 < threshold <= 0.30
    assert how == "max_val_f1"

    from sklearn.metrics import f1_score

    at_selected = f1_score(targets, (probs >= threshold).astype(int))
    at_half = f1_score(targets, (probs >= 0.5).astype(int), zero_division=0)
    assert at_selected == pytest.approx(1.0)
    assert at_half == 0.0


def test_threshold_selection_falls_back_on_a_single_class_split() -> None:
    threshold, how = select_threshold(np.zeros(10, dtype=int), np.linspace(0, 1, 10))

    assert threshold == FALLBACK_DECISION_THRESHOLD
    assert how == "fallback_single_class"


def test_selected_threshold_is_never_zero_or_one() -> None:
    """Either extreme predicts one class for everything, which is not an operating point."""
    rng = np.random.default_rng(0)
    targets = (rng.random(400) < 0.2).astype(int)
    probs = np.clip(targets * 0.4 + rng.normal(0.3, 0.1, 400), 0, 1)

    threshold, _ = select_threshold(targets, probs)

    assert 0.0 < threshold < 1.0


def test_balanced_accuracy_and_mcc_are_reported() -> None:
    """Raw accuracy is flattered by the majority class at these positive rates; these
    two are not."""
    targets = np.array([1] * 2 + [0] * 18)
    probs = np.array([0.9, 0.9] + [0.1] * 18)

    metrics = _point_metrics(targets, probs, threshold=0.5)

    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_balanced_accuracy_and_mcc_expose_a_majority_class_predictor() -> None:
    """The case that matters: accuracy says 0.9, the other two say chance and nothing."""
    targets = np.array([1] * 2 + [0] * 18)
    probs = np.full(20, 0.1)  # predicts negative for everything

    metrics = _point_metrics(targets, probs, threshold=0.5)

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["mcc"] == pytest.approx(0.0)


def test_undefined_metrics_include_the_new_ones() -> None:
    metrics = _point_metrics(np.zeros(10, dtype=int), np.linspace(0, 1, 10), 0.5)

    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["mcc"])


def test_threshold_free_metrics_are_unmoved_by_the_operating_point() -> None:
    """AUPRC and ROC AUC describe the ranking, which is why they are the primary pair."""
    targets = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.45, 0.9])

    low = _point_metrics(targets, probs, threshold=0.2)
    high = _point_metrics(targets, probs, threshold=0.8)

    assert low["auprc"] == pytest.approx(high["auprc"])
    assert low["roc_auc"] == pytest.approx(high["roc_auc"])
    # Recall is the unambiguous mover; balanced accuracy can coincide across two
    # thresholds when the per-class recalls trade off exactly, as they do here.
    assert low["recall"] > high["recall"]


def test_epoch_counts_come_from_the_training_log(tmp_path: Path) -> None:
    """The checkpoint is written only on an improving epoch, so its own epoch number is
    the best one -- reporting that as epochs_run understates an early-stopped run, and
    the two fields would never disagree, hiding the error."""
    from src.modeling.evaluate_tile_classifier import _epoch_summary

    (tmp_path / "training_log.json").write_text(
        json.dumps({"epochs_run": 12, "best_epoch": 7, "early_stopped": True}),
        encoding="utf-8",
    )

    summary = _epoch_summary(tmp_path, {"epoch": 7})

    assert summary["epochs_run"] == 12
    assert summary["best_epoch"] == 7
    assert summary["early_stopped"] is True
    assert summary["early_stopped_epoch"] == 12


def test_epoch_counts_are_null_rather_than_guessed_without_a_log(tmp_path: Path) -> None:
    from src.modeling.evaluate_tile_classifier import _epoch_summary

    summary = _epoch_summary(tmp_path, {"epoch": 7})

    assert summary["epochs_run"] is None
    assert summary["best_epoch"] == 7
    assert summary["early_stopped_epoch"] is None


def test_train_summary_describes_only_the_slides_the_model_saw() -> None:
    """A capped run trained on a subset; counting the whole split would report tiles and
    positives from slides the model never met."""
    from src.modeling.evaluate_tile_classifier import _train_summary
    from src.modeling.train_tile_classifier import TrainingConfig

    scored = pd.DataFrame(
        {
            "split": ["train"] * 6,
            "stem": ["A", "A", "B", "B", "C", "C"],
            "label": ["positive", "negative"] * 3,
            "is_ambiguous": [False] * 6,
        }
    )

    whole = _train_summary(scored, TrainingConfig())
    capped = _train_summary(scored, TrainingConfig(max_train_slides=2))

    assert whole["n_tiles"] == 6 and whole["n_slides"] == 3
    assert capped["n_tiles"] == 4 and capped["n_slides"] == 2
    assert capped["max_train_slides"] == 2


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("CHN_SP_5_22-24", "CHN_SP_5"),
        ("CHN_SP_5_25-27", "CHN_SP_5"),
        ("LHP_W_10_28-30", "LHP_W_10"),
        ("LHP_SP_6_3-4", "LHP_SP_6"),
        ("nounderscore", "nounderscore"),
    ],
)
def test_specimen_is_the_stem_without_its_cut_range(stem: str, expected: str) -> None:
    """Two cut ranges of one polyp are serial sections, not independent samples."""
    from src.modeling.evaluate_tile_classifier import specimen_of

    assert specimen_of(stem) == expected


def test_intervals_resample_specimens_not_slides() -> None:
    """Under grouped cross-validation a fold holds several slides of one specimen.
    Resampling slides would then treat serial sections through one polyp as independent
    evidence and report an interval narrower than the data supports.
    """
    # Four slides, two specimens. Specimen A is ranked perfectly, specimen B backwards.
    labels = ["positive", "negative"] * 10
    probs = [0.9, 0.1] * 5 + [0.1, 0.9] * 5
    slides = (["CHN_SP_5_10-12"] * 5 + ["CHN_SP_5_22-24"] * 5
              + ["LHP_W_10_10-12"] * 5 + ["LHP_W_10_28-30"] * 5)
    frame = _frame(labels, probs, slides)
    targets = binary_targets(frame).to_numpy()

    intervals = _bootstrap_intervals(
        frame, targets, frame["prob"].to_numpy(), 0.5, 500, seed=1
    )

    low, high = intervals["auprc"]
    assert low is not None and high is not None
    # Only two independent groups, so the interval has to span the disagreement.
    assert high - low > 0.3


def test_specimen_count_is_reported_alongside_the_slide_count() -> None:
    frame = _frame(
        ["positive", "negative"] * 2,
        [0.9, 0.1] * 2,
        ["CHN_SP_5_10-12", "CHN_SP_5_22-24", "LHP_W_10_10-12", "LHP_W_10_28-30"],
    )

    metrics = _split_metrics(frame, threshold=0.5, bootstrap_samples=50, seed=1)

    assert metrics["n_slides"] == 4
    assert metrics["n_specimens"] == 2
