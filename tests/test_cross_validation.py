"""Cross-validation harness: isolation, coverage, lift and aggregation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.cross_validation import (
    aggregate,
    check_coverage,
    check_fold_isolation,
    cv_run_id,
    fold_config,
    fold_summary,
    load_fold_manifest,
    summarise,
)
from src.modeling.train_tile_classifier import TrainingConfig

# Each fold's LHP specimen is its inner-val one.
STEMS = {
    "CHN_A_1_1-2": 1, "LHP_A_1_1-2": 1,
    "CHN_A_2_1-2": 2, "LHP_A_2_1-2": 2,
}


def _fold_manifest(tiles_per_slide: int = 4) -> dict:
    folds = []
    for fold in (1, 2):
        slides = sorted(s for s, f in STEMS.items() if f == fold)
        folds.append({
            "fold": fold,
            "specimens": sorted({s.rsplit("_", 1)[0] for s in slides}),
            "slides": slides,
            "inner_val_specimen": f"LHP_A_{fold}",
            "tiles": {"512": tiles_per_slide * len(slides)},
            "ambiguous": {"512": 0},
            "split_manifest_path": f"splits/fold{fold}.json",
            "split_manifest_sha256": f"sha-fold{fold}",
        })
    return {"manifest_version": "cv-v1", "n_folds": 2, "folds": folds,
            "_sha256": "f" * 64, "_path": "splits/fold_manifest.json"}


def _split(held_out: int, stem: str) -> str:
    if STEMS[stem] == held_out:
        return "test"
    return "val" if stem.startswith("LHP") else "train"


def _predictions(held_out: int, tiles_per_slide: int = 4) -> pd.DataFrame:
    rows = []
    for stem in STEMS:
        split = _split(held_out, stem)
        if split == "train":
            continue
        for i in range(tiles_per_slide):
            rows.append({
                "tile_id": f"{stem}_t{i}", "stem": stem, "split": split,
                "label": "positive" if i == 0 else "negative",
                "prob": 0.9 if i == 0 else 0.1,
            })
    return pd.DataFrame(rows)


def _results(held_out: int, predictions_path: Path, auprc: float = 0.8,
             positive_rate: float = 0.25) -> dict:
    return {
        "run_id": f"run-fold{held_out}",
        "split_manifest_sha256": f"sha-fold{held_out}",
        "decision_threshold": 0.5,
        "threshold_selection": "max_val_f1",
        "best_epoch": 3,
        "epochs_run": 8,
        "is_subset_run": False,
        "predictions_path": str(predictions_path),
        "metrics": {
            "val": {"n_tiles": 4, "n_specimens": 1},
            "test": {
                "n_tiles": 8, "n_positive": 2, "n_slides": 2, "n_specimens": 2,
                "auprc": auprc, "roc_auc": 0.9, "f1": 0.7, "precision": 0.6,
                "recall": 0.8, "accuracy": 0.9, "balanced_accuracy": 0.85, "mcc": 0.6,
                "trivial_baseline": {"positive_rate": positive_rate,
                                     "all_negative_accuracy": 1 - positive_rate},
            },
        },
    }


def _fold_results(tmp_path: Path, folds=(1, 2)) -> dict[int, dict]:
    out = {}
    for fold in folds:
        path = tmp_path / f"predictions_fold{fold}.csv"
        _predictions(fold).to_csv(path, index=False)
        out[fold] = _results(fold, path)
    return out


def test_fold_config_points_at_the_folds_manifest(tmp_path: Path) -> None:
    manifest = _fold_manifest()
    config = fold_config(TrainingConfig(), manifest, 2)
    assert config.split_manifest_path == "splits/fold2.json"


def test_fold_config_refuses_a_base_that_already_names_a_manifest(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text("{}")
    with pytest.raises(ValueError, match="sets the split manifest per fold"):
        fold_config(TrainingConfig(split_manifest_path=str(other)), _fold_manifest(), 1)


def test_fold_manifest_is_refused_when_a_fold_split_was_regenerated(tmp_path: Path) -> None:
    split = tmp_path / "fold1.json"
    split.write_text('{"slides": {}}')
    manifest = {"manifest_version": "cv-v1", "n_folds": 1, "folds": [{
        "fold": 1, "split_manifest_path": str(split),
        "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
    }]}
    path = tmp_path / "fold_manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_fold_manifest(path)["folds"][0]["fold"] == 1

    split.write_text('{"slides": {"X_1_1-2": {"split": "test"}}}')
    with pytest.raises(ValueError, match="does not match the hash"):
        load_fold_manifest(path)


def _index(held_out: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"tile_id": f"{stem}_t{i}", "stem": stem, "split": _split(held_out, stem)}
        for stem in STEMS for i in range(3)
    ])


@pytest.mark.parametrize("held_out", [1, 2])
def test_a_correct_fold_index_passes(held_out: int) -> None:
    found = check_fold_isolation(_index(held_out), _fold_manifest(), held_out)
    assert set(found["test"]) == {f"CHN_A_{held_out}", f"LHP_A_{held_out}"}


def test_training_on_a_held_out_specimen_is_refused() -> None:
    index = _index(1)
    index.loc[index["stem"] == "LHP_A_1_1-2", "split"] = "train"
    with pytest.raises(ValueError, match="specimens do not match"):
        check_fold_isolation(index, _fold_manifest(), 1)


def test_one_tile_of_a_specimen_on_the_wrong_side_is_caught() -> None:
    index = _index(1)
    index.loc[0, "split"] = "train"          # one CHN_A_1 tile moved out of test
    with pytest.raises(ValueError, match="more than one split"):
        check_fold_isolation(index, _fold_manifest(), 1)


def test_validating_on_anything_but_inner_val_is_refused() -> None:
    index = _index(1)
    index.loc[index["stem"] == "CHN_A_2_1-2", "split"] = "val"
    with pytest.raises(ValueError, match="specimens do not match"):
        check_fold_isolation(index, _fold_manifest(), 1)


def test_an_entire_missing_specimen_is_caught() -> None:
    index = _index(1)
    index = index.loc[index["stem"] != "CHN_A_1_1-2"]
    with pytest.raises(ValueError, match=r"test specimens.*missing \['CHN_A_1'\]"):
        check_fold_isolation(index, _fold_manifest(), 1)


def test_prediction_log_requires_exact_val_and_test_sets() -> None:
    predictions = _predictions(1)
    check_fold_isolation(
        predictions, _fold_manifest(), 1, required_splits=("val", "test")
    )
    incomplete = predictions.loc[predictions["stem"] != "LHP_A_2_1-2"]
    with pytest.raises(ValueError, match=r"val specimens.*missing \['LHP_A_2'\]"):
        check_fold_isolation(
            incomplete, _fold_manifest(), 1, required_splits=("val", "test")
        )


SYNTHETIC_TILE_IDS = {f"{stem}_t{i}" for stem in STEMS for i in range(4)}


@pytest.fixture
def synthetic_scorable_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.modeling.cross_validation as module

    monkeypatch.setattr(module, "scorable_tile_ids", lambda *a: set(SYNTHETIC_TILE_IDS))


def test_coverage_counts_each_held_out_tile_once() -> None:
    held_out = {f: _predictions(f).query("split == 'test'") for f in (1, 2)}
    assert check_coverage(held_out, expected_tile_ids=SYNTHETIC_TILE_IDS) == 16


def test_a_tile_held_out_twice_is_refused() -> None:
    held_out = {1: _predictions(1).query("split == 'test'")}
    held_out[2] = held_out[1].head(1)
    with pytest.raises(ValueError, match="more than one fold"):
        check_coverage(held_out)


def test_coverage_short_of_the_corpus_is_refused() -> None:
    held_out = {1: _predictions(1).query("split == 'test'")}
    with pytest.raises(ValueError, match=r"8 never held out"):
        check_coverage(held_out, expected_tile_ids=SYNTHETIC_TILE_IDS)


def test_lift_is_auprc_over_the_scored_positive_rate(tmp_path: Path) -> None:
    summary = fold_summary(_results(1, tmp_path / "p.csv", auprc=0.6, positive_rate=0.2), 1)
    assert summary["auprc_lift"] == pytest.approx(3.0)
    assert summary["positive_rate"] == 0.2
    assert summary["auprc"] == 0.6


def test_a_single_class_fold_is_reported_not_scored(tmp_path: Path) -> None:
    summary = fold_summary(_results(1, tmp_path / "p.csv", auprc=None, positive_rate=0.0), 1)
    assert summary["undefined"] is True
    assert summary["auprc_lift"] is None


def test_summary_skips_undefined_folds_and_says_how_many_it_used() -> None:
    assert summarise([2.0, 4.0, None]) == {
        "mean": 3.0, "min": 2.0, "max": 4.0, "iqr": [2.5, 3.5], "n_defined": 2,
    }


@pytest.mark.usefixtures("synthetic_scorable_tiles")
def test_aggregate_over_every_fold(tmp_path: Path) -> None:
    results = aggregate(TrainingConfig(), _fold_manifest(), _fold_results(tmp_path))

    assert results["is_partial"] is False
    assert results["n_tiles_held_out"] == 16
    assert [f["fold"] for f in results["per_fold"]] == [1, 2]
    assert results["summary"]["auprc_lift"]["mean"] == pytest.approx(0.8 / 0.25)
    assert "not a confidence interval" in results["summary"]["note"]
    assert "confidence_interval_95" not in json.dumps(results["summary"])


def test_aggregate_refuses_missing_folds_unless_told_it_is_partial(tmp_path: Path) -> None:
    partial = _fold_results(tmp_path, folds=(1,))
    with pytest.raises(ValueError, match="missing for fold"):
        aggregate(TrainingConfig(), _fold_manifest(), partial)

    results = aggregate(TrainingConfig(), _fold_manifest(), partial, allow_partial=True)
    assert results["is_partial"] is True
    assert results["folds_missing"] == [2]


def test_aggregate_refuses_a_fold_scored_against_another_manifest(tmp_path: Path) -> None:
    fold_results = _fold_results(tmp_path)
    fold_results[2]["split_manifest_sha256"] = "something-else"
    with pytest.raises(ValueError, match="split manifest other than"):
        aggregate(TrainingConfig(), _fold_manifest(), fold_results)


def test_aggregate_refuses_when_the_folds_miss_scorable_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.modeling.cross_validation as module

    monkeypatch.setattr(module, "scorable_tile_ids",
                        lambda *a: SYNTHETIC_TILE_IDS | {"CHN_A_1_1-2_t9"})
    with pytest.raises(ValueError, match=r"1 never held out .*0 not scorable"):
        aggregate(TrainingConfig(), _fold_manifest(), _fold_results(tmp_path))


def test_coverage_uses_the_labelling_the_folds_were_scored_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.modeling.cross_validation as module

    seen = []
    monkeypatch.setattr(module, "scorable_tile_ids",
                        lambda *args: seen.append(args[1:]) or set(SYNTHETIC_TILE_IDS))
    fold_results = _fold_results(tmp_path)
    for results in fold_results.values():
        results.update(label_rule="area", eval_min_oocyte_area_fraction=0.25)
    aggregate(TrainingConfig(min_oocyte_area_fraction=0.05), _fold_manifest(), fold_results)
    assert seen == [(512, "area", 0.25)]


def test_folds_scored_under_different_labellings_are_refused(tmp_path: Path) -> None:
    fold_results = _fold_results(tmp_path)
    fold_results[1]["eval_min_oocyte_area_fraction"] = 0.05
    fold_results[2]["eval_min_oocyte_area_fraction"] = 0.25
    with pytest.raises(ValueError, match="different labellings"):
        aggregate(TrainingConfig(), _fold_manifest(), fold_results)


@pytest.mark.usefixtures("synthetic_scorable_tiles")
def test_a_substituted_tile_with_the_right_count_is_refused(tmp_path: Path) -> None:
    fold_results = _fold_results(tmp_path)
    path = Path(fold_results[2]["predictions_path"])
    predictions = pd.read_csv(path)
    predictions.loc[predictions["tile_id"] == "CHN_A_2_1-2_t0", "tile_id"] = "CHN_A_2_1-2_t9"
    predictions.to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"1 never held out .*1 not scorable"):
        aggregate(TrainingConfig(), _fold_manifest(), fold_results)


@pytest.mark.usefixtures("synthetic_scorable_tiles")
def test_a_subset_fold_marks_the_whole_aggregate(tmp_path: Path) -> None:
    fold_results = _fold_results(tmp_path)
    fold_results[1]["is_subset_run"] = True
    assert aggregate(TrainingConfig(), _fold_manifest(), fold_results)["is_subset_run"] is True


def test_cv_run_id_names_the_fold_assignment_not_the_per_fold_manifest() -> None:
    base = TrainingConfig()
    assert cv_run_id(base, "a" * 64) == cv_run_id(base, "a" * 64)
    assert cv_run_id(base, "a" * 64) != cv_run_id(base, "b" * 64)
    assert cv_run_id(base, "a" * 64) != cv_run_id(TrainingConfig(seed=43), "a" * 64)
    assert cv_run_id(base, "a" * 64).startswith("resnet18_none_0512_area0500_seed42_cv_")


def test_real_fold_splits_isolate_specimens_and_cover_every_tile_once() -> None:
    from src.modeling.tile_index import load_tile_index, training_rows

    manifest = load_fold_manifest()
    index = training_rows(load_tile_index(tile_sizes=(512,)))
    held_out = {}
    for fold in manifest["folds"]:
        split = json.loads((REPO_ROOT / fold["split_manifest_path"]).read_text())["slides"]
        fold_index = index.assign(split=index["stem"].map(lambda s: split[s]["split"]))
        check_fold_isolation(fold_index, manifest, fold["fold"])
        held_out[fold["fold"]] = fold_index.loc[fold_index["split"] == "test"]

    from src.modeling.cross_validation import scorable_tile_ids

    expected = scorable_tile_ids(manifest, 512, "area", 0.05)
    assert check_coverage(held_out, expected_tile_ids=expected) == len(index)


@pytest.mark.parametrize(
    ("folds", "aggregate", "allow_partial", "refused"),
    [
        ([2], True, False, True),
        ([2], False, False, False),
        ([1, 2], True, True, False),
        ([1, 2, 3, 4], True, False, False),
        ([5], False, False, True),
        ([1, 1], False, False, True),
    ],
)
def test_a_fold_subset_that_could_not_aggregate_is_refused_before_training(
    folds: list[int], aggregate: bool, allow_partial: bool, refused: bool
) -> None:
    from scripts.run_cross_validation import check_fold_selection

    call = lambda: check_fold_selection(  # noqa: E731
        folds, [1, 2, 3, 4], aggregate=aggregate, allow_partial=allow_partial
    )
    if refused:
        with pytest.raises(SystemExit):
            call()
    else:
        call()


def test_an_unsupported_tile_size_fails_before_the_index_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.modeling.tile_index as tile_index
    from src.modeling.cross_validation import run_fold

    def _explode(**kwargs):
        raise AssertionError("the index must not be parsed for a refused tile size")

    monkeypatch.setattr(tile_index, "load_tile_index", _explode)
    base = TrainingConfig(architecture="frozen_encoder", tile_size=128)
    with pytest.raises(ValueError, match="supports tile sizes"):
        run_fold(base, _fold_manifest(), 1)
