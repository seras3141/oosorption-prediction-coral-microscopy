"""Held-out tiles of a finished CV run: selection and the checks against what it recorded."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling import held_out_predictions
from src.modeling.cut_exclusions import LEGACY_CORPUS
from src.modeling.held_out_predictions import (
    OUTPUT_COLUMNS,
    check_held_out_coverage,
    check_reproduces_saved_predictions,
    collect_held_out_tiles,
    load_cv_run,
    predict_fold,
    predict_held_out_tiles,
    run_held_out_predictions,
    write_held_out_predictions,
)
from src.modeling.tile_index import binary_targets
from src.modeling.train_tile_classifier import TrainingConfig

STEMS = {"CHN_A_1_1-2": 1, "LHP_A_1_1-2": 1, "CHN_A_2_1-2": 2, "LHP_A_2_1-2": 2}
TILE_LABELS = {"t0": "positive", "t1": "negative", "t2": "ambiguous"}
TILE_COVERAGE = {"t0": 0.5, "t1": 0.0, "t2": 0.01}
FOLDS = (1, 2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _split(held_out: int, stem: str) -> str:
    if STEMS[stem] == held_out:
        return "test"
    return "val" if stem.startswith("LHP") else "train"


def _index(held_out: int) -> pd.DataFrame:
    rows = []
    for stem in STEMS:
        for suffix, label in TILE_LABELS.items():
            rows.append({"tile_id": f"{stem}_{suffix}", "stem": stem, "cut_name": f"{stem}_cut000",
                         "tile_size": 512, "oocyte_area_fraction": TILE_COVERAGE[suffix],
                         "label": label, "is_ambiguous": label == "ambiguous",
                         "split": _split(held_out, stem)})
    return pd.DataFrame(rows)


class CvRunFiles:
    """A two-fold CV run on disk: manifests, checkpoints, evaluations and saved predictions."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.split_paths = {}
        for fold in FOLDS:
            slides = {stem: {"split": _split(fold, stem)} for stem in STEMS}
            self.split_paths[fold] = _write_json(root / "splits" / f"fold{fold}.json",
                                                 {"slides": slides})
        self.fold_manifest_path = _write_json(root / "fold_manifest.json", {
            "manifest_version": "cv-v1",
            "n_folds": len(FOLDS),
            "folds": [{
                "fold": fold,
                "specimens": sorted({stem.rsplit("_", 1)[0] for stem, f in STEMS.items()
                                     if f == fold}),
                "inner_val_specimen": f"LHP_A_{fold}",
                "split_manifest_path": str(self.split_paths[fold]),
                "split_manifest_sha256": _sha256(self.split_paths[fold]),
            } for fold in FOLDS],
        })
        self.results_paths = {}
        self.predictions_paths = {}
        self.checkpoint_paths = {}
        self.run_ids = {}
        for fold in FOLDS:
            self.write_fold(fold, TrainingConfig(split_manifest_path=str(self.split_paths[fold])))
        self.write_cv_results()

    def write_fold(self, fold: int, config: TrainingConfig) -> None:
        run_dir = self.root / "runs" / f"fold{fold}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_paths[fold] = run_dir / "checkpoint.pt"
        torch.save({"config": asdict(config)}, self.checkpoint_paths[fold])
        self.run_ids[fold] = config.run_id()
        index = _index(fold)
        saved = index.loc[index["split"].isin(["test", "val"]) & ~index["is_ambiguous"]]
        self.predictions_paths[fold] = run_dir / "predictions.csv"
        saved.assign(prob=0.5).to_csv(self.predictions_paths[fold], index=False)
        self.results_paths[fold] = _write_json(run_dir / "results.json", {
            "run_id": self.run_ids[fold],
            "predictions_path": str(self.predictions_paths[fold]),
            "checkpoint_path": str(self.checkpoint_paths[fold]),
            "label_rule": "area",
            "min_oocyte_area_fraction": 0.05,
            "eval_min_oocyte_area_fraction": 0.05,
            "split_manifest_sha256": _sha256(self.split_paths[fold]),
            "n_ambiguous_excluded": {"train": 0, "val": 2,
                                     "test": int(index.query("split == 'test'")["is_ambiguous"].sum())},
        })

    def write_cv_results(self, **overrides) -> Path:
        scorable = sum(int((~_index(fold).query("split == 'test'")["is_ambiguous"]).sum())
                       for fold in FOLDS)
        data = {
            "fold_manifest_path": str(self.fold_manifest_path),
            "fold_manifest_sha256": _sha256(self.fold_manifest_path),
            "is_partial": False,
            "folds_missing": [],
            "cv_run_id": "cv1",
            "n_tiles_held_out": scorable,
            "per_fold": [{"fold": fold, "run_id": self.run_ids[fold],
                          "predictions_path": str(self.predictions_paths[fold])}
                         for fold in FOLDS],
        }
        data.update(overrides)
        self.cv_results_path = _write_json(self.root / "cv" / "cv_results.json", data)
        return self.cv_results_path

    def edit_json(self, path: Path, **changes) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(changes)
        path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def index_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_load_tile_index(**kwargs) -> pd.DataFrame:
        calls.append(kwargs)
        return _index(_fold_of(str(kwargs["split_manifest_path"])))

    monkeypatch.setattr(held_out_predictions, "load_tile_index", fake_load_tile_index)
    return calls


def _fold_of(split_path: str) -> int:
    return int(Path(split_path).stem.removeprefix("fold"))


@pytest.fixture
def run_files(tmp_path: Path) -> CvRunFiles:
    return CvRunFiles(tmp_path)


def test_every_held_out_tile_is_collected_ambiguous_included(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    held_out = collect_held_out_tiles(load_cv_run(run_files.cv_results_path))

    for fold in FOLDS:
        expected = set(_index(fold).query("split == 'test'")["tile_id"])
        assert set(held_out[fold]["tile_id"]) == expected
        assert held_out[fold]["is_ambiguous"].sum() == 2


def test_the_index_uses_the_runs_own_split_labelling_and_corpus(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    collect_held_out_tiles(load_cv_run(run_files.cv_results_path))

    assert [Path(call["split_manifest_path"]).resolve() for call in index_calls] == [
        run_files.split_paths[fold].resolve() for fold in FOLDS
    ]
    for call in index_calls:
        assert call["tile_sizes"] == (512,)
        assert call["label_rule"] == "area"
        assert call["min_oocyte_area_fraction"] == 0.05
        assert call["cut_exclusions"] == LEGACY_CORPUS


def test_a_rescored_evaluation_is_read_with_its_scoring_threshold(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    for fold in FOLDS:
        rescored = run_files.predictions_paths[fold].with_name("predictions_eval_area2500.csv")
        run_files.predictions_paths[fold].rename(rescored)
        run_files.predictions_paths[fold] = rescored
        results = run_files.results_paths[fold]
        results.rename(results.with_name("results_eval_area2500.json"))
        run_files.edit_json(results.with_name("results_eval_area2500.json"),
                            eval_min_oocyte_area_fraction=0.25, predictions_path=str(rescored))
    run_files.write_cv_results()

    cv_run = load_cv_run(run_files.cv_results_path)
    collect_held_out_tiles(cv_run)

    assert {fold.scoring_area_fraction for fold in cv_run.folds} == {0.25}
    assert {call["min_oocyte_area_fraction"] for call in index_calls} == {0.25}


def test_an_evaluation_without_a_recorded_labelling_falls_back_to_the_checkpoint(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    for fold in FOLDS:
        run_files.write_fold(fold, TrainingConfig(split_manifest_path=str(run_files.split_paths[fold]),
                                                  label_rule="centroid"))
        results = json.loads(run_files.results_paths[fold].read_text(encoding="utf-8"))
        for key in ("label_rule", "eval_min_oocyte_area_fraction"):
            results.pop(key)
        results["min_oocyte_area_fraction"] = 0.4
        run_files.results_paths[fold].write_text(json.dumps(results), encoding="utf-8")
    run_files.write_cv_results()

    cv_run = load_cv_run(run_files.cv_results_path)

    assert {fold.label_rule for fold in cv_run.folds} == {"centroid"}
    assert {fold.scoring_area_fraction for fold in cv_run.folds} == {
        cv_run.folds[0].config.min_oocyte_area_fraction
    }


def test_a_partial_cv_run_is_refused(run_files: CvRunFiles, index_calls: list[dict]) -> None:
    run_files.write_cv_results(is_partial=True, folds_missing=[2])

    with pytest.raises(ValueError, match="partial CV run"):
        load_cv_run(run_files.cv_results_path)


def test_a_cv_run_missing_a_fold_is_refused_even_unflagged(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    cv_results = json.loads(run_files.cv_results_path.read_text(encoding="utf-8"))
    run_files.write_cv_results(per_fold=cv_results["per_fold"][:1])

    with pytest.raises(ValueError, match="but its fold manifest has"):
        load_cv_run(run_files.cv_results_path)


def test_a_changed_fold_manifest_is_refused(run_files: CvRunFiles, index_calls: list[dict]) -> None:
    run_files.edit_json(run_files.fold_manifest_path, created="later")

    with pytest.raises(ValueError, match="has changed since"):
        load_cv_run(run_files.cv_results_path)


def test_a_checkpoint_from_another_run_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    recorded_run_id = run_files.run_ids[1]
    run_files.write_fold(1, TrainingConfig(split_manifest_path=str(run_files.split_paths[1]),
                                           seed=43))
    run_files.edit_json(run_files.results_paths[1], run_id=recorded_run_id)

    with pytest.raises(ValueError, match="rebuilds as run"):
        load_cv_run(run_files.cv_results_path)


def test_a_checkpoint_naming_another_split_manifest_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    copy = run_files.root / "elsewhere" / "fold1.json"
    copy.parent.mkdir()
    copy.write_bytes(run_files.split_paths[1].read_bytes())
    run_files.write_fold(1, TrainingConfig(split_manifest_path=str(copy)))
    run_files.write_cv_results()

    with pytest.raises(ValueError, match="but the fold manifest names"):
        load_cv_run(run_files.cv_results_path)


def test_an_evaluation_against_another_split_manifest_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    run_files.edit_json(run_files.results_paths[2], split_manifest_sha256="0" * 64)

    with pytest.raises(ValueError, match="the evaluation records split manifest"):
        load_cv_run(run_files.cv_results_path)


def test_held_out_specimens_other_than_the_folds_are_refused(
    run_files: CvRunFiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(held_out_predictions, "load_tile_index", lambda **kwargs: _index(2))

    with pytest.raises(ValueError, match="test specimens do not match"):
        collect_held_out_tiles(load_cv_run(run_files.cv_results_path))


def test_a_scored_tile_no_longer_held_out_is_refused(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    def index_without_a_tile(**kwargs) -> pd.DataFrame:
        frame = _index(_fold_of(str(kwargs["split_manifest_path"])))
        return frame.loc[frame["tile_id"] != "CHN_A_1_1-2_t0"]

    monkeypatch.setattr(held_out_predictions, "load_tile_index", index_without_a_tile)

    with pytest.raises(ValueError, match="tile\\(s\\) the run scored are not held out"):
        collect_held_out_tiles(load_cv_run(run_files.cv_results_path))


def test_an_unscored_tile_outside_the_ambiguous_band_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    saved = pd.read_csv(run_files.predictions_paths[1])
    saved.loc[saved["tile_id"] != "CHN_A_1_1-2_t1"].to_csv(run_files.predictions_paths[1],
                                                         index=False)

    with pytest.raises(ValueError, match="outside the ambiguous band were never scored"):
        collect_held_out_tiles(load_cv_run(run_files.cv_results_path))


def test_a_scored_tile_now_ambiguous_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    saved = pd.read_csv(run_files.predictions_paths[1])
    extra = saved.iloc[:1].assign(tile_id="CHN_A_1_1-2_t2", label="ambiguous")
    pd.concat([saved, extra]).to_csv(run_files.predictions_paths[1], index=False)

    with pytest.raises(ValueError, match="ambiguous under its labelling now"):
        collect_held_out_tiles(load_cv_run(run_files.cv_results_path))


def test_a_scorable_total_other_than_the_run_recorded_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    run_files.write_cv_results(n_tiles_held_out=7)

    with pytest.raises(ValueError, match="but the run recorded 7"):
        collect_held_out_tiles(load_cv_run(run_files.cv_results_path))


def test_a_tile_held_out_by_two_folds_is_refused() -> None:
    fold1 = _index(1).query("split == 'test'")
    fold2 = pd.concat([_index(2).query("split == 'test'"), fold1.iloc[:1]])

    with pytest.raises(ValueError, match="held out by more than one fold"):
        check_held_out_coverage({1: fold1, 2: fold2}, n_tiles_held_out=5)


def test_coverage_returns_every_held_out_tile_ambiguous_included() -> None:
    held_out = {fold: _index(fold).query("split == 'test'") for fold in FOLDS}

    assert check_held_out_coverage(held_out, n_tiles_held_out=8) == 12


def test_the_index_uses_the_exclusion_list_the_run_trained_with(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    for fold in FOLDS:
        run_files.write_fold(fold, TrainingConfig(split_manifest_path=str(run_files.split_paths[fold]),
                                                  excluded_cuts=("CHN_A_9_1-2_cut000",),
                                                  cut_exclusions_applied=True))
    run_files.write_cv_results()

    cv_run = load_cv_run(run_files.cv_results_path)
    collect_held_out_tiles(cv_run)

    assert {call["cut_exclusions"] for call in index_calls} == {cv_run.folds[0].config.cut_exclusions()}
    assert cv_run.folds[0].config.cut_exclusions() != LEGACY_CORPUS


@pytest.mark.parametrize("column, value", [("label", "negative"), ("oocyte_area_fraction", 0.3)])
def test_a_scored_tile_relabelled_since_the_run_is_refused(
    run_files: CvRunFiles, index_calls: list[dict], column: str, value
) -> None:
    saved = pd.read_csv(run_files.predictions_paths[1])
    saved.loc[saved["tile_id"] == "CHN_A_1_1-2_t0", column] = value
    saved.to_csv(run_files.predictions_paths[1], index=False)

    with pytest.raises(ValueError, match="different label or oocyte coverage"):
        collect_held_out_tiles(load_cv_run(run_files.cv_results_path))


def test_a_centroid_run_with_no_coverage_on_either_side_passes(
    run_files: CvRunFiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    def centroid_index(**kwargs) -> pd.DataFrame:
        return _index(_fold_of(str(kwargs["split_manifest_path"]))).assign(oocyte_area_fraction=None)

    for fold in FOLDS:
        saved = pd.read_csv(run_files.predictions_paths[fold]).assign(oocyte_area_fraction=None)
        saved.to_csv(run_files.predictions_paths[fold], index=False)
    monkeypatch.setattr(held_out_predictions, "load_tile_index", centroid_index)

    held_out = collect_held_out_tiles(load_cv_run(run_files.cv_results_path))

    assert sum(len(frame) for frame in held_out.values()) == 12


def test_coverage_missing_on_one_side_only_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    saved = pd.read_csv(run_files.predictions_paths[1])
    saved.loc[saved["tile_id"] == "CHN_A_1_1-2_t0", "oocyte_area_fraction"] = None
    saved.to_csv(run_files.predictions_paths[1], index=False)

    with pytest.raises(ValueError, match="different label or oocyte coverage"):
        collect_held_out_tiles(load_cv_run(run_files.cv_results_path))


def test_a_checkpoint_that_recorded_no_split_manifest_hash_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    checkpoint = torch.load(run_files.checkpoint_paths[1], weights_only=True)
    checkpoint["config"]["split_manifest_sha256"] = None
    torch.save(checkpoint, run_files.checkpoint_paths[1])

    with pytest.raises(ValueError, match="the checkpoint records split manifest"):
        load_cv_run(run_files.cv_results_path)


@pytest.mark.parametrize("field, value", [("run_id", "another_run"),
                                          ("predictions_path", "elsewhere/predictions.csv")])
def test_an_evaluation_file_from_another_run_or_predictions_is_refused(
    run_files: CvRunFiles, index_calls: list[dict], field: str, value: str
) -> None:
    run_files.edit_json(run_files.results_paths[1], **{field: value})

    with pytest.raises(ValueError, match="not the run and predictions the CV run recorded"):
        load_cv_run(run_files.cv_results_path)


def test_an_ambiguous_count_other_than_the_run_excluded_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    run_files.edit_json(run_files.results_paths[1], n_ambiguous_excluded={"test": 1})

    with pytest.raises(ValueError, match="but the run excluded 1 as ambiguous"):
        collect_held_out_tiles(load_cv_run(run_files.cv_results_path))


def test_an_evaluation_without_an_ambiguous_count_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    run_files.edit_json(run_files.results_paths[1], n_ambiguous_excluded={"train": 0})

    with pytest.raises(ValueError, match="does not record how many test tiles were ambiguous"):
        load_cv_run(run_files.cv_results_path)


def test_folds_scored_under_different_labellings_are_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    run_files.edit_json(run_files.results_paths[2], eval_min_oocyte_area_fraction=0.25)

    with pytest.raises(ValueError, match="scored under different labellings"):
        load_cv_run(run_files.cv_results_path)



SAVED_PROB = 0.5


def _probability_of(tile_id: str) -> float:
    if tile_id.endswith("_t2"):
        return 0.9 if tile_id.startswith("CHN") else 0.8
    return SAVED_PROB


class StubModel:
    def to(self, device):
        return self

    def eval(self):
        return self


class StubPrediction:
    """Stands in for model loading and inference, recording what reaches them."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, shift: dict[str, float] | None = None):
        self.restored_with_cache: list = []
        self.predicted: list[pd.DataFrame] = []
        self.predicted_with_cache: list = []
        self.shift = shift or {}
        monkeypatch.setattr(held_out_predictions, "_restore_model", self.restore)
        monkeypatch.setattr(held_out_predictions, "_predict", self.predict)

    def restore(self, config, checkpoint, cache=None):
        self.restored_with_cache.append(cache)
        return StubModel()

    def predict(self, model, frame, config, device, num_workers, cache=None):
        binary_targets(frame)
        self.predicted.append(frame)
        self.predicted_with_cache.append(cache)
        return np.array([_probability_of(t) + self.shift.get(t, 0.0) for t in frame["tile_id"]])


def test_every_held_out_tile_is_predicted_and_written(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    StubPrediction(monkeypatch)

    out_dir = run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                                       device="cpu")

    predictions = pd.read_csv(out_dir / "predictions.csv")
    assert tuple(predictions.columns) == OUTPUT_COLUMNS
    assert len(predictions) == 12
    assert predictions["is_ambiguous"].sum() == 4
    assert dict(zip(predictions["tile_id"], predictions["prob"])) == {
        t: _probability_of(t) for t in predictions["tile_id"]
    }
    assert set(zip(predictions["fold"], predictions["run_id"])) == {
        (fold, run_files.run_ids[fold]) for fold in FOLDS
    }
    assert out_dir.name == "cv1"
    assert [p.name for p in out_dir.parent.iterdir()] == ["cv1"]


def test_provenance_records_the_run_its_models_and_the_reproduction_gap(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    StubPrediction(monkeypatch, shift={"CHN_A_1_1-2_t0": 5e-4})

    out_dir = run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                                       device="cpu")

    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["cv_results_sha256"] == _sha256(run_files.cv_results_path)
    assert provenance["fold_manifest_sha256"] == _sha256(run_files.fold_manifest_path)
    assert (provenance["n_tiles"], provenance["n_ambiguous"]) == (12, 4)
    folds = {record["fold"]: record for record in provenance["folds"]}
    assert folds[1]["checkpoint_sha256"] == _sha256(run_files.checkpoint_paths[1])
    assert folds[1]["max_reproduction_difference"] == pytest.approx(5e-4)
    assert folds[2]["max_reproduction_difference"] == 0.0
    assert (folds[1]["n_tiles"], folds[1]["n_ambiguous"]) == (6, 2)


def test_ambiguous_tiles_reach_the_model_without_their_label(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubPrediction(monkeypatch)

    run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                             device="cpu")

    assert sum(len(frame) for frame in stub.predicted) == 12
    assert not any(frame["is_ambiguous"].any() for frame in stub.predicted)


def test_probabilities_beyond_the_tolerance_are_refused_and_nothing_is_written(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    StubPrediction(monkeypatch, shift={"CHN_A_1_1-2_t0": 2e-3})

    with pytest.raises(ValueError, match="differ from the run's saved ones"):
        run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                                 device="cpu")
    assert not (run_files.root / "out").exists() or not any((run_files.root / "out").iterdir())


def test_a_saved_tile_left_unpredicted_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    cv_run = load_cv_run(run_files.cv_results_path)
    held_out = collect_held_out_tiles(cv_run)[1]
    partial = held_out.loc[held_out["tile_id"] != "CHN_A_1_1-2_t0"]

    with pytest.raises(ValueError, match="some saved tiles were not predicted"):
        check_reproduces_saved_predictions(cv_run.folds[0], partial,
                                           np.full(len(partial), SAVED_PROB))


def test_an_existing_output_is_refused_before_any_prediction(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubPrediction(monkeypatch)
    existing = run_files.root / "out" / "cv1"
    existing.mkdir(parents=True)
    (existing / "predictions.csv").write_text("kept", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                                 device="cpu")
    assert stub.predicted == []
    assert (existing / "predictions.csv").read_text(encoding="utf-8") == "kept"


def test_writing_over_an_existing_output_is_refused_without_overwrite(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    cv_run = load_cv_run(run_files.cv_results_path)
    existing = run_files.root / "out" / "cv1"
    existing.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        write_held_out_predictions(cv_run, pd.DataFrame({"is_ambiguous": [False]}), [], "cpu",
                                   output_root=run_files.root / "out")
    assert [p.name for p in existing.parent.iterdir()] == ["cv1"]


def test_overwrite_replaces_an_existing_output(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    StubPrediction(monkeypatch)
    existing = run_files.root / "out" / "cv1"
    existing.mkdir(parents=True)
    (existing / "stale.txt").write_text("old", encoding="utf-8")

    out_dir = run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                                       device="cpu", overwrite=True)

    assert sorted(p.name for p in out_dir.iterdir()) == ["predictions.csv", "provenance.json"]


@pytest.mark.parametrize("overwrite, failing", [(False, "rename"), (True, "_swap_into_place")])
def test_a_failed_write_leaves_no_staging_directory(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch,
    overwrite: bool, failing: str
) -> None:
    def fail(*args):
        raise OSError("disk full")

    target = held_out_predictions.os if failing == "rename" else held_out_predictions
    monkeypatch.setattr(target, failing, fail)
    cv_run = load_cv_run(run_files.cv_results_path)

    with pytest.raises(OSError, match="disk full"):
        write_held_out_predictions(cv_run, pd.DataFrame({"is_ambiguous": [False]}), [], "cpu",
                                   output_root=run_files.root / "out", overwrite=overwrite)
    assert list((run_files.root / "out").iterdir()) == []


def test_a_frozen_run_is_scored_from_its_embedding_cache(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubPrediction(monkeypatch)
    cache = object()
    monkeypatch.setattr(held_out_predictions, "load_config_cache", lambda config: cache)
    cv_run = load_cv_run(run_files.cv_results_path)
    frozen = replace(cv_run.folds[0], config=SimpleNamespace(embedding_cache_dir="cache"))

    predict_fold(frozen, collect_held_out_tiles(cv_run)[1], torch.device("cpu"), 0)

    assert stub.restored_with_cache == [cache]
    assert stub.predicted_with_cache == [cache]


def test_tiles_the_run_scored_keep_their_saved_probability(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    StubPrediction(monkeypatch, shift={"CHN_A_1_1-2_t0": 5e-4, "CHN_A_1_1-2_t2": 5e-4})

    out_dir = run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                                       device="cpu")

    predictions = pd.read_csv(out_dir / "predictions.csv").set_index("tile_id")
    assert predictions.loc["CHN_A_1_1-2_t0", "prob"] == SAVED_PROB
    assert predictions.loc["CHN_A_1_1-2_t0", "prob_source"] == "saved"
    assert predictions.loc["CHN_A_1_1-2_t2", "prob"] == pytest.approx(0.9 + 5e-4)
    assert predictions.loc["CHN_A_1_1-2_t2", "prob_source"] == "recomputed"
    assert (predictions["prob_source"] == "recomputed").sum() == 4


def test_a_checkpoint_replaced_after_its_run_was_checked_is_refused(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubPrediction(monkeypatch)
    cv_run = load_cv_run(run_files.cv_results_path)
    checkpoint = torch.load(run_files.checkpoint_paths[1], weights_only=True)
    checkpoint["retrained"] = True
    torch.save(checkpoint, run_files.checkpoint_paths[1])

    with pytest.raises(ValueError, match="has changed since its run was checked"):
        predict_fold(cv_run.folds[0], collect_held_out_tiles(cv_run)[1], torch.device("cpu"), 0)
    assert stub.predicted == []


def test_a_missing_gpu_is_refused_before_any_work_unless_cpu_is_asked_for(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubPrediction(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="no GPU is available"):
        run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out")
    assert stub.predicted == [] and index_calls == []


def test_a_blank_saved_probability_is_refused(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    StubPrediction(monkeypatch)
    saved = pd.read_csv(run_files.predictions_paths[1])
    saved.loc[saved["tile_id"] == "CHN_A_1_1-2_t0", "prob"] = None
    saved.to_csv(run_files.predictions_paths[1], index=False)

    with pytest.raises(ValueError, match="saved held-out probabilities are blank"):
        run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                                 device="cpu")


def test_a_fold_with_no_scored_held_out_tiles_is_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    cv_run = load_cv_run(run_files.cv_results_path)
    empty = replace(cv_run.folds[0], saved_held_out=cv_run.folds[0].saved_held_out.iloc[:0])

    with pytest.raises(ValueError, match="scored no held-out tiles"):
        check_reproduces_saved_predictions(empty, collect_held_out_tiles(cv_run)[1], np.array([]))


def test_files_changed_after_loading_do_not_reach_the_output_or_its_provenance(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    StubPrediction(monkeypatch)
    cv_results_sha256 = _sha256(run_files.cv_results_path)
    predictions_sha256 = _sha256(run_files.predictions_paths[1])
    cv_run = load_cv_run(run_files.cv_results_path)
    held_out = collect_held_out_tiles(cv_run)
    pd.read_csv(run_files.predictions_paths[1]).assign(prob=0.1).to_csv(
        run_files.predictions_paths[1], index=False)
    run_files.edit_json(run_files.cv_results_path, note="rewritten")

    predictions, records = predict_held_out_tiles(cv_run, held_out, torch.device("cpu"), 0)
    out_dir = write_held_out_predictions(cv_run, predictions, records, "cpu",
                                         output_root=run_files.root / "out")

    written = pd.read_csv(out_dir / "predictions.csv")
    assert set(written.loc[written["prob_source"] == "saved", "prob"]) == {SAVED_PROB}
    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["cv_results_sha256"] == cv_results_sha256
    assert {f["fold"]: f["saved_predictions_sha256"] for f in provenance["folds"]}[1] == \
        predictions_sha256


def test_the_output_follows_the_umask_and_leaves_other_staging_alone(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    StubPrediction(monkeypatch)
    concurrent = run_files.root / "out" / ".cv1.staging-otherjob"
    concurrent.mkdir(parents=True)
    (concurrent / "predictions.csv").write_text("in progress", encoding="utf-8")
    previous = os.umask(0o002)
    try:
        out_dir = run_held_out_predictions(run_files.cv_results_path,
                                           output_root=run_files.root / "out", device="cpu")
    finally:
        os.umask(previous)

    assert stat.S_IMODE(out_dir.stat().st_mode) & 0o070 == 0o070
    assert sorted(p.name for p in out_dir.parent.iterdir()) == [".cv1.staging-otherjob", "cv1"]
    assert (concurrent / "predictions.csv").read_text(encoding="utf-8") == "in progress"


def test_saved_predictions_listing_a_tile_twice_are_refused(
    run_files: CvRunFiles, index_calls: list[dict]
) -> None:
    saved = pd.read_csv(run_files.predictions_paths[1])
    pd.concat([saved, saved.iloc[:1]]).to_csv(run_files.predictions_paths[1], index=False)

    with pytest.raises(ValueError, match="more than once"):
        load_cv_run(run_files.cv_results_path)


def test_provenance_records_each_folds_evaluation_file_hash(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    StubPrediction(monkeypatch)

    out_dir = run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                                       device="cpu")

    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert {f["fold"]: f["results_sha256"] for f in provenance["folds"]} == {
        fold: _sha256(run_files.results_paths[fold]) for fold in FOLDS
    }


def test_a_non_finite_recomputed_probability_is_refused(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    StubPrediction(monkeypatch, shift={"CHN_A_1_1-2_t2": float("nan")})

    with pytest.raises(ValueError, match="non-finite probabilities"):
        run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                                 device="cpu")
    assert not (run_files.root / "out" / "cv1").exists()


def test_an_output_written_concurrently_survives_a_run_without_overwrite(
    run_files: CvRunFiles, index_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    cv_run = load_cv_run(run_files.cv_results_path)
    concurrent = run_files.root / "out" / "cv1"
    concurrent.mkdir(parents=True)
    (concurrent / "predictions.csv").write_text("other job", encoding="utf-8")
    monkeypatch.setattr(held_out_predictions, "_refuse_existing", lambda out_dir, overwrite: None)

    with pytest.raises(OSError):
        write_held_out_predictions(cv_run, pd.DataFrame({"is_ambiguous": [False]}), [], "cpu",
                                   output_root=run_files.root / "out")
    assert (concurrent / "predictions.csv").read_text(encoding="utf-8") == "other job"
    assert [p.name for p in concurrent.parent.iterdir()] == ["cv1"]


@pytest.mark.parametrize("cv_run_id", ["", ".", "..", "../escape", "a/b"])
def test_a_cv_run_id_that_is_not_a_plain_name_is_refused(
    run_files: CvRunFiles, index_calls: list[dict], cv_run_id: str
) -> None:
    run_files.write_cv_results(cv_run_id=cv_run_id)

    with pytest.raises(ValueError, match="not a plain name"):
        run_held_out_predictions(run_files.cv_results_path, output_root=run_files.root / "out",
                                 device="cpu")
