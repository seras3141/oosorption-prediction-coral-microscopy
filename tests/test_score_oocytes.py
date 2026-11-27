"""Scoring a CV run per oocyte from its held-out predictions, its outputs and noise floor."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import score_oocytes
from src.modeling.cut_exclusions import CutExclusions
from src.modeling.oocyte_scoring import (
    noise_floor,
    score_held_out_predictions,
    write_noise_floor,
    write_oocyte_scores,
)

TILE, WIDTH, HEIGHT = 10, 30, 20
STEM = "S_A_1_1-2"
CUTS = {f"{STEM}_cut000": 1, f"{STEM}_cut001": 2, f"{STEM}_cut002": 2}
EXCLUDED = f"{STEM}_cut002"


def _square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _tiles(cut_name: str) -> list[dict]:
    tiles = []
    for row in range(3):
        for col in range(4):
            x0, y0 = col * 8, row * 8
            x1, y1 = min(x0 + TILE, WIDTH), min(y0 + TILE, HEIGHT)
            tiles.append({"tile_id": f"{cut_name}_r{row}_c{col}", "tile_size": TILE, "row": row,
                          "col": col, "cut_local_bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                          "pad_right": TILE - (x1 - x0), "pad_bottom": TILE - (y1 - y0)})
    return tiles


class HeldOutRun:
    """Held-out predictions over three cuts in two folds, with manifests and annotations."""

    def __init__(self, root: Path, cv_run_id: str = "cvA", shift: float = 0.0) -> None:
        self.tiles_dir, self.cuts_dir = root / "tiles", root / "cuts"
        rows = []
        for cut_name, fold in CUTS.items():
            tiles = _tiles(cut_name)
            manifest_dir = self.tiles_dir / STEM / cut_name
            manifest_dir.mkdir(parents=True)
            (manifest_dir / f"{cut_name}_tile_manifest.json").write_text(json.dumps({
                "cut_name": cut_name, "stem": STEM, "overlap": 0.2, "mpp_x": 0.5, "mpp_y": 0.5,
                "level0_bbox": {"x0": 0, "y0": 0, "x1": WIDTH, "y1": HEIGHT}, "tiles": tiles}))
            (self.cuts_dir / STEM).mkdir(parents=True, exist_ok=True)
            (self.cuts_dir / STEM / f"{cut_name}_annotations.geojson").write_text(json.dumps({
                "type": "FeatureCollection", "features": [
                    {"type": "Feature", "properties": {"name": "Oosorption Stage 0"},
                     "geometry": {"type": "Polygon", "coordinates": [_square(10, 10, 14, 14)]}},
                    {"type": "Feature", "properties": {"name": "Oosorption Stage 4"},
                     "geometry": {"type": "Polygon", "coordinates": [_square(1, 1, 7, 7)]}}]}))
            for tile in tiles:
                prob = 0.9 if tile["row"] == 1 and tile["col"] == 1 else 0.1
                rows.append({"tile_id": tile["tile_id"], "stem": STEM, "cut_name": cut_name,
                             "tile_size": TILE, "fold": fold, "run_id": f"run{fold}",
                             "label": "negative", "oocyte_area_fraction": 0.0,
                             "is_ambiguous": False, "prob": min(1.0, prob + shift),
                             "prob_source": "saved"})
        self.dir = root / "held_out" / cv_run_id
        self.dir.mkdir(parents=True)
        self.predictions = pd.DataFrame(rows)
        self.predictions.to_csv(self.dir / "predictions.csv", index=False)
        self.provenance = {"cv_run_id": cv_run_id, "n_tiles": len(rows), "n_ambiguous": 0,
                           "fold_manifest_sha256": "f" * 64}
        self.write_provenance()

    def write_provenance(self, **changes) -> None:
        (self.dir / "provenance.json").write_text(json.dumps({**self.provenance, **changes}))

    def score(self, exclusions: CutExclusions | None = None):
        exclusions = exclusions or CutExclusions(frozenset({EXCLUDED}), applied=True)
        return score_held_out_predictions(self.dir, exclusions, self.tiles_dir, self.cuts_dir)


@pytest.fixture
def run(tmp_path: Path) -> HeldOutRun:
    return HeldOutRun(tmp_path)


def test_a_run_is_scored_per_oocyte_without_its_excluded_cuts(run: HeldOutRun) -> None:
    scored = run.score()

    assert sorted(scored.per_oocyte["cut_name"].unique()) == [f"{STEM}_cut000", f"{STEM}_cut001"]
    assert scored.summary["excluded_cuts_in_predictions"] == [EXCLUDED]
    assert scored.summary["tile_size"] == TILE
    assert scored.summary["oocytes"] == {"total": 4, "judged": 4, "no_scored_area": 0,
                                         "under_min_share": 0}
    assert sorted(scored.curves["fold"].unique()) == [1, 2]
    found = scored.per_oocyte.set_index(["cut_name", "oocyte"])["found_threshold"]
    assert found[(f"{STEM}_cut000", 0)] == 0.9 and found[(f"{STEM}_cut000", 1)] == 0.1


def test_the_summary_carries_operating_points_breakdowns_and_inputs(run: HeldOutRun) -> None:
    summary = run.score().summary

    point = summary["operating_points"]["recall_0.90"]
    assert set(point["per_fold"]) == {1, 2} and point["folds_not_reaching"] == []
    assert summary["breakdowns"]["recall_0.90"]["stage"] == [
        {"value": 0, "n_scorable": 2, "n_found": 2, "recall": 1.0},
        {"value": 4, "n_scorable": 2, "n_found": 2, "recall": 1.0},
    ]
    assert len(summary["size_bin_edges_um"]) >= 1
    assert summary["fold_manifest_sha256"] == "f" * 64
    assert summary["min_scorable_share"] == 0.5 and summary["found_coverage"] == 0.5
    assert set(summary) >= {"predictions_sha256", "provenance_sha256", "cv_run_id"}
    assert summary["cut_exclusions"] == {"excluded_cuts": [EXCLUDED], "known_empty_cuts": [],
                                         "cut_exclusions_applied": True}
    assert summary["breakdowns"]["recall_0.90"]["folds"] == [1, 2]


def test_an_applied_list_that_excludes_nothing_scores_every_cut(run: HeldOutRun) -> None:
    scored = run.score(CutExclusions(applied=True))

    assert scored.per_oocyte["cut_name"].nunique() == 3 and scored.summary["excluded_cuts_in_predictions"] == []


def test_scoring_without_an_applied_exclusion_list_is_refused(run: HeldOutRun) -> None:
    with pytest.raises(ValueError, match="needs an applied cut exclusion list"):
        score_held_out_predictions(run.dir, CutExclusions(), run.tiles_dir, run.cuts_dir)


def test_a_run_whose_cuts_are_all_excluded_is_refused(run: HeldOutRun) -> None:
    with pytest.raises(ValueError, match="nothing to score"):
        run.score(CutExclusions(frozenset(CUTS), applied=True))


def test_predictions_listing_a_tile_twice_are_refused(run: HeldOutRun) -> None:
    pd.concat([run.predictions, run.predictions.iloc[:1]]).to_csv(run.dir / "predictions.csv",
                                                                index=False)
    run.write_provenance(n_tiles=len(run.predictions) + 1)

    with pytest.raises(ValueError, match="more than once"):
        run.score()


@pytest.mark.parametrize("key, value", [("n_tiles", 1), ("n_ambiguous", 5)])
def test_predictions_that_disagree_with_their_provenance_are_refused(
    run: HeldOutRun, key: str, value: int
) -> None:
    run.write_provenance(**{key: value})

    with pytest.raises(ValueError, match="but provenance.json records"):
        run.score()


def test_predictions_mixing_tile_sizes_are_refused(run: HeldOutRun) -> None:
    run.predictions.assign(tile_size=[TILE] * (len(run.predictions) - 1) + [20]).to_csv(
        run.dir / "predictions.csv", index=False)

    with pytest.raises(ValueError, match="mixes tile sizes"):
        run.score()


def test_a_cut_scored_by_two_folds_is_refused(run: HeldOutRun) -> None:
    predictions = run.predictions.copy()
    predictions.loc[predictions.index[0], "fold"] = 2
    predictions.to_csv(run.dir / "predictions.csv", index=False)

    with pytest.raises(ValueError, match="more than one fold"):
        run.score()


def test_scores_are_written_group_readable_and_never_over_an_existing_output(
    run: HeldOutRun, tmp_path: Path
) -> None:
    scored = run.score()
    previous = os.umask(0o002)
    try:
        out_dir = write_oocyte_scores(scored, tmp_path / "out")
    finally:
        os.umask(previous)

    assert sorted(p.name for p in out_dir.iterdir()) == ["curves.csv", "per_oocyte.csv",
                                                         "summary.json"]
    assert stat.S_IMODE(out_dir.stat().st_mode) & 0o070 == 0o070
    assert json.loads((out_dir / "summary.json").read_text())["cv_run_id"] == "cvA"
    with pytest.raises(FileExistsError):
        write_oocyte_scores(scored, tmp_path / "out")
    write_oocyte_scores(scored, tmp_path / "out", overwrite=True)


def test_a_cv_run_id_that_is_not_a_plain_name_is_refused(run: HeldOutRun, tmp_path: Path) -> None:
    run.write_provenance(cv_run_id="../escape")

    with pytest.raises(ValueError, match="not a plain directory name"):
        write_oocyte_scores(run.score(), tmp_path / "out")


def _summaries(tmp_path: Path) -> list[dict]:
    return [HeldOutRun(tmp_path / name, cv_run_id=name, shift=shift).score().summary
            for name, shift in (("seed42", 0.0), ("seed43", 0.05))]


def test_the_noise_floor_is_the_range_of_fold_means_across_runs(tmp_path: Path) -> None:
    summaries = _summaries(tmp_path)

    floor = noise_floor(summaries)

    entry = floor["recall_0.90"]["threshold"]
    means = [s["operating_points"]["recall_0.90"]["threshold"]["mean"] for s in summaries]
    assert entry["per_run"] == {"seed42": means[0], "seed43": means[1]}
    assert entry["range"] == pytest.approx(max(means) - min(means))
    assert floor["runs"] == ["seed42", "seed43"] and floor["tile_size"] == TILE


@pytest.mark.parametrize("problem", ["one_run", "same_run", "other_size", "other_folds",
                                     "other_exclusions", "other_targets", "other_corpus"])
def test_a_noise_floor_over_incomparable_runs_is_refused(tmp_path: Path, problem: str) -> None:
    summaries = _summaries(tmp_path)
    if problem == "one_run":
        summaries = summaries[:1]
    elif problem == "same_run":
        summaries = [summaries[0], summaries[0]]
    elif problem == "other_size":
        summaries[1] = {**summaries[1], "tile_size": 20}
    elif problem == "other_exclusions":
        summaries[1] = {**summaries[1], "cut_exclusions": {**summaries[1]["cut_exclusions"],
                                                           "excluded_cuts": []}}
    elif problem == "other_corpus":
        summaries[1] = {**summaries[1], "oocytes": {**summaries[1]["oocytes"], "total": 99}}
    elif problem == "other_targets":
        points = summaries[1]["operating_points"]
        summaries[1] = {**summaries[1],
                        "operating_points": {**points, "recall_0.95": points["recall_0.90"]}}
    else:
        summaries[1] = {**summaries[1], "fold_manifest_sha256": "0" * 64}

    with pytest.raises(ValueError):
        noise_floor(summaries)


def test_runs_sharing_no_reached_fold_have_no_noise_floor_value(tmp_path: Path) -> None:
    summaries = _summaries(tmp_path)
    summaries[1]["operating_points"]["recall_0.90"]["per_fold"] = {}

    floor = noise_floor(summaries)

    assert floor["recall_0.90"]["folds_used"] == []
    assert floor["recall_0.90"]["spill"]["range"] is None
    assert floor["recall_0.90"]["spill"]["runs_without_value"] == ["seed42", "seed43"]


def test_the_noise_floor_is_written_under_its_size_and_fold_manifest(tmp_path: Path) -> None:
    out_dir = write_noise_floor(noise_floor(_summaries(tmp_path)), tmp_path / "out")

    assert out_dir.name.startswith("noise_floor_0010_ffffffff_") and len(out_dir.name) == 34
    assert (out_dir / "noise_floor.json").exists()


@pytest.mark.parametrize("arguments", [[], ["--held-out-predictions", "a", "--noise-floor", "b"]])
def test_the_command_needs_exactly_one_mode(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        score_oocytes.parse_args(arguments)


def test_the_command_refuses_existing_or_duplicate_outputs_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(score_oocytes, "score_held_out_predictions",
                        lambda *args, **kwargs: calls.append(args))
    a, b = HeldOutRun(tmp_path / "a", cv_run_id="cvA"), HeldOutRun(tmp_path / "b", cv_run_id="cvA")
    (tmp_path / "out" / "cvB").mkdir(parents=True)
    c = HeldOutRun(tmp_path / "c", cv_run_id="cvB")

    with pytest.raises(ValueError, match="more than one input writes to"):
        score_oocytes.main(["--held-out-predictions", str(a.dir), str(b.dir),
                            "--output-root", str(tmp_path / "out")])
    with pytest.raises(FileExistsError, match="--overwrite"):
        score_oocytes.main(["--held-out-predictions", str(c.dir),
                            "--output-root", str(tmp_path / "out")])
    assert calls == []


def test_the_command_scores_each_run_with_the_exclusion_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exclusions = tmp_path / "exclusions.csv"
    exclusions.write_text(f"cut_name,status\n{EXCLUDED},excluded\n")
    run = HeldOutRun(tmp_path / "a")
    real_score = score_oocytes.score_held_out_predictions
    monkeypatch.setattr(score_oocytes, "score_held_out_predictions",
                        lambda held_out, excl: real_score(held_out, excl, run.tiles_dir, run.cuts_dir))

    score_oocytes.main(["--held-out-predictions", str(run.dir), "--output-root",
                        str(tmp_path / "out"), "--cut-exclusions", str(exclusions)])

    summary = json.loads((tmp_path / "out" / "cvA" / "summary.json").read_text())
    assert summary["excluded_cuts_in_predictions"] == [EXCLUDED]



def test_the_noise_floor_records_which_folds_each_run_averaged(tmp_path: Path) -> None:
    summaries = _summaries(tmp_path)
    point = summaries[1]["operating_points"]["recall_0.90"]
    point["per_fold"] = {fold: values for fold, values in point["per_fold"].items() if fold != 2}

    floor = noise_floor(summaries)

    assert floor["recall_0.90"]["folds_per_run"] == {"seed42": [1, 2], "seed43": [1]}
    assert floor["recall_0.90"]["fold_sets_match"] is False
    assert floor["recall_0.90"]["folds_used"] == [1]
    fold1 = [summaries[i]["operating_points"]["recall_0.90"]["per_fold"][1]["spill"] for i in (0, 1)]
    assert floor["recall_0.90"]["spill"]["per_run"] == {"seed42": fold1[0], "seed43": fold1[1]}
    assert floor["recall_0.80"]["fold_sets_match"] is True


def test_two_noise_floors_over_different_runs_get_different_names(tmp_path: Path) -> None:
    summaries = _summaries(tmp_path)
    third = HeldOutRun(tmp_path / "seed44", cv_run_id="seed44", shift=0.02).score().summary

    first = write_noise_floor(noise_floor(summaries), tmp_path / "out")
    second = write_noise_floor(noise_floor([summaries[0], third]), tmp_path / "out")

    assert first != second


def test_cut_exclusions_with_a_noise_floor_are_refused() -> None:
    with pytest.raises(SystemExit):
        score_oocytes.parse_args(["--noise-floor", "a.json", "b.json", "--cut-exclusions", "x.csv"])



def test_a_cut_without_annotations_must_be_listed_as_known_empty(run: HeldOutRun) -> None:
    empty = {"type": "FeatureCollection", "features": []}
    (run.cuts_dir / STEM / f"{STEM}_cut001_annotations.geojson").write_text(json.dumps(empty))

    with pytest.raises(ValueError, match="not listed as known empty"):
        run.score()
    scored = run.score(CutExclusions(frozenset({EXCLUDED}), frozenset({f"{STEM}_cut001"}),
                                     applied=True))
    assert scored.per_oocyte["cut_name"].unique().tolist() == [f"{STEM}_cut000"]


def test_empty_predictions_are_refused(run: HeldOutRun) -> None:
    run.predictions.iloc[:0].to_csv(run.dir / "predictions.csv", index=False)
    run.write_provenance(n_tiles=0)

    with pytest.raises(ValueError, match="has no rows"):
        run.score()


def test_a_run_left_with_no_oocytes_is_refused(run: HeldOutRun) -> None:
    empty = {"type": "FeatureCollection", "features": []}
    for cut_name in CUTS:
        (run.cuts_dir / STEM / f"{cut_name}_annotations.geojson").write_text(json.dumps(empty))

    with pytest.raises(ValueError, match="hold no oocytes"):
        run.score(CutExclusions(frozenset({EXCLUDED}), frozenset(CUTS) - {EXCLUDED}, applied=True))


def test_the_summary_records_its_input_relative_to_the_repo(
    run: HeldOutRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.modeling import cross_validation

    monkeypatch.setattr(cross_validation, "REPO_ROOT", run.dir.parents[1])

    assert run.score().summary["held_out_predictions"] == "held_out/cvA"



def test_a_cut_listed_as_known_empty_that_has_annotations_is_refused(run: HeldOutRun) -> None:
    with pytest.raises(ValueError, match="listed as known empty but has annotations"):
        run.score(CutExclusions(frozenset({EXCLUDED}), frozenset({f"{STEM}_cut000"}), applied=True))


def test_scoring_with_no_exclusion_list_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        score_oocytes.parse_args(["--held-out-predictions", "a", "--cut-exclusions", "none"])


def test_the_noise_floor_records_its_inputs_and_exclusion_list(tmp_path: Path) -> None:
    summaries = _summaries(tmp_path)

    floor = noise_floor(summaries)

    assert floor["inputs"]["seed42"]["predictions_sha256"] == summaries[0]["predictions_sha256"]
    assert floor["cut_exclusions"] == summaries[0]["cut_exclusions"]



def test_a_noise_floor_over_summaries_scored_before_a_metric_existed_is_refused(tmp_path: Path) -> None:
    summaries = _summaries(tmp_path)
    for values in summaries[1]["operating_points"]["recall_0.90"]["per_fold"].values():
        values.pop("false_blobs_2plus_per_cm2")

    with pytest.raises(ValueError, match="rescore it with --overwrite"):
        noise_floor(summaries)
