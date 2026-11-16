"""The cut exclusion CSV: what it lists and the checks on load."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.cut_exclusions import (
    DEFAULT_CUT_EXCLUSIONS,
    LEGACY_CORPUS,
    cut_exclusion_fields,
    load_cut_exclusions,
    recorded_cut_exclusions,
)

HEADER = "cut_name,status,reason,date\n"


def _csv(tmp_path: Path, body: str, header: str = HEADER) -> Path:
    path = tmp_path / "cut_exclusions.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


def test_the_csv_splits_excluded_from_known_empty_cuts_and_hashes_the_listing(tmp_path: Path) -> None:
    path = _csv(tmp_path, "A_1-2_cut000,excluded,out of focus,2026-10-27\n"
                          "B_3-4_cut001,known_empty,checked: no oocytes,2026-10-27\n")
    exclusions = load_cut_exclusions(path)
    assert exclusions.excluded == {"A_1-2_cut000"}
    assert exclusions.known_empty == {"B_3-4_cut001"}
    assert exclusions.reasons["A_1-2_cut000"] == "out of focus"
    assert exclusions.applied and not LEGACY_CORPUS.applied
    before = exclusions.sha256
    path.write_text(path.read_text() + "C_5-6_cut002,excluded,folded,2026-10-28\n")
    assert load_cut_exclusions(path).sha256 != before


def test_the_hash_covers_the_listed_cuts_not_their_reasons_dates_or_line_endings(
        tmp_path: Path) -> None:
    first = load_cut_exclusions(_csv(tmp_path, "A_cut000,excluded,blur,2026-10-27\n")).sha256
    reworded = _csv(tmp_path, "A_cut000,excluded,out of focus,2026-11-02\r\n",
                    header=HEADER.replace("\n", "\r\n"))
    assert load_cut_exclusions(reworded).sha256 == first
    assert load_cut_exclusions(_csv(tmp_path, "A_cut000,known_empty,blur,2026-10-27\n")).sha256 != first


def test_a_byte_order_mark_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "cut_exclusions.csv"
    path.write_bytes(b"\xef\xbb\xbf" + (HEADER + "A_cut000,excluded,blur,2026-10-27\n").encode())
    assert load_cut_exclusions(path).excluded == {"A_cut000"}


def test_an_error_names_the_line_it_is_on_after_a_blank_line(tmp_path: Path) -> None:
    path = _csv(tmp_path, "A_cut000,excluded,blur,2026-10-27\n\nB_cut000,dropped,blur,2026-10-27\n")
    with pytest.raises(ValueError, match="line 4: status 'dropped'"):
        load_cut_exclusions(path)


def test_reason_and_date_are_optional_as_values_and_as_columns(tmp_path: Path) -> None:
    blank = load_cut_exclusions(_csv(tmp_path, "A_cut000,excluded,,\n"))
    bare = load_cut_exclusions(_csv(tmp_path, "A_cut000,excluded\n", header="cut_name,status\n"))
    assert blank.excluded == bare.excluded == {"A_cut000"}
    assert blank.reasons["A_cut000"] == "" and blank.sha256 == bare.sha256


@pytest.mark.parametrize(("header", "body", "message"), [
    ("cut_name,status,reason,source,date\n", "", "may add reason, date; got"),
    ("cut_name,reason,date\n", "", "header needs cut_name, status"),
    ("cut_name,status,status\n", "", "header needs"),
    (HEADER, "A_cut000,excluded,blur,2026-10-27\nA_cut000,excluded,blur,2026-10-27\n",
     "listed twice"),
    (HEADER, "A_cut000,dropped,blur,2026-10-27\n", "status 'dropped'"),
    (HEADER, " ,excluded,blur,2026-10-27\n", "'' is not a cut name"),
    (HEADER, "LHP_SU_9_13-15_cut00,excluded,,\n", "is not a cut name"),
    (HEADER, '"A_cut000,excluded\nB_cut000",known_empty,,\n', "a field spans lines"),
    (HEADER, "A_cut000,,blur,2026-10-27\n", "status '' is not one of"),
    (HEADER, "A_cut000,excluded,not annotated, out of focus,2026-10-27\n",
     "more fields than the header"),
])
def test_a_malformed_csv_is_refused(tmp_path: Path, header: str, body: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_cut_exclusions(_csv(tmp_path, body, header))


def test_a_quoted_reason_may_contain_a_comma(tmp_path: Path) -> None:
    path = _csv(tmp_path, 'A_cut000,excluded,"not annotated, out of focus",2026-10-27\n')
    assert load_cut_exclusions(path).reasons["A_cut000"] == "not annotated, out of focus"


def test_the_versioned_csv_leaves_out_the_cut_the_annotators_skipped() -> None:
    exclusions = load_cut_exclusions(DEFAULT_CUT_EXCLUSIONS)
    assert exclusions.excluded == {"LHP_SU_9_13-15_cut000"}
    assert exclusions.reasons["LHP_SU_9_13-15_cut000"] == "not annotated: out of focus"
    assert not exclusions.known_empty


def test_a_list_must_be_applied_to_list_cuts_and_its_hash_follows_its_cuts() -> None:
    from src.modeling.cut_exclusions import CutExclusions

    with pytest.raises(ValueError, match="must be applied"):
        CutExclusions(excluded=frozenset({"A_cut000"}))
    applied = CutExclusions(excluded=frozenset({"A_cut000"}), applied=True)
    assert applied.sha256 is not None
    assert applied.sha256 != CutExclusions(known_empty=frozenset({"A_cut000"}), applied=True).sha256
    assert CutExclusions(applied=True).sha256 is not None and LEGACY_CORPUS.sha256 is None
    with pytest.raises(ValueError, match="listed both"):
        CutExclusions(excluded=frozenset({"A_cut000"}), known_empty=frozenset({"A_cut000"}),
                      applied=True)


def test_errors_name_the_file_that_was_read(tmp_path: Path, monkeypatch) -> None:
    import src.modeling.cut_exclusions as module

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    _csv(tmp_path, "A_cut000,dropped,,\n")
    with pytest.raises(ValueError, match=re.escape(f"{tmp_path / 'cut_exclusions.csv'} line 2")):
        load_cut_exclusions("cut_exclusions.csv")


def test_blank_rows_and_an_empty_trailing_column_from_a_spreadsheet_are_ignored(
        tmp_path: Path) -> None:
    path = _csv(tmp_path, "A_cut000,excluded,blur,2026-10-27,\n,,,,\n",
                header="cut_name,status,reason,date,\n")
    assert load_cut_exclusions(path).excluded == {"A_cut000"}
    with pytest.raises(ValueError, match="a column without a name"):
        load_cut_exclusions(_csv(tmp_path, "A_cut000,excluded,blur,2026-10-27,x\n",
                                 header="cut_name,status,reason,date,\n"))
    with pytest.raises(ValueError, match="line 2: a value in a column without a name"):
        load_cut_exclusions(_csv(tmp_path, "A_cut000,excluded,blur,2026-10-27,see ticket 12,\n",
                                 header="cut_name,status,reason,date,,\n"))


def test_a_quote_left_open_is_refused_rather_than_swallowing_the_rows_after_it(
        tmp_path: Path) -> None:
    path = _csv(tmp_path, 'A_cut000,excluded,"blur,2026-10-27\nB_cut001,excluded,x,2026-10-27\n')
    with pytest.raises(ValueError, match="line 3: unexpected end of data"):
        load_cut_exclusions(path)


def test_a_quote_closed_in_a_later_row_is_refused(tmp_path: Path) -> None:
    path = _csv(tmp_path, 'A_cut000,excluded,"blur,\nB_cut001,excluded,x,\nC_cut002,excluded,end",\n')
    with pytest.raises(ValueError, match="a field spans lines"):
        load_cut_exclusions(path)
    longer = _csv(tmp_path, 'A_cut000,excluded,"blur\nB_cut001,excluded,x",2026-10-27,extra\n')
    with pytest.raises(ValueError, match="a field spans lines"):
        load_cut_exclusions(longer)


def test_recorded_exclusions_round_trip(tmp_path: Path) -> None:
    exclusions = load_cut_exclusions(_csv(tmp_path, "A_cut000,excluded,blur,2026-10-27\n"))
    fields = cut_exclusion_fields(exclusions)
    restored = recorded_cut_exclusions(fields["excluded_cuts"], fields["known_empty_cuts"],
                                       fields["cut_exclusions_applied"])
    assert restored == exclusions and restored.sha256 == exclusions.sha256
    assert recorded_cut_exclusions((), (), False) == LEGACY_CORPUS


def test_a_config_saved_before_the_list_keeps_its_run_id_and_corpus() -> None:
    from dataclasses import asdict

    from src.modeling.train_tile_classifier import TrainingConfig

    new_fields = ("excluded_cuts", "known_empty_cuts", "cut_exclusions_applied")
    saved = {key: value for key, value in asdict(TrainingConfig()).items() if key not in new_fields}
    reloaded = TrainingConfig(**saved)
    assert reloaded.run_id() == TrainingConfig().run_id()
    assert reloaded.cut_exclusions() == LEGACY_CORPUS


def test_a_run_with_exclusions_gets_its_own_id_and_reloads_with_it(tmp_path: Path) -> None:
    from dataclasses import asdict

    from src.modeling.train_tile_classifier import TrainingConfig

    exclusions = load_cut_exclusions(_csv(tmp_path, "A_cut000,excluded,blur,2026-10-27\n"))
    config = TrainingConfig(**cut_exclusion_fields(exclusions))
    assert config.run_id() != TrainingConfig().run_id()
    empty_list = TrainingConfig(**cut_exclusion_fields(load_cut_exclusions(_csv(tmp_path, ""))))
    assert empty_list.run_id() != TrainingConfig().run_id()
    as_json = {**asdict(config), "excluded_cuts": ["A_cut000"], "known_empty_cuts": []}
    reloaded = TrainingConfig(**as_json)
    assert reloaded.run_id() == config.run_id()
    assert reloaded.cut_exclusions() == exclusions


def test_the_training_flag_records_the_csv_or_the_legacy_corpus(tmp_path: Path) -> None:
    from scripts.train_tile_classifier import build_parser, config_from_args

    path = _csv(tmp_path, "A_cut000,excluded,blur,2026-10-27\n")
    config = config_from_args(build_parser().parse_args(["--cut-exclusions", str(path)]))
    assert config.excluded_cuts == ("A_cut000",) and config.cut_exclusions_applied
    legacy = config_from_args(build_parser().parse_args(["--cut-exclusions", "none"]))
    assert legacy.cut_exclusions() == LEGACY_CORPUS
    default = config_from_args(build_parser().parse_args([]))
    assert default.excluded_cuts == ("LHP_SU_9_13-15_cut000",)


def test_a_fold_manifest_built_with_another_list_is_never_overwritten(tmp_path: Path,
                                                                      monkeypatch) -> None:
    import json

    import scripts.generate_fold_manifest as cli

    listed = load_cut_exclusions(_csv(tmp_path, "A_cut000,excluded,,\n"))
    output = tmp_path / "fold_manifest.json"
    assert cli.other_exclusions_at(output, [], listed, manifests=()) is None
    output.write_text(json.dumps({"folds": []}))
    assert cli.other_exclusions_at(output, [], LEGACY_CORPUS, manifests=()) is None
    assert "orphan the runs made on it" in cli.other_exclusions_at(output, [], listed, manifests=())

    split_file = tmp_path / "cv-v1" / "fold1_split_manifest.json"
    v1 = tmp_path / "v1_fold_manifest.json"
    v1.write_text(json.dumps({"folds": [{"split_manifest_path": str(split_file)}]}))
    new_output = tmp_path / "fold_manifest_v2.json"
    assert str(v1) in cli.other_exclusions_at(new_output, [split_file], listed, manifests=(v1,))
    assert cli.other_exclusions_at(new_output, [tmp_path / "cv-v2" / "fold1.json"], listed,
                                   manifests=(v1,)) is None
    roundabout = tmp_path / "other" / ".." / "cv-v1" / "fold1_split_manifest.json"
    assert cli.other_exclusions_at(new_output, [roundabout], listed, manifests=(v1,))
    assert str(v1) in cli.other_exclusions_at(new_output, [split_file], listed)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    elsewhere = tmp_path / "v2" / "fold_manifest.json"
    (tmp_path / "data" / "splits").mkdir(parents=True)
    v1.rename(tmp_path / "data" / "splits" / "fold_manifest.json")
    assert "fold_manifest.json" in cli.other_exclusions_at(elsewhere, [split_file], listed)

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"slides": {}}))
    for name in ("load_tile_index", "specimen_counts", "assign_folds", "choose_inner_val"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "build_fold_manifest", lambda *args, **kwargs: {
        "n_specimens": 0, "n_slides": 0, "n_folds": 0, "folds": []})
    monkeypatch.setattr(cli, "format_fold_table", lambda manifest: "")
    before = output.read_text()
    assert cli.main(["--split-manifest", str(split), "--output", str(output), "--force",
                     "--cut-exclusions", str(tmp_path / "cut_exclusions.csv")]) == 1
    assert output.read_text() == before
