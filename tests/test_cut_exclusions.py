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
    load_cut_exclusions,
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
    (HEADER, '"A_cut000,excluded\nB_cut000",known_empty,,\n', "is not a cut name"),
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
