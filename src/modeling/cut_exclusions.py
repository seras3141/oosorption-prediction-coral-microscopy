"""Cuts the tile index leaves out, read from a versioned CSV: annotators skip cuts they
cannot read, so an unannotated cut is listed as ``excluded`` or as ``known_empty``."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUT_EXCLUSIONS = "configs/cut_exclusions.csv"
REQUIRED_COLUMNS = ("cut_name", "status")
OPTIONAL_COLUMNS = ("reason", "date")
EXCLUDED = "excluded"
KNOWN_EMPTY = "known_empty"
STATUSES = (EXCLUDED, KNOWN_EMPTY)
CUT_NAME = re.compile(r"[A-Za-z0-9_-]+_cut\d{3}")


@dataclass(frozen=True)
class CutExclusions:
    """Cuts left out and cuts indexed without annotations; ``applied`` is False only for
    the legacy corpus, indexed before the list existed."""

    excluded: frozenset[str] = frozenset()
    known_empty: frozenset[str] = frozenset()
    applied: bool = False
    reasons: dict[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.applied and (self.excluded or self.known_empty):
            raise ValueError("a cut exclusion list that lists cuts must be applied")
        both = sorted(self.excluded & self.known_empty)
        if both:
            raise ValueError(f"{', '.join(both)} listed both as excluded and as known empty")

    @property
    def sha256(self) -> str | None:
        """Over the cut names and statuses only; None for the legacy corpus."""
        if not self.applied:
            return None
        statuses = {**dict.fromkeys(self.excluded, EXCLUDED),
                    **dict.fromkeys(self.known_empty, KNOWN_EMPTY)}
        return _listing_sha256(statuses)


LEGACY_CORPUS = CutExclusions()


def load_cut_exclusions(path: str | Path = DEFAULT_CUT_EXCLUSIONS) -> CutExclusions:
    """Read and check the exclusion CSV; ``reason`` and ``date`` are optional.

    Raises
    ------
    ValueError
        If ``cut_name`` or ``status`` is missing from the header, the header has any other
        column, a quote is left open or spans lines, a row has more fields than the header, a cut is
        listed twice, a cut name is not of the form ``<slide>_cutNNN``, or a status is
        unknown.
    """
    resolved = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    # Spreadsheets save "CSV UTF-8" with a BOM.
    rows = csv.reader(io.StringIO(resolved.read_bytes().decode("utf-8-sig")), strict=True)
    try:
        return _parse_rows(rows, resolved)
    except csv.Error as error:
        raise ValueError(f"{resolved} line {rows.line_num}: {error}") from error


def _parse_rows(rows: Any, resolved: Path) -> CutExclusions:
    header = [name.strip() for name in next(rows, [])]
    named = [name for name in header if name]
    missing = [name for name in REQUIRED_COLUMNS if name not in named]
    unknown = [name for name in named if name not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    if missing or unknown or len(set(named)) != len(named):
        raise ValueError(
            f"{resolved}: header needs {', '.join(REQUIRED_COLUMNS)} and may add "
            f"{', '.join(OPTIONAL_COLUMNS)}; got {', '.join(header)}"
        )
    position = {name: header.index(name) for name in named}
    unnamed = [i for i, name in enumerate(header) if not name]
    statuses: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for row in rows:
        where = f"{resolved} line {rows.line_num}"
        if any("\n" in value or "\r" in value for value in row):
            raise ValueError(f"{where}: a field spans lines; close the quote it opens")
        if len(row) > len(header):
            raise ValueError(f"{where}: more fields than the header; quote a reason that "
                             "contains a comma")
        if any(row[i].strip() for i in unnamed if i < len(row)):
            raise ValueError(f"{where}: a value in a column without a name")
        if not any(value.strip() for value in row):
            continue
        field_of = {name: row[i].strip() if i < len(row) else "" for name, i in position.items()}
        cut_name, status = field_of["cut_name"], field_of["status"]
        if not CUT_NAME.fullmatch(cut_name):
            raise ValueError(f"{where}: {cut_name!r} is not a cut name like "
                             "LHP_SU_9_13-15_cut000")
        if status not in STATUSES:
            raise ValueError(f"{where}: status {status!r} is not one of {STATUSES}")
        if cut_name in statuses:
            raise ValueError(f"{where}: {cut_name} is listed twice")
        statuses[cut_name] = status
        reasons[cut_name] = field_of.get("reason", "")
    return CutExclusions(
        excluded=frozenset(c for c, s in statuses.items() if s == EXCLUDED),
        known_empty=frozenset(c for c, s in statuses.items() if s == KNOWN_EMPTY),
        applied=True,
        reasons=reasons,
    )


def _listing_sha256(statuses: dict[str, str]) -> str:
    # Only names and statuses define the corpus.
    listing = "".join(f"{cut},{status}\n" for cut, status in sorted(statuses.items()))
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def cut_exclusions_from_flag(value: str) -> CutExclusions:
    """``none`` for the legacy corpus, else the path of an exclusion CSV."""
    return LEGACY_CORPUS if value == "none" else load_cut_exclusions(value)


def cut_exclusion_fields(exclusions: CutExclusions) -> dict[str, object]:
    """What a run or a fold manifest records about the cut exclusions it used."""
    return {
        "excluded_cuts": tuple(sorted(exclusions.excluded)),
        "known_empty_cuts": tuple(sorted(exclusions.known_empty)),
        "cut_exclusions_applied": exclusions.applied,
    }


def recorded_cut_exclusions(
    excluded: tuple[str, ...], known_empty: tuple[str, ...], applied: bool
) -> CutExclusions:
    """The cut exclusions a run or fold manifest recorded."""
    return CutExclusions(frozenset(excluded), frozenset(known_empty), applied=applied)


def cut_exclusions_recorded_in(record: dict[str, Any]) -> CutExclusions:
    """The exclusions a fold manifest records; one without the fields is the legacy corpus."""
    return recorded_cut_exclusions(tuple(record.get("excluded_cuts", ())),
                                   tuple(record.get("known_empty_cuts", ())),
                                   bool(record.get("cut_exclusions_applied", False)))


def describe_cut_exclusions(exclusions: CutExclusions) -> str:
    """The exclusions in words, for error messages."""
    if not exclusions.applied:
        return "no cut exclusion list (the legacy corpus)"
    return (f"a cut exclusion list excluding {sorted(exclusions.excluded) or 'nothing'} "
            f"and confirming empty {sorted(exclusions.known_empty) or 'nothing'}")
