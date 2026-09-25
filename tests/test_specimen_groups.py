"""Specimen grouping and the cross-validation fold assignment.

Synthetic tests pin the rules; the real-corpus tests pin the published fold table, so a
change to the tile index, the labels or the search that moves any fold shows up here.
"""

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

from src.modeling.specimen_groups import (
    ANCHOR_SITE,
    assign_folds,
    build_fold_manifest,
    choose_inner_val,
    fold_split_manifest,
    site_of,
    specimen_counts,
    specimen_of,
    straddling_specimens,
    write_fold_manifests,
)
from src.modeling.tile_index import load_tile_index


# ---------------------------------------------------------------------------
# Synthetic fixtures


def _counts(rows: dict[str, tuple[int, int]], size: int = 512) -> pd.DataFrame:
    """A specimen_counts-shaped frame from {specimen: (tiles, positives)} at one size."""
    return pd.DataFrame(
        {
            "site": [site_of(s) for s in rows],
            "slides": [(f"{s}_1-2",) for s in rows],
            f"tiles_{size}": [t for t, _ in rows.values()],
            f"positives_{size}": [p for _, p in rows.values()],
            f"ambiguous_{size}": [0 for _ in rows],
        },
        index=list(rows),
    )


def _index(stems_labels: dict[str, list[str]], size: int = 512) -> pd.DataFrame:
    rows = [
        {"stem": stem, "tile_size": size, "label": label}
        for stem, labels in stems_labels.items()
        for label in labels
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Keys


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("CHN_SP_5_22-24", "CHN_SP_5"),
        ("CHN_SP_5_25-27", "CHN_SP_5"),
        ("LHP_SU_10_52-54", "LHP_SU_10"),
        ("LHP_SP_6_3-4", "LHP_SP_6"),
    ],
)
def test_specimen_drops_only_the_cut_range(stem: str, expected: str) -> None:
    assert specimen_of(stem) == expected
    assert site_of(stem) == expected.split("_")[0]


def test_evaluator_uses_the_same_specimen_key() -> None:
    """One definition, not two: the evaluator's bootstrap groups by this key."""
    from src.modeling import evaluate_tile_classifier

    assert evaluate_tile_classifier.specimen_of is specimen_of


def test_straddling_specimens_names_every_split_they_touch() -> None:
    assignments = {"A_X_1_1-2": "train", "A_X_1_3-4": "test", "B_X_2_1-2": "val"}
    assert straddling_specimens(assignments) == {"A_X_1": ["test", "train"]}


# ---------------------------------------------------------------------------
# Counts


def test_counts_group_slides_into_their_specimen() -> None:
    index = _index({
        "CHN_A_1_1-2": ["positive", "negative"],
        "CHN_A_1_3-4": ["ambiguous", "negative"],
        "LHP_A_2_1-2": ["negative"],
    })
    counts = specimen_counts(index)

    assert list(counts.index) == ["CHN_A_1", "LHP_A_2"]
    assert counts.at["CHN_A_1", "slides"] == ("CHN_A_1_1-2", "CHN_A_1_3-4")
    assert counts.at["CHN_A_1", "tiles_512"] == 4
    assert counts.at["CHN_A_1", "positives_512"] == 1
    assert counts.at["CHN_A_1", "ambiguous_512"] == 1


# ---------------------------------------------------------------------------
# Fold assignment


def _eight_specimens() -> pd.DataFrame:
    return _counts({
        "CHN_A_1": (100, 10), "CHN_A_2": (100, 20),
        "LHP_A_1": (100, 30), "LHP_A_2": (100, 5), "LHP_A_3": (100, 15),
        "LHP_A_4": (100, 25), "LHP_A_5": (100, 10), "LHP_A_6": (100, 20),
    })


def test_every_fold_gets_exactly_one_anchor_specimen() -> None:
    assignment = assign_folds(_eight_specimens(), n_folds=2, balance_sizes=(512,),
                              min_positives={})
    anchors = [f for s, f in assignment.items() if site_of(s) == ANCHOR_SITE]
    assert sorted(anchors) == [1, 2]


def test_assignment_is_deterministic() -> None:
    counts = _eight_specimens()
    kwargs = dict(n_folds=2, balance_sizes=(512,), min_positives={})
    assert assign_folds(counts, **kwargs) == assign_folds(counts.sample(frac=1, random_state=1),
                                                         **kwargs)


def test_positive_floor_is_enforced() -> None:
    counts = _counts({"CHN_A_1": (100, 1), "CHN_A_2": (100, 1),
                      "LHP_A_1": (100, 40), "LHP_A_2": (100, 1)})
    # Each anchor holds 1 positive, so a floor of 2 forces one LHP specimen into each
    # fold, even though parking both with one anchor would balance rates no worse.
    assignment = assign_folds(counts, n_folds=2, balance_sizes=(512,),
                              min_positives={512: 2}, max_size_ratio=None)
    assert assignment["LHP_A_1"] != assignment["LHP_A_2"]
    # Only the fold holding LHP_A_1 can reach 30, so a floor of 30 in both is infeasible.
    with pytest.raises(ValueError, match="positive-tile floor"):
        assign_folds(counts, n_folds=2, balance_sizes=(512,), min_positives={512: 30},
                     max_size_ratio=None)


def test_size_cap_stops_the_search_parking_specimens_in_one_fold() -> None:
    """Rate balance alone is won by one huge mixed fold; the cap forbids it."""
    counts = _counts({"CHN_A_1": (100, 10), "CHN_A_2": (100, 10),
                      "LHP_A_1": (1000, 500), "LHP_A_2": (1000, 0), "LHP_A_3": (100, 10)})
    free = assign_folds(counts, n_folds=2, balance_sizes=(512,), min_positives={},
                        max_size_ratio=None)
    capped = assign_folds(counts, n_folds=2, balance_sizes=(512,), min_positives={},
                          max_size_ratio=1.5)

    def sizes(assignment: dict[str, int]) -> list[int]:
        return [sum(counts.at[s, "tiles_512"] for s, f in assignment.items() if f == fold)
                for fold in (1, 2)]

    assert max(sizes(free)) / min(sizes(free)) > 1.5
    assert max(sizes(capped)) / min(sizes(capped)) <= 1.5


def test_anchor_count_must_match_fold_count() -> None:
    with pytest.raises(ValueError, match="exactly 3 CHN"):
        assign_folds(_eight_specimens(), n_folds=3, balance_sizes=(512,), min_positives={})


# ---------------------------------------------------------------------------
# Inner validation


def test_inner_val_is_never_the_anchor_and_skips_a_dominant_specimen() -> None:
    counts = _counts({
        "CHN_A_1": (100, 15),               # closest rate, but the fold's anchor
        "LHP_A_1": (100, 90),               # holds most positives: excluded
        "LHP_A_2": (100, 20),
        "LHP_A_3": (100, 2),
    })
    chosen = choose_inner_val(counts, {s: 1 for s in counts.index})
    assert chosen == {1: "LHP_A_2"}


def test_inner_val_refuses_a_fold_with_no_eligible_specimen() -> None:
    counts = _counts({"CHN_A_1": (100, 10), "LHP_A_1": (100, 10)})
    with pytest.raises(ValueError, match="no specimen eligible"):
        choose_inner_val(counts, {"CHN_A_1": 1, "LHP_A_1": 2})


# ---------------------------------------------------------------------------
# Per-fold split manifests


def _manifest() -> dict:
    counts = _eight_specimens()
    assignment = assign_folds(counts, n_folds=2, balance_sizes=(512,), min_positives={})
    return build_fold_manifest(counts, assignment, choose_inner_val(counts, assignment),
                               split_manifest_dir="splits/cv")


@pytest.mark.parametrize("held_out", [1, 2])
def test_fold_split_holds_every_specimen_on_one_side(held_out: int) -> None:
    manifest = _manifest()
    split = fold_split_manifest(manifest, held_out)
    by_specimen: dict[str, set[str]] = {}
    for entry in split["slides"].values():
        by_specimen.setdefault(entry["specimen"], set()).add(entry["split"])

    assert all(len(v) == 1 for v in by_specimen.values())
    folds = {f["fold"]: f for f in manifest["folds"]}
    assert {s for s, v in by_specimen.items() if v == {"test"}} == set(folds[held_out]["specimens"])
    assert {s for s, v in by_specimen.items() if v == {"val"}} == {
        f["inner_val_specimen"] for n, f in folds.items() if n != held_out
    }
    assert split["manifest_version"] == f"cv-v1-fold{held_out}"


def test_every_slide_is_tested_exactly_once_across_folds() -> None:
    manifest = _manifest()
    tested = [
        stem
        for fold in manifest["folds"]
        for stem, entry in fold_split_manifest(manifest, fold["fold"])["slides"].items()
        if entry["split"] == "test"
    ]
    all_slides = [stem for fold in manifest["folds"] for stem in fold["slides"]]
    assert sorted(tested) == sorted(all_slides)
    assert len(set(tested)) == len(tested)


def test_written_hashes_are_the_hashes_of_the_written_files(tmp_path: Path) -> None:
    manifest = _manifest()
    written = write_fold_manifests(manifest, "splits/fold_manifest.json", tmp_path)

    reloaded = json.loads(written[-1].read_text())
    for fold in reloaded["folds"]:
        path = tmp_path / fold["split_manifest_path"]
        assert fold["split_manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_fold_split_manifest_drives_the_tile_index(tmp_path: Path) -> None:
    """The per-fold manifest is consumed by load_tile_index exactly like v1."""
    stems = ("CHN_A_1_1-2", "LHP_A_1_1-2")
    for stem in stems:
        cut = f"{stem}_cut000"
        tile = {
            "tile_id": f"{cut}_s0512_r0000_c0000", "tile_size": 512,
            "cut_local_bbox": {"x0": 0, "y0": 0, "x1": 512, "y1": 512},
            "tissue_fraction": 0.9, "has_oocyte": False, "n_oocytes": 0,
            "png_path": f"data/tiles/{stem}/{cut}/0512/{cut}_s0512_r0000_c0000.png",
        }
        manifest = tmp_path / "tiles" / stem / cut / f"{cut}_tile_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"cut_name": cut, "stem": stem, "tiles": [tile]}))
        geojson = tmp_path / "cuts" / stem / f"{cut}_annotations.geojson"
        geojson.parent.mkdir(parents=True)
        geojson.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    paths = {"tiles_dir": tmp_path / "tiles", "cuts_dir": tmp_path / "cuts"}
    fold_manifest = {
        "manifest_version": "cv-v1", "created": "2026-09-25",
        "folds": [
            {"fold": 1, "slides": [stems[0]], "inner_val_specimen": "none"},
            {"fold": 2, "slides": [stems[1]], "inner_val_specimen": "LHP_A_1"},
        ],
    }
    split_path = tmp_path / "fold1.json"
    split_path.write_text(json.dumps(fold_split_manifest(fold_manifest, 1)))

    index = load_tile_index(**{**paths, "split_manifest_path": split_path})
    assert dict(index.groupby("stem")["split"].first()) == {stems[0]: "test", stems[1]: "val"}


# ---------------------------------------------------------------------------
# Real corpus -- pins the fold table in the M7b plan


@pytest.fixture(scope="module")
def real_counts() -> pd.DataFrame:
    return specimen_counts(load_tile_index(tile_sizes=(512, 1024)))


def test_real_corpus_has_14_specimens_from_26_slides(real_counts: pd.DataFrame) -> None:
    assert len(real_counts) == 14
    assert sum(len(s) for s in real_counts["slides"]) == 26
    assert (real_counts["site"] == ANCHOR_SITE).sum() == 4


def test_v1_split_has_the_five_straddling_specimens() -> None:
    manifest = json.loads((REPO_ROOT / "data/splits/split_manifest.json").read_text())
    straddling = straddling_specimens({s: e["split"] for s, e in manifest["slides"].items()})
    assert straddling == {
        "CHN_AU_10": ["test", "train"],
        "CHN_SP_5": ["train", "val"],
        "CHN_SU_9": ["train", "val"],
        "LHP_SU_9": ["train", "val"],
        "LHP_W_10": ["train", "val"],
    }


def test_real_corpus_fold_table(real_counts: pd.DataFrame) -> None:
    assignment = assign_folds(real_counts)
    inner_val = choose_inner_val(real_counts, assignment)
    manifest = build_fold_manifest(real_counts, assignment, inner_val)

    expected = {
        1: (["CHN_AU_10", "LHP_SP_3", "LHP_SU_3", "LHP_W_10"], 6, 9999, 1326, 2416, 441, "LHP_SP_3"),
        2: (["CHN_AU_8", "LHP_AU_5", "LHP_SP_6", "LHP_SU_9"], 7, 13561, 2658, 3385, 832, "LHP_SP_6"),
        3: (["CHN_SP_5", "LHP_SU_10", "LHP_W_17"], 7, 9835, 1290, 2443, 427, "LHP_W_17"),
        4: (["CHN_SU_9", "LHP_AU_9", "LHP_SP_2"], 6, 10798, 1491, 2633, 496, "LHP_AU_9"),
    }
    for fold in manifest["folds"]:
        specimens, n_slides, t512, p512, t1024, p1024, iv = expected[fold["fold"]]
        assert fold["specimens"] == specimens
        assert fold["n_slides"] == n_slides
        assert (fold["tiles"]["512"], fold["positives"]["512"]) == (t512, p512)
        assert (fold["tiles"]["1024"], fold["positives"]["1024"]) == (t1024, p1024)
        assert fold["inner_val_specimen"] == iv


def test_a_fold_without_tiles_at_a_recorded_size_gets_a_null_rate() -> None:
    counts = _counts({"CHN_A_1": (100, 10), "LHP_A_1": (100, 10)})
    counts["tiles_128"] = [50, 0]
    counts["positives_128"] = [5, 0]
    counts["ambiguous_128"] = [0, 0]
    manifest = build_fold_manifest(counts, {"CHN_A_1": 1, "LHP_A_1": 2},
                                   {1: "LHP_A_1", 2: "LHP_A_1"})
    rates = {f["fold"]: f["positive_rate"]["128"] for f in manifest["folds"]}
    assert rates == {1: 0.1, 2: None}
