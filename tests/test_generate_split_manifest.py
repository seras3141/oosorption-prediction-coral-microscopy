"""Tests for the slide-level split manifest generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from src.data_preparation.generate_split_manifest import (
    assign_splits,
    compute_stage_distribution,
)


def _feature(name: str) -> dict:
    return {
        "type": "Feature",
        "properties": {"name": name},
        "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 0], [0, 1], [0, 0]]},
    }


def _write_geojson(path: Path, feature_names: list[str]) -> None:
    fc = {"type": "FeatureCollection", "features": [_feature(n) for n in feature_names]}
    with path.open("w", encoding="utf-8") as fp:
        json.dump(fc, fp)


def test_compute_stage_distribution_synthetic(tmp_path: Path):
    _write_geojson(tmp_path / "LHP_W_1_1-2.geojson", ["Oosorption Stage 1", "Oosorption Stage 0"])
    _write_geojson(tmp_path / "CHN_AU_2_3-4.geojson", ["Oosorption Stage 0", "Oosorption Stage 0"])
    _write_geojson(
        tmp_path / "LHP_SU_3_5-6.geojson",
        ["Oosorption Stage 0", "Oosorption Stage 2", "Oosorption Stage 3"],
    )

    dist = compute_stage_distribution(tmp_path)

    assert dist["LHP_W_1_1-2"]["location"] == "LHP"
    assert dist["LHP_W_1_1-2"]["stage_counts"] == {1: 1, 0: 1}

    assert dist["CHN_AU_2_3-4"]["location"] == "CHN"
    assert dist["CHN_AU_2_3-4"]["stage_counts"] == {0: 2}

    assert dist["LHP_SU_3_5-6"]["location"] == "LHP"
    assert dist["LHP_SU_3_5-6"]["stage_counts"] == {0: 1, 2: 1, 3: 1}


def test_compute_stage_distribution_skips_unlabelled(tmp_path: Path):
    _write_geojson(
        tmp_path / "CHN_AU_1_1-2.geojson",
        ["Oosorption Stage 0", "no stage information here"],
    )

    dist = compute_stage_distribution(tmp_path)

    assert dist["CHN_AU_1_1-2"]["stage_counts"] == {0: 1}


def test_compute_stage_distribution_skips_out_of_range_stage(tmp_path: Path, caplog):
    _write_geojson(
        tmp_path / "CHN_AU_1_1-2.geojson",
        ["Oosorption Stage 0", "Oosorption Stage 12"],
    )

    with caplog.at_level("WARNING"):
        dist = compute_stage_distribution(tmp_path)

    assert dist["CHN_AU_1_1-2"]["stage_counts"] == {0: 1}
    assert any("outside the valid 0-4 range" in rec.message for rec in caplog.records)


def test_assign_splits_forces_scarce_stage_to_train():
    dist = {
        "LHP_A_1_1-2": {"location": "LHP", "stage_counts": {1: 1, 0: 5}},
        "LHP_A_2_1-2": {"location": "LHP", "stage_counts": {0: 5}},
        "LHP_A_3_1-2": {"location": "LHP", "stage_counts": {0: 5}},
    }
    quotas = {("LHP", False): (1, 1)}

    for seed in (0, 1, 42):
        result = assign_splits(dist, quotas=quotas, seed=seed)
        assert result["LHP_A_1_1-2"] == "train"


def test_assign_splits_quota_exact():
    dist = {
        f"CHN_A_{i}_1-2": {"location": "CHN", "stage_counts": {0: 1}} for i in range(4)
    }
    quotas = {("CHN", False): (1, 1)}

    result = assign_splits(dist, quotas=quotas, seed=42)

    counts = {"train": 0, "val": 0, "test": 0}
    for split in result.values():
        counts[split] += 1
    assert counts == {"train": 2, "val": 1, "test": 1}


def test_assign_splits_deterministic():
    dist = {
        f"LHP_A_{i}_1-2": {"location": "LHP", "stage_counts": {0: 1, 2: i % 2}}
        for i in range(6)
    }
    quotas = {("LHP", False): (1, 1), ("LHP", True): (1, 0)}

    result_a = assign_splits(dist, quotas=quotas, seed=42)
    result_b = assign_splits(dist, quotas=quotas, seed=42)

    assert result_a == result_b


def test_assign_splits_missing_stratum_quota_raises():
    dist = {
        "LHP_A_1_1-2": {"location": "LHP", "stage_counts": {0: 5}},
    }
    with pytest.raises(KeyError):
        assign_splits(dist, quotas={}, seed=42)


def test_assign_splits_real_corpus_golden():
    dist = compute_stage_distribution(REPO_ROOT / "data" / "dataset_28_04")
    result = assign_splits(dist, seed=42)

    expected_train = {
        "CHN_AU_10_31-33", "CHN_AU_8_40-42", "CHN_AU_8_7-9", "CHN_SP_5_13-15",
        "CHN_SP_5_22-24", "CHN_SP_5_37-39", "CHN_SP_5_58-60", "CHN_SU_9_34-36",
        "CHN_SU_9_37-39", "CHN_SU_9_4-6", "LHP_AU_9_10-12", "LHP_SP_2_34-36",
        "LHP_SP_3_25-27", "LHP_SP_6_3-4", "LHP_SU_10_52-54", "LHP_SU_9_25-27",
        "LHP_SU_9_40-42", "LHP_W_10_10-12", "LHP_W_17_22-24",
    }
    expected_val = {"CHN_SP_5_25-27", "CHN_SU_9_10-12", "LHP_SU_9_13-15", "LHP_W_10_28-30"}
    expected_test = {"CHN_AU_10_19-21", "LHP_AU_5_16-18", "LHP_SU_3_22-24"}

    actual_train = {s for s, split in result.items() if split == "train"}
    actual_val = {s for s, split in result.items() if split == "val"}
    actual_test = {s for s, split in result.items() if split == "test"}

    assert actual_train == expected_train
    assert actual_val == expected_val
    assert actual_test == expected_test
