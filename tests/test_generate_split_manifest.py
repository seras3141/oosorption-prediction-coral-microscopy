"""Tests for the slide-level split manifest generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.generate_split_manifest import compute_stage_distribution


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
