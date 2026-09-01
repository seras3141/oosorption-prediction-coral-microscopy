"""Tests for the shared stage-extraction helper."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.annotations import extract_stage


def test_extract_stage_from_name():
    assert extract_stage({"name": "Oosorption Stage 0"}) == 0


def test_extract_stage_no_matching_field():
    assert extract_stage({"foo": "bar"}) is None


def test_extract_stage_from_classification_name():
    assert extract_stage({"classification": {"name": "Stage 3"}}) == 3


def test_extract_stage_metadata_takes_priority_over_name():
    # Regression case from the real conflicting feature in
    # data/dataset_28_04/CHN_SP_5_13-15.geojson: metadata.ANNOTATION_DESCRIPTION
    # must win over `name` when both are present and disagree.
    props = {
        "metadata": {"ANNOTATION_DESCRIPTION": "Oosorption Stage 0"},
        "name": "Oosorption Stage 3",
    }
    assert extract_stage(props) == 0
