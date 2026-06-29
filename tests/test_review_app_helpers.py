"""Focused tests for review-app session helper functions."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import streamlit  # noqa: F401
except ModuleNotFoundError:
    sys.modules["streamlit"] = types.SimpleNamespace()

from app.review_tiles import (  # noqa: E402
    _join_tiles_and_labels,
    _load_review_session,
    _validate_review_session,
    _write_json_atomic,
)


def _session_meta() -> dict:
    return {
        "session_id": "review001",
        "tiles": [
            {
                "display_index": 0,
                "tile_id": "tile-a",
                "png_filename": "tiles/tile_001.png",
                "tile_size": 256,
                "ground_truth": True,
            }
        ],
    }


def _labels() -> dict:
    return {
        "session_id": "review001",
        "labels": [
            {
                "display_index": 0,
                "tile_id": "tile-a",
                "collaborator_label": None,
                "labelled_at": None,
            }
        ],
    }


def test_validate_review_session_rejects_mismatched_session_id() -> None:
    meta = _session_meta()
    labels = _labels()
    labels["session_id"] = "other"

    with pytest.raises(ValueError, match="different session_id"):
        _validate_review_session(meta, labels)


def test_join_tiles_and_labels_adds_collaborator_fields() -> None:
    meta = _session_meta()
    labels = _labels()
    labels["labels"][0]["collaborator_label"] = True
    labels["labels"][0]["labelled_at"] = "2026-05-27T13:00:00"

    session_data = _join_tiles_and_labels(meta, labels)

    assert session_data["tiles"][0]["ground_truth"] is True
    assert session_data["tiles"][0]["collaborator_label"] is True
    assert session_data["tiles"][0]["labelled_at"] == "2026-05-27T13:00:00"


def test_load_review_session_reads_split_files(tmp_path: Path) -> None:
    meta_path = tmp_path / "session_meta.json"
    labels_path = tmp_path / "labels.json"
    meta_path.write_text(json.dumps(_session_meta()), encoding="utf-8")
    labels_path.write_text(json.dumps(_labels()), encoding="utf-8")

    session_data = _load_review_session(meta_path, labels_path)

    assert session_data["session_id"] == "review001"
    assert session_data["tiles"][0]["collaborator_label"] is None


def test_write_json_atomic_replaces_target(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    _write_json_atomic(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert not (tmp_path / "labels.json.tmp").exists()
