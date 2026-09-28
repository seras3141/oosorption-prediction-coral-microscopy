"""Synthetic tests for the tile index: labels, split join, and path resolution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.tile_index import (
    INDEX_COLUMNS,
    binary_targets,
    load_tile_index,
    training_rows,
)

TILE_SIZE_DIR_WIDTH = 4


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp)


def _square(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _tile(cut: str, size: int, row: int, col: int, x0: int, y0: int, *, has_oocyte=False) -> dict:
    tile_id = f"{cut}_s{size:04d}_r{row:04d}_c{col:04d}"
    return {
        "tile_id": tile_id,
        "tile_size": size,
        "row": row,
        "col": col,
        "cut_local_bbox": {"x0": x0, "y0": y0, "x1": x0 + size, "y1": y0 + size},
        "pad_right": 0,
        "pad_bottom": 0,
        "tissue_fraction": 0.9,
        "has_oocyte": has_oocyte,
        "n_oocytes": 1 if has_oocyte else 0,
        "stages": ["Stage 0"] if has_oocyte else [],
        "annotations": [],
        "png_path": (
            f"data/tiles/{cut.rsplit('_cut', 1)[0]}/{cut}/"
            f"{size:0{TILE_SIZE_DIR_WIDTH}d}/{tile_id}.png"
        ),
    }


def _corpus(root: Path, *, stems=("SLIDE_A", "SLIDE_B"), splits=("train", "val")) -> dict:
    """Two slides, one cut each, with a 2000 px annotation at the origin.

    Tile (0, 0) is fully inside the annotation but holds no centroid, so the centroid
    rule calls it negative while the area rule calls it positive. Tile at 5000 px is
    clear of every annotation.
    """
    for stem in stems:
        cut = f"{stem}_cut000"
        tiles = [
            _tile(cut, 512, 0, 0, 0, 0, has_oocyte=False),
            _tile(cut, 512, 9, 9, 5000, 5000, has_oocyte=False),
            _tile(cut, 512, 1, 1, 900, 900, has_oocyte=True),
            _tile(cut, 128, 0, 0, 0, 0, has_oocyte=False),
        ]
        _write_json(
            root / "data" / "tiles" / stem / cut / f"{cut}_tile_manifest.json",
            {"cut_name": cut, "stem": stem, "tiles": tiles},
        )
        _write_json(
            root / "data" / "cuts" / stem / f"{cut}_annotations.geojson",
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Oosorption Stage 0"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": _square(0, 0, 2000, 2000),
                        },
                    }
                ],
            },
        )
    manifest = {
        "manifest_version": "v1",
        "slides": {
            stem: {"split": split, "location": "CHN", "stage_counts": {"0": 1}}
            for stem, split in zip(stems, splits)
        },
    }
    _write_json(root / "data" / "splits" / "split_manifest.json", manifest)
    return {
        "tiles_dir": root / "data" / "tiles",
        "cuts_dir": root / "data" / "cuts",
        "split_manifest_path": root / "data" / "splits" / "split_manifest.json",
    }


def test_index_has_the_declared_columns(tmp_path: Path) -> None:
    index = load_tile_index(**_corpus(tmp_path))
    assert list(index.columns) == list(INDEX_COLUMNS)


def test_only_requested_tile_sizes_are_indexed(tmp_path: Path) -> None:
    """The 128 px tiles in the fixture must not appear."""
    index = load_tile_index(**_corpus(tmp_path), tile_sizes=(512,))
    assert set(index["tile_size"]) == {512}
    assert len(index) == 6  # 3 tiles at 512 px, 2 slides


def test_split_join_follows_the_manifest(tmp_path: Path) -> None:
    index = load_tile_index(**_corpus(tmp_path), tile_sizes=(512,))
    assert dict(index.groupby("stem")["split"].first()) == {
        "SLIDE_A": "train",
        "SLIDE_B": "val",
    }
    # Every tile of a slide lands in that slide's split; no tile-level leakage.
    assert index.groupby("stem")["split"].nunique().eq(1).all()


def test_area_rule_labels_a_tile_the_centroid_rule_misses(tmp_path: Path) -> None:
    """The correction, end to end: interior tiles are positive under the area rule."""
    paths = _corpus(tmp_path)
    interior = "SLIDE_A_cut000_s0512_r0000_c0000"

    area = load_tile_index(**paths, tile_sizes=(512,)).set_index("tile_id")
    centroid = load_tile_index(**paths, tile_sizes=(512,), label_rule="centroid").set_index(
        "tile_id"
    )

    assert area.loc[interior, "oocyte_area_fraction"] == pytest.approx(1.0)
    assert area.loc[interior, "label"] == "positive"
    assert centroid.loc[interior, "label"] == "negative"


def test_centroid_rule_reports_no_coverage_and_no_ambiguity(tmp_path: Path) -> None:
    index = load_tile_index(**_corpus(tmp_path), tile_sizes=(512,), label_rule="centroid")
    assert index["oocyte_area_fraction"].isna().all()
    assert not index["is_ambiguous"].any()


def test_tiles_clear_of_annotations_are_negative(tmp_path: Path) -> None:
    index = load_tile_index(**_corpus(tmp_path), tile_sizes=(512,)).set_index("tile_id")
    assert index.loc["SLIDE_A_cut000_s0512_r0009_c0009", "label"] == "negative"
    assert index.loc["SLIDE_A_cut000_s0512_r0009_c0009", "oocyte_area_fraction"] == 0.0


def test_threshold_moves_the_ambiguous_band(tmp_path: Path) -> None:
    """A tile with 3% coverage is ambiguous by default and positive at a 1% threshold."""
    paths = _corpus(tmp_path, stems=("SLIDE_A",), splits=("train",))
    cut = "SLIDE_A_cut000"
    # Tile spans x 1985..2497 against a square ending at 2000: a 15 px wide, full-height
    # strip, so 15 * 512 / 512**2 = 2.9% coverage.
    sliver = _tile(cut, 512, 5, 5, 1985, 0)
    manifest_path = paths["tiles_dir"] / "SLIDE_A" / cut / f"{cut}_tile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tiles"].append(sliver)
    _write_json(manifest_path, manifest)

    default = load_tile_index(**paths, tile_sizes=(512,)).set_index("tile_id")
    lenient = load_tile_index(
        **paths, tile_sizes=(512,), min_oocyte_area_fraction=0.01
    ).set_index("tile_id")

    fraction = default.loc[sliver["tile_id"], "oocyte_area_fraction"]
    assert 0.01 < fraction < 0.05
    assert default.loc[sliver["tile_id"], "label"] == "ambiguous"
    assert bool(default.loc[sliver["tile_id"], "is_ambiguous"]) is True
    assert lenient.loc[sliver["tile_id"], "label"] == "positive"


def test_png_paths_use_zero_padded_size_directories(tmp_path: Path) -> None:
    index = load_tile_index(**_corpus(tmp_path), tile_sizes=(512, 1024))
    assert index["png_path"].str.contains("/0512/").all()
    assert not index["png_path"].str.contains("/512/").any()


def test_stem_missing_from_the_split_manifest_raises(tmp_path: Path) -> None:
    """An unassigned slide is a broken split, not a slide to drop quietly."""
    paths = _corpus(tmp_path)
    manifest_path = paths["split_manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["slides"]["SLIDE_B"]
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="SLIDE_B has tiles .* no entry in the split"):
        load_tile_index(**paths, tile_sizes=(512,))


def test_unknown_label_rule_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="label_rule must be one of"):
        load_tile_index(**_corpus(tmp_path), label_rule="whatever")


def test_no_manifests_raises(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    paths["tiles_dir"] = tmp_path / "empty"
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No tile manifests found"):
        load_tile_index(**paths)


def test_split_manifest_without_slides_raises(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    _write_json(paths["split_manifest_path"], {"manifest_version": "v1"})
    with pytest.raises(ValueError, match="has no slides"):
        load_tile_index(**paths)


def test_training_rows_drops_only_the_ambiguous_band() -> None:
    frame = pd.DataFrame(
        {
            "label": ["positive", "ambiguous", "negative"],
            "is_ambiguous": [False, True, False],
        }
    )
    kept = training_rows(frame)
    assert list(kept["label"]) == ["positive", "negative"]
    assert list(kept.index) == [0, 1]


def test_binary_targets_maps_positive_to_one() -> None:
    frame = pd.DataFrame(
        {"label": ["positive", "negative"], "is_ambiguous": [False, False]}
    )
    assert list(binary_targets(frame)) == [1, 0]


def test_binary_targets_refuses_ambiguous_rows() -> None:
    frame = pd.DataFrame({"label": ["ambiguous"], "is_ambiguous": [True]})
    with pytest.raises(ValueError, match="call training_rows"):
        binary_targets(frame)


def test_tile_index_does_not_require_torch() -> None:
    """The layering rule, enforced rather than documented.

    A fresh interpreter must be able to import the index and label modules with torch
    blocked, so the detection milestones can reuse the corrected label without the ML
    stack.
    """
    # find_spec, not find_module: meta-path find_module support was removed in Python
    # 3.12, so a find_module-based blocker is inert and the test would only be checking
    # sys.modules after the fact.
    program = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch is blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "try:\n"
        "    import torch\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('blocker is inert; the test proves nothing')\n"
        "import src.modeling.tile_index\n"
        "import src.modeling.tile_labels\n"
        "import src.modeling.specimen_groups\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_split_manifest_slide_without_tiles_raises(tmp_path: Path) -> None:
    """A partial data/ sync must not quietly shrink the evaluation splits."""
    paths = _corpus(tmp_path)
    manifest_path = paths["split_manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["slides"]["SLIDE_C"] = {
        "split": "test",
        "location": "LHP",
        "stage_counts": {"0": 1},
    }
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="SLIDE_C"):
        load_tile_index(**paths, tile_sizes=(512,))

    index = load_tile_index(**paths, tile_sizes=(512,), allow_missing_slides=True)
    assert "SLIDE_C" not in set(index["stem"])


def test_tile_size_matching_no_manifest_raises(tmp_path: Path) -> None:
    """A typo'd size would otherwise return an empty index and fail later, far from
    the cause."""
    with pytest.raises(ValueError, match=r"No tiles at sizes \[768\]"):
        load_tile_index(**_corpus(tmp_path), tile_sizes=(768,))


def test_slide_with_no_tiles_at_the_requested_size_is_not_counted_as_present(
    tmp_path: Path,
) -> None:
    """Otherwise a slide tiled only at other sizes vanishes from the index without
    tripping the evaluation-shrink guard."""
    paths = _corpus(tmp_path)
    # Strip SLIDE_B down to 128 px tiles only, then ask for 512.
    cut = "SLIDE_B_cut000"
    manifest_path = paths["tiles_dir"] / "SLIDE_B" / cut / f"{cut}_tile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tiles"] = [t for t in manifest["tiles"] if t["tile_size"] == 128]
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="SLIDE_B"):
        load_tile_index(**paths, tile_sizes=(512,))

    index = load_tile_index(**paths, tile_sizes=(512,), allow_missing_slides=True)
    assert set(index["stem"]) == {"SLIDE_A"}
