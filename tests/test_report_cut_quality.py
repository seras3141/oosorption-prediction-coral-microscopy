from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import report_cut_quality
from src.modeling.cut_exclusions import LEGACY_CORPUS
from src.modeling import tile_index
from src.modeling.tile_index import LABEL_RULE_CENTROID

STEM = "S_A_1_1-2"


def _tiles(tmp_path: Path) -> pd.DataFrame:
    flat = np.tile(np.array((120, 60, 140), dtype=np.uint8), (32, 32, 1))
    board = np.tile(np.array((60, 20, 80), dtype=np.uint8), (32, 32, 1))
    board[::2, ::2] = (160, 60, 180)
    rows = []
    for index, image in enumerate((flat, flat, flat, board, board, board, board)):
        cut = f"{STEM}_cut{index:03d}"
        path = tmp_path / "tiles" / f"{cut}_t0.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)
        rows.append({"tile_id": f"{cut}_t0", "png_path": str(path), "stem": STEM,
                     "cut_name": cut, "tile_size": 512})
    return pd.DataFrame(rows)


def test_the_report_measures_listed_cuts_and_names_unlisted_flagged_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    requested: dict[str, Any] = {}

    def fake_index(**kwargs: Any) -> pd.DataFrame:
        requested.update(kwargs)
        return _tiles(tmp_path)

    monkeypatch.setattr(report_cut_quality, "load_tile_index", fake_index)
    exclusions = tmp_path / "cut_exclusions.csv"
    exclusions.write_text(f"cut_name,status\n{STEM}_cut000,excluded\n{STEM}_cut001,known_empty\n"
                          f"{STEM}_cut009,excluded\n")
    output = tmp_path / "out"
    with caplog.at_level(logging.WARNING):
        assert report_cut_quality.main(["--cut-exclusions", str(exclusions), "--output-dir",
                                        str(output), "--min-sharpness-tiles", "1"]) == 0

    assert requested["cut_exclusions"] is LEGACY_CORPUS
    assert requested["tile_sizes"] == (512,)
    assert requested["label_rule"] == LABEL_RULE_CENTROID
    report = pd.read_csv(output / "cut_quality.csv", keep_default_na=False).set_index("cut_name")
    assert report["flag"].to_dict() == {
        f"{STEM}_cut000": "blurred", f"{STEM}_cut001": "blurred", f"{STEM}_cut002": "blurred",
        f"{STEM}_cut003": "", f"{STEM}_cut004": "", f"{STEM}_cut005": "", f"{STEM}_cut006": ""}
    assert report["listed"].to_dict()[f"{STEM}_cut000"] == "excluded"
    markdown = (output / "cut_quality.md").read_text()
    assert f"| {STEM}_cut000 | 0.00000 | 0.00 |" in markdown
    assert "cuts not excluded with at least 1 tiles" in markdown
    warned = [record.getMessage() for record in caplog.records]
    assert warned == [
        f"{STEM}_cut001 looks out of focus (0.00 x the median sharpness) and is not excluded",
        f"{STEM}_cut002 looks out of focus (0.00 x the median sharpness) and is not excluded",
        f"{STEM}_cut009 listed in the cut exclusions but no 512 px tile belongs to it; a typo?",
    ]


def test_a_cut_with_too_few_sharpness_tiles_is_named_for_a_look_not_for_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tiles = _tiles(tmp_path)
    thin = np.tile(np.array((250, 250, 250), dtype=np.uint8), (32, 32, 1))
    thin[:, 10:13] = (200, 100, 50)
    Image.fromarray(thin).save(tiles["png_path"].iloc[0])
    monkeypatch.setattr(report_cut_quality, "load_tile_index", lambda **kwargs: tiles)
    with caplog.at_level(logging.WARNING):
        report_cut_quality.main(["--cut-exclusions", "none", "--output-dir",
                                 str(tmp_path / "out"), "--min-sharpness-tiles", "1"])
    assert (f"{STEM}_cut000 has 0 tiles that give a sharpness, too few to judge; look at it"
            in [record.getMessage() for record in caplog.records])


def test_a_relative_output_dir_is_under_the_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report_cut_quality, "load_tile_index", lambda **kwargs: _tiles(tmp_path))
    monkeypatch.setattr(tile_index, "REPO_ROOT", tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    report_cut_quality.main(["--cut-exclusions", "none", "--output-dir", "report",
                             "--min-sharpness-tiles", "1"])
    assert (tmp_path / "report" / "cut_quality.csv").is_file()
    assert "no exclusion list was given" in (tmp_path / "report" / "cut_quality.md").read_text()


@pytest.mark.parametrize("arguments, message", [
    (["--cutoff-fraction", "50"], "between 0 and 1"),
    (["--cutoff-fraction", "0"], "between 0 and 1"),
    (["--per-cut", "5"], "fewer tiles than --min-sharpness-tiles 10"),
    (["--per-cut", "0", "--min-sharpness-tiles", "0"], "must be at least 1, got 0"),
])
def test_arguments_that_cannot_give_a_meaningful_report_are_refused(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        report_cut_quality.parse_args(arguments)
    assert message in capsys.readouterr().err
