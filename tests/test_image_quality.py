from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling import tile_index
from src.modeling.cut_exclusions import CutExclusions
from src.qc.image_quality import cut_image_quality, flag_cuts, tile_image_stats


def _tile(colour: tuple[int, int, int], size: int = 32) -> np.ndarray:
    return np.tile(np.array(colour, dtype=np.uint8), (size, size, 1))


def test_a_flat_tile_has_no_sharpness_and_a_checkerboard_has_some() -> None:
    flat = tile_image_stats(_tile((120, 60, 140)))
    assert flat["sharpness"] == 0.0
    board = _tile((40, 40, 40))
    board[::2, ::2] = 140
    assert tile_image_stats(board)["sharpness"] > 0


def test_tile_stats_measure_tissue_pixels_only() -> None:
    image = _tile((250, 250, 250))
    image[:, :16] = (200, 100, 50)
    stats = tile_image_stats(image)
    assert stats["red"] == pytest.approx(200 / 255)
    assert stats["blue"] == pytest.approx(50 / 255)
    assert stats["saturation"] == pytest.approx(150 / 200)
    assert stats["brightness"] == pytest.approx((0.299 * 200 + 0.587 * 100 + 0.114 * 50) / 255, abs=1e-6)
    assert tile_image_stats(_tile((250, 250, 250))) is None


def _quality_tiles(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for cut, colour in (("S_A_1_1-2_cut000", (200, 100, 50)), ("S_A_1_1-2_cut001", (100, 50, 150))):
        for i in range(5):
            tile_id = f"{cut}_t{i}"
            path = tmp_path / "S_A_1_1-2" / cut / "0512" / f"{tile_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_tile(colour)).save(path)
            rows.append({"tile_id": tile_id, "png_path": str(path), "stem": "S_A_1_1-2",
                         "cut_name": cut, "tile_size": 512})
    return pd.DataFrame(rows)


def test_cut_quality_samples_each_cut_with_the_seed_and_ranks_within_the_table(tmp_path: Path) -> None:
    tiles = _quality_tiles(tmp_path)
    tiles = tiles[(tiles["cut_name"] == "S_A_1_1-2_cut000") | ~tiles["tile_id"].str.endswith(("t3", "t4"))]
    table = cut_image_quality(tiles, per_cut=4, seed=1).set_index("cut_name")
    assert table["tiles_measured"].to_dict() == {"S_A_1_1-2_cut000": 4, "S_A_1_1-2_cut001": 3}
    assert table.loc["S_A_1_1-2_cut000", "red"] == pytest.approx(200 / 255)
    assert table["red_percentile"].to_dict() == {"S_A_1_1-2_cut000": 100.0, "S_A_1_1-2_cut001": 50.0}
    with pytest.raises(ValueError, match="more than once"):
        cut_image_quality(pd.concat([tiles, tiles]))


def test_the_sample_depends_on_the_seed_only(tmp_path: Path) -> None:
    tiles = _quality_tiles(tmp_path)
    for i, path in enumerate(tiles["png_path"]):
        Image.fromarray(_tile((10 * i, 50, 50))).save(path)
    first = cut_image_quality(tiles, per_cut=2, seed=1)
    assert cut_image_quality(tiles.iloc[::-1], per_cut=2, seed=1).equals(first)
    assert not cut_image_quality(tiles, per_cut=2, seed=3).equals(first)


def test_a_relative_png_path_is_read_from_the_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiles = _quality_tiles(tmp_path)
    monkeypatch.setattr(tile_index, "REPO_ROOT", tmp_path)
    relative = tiles["png_path"].map(lambda path: str(Path(path).relative_to(tmp_path)))
    table = cut_image_quality(tiles.assign(png_path=relative)).set_index("cut_name")
    assert table.loc["S_A_1_1-2_cut000", "red"] == pytest.approx(200 / 255)


def test_tiles_of_different_sizes_are_refused(tmp_path: Path) -> None:
    tiles = _quality_tiles(tmp_path)
    tiles.loc[0, "tile_size"] = 1024
    with pytest.raises(ValueError, match=r"mix tile sizes \[512, 1024\]"):
        cut_image_quality(tiles)


def test_sharpness_ignores_the_background_share_of_a_tile() -> None:
    board = _tile((40, 40, 40), size=64)
    board[::2, ::2] = 140
    half = board.copy()
    half[:, 32:] = 250
    full, partial = tile_image_stats(board)["sharpness"], tile_image_stats(half)["sharpness"]
    assert partial == pytest.approx(full, rel=0.05)


def test_black_padding_is_not_tissue() -> None:
    image = _tile((250, 250, 250), size=64)
    image[:, :32] = (40, 40, 40)
    image[:, :32][::2, ::2] = 140
    padded = image.copy()
    padded[:, 32:] = 0
    plain, with_padding = tile_image_stats(image), tile_image_stats(padded)
    assert plain["sharpness"] > 0
    assert with_padding["brightness"] == pytest.approx(plain["brightness"])
    assert with_padding["sharpness"] == pytest.approx(plain["sharpness"], rel=0.05)


def test_sharpness_tiles_count_only_tiles_that_give_a_sharpness(tmp_path: Path) -> None:
    tiles = _quality_tiles(tmp_path).head(2)
    thin = _tile((250, 250, 250))
    thin[:, 10:13] = (200, 100, 50)
    Image.fromarray(thin).save(tiles["png_path"].iloc[0])
    table = cut_image_quality(tiles)
    assert table.loc[0, ["tiles_measured", "sharpness_tiles"]].tolist() == [2, 1]


def test_a_flat_tissue_block_on_background_has_no_sharpness() -> None:
    image = _tile((250, 250, 250))
    image[8:24, 8:24] = (120, 60, 140)
    assert tile_image_stats(image)["sharpness"] == pytest.approx(0.0)


def test_thin_tissue_keeps_its_colour_but_has_no_sharpness() -> None:
    image = _tile((250, 250, 250))
    image[:, 10:13] = (200, 100, 50)
    stats = tile_image_stats(image)
    assert stats["red"] == pytest.approx(200 / 255)
    assert np.isnan(stats["sharpness"])


def _quality(rows: list[tuple[str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame([{"cut_name": cut, "tiles_measured": 60, "sharpness_tiles": tiles,
                          "sharpness": sharpness} for cut, sharpness, tiles in rows])


def test_a_cut_is_blurred_below_the_cutoff_of_the_median_of_cuts_with_enough_tiles() -> None:
    quality = _quality([("S_A_1_1-2_cut000", 0.49, 60), ("S_A_1_1-2_cut001", 0.5, 10),
                        ("S_A_1_1-2_cut002", 1.0, 60), ("S_A_1_1-2_cut003", 1.0, 60),
                        ("S_A_1_1-2_cut004", 1.0, 60), ("S_A_1_1-2_cut005", 0.01, 9),
                        ("S_A_1_1-2_cut006", np.nan, 0)])
    report = flag_cuts(quality, CutExclusions(), cutoff_fraction=0.5, min_sharpness_tiles=10)
    assert report["cut_name"].tolist() == [f"S_A_1_1-2_cut00{i}" for i in (6, 5, 0, 1, 2, 3, 4)]
    assert report.set_index("cut_name")["flag"].to_dict() == {
        "S_A_1_1-2_cut006": "too few tiles", "S_A_1_1-2_cut005": "too few tiles",
        "S_A_1_1-2_cut000": "blurred", "S_A_1_1-2_cut001": "",
        "S_A_1_1-2_cut002": "", "S_A_1_1-2_cut003": "", "S_A_1_1-2_cut004": ""}
    assert report.set_index("cut_name").loc["S_A_1_1-2_cut001", "sharpness_vs_median"] == 0.5


def test_each_cut_carries_its_status_in_the_exclusion_list() -> None:
    quality = _quality([("S_A_1_1-2_cut000", 0.1, 60), ("S_A_1_1-2_cut001", 1.0, 60),
                        ("S_A_1_1-2_cut002", 1.0, 60)])
    exclusions = CutExclusions(excluded=frozenset({"S_A_1_1-2_cut000"}),
                               known_empty=frozenset({"S_A_1_1-2_cut001"}), applied=True)
    report = flag_cuts(quality, exclusions)
    assert report.set_index("cut_name")["listed"].to_dict() == {
        "S_A_1_1-2_cut000": "excluded", "S_A_1_1-2_cut001": "known_empty", "S_A_1_1-2_cut002": ""}


def test_excluded_cuts_do_not_set_the_median() -> None:
    quality = _quality([("S_A_1_1-2_cut000", 0.01, 60), ("S_A_1_1-2_cut001", 0.01, 60),
                        ("S_A_1_1-2_cut002", 0.45, 60), ("S_A_1_1-2_cut003", 1.0, 60),
                        ("S_A_1_1-2_cut004", 1.0, 60)])
    exclusions = CutExclusions(excluded=frozenset({"S_A_1_1-2_cut000", "S_A_1_1-2_cut001"}),
                               applied=True)
    report = flag_cuts(quality, exclusions, cutoff_fraction=0.5).set_index("cut_name")
    assert report.loc["S_A_1_1-2_cut002", "flag"] == "blurred"
    assert report.loc["S_A_1_1-2_cut003", "sharpness_vs_median"] == 1.0


@pytest.mark.parametrize("sharpness, tiles", [(0.0, 60), (1.0, 3)])
def test_without_a_reference_sharpness_nothing_is_judged(sharpness: float, tiles: int) -> None:
    quality = _quality([("S_A_1_1-2_cut000", sharpness, tiles),
                        ("S_A_1_1-2_cut001", sharpness, tiles)])
    with pytest.raises(ValueError, match="no reference sharpness"):
        flag_cuts(quality, CutExclusions())


def test_a_cut_without_a_sharpness_is_never_judged_sharp() -> None:
    quality = _quality([("S_A_1_1-2_cut000", np.nan, 0), ("S_A_1_1-2_cut001", 1.0, 60)])
    report = flag_cuts(quality, CutExclusions(), min_sharpness_tiles=0).set_index("cut_name")
    assert report.loc["S_A_1_1-2_cut000", "flag"] == "too few tiles"
