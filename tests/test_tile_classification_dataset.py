"""Synthetic tests for the tensor-serving tile dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling import tile_classification_dataset as module
from src.modeling.tile_classification_dataset import TileClassificationDataset

TILE_PX = 8


def _write_png(path: Path, value: int) -> Path:
    """A flat single-colour tile, so pixel values are predictable."""
    array = np.full((TILE_PX, TILE_PX, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def _index(tmp_path: Path, labels: list[str]) -> pd.DataFrame:
    rows = []
    for position, label in enumerate(labels):
        png = _write_png(tmp_path / f"tile{position}.png", 255 if label == "positive" else 0)
        rows.append(
            {
                "tile_id": f"tile{position}",
                "png_path": str(png),
                "stem": "SLIDE_A",
                "cut_name": "SLIDE_A_cut000",
                "tile_size": 512,
                "n_oocytes": 1 if label == "positive" else 0,
                "tissue_fraction": 0.9,
                "oocyte_area_fraction": 1.0 if label == "positive" else 0.0,
                "label": label,
                "is_ambiguous": label == "ambiguous",
                "split": "train",
            }
        )
    return pd.DataFrame(rows)


def test_yields_chw_float_tensors_in_unit_range(tmp_path: Path) -> None:
    dataset = TileClassificationDataset(_index(tmp_path, ["positive"]))

    image, target = dataset[0]

    assert image.shape == (3, TILE_PX, TILE_PX)
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    assert image.allclose(torch.ones_like(image))
    assert target.dtype == torch.float32


def test_targets_follow_the_label(tmp_path: Path) -> None:
    dataset = TileClassificationDataset(_index(tmp_path, ["positive", "negative"]))

    assert float(dataset[0][1]) == 1.0
    assert float(dataset[1][1]) == 0.0


def test_length_matches_the_index(tmp_path: Path) -> None:
    dataset = TileClassificationDataset(_index(tmp_path, ["positive", "negative", "negative"]))
    assert len(dataset) == 3


def test_positive_and_negative_positions_partition_the_index(tmp_path: Path) -> None:
    """The forced-ratio sampler draws from these two pools."""
    dataset = TileClassificationDataset(
        _index(tmp_path, ["negative", "positive", "negative", "positive"])
    )

    assert dataset.positive_positions() == [1, 3]
    assert dataset.negative_positions() == [0, 2]
    assert len(dataset.positive_positions()) + len(dataset.negative_positions()) == len(dataset)


def test_ambiguous_rows_are_refused(tmp_path: Path) -> None:
    """Serving these as negatives would corrupt the negative pool."""
    with pytest.raises(ValueError, match="call training_rows"):
        TileClassificationDataset(_index(tmp_path, ["positive", "ambiguous"]))


def test_empty_index_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="index is empty"):
        TileClassificationDataset(_index(tmp_path, []))


def test_transform_is_applied(tmp_path: Path) -> None:
    calls: list[tuple[int, int]] = []

    def transform(image: Image.Image) -> torch.Tensor:
        calls.append(image.size)
        return torch.zeros(3, 4, 4)

    dataset = TileClassificationDataset(_index(tmp_path, ["positive"]), transform=transform)
    image, _ = dataset[0]

    assert calls == [(TILE_PX, TILE_PX)]
    assert image.shape == (3, 4, 4)


def test_relative_png_paths_resolve_against_the_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifests store png_path repo-relative, so the dataset must anchor it."""
    index = _index(tmp_path, ["positive"])
    absolute = Path(index.loc[0, "png_path"])
    index.loc[0, "png_path"] = absolute.name
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    image, _ = TileClassificationDataset(index)[0]

    assert image.shape == (3, TILE_PX, TILE_PX)


def test_index_property_is_positionally_aligned(tmp_path: Path) -> None:
    dataset = TileClassificationDataset(_index(tmp_path, ["negative", "positive"]))

    assert list(dataset.index["tile_id"]) == ["tile0", "tile1"]
    assert float(dataset[1][1]) == 1.0
    assert dataset.index.loc[1, "label"] == "positive"


def test_greyscale_and_rgba_tiles_are_converted_to_three_channels(tmp_path: Path) -> None:
    index = _index(tmp_path, ["positive"])
    Image.fromarray(np.full((TILE_PX, TILE_PX), 128, dtype=np.uint8), mode="L").save(
        index.loc[0, "png_path"]
    )

    image, _ = TileClassificationDataset(index)[0]

    assert image.shape == (3, TILE_PX, TILE_PX)
    assert float(image[0, 0, 0]) == pytest.approx(128 / 255, abs=1e-6)


def test_mixed_tile_sizes_are_refused(tmp_path: Path) -> None:
    """load_tile_index indexes 512 and 1024 together and nothing here resizes, so a
    mixed index must fail at construction rather than one batch into training."""
    index = _index(tmp_path, ["positive", "negative"])
    index.loc[1, "tile_size"] = 1024

    with pytest.raises(ValueError, match="mixes tile sizes"):
        TileClassificationDataset(index)
