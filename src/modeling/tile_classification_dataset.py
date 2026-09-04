"""Serve indexed tiles as tensors for the tile-level classifier.

This is the torch-dependent half of the data layer; :mod:`src.modeling.tile_index`
builds the index it consumes and stays importable without torch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.modeling.tile_index import binary_targets

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


class TileClassificationDataset(Dataset):
    """Map-style dataset over a tile index, yielding ``(image, target)`` pairs.

    Parameters
    ----------
    index : pandas.DataFrame
        Rows from :func:`src.modeling.tile_index.load_tile_index`, with the ambiguous
        band already dropped via
        :func:`src.modeling.tile_index.training_rows`.
    transform : callable, optional
        Applied to the loaded ``PIL.Image``; must return a tensor. When omitted the
        image is converted to a float tensor in ``[0, 1]`` with no resizing, so the
        caller sees the tile at its native size rather than a silent default.

    Raises
    ------
    ValueError
        If ``index`` is empty, still carries ambiguous rows, or mixes tile sizes. The
        default transform does not resize, and ``load_tile_index`` indexes 512 px and
        1024 px together, so a mixed index would reach the default collate and fail
        there with a shape error one batch into training rather than at construction.

    Notes
    -----
    Tiles overlap by 20% and one oocyte spans several tiles, so rows are not
    independent samples. That does not affect training, but any evaluation resampling
    over this dataset must group by slide rather than by tile.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if index.empty:
            raise ValueError("index is empty; nothing to serve")
        sizes = sorted(index["tile_size"].unique())
        if len(sizes) > 1:
            raise ValueError(
                f"index mixes tile sizes {sizes}; the default transform does not "
                "resize, so batching them would fail in collate. Build one dataset "
                "per tile size."
            )
        # binary_targets raises on leftover ambiguous rows, which is the check we want
        # here too -- serving them as negatives would corrupt the negative pool.
        self._targets = binary_targets(index).to_numpy()
        self._index = index.reset_index(drop=True)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._index.iloc[position]
        path = Path(row["png_path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            tensor = self._transform(rgb) if self._transform else _to_tensor(rgb)
        target = torch.tensor(float(self._targets[position]), dtype=torch.float32)
        return tensor, target

    @property
    def index(self) -> pd.DataFrame:
        """The index backing this dataset, positionally aligned with ``__getitem__``."""
        return self._index

    def positive_positions(self) -> list[int]:
        """Positions of the positive rows, for the forced-ratio batch sampler."""
        return [int(pos) for pos in (self._targets == 1).nonzero()[0]]

    def negative_positions(self) -> list[int]:
        """Positions of the negative rows, for the forced-ratio batch sampler."""
        return [int(pos) for pos in (self._targets == 0).nonzero()[0]]


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB image to a CHW float tensor in ``[0, 1]``.

    Avoids a torchvision dependency in the data layer; the training arms bring their own
    transforms.
    """
    array = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(array.copy())
    return tensor.permute(2, 0, 1).float().div_(255.0)
