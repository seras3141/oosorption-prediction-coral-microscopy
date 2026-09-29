"""Embedding cache: build, pooling, refusals, and training a head from it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling.embedding_cache as embedding_cache
import src.modeling.encoders as encoders
from src.modeling.embedding_cache import (
    CachedEmbeddingDataset,
    build_cache,
    check_covers,
    load_cache,
)
from src.modeling.encoders import EmbeddingHead, FrozenEncoderClassifier

EMBED_DIM = 8
ENCODER = "owkin/phikon-v2"


class _StubEncoder(nn.Module):

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        channel_means = images.mean(dim=(2, 3))
        return channel_means.repeat(1, EMBED_DIM // 3 + 1)[:, :EMBED_DIM]


@pytest.fixture
def stub_encoder(monkeypatch: pytest.MonkeyPatch):
    def _load(encoder_name: str, tile_size: int, **kwargs) -> FrozenEncoderClassifier:
        encoders.encoder_input_px(tile_size)
        return FrozenEncoderClassifier(_StubEncoder(), EMBED_DIM, tile_size=tile_size)

    monkeypatch.setattr(encoders, "load_encoder", _load)
    monkeypatch.setattr(embedding_cache, "local_encoder_revision", lambda name: "rev-a")


def _tiles(tmp_path: Path, tile_size: int, n: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n):
        array = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        half = tile_size // 2
        array[:half, :half] = (10 * i, 0, 0)
        array[:half, half:] = (0, 200, 0)
        array[half:, :half] = (0, 0, 100)
        array[half:, half:] = (250, 250, 250)
        path = tmp_path / "tiles" / f"t{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)
        rows.append({
            "tile_id": f"S_A_1_1-2_cut000_s{tile_size:04d}_r0000_c{i:04d}",
            "png_path": str(path), "stem": "S_A_1_1-2", "tile_size": tile_size,
            "label": "negative", "is_ambiguous": False,
        })
    return pd.DataFrame(rows)


def test_1024_cache_keeps_quadrants_and_mean_matches_the_live_arm(
    tmp_path: Path, stub_encoder
) -> None:
    tiles = _tiles(tmp_path, 1024)
    path = build_cache(ENCODER, 1024, tmp_path / "cache", batch_size=4, num_workers=0,
                       tiles=tiles, device="cpu")
    cache = load_cache(path, 1024, ENCODER)
    assert cache.embeddings.shape == (6, 4, EMBED_DIM)

    live = FrozenEncoderClassifier(_StubEncoder(), EMBED_DIM, tile_size=1024)
    from src.modeling.train_tile_classifier import build_transforms

    transform = build_transforms(train=False, input_px=448, mean=encoders.encoder_spec(ENCODER).mean,
                                 std=encoders.encoder_spec(ENCODER).std)
    images = torch.stack([transform(Image.open(p).convert("RGB")) for p in tiles["png_path"]])
    with torch.no_grad():
        expected_mean = live._embed(images).numpy()
    ids = list(tiles["tile_id"])
    np.testing.assert_allclose(cache.pooled(ids, "mean"), expected_mean, rtol=1e-5, atol=1e-6)
    assert not np.allclose(cache.pooled(ids, "max"), cache.pooled(ids, "mean"))


def test_512_cache_holds_one_sub_tile(tmp_path: Path, stub_encoder) -> None:
    path = build_cache(ENCODER, 512, tmp_path / "cache", batch_size=4, num_workers=0,
                       tiles=_tiles(tmp_path, 512), device="cpu")
    cache = load_cache(path, 512, ENCODER)
    assert cache.embeddings.shape == (6, 1, EMBED_DIM)
    ids = list(cache.rows)
    np.testing.assert_array_equal(cache.pooled(ids, "mean"), cache.pooled(ids, "max"))


def test_128_px_is_refused_before_encoding(tmp_path: Path, stub_encoder) -> None:
    with pytest.raises(ValueError, match="supports tile sizes"):
        build_cache(ENCODER, 128, tmp_path / "cache", tiles=_tiles(tmp_path, 128), device="cpu")


@pytest.fixture
def built(tmp_path: Path, stub_encoder) -> tuple[Path, pd.DataFrame]:
    tiles = _tiles(tmp_path, 512)
    return build_cache(ENCODER, 512, tmp_path / "cache", batch_size=4, num_workers=0,
                       tiles=tiles, device="cpu"), tiles


def test_a_cache_for_another_size_or_encoder_is_refused(built) -> None:
    path, _ = built
    with pytest.raises(ValueError, match="not 1024 px"):
        load_cache(path, 1024, ENCODER)
    with pytest.raises(ValueError, match="not bioptimus"):
        load_cache(path, 512, "bioptimus/H-optimus-0")


def test_a_cache_from_other_weights_is_refused(built, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = built
    monkeypatch.setattr(embedding_cache, "local_encoder_revision", lambda name: "rev-b")
    with pytest.raises(ValueError, match="local weights are"):
        load_cache(path, 512, ENCODER)


def test_an_edited_tile_list_is_refused(built) -> None:
    path, _ = built
    listing = (path / "tiles.csv").read_text().replace("c0000", "c9999")
    (path / "tiles.csv").write_text(listing)
    with pytest.raises(ValueError, match="does not match the hash"):
        load_cache(path, 512, ENCODER)


def test_a_stale_cache_is_refused_for_a_moved_or_missing_tile(built) -> None:
    path, tiles = built
    cache = load_cache(path, 512, ENCODER)
    moved = tiles.assign(png_path=tiles["png_path"].str.replace("tiles", "elsewhere"))
    with pytest.raises(ValueError, match="now points at"):
        check_covers(cache, moved)
    extra = pd.concat([tiles, tiles.head(1).assign(tile_id="not-cached")])
    with pytest.raises(ValueError, match="not in the cache"):
        check_covers(cache, extra)


def test_the_dataset_serves_embeddings_with_the_index_targets(built) -> None:
    path, tiles = built
    labelled = tiles.assign(label=["positive", "negative"] * 3)
    dataset = CachedEmbeddingDataset(labelled, load_cache(path, 512, ENCODER))
    embedding, target = dataset[0]
    assert embedding.shape == (EMBED_DIM,)
    assert target.item() == 1.0
    assert dataset.positive_positions() == [0, 2, 4]


def test_cached_and_live_heads_share_checkpoint_keys() -> None:
    live = FrozenEncoderClassifier(_StubEncoder(), EMBED_DIM, tile_size=512)
    cached = EmbeddingHead(EMBED_DIM)
    assert live.head.state_dict().keys() == cached.head.state_dict().keys()
    cached.head.load_state_dict(live.head.state_dict())


def test_config_binds_the_cache_and_keeps_old_run_ids(built, tmp_path: Path) -> None:
    from src.modeling.train_tile_classifier import TrainingConfig

    path, _ = built
    plain = TrainingConfig(architecture="frozen_encoder")
    cached = TrainingConfig(architecture="frozen_encoder", embedding_cache_dir=str(path))
    assert plain.embedding_cache_sha256 is None
    assert cached.embedding_cache_sha256 is not None
    assert cached.run_id() != plain.run_id()

    (path / "meta.json").write_text((path / "meta.json").read_text().replace("none", "flips"))
    from dataclasses import asdict

    with pytest.raises(ValueError, match="rebuilt since this run was trained"):
        TrainingConfig(**asdict(cached))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"architecture": "resnet18", "embedding_cache_dir": "x"}, "frozen_encoder"),
        ({"architecture": "frozen_encoder", "quadrant_pooling": "max"}, "only configurable"),
        ({"architecture": "frozen_encoder", "quadrant_pooling": "median"}, "mean or max"),
    ],
)
def test_invalid_cache_options_are_refused(kwargs: dict, match: str) -> None:
    from src.modeling.train_tile_classifier import TrainingConfig

    with pytest.raises(ValueError, match=match):
        TrainingConfig(**kwargs)


def test_max_pooling_is_a_distinct_run_only_at_1024_px(tmp_path: Path) -> None:
    from src.modeling.train_tile_classifier import TrainingConfig

    kwargs = dict(architecture="frozen_encoder", embedding_cache_dir=str(tmp_path / "none"))
    assert (TrainingConfig(tile_size=512, quadrant_pooling="max", **kwargs).run_id()
            == TrainingConfig(tile_size=512, **kwargs).run_id())
    assert (TrainingConfig(tile_size=1024, quadrant_pooling="max", **kwargs).run_id()
            != TrainingConfig(tile_size=1024, **kwargs).run_id())


def test_a_head_trains_and_evaluates_from_the_cache(
    tmp_path: Path, stub_encoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.modeling.evaluate_tile_classifier as evaluate_module
    import src.modeling.train_tile_classifier as train_module
    from src.modeling.evaluate_tile_classifier import evaluate
    from src.modeling.train_tile_classifier import TrainingConfig, train

    tiles = _tiles(tmp_path, 512, n=12)
    path = build_cache(ENCODER, 512, tmp_path / "cache", batch_size=4, num_workers=0,
                       tiles=tiles, device="cpu")
    labels = ["positive" if i % 2 else "negative" for i in range(12)]
    splits = ["train"] * 6 + ["val"] * 3 + ["test"] * 3
    index = tiles.assign(label=labels, split=splits, cut_name="S_A_1_1-2_cut000",
                         oocyte_area_fraction=[0.5 if lab == "positive" else 0.0 for lab in labels])
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"manifest_version": "test", "slides": {}}))

    monkeypatch.setattr(train_module, "load_tile_index", lambda **kw: index)
    monkeypatch.setattr(evaluate_module, "load_tile_index", lambda **kw: index)
    config = TrainingConfig(
        architecture="frozen_encoder", tile_size=512, embedding_cache_dir=str(path),
        epochs=2, patience=5, batch_size=4, num_workers=0, split_manifest_path=str(manifest),
        output_dir=str(tmp_path / "runs"),
    )
    result = train(config)
    results = evaluate(tmp_path / "runs" / result.run_id, bootstrap_samples=5, num_workers=0)

    assert results["augmentation"] == "none (cached embeddings)"
    assert results["embedding_cache"]["meta_sha256"] == config.embedding_cache_sha256
    assert results["metrics"]["test"]["n_tiles"] == 3


def test_a_rewritten_tile_manifest_invalidates_the_cache(
    tmp_path: Path, stub_encoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    manifest = tmp_path / "data/tiles/S_A_1_1-2/S_A_1_1-2_cut000/S_A_1_1-2_cut000_tile_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    monkeypatch.setattr(embedding_cache, "_resolve_repo_path",
                        lambda p: Path(p) if Path(p).is_absolute() else tmp_path / p)
    tiles = _tiles(tmp_path, 512).assign(cut_name="S_A_1_1-2_cut000")
    path = build_cache(ENCODER, 512, tmp_path / "cache", batch_size=4, num_workers=0,
                       tiles=tiles, device="cpu")
    load_cache(path, 512, ENCODER)

    manifest.write_text('{"regenerated": true}')
    os.utime(manifest, ns=(1, 1))
    with pytest.raises(ValueError, match="changed since the cache was built"):
        load_cache(path, 512, ENCODER)
