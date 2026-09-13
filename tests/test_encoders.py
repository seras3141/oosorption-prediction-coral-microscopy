"""Tests for the frozen-encoder arm, using a stub backbone.

Deliberately does not load real weights: Phikon-v2 is a 1.2 GB download and 300M
parameters, which would make the suite unusable. What matters here is the wiring -- the
2x2 split geometry, the pooling, and that the backbone really is frozen -- all of which
a stub exercises exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.encoders import (
    ENCODER_LICENCES,
    PHIKON_V2,
    SUPPORTED_ENCODERS,
    FrozenEncoderClassifier,
    _pooled_embedding,
    _split_into_quadrants,
    encoder_input_px,
    load_encoder,
)

EMBED_DIM = 16


class _StubEncoder(nn.Module):
    """Returns a per-image embedding derived from the mean pixel, plus a live parameter.

    The mean makes pooling checkable by hand; the parameter lets the freezing assertions
    mean something.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(EMBED_DIM))
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        # (batch, EMBED_DIM), the shape a pooled backbone output actually has.
        per_image = images.mean(dim=(1, 2, 3)).unsqueeze(1)
        return per_image * self.scale


def _classifier(tile_size: int) -> FrozenEncoderClassifier:
    return FrozenEncoderClassifier(_StubEncoder(), EMBED_DIM, tile_size=tile_size)


def test_512_arm_emits_one_logit_per_tile() -> None:
    model = _classifier(512)
    logits = model(torch.randn(3, 3, 224, 224))

    assert logits.shape == (3,)
    assert not model.pools_sub_tiles


def test_1024_arm_splits_into_four_and_pools() -> None:
    model = _classifier(1024)
    assert model.pools_sub_tiles

    logits = model(torch.randn(2, 3, 448, 448))

    assert logits.shape == (2,)
    # One backbone pass over batch*4 sub-tiles, not four separate passes.
    assert model.encoder.calls == 1


def test_1024_arm_pools_the_mean_of_its_quadrants() -> None:
    """Not a running average or a concat: the four sub-tile embeddings are averaged."""
    model = _classifier(1024)
    images = torch.randn(2, 3, 448, 448)

    quadrants = _split_into_quadrants(images)
    with torch.no_grad():
        manual = torch.stack(
            [_pooled_embedding(model.encoder, quadrants[:, i]) for i in range(4)], dim=1
        ).mean(dim=1)
        internal = model._embed(images)

    assert torch.allclose(manual, internal, atol=1e-6)


def test_quadrants_are_equal_and_cover_the_whole_tile() -> None:
    images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

    quadrants = _split_into_quadrants(images)

    assert quadrants.shape == (2, 4, 3, 2, 2)
    # Every pixel appears exactly once across the four quadrants.
    assert quadrants.sum() == images.sum()


def test_odd_dimensions_are_refused() -> None:
    """An uneven split would silently drop a row or column of pixels."""
    with pytest.raises(ValueError, match="even spatial dimensions"):
        _split_into_quadrants(torch.randn(1, 3, 225, 224))


def test_backbone_is_frozen_and_only_the_head_trains() -> None:
    model = _classifier(512)

    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for p in model.head.parameters())


def test_train_mode_leaves_the_backbone_in_eval() -> None:
    """model.train() would otherwise re-enable backbone dropout and batch-norm updates,
    so a "frozen" encoder would still drift between training and evaluation."""
    model = _classifier(512)

    model.train()

    assert model.head.training is True
    assert model.encoder.training is False


def test_no_gradient_reaches_the_backbone() -> None:
    model = _classifier(512)
    logits = model(torch.randn(2, 3, 224, 224))

    logits.sum().backward()

    assert model.encoder.scale.grad is None
    assert any(p.grad is not None for p in model.head.parameters())


@pytest.mark.parametrize("tile_size", [128, 256, 2048, 0])
def test_unsupported_tile_size_is_refused(tile_size: int) -> None:
    with pytest.raises(ValueError, match="tile_size must be"):
        _classifier(tile_size)

    with pytest.raises(ValueError, match="tile_size must be"):
        encoder_input_px(tile_size)


def test_input_px_keeps_sub_tiles_at_the_backbone_magnification() -> None:
    assert encoder_input_px(512) == 224
    assert encoder_input_px(1024) == 448


def test_unknown_encoder_is_refused_before_any_download() -> None:
    """An unknown backbone's output shape cannot be pooled reliably."""
    with pytest.raises(ValueError, match="encoder_name must be one of"):
        load_encoder("some/other-model")


def test_every_supported_encoder_has_its_licence_recorded() -> None:
    """Phikon-v2 is non-commercial; that must not be discovered after productisation."""
    assert set(ENCODER_LICENCES) == set(SUPPORTED_ENCODERS)
    assert "non-commercial" in ENCODER_LICENCES[PHIKON_V2]


def test_pooled_embedding_handles_the_shapes_these_backbones_return() -> None:
    class _Tokens(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.arange(
                images.shape[0] * 5 * EMBED_DIM, dtype=torch.float32
            ).reshape(images.shape[0], 5, EMBED_DIM)

    images = torch.randn(2, 3, 8, 8)
    # A 3-D tensor is token states: take CLS.
    pooled = _pooled_embedding(_Tokens(), images)
    assert pooled.shape == (2, EMBED_DIM)
    assert torch.equal(pooled, _Tokens()(images)[:, 0])


def test_pooled_embedding_rejects_an_unpoolable_output() -> None:
    class _Odd(nn.Module):
        def forward(self, images: torch.Tensor) -> dict:
            return {"unexpected": images}

    with pytest.raises(TypeError, match="cannot pool an embedding"):
        _pooled_embedding(_Odd(), torch.randn(1, 3, 8, 8))


def test_unverified_encoder_is_declared_as_such() -> None:
    """H-optimus-0 is gated: its card and config both return 401 without a token, so
    its weights have never been fetched and its path has never run here. Recording that
    keeps an unvalidated result from being read as a validated one."""
    from src.modeling.encoders import ENCODER_SPECS, H_OPTIMUS_0

    assert ENCODER_SPECS[PHIKON_V2].verified is True
    assert ENCODER_SPECS[H_OPTIMUS_0].verified is False


def test_each_encoder_carries_its_own_normalisation() -> None:
    from src.modeling.encoders import ENCODER_SPECS, H_OPTIMUS_0, IMAGENET_MEAN

    assert ENCODER_SPECS[PHIKON_V2].mean == IMAGENET_MEAN
    assert ENCODER_SPECS[H_OPTIMUS_0].mean != IMAGENET_MEAN
    for spec in ENCODER_SPECS.values():
        assert len(spec.mean) == 3 and len(spec.std) == 3
        assert all(s > 0 for s in spec.std)


def test_h_optimus_carries_the_model_kwargs_its_card_documents() -> None:
    from src.modeling.encoders import ENCODER_SPECS, H_OPTIMUS_0

    kwargs = ENCODER_SPECS[H_OPTIMUS_0].model_kwargs
    assert kwargs["init_values"] == pytest.approx(1e-5)
    assert kwargs["dynamic_img_size"] is False


def test_encoder_spec_refuses_an_unknown_backbone() -> None:
    from src.modeling.encoders import encoder_spec

    with pytest.raises(ValueError, match="no known normalisation statistics"):
        encoder_spec("some/other-model")


def test_bad_tile_size_is_refused_before_any_weight_fetch(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise a typo pays a 1.2 GB load first and only then raises."""
    import src.modeling.encoders as module

    def _explode(*args, **kwargs):
        raise AssertionError("weights must not be fetched for a bad tile size")

    monkeypatch.setattr(module, "FrozenEncoderClassifier", _explode)
    with pytest.raises(ValueError, match="tile_size must be"):
        load_encoder(PHIKON_V2, tile_size=256)


def test_cls_token_is_preferred_over_the_pooler() -> None:
    """A ViT-based encoder pools CLS through a Linear+Tanh absent from the released
    checkpoint, so HF initialises it randomly; the head would then train on a random
    projection with no error and no warning."""

    class _Output:
        def __init__(self, cls: torch.Tensor, pooled: torch.Tensor) -> None:
            self.last_hidden_state = cls
            self.pooler_output = pooled

    class _Model(nn.Module):
        def forward(self, images: torch.Tensor):
            batch = images.shape[0]
            hidden = torch.zeros(batch, 3, EMBED_DIM)
            hidden[:, 0] = 1.0  # the real CLS token
            return _Output(hidden, torch.full((batch, EMBED_DIM), 99.0))

    pooled = _pooled_embedding(_Model(), torch.randn(2, 3, 8, 8))

    assert torch.allclose(pooled, torch.ones(2, EMBED_DIM))
    assert not torch.allclose(pooled, torch.full((2, EMBED_DIM), 99.0))


def test_pooler_is_still_used_when_there_is_no_hidden_state() -> None:
    class _PoolerOnly:
        pooler_output = torch.ones(2, EMBED_DIM)

    class _Model(nn.Module):
        def forward(self, images: torch.Tensor):
            return _PoolerOnly()

    assert torch.allclose(_pooled_embedding(_Model(), torch.randn(2, 3, 8, 8)),
                          torch.ones(2, EMBED_DIM))


def test_forcing_offline_mode_actually_takes_effect() -> None:
    """Writing HF_HUB_OFFLINE to the environment is not enough: huggingface_hub reads it
    once at import into a module constant, and importing timm already imported the hub,
    so a later env write lands too late and the fetch proceeds anyway."""
    from huggingface_hub import constants

    from src.modeling.encoders import _force_offline_hub

    original = constants.HF_HUB_OFFLINE
    try:
        constants.HF_HUB_OFFLINE = False
        _force_offline_hub()
        assert constants.HF_HUB_OFFLINE is True
        assert constants.is_offline_mode() is True
    finally:
        constants.HF_HUB_OFFLINE = original


def test_head_geometry_is_exposed_for_the_checkpoint_to_record() -> None:
    """A head-only checkpoint stores exactly these layers, so their shape has to be
    recoverable rather than left to a default that could change."""
    model = FrozenEncoderClassifier(_StubEncoder(), EMBED_DIM, tile_size=512,
                                    hidden_dim=64, dropout=0.1)

    assert model.hidden_dim == 64
    assert model.dropout == pytest.approx(0.1)
    assert model.head[0].out_features == 64
