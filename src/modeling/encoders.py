"""Frozen pathology-encoder arm: a histology backbone with a light trained head.

The comparison arm to the fine-tuned ResNet-18. The backbone stays frozen and only a
two-layer MLP head is trained, because the supporting evidence for these encoders covers
low-resource *probing* of in-domain human classification tasks, not full fine-tuning --
and coral gonad is a large domain shift from the human H&E these models were trained on.

Two input details are specified rather than left to the shipped preprocessor, because
both would silently corrupt the labels:

**No centre crop.** Phikon-v2's ``preprocessor_config.json`` sets ``do_resize`` *and*
``do_center_crop`` with a 224 crop. Applied as shipped to a 512 px tile that is resized
to 256 and then cropped to 224, roughly 58% of the tile area is discarded -- so an oocyte
near the tile border is cropped away while the label still says positive. The whole tile
is resized instead.

**A 2x2 grid at 1024 px.** At 0.2293 um/px, downsampling a 512 px tile to 224 lands at
about 0.52 um/px, essentially the 20x regime Phikon-v2 was trained on. Doing the same to
a 1024 px tile lands at about 1.05 um/px -- roughly 10x, outside the encoder's training
domain. So a 1024 px tile is split into four 512 px sub-tiles, each embedded at its
native magnification, and the four embeddings are mean-pooled.

**256 px is accepted off-magnification; 128 px is refused.** A 256 px tile resized to
224 lands at about 0.26 um/px, roughly twice the magnification the encoder was trained
at. That mismatch is accepted deliberately, so the smallest scale the ResNet arm is
compared at also has a frozen-encoder point, and it has to be read as "encoder off its
training distribution" as much as "less context". At 128 px it would be about 4x, far
enough outside that a result would say more about the encoder than the scale.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

LOG = logging.getLogger(__name__)

PHIKON_V2 = "owkin/phikon-v2"
H_OPTIMUS_0 = "bioptimus/H-optimus-0"
DEFAULT_ENCODER = PHIKON_V2

ENCODER_INPUT_PX = 224
SUB_TILE_PX = 512
#: Resized straight to the encoder input, at ~2x its training magnification -- see the
#: module docstring.
SMALL_TILE_PX = 256
FROZEN_TILE_SIZES = (SMALL_TILE_PX, SUB_TILE_PX, 2 * SUB_TILE_PX)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EncoderSpec:
    """How to load one backbone and what to feed it.

    The normalisation statistics belong here rather than as one global constant. Feeding
    a backbone inputs normalised for a different dataset shifts every pixel outside the
    range it was trained on, degrading every embedding with no error raised -- the same
    class of silent domain shift the 2x2 split at 1024 px exists to avoid.

    Attributes
    ----------
    loader : str
        ``"transformers"`` or ``"timm"``.
    mean, std : tuple of float
        Channel normalisation statistics.
    licence : str
        Recorded with the run so a later productisation decision is not made unaware of
        it -- Phikon-v2 is non-commercial.
    verified : bool
        Whether this project has actually loaded these weights and run a forward pass.
        False means the path is written but unexercised, and the statistics are taken
        from published documentation rather than confirmed locally.
    model_kwargs : dict
        Extra arguments the backbone's documentation requires.
    """

    loader: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    licence: str
    verified: bool
    model_kwargs: dict[str, Any] = field(default_factory=dict)


#: Encoders this arm knows how to load. Phikon-v2 is the primary choice: open weights,
#: ungated, no data-use agreement, and verified here. H-optimus-0 is the documented
#: fallback, but it is gated -- its model card and config both return 401 without a
#: token -- so its weights have never been fetched and its path has never run in this
#: project. Its statistics below come from its published card, not from a local check.
ENCODER_SPECS: dict[str, EncoderSpec] = {
    PHIKON_V2: EncoderSpec(
        loader="transformers",
        # Confirmed against the cached preprocessor_config.json: Phikon-v2 uses the
        # ImageNet statistics, which is why a single global constant happened to work.
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        licence="Owkin non-commercial research licence",
        verified=True,
    ),
    H_OPTIMUS_0: EncoderSpec(
        loader="timm",
        # From the published model card. UNVERIFIED here: the card is gated and returns
        # 401 without a token, so these could not be confirmed against the real config.
        mean=(0.707223, 0.578729, 0.703617),
        std=(0.211883, 0.230117, 0.177517),
        licence="Apache-2.0 (gated: requires terms acceptance and an HF token)",
        verified=False,
        model_kwargs={"init_values": 1e-5, "dynamic_img_size": False},
    ),
}

SUPPORTED_ENCODERS = tuple(ENCODER_SPECS)

#: Kept for callers that only want the licence string.
ENCODER_LICENCES = {name: spec.licence for name, spec in ENCODER_SPECS.items()}


@contextlib.contextmanager
def _offline_hub() -> Iterator[None]:
    """Make ``huggingface_hub`` refuse network access, for this block only.

    ``timm.create_model`` has no ``local_files_only``, so the only lever is the hub's
    offline mode. Setting ``HF_HUB_OFFLINE`` in the environment is not enough:
    ``huggingface_hub`` reads it once at import into a module constant, and importing
    timm already imported the hub, so the write lands too late and the fetch proceeds
    anyway -- the mid-run network hang the cache-only promise exists to prevent. The
    constant has to be assigned directly.

    Scoped rather than permanent: leaving it set would mean a later
    ``load_encoder(..., local_files_only=False)`` silently could not fetch, which
    contradicts that opt-out and breaks any process loading more than one encoder.
    """
    previous_env = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    constants = None
    previous_constant = None
    try:
        from huggingface_hub import constants as hub_constants

        constants = hub_constants
        previous_constant = constants.HF_HUB_OFFLINE
        constants.HF_HUB_OFFLINE = True
    except Exception:  # pragma: no cover - hub layout changed
        LOG.warning(
            "Could not force huggingface_hub offline mode; a missing weight may reach "
            "the network instead of failing at startup"
        )
    try:
        yield
    finally:
        if constants is not None:
            constants.HF_HUB_OFFLINE = previous_constant
        if previous_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_env


def encoder_spec(encoder_name: str) -> EncoderSpec:
    """Return the loading and preprocessing spec for a supported encoder.

    Raises
    ------
    ValueError
        If the encoder is unknown. An unrecognised backbone has no known normalisation
        statistics, and guessing them would silently degrade every embedding.
    """
    try:
        return ENCODER_SPECS[encoder_name]
    except KeyError:
        raise ValueError(
            f"encoder_name must be one of {SUPPORTED_ENCODERS}, got {encoder_name!r}; "
            "an unknown backbone has no known normalisation statistics or output shape"
        ) from None


#: Defaults for the trained head. Recorded with each run rather than left implicit:
#: a head checkpoint stores only these layers, so changing the width later would make
#: every existing checkpoint fail to load, and changing the dropout would reload it with
#: different regularisation and no error at all.
DEFAULT_HEAD_HIDDEN_DIM = 512
DEFAULT_HEAD_DROPOUT = 0.25


class FrozenEncoderClassifier(nn.Module):
    """A frozen histology backbone with a trained two-layer MLP head.

    Parameters
    ----------
    encoder : nn.Module
        The backbone. Put in eval mode with gradients disabled by this class.
    embedding_dim : int
        Width of the backbone's pooled output.
    tile_size : int
        512 embeds the tile once; 1024 embeds four 512 px sub-tiles and mean-pools them,
        so both scales reach the backbone at the magnification it was trained on. 256
        embeds the tile once at about twice that magnification.
    hidden_dim : int, optional
        Width of the head's hidden layer.
    dropout : float, optional
        Applied between the head's two layers.

    Raises
    ------
    ValueError
        If ``tile_size`` is not one of :data:`FROZEN_TILE_SIZES`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        tile_size: int,
        hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
        dropout: float = DEFAULT_HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        encoder_input_px(tile_size)
        self.encoder = encoder
        self.tile_size = tile_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def pools_sub_tiles(self) -> bool:
        """Whether the forward pass splits each tile into a 2x2 grid first."""
        return self.tile_size == 2 * SUB_TILE_PX

    def train(self, mode: bool = True) -> "FrozenEncoderClassifier":
        """Put the head in the requested mode but keep the backbone in eval.

        Without this, ``model.train()`` would re-enable the backbone's dropout and
        batch-norm updates, so a "frozen" encoder would still drift and its embeddings
        would differ between training and evaluation.
        """
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one logit per tile.

        Parameters
        ----------
        images : torch.Tensor
            ``(batch, 3, H, W)``. At 1024 px, ``H`` and ``W`` are the sub-tile grid's
            input size and the batch is split into quadrants internally.
        """
        with torch.no_grad():
            embeddings = self._embed(images)
        return self.head(embeddings).squeeze(-1)

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        if not self.pools_sub_tiles:
            return _pooled_embedding(self.encoder, images)
        quadrants = _split_into_quadrants(images)
        # One backbone pass over batch*4 sub-tiles, then mean-pool per tile. Passing the
        # quadrants as one batch rather than looping keeps the GPU busy and makes the
        # pooling exact rather than a running average.
        batch, _, _, _ = images.shape
        flat = quadrants.reshape(batch * 4, *quadrants.shape[2:])
        embedded = _pooled_embedding(self.encoder, flat)
        return embedded.reshape(batch, 4, -1).mean(dim=1)


def _split_into_quadrants(images: torch.Tensor) -> torch.Tensor:
    """Split each image into a 2x2 grid, returning ``(batch, 4, 3, H/2, W/2)``.

    Raises
    ------
    ValueError
        If the spatial dimensions are not even, which would make the quadrants unequal
        and silently drop a row or column of pixels.
    """
    _, _, height, width = images.shape
    if height % 2 or width % 2:
        raise ValueError(
            f"a 2x2 split needs even spatial dimensions, got {height}x{width}"
        )
    half_h, half_w = height // 2, width // 2
    return torch.stack(
        [
            images[:, :, :half_h, :half_w],
            images[:, :, :half_h, half_w:],
            images[:, :, half_h:, :half_w],
            images[:, :, half_h:, half_w:],
        ],
        dim=1,
    )


def _pooled_embedding(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run the backbone and reduce its output to one vector per image.

    Handles both shapes these encoders return: a HuggingFace output object carrying
    ``pooler_output`` or token states needing a CLS take, and a timm model returning a
    plain tensor.
    """
    output = encoder(images)
    if isinstance(output, torch.Tensor):
        return output if output.ndim == 2 else output[:, 0]

    # CLS token first, pooler second. Both supported cards document
    # last_hidden_state[:, 0]. For Dinov2 (Phikon-v2) the two are identical -- checked,
    # max absolute difference 0.0 -- but a ViTModel-based encoder pools CLS through a
    # Linear+Tanh whose weights are absent from the released checkpoint and therefore
    # randomly initialised, so the head would train on a random projection with no error
    # and no warning, and an embedding-dim check could not catch it.
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0]
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        LOG.warning(
            "%s exposes no last_hidden_state; falling back to pooler_output, which on "
            "some architectures is a randomly initialised projection",
            type(output).__name__,
        )
        return pooled
    raise TypeError(
        f"cannot pool an embedding out of {type(output).__name__}; expected a tensor, "
        "last_hidden_state or pooler_output"
    )


def load_encoder(
    encoder_name: str = DEFAULT_ENCODER,
    tile_size: int = SUB_TILE_PX,
    local_files_only: bool = True,
    **head_kwargs: Any,
) -> FrozenEncoderClassifier:
    """Load a supported frozen encoder and wrap it with a trainable head.

    Weights come from the local HuggingFace cache: ``local_files_only`` defaults to True
    so this is enforced rather than merely hoped for. Compute nodes are not assumed to
    have egress, so prefetch on the login node; a missing weight then fails at startup
    instead of hanging on a network call partway through a multi-hour run. Pass
    ``local_files_only=False`` only when deliberately fetching on a machine with egress.

    Raises
    ------
    ValueError
        If ``encoder_name`` is unknown or ``tile_size`` unsupported -- both checked
        before any weights are fetched.
    """
    spec = encoder_spec(encoder_name)
    # Check the tile size before pulling 1.2-4.5 GB of weights: otherwise a bad size
    # pays the whole load first and only then raises.
    encoder_input_px(tile_size)

    LOG.info("Loading %s (licence: %s)", encoder_name, spec.licence)
    if not spec.verified:
        LOG.warning(
            "%s has never been loaded or run in this project, and its normalisation "
            "statistics come from its published card rather than a local check. Treat "
            "any result from it as unvalidated until a forward pass is confirmed.",
            encoder_name,
        )

    if spec.loader == "transformers":
        from transformers import AutoModel

        # local_files_only makes the cache-only claim real. Without it a missing weight
        # reaches the network, which on a compute node without egress is the mid-run
        # hang this is meant to avoid.
        encoder = AutoModel.from_pretrained(
            encoder_name, local_files_only=local_files_only, **spec.model_kwargs
        )
        embedding_dim = int(encoder.config.hidden_size)
    else:
        import timm

        with _offline_hub() if local_files_only else contextlib.nullcontext():
            encoder = timm.create_model(
                f"hf-hub:{encoder_name}",
                pretrained=True,
                num_classes=0,
                **spec.model_kwargs,
            )
        embedding_dim = int(encoder.num_features)

    LOG.info("%s embedding dim %d", encoder_name, embedding_dim)
    return FrozenEncoderClassifier(
        encoder=encoder,
        embedding_dim=embedding_dim,
        tile_size=tile_size,
        **head_kwargs,
    )


def encoder_input_px(tile_size: int) -> int:
    """Input size the transform should produce for this tile size.

    512 px resizes to the backbone's 224 directly. 1024 px resizes to 448 so that each
    quadrant of the 2x2 split arrives at 224, keeping every sub-tile at the ~0.52 um/px
    magnification the backbone was trained on. 256 px also resizes to 224, which lands
    at ~0.26 um/px -- about twice that magnification, accepted deliberately.

    Raises
    ------
    ValueError
        For any other size, including 128 px, which would reach the backbone at ~4x.
    """
    if tile_size in (SMALL_TILE_PX, SUB_TILE_PX):
        return ENCODER_INPUT_PX
    if tile_size == 2 * SUB_TILE_PX:
        return 2 * ENCODER_INPUT_PX
    raise ValueError(
        f"the frozen-encoder arm supports tile sizes {FROZEN_TILE_SIZES}, got {tile_size}"
    )
