"""Train the tile-level oocyte classifier.

Fine-tunes an ImageNet-pretrained ResNet-18 on the geometry-derived tile label, with the
forced-ratio batch sampler and optional hard-negative mining. Validation runs on the
natural class distribution, so the metric that drives early stopping describes deployment
conditions rather than the training-time balance.

Early stopping watches **validation average precision**, not loss. At the positive rates
this corpus has -- roughly 1 in 6.5 at 512 px, 1 in 5 at 1024 px -- a model can improve
its loss while getting worse at the minority class, and average precision is the metric
the results schema treats as primary.

Only one tile size per run: the dataset refuses a mixed index because nothing here
resizes beyond the encoder's input, so 512 px and 1024 px are separate runs whose results
this milestone compares.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
import re
import time
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.modeling.encoders import (
    DEFAULT_ENCODER,
    DEFAULT_HEAD_DROPOUT,
    DEFAULT_HEAD_HIDDEN_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    encoder_input_px,
    encoder_spec,
    load_encoder,
)
from src.modeling.samplers import ForcedRatioBatchSampler, check_mining_allowed
from src.modeling.tile_classification_dataset import TileClassificationDataset
from src.modeling.tile_index import (
    LABEL_RULE_AREA,
    LABEL_RULE_CENTROID,
    binary_targets,
    load_tile_index,
    training_rows,
)
from src.modeling.tile_labels import DEFAULT_MIN_OOCYTE_AREA_FRACTION, LABEL_POSITIVE

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

ARCHITECTURE_RESNET18 = "resnet18"
ARCHITECTURE_FROZEN_ENCODER = "frozen_encoder"
ARCHITECTURES = (ARCHITECTURE_RESNET18, ARCHITECTURE_FROZEN_ENCODER)

# The ResNet arm always sees 224x224. The frozen arm's input size depends on the tile
# size -- see input_px_for -- so at 1024 px the two arms do NOT see the same
# magnification: the ResNet sees the whole tile at ~1.05 um/px while the frozen encoder
# sees four sub-tiles at ~0.52 um/px. That is deliberate, because feeding the frozen
# backbone ~10x input would take it outside its training domain, but it means the
# 1024 px comparison is encoder-and-magnification, not encoder alone.
ENCODER_INPUT_PX = 224

#: Per-arm learning rates. The frozen arm trains a small probe head from scratch and
#: needs an order of magnitude more than fine-tuning a pretrained ResNet; at 1e-4 with
#: early stopping on val AP it would plausibly stop before converging, and lose the
#: comparison for a reason that has nothing to do with the encoder.
DEFAULT_LR = {ARCHITECTURE_RESNET18: 1e-4, ARCHITECTURE_FROZEN_ENCODER: 1e-3}


def _slug(value: str) -> str:
    """Reduce a value to something safe to use as one path segment."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unnamed"


@dataclass
class TrainingConfig:
    """Everything that affects a run's numbers, recorded alongside its results."""

    architecture: str = ARCHITECTURE_RESNET18
    encoder_name: str | None = None
    tile_size: int = 512
    label_rule: str = LABEL_RULE_AREA
    min_oocyte_area_fraction: float = DEFAULT_MIN_OOCYTE_AREA_FRACTION
    epochs: int = 30
    patience: int = 5
    batch_size: int = 64
    #: None means "the default for this architecture" -- see __post_init__.
    lr: float | None = None
    weight_decay: float = 1e-4
    positive_fraction: float = 0.25
    hard_negative_mining: bool = False
    warmup_epochs: int = 3
    hard_negative_pool_fraction: float = 0.25
    hard_negative_share: float = 0.5
    #: Head geometry, recorded because a head-only checkpoint stores exactly these
    #: layers: a later change to the width would break every existing checkpoint, and a
    #: change to the dropout would reload it with different regularisation, silently.
    head_hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM
    head_dropout: float = DEFAULT_HEAD_DROPOUT
    seed: int = 42
    num_workers: int = 4
    max_train_slides: int | None = None
    max_batches_per_epoch: int | None = None
    max_val_tiles: int | None = None
    output_dir: str = "data/tile_classifier"

    #: Fields that do not change a run's numbers and so stay out of its fingerprint.
    #:
    #: num_workers is NOT here. Augmentation runs inside the DataLoader workers, which
    #: are seeded per worker, so the worker count changes which RNG stream each sample
    #: draws from -- and 0 runs augmentation in the main stream instead. Two runs with
    #: different worker counts follow different training trajectories, so they must not
    #: share a run id and overwrite each other.
    RUN_ID_EXCLUDED: ClassVar[tuple[str, ...]] = ("output_dir",)

    def __post_init__(self) -> None:
        """Resolve defaults here so the recorded config describes what actually ran.

        Two things were being resolved late, at the point of use:

        ``encoder_name`` was left None on the frozen arm and only defaulted inside
        ``build_model``, so the run id, the fingerprint and the saved config all said
        ``null`` while Phikon-v2 weights were trained. The same experiment then got two
        different run directories depending on whether the flag was passed, and a later
        change to the default would leave old checkpoints unable to name their backbone.

        ``lr`` had one default for both arms, though the frozen arm's probe head needs
        an order of magnitude more.
        """
        if self.architecture not in ARCHITECTURES:
            # Validated here rather than only in build_model, so resolving the per-arm
            # learning rate below cannot raise a bare KeyError over a clear message.
            raise ValueError(
                f"architecture must be one of {ARCHITECTURES}, got {self.architecture!r}"
            )
        if self.architecture == ARCHITECTURE_FROZEN_ENCODER and not self.encoder_name:
            self.encoder_name = DEFAULT_ENCODER
        if self.architecture != ARCHITECTURE_FROZEN_ENCODER:
            self.encoder_name = None
        if self.lr is None:
            self.lr = DEFAULT_LR[self.architecture]
        if not self.hard_negative_mining:
            # With mining off these change nothing, so two otherwise identical runs
            # would get different ids and separate output directories -- one experiment
            # presented as several. Normalised for the same reason as encoder_name.
            self.warmup_epochs = 0
            self.hard_negative_pool_fraction = 0.0
            self.hard_negative_share = 0.0
        if self.architecture != ARCHITECTURE_FROZEN_ENCODER:
            # Head geometry only exists on the frozen arm.
            self.head_hidden_dim = DEFAULT_HEAD_HIDDEN_DIM
            self.head_dropout = DEFAULT_HEAD_DROPOUT
        if self.label_rule == LABEL_RULE_CENTROID:
            # The centroid rule ignores the coverage threshold entirely, so two runs
            # differing only in it train on identical labels and must not be filed as
            # separate experiments.
            self.min_oocyte_area_fraction = DEFAULT_MIN_OOCYTE_AREA_FRACTION

    def run_id(self) -> str:
        """Identify a run by everything that changes its numbers.

        A readable prefix names the axes this milestone compares, followed by a short
        fingerprint over the whole configuration. The fingerprint is what actually makes
        the id unique: a readable prefix alone collided across six different experiments
        -- mining on/off and sweeps of positive fraction, batch size and learning rate
        all produced the same name -- so each run silently overwrote the previous one's
        checkpoint and log in a shared directory.

        The threshold is rendered in basis points, so 0.05 and 0.054 do not both become
        "005" and a sub-1% threshold does not become "000".
        """
        # A run id is used as a single directory name, and encoder ids carry a slash
        # ("owkin/phikon-v2"), which would silently nest the run one level deeper and
        # split its artifacts across two directories.
        encoder = _slug(self.encoder_name) if self.encoder_name else "none"
        threshold = f"{round(self.min_oocyte_area_fraction * 10000):04d}"
        return (
            f"{self.architecture}_{encoder}_{self.tile_size:04d}_"
            f"{self.label_rule}{threshold}_seed{self.seed}_{self.fingerprint()}"
        )

    def fingerprint(self, length: int = 8) -> str:
        """Short stable hash over every field that affects the numbers."""
        payload = {
            key: value
            for key, value in sorted(asdict(self).items())
            if key not in self.RUN_ID_EXCLUDED
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        )
        return digest.hexdigest()[:length]


@dataclass
class EpochRecord:
    """One epoch's training and validation summary, for the run log."""

    epoch: int
    train_loss: float
    val_loss: float
    val_average_precision: float
    val_roc_auc: float
    mining_active: bool
    observed_positive_fraction: float
    seconds: float


@dataclass
class TrainingResult:
    """What a completed run leaves behind for evaluation to pick up."""

    run_id: str
    config: dict[str, Any]
    checkpoint_path: str
    epochs_run: int
    best_epoch: int
    best_val_average_precision: float
    early_stopped: bool
    split_sizes: dict[str, dict[str, int]]
    batch_composition_log_path: str
    encoder_licence: str | None = None
    epochs: list[EpochRecord] = field(default_factory=list)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch so a rerun reproduces the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def input_px_for(config: TrainingConfig) -> int:
    """Spatial size the transform should produce for this run.

    The ResNet arm always sees 224. The frozen-encoder arm sees 224 at 512 px, and 448
    at 1024 px so that each quadrant of its 2x2 split arrives at 224 -- keeping every
    sub-tile at the magnification the backbone was trained on instead of downsampling
    the whole tile to roughly 10x.
    """
    if config.architecture == ARCHITECTURE_FROZEN_ENCODER:
        return encoder_input_px(config.tile_size)
    return ENCODER_INPUT_PX


def normalisation_for(config: TrainingConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Channel statistics for this run's backbone.

    Per-encoder rather than one global constant: normalising with another dataset's
    statistics shifts every pixel outside the range the backbone was trained on and
    degrades every embedding with no error raised.
    """
    if config.architecture == ARCHITECTURE_FROZEN_ENCODER:
        spec = encoder_spec(config.encoder_name or DEFAULT_ENCODER)
        return spec.mean, spec.std
    return IMAGENET_MEAN, IMAGENET_STD


def build_transforms(
    train: bool,
    input_px: int = ENCODER_INPUT_PX,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> Any:
    """Return the tile-to-tensor transform for one phase.

    Resizes the whole tile to ``input_px`` rather than centre-cropping it. A crop would
    discard the tile's border, so an oocyte near the edge could be cropped away while the
    label still says positive -- label noise introduced by preprocessing. This is also
    why the shipped Phikon-v2 processor is not used as-is: it centre-crops by default.

    Augmentation is flips, rotation and colour jitter only. Macenko normalisation is a
    separate ablation and is deliberately not entangled with this experiment.
    """
    from torchvision import transforms

    steps: list[Any] = [transforms.Resize((input_px, input_px))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Quarter turns only. RandomRotation(degrees=90) draws a uniform angle in
            # [-90, +90], and any off-axis angle rotates the tile's corners out of frame
            # and black-fills them -- the same preprocessing-introduced label noise this
            # function avoids centre-cropping for, and worse at 512 px where only 11.2%
            # of oocytes fit inside a tile to begin with.
            transforms.RandomChoice(
                [transforms.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)]
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(steps)


def build_model(config: TrainingConfig, pretrained: bool = True) -> nn.Module:
    """Build the classifier for the configured architecture.

    Returns a model whose forward gives one logit per tile.

    Parameters
    ----------
    config : TrainingConfig
        The run's configuration.
    pretrained : bool, optional
        Load the backbone's pretrained weights. Evaluation passes False: a checkpoint
        already carries every parameter, so fetching ImageNet weights only to overwrite
        them costs a download that can fail outright on a node with no cache and no
        egress.
    """
    if config.architecture == ARCHITECTURE_RESNET18:
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if config.architecture == ARCHITECTURE_FROZEN_ENCODER:
        return load_encoder(
            encoder_name=config.encoder_name or DEFAULT_ENCODER,
            tile_size=config.tile_size,
            hidden_dim=config.head_hidden_dim,
            dropout=config.head_dropout,
        )
    raise ValueError(f"architecture must be one of {ARCHITECTURES}")


def prepare_splits(config: TrainingConfig) -> dict[str, pd.DataFrame]:
    """Build the tile index and return the ambiguity-free train and val frames.

    Raises
    ------
    ValueError
        If either split ends up empty, which would otherwise surface as an opaque
        failure inside the sampler or the metric.
    """
    index = training_rows(
        load_tile_index(
            tile_sizes=(config.tile_size,),
            label_rule=config.label_rule,
            min_oocyte_area_fraction=config.min_oocyte_area_fraction,
        )
    )
    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val"):
        frame = index.loc[index["split"] == name].reset_index(drop=True)
        if name == "train" and config.max_train_slides is not None:
            keep = sorted(frame["stem"].unique())[: config.max_train_slides]
            frame = frame.loc[frame["stem"].isin(keep)].reset_index(drop=True)
            LOG.info("Smoke-test subset: %d train slide(s) %s", len(keep), keep)
        if name == "val" and config.max_val_tiles is not None:
            # Class-stratified so a subsampled validation set keeps both classes and the
            # metric stays defined; a head() would often take negatives only.
            frame = _stratified_subsample(frame, config.max_val_tiles, config.seed)
            LOG.info("Smoke-test subset: %d val tile(s)", len(frame))
        if frame.empty:
            raise ValueError(f"the {name} split has no tiles at {config.tile_size} px")
        splits[name] = frame
    return splits


def _stratified_subsample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Take at most ``limit`` rows, preserving the positive rate.

    Smoke tests only. Keeping the natural rate matters even here: a subsample that lost
    the positives would make average precision undefined and hide whether the loop is
    actually learning anything.
    """
    if limit < 2:
        raise ValueError(
            f"limit must be at least 2 to keep one tile of each class, got {limit}; "
            "a single-class validation split makes average precision undefined and "
            "saves no checkpoint"
        )
    if len(frame) <= limit:
        return frame
    positives = frame.loc[frame["label"] == LABEL_POSITIVE]
    negatives = frame.loc[frame["label"] != LABEL_POSITIVE]

    if len(positives) == 0 or len(negatives) == 0:
        # Nothing to preserve; the caller's split is already single-class and
        # _val_metrics will report that.
        return frame.sample(limit, random_state=seed).reset_index(drop=True)

    positive_rate = len(positives) / len(frame)
    # Floor BOTH classes at one. Flooring only positives let a positive-heavy split
    # round the whole limit into positives and return a single-class val set, which
    # makes average precision NaN every epoch, saves no checkpoint, and aborts the run
    # with a message blaming the tile size rather than this subsample.
    n_positive = min(len(positives), max(1, round(limit * positive_rate)))
    n_positive = min(n_positive, limit - 1)
    # Fill the rest with negatives, then top back up from whichever class still has rows,
    # so the result is exactly `limit` long rather than over it (a positive-heavy split
    # could round n_positive up to the whole limit and then add a negative on top) or
    # short of it when one class is scarce.
    n_negative = min(len(negatives), limit - n_positive)
    shortfall = limit - n_positive - n_negative
    if shortfall > 0:
        headroom_positive = min(shortfall, len(positives) - n_positive)
        n_positive += headroom_positive
        n_negative += min(shortfall - headroom_positive, len(negatives) - n_negative)

    taken = pd.concat(
        [
            positives.sample(n_positive, random_state=seed),
            negatives.sample(n_negative, random_state=seed),
        ]
    )
    return taken.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train(config: TrainingConfig) -> TrainingResult:
    """Run one training configuration to completion and write its checkpoint.

    Returns
    -------
    TrainingResult
        Per-epoch history plus the best epoch and checkpoint path.
    """
    check_mining_allowed(config.label_rule, config.hard_negative_mining)
    if config.architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    if config.hard_negative_mining:
        _check_mining_configuration(config)
    seed_everything(config.seed)

    # Build the model before reading the tile index. On the frozen arm this is a
    # multi-GB weight load rather than a cheap construction, but a bad architecture,
    # encoder or tile size should still surface before the index is parsed -- and
    # load_encoder validates all three before fetching anything.
    model = build_model(config)

    run_dir = _resolve_repo_path(config.output_dir) / config.run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    composition_log = run_dir / "batch_composition.jsonl"
    # The sampler appends, and run_dir is deterministic, so a re-run of the same
    # configuration would interleave two runs' batch records in the one file the
    # verification pass reads the realised ratio from.
    checkpoint_path = run_dir / "checkpoint.pt"
    run_log_path = run_dir / "training_log.json"
    # The run directory is deterministic, so a re-run of the same configuration would
    # otherwise leave the previous execution's log and evaluation beside this one's
    # partial composition log -- and anything reading the checkpoint would score a
    # different execution than the logs describe.
    #
    # checkpoint.pt is deliberately NOT cleared. It is overwritten on the first
    # improving epoch, and pre-deleting it would mean a re-run killed early -- a
    # preempted GPU job, or a failure inside prepare_splits -- left the directory with
    # no checkpoint at all, destroying a result that was still scoreable.
    for stale in (
        composition_log,
        run_log_path,
        run_dir / "results.json",
        run_dir / "predictions.csv",
    ):
        stale.unlink(missing_ok=True)

    splits = prepare_splits(config)
    input_px = input_px_for(config)
    mean, std = normalisation_for(config)
    LOG.info("Input %d px, normalisation mean=%s std=%s", input_px, mean, std)
    train_dataset = TileClassificationDataset(
        splits["train"],
        transform=build_transforms(train=True, input_px=input_px, mean=mean, std=std),
    )
    # Same tiles, deterministic transform. Ranking negatives for mining must not depend
    # on which random flip, rotation and jitter each one happened to draw: that puts
    # noise straight into which tiles enter the hard pool, and makes the ranking
    # irreproducible from the checkpoint.
    scoring_dataset = TileClassificationDataset(
        splits["train"],
        transform=build_transforms(train=False, input_px=input_px, mean=mean, std=std),
    )
    val_dataset = TileClassificationDataset(
        splits["val"],
        transform=build_transforms(train=False, input_px=input_px, mean=mean, std=std),
    )

    sampler = ForcedRatioBatchSampler(
        train_dataset.positive_positions(),
        train_dataset.negative_positions(),
        batch_size=config.batch_size,
        positive_fraction=config.positive_fraction,
        batches_per_epoch=config.max_batches_per_epoch,
        seed=config.seed,
        composition_log_path=composition_log,
        label_rule=config.label_rule,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=config.num_workers
    )
    # Validation is not resampled: the natural distribution is what the metric should
    # describe.
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Training %s on %s", config.run_id(), device)
    model = model.to(device)
    # Positives are already oversampled by the sampler, so the loss is unweighted;
    # weighting on top would double-count the correction.
    criterion = nn.BCEWithLogitsLoss()
    # Only parameters that actually require grad: handing the frozen backbone's weights
    # to AdamW would build optimiser state for 300M frozen parameters and let weight
    # decay touch them, so the "frozen" encoder would not be.
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    LOG.info("Training %d of %d parameters", n_trainable, n_total)
    optimizer = torch.optim.AdamW(
        trainable, lr=config.lr, weight_decay=config.weight_decay
    )

    history: list[EpochRecord] = []
    best_ap = -1.0
    best_epoch = 0
    epochs_without_gain = 0
    early_stopped = False

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        if (
            config.hard_negative_mining
            and epoch == config.warmup_epochs + 1
            and not sampler.mining_active
        ):
            _enable_mining(model, scoring_dataset, sampler, config, device)

        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, probabilities, targets = _evaluate(model, val_loader, criterion, device)
        average_precision, roc_auc = _val_metrics(targets, probabilities)

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_average_precision=average_precision,
                val_roc_auc=roc_auc,
                mining_active=sampler.mining_active,
                observed_positive_fraction=sampler.observed_positive_fraction(
                    current_epoch_only=True
                ),
                seconds=time.time() - started,
            )
        )
        LOG.info(
            "epoch %d/%d  train_loss %.4f  val_loss %.4f  val_AP %.4f  val_AUROC %.4f%s",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            average_precision,
            roc_auc,
            "  [mining]" if sampler.mining_active else "",
        )

        # An explicit NaN check, not just `>`: NaN compares false against everything, so
        # a val split with one class would leave best_ap at its sentinel, save no
        # checkpoint, and still report a checkpoint_path that does not exist.
        improved = not math.isnan(average_precision) and average_precision > best_ap
        if improved:
            best_ap, best_epoch, epochs_without_gain = average_precision, epoch, 0
            _save_checkpoint_atomically(
                _checkpoint_payload(model, config, epoch, average_precision),
                checkpoint_path,
            )
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.patience:
                LOG.info(
                    "Early stopping at epoch %d; no val AP gain since epoch %d",
                    epoch,
                    best_epoch,
                )
                early_stopped = True
                break

    if best_epoch == 0:
        raise RuntimeError(
            f"no checkpoint was saved for {config.run_id()}: validation average "
            f"precision never improved over {len(history)} epoch(s). It is NaN when the "
            "validation split holds a single class -- check that the split has both "
            "positives and negatives at this tile size and threshold."
        )

    result = TrainingResult(
        run_id=config.run_id(),
        config=asdict(config),
        checkpoint_path=_path_relative_to_repo(checkpoint_path),
        epochs_run=len(history),
        best_epoch=best_epoch,
        best_val_average_precision=best_ap,
        early_stopped=early_stopped,
        split_sizes={
            name: {
                "n_tiles": len(frame),
                "n_positive": int(binary_targets(frame).sum()),
                "n_slides": int(frame["stem"].nunique()),
            }
            for name, frame in splits.items()
        },
        batch_composition_log_path=_path_relative_to_repo(composition_log),
        encoder_licence=(
            encoder_spec(config.encoder_name or DEFAULT_ENCODER).licence
            if config.architecture == ARCHITECTURE_FROZEN_ENCODER
            else None
        ),
        epochs=history,
    )
    _write_run_log(run_log_path, result)
    return result


def _check_mining_configuration(config: TrainingConfig) -> None:
    """Reject any mining configuration that would quietly do nothing.

    Every case here ends the same way: the run trains identically to the mining-off arm
    while ``hard_negative_mining=True`` sits in the fingerprint and the recorded config,
    so the comparison would show two arms where only one experiment happened. The
    proportion checks are duplicated from the sampler deliberately -- there they are
    reached only on the epoch after the warm-up, which on a real run means dying several
    GPU-hours in.

    Raises
    ------
    ValueError
        If the warm-up leaves no epoch for mining, if early stopping could fire before
        mining starts, if the head would be ranked before it has learned anything, or if
        either proportion is out of range or leaves the hard pool unused.
    """
    if config.warmup_epochs < 1:
        raise ValueError(
            "warmup_epochs must be at least 1: ranking negatives with a randomly "
            "initialised head would make the hard pool meaningless"
        )
    if config.warmup_epochs >= config.epochs:
        raise ValueError(
            f"warmup_epochs {config.warmup_epochs} leaves no epoch for mining in a "
            f"{config.epochs}-epoch run, so mining would never activate while the run "
            "still recorded itself as the mining arm"
        )
    if config.patience <= config.warmup_epochs:
        raise ValueError(
            f"patience {config.patience} can stop the run at or before epoch "
            f"{config.warmup_epochs + 1}, when mining would first activate; raise "
            "patience above warmup_epochs or shorten the warm-up"
        )
    if not 0.0 < config.hard_negative_pool_fraction <= 1.0:
        raise ValueError(
            "hard_negative_pool_fraction must be in (0, 1], got "
            f"{config.hard_negative_pool_fraction}"
        )
    if not 0.0 < config.hard_negative_share <= 1.0:
        raise ValueError(
            f"hard_negative_share must be in (0, 1], got {config.hard_negative_share}; "
            "a share of 0 would score the whole negative pool and then never draw from "
            "it, leaving the run indistinguishable from mining-off"
        )
    # The realised quota, not just the proportion: a small share against a small batch
    # rounds to zero hard negatives, and the run then reports mining active while
    # drawing none -- the mining-off arm wearing the mining arm's label.
    negatives_per_batch = config.batch_size - round(
        config.batch_size * config.positive_fraction
    )
    if round(negatives_per_batch * config.hard_negative_share) < 1:
        raise ValueError(
            f"hard_negative_share {config.hard_negative_share} against "
            f"{negatives_per_batch} negative(s) per batch rounds to zero hard "
            "negatives, so mining would draw none while the run recorded itself as the "
            "mining arm; raise the share or the batch size"
        )


def _save_checkpoint_atomically(payload: dict[str, Any], path: Path) -> None:
    """Write the checkpoint to a sibling temp file, then replace in one step.

    Saving in place truncates the previous checkpoint before the replacement is
    complete, so a preemption mid-write leaves a corrupt file -- and this run directory
    deliberately keeps the previous checkpoint precisely so a killed re-run does not
    destroy a still-scoreable result. An in-place write would defeat that. os.replace is
    atomic within a filesystem, and the temp file is a sibling to stay on one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _checkpoint_payload(
    model: nn.Module,
    config: TrainingConfig,
    epoch: int,
    average_precision: float,
) -> dict[str, Any]:
    """Build the checkpoint, saving only what training actually changed.

    On the frozen arm the backbone is 300M-1.1B unchanged parameters -- 1.2 GB for
    Phikon-v2 ViT-L, more for H-optimus-0 -- and saving it on every improving epoch
    means repeated multi-GB writes to the group lustre share for bytes that are already
    in the HuggingFace cache. Only the head is stored; ``encoder_name`` in the config
    says which backbone to reload it onto.
    """
    head_only = config.architecture == ARCHITECTURE_FROZEN_ENCODER
    state = model.head.state_dict() if head_only else model.state_dict()
    return {
        "model_state_dict": state,
        "state_dict_scope": "head" if head_only else "model",
        "config": asdict(config),
        "epoch": epoch,
        "val_average_precision": average_precision,
    }


def _enable_mining(
    model: nn.Module,
    dataset: TileClassificationDataset,
    sampler: ForcedRatioBatchSampler,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    """Score the whole negative pool and hand the hardest to the sampler.

    ``dataset`` must carry the deterministic transform, not the training augmentation.
    """
    positions = dataset.negative_positions()
    LOG.info("Scoring %d negatives for hard-negative mining", len(positions))
    scores = _score_positions(model, dataset, positions, config, device)
    sampler.set_hard_negatives(
        scores,
        pool_fraction=config.hard_negative_pool_fraction,
        share=config.hard_negative_share,
    )


def _score_positions(
    model: nn.Module,
    dataset: TileClassificationDataset,
    positions: list[int],
    config: TrainingConfig,
    device: torch.device,
) -> list[float]:
    """Predicted positive probability for the given dataset positions, in order."""
    from torch.utils.data import Subset

    loader = DataLoader(
        Subset(dataset, positions),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for images, _ in loader:
            logits = _logits(model, images.to(device))
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return scores


def _logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """One logit per tile, whatever shape the arm's head returns.

    The ResNet head gives ``(batch, 1)``; the frozen-encoder head already squeezes to
    ``(batch,)``. Squeezing dim 1 unconditionally would fail on the latter.
    """
    output = model(images)
    return output.squeeze(-1) if output.ndim > 1 else output


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(_logits(model, images), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    probabilities: list[float] = []
    targets_seen: list[float] = []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = _logits(model, images)
            # Weight by samples, not batches: validation is not resampled and its last
            # batch is partial, so a per-batch mean makes val_loss depend on batch size
            # and stop being comparable across a sweep.
            total_loss += float(criterion(logits, targets).item()) * targets.numel()
            n_samples += int(targets.numel())
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            targets_seen.extend(targets.cpu().tolist())
    return (
        total_loss / max(1, n_samples),
        np.asarray(probabilities, dtype=float),
        np.asarray(targets_seen, dtype=float),
    )


def _val_metrics(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Average precision and ROC AUC, or NaN where a split has only one class.

    A smoke-test subset can easily hold a single class; returning NaN keeps the loop
    running and makes the gap visible in the log rather than raising mid-run.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(targets)) < 2:
        LOG.warning("Validation targets hold one class only; metrics are undefined")
        return float("nan"), float("nan")
    return (
        float(average_precision_score(targets, probabilities)),
        float(roc_auc_score(targets, probabilities)),
    )


def _write_run_log(path: Path, result: TrainingResult) -> None:
    """Write the run log, rendering an undefined metric as null rather than NaN.

    ``json.dump`` emits a bare ``NaN`` literal by default. Python reads it back, but it
    is not valid JSON, and the results schema uses ``null`` for a metric that could not
    be computed -- so any strict consumer would reject the file.
    """
    payload = _nan_to_none(asdict(result))
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, allow_nan=False)
        fp.write("\n")
    LOG.info("Wrote %s", path)


def _nan_to_none(value: Any) -> Any:
    """Recursively replace NaN floats with None so the payload is valid JSON."""
    if isinstance(value, dict):
        return {key: _nan_to_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _path_relative_to_repo(path: Path) -> str:
    """Return a repository-relative POSIX path, so a results file survives a move.

    Deliberately does **not** call ``resolve()``: ``data/`` is a symlink to the group
    lustre share, so resolving follows it out of the repository and every artifact path
    would be recorded absolute and machine-specific. Matches
    ``generate_tiles._path_relative_to_repo``, which records tile paths the same way.
    """
    candidate = path if path.is_absolute() else REPO_ROOT / path
    try:
        return candidate.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        LOG.warning("%s is outside %s; recording it absolute", candidate, REPO_ROOT)
        return candidate.as_posix()
