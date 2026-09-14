"""Tests for the training loop's pure parts: run identity, subsampling, guards."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.train_tile_classifier import (
    TrainingConfig,
    _stratified_subsample,
    build_model,
    build_transforms,
)


def _frame(n_positive: int, n_negative: int) -> pd.DataFrame:
    labels = ["positive"] * n_positive + ["negative"] * n_negative
    return pd.DataFrame(
        {
            "tile_id": [f"t{i}" for i in range(len(labels))],
            "label": labels,
            "is_ambiguous": [False] * len(labels),
        }
    )


def test_run_id_starts_with_the_axes_this_milestone_compares() -> None:
    assert TrainingConfig().run_id().startswith("resnet18_none_0512_area0500_seed42_")


@pytest.mark.parametrize(
    "changed",
    [
        {"tile_size": 1024},
        {"min_oocyte_area_fraction": 0.25},
        {"seed": 7},
        {"label_rule": "centroid"},
        {"hard_negative_mining": True},
        {"positive_fraction": 0.5},
        {"batch_size": 32},
        {"lr": 1e-3},
        {"weight_decay": 1e-3},
        {"epochs": 10},
        {"patience": 2},
        {"max_train_slides": 2},
        {"max_batches_per_epoch": 5},
        {"max_val_tiles": 100},
    ],
)
def test_every_setting_that_changes_the_numbers_changes_the_run_id(changed: dict) -> None:
    """A readable prefix alone collapsed six different experiments onto one id, so each
    silently overwrote the previous run's checkpoint and log."""
    assert TrainingConfig(**changed).run_id() != TrainingConfig().run_id()


@pytest.mark.parametrize("field", ["num_workers", "output_dir"])
def test_settings_that_do_not_change_the_numbers_keep_the_run_id(field: str) -> None:
    value = 1 if field == "num_workers" else "somewhere/else"
    assert TrainingConfig(**{field: value}).run_id() == TrainingConfig().run_id()


@pytest.mark.parametrize(
    "changed",
    [
        {"warmup_epochs": 7},
        {"hard_negative_pool_fraction": 0.5},
        {"hard_negative_share": 0.9},
    ],
)
def test_mining_knobs_only_affect_the_run_id_when_mining_is_on(changed: dict) -> None:
    """With mining off they change nothing, so two identical experiments would otherwise
    get separate run directories and appear as two."""
    assert TrainingConfig(**changed).run_id() == TrainingConfig().run_id()

    on = TrainingConfig(hard_negative_mining=True)
    assert TrainingConfig(hard_negative_mining=True, **changed).run_id() != on.run_id()
    assert on.run_id() != TrainingConfig().run_id()


@pytest.mark.parametrize(
    ("changed", "affects_run_id"),
    [({"head_hidden_dim": 256}, True), ({"head_dropout": 0.5}, True)],
)
def test_head_geometry_is_fingerprinted_on_the_frozen_arm_only(
    changed: dict, affects_run_id: bool
) -> None:
    """A head-only checkpoint stores exactly these layers, so a later change to the
    width would break every existing one and a change to the dropout would reload it
    with different regularisation and no error."""
    frozen = TrainingConfig(architecture="frozen_encoder")
    assert (
        TrainingConfig(architecture="frozen_encoder", **changed).run_id() != frozen.run_id()
    ) is affects_run_id

    resnet = TrainingConfig(architecture="resnet18")
    assert TrainingConfig(architecture="resnet18", **changed).run_id() == resnet.run_id()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"hard_negative_share": 0.0}, r"hard_negative_share must be in \(0, 1\]"),
        ({"hard_negative_share": 1.5}, r"hard_negative_share must be in \(0, 1\]"),
        ({"hard_negative_pool_fraction": 0.0}, r"pool_fraction must be in \(0, 1\]"),
        ({"hard_negative_pool_fraction": 1.5}, r"pool_fraction must be in \(0, 1\]"),
        ({"patience": 2, "warmup_epochs": 3}, "can stop the run at or before epoch"),
        ({"warmup_epochs": 30}, "leaves no epoch for mining"),
        ({"warmup_epochs": 0}, "warmup_epochs must be at least 1"),
    ],
)
def test_mining_configurations_that_would_do_nothing_are_refused(
    kwargs: dict, match: str
) -> None:
    """Each of these trains identically to the mining-off arm while the fingerprint and
    the recorded config still claim to be the mining arm. The sampler's own range checks
    are not reached until the epoch after the warm-up, i.e. GPU-hours into a real run."""
    from src.modeling.train_tile_classifier import _check_mining_configuration

    config = TrainingConfig(hard_negative_mining=True, **kwargs)
    with pytest.raises(ValueError, match=match):
        _check_mining_configuration(config)


def test_default_mining_configuration_is_accepted() -> None:
    from src.modeling.train_tile_classifier import _check_mining_configuration

    assert _check_mining_configuration(TrainingConfig(hard_negative_mining=True)) is None


@pytest.mark.parametrize(
    ("n_positive", "n_negative", "limit"),
    [(95, 5, 10), (100, 2, 50), (20, 200, 40), (1, 999, 10), (999, 1, 10)],
)
def test_stratified_subsample_never_drops_a_class(
    n_positive: int, n_negative: int, limit: int
) -> None:
    """Flooring only positives let a positive-heavy split round the whole limit into
    positives, making average precision NaN every epoch, saving no checkpoint, and
    aborting with a message that blamed the tile size instead of this subsample."""
    taken = _stratified_subsample(_frame(n_positive, n_negative), limit=limit, seed=1)

    assert len(taken) == limit
    assert (taken["label"] == "positive").sum() >= 1
    assert (taken["label"] != "positive").sum() >= 1


def test_threshold_is_encoded_finely_enough_not_to_collide() -> None:
    """Whole percent made 0.05 and 0.054 identical, and any sub-1% threshold "000"."""
    ids = {
        TrainingConfig(min_oocyte_area_fraction=f).run_id()
        for f in (0.05, 0.054, 0.004, 0.10, 0.25)
    }
    assert len(ids) == 5
    assert "area0500" in TrainingConfig(min_oocyte_area_fraction=0.05).run_id()
    assert "area0040" in TrainingConfig(min_oocyte_area_fraction=0.004).run_id()


def test_stratified_subsample_keeps_both_classes() -> None:
    """A head() would take negatives only and make average precision undefined."""
    frame = _frame(n_positive=20, n_negative=200)

    taken = _stratified_subsample(frame, limit=40, seed=1)

    assert len(taken) == 40
    assert set(taken["label"]) == {"positive", "negative"}


def test_stratified_subsample_preserves_the_positive_rate() -> None:
    frame = _frame(n_positive=100, n_negative=900)

    taken = _stratified_subsample(frame, limit=200, seed=1)

    rate = (taken["label"] == "positive").mean()
    assert rate == pytest.approx(0.10, abs=0.02)


def test_stratified_subsample_is_a_no_op_below_the_limit() -> None:
    frame = _frame(n_positive=5, n_negative=5)
    assert _stratified_subsample(frame, limit=100, seed=1) is frame


def test_stratified_subsample_keeps_a_positive_even_when_rare() -> None:
    """Rounding must not drop the minority class entirely."""
    frame = _frame(n_positive=1, n_negative=999)

    taken = _stratified_subsample(frame, limit=10, seed=1)

    assert (taken["label"] == "positive").sum() >= 1


def test_frozen_arm_input_size_keeps_sub_tiles_at_the_backbone_magnification() -> None:
    """512 goes straight to 224. 1024 goes to 448 so each quadrant of the 2x2 split
    arrives at 224, instead of downsampling the whole tile to roughly 10x."""
    from src.modeling.train_tile_classifier import input_px_for

    frozen = TrainingConfig(architecture="frozen_encoder")
    assert input_px_for(replace(frozen, tile_size=512)) == 224
    assert input_px_for(replace(frozen, tile_size=1024)) == 448

    resnet = TrainingConfig(architecture="resnet18")
    assert input_px_for(replace(resnet, tile_size=512)) == 224
    assert input_px_for(replace(resnet, tile_size=1024)) == 224


def test_resnet18_head_is_a_single_logit() -> None:
    model = build_model(TrainingConfig())
    assert model.fc.out_features == 1


def test_train_transform_augments_and_eval_does_not() -> None:
    """Augmentation at evaluation time would make the metric non-deterministic."""
    train_names = [type(step).__name__ for step in build_transforms(train=True).transforms]
    eval_names = [type(step).__name__ for step in build_transforms(train=False).transforms]

    assert "RandomHorizontalFlip" in train_names
    assert "ColorJitter" in train_names
    assert not any(name.startswith("Random") for name in eval_names)
    assert not any("Jitter" in name for name in eval_names)


def test_neither_transform_centre_crops() -> None:
    """A crop would discard the tile border, so an edge oocyte could be cropped away
    while the label still says positive."""
    for train in (True, False):
        names = [type(step).__name__ for step in build_transforms(train=train).transforms]
        assert not any("Crop" in name for name in names)
        assert "Resize" in names


def test_paths_under_the_data_symlink_stay_repo_relative() -> None:
    """data/ is a symlink to the group lustre share, so resolving it would take every
    artifact path out of the repo and record it absolute and machine-specific."""
    from src.modeling.train_tile_classifier import REPO_ROOT, _path_relative_to_repo

    inside = REPO_ROOT / "data" / "tile_classifier" / "run" / "checkpoint.pt"
    assert _path_relative_to_repo(inside) == "data/tile_classifier/run/checkpoint.pt"
    assert not Path(_path_relative_to_repo(inside)).is_absolute()


def test_relative_input_is_anchored_at_the_repo_root() -> None:
    from src.modeling.train_tile_classifier import _path_relative_to_repo

    assert _path_relative_to_repo(Path("data/x/y.pt")) == "data/x/y.pt"


def test_path_outside_the_repo_stays_absolute() -> None:
    from src.modeling.train_tile_classifier import _path_relative_to_repo

    recorded = _path_relative_to_repo(Path("/tmp/elsewhere/checkpoint.pt"))
    assert Path(recorded).is_absolute()


def test_rotation_is_quarter_turns_only() -> None:
    """RandomRotation(degrees=90) draws a uniform angle in [-90, +90]; any off-axis
    angle rotates the tile's corners out of frame and black-fills them -- the same
    preprocessing label noise centre-cropping was rejected for."""
    from torchvision import transforms

    steps = build_transforms(train=True).transforms
    choice = [s for s in steps if isinstance(s, transforms.RandomChoice)]
    assert choice, "expected a RandomChoice over fixed rotations"
    angles = {tuple(t.degrees) for t in choice[0].transforms}
    assert angles == {(0.0, 0.0), (90.0, 90.0), (180.0, 180.0), (270.0, 270.0)}


def test_no_bare_random_rotation_range_remains() -> None:
    from torchvision import transforms

    for train in (True, False):
        for step in build_transforms(train=train).transforms:
            if isinstance(step, transforms.RandomRotation):
                low, high = step.degrees
                assert low == high, "a rotation range would black-fill tile corners"


@pytest.mark.parametrize(
    ("n_positive", "n_negative", "limit"),
    [(95, 5, 10), (100, 2, 50), (20, 200, 40), (1, 999, 10), (5, 5, 3)],
)
def test_stratified_subsample_returns_exactly_the_limit(
    n_positive: int, n_negative: int, limit: int
) -> None:
    """A positive-heavy split could round the positive quota up to the whole limit and
    then add a negative on top, exceeding it."""
    taken = _stratified_subsample(_frame(n_positive, n_negative), limit=limit, seed=1)

    assert len(taken) == limit
    assert (taken["label"] == "positive").sum() >= 1 or n_positive == 0


def test_nan_metrics_are_written_as_null_not_nan() -> None:
    """json.dump emits a bare NaN literal by default, which is not valid JSON."""
    import json

    from src.modeling.train_tile_classifier import _nan_to_none

    payload = _nan_to_none(
        {"ap": float("nan"), "nested": [{"auc": float("nan")}], "ok": 0.5}
    )
    text = json.dumps(payload, allow_nan=False)

    assert "NaN" not in text
    assert json.loads(text) == {"ap": None, "nested": [{"auc": None}], "ok": 0.5}


def test_encoder_name_is_refused_on_the_resnet_arm() -> None:
    """It would change the run id and recorded config while the model ignored it."""
    from scripts.train_tile_classifier import main

    with pytest.raises(SystemExit, match="only meaningful with"):
        main(["--architecture", "resnet18", "--encoder-name", "owkin/phikon-v2"])


def test_run_id_is_a_single_path_segment() -> None:
    """Encoder ids carry a slash ("owkin/phikon-v2"), which would nest the run one level
    deeper and split its checkpoint, log and composition file across two directories."""
    config = TrainingConfig(
        architecture="frozen_encoder", encoder_name="owkin/phikon-v2"
    )
    run_id = config.run_id()

    assert "/" not in run_id
    assert "owkin-phikon-v2" in run_id
    assert Path(run_id).name == run_id


def test_encoder_slug_does_not_collapse_distinct_encoders() -> None:
    ids = {
        TrainingConfig(architecture="frozen_encoder", encoder_name=name).run_id()
        for name in ("owkin/phikon-v2", "bioptimus/H-optimus-0")
    }
    assert len(ids) == 2


def test_frozen_arm_resolves_its_encoder_at_construction() -> None:
    """Left to build_model, the run id, fingerprint and saved config all said null while
    Phikon-v2 weights were trained -- and the same experiment got two run directories
    depending on whether the flag was passed."""
    implicit = TrainingConfig(architecture="frozen_encoder")
    explicit = TrainingConfig(
        architecture="frozen_encoder", encoder_name="owkin/phikon-v2"
    )

    assert implicit.encoder_name == "owkin/phikon-v2"
    assert implicit.run_id() == explicit.run_id()


def test_resnet_arm_records_no_encoder() -> None:
    """It would change the run id and recorded config while the model ignored it."""
    assert TrainingConfig(architecture="resnet18", encoder_name="owkin/phikon-v2").encoder_name is None


def test_learning_rate_default_is_per_arm() -> None:
    """A probe head trained from scratch at the fine-tuning rate would plausibly early
    stop before converging, losing the comparison for a non-encoder reason."""
    assert TrainingConfig(architecture="resnet18").lr == pytest.approx(1e-4)
    assert TrainingConfig(architecture="frozen_encoder").lr == pytest.approx(1e-3)
    # An explicit value still wins.
    assert TrainingConfig(architecture="frozen_encoder", lr=5e-4).lr == pytest.approx(5e-4)


def test_unknown_architecture_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="architecture must be one of"):
        TrainingConfig(architecture="vit_huge")


def test_normalisation_is_per_encoder() -> None:
    """Normalising with another dataset's statistics shifts every pixel outside the
    range the backbone was trained on, degrading embeddings with no error."""
    from src.modeling.encoders import IMAGENET_MEAN
    from src.modeling.train_tile_classifier import normalisation_for

    resnet_mean, _ = normalisation_for(TrainingConfig(architecture="resnet18"))
    phikon_mean, _ = normalisation_for(TrainingConfig(architecture="frozen_encoder"))
    optimus_mean, _ = normalisation_for(
        TrainingConfig(architecture="frozen_encoder", encoder_name="bioptimus/H-optimus-0")
    )

    assert resnet_mean == IMAGENET_MEAN
    assert phikon_mean == IMAGENET_MEAN  # Phikon-v2 genuinely uses ImageNet stats
    assert optimus_mean != IMAGENET_MEAN  # H-optimus-0 does not


def test_transform_uses_the_normalisation_it_is_given() -> None:
    from torchvision import transforms

    steps = build_transforms(train=False, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6)).transforms
    norm = [s for s in steps if isinstance(s, transforms.Normalize)][0]
    assert tuple(norm.mean) == (0.1, 0.2, 0.3)
    assert tuple(norm.std) == (0.4, 0.5, 0.6)


@pytest.mark.parametrize(
    ("warmup", "epochs"), [(0, 5), (5, 5), (6, 5)]
)
def test_mining_that_could_never_activate_is_refused(warmup: int, epochs: int) -> None:
    """It would record itself as the mining arm while being the mining-off experiment."""
    from src.modeling.train_tile_classifier import train

    config = TrainingConfig(
        hard_negative_mining=True, warmup_epochs=warmup, epochs=epochs
    )
    with pytest.raises(ValueError, match="warmup_epochs"):
        train(config)


def test_frozen_checkpoint_stores_only_the_trained_head() -> None:
    """The backbone is 300M unchanged parameters already in the HuggingFace cache;
    re-serialising it every improving epoch is repeated multi-GB writes to lustre."""
    from src.modeling.encoders import FrozenEncoderClassifier
    from src.modeling.train_tile_classifier import _checkpoint_payload

    class _Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.big = nn.Parameter(torch.zeros(4096))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, 8)

    model = FrozenEncoderClassifier(_Stub(), 8, tile_size=512)
    config = TrainingConfig(architecture="frozen_encoder")

    payload = _checkpoint_payload(model, config, epoch=1, average_precision=0.5)

    assert payload["state_dict_scope"] == "head"
    saved = sum(v.numel() for v in payload["model_state_dict"].values())
    assert saved == sum(p.numel() for p in model.head.parameters())
    assert saved < sum(p.numel() for p in model.parameters())
    # The config names the backbone so the head can be reloaded onto it.
    assert payload["config"]["encoder_name"] == "owkin/phikon-v2"


def test_resnet_checkpoint_stores_the_whole_model() -> None:
    from src.modeling.train_tile_classifier import _checkpoint_payload

    config = TrainingConfig(architecture="resnet18")
    payload = _checkpoint_payload(build_model(config), config, 1, 0.5)

    assert payload["state_dict_scope"] == "model"


def test_subsample_limit_below_two_is_refused() -> None:
    """One row cannot hold both classes, and a single-class val split makes average
    precision undefined every epoch, saves no checkpoint, and aborts with a message
    that blames the tile size."""
    with pytest.raises(ValueError, match="at least 2"):
        _stratified_subsample(_frame(5, 20), limit=1, seed=1)


def test_subsample_of_exactly_two_keeps_one_of_each() -> None:
    taken = _stratified_subsample(_frame(5, 20), limit=2, seed=1)

    assert len(taken) == 2
    assert (taken["label"] == "positive").sum() == 1
    assert (taken["label"] != "positive").sum() == 1


@pytest.mark.parametrize(
    "args",
    [
        ["--positive-fraction", "1.5"],
        ["--positive-fraction", "0"],
        ["--hard-negative-share", "0"],
        ["--hard-negative-pool-fraction", "1.2"],
        ["--min-oocyte-area-fraction", "1.2"],
        ["--max-val-tiles", "1"],
    ],
)
def test_proportions_and_caps_are_validated_at_parse_time(args: list[str]) -> None:
    """The sampler's own range checks are only reached after the encoder weights and the
    whole tile index have loaded, so an out-of-range flag would cost a multi-GB load and
    a full index parse before failing."""
    from scripts.train_tile_classifier import _parse_args

    with pytest.raises(SystemExit):
        _parse_args(args)
