"""Tests for the training loop's pure parts: run identity, subsampling, guards."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

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
        {"hard_negative_pool_fraction": 0.5},
        {"hard_negative_share": 0.25},
        {"warmup_epochs": 1},
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


def test_frozen_encoder_arm_is_not_silently_a_resnet() -> None:
    with pytest.raises(NotImplementedError, match="frozen-encoder arm"):
        build_model(TrainingConfig(architecture="frozen_encoder"))


def test_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError, match="architecture must be one of"):
        build_model(TrainingConfig(architecture="vit_huge"))


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
        main(["--architecture", "resnet18", "--encoder-name", "phikon-v2"])
